# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial import KDTree

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._objects_dc = torch.empty(0)
        self.num_objects = 16
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.inpaint_mask = torch.empty(0, dtype=torch.bool)
        self.height_constrained_mask = torch.empty(0, dtype=torch.bool)
        self.height_constraint_values = torch.empty(0)
        self.plane_normal = None  # For axis-agnostic constraints: normal vector of the constraint plane
        self.plane_centroid = None  # Centroid of the constraint plane
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def get_inpaint_mask(self):
        n_points = self.get_xyz.shape[0]
        if self.inpaint_mask.numel() != n_points:
            self.inpaint_mask = torch.zeros((n_points), device=self.get_xyz.device, dtype=torch.bool)
        else:
            self.inpaint_mask = self.inpaint_mask.to(device=self.get_xyz.device, dtype=torch.bool)
        return self.inpaint_mask

    def _sync_height_constraint_tensors(self):
        n_points = self.get_xyz.shape[0]
        device = self.get_xyz.device
        if self.height_constrained_mask.numel() != n_points:
            self.height_constrained_mask = torch.zeros((n_points), device=device, dtype=torch.bool)
        else:
            self.height_constrained_mask = self.height_constrained_mask.to(device=device, dtype=torch.bool)

        if self.height_constraint_values.numel() != n_points:
            self.height_constraint_values = torch.zeros((n_points), device=device, dtype=self.get_xyz.dtype)
        else:
            self.height_constraint_values = self.height_constraint_values.to(device=device, dtype=self.get_xyz.dtype)

    def set_height_constraint(self, mask, height_value, plane_info=None):
        """
        Set height constraints for Gaussians.
        
        Args:
            mask: Boolean tensor indicating constrained points
            height_value: Scalar or per-point height values
            plane_info: Optional dict with 'normal', 'centroid', 'd' from fit_plane_to_points_axis_agnostic()
        """
        self._sync_height_constraint_tensors()
        constraint_mask = mask.to(device=self.get_xyz.device, dtype=torch.bool)
        if constraint_mask.numel() != self.get_xyz.shape[0]:
            raise ValueError("Constraint mask size must match number of Gaussians.")

        self.height_constrained_mask = constraint_mask
        
        # Store plane info for axis-agnostic constraints
        if plane_info is not None:
            self.plane_normal = plane_info.get('normal')  # numpy array (3,)
            self.plane_centroid = plane_info.get('centroid')  # numpy array (3,)
        
        if torch.is_tensor(height_value):
            if height_value.numel() == 1:
                self.height_constraint_values[constraint_mask] = height_value.to(device=self.get_xyz.device, dtype=self.get_xyz.dtype)
            elif height_value.numel() == self.get_xyz.shape[0]:
                self.height_constraint_values[constraint_mask] = height_value.to(device=self.get_xyz.device, dtype=self.get_xyz.dtype)[constraint_mask]
            else:
                raise ValueError("height_value tensor must be scalar or per-point with matching length.")
        else:
            self.height_constraint_values[constraint_mask] = float(height_value)

    def clear_height_constraint(self):
        self._sync_height_constraint_tensors()
        self.height_constrained_mask.zero_()
        self.height_constraint_values.zero_()
        self.plane_normal = None
        self.plane_centroid = None

    @staticmethod
    def _quaternion_to_rotation_matrix(quaternions):
        q = quaternions / torch.clamp(torch.norm(quaternions, dim=1, keepdim=True), min=1e-12)
        w, x, y, z = q.unbind(dim=1)

        R = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    def apply_height_constraint_to_gradients(self, blend=1.0):
        """
        Apply axis-agnostic covariance constraints along the plane normal.
        Supports legacy Z-axis center constraint if no plane normal is set.
        
        Args:
            blend: Blending factor (0.0 = no constraint, 1.0 = full constraint)
        """
        self._sync_height_constraint_tensors()
        if not self.height_constrained_mask.any():
            return

        blend = max(0.0, min(1.0, float(blend)))
        with torch.no_grad():
            constrained_mask = self.height_constrained_mask
            
            if self.plane_normal is not None:
                plane_normal = torch.from_numpy(self.plane_normal).to(device=self._xyz.device, dtype=self._xyz.dtype)
                plane_normal = plane_normal / torch.clamp(torch.norm(plane_normal), min=1e-12)

                constrained_scaling = self.get_scaling[constrained_mask]
                constrained_rotation = self.get_rotation[constrained_mask]
                if constrained_scaling.numel() == 0:
                    return

                R = self._quaternion_to_rotation_matrix(constrained_rotation)
                normal_batch = plane_normal.view(1, 3, 1).expand(R.shape[0], -1, -1)
                normal_local = torch.bmm(R.transpose(1, 2), normal_batch).squeeze(-1)

                current_var_along_normal = (constrained_scaling.pow(2) * normal_local.pow(2)).sum(dim=1)
                target_var_along_normal = self.height_constraint_values[constrained_mask].clamp_min(0.0)

                exceed_mask = current_var_along_normal > (target_var_along_normal + 1e-12)
                if not exceed_mask.any():
                    return

                ratio = torch.ones_like(current_var_along_normal)
                ratio[exceed_mask] = torch.sqrt(
                    torch.clamp(
                        target_var_along_normal[exceed_mask] / torch.clamp(current_var_along_normal[exceed_mask], min=1e-12),
                        min=0.0,
                        max=1.0,
                    )
                )
                blended_ratio = (1.0 - blend) + blend * ratio

                new_scaling = constrained_scaling * blended_ratio.unsqueeze(1)
                self._scaling.data[constrained_mask] = self.scaling_inverse_activation(torch.clamp(new_scaling, min=1e-8))

                max_var_before = current_var_along_normal.max().item()
                max_var_after = (
                    (new_scaling.pow(2) * normal_local.pow(2)).sum(dim=1)
                ).max().item()
                print(
                    "[HeightConstraint] Applied covariance cap along plane normal "
                    "with blend {:.4f}. Max variance {:.6f} -> {:.6f}".format(
                        blend, max_var_before, max_var_after
                    )
                )
            else:
                # LEGACY: Constrain Z-axis only
                constrained_z = self._xyz.data[constrained_mask, 2]
                target_z = self.height_constraint_values[constrained_mask]
                self._xyz.data[constrained_mask, 2] = (1.0 - blend) * constrained_z + blend * target_z

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._objects_dc,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._objects_dc,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_objects(self):
        return self._objects_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # random init obj_id now
        fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0],self.num_objects), device="cuda"))
        fused_objects = fused_objects[:,:,None]

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._objects_dc = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True))
        self.inpaint_mask = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        self.height_constrained_mask = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        self.height_constraint_values = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=self.get_xyz.dtype)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def finetune_setup(self, training_args, mask3d):
        # Define a function that applies the mask to the gradients
        def mask_hook(grad):
            return grad * mask3d
        def mask_hook2(grad):
            return grad * mask3d.squeeze(-1)
        

        # Register the hook to the parameter (only once!)
        hook_xyz = self._xyz.register_hook(mask_hook2)
        hook_dc = self._features_dc.register_hook(mask_hook)
        hook_rest = self._features_rest.register_hook(mask_hook)
        hook_opacity = self._opacity.register_hook(mask_hook2)
        hook_scaling = self._scaling.register_hook(mask_hook2)
        hook_rotation = self._rotation.register_hook(mask_hook2)

        self._objects_dc.requires_grad = False
        self.inpaint_mask = mask3d.bool().squeeze().to(device=self.get_xyz.device)
        self.height_constrained_mask = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)
        self.height_constraint_values = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=self.get_xyz.dtype)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def removal_setup(self, training_args, mask3d):

        mask3d = ~mask3d.bool().squeeze()

        # Extracting subsets using the mask
        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        features_rest_sub = self._features_rest[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()
        objects_dc_sub = self._objects_dc[mask3d].detach()


        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(set_requires_grad(xyz_sub, False))
        self._features_dc = nn.Parameter(set_requires_grad(features_dc_sub, False))
        self._features_rest = nn.Parameter(set_requires_grad(features_rest_sub, False))
        self._opacity = nn.Parameter(set_requires_grad(opacity_sub, False))
        self._scaling = nn.Parameter(set_requires_grad(scaling_sub, False))
        self._rotation = nn.Parameter(set_requires_grad(rotation_sub, False))
        self._objects_dc = nn.Parameter(set_requires_grad(objects_dc_sub, False))
        self.inpaint_mask = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)
        self.height_constrained_mask = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)
        self.height_constraint_values = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=self.get_xyz.dtype)


    def inpaint_setup(self, training_args, mask3d):

        def initialize_new_features(features, num_new_points, mask_xyz_values, distance_threshold=0.25, max_distance_threshold=1, k=5):
            """Initialize new points for multiple features based on neighbouring points in the remaining area."""
            new_features = {}
            
            if num_new_points == 0:
                for key in features:
                    new_features[key] = torch.empty((0, *features[key].shape[1:]), device=features[key].device)
                return new_features

            # Get remaining points from features
            remaining_xyz_values = features["xyz"]
            remaining_xyz_values_np = remaining_xyz_values.cpu().numpy()
            
            # Build a KD-Tree for fast nearest-neighbor lookup
            kdtree = KDTree(remaining_xyz_values_np)
            
            # Sample random points from mask_xyz_values as query points
            mask_xyz_values_np = mask_xyz_values.cpu().numpy()
            query_points = mask_xyz_values_np

            # Find the k nearest neighbors in the remaining points for each query point
            distances, indices = kdtree.query(query_points, k=k)
            selected_indices = indices

            # Initialize new points for each feature
            for key, feature in features.items():
                # Convert feature to numpy array
                feature_np = feature.cpu().numpy()
                
                # If we have valid neighbors, calculate the mean of neighbor points
                if feature_np.ndim == 2:
                    neighbor_points = feature_np[selected_indices]
                elif feature_np.ndim == 3:
                    neighbor_points = feature_np[selected_indices, :, :]
                else:
                    raise ValueError(f"Unsupported feature dimension: {feature_np.ndim}")
                new_points_np = np.mean(neighbor_points, axis=1)
                
                # Convert back to tensor
                new_features[key] = torch.tensor(new_points_np, device=feature.device, dtype=feature.dtype)
            
            return new_features['xyz'], new_features['features_dc'], new_features['scaling'], new_features['objects_dc'], new_features['features_rest'], new_features['opacity'], new_features['rotation']
        
        mask3d = ~mask3d.bool().squeeze()
        mask_xyz_values = self._xyz[~mask3d]

        # Extracting subsets using the mask
        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        features_rest_sub = self._features_rest[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()
        objects_dc_sub = self._objects_dc[mask3d].detach()

        # Add new points with random initialization
        sub_features = {
            'xyz': xyz_sub,
            'features_dc': features_dc_sub,
            'scaling': scaling_sub,
            'objects_dc': objects_dc_sub,
            'features_rest': features_rest_sub,
            'opacity': opacity_sub,
            'rotation': rotation_sub,
        }

        num_new_points = len(mask_xyz_values)
        with torch.no_grad():
            new_xyz, new_features_dc, new_scaling, new_objects_dc, new_features_rest, new_opacity, new_rotation = initialize_new_features(sub_features, num_new_points, mask_xyz_values)


        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(torch.cat([set_requires_grad(xyz_sub, False), set_requires_grad(new_xyz, True)]))
        self._features_dc = nn.Parameter(torch.cat([set_requires_grad(features_dc_sub, False), set_requires_grad(new_features_dc, True)]))
        self._features_rest = nn.Parameter(torch.cat([set_requires_grad(features_rest_sub, False), set_requires_grad(new_features_rest, True)]))
        self._opacity = nn.Parameter(torch.cat([set_requires_grad(opacity_sub, False), set_requires_grad(new_opacity, True)]))
        self._scaling = nn.Parameter(torch.cat([set_requires_grad(scaling_sub, False), set_requires_grad(new_scaling, True)]))
        self._rotation = nn.Parameter(torch.cat([set_requires_grad(rotation_sub, False), set_requires_grad(new_rotation, True)]))
        self._objects_dc = nn.Parameter(torch.cat([set_requires_grad(objects_dc_sub, False), set_requires_grad(new_objects_dc, True)]))

        num_fixed = xyz_sub.shape[0]
        num_inpaint = new_xyz.shape[0]
        self.inpaint_mask = torch.cat([
            torch.zeros((num_fixed), device=self.get_xyz.device, dtype=torch.bool),
            torch.ones((num_inpaint), device=self.get_xyz.device, dtype=torch.bool)
        ], dim=0)
        self.height_constrained_mask = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)
        self.height_constraint_values = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=self.get_xyz.dtype)

        # for optimize
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Setup optimizer. Only the new points will have gradients.
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"}  # Assuming there's a learning rate for objects_dc in training_args
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._objects_dc.shape[1]*self._objects_dc.shape[2]):
            l.append('obj_dc_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        obj_dc = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        objects_dc = np.zeros((xyz.shape[0], self.num_objects, 1))
        for idx in range(self.num_objects):
            objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.inpaint_mask = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        self.height_constrained_mask = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.bool)
        self.height_constraint_values = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=self.get_xyz.dtype)

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        if self.optimizer is None:
            optimizable_tensors["xyz"] = nn.Parameter(self._xyz[mask].requires_grad_(True))
            optimizable_tensors["f_dc"] = nn.Parameter(self._features_dc[mask].requires_grad_(True))
            optimizable_tensors["f_rest"] = nn.Parameter(self._features_rest[mask].requires_grad_(True))
            optimizable_tensors["opacity"] = nn.Parameter(self._opacity[mask].requires_grad_(True))
            optimizable_tensors["scaling"] = nn.Parameter(self._scaling[mask].requires_grad_(True))
            optimizable_tensors["rotation"] = nn.Parameter(self._rotation[mask].requires_grad_(True))
            optimizable_tensors["obj_dc"] = nn.Parameter(self._objects_dc[mask].requires_grad_(True))
            return optimizable_tensors

        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = optimizable_tensors["obj_dc"]

        new_num_points = self.get_xyz.shape[0]

        if self.xyz_gradient_accum.numel() == valid_points_mask.shape[0]:
            grad_mask = valid_points_mask.to(device=self.xyz_gradient_accum.device)
            self.xyz_gradient_accum = self.xyz_gradient_accum[grad_mask]
        else:
            self.xyz_gradient_accum = torch.zeros((new_num_points, 1), device=self.get_xyz.device)

        if self.denom.numel() == valid_points_mask.shape[0]:
            denom_mask = valid_points_mask.to(device=self.denom.device)
            self.denom = self.denom[denom_mask]
        else:
            self.denom = torch.zeros((new_num_points, 1), device=self.get_xyz.device)

        if self.max_radii2D.numel() == valid_points_mask.shape[0]:
            radii_mask = valid_points_mask.to(device=self.max_radii2D.device)
            self.max_radii2D = self.max_radii2D[radii_mask]
        else:
            self.max_radii2D = torch.zeros((new_num_points), device=self.get_xyz.device)

        if self.inpaint_mask.numel() == valid_points_mask.shape[0]:
            self.inpaint_mask = self.inpaint_mask[valid_points_mask]
        else:
            self.inpaint_mask = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)

        if self.height_constrained_mask.numel() == valid_points_mask.shape[0]:
            self.height_constrained_mask = self.height_constrained_mask[valid_points_mask]
        else:
            self.height_constrained_mask = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)

        if self.height_constraint_values.numel() == valid_points_mask.shape[0]:
            self.height_constraint_values = self.height_constraint_values[valid_points_mask]
        else:
            self.height_constraint_values = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=self.get_xyz.dtype)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc, new_inpaint_mask=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "obj_dc": new_objects_dc}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = optimizable_tensors["obj_dc"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if self.inpaint_mask.numel() == (self.get_xyz.shape[0] - new_xyz.shape[0]):
            if new_inpaint_mask is None:
                new_inpaint_mask = torch.zeros((new_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)
            else:
                new_inpaint_mask = new_inpaint_mask.to(device=self.get_xyz.device, dtype=torch.bool)
            self.inpaint_mask = torch.cat((self.inpaint_mask, new_inpaint_mask), dim=0)
        else:
            self.inpaint_mask = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)

        if self.height_constrained_mask.numel() == (self.get_xyz.shape[0] - new_xyz.shape[0]):
            new_height_mask = torch.zeros((new_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)
            new_height_values = torch.zeros((new_xyz.shape[0]), device=self.get_xyz.device, dtype=self.get_xyz.dtype)
            self.height_constrained_mask = torch.cat((self.height_constrained_mask, new_height_mask), dim=0)
            self.height_constraint_values = torch.cat((self.height_constraint_values, new_height_values), dim=0)
        else:
            self.height_constrained_mask = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=torch.bool)
            self.height_constraint_values = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device, dtype=self.get_xyz.dtype)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N,1,1)
        parent_inpaint_mask = self.get_inpaint_mask()[selected_pts_mask]
        new_inpaint_mask = parent_inpaint_mask.repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc, new_inpaint_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_objects_dc = self._objects_dc[selected_pts_mask]
        new_inpaint_mask = self.get_inpaint_mask()[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc, new_inpaint_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1