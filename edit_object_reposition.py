# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import json
import os
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path
from random import randint

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from edit_object_removal import points_inside_convex_hull
from gaussian_renderer import GaussianModel, render
from render import feature_to_rgb, visualize_obj
from scene import Scene
from utils.general_utils import safe_state
from utils.loss_utils import ssim


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class PseudoGTSupervision:
    def __init__(self, pseudo_gt_path):
        self.enabled = bool(pseudo_gt_path)
        self.pseudo_gt_path = Path(pseudo_gt_path) if self.enabled else None
        self._cache = {}

    def _load_image(self, image_name, image_height, image_width, device):
        if not self.enabled:
            return None
        if image_name in self._cache:
            cached = self._cache[image_name]
            if cached is None:
                return None
            return cached.to(device)

        image_file = self.pseudo_gt_path / f"{image_name}.png"
        if not image_file.exists():
            self._cache[image_name] = None
            return None

        image_np = np.array(Image.open(image_file).convert("RGB"), dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        if image_tensor.shape[1] != image_height or image_tensor.shape[2] != image_width:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(image_height, image_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        self._cache[image_name] = image_tensor
        return image_tensor.to(device)


def apply_translation_to_selected_gaussians(gaussians, mask3d, translation):
    translation_tensor = torch.tensor(translation, dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)
    if translation_tensor.abs().sum().item() == 0:
        return
    with torch.no_grad():
        gaussians._xyz.data[mask3d] = gaussians._xyz.data[mask3d] + translation_tensor


def duplicate_and_translate_selected_gaussians(gaussians, mask3d, translation):
    """
    Keep the original gaussians in place and create a translated duplicate copy.

    Returns:
        train_mask_expanded: Bool mask of length N+M for optimization.
            Marks original selected gaussians and translated duplicates.
        translated_only_mask: Bool mask of length N+M marking only translated duplicates.
    """
    translation_tensor = torch.tensor(translation, dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)
    n_original = gaussians._xyz.shape[0]
    n_selected = int(mask3d.sum().item())

    if translation_tensor.abs().sum().item() == 0 or n_selected == 0:
        translated_only_mask = torch.zeros_like(mask3d, dtype=torch.bool)
        return mask3d, translated_only_mask

    with torch.no_grad():
        xyz_new = gaussians._xyz[mask3d].detach().clone() + translation_tensor
        features_dc_new = gaussians._features_dc[mask3d].detach().clone()
        features_rest_new = gaussians._features_rest[mask3d].detach().clone()
        opacity_new = gaussians._opacity[mask3d].detach().clone()
        scaling_new = gaussians._scaling[mask3d].detach().clone()
        rotation_new = gaussians._rotation[mask3d].detach().clone()
        objects_dc_new = gaussians._objects_dc[mask3d].detach().clone()

        gaussians._xyz = torch.nn.Parameter(torch.cat([gaussians._xyz.detach(), xyz_new], dim=0).requires_grad_(True))
        gaussians._features_dc = torch.nn.Parameter(
            torch.cat([gaussians._features_dc.detach(), features_dc_new], dim=0).requires_grad_(True)
        )
        gaussians._features_rest = torch.nn.Parameter(
            torch.cat([gaussians._features_rest.detach(), features_rest_new], dim=0).requires_grad_(True)
        )
        gaussians._opacity = torch.nn.Parameter(torch.cat([gaussians._opacity.detach(), opacity_new], dim=0).requires_grad_(True))
        gaussians._scaling = torch.nn.Parameter(torch.cat([gaussians._scaling.detach(), scaling_new], dim=0).requires_grad_(True))
        gaussians._rotation = torch.nn.Parameter(torch.cat([gaussians._rotation.detach(), rotation_new], dim=0).requires_grad_(True))
        gaussians._objects_dc = torch.nn.Parameter(torch.cat([gaussians._objects_dc.detach(), objects_dc_new], dim=0).requires_grad_(True))

    train_mask_expanded = torch.zeros((n_original + n_selected), device=mask3d.device, dtype=torch.bool)
    train_mask_expanded[:n_original] = mask3d
    train_mask_expanded[n_original:] = True

    translated_only_mask = torch.zeros((n_original + n_selected), device=mask3d.device, dtype=torch.bool)
    translated_only_mask[n_original:] = True

    return train_mask_expanded, translated_only_mask


def reduce_opacity_in_destination(gaussians, translated_mask3d, target_opacity=0.05, blend_radius=0.05):
    """
    Reduce opacity of gaussians in the destination area to create space for blending.
    
    Args:
        gaussians: GaussianModel instance
        translated_mask3d: Boolean mask of gaussians that were translated
        target_opacity: Target opacity value for gaussians in destination (default 0.05, very transparent)
        blend_radius: Kept for backward compatibility; convex-hull based blending no longer uses a radius search
    """
    with torch.no_grad():
        translated_positions = gaussians._xyz.data[translated_mask3d]
        all_positions = gaussians._xyz.data

        if translated_positions.shape[0] == 0:
            return

        # Build a convex hull around the translated gaussians and affect the
        # other gaussians that fall inside this destination region.
        from scipy.spatial import Delaunay

        translated_points = translated_positions.detach().cpu().numpy()
        if translated_points.shape[0] < 4:
            # A 3D convex hull needs at least four non-coplanar points.
            # Fall back to affecting the translated gaussians themselves only.
            destination_mask = translated_mask3d.clone()
        else:
            try:
                hull = Delaunay(translated_points)
                inside_mask = torch.from_numpy(
                    hull.find_simplex(all_positions.detach().cpu().numpy()) >= 0
                ).to(device=translated_mask3d.device)
                destination_mask = inside_mask & (~translated_mask3d)
            except Exception:
                # If the hull is degenerate, fall back to a conservative mask.
                destination_mask = translated_mask3d.clone()

        if destination_mask.sum().item() == 0:
            return

        from utils.general_utils import inverse_sigmoid
        target_opacity_internal = inverse_sigmoid(torch.tensor(target_opacity, device=gaussians._opacity.device))

        gaussians._opacity.data[destination_mask] = target_opacity_internal
        print(f"Reduced opacity for {destination_mask.sum().item()} gaussians in destination area")



# def remove_gaussians_in_destination(
#     gaussians,
#     source_xyz_before_translation,
#     source_anchor_mask_original,
#     translation,
#     protected_mask=None,
#     match_radius=0.015,
# ):
#     """
#     Translate source-anchor centers using the translation vector and remove
#     other gaussians that overlap those translated positions.
#     """
#     with torch.no_grad():
#         source_anchor_mask_original = source_anchor_mask_original.to(
#             device=source_xyz_before_translation.device, dtype=torch.bool
#         ).flatten()
#         if source_anchor_mask_original.sum().item() == 0:
#             return None

#         from scipy.spatial import cKDTree

#         translation_tensor = torch.tensor(
#             translation,
#             dtype=source_xyz_before_translation.dtype,
#             device=source_xyz_before_translation.device,
#         )
#         translated_positions = source_xyz_before_translation[source_anchor_mask_original] + translation_tensor
#         if translated_positions.shape[0] == 0:
#             return None

#         translated_points = translated_positions.detach().cpu().numpy().astype(np.float32)
#         all_points = gaussians._xyz.detach().cpu().numpy().astype(np.float32)

#         if match_radius is None or match_radius <= 0:
#             if translated_points.shape[0] >= 2:
#                 translated_tree = cKDTree(translated_points)
#                 nn_distances, _ = translated_tree.query(translated_points, k=2)
#                 match_radius = float(np.median(nn_distances[:, 1]) * 0.1)
#             else:
#                 match_radius = 1e-3

#         match_radius = max(float(match_radius), 1e-6)

#         try:
#             all_tree = cKDTree(all_points)
#             neighbor_lists = all_tree.query_ball_point(translated_points, r=match_radius)
#             if len(neighbor_lists) == 0:
#                 destination_mask = torch.zeros((all_points.shape[0],), device=gaussians._xyz.device, dtype=torch.bool)
#             else:
#                 non_empty_neighbors = [np.asarray(ids, dtype=np.int64) for ids in neighbor_lists if len(ids) > 0]
#                 destination_mask = torch.zeros((all_points.shape[0],), device=gaussians._xyz.device, dtype=torch.bool)
#                 if len(non_empty_neighbors) == 0:
#                     return None
#                 neighbor_indices = np.unique(np.concatenate(non_empty_neighbors))
#                 if neighbor_indices.size > 0:
#                     destination_mask[torch.from_numpy(neighbor_indices).to(device=gaussians._xyz.device)] = True
#         except Exception:
#             return None

#         if protected_mask is not None:
#             protected_mask = protected_mask.to(device=destination_mask.device, dtype=torch.bool).flatten()
#             if protected_mask.shape[0] == destination_mask.shape[0]:
#                 destination_mask = destination_mask & (~protected_mask)

#         if destination_mask.sum().item() == 0:
#             return None

#         gaussians.prune_points(destination_mask)
#         print(f"Removed {destination_mask.sum().item()} gaussians in destination area")
#         return destination_mask


def remove_gaussians_in_destination(gaussians, translated_mask3d):
    """
    Remove gaussians that fall inside the destination convex hull, excluding
    the translated gaussians themselves.
    """
    with torch.no_grad():
        translated_positions = gaussians._xyz.data[translated_mask3d]
        if translated_positions.shape[0] == 0:
            return None

        from scipy.spatial import Delaunay

        translated_points = translated_positions.detach().cpu().numpy()
        if translated_points.shape[0] < 4:
            destination_mask = translated_mask3d.clone()
        else:
            try:
                center = translated_points.mean(axis=0)
                shrink_factor = 0.8  # < 1.0 shrinks the hull

                translated_points_shrunk = center + shrink_factor * (translated_points - center)
                hull = Delaunay(translated_points_shrunk)
                # hull = Delaunay(translated_points)
                inside_mask = torch.from_numpy(
                    hull.find_simplex(gaussians._xyz.detach().cpu().numpy()) >= 0
                ).to(device=translated_mask3d.device)
                destination_mask = inside_mask & (~translated_mask3d)
            except Exception:
                destination_mask = translated_mask3d.clone()

        if destination_mask.sum().item() == 0:
            return None

        gaussians.prune_points(destination_mask)
        print(f"Removed {destination_mask.sum().item()} gaussians in destination area")
        return destination_mask

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


def cap_covariances_toward_target(gaussians, source_mask, target_mask, blend=1.0, safety_margin=0.05):
    """
    Prevent source-side Gaussian covariances from stretching into the target region's
    bounding box. Uses axis-aligned bounding box to define the protected destination area.
    
    Args:
        gaussians: GaussianModel instance
        source_mask: Boolean mask of source Gaussians to constrain
        target_mask: Boolean mask defining target region via bounding box
        blend: Blending parameter (1.0 = full constraint, 0.0 = no constraint)
        safety_margin: Extra margin around bbox to maintain (default 0.05)
    """
    with torch.no_grad():
        source_mask = source_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).flatten()
        target_mask = target_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).flatten()

        if not source_mask.any() or not target_mask.any():
            return 0

        # Compute bounding box from target region
        target_positions = gaussians.get_xyz[target_mask]
        bbox_min = target_positions.min(dim=0).values
        bbox_max = target_positions.max(dim=0).values
        bbox_min = bbox_min - safety_margin
        bbox_max = bbox_max + safety_margin
        
        source_indices = torch.where(source_mask)[0]
        source_positions = gaussians.get_xyz[source_mask]
        
        if source_positions.shape[0] == 0:
            return 0

        # Find closest point on bbox for each source Gaussian
        # Clamp position to bbox to get closest point
        closest_points = torch.clamp(source_positions, bbox_min, bbox_max)
        distances_to_bbox = torch.norm(source_positions - closest_points, dim=1)

        constrained_scaling = gaussians.get_scaling[source_mask]
        constrained_rotation = gaussians.get_rotation[source_mask]
        
        # Compute uniform scaling estimate as average scale
        uniform_scale = constrained_scaling.mean(dim=1)
        
        # Identify Gaussians that would stretch into the bbox
        # A Gaussian stretches into bbox if its scale is comparable to or larger than distance to bbox
        shrink_mask = distances_to_bbox <= (uniform_scale + safety_margin)
        
        if not shrink_mask.any():
            return 0

        # Compute shrinking ratios for violating Gaussians
        active_distances = distances_to_bbox[shrink_mask]
        active_scaling = constrained_scaling[shrink_mask]
        
        # Target extent should not reach the bbox
        target_extent = torch.clamp(active_distances - safety_margin, min=1e-8)
        current_extent = active_scaling.mean(dim=1)
        
        # Compute shrinking ratio needed
        ratio = torch.sqrt(
            torch.clamp(
                target_extent / torch.clamp(current_extent, min=1e-12),
                min=0.0,
                max=1.0,
            )
        )
        blended_ratio = (1.0 - blend) + blend * ratio

        # Apply shrinking to violating Gaussians
        shrink_indices = source_indices[shrink_mask]
        new_scaling = active_scaling * blended_ratio.unsqueeze(1)
        gaussians._scaling.data[shrink_indices] = gaussians.scaling_inverse_activation(
            torch.clamp(new_scaling, min=1e-8)
        )

        print(
            f"[Reposition] Capped covariance for {int(shrink_mask.sum().item())} gaussians "
            f"to avoid stretching into target bbox."
        )
        return int(shrink_mask.sum().item())


def finetune_reposition(
    opt,
    model_path,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    classifier,
    selected_obj_ids,
    removal_thresh,
    finetune_iteration,
    translation,
    pseudo_gt_path,
    lambda_ssim=0.2,
    enable_opacity_blending=False,
    opacity_blend_target=0.05,
    opacity_blend_radius=0.1,
    keep_original_gaussians=False,
):
    supervision = PseudoGTSupervision(pseudo_gt_path)

    selected_obj_ids = torch.tensor(selected_obj_ids).cuda()
    source_xyz_before_translation = gaussians._xyz.detach().clone()
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
        prob_obj3d = torch.softmax(logits3d, dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        mask3d = mask.any(dim=0).squeeze()
        # mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(), mask3d, outlier_factor=1.0)
        # mask3d = torch.logical_or(mask3d, mask3d_convex)

    source_anchor_mask_original = mask3d.clone()
    target_anchor_mask = mask3d
    if keep_original_gaussians:
        mask3d_for_optimizer, target_anchor_mask = duplicate_and_translate_selected_gaussians(
            gaussians, mask3d, translation
        )
    else:
        apply_translation_to_selected_gaussians(gaussians, mask3d, translation)
        mask3d_for_optimizer = mask3d
    
    mask3d_for_pseudo_repositioned = mask3d_for_optimizer

    source_neighborhood_mask = points_inside_convex_hull(
        source_xyz_before_translation, source_anchor_mask_original, outlier_factor=1.0
    )
    if source_neighborhood_mask.shape[0] != target_anchor_mask.shape[0]:
        source_neighborhood_mask_expanded = torch.zeros_like(target_anchor_mask, dtype=torch.bool)
        source_neighborhood_mask_expanded[: source_neighborhood_mask.shape[0]] = source_neighborhood_mask
        source_neighborhood_mask = source_neighborhood_mask_expanded

    # Remove gaussians that already occupy the destination region before
    # finetune_setup(), because pruning requires an initialized optimizer.
    # removed_mask = remove_gaussians_in_destination(gaussians, target_anchor_mask)
    # if removed_mask is not None:
    #     keep_mask = ~removed_mask
    #     mask3d_for_optimizer = mask3d_for_optimizer[keep_mask]
    #     target_anchor_mask = target_anchor_mask[keep_mask]
    #     source_neighborhood_mask = source_neighborhood_mask[keep_mask]

    target_neighborhood_mask = points_inside_convex_hull(
        gaussians._xyz.detach(), target_anchor_mask, outlier_factor=1.0
    )
    mask3d_for_optimizer = torch.logical_or(source_neighborhood_mask, target_neighborhood_mask)
    mask3d_for_optimizer = torch.logical_and(mask3d_for_optimizer, ~target_anchor_mask)

    finetune_mask = mask3d_for_optimizer.clone()
    #change mask3d_for_pseudo_repositioned to finetune mask if repositioning pseudo gt is not used 
    gaussians.finetune_setup(opt, finetune_mask.float()[:, None, None])

    lpips_metric = lpips.LPIPS(net="vgg")
    for param in lpips_metric.parameters():
        param.requires_grad = False
    lpips_metric.cuda()

    progress_bar = tqdm(range(finetune_iteration), desc="Reposition finetune")
    for i in range(finetune_iteration):
        viewpoint_stack = views.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
        rendering = render_pkg["render"]

        pseudo_gt = supervision._load_image(
            viewpoint_cam.image_name,
            int(viewpoint_cam.image_height),
            int(viewpoint_cam.image_width),
            rendering.device,
        )
        if pseudo_gt is None:
            pseudo_gt = viewpoint_cam.original_image.cuda()

        l1 = torch.abs(rendering - pseudo_gt).mean()

        rendering_lpips = F.interpolate(
            rendering.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
        )
        pseudo_gt_lpips = F.interpolate(
            pseudo_gt.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False
        )
        lpips_loss = lpips_metric(rendering_lpips * 2 - 1, pseudo_gt_lpips * 2 - 1).mean()
        ssim_loss = 1.0 - ssim(rendering.unsqueeze(0), pseudo_gt.unsqueeze(0))

        loss = (0.8 - opt.lambda_dssim) * l1 + opt.lambda_dssim * lpips_loss + lambda_ssim * ssim_loss
        loss.backward()

        # with torch.no_grad():
        #     if iteration < 5000 :
        #         gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        #         gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        #         if  iteration % 300 == 0:
        #             size_threshold = 20 
        #             gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
  

        if enable_opacity_blending:
            reduce_opacity_in_destination(
                gaussians, 
                target_anchor_mask,
                target_opacity=opacity_blend_target,
                blend_radius=opacity_blend_radius
            )

        gaussians.optimizer.step()

        # cap_covariances_toward_target(
        #     gaussians,
        #     finetune_mask,
        #     target_anchor_mask,
        #     blend=0.3,
        #     safety_margin=0.0,
        # )
        
        gaussians.optimizer.zero_grad(set_to_none=True)

        if i % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.7f}"})
            progress_bar.update(10)
    progress_bar.close()

    # removed_mask = remove_gaussians_in_destination(
    #     gaussians,
    #     source_xyz_before_translation,
    #     source_anchor_mask_original,
    #     translation,
    # )
    # if removed_mask is not None:
    #     keep_mask = ~removed_mask
    #     mask3d_for_optimizer = mask3d_for_optimizer[keep_mask]
    #     target_anchor_mask = target_anchor_mask[keep_mask]
    #     source_neighborhood_mask = source_neighborhood_mask[keep_mask]


    removed_mask = remove_gaussians_in_destination(gaussians, target_anchor_mask)
    if removed_mask is not None:
            keep_mask = ~removed_mask
            mask3d_for_optimizer = mask3d_for_optimizer[keep_mask]
            target_anchor_mask = target_anchor_mask[keep_mask]
            source_neighborhood_mask = source_neighborhood_mask[keep_mask]

    


    point_cloud_path = os.path.join(model_path, f"point_cloud_object_reposition_copy/iteration_{iteration}")
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    return gaussians


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier):
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits, dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))

        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, "{0:05d}".format(idx) + ".png"))
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, "{0:05d}".format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, "{0:05d}".format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))

    out_path = os.path.join(render_path[:-8], "concat")
    makedirs(out_path, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"DIVX")
    size = (gt.shape[-1] * 5, gt.shape[-2])
    fps = float(5) if "train" in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path, "result.mp4"), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        gt = np.array(Image.open(os.path.join(gts_path, file_name)))
        rgb = np.array(Image.open(os.path.join(render_path, file_name)))
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path, file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path, file_name)))
        pred_obj = np.array(Image.open(os.path.join(pred_obj_path, file_name)))

        result = np.hstack([gt, rgb, gt_obj, pred_obj, render_obj]).astype("uint8")
        Image.fromarray(result).save(os.path.join(out_path, file_name))
        writer.write(result[:, :, ::-1])

    writer.release()


def reposition(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    opt: OptimizationParams,
    select_obj_id,
    removal_thresh: float,
    finetune_iteration: int,
    translation,
    pseudo_gt_path,
    enable_opacity_blending: bool = False,
    opacity_blend_target: float = 0.0,
    opacity_blend_radius: float = 0.1,
    keep_original_gaussians: bool = False,
):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    num_classes = dataset.num_classes
    print("Num classes:", num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(
        safe_torch_load(os.path.join(dataset.model_path, "point_cloud", f"iteration_{scene.loaded_iter}", "classifier.pth"))
    )
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = finetune_reposition(
        opt,
        dataset.model_path,
        scene.loaded_iter,
        scene.getTrainCameras(),
        gaussians,
        pipeline,
        background,
        classifier,
        select_obj_id,
        removal_thresh,
        finetune_iteration,
        translation,
        pseudo_gt_path,
        lambda_ssim=getattr(opt, "reposition_lambda_ssim", 0.2),
        enable_opacity_blending=enable_opacity_blending,
        opacity_blend_target=opacity_blend_target,
        opacity_blend_radius=opacity_blend_radius,
        keep_original_gaussians=keep_original_gaussians,
    )

    dataset.object_path = "object_mask"
    dataset.images = "images"
    scene = Scene(dataset, gaussians, load_iteration=f"_object_reposition_copy/iteration_{scene.loaded_iter}", shuffle=False)

    with torch.no_grad():
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                classifier,
            )
        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                classifier,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Road damage reposition with pseudo-GT supervision")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--config_file", type=str, default="config/object_reposition/bear.json", help="Path to the configuration file")
    parser.add_argument("--translation_dx", type=float, default=0.0, help="Translation in world x-axis")
    parser.add_argument("--translation_dy", type=float, default=0.0, help="Translation in world y-axis")
    parser.add_argument("--translation_dz", type=float, default=0.0, help="Translation in world z-axis")
    parser.add_argument("--pseudo_gt_path", type=str, default="", help="Directory containing pseudo-GT images as <image_name>.png")
    parser.add_argument("--enable_opacity_blending", action="store_true", help="Enable opacity reduction in destination area for better blending")
    parser.add_argument("--opacity_blend_target", type=float, default=0.05, help="Target opacity for gaussians in destination area")
    parser.add_argument("--opacity_blend_radius", type=float, default=0.1, help="Radius around translated gaussians to affect for blending")
    parser.add_argument("--keep_original_gaussians", action="store_true", help="Keep the original gaussians in place and duplicate them at the translated location")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    try:
        with open(args.config_file, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as error:
        print(f"Error: Failed to parse the JSON configuration file: {error}")
        exit(1)

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])
    args.images = config.get("images", "images")
    args.object_path = config.get("object_path", "object_mask")
    args.resolution = config.get("r", 1)
    args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    args.finetune_iteration = config.get("finetune_iteration", 10000)
    args.reposition_lambda_ssim = config.get("reposition_lambda_ssim", 0.2)

    cfg_translation = config.get("translation", None)
    if cfg_translation is not None:
        if not isinstance(cfg_translation, list) or len(cfg_translation) != 3:
            raise ValueError("Config key 'translation' must be a list [dx, dy, dz].")
        args.translation_dx = float(cfg_translation[0])
        args.translation_dy = float(cfg_translation[1])
        args.translation_dz = float(cfg_translation[2])
    else:
        args.translation_dx = config.get("translation_dx", args.translation_dx)
        args.translation_dy = config.get("translation_dy", args.translation_dy)
        args.translation_dz = config.get("translation_dz", args.translation_dz)

    args.pseudo_gt_path = config.get("pseudo_gt_path", args.pseudo_gt_path)

    args.enable_opacity_blending = config.get("enable_opacity_blending", args.enable_opacity_blending)
    args.opacity_blend_target = config.get("opacity_blend_target", args.opacity_blend_target)
    args.opacity_blend_radius = config.get("opacity_blend_radius", args.opacity_blend_radius)
    args.keep_original_gaussians = config.get("keep_original_gaussians", args.keep_original_gaussians)

    translation = [args.translation_dx, args.translation_dy, args.translation_dz]
    print(f"Using translation: {translation}")
    print(f"Pseudo-GT path: {args.pseudo_gt_path}")

    safe_state(args.quiet)

    reposition(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        opt.extract(args),
        args.select_obj_id,
        args.removal_thresh,
        args.finetune_iteration,
        translation,
        args.pseudo_gt_path,
        enable_opacity_blending=args.enable_opacity_blending,
        opacity_blend_target=args.opacity_blend_target,
        opacity_blend_radius=args.opacity_blend_radius,
        keep_original_gaussians=args.keep_original_gaussians,
    )
