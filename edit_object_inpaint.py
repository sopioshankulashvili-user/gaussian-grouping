# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import cv2
from utils.loss_utils import masked_l1_loss, ssim
from random import randint
import lpips
import json
import torch.nn.functional as F

from render import feature_to_rgb, visualize_obj
from edit_object_removal import points_inside_convex_hull

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from pathlib import Path
from utils.graphics_utils import geom_transform_points
from height_constraint import create_road_height_constraint


class RoadConstraintManager:
    def __init__(self, road_mask_path):
        self.enabled = bool(road_mask_path)
        self.road_mask_path = Path(road_mask_path) if self.enabled else None
        self._mask_cache = {}

    def _load_mask(self, image_name):
        if not self.enabled:
            return None
        if image_name in self._mask_cache:
            return self._mask_cache[image_name]
        mask_file = self.road_mask_path / f"{image_name}.png"
        if not mask_file.exists():
            self._mask_cache[image_name] = None
            return None
        mask_np = np.array(Image.open(mask_file).convert("L"), dtype=np.uint8)
        mask_tensor = torch.from_numpy((mask_np > 0).astype(np.bool_)).cuda()
        self._mask_cache[image_name] = mask_tensor
        return mask_tensor

    def build_visible_road_mask(self, gaussians, viewpoint_cam, visibility_filter):
        road_mask_2d = self._load_mask(viewpoint_cam.image_name)
        if road_mask_2d is None:
            return None
        width = int(viewpoint_cam.image_width)
        height = int(viewpoint_cam.image_height)
        if road_mask_2d.shape[0] != height or road_mask_2d.shape[1] != width:
            road_mask_2d = F.interpolate(
                road_mask_2d.float().unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="nearest"
            ).squeeze(0).squeeze(0).bool()
        visible_indices = torch.where(visibility_filter)[0]
        if visible_indices.numel() == 0:
            return None
        xyz_visible = gaussians.get_xyz[visible_indices]
        projected_ndc = geom_transform_points(xyz_visible, viewpoint_cam.full_proj_transform)
        px = ((projected_ndc[:, 0] + 1.0) * 0.5 * (width - 1)).long()
        py = ((1.0 - projected_ndc[:, 1]) * 0.5 * (height - 1)).long()
        in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        if not in_bounds.any():
            return None
        visible_indices = visible_indices[in_bounds]
        px = px[in_bounds]
        py = py[in_bounds]
        on_road = road_mask_2d[py, px]
        if not on_road.any():
            return None
        road_indices = visible_indices[on_road]
        full_mask = torch.zeros((gaussians.get_xyz.shape[0]), device=gaussians.get_xyz.device, dtype=torch.bool)
        full_mask[road_indices] = True
        return full_mask


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        # We use a standard pre-trained VGG16 for texture features
        vgg = models.vgg16(pretrained=True).features.cuda().eval()
        # We only need the first few layers for texture/style
        self.layers = nn.Sequential(*list(vgg.children())[:16]) 
        for param in self.parameters():
            param.requires_grad = False

    def gram_matrix(self, x):
        (b, c, h, w) = x.size()
        features = x.view(b, c, h * w)
        # Matrix multiplication of features with their transpose
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def forward(self, input, target):
        # Normalize input to VGG expectations (0-1 -> VGG Mean/Std)
        # Note: input/target are already in [-1, 1] if coming from LPIPS logic
        input_vgg = (input + 1) / 2
        target_vgg = (target + 1) / 2
        
        feat_in = self.layers(input_vgg)
        feat_tar = self.layers(target_vgg)
        
        gram_in = self.gram_matrix(feat_in)
        gram_tar = self.gram_matrix(feat_tar)
        
        return F.mse_loss(gram_in, gram_tar)

def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    
    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

# Function to divide image into K x K patches
def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)

def finetune_inpaint(opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, removal_thresh, finetune_iteration, inpaint_strategy="direct", road_mask_path="", min_road_points=128, height_loss_weight=0.2, hole_loss_weight=0.05, min_opacity=0.05):

    # Initialize road constraint manager
    road_manager = RoadConstraintManager(road_mask_path)
    print("Road constraints active:", road_manager.enabled)
    print("Road mask path:", road_mask_path)
    print(bool(road_mask_path))
    road_constraints_active = bool(road_mask_path) and road_manager.enabled

    # get 3d gaussians idx corresponding to select obj id
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d,dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > removal_thresh
        mask3d = mask.any(dim=0).squeeze()

        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
        mask3d = torch.logical_or(mask3d,mask3d_convex)
        mask3d = mask3d.float()[:,None,None]

    strategy = str(inpaint_strategy).lower()
    valid_strategies = {"reseed", "direct", "none"}
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown inpaint_strategy: {inpaint_strategy}. Use one of ['reseed', 'direct']")
    
    def apply_strategy(strategy_name):
        if strategy_name == "reseed":
            gaussians.inpaint_setup(opt, mask3d)
        elif strategy_name == "direct":
            gaussians.finetune_setup(opt, mask3d)
        else:
            raise ValueError(f"Invalid inpaint setup strategy: {strategy_name}")

    apply_strategy(strategy)
    print(f"Inpaint strategy: {strategy}")
    

    iterations = finetune_iteration
    progress_bar = tqdm(range(iterations), desc="Finetuning progress")
    LPIPS = lpips.LPIPS(net='vgg')
    for param in LPIPS.parameters():
        param.requires_grad = False
    LPIPS.cuda()

    STYLE = StyleLoss().cuda()

    for iteration in range(iterations):
        viewpoint_stack = views.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

        mask2d = viewpoint_cam.objects
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = masked_l1_loss(image, gt_image, ~mask2d)

        bbox = mask_to_bbox(mask2d)
        cropped_image = crop_using_bbox(image, bbox)
        cropped_gt_image = crop_using_bbox(gt_image, bbox)
        K = 2
        rendering_patches = divide_into_patches(cropped_image[None, ...], K)
        gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)

        #sopio: fix "patches too small error"
        rendering_patches = rendering_patches.squeeze()
        gt_patches = gt_patches.squeeze()

        # Resize patches to safe size
        rendering_patches = F.interpolate(rendering_patches, size=(32, 32), mode='bilinear', align_corners=False)
        gt_patches = F.interpolate(gt_patches, size=(32,32), mode='bilinear', align_corners=False)

        lpips_loss = LPIPS(rendering_patches*2-1, gt_patches*2-1).mean()
        # lpips_loss = LPIPS(rendering_patches.squeeze()*2-1,gt_patches.squeeze()*2-1).mean()


        # add ssim loss
        lambda_ssim = 0.2
        ssim_val = ssim(rendering_patches, gt_patches)

        loss_ssim = 1.0 - ssim_val

        
        """Style loss logic"""
        loss_style = STYLE(rendering_patches, gt_patches)

        loss = (0.8 - opt.lambda_dssim) * Ll1 + \
            (opt.lambda_dssim * lpips_loss) + \
            (lambda_ssim * loss_ssim) 
            # (0.1 * loss_style)


        ### ROAD CONSTRAINTS ###
        # Apply road constraint losses
        loss_road_height = None
        loss_road_hole = None
        if road_constraints_active and iteration % max(1, 1) == 0:
            road_gaussian_mask = road_manager.build_visible_road_mask(gaussians, viewpoint_cam, visibility_filter)
            if road_gaussian_mask is not None and road_gaussian_mask.sum().item() >= min_road_points:
                create_road_height_constraint(
                    gaussians,
                    road_gaussian_mask,
                    method='fit_plane_axis_agnostic'
                )

                if gaussians.plane_normal is not None and gaussians.plane_centroid is not None:
                        n = gaussians.plane_normal.astype(np.float64)   # [a, b, c]
                        c0 = gaussians.plane_centroid.astype(np.float64)
                        d = -float(np.dot(n, c0))
                        cam_name = getattr(viewpoint_cam, "image_name", "unknown_view")
                        print(
                            f"[Iter {iteration:06d}] [View {cam_name}] "
                            f"Plane: {n[0]:+.6f}x {n[1]:+.6f}y {n[2]:+.6f}z {d:+.6f} = 0"
                        )

                constrained_mask = gaussians.height_constrained_mask
                constrained_xyz = gaussians.get_xyz[constrained_mask]
                constrained_targets = gaussians.height_constraint_values[constrained_mask]

                if gaussians.plane_normal is not None:
                    plane_normal = torch.from_numpy(gaussians.plane_normal).float().to(device=constrained_xyz.device)
                    plane_centroid = torch.from_numpy(gaussians.plane_centroid).float().to(device=constrained_xyz.device)
                    current_vec = constrained_xyz - plane_centroid.unsqueeze(0)
                    current_distance = (current_vec * plane_normal.unsqueeze(0)).sum(dim=1)
                    loss_road_height = torch.mean((current_distance - constrained_targets) ** 2)
                else:
                    loss_road_height = torch.mean((constrained_xyz[:, 2] - constrained_targets) ** 2)

                loss = loss + height_loss_weight * loss_road_height

                constrained_opacity = gaussians.get_opacity[constrained_mask].squeeze(-1)
                loss_road_hole = torch.relu(min_opacity - constrained_opacity).mean()
                loss = loss + hole_loss_weight * loss_road_hole
            else:
                gaussians.clear_height_constraint()

        loss.backward()

        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_loss
        # loss.backward()

        with torch.no_grad():
            # if iteration < 5000 :
                # Keep track of max radii in image-space for pruning
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # if  iteration % 300 == 0:
                #     size_threshold = 20 
                #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
            pass
                
        gaussians.optimizer.step()

        #### ROAD CONSTRAINTS ####
        if road_constraints_active:
            gaussians.apply_height_constraint_to_gradients(blend=1.0)
            print("road constraint active:", road_constraints_active)

        gaussians.optimizer.zero_grad(set_to_none = True)

        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(10)
    progress_bar.close()
    
    # save gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud_object_inpaint/iteration_{}".format(iteration))
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
        pred_obj = torch.argmax(logits,dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))

        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    out_path = os.path.join(render_path[:-8],'concat')
    makedirs(out_path,exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
    size = (gt.shape[-1]*5,gt.shape[-2])
    fps = float(5) if 'train' in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

    for file_name in sorted(os.listdir(gts_path)):
        gt = np.array(Image.open(os.path.join(gts_path,file_name)))
        rgb = np.array(Image.open(os.path.join(render_path,file_name)))
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path,file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path,file_name)))
        pred_obj = np.array(Image.open(os.path.join(pred_obj_path,file_name)))

        result = np.hstack([gt,rgb,gt_obj,pred_obj,render_obj])
        result = result.astype('uint8')

        Image.fromarray(result).save(os.path.join(out_path,file_name))
        writer.write(result[:,:,::-1])

    writer.release()


def inpaint(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, removal_thresh : float,  finetune_iteration: int, inpaint_strategy: str):
    # 1. load gaussian checkpoint
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    # 2. inpaint selected object
    gaussians = finetune_inpaint(
        opt,
        dataset.model_path,
        scene.loaded_iter,
        scene.getTrainCameras(),
        gaussians,
        pipeline,
        background,
        classifier,
        select_obj_id,
        scene.cameras_extent,
        removal_thresh,
        finetune_iteration,
        inpaint_strategy,
        road_mask_path=getattr(dataset, "road_mask_path", ""),
        min_road_points=getattr(opt, "road_min_points", 128),
        height_loss_weight=getattr(opt, "road_height_loss_weight", 0.2),
        hole_loss_weight=getattr(opt, "road_hole_loss_weight", 0.05),
        min_opacity=getattr(opt, "road_min_opacity", 0.05),
    )

    # 3. render new result
    dataset.object_path = 'object_mask'
    dataset.images = 'images'
    scene = Scene(dataset, gaussians, load_iteration='_object_inpaint/iteration_'+str(finetune_iteration-1), shuffle=False)
    with torch.no_grad():
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--inpaint_strategy", type=str, default="direct", choices=["reseed", "direct"], help="Inpainting strategy: reseed (remove + synthesize) or direct (edit selected gaussians)")

    parser.add_argument("--config_file", type=str, default="config/object_removal/bear.json", help="Path to the configuration file")


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.num_classes = config.get("num_classes", 200)
    args.removal_thresh = config.get("removal_thresh", 0.3)
    args.select_obj_id = config.get("select_obj_id", [34])
    args.images = config.get("images", "images")
    args.object_path = config.get("object_path", "object_mask")
    args.resolution = config.get("r", 1)
    args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    args.finetune_iteration = config.get("finetune_iteration", 10_000)
    args.inpaint_strategy = config.get("inpaint_strategy", args.inpaint_strategy)
    args.road_mask_path = config.get("road_mask_path", "")
    args.road_min_points = config.get("road_min_points", 128)
    args.road_height_loss_weight = config.get("road_height_loss_weight", 0.2)
    args.road_hole_loss_weight = config.get("road_hole_loss_weight", 0.05)
    args.road_min_opacity = config.get("road_min_opacity", 0.05)

    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    inpaint(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        opt.extract(args),
        args.select_obj_id,
        args.removal_thresh,
        args.finetune_iteration,
        args.inpaint_strategy
        )