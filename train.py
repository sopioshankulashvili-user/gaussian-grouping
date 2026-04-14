# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json
from pathlib import Path
from PIL import Image
import numpy as np
from utils.graphics_utils import geom_transform_points
from height_constraint import (
    detect_flat_surface_height,
    create_road_height_constraint,
    fit_plane_to_points,
    fit_plane_to_points_axis_agnostic,
    compute_constraint_values_along_normal
)


class RoadConstraintManager:
    def __init__(self, road_mask_path, source_path=None):
        self.enabled = bool(road_mask_path)
        self.source_path = Path(source_path) if source_path else None
        self.road_mask_path = Path(road_mask_path) if self.enabled else None
        if self.enabled and self.road_mask_path is not None and not self.road_mask_path.is_absolute() and self.source_path is not None:
            self.road_mask_path = self.source_path / self.road_mask_path
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
        print("road_mask_2d size of mask ", road_mask_2d.shape if road_mask_2d is not None else "None")
        if road_mask_2d is None:
            return None

        width = int(viewpoint_cam.image_width)
        height = int(viewpoint_cam.image_height)
        if road_mask_2d.shape[0] != height or road_mask_2d.shape[1] != width:
            print(f"[RoadConstraint] Warning: Road mask for view '{viewpoint_cam.image_name}' has shape {road_mask_2d.shape}, expected ({height}, {width}). Resizing mask to fit.")
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


def build_road_class_mask(gaussians, classifier, road_class_id=1, probability_threshold=0.5, visible_mask=None):
    with torch.no_grad():
        logits = classifier(gaussians._objects_dc.permute(2, 0, 1))
        probs = torch.softmax(logits, dim=0)

    if road_class_id < 0 or road_class_id >= probs.shape[0]:
        return None

    road_mask = probs[road_class_id, :, 0] >= float(probability_threshold)
    if visible_mask is not None:
        if visible_mask.numel() != road_mask.numel():
            return None
        road_mask = road_mask & visible_mask.to(device=road_mask.device, dtype=torch.bool)
    return road_mask


def prune_gaussians_above_road_plane(
    gaussians,
    margin=0.05,
    up_axis=(0.0, 0.0, 1.0),
    below_margin=None,
    candidate_mask=None,
    protect_mask=None,
):
    if gaussians.plane_normal is None or gaussians.plane_centroid is None:
        return 0

    xyz = gaussians.get_xyz
    if xyz.numel() == 0:
        return 0

    plane_normal = torch.from_numpy(gaussians.plane_normal).float().to(device=xyz.device)
    plane_centroid = torch.from_numpy(gaussians.plane_centroid).float().to(device=xyz.device)

    # up_axis_tensor = torch.tensor(up_axis, dtype=torch.float32, device=xyz.device)
    # up_norm = torch.norm(up_axis_tensor)
    # if up_norm > 0:
    #     up_axis_tensor = up_axis_tensor / up_norm
    #     if torch.dot(plane_normal, up_axis_tensor) < 0:
    #         plane_normal = -plane_normal

    above_margin = float(margin)
    below_margin = float(above_margin if below_margin is None else below_margin)
    signed_distance = ((xyz - plane_centroid.unsqueeze(0)) * plane_normal.unsqueeze(0)).sum(dim=1)
    prune_mask = (signed_distance > above_margin) | (signed_distance < -below_margin)

    if candidate_mask is not None:
        if candidate_mask.numel() != prune_mask.numel():
            return 0
        prune_mask = prune_mask & candidate_mask.to(device=prune_mask.device, dtype=torch.bool)

    if protect_mask is not None:
        if protect_mask.numel() != prune_mask.numel():
            return 0
        prune_mask = prune_mask & (~protect_mask.to(device=prune_mask.device, dtype=torch.bool))

    if not prune_mask.any():
        return 0

    prune_count = int(prune_mask.sum().item())
    gaussians.prune_points(prune_mask)
    return prune_count


def trim_above_plane_from_mask(gaussians, protect_mask, above_margin=0.0):
    if protect_mask is None:
        return None
    if gaussians.plane_normal is None or gaussians.plane_centroid is None:
        return protect_mask

    xyz = gaussians.get_xyz
    if xyz.numel() == 0 or protect_mask.numel() != xyz.shape[0]:
        return protect_mask

    plane_normal = torch.from_numpy(gaussians.plane_normal).float().to(device=xyz.device)
    plane_centroid = torch.from_numpy(gaussians.plane_centroid).float().to(device=xyz.device)
    signed_distance = ((xyz - plane_centroid.unsqueeze(0)) * plane_normal.unsqueeze(0)).sum(dim=1)

    safe_mask = signed_distance <= float(above_margin)
    return protect_mask.to(device=xyz.device, dtype=torch.bool) & safe_mask

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    classifier.cuda()
    road_manager = RoadConstraintManager(
        getattr(dataset, "road_mask_path", "object_mask_road"),
        source_path=getattr(dataset, "source_path", None)
    )
    road_mask_source = str(getattr(opt, "road_mask_source", "class")).lower()
    road_use_class_mask = road_mask_source == "class"
    road_constraints_active = opt.flattened_road and (road_use_class_mask or road_manager.enabled)

    if opt.flattened_road and not road_constraints_active:
        print("[RoadConstraint] Disabled: set road_mask_source='class' or provide --road_mask_path for image-mask mode.")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

        if road_constraints_active and iteration % max(1, opt.road_constraint_every) == 0:
            if road_use_class_mask:
                # print(f"[RoadConstraint] Using classifier-based road mask with class ID {opt.road_class_id} and probability threshold {opt.road_class_prob_threshold}.")
                road_gaussian_mask = build_road_class_mask(
                    gaussians,
                    classifier,
                    road_class_id=opt.road_class_id,
                    probability_threshold=opt.road_class_prob_threshold,
                    visible_mask=visibility_filter,
                )
            else:
                road_gaussian_mask = road_manager.build_visible_road_mask(gaussians, viewpoint_cam, visibility_filter)

            if road_gaussian_mask is not None and road_gaussian_mask.sum().item() >= opt.road_min_points:
                # Use AXIS-AGNOSTIC plane fitting (works with any axis orientation)
                print(f"[RoadConstraint] Applying height constraint using {road_gaussian_mask.sum().item()} visible road points.")
                constraint_values, plane_info = create_road_height_constraint(
                    gaussians, 
                    road_gaussian_mask, 
                    # height_value=1.4, #if we want to hardcode height
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
            else:
                gaussians.clear_height_constraint()
                if road_gaussian_mask is None:
                    print("road gaussian mask: ", road_gaussian_mask)
                    print(f"[RoadConstraint] No usable mask for view '{viewpoint_cam.image_name}'. Skipping until masks are available.")

        # Object Loss
        gt_obj = viewpoint_cam.objects.cuda().long()
        logits = classifier(objects)
        loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        loss_obj = loss_obj / torch.log(torch.tensor(num_classes))  # normalize to (0,1)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss_obj_3d = None
        loss_road_height = None
        loss_road_hole = None
        if iteration % opt.reg3d_interval == 0:
            # regularize at certain intervals
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
            loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj + loss_obj_3d
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj

        # if road_constraints_active and gaussians.height_constrained_mask.numel() == gaussians.get_xyz.shape[0] and gaussians.height_constrained_mask.any():
        #     constrained_mask = gaussians.height_constrained_mask
        #     constrained_xyz = gaussians.get_xyz[constrained_mask]
        #     constrained_targets = gaussians.height_constraint_values[constrained_mask]
            
        #     # Compute height loss based on constraint type (axis-agnostic or legacy)
        #     if gaussians.plane_normal is not None:
        #         # AXIS-AGNOSTIC: Use plane normal direction
        #         import torch as torch_module
        #         plane_normal = torch_module.from_numpy(gaussians.plane_normal).float().to(device=constrained_xyz.device)
        #         plane_centroid = torch_module.from_numpy(gaussians.plane_centroid).float().to(device=constrained_xyz.device)
                
        #         # Compute current distance along plane normal
        #         current_vec = constrained_xyz - plane_centroid.unsqueeze(0)
        #         current_distance = (current_vec * plane_normal.unsqueeze(0)).sum(dim=1)
                
        #         # Distance error from target
        #         loss_road_height = torch.mean((current_distance - constrained_targets) ** 2)
        #     else:
        #         # LEGACY: Z-axis only
        #         loss_road_height = torch.mean((constrained_xyz[:, 2] - constrained_targets) ** 2)
            
        #     loss = loss + opt.road_height_loss_weight * loss_road_height

        #     constrained_opacity = gaussians.get_opacity[constrained_mask].squeeze(-1)
        #     loss_road_hole = torch.relu(opt.road_min_opacity - constrained_opacity).mean()
        #     loss = loss + opt.road_hole_loss_weight * loss_road_hole

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d, use_wandb)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if road_constraints_active and opt.remove_above_road and iteration % max(1, opt.road_remove_every) == 0:
                    prune_protect_mask = None
                    if gaussians.height_constrained_mask.numel() == gaussians.get_xyz.shape[0]:
                        prune_protect_mask = gaussians.height_constrained_mask

                    if road_use_class_mask:
                        class_protect_mask = build_road_class_mask(
                            gaussians,
                            classifier,
                            road_class_id=opt.road_class_id,
                            probability_threshold=opt.road_class_prob_threshold,
                        )
                        if class_protect_mask is not None:
                            prune_protect_mask = class_protect_mask if prune_protect_mask is None else (prune_protect_mask | class_protect_mask)

                    prune_protect_mask = trim_above_plane_from_mask(
                        gaussians,
                        prune_protect_mask,
                        above_margin=opt.road_above_margin,
                    )

                    pruned_count = prune_gaussians_above_road_plane(
                        gaussians,
                        margin=opt.road_above_margin,
                        below_margin=getattr(opt, "road_below_margin", opt.road_above_margin),
                        up_axis=opt.road_up_axis,
                        protect_mask=prune_protect_mask,
                    )
                    if pruned_count > 0:
                        print(
                            f"[RoadConstraint] Pruned {pruned_count} off-plane gaussians "
                            f"(above={opt.road_above_margin:.4f}, below={getattr(opt, 'road_below_margin', opt.road_above_margin):.4f})."
                        )
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                # for every thousand iterations
                if road_constraints_active and iteration % 1000 == 0:
                    gaussians.apply_height_constraint_to_gradients(blend=opt.road_height_blend)
                gaussians.optimizer.zero_grad(set_to_none = True)
                cls_optimizer.step()
                cls_optimizer.zero_grad()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if road_constraints_active and opt.remove_above_road:
        with torch.no_grad():
            prune_protect_mask = None
            if gaussians.height_constrained_mask.numel() == gaussians.get_xyz.shape[0]:
                prune_protect_mask = gaussians.height_constrained_mask

            if road_use_class_mask:
                class_protect_mask = build_road_class_mask(
                    gaussians,
                    classifier,
                    road_class_id=opt.road_class_id,
                    probability_threshold=opt.road_class_prob_threshold,
                )
                if class_protect_mask is not None:
                    prune_protect_mask = class_protect_mask if prune_protect_mask is None else (prune_protect_mask | class_protect_mask)

            prune_protect_mask = trim_above_plane_from_mask(
                gaussians,
                prune_protect_mask,
                above_margin=opt.road_above_margin,
            )

            pruned_count = prune_gaussians_above_road_plane(
                gaussians,
                margin=opt.road_above_margin,
                below_margin=getattr(opt, "road_below_margin", opt.road_above_margin),
                up_axis=opt.road_up_axis,
                protect_mask=prune_protect_mask,
            )
            if pruned_count > 0:
                print(
                    f"[RoadConstraint] Post-training prune removed {pruned_count} off-plane gaussians "
                    f"(above={opt.road_above_margin:.4f}, below={getattr(opt, 'road_below_margin', opt.road_above_margin):.4f})."
                )
            else:
                print("[RoadConstraint] Post-training prune removed 0 gaussians.")

            print("\n[POST] Saving pruned final Gaussians")
            scene.save(opt.iterations)
            torch.save(
                classifier.state_dict(),
                os.path.join(scene.model_path, f"point_cloud/iteration_{opt.iterations}", "classifier.pth")
            )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, use_wandb):

    if use_wandb:
        if loss_obj_3d:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "train_loss_patches/loss_obj_3d": loss_obj_3d.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "iter_time": elapsed, "iter": iteration})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if use_wandb:
                        if idx < 5:
                            wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)]})
                            if iteration == testing_iterations[0]:
                                wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [wandb.Image(gt_image)]})
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test, config['name'] + "/loss_viewpoint - psnr": psnr_test})
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity, "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

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

    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", 200)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)
    args.flattened_road = config.get("flattened_road", args.flattened_road)
    args.road_constraint_every = config.get("road_constraint_every", args.road_constraint_every)
    args.road_min_points = config.get("road_min_points", args.road_min_points)
    args.road_height_loss_weight = config.get("road_height_loss_weight", args.road_height_loss_weight)
    args.road_hole_loss_weight = config.get("road_hole_loss_weight", args.road_hole_loss_weight)
    args.road_min_opacity = config.get("road_min_opacity", args.road_min_opacity)
    args.road_height_blend = config.get("road_height_blend", args.road_height_blend)
    args.remove_above_road = config.get("remove_above_road", args.remove_above_road)
    args.road_remove_every = config.get("road_remove_every", args.road_remove_every)
    args.road_above_margin = config.get("road_above_margin", args.road_above_margin)
    args.road_below_margin = config.get("road_below_margin", getattr(args, "road_below_margin", args.road_above_margin))
    args.road_mask_source = config.get("road_mask_source", args.road_mask_source)
    args.road_class_id = config.get("road_class_id", args.road_class_id)
    args.road_class_prob_threshold = config.get("road_class_prob_threshold", args.road_class_prob_threshold)

    road_up_axis_cfg = config.get("road_up_axis", args.road_up_axis)
    if isinstance(road_up_axis_cfg, str):
        axis_vals = [float(v.strip()) for v in road_up_axis_cfg.split(",") if v.strip()]
    else:
        axis_vals = list(road_up_axis_cfg)
    if len(axis_vals) != 3:
        raise ValueError("road_up_axis must contain 3 values, e.g. [0, 0, 1] or '0,0,1'.")
    args.road_up_axis = tuple(axis_vals)
    
    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="gaussian-splatting")
        wandb.config.args = args
        wandb.run.name = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb)

    # All done
    print("\nTraining complete.")
