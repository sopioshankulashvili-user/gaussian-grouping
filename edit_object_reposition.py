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
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from edit_object_removal import points_inside_convex_hull
from gaussian_renderer import GaussianModel, render
from height_constraint import create_road_height_constraint
from render import feature_to_rgb, visualize_obj
from scene import Scene
from scene.dataset_readers import loadCameras
from utils.general_utils import safe_state
from utils.camera_utils import generate_interpolated_path, visualizer
from utils.loss_utils import ssim


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

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

        image_file = self.pseudo_gt_path / f"{image_name}.jpeg" if (self.pseudo_gt_path / f"{image_name}.jpeg").exists() else self.pseudo_gt_path / f"{image_name}.png"

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


def _normalize_quaternions(quaternions):
    return quaternions / torch.clamp(torch.norm(quaternions, dim=1, keepdim=True), min=1e-12)


def _quaternion_multiply(q1, q2):
    """Multiply two batches of quaternions in (w, x, y, z) format."""
    w1, x1, y1, z1 = q1.unbind(dim=1)
    w2, x2, y2, z2 = q2.unbind(dim=1)

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=1,
    )


def _rotation_vector_to_axis_angle(rotation_vector, device, dtype):
    """Interpret a 3D Rodrigues vector as axis * angle.

    The vector direction is the rotation axis (plane normal), and its norm is
    the rotation angle in degrees.
    """
    rotation_tensor = torch.as_tensor(rotation_vector, dtype=dtype, device=device).flatten()
    if rotation_tensor.numel() != 3:
        raise ValueError("Rotation must contain exactly three values: a Rodrigues vector [rx, ry, rz].")

    angle_degrees = torch.norm(rotation_tensor)
    if angle_degrees.item() == 0:
        return None, None

    axis = rotation_tensor / torch.clamp(angle_degrees, min=1e-12)
    return axis, angle_degrees


def _axis_angle_to_rotation_matrix(axis, angle_degrees, device, dtype):
    axis = torch.as_tensor(axis, dtype=dtype, device=device).flatten()
    if axis.numel() != 3:
        raise ValueError("Rotation axis must contain exactly three values.")

    axis_norm = torch.norm(axis)
    if axis_norm.item() == 0:
        return torch.eye(3, dtype=dtype, device=device)

    axis = axis / torch.clamp(axis_norm, min=1e-12)
    theta = torch.deg2rad(torch.as_tensor(angle_degrees, dtype=dtype, device=device))
    c = torch.cos(theta)
    s = torch.sin(theta)
    one_minus_c = 1.0 - c

    x, y, z = axis
    K = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=dtype,
        device=device,
    )
    axis_outer = axis.unsqueeze(1) @ axis.unsqueeze(0)
    identity = torch.eye(3, dtype=dtype, device=device)
    return c * identity + one_minus_c * axis_outer + s * K


def _axis_angle_to_quaternion(axis, angle_degrees, device, dtype):
    axis = torch.as_tensor(axis, dtype=dtype, device=device).flatten()
    if axis.numel() != 3:
        raise ValueError("Rotation axis must contain exactly three values.")

    axis_norm = torch.norm(axis)
    if axis_norm.item() == 0:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)

    axis = axis / torch.clamp(axis_norm, min=1e-12)
    theta = torch.deg2rad(torch.as_tensor(angle_degrees, dtype=dtype, device=device))
    half_theta = theta * 0.5
    quat = torch.cat([torch.cos(half_theta).unsqueeze(0), axis * torch.sin(half_theta)])
    return _normalize_quaternions(quat.unsqueeze(0))[0]


def _resolve_rotation_axis_and_angle(gaussians, rotation_vector, device, dtype):
    """Resolve axis-angle rotation using plane normal as axis when available.

    `rotation_vector` can be either:
      - scalar angle in degrees, or
      - Rodrigues vector [rx, ry, rz] (legacy path)
    """
    rotation_tensor = torch.as_tensor(rotation_vector, dtype=dtype, device=device).flatten()
    if rotation_tensor.numel() == 1:
        angle_degrees = torch.abs(rotation_tensor[0])
        angle_sign = torch.sign(rotation_tensor[0])
    elif rotation_tensor.numel() == 3:
        angle_degrees = torch.norm(rotation_tensor)
        angle_sign = torch.tensor(1.0, dtype=dtype, device=device)
    else:
        raise ValueError("Rotation must be either a scalar angle (degrees) or a Rodrigues vector [rx, ry, rz].")

    if angle_degrees.item() == 0:
        return None, None

    plane_normal = getattr(gaussians, "plane_normal", None)
    if plane_normal is not None:
        axis = torch.as_tensor(plane_normal, dtype=dtype, device=device).flatten()
        if axis.numel() != 3:
            raise ValueError("gaussians.plane_normal must contain exactly three values.")
        axis_norm = torch.norm(axis)
        if axis_norm.item() > 0:
            axis = axis / torch.clamp(axis_norm, min=1e-12)
            if angle_sign.item() < 0:
                axis = -axis
            return axis, angle_degrees

    if rotation_tensor.numel() == 1:
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        if angle_sign.item() < 0:
            axis = -axis
        return axis, angle_degrees

    axis = rotation_tensor / torch.clamp(angle_degrees, min=1e-12)
    return axis, angle_degrees


def _store_plane_normal_and_centroid(gaussians, axis, centroid):
    if axis is None or centroid is None:
        return
    gaussians.plane_normal = axis.detach().cpu().numpy()
    gaussians.plane_centroid = centroid.detach().cpu().numpy()


def _ensure_plane_info_for_rotation(gaussians, plane_mask):
    """Populate gaussians.plane_normal/plane_centroid using height-constraint logic."""
    if getattr(gaussians, "plane_normal", None) is not None and getattr(gaussians, "plane_centroid", None) is not None:
        print("[Reposition] Plane normal and centroid already exist, skipping plane estimation.")
        print(f"[Reposition] Stored plane normal: {gaussians.plane_normal}")
        print(f"[Reposition] Stored plane centroid: {gaussians.plane_centroid}")
        return True

    if plane_mask is None:
        return False

    plane_mask = plane_mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).flatten()
    if plane_mask.numel() != gaussians.get_xyz.shape[0]:
        return False
    if int(plane_mask.sum().item()) < 3:
        return False

    try:
        create_road_height_constraint(
            gaussians,
            plane_mask,
            height_value=1e-2,
            method="fit_plane_axis_agnostic",
        )
    except Exception as exc:
        print(f"[Reposition] Failed to estimate plane info for rotation: {exc}")
        return False
    
    return getattr(gaussians, "plane_normal", None) is not None and getattr(gaussians, "plane_centroid", None) is not None


def _apply_rotation_to_gaussian_subset(gaussians, mask3d, rotation_degrees, translate_center=None):
    rotation_axis, rotation_angle = _resolve_rotation_axis_and_angle(
        gaussians, rotation_degrees, gaussians.get_xyz.device, gaussians.get_xyz.dtype
    )
    if rotation_axis is None:
        return None

    selected_xyz = gaussians._xyz.data[mask3d].detach().clone()
    if selected_xyz.shape[0] == 0:
        return None

    if translate_center is None:
        center = selected_xyz.mean(dim=0)
    else:
        center = torch.as_tensor(translate_center, dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)

    _store_plane_normal_and_centroid(gaussians, rotation_axis, center)
    delta_quaternion = _axis_angle_to_quaternion(rotation_axis, rotation_angle, gaussians.get_xyz.device, gaussians.get_xyz.dtype)
    rotation_matrix = _axis_angle_to_rotation_matrix(rotation_axis, rotation_angle, gaussians.get_xyz.device, gaussians.get_xyz.dtype)

    with torch.no_grad():
        centered_xyz = selected_xyz - center.unsqueeze(0)
        rotated_xyz = centered_xyz @ rotation_matrix.t() + center.unsqueeze(0)
        gaussians._xyz.data[mask3d] = rotated_xyz

        selected_rotation = gaussians._rotation.data[mask3d].detach().clone()
        if selected_rotation.shape[1] != 4:
            raise ValueError("Gaussian rotations are expected to be quaternions with four components.")

        delta_quaternion_batch = delta_quaternion.unsqueeze(0).expand(selected_rotation.shape[0], -1)
        rotated_quaternion = _quaternion_multiply(delta_quaternion_batch, selected_rotation)
        gaussians._rotation.data[mask3d] = _normalize_quaternions(rotated_quaternion)

    return center


def apply_rotation_to_selected_gaussians(gaussians, mask3d, rotation_degrees):
    _apply_rotation_to_gaussian_subset(gaussians, mask3d, rotation_degrees)


def duplicate_and_rotate_selected_gaussians(gaussians, mask3d, rotation_degrees):
    """
    Keep the original gaussians in place and create a rotated duplicate copy.

    Returns:
        train_mask_expanded: Bool mask of length N+M for optimization.
            Marks original selected gaussians and rotated duplicates.
        rotated_only_mask: Bool mask of length N+M marking only rotated duplicates.
    """
    rotation_axis, rotation_angle = _resolve_rotation_axis_and_angle(
        gaussians, rotation_degrees, gaussians.get_xyz.device, gaussians.get_xyz.dtype
    )
    n_original = gaussians._xyz.shape[0]
    n_selected = int(mask3d.sum().item())

    if rotation_axis is None or n_selected == 0:
        rotated_only_mask = torch.zeros_like(mask3d, dtype=torch.bool)
        return mask3d, rotated_only_mask

    with torch.no_grad():
        selected_xyz = gaussians._xyz[mask3d].detach().clone()
        center = selected_xyz.mean(dim=0)
        _store_plane_normal_and_centroid(gaussians, rotation_axis, center)
        delta_quaternion = _axis_angle_to_quaternion(rotation_axis, rotation_angle, gaussians.get_xyz.device, gaussians.get_xyz.dtype)
        rotation_matrix = _axis_angle_to_rotation_matrix(rotation_axis, rotation_angle, gaussians.get_xyz.device, gaussians.get_xyz.dtype)

        xyz_new = (selected_xyz - center.unsqueeze(0)) @ rotation_matrix.t() + center.unsqueeze(0)
        features_dc_new = gaussians._features_dc[mask3d].detach().clone()
        features_rest_new = gaussians._features_rest[mask3d].detach().clone()
        opacity_new = gaussians._opacity[mask3d].detach().clone()
        scaling_new = gaussians._scaling[mask3d].detach().clone()
        rotation_new = gaussians._rotation[mask3d].detach().clone()
        objects_dc_new = gaussians._objects_dc[mask3d].detach().clone()

        delta_quaternion_batch = delta_quaternion.unsqueeze(0).expand(rotation_new.shape[0], -1)
        rotation_new = _normalize_quaternions(_quaternion_multiply(delta_quaternion_batch, rotation_new))

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

    rotated_only_mask = torch.zeros((n_original + n_selected), device=mask3d.device, dtype=torch.bool)
    rotated_only_mask[n_original:] = True

    return train_mask_expanded, rotated_only_mask


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


def _render_with_active_mask(view, gaussians, pipeline, background, active_mask):
    """Render while temporarily disabling gaussians outside active_mask."""
    active_mask = active_mask.to(device=gaussians._opacity.device, dtype=torch.bool).flatten()
    original_opacity = gaussians._opacity.data.clone()
    try:
        inactive_mask = ~active_mask
        if inactive_mask.any():
            from utils.general_utils import inverse_sigmoid

            gaussians._opacity.data[inactive_mask] = inverse_sigmoid(
                torch.full((int(inactive_mask.sum().item()), 1), 1e-6, device=gaussians._opacity.device, dtype=gaussians._opacity.dtype)
            )
        return render(view, gaussians, pipeline, background)
    finally:
        gaussians._opacity.data.copy_(original_opacity)


def _composite_two_passes(bg_pkg, fg_pkg, background):
    bg_render = bg_pkg["render"]      # [3, H, W]
    fg_render = fg_pkg["render"]      # [3, H, W]
    
    # Use the foreground's actual accumulated opacity (Alpha)
    # Ensure this is [1, H, W]
    fg_alpha = fg_pkg["opacity"] 

    # Standard "Over" operator: Result = FG + (1 - Alpha_FG) * BG
    # This assumes FG is already premultiplied (Standard in 3DGS)
    render_out = fg_render + (1.0 - fg_alpha) * bg_render

    # Composite the object IDs/Auxiliary maps similarly or with a hard threshold
    fg_mask = fg_alpha > 0.5
    render_obj_out = torch.where(fg_mask, fg_pkg["render_object"], bg_pkg["render_object"])
    
    return render_out, render_obj_out


def _render_with_optional_two_pass(view, gaussians, pipeline, background, fg_mask=None):
    fg_mask = getattr(gaussians, "reposition_foreground_mask", fg_mask)
    if fg_mask is None or fg_mask.numel() != gaussians._xyz.shape[0]:
        return render(view, gaussians, pipeline, background)

    fg_mask = fg_mask.to(device=gaussians._xyz.device, dtype=torch.bool).flatten()
    bg_pkg = _render_with_active_mask(view, gaussians, pipeline, background, ~fg_mask)
    fg_pkg = _render_with_active_mask(view, gaussians, pipeline, background, fg_mask)
    rendering, rendering_obj = _composite_two_passes(bg_pkg, fg_pkg, background)
    return {"render": rendering, "render_object": rendering_obj}


def save_interpolate_pose(model_path, iteration, num_views):
    """Generate smooth interpolated camera poses from optimized poses."""
    model_path = Path(model_path)
    org_pose = np.load(model_path / f"pose/ours_{iteration}/pose_optimized.npy")
    visualizer(org_pose, ["green" for _ in org_pose], model_path / f"pose/ours_{iteration}/poses_optimized.png")

    n_interp = int(10 * 30 / num_views)
    all_inter_pose = []
    for i in range(num_views - 1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i : i + 2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.concatenate(all_inter_pose, axis=0)
    all_inter_pose = np.concatenate([all_inter_pose, org_pose[-1][:3, :].reshape(1, 3, 4)], axis=0)

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)

    inter_pose = np.stack(inter_pose_list, 0)
    visualizer(inter_pose, ["blue" for _ in inter_pose], model_path / f"pose/ours_{iteration}/poses_interpolated.png")
    np.save(model_path / f"pose/ours_{iteration}/pose_interpolated.npy", inter_pose)


def images_to_video(image_folder, output_video_path, fps=30):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith((".png", ".jpg", ".jpeg", ".JPG", ".PNG")):
            image_path = os.path.join(image_folder, filename)
            images.append(imageio.imread(image_path))
    imageio.mimwrite(output_video_path, images, fps=fps)


def render_interpolated_set(model_path, name, iteration, pose_iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, f"ours_interp_{iteration}", "renders")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering interpolated progress")):
        results = _render_with_optional_two_pass(view, gaussians, pipeline, background)
        rendering = results["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        #image saved on path
        print(f"Saved rendered image to {os.path.join(render_path, f'{idx:05d}.png')}")

    output_video_name = f"{name}_interp_video.mp4"
    images_to_video(render_path, os.path.join(render_path[:-8], output_video_name), fps=30)
    print(f"Video saved to {os.path.join(render_path[:-8], output_video_name)}")


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
                center = translated_points.mean(axis=0)
                shrink_factor = 0.8  # < 1.0 shrinks the hull

                translated_points_shrunk = center + shrink_factor * (translated_points - center)
                hull = Delaunay(translated_points_shrunk)
                # hull = Delaunay(translated_points)
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
    rotation,
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

    has_translation = np.abs(np.asarray(translation, dtype=np.float32)).sum() > 0
    has_rotation = abs(float(rotation)) > 0

    if has_rotation:
        road_gaussian_mask = build_road_class_mask(
                    gaussians,
                    classifier,
                    road_class_id=1,
                    probability_threshold=0.4,
                    visible_mask=None,
                )
        _ensure_plane_info_for_rotation(gaussians, road_gaussian_mask)

    if keep_original_gaussians:
        if has_translation:
            mask3d_for_optimizer, target_anchor_mask = duplicate_and_translate_selected_gaussians(
                gaussians, mask3d, translation
            )
        else:
            mask3d_for_optimizer = mask3d

        if has_rotation:
            if has_translation:
                apply_rotation_to_selected_gaussians(gaussians, target_anchor_mask, rotation)
            else:
                mask3d_for_optimizer, target_anchor_mask = duplicate_and_rotate_selected_gaussians(
                    gaussians, mask3d, rotation
                )
    else:
        if has_translation:
            apply_translation_to_selected_gaussians(gaussians, mask3d, translation)
        if has_rotation:
            apply_rotation_to_selected_gaussians(gaussians, mask3d, rotation)
        mask3d_for_optimizer = mask3d

    # if has_rotation and getattr(gaussians, "plane_normal", None) is not None:
    #     print(f"[Reposition] Stored plane normal: {gaussians.plane_normal}")
    #     if getattr(gaussians, "plane_centroid", None) is not None:
    #         print(f"[Reposition] Stored plane centroid: {gaussians.plane_centroid}")
    
    mask3d_for_pseudo_repositioned = mask3d_for_optimizer

    # Store the translated foreground mask for two-pass rendering.
    gaussians.reposition_foreground_mask = target_anchor_mask.clone()

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

        with torch.no_grad():
            if iteration < 5000 :
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if  iteration % 100 == 0:
                    size_threshold = 20 
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
  

        if enable_opacity_blending:
            reduce_opacity_in_destination(
                gaussians, 
                target_anchor_mask,
                target_opacity=opacity_blend_target,
                blend_radius=opacity_blend_radius
            )

        gaussians.optimizer.step()

        
        gaussians.optimizer.zero_grad(set_to_none=True)

        if i % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.7f}"})
            progress_bar.update(10)
    progress_bar.close()


    # removed_mask = remove_gaussians_in_destination(gaussians, target_anchor_mask)
    #                 if removed_mask is not None:
    #                         keep_mask = ~removed_mask
    #                         mask3d_for_optimizer = mask3d_for_optimizer[keep_mask]
    #                         target_anchor_mask = target_anchor_mask[keep_mask]
    #                         source_neighborhood_mask = source_neighborhood_mask[keep_mask]


    point_cloud_path = os.path.join(model_path, f"point_cloud_object_reposition/iteration_{iteration}")
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    return gaussians


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, fix_boundary_stretching=True, boundary_shrink_factor=0.85):
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred")
    pointcloud_path = os.path.join(model_path, name, "ours{}".format(iteration), "point_cloud")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(pointcloud_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = _render_with_optional_two_pass(view, gaussians, pipeline, background)
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
    rotation,
    pseudo_gt_path,
    enable_opacity_blending: bool = False,
    opacity_blend_target: float = 0.0,
    opacity_blend_radius: float = 0.1,
    keep_original_gaussians: bool = False,
    infer_video: bool = False,
    num_views: int = 10,
):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    pose_iteration = scene.loaded_iter
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
        rotation,
        pseudo_gt_path,
        lambda_ssim=getattr(opt, "reposition_lambda_ssim", 0.2),
        enable_opacity_blending=enable_opacity_blending,
        opacity_blend_target=opacity_blend_target,
        opacity_blend_radius=opacity_blend_radius,
        keep_original_gaussians=keep_original_gaussians,
    )

    dataset.object_path = "object_mask"
    dataset.images = "images"
    scene = Scene(dataset, gaussians, load_iteration=f"_object_reposition/iteration_{scene.loaded_iter}", shuffle=False)

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

        if infer_video:
            try:
                # save_interpolate_pose(Path(dataset.model_path), pose_iteration, num_views)
                interp_pose = np.load(Path(dataset.model_path) / "pose" / f"ours_{pose_iteration}" / "pose_interpolated.npy")
                viewpoint_stack = loadCameras(interp_pose, scene.getTrainCameras())
                render_interpolated_set(
                    dataset.model_path,
                    "interp",
                    scene.loaded_iter,
                    pose_iteration,
                    viewpoint_stack,
                    gaussians,
                    pipeline,
                    background,
                )
            except Exception as e:
                print(f"Warning: Could not render interpolated poses: {e}")
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
    parser.add_argument("--rotation_angle", type=float, default=0.0, help="Rotation angle in degrees around plane normal")
    parser.add_argument("--pseudo_gt_path", type=str, default="", help="Directory containing pseudo-GT images as <image_name>.png")
    parser.add_argument("--enable_opacity_blending", action="store_true", help="Enable opacity reduction in destination area for better blending")
    parser.add_argument("--opacity_blend_target", type=float, default=1e-5, help="Target opacity for gaussians in destination area")
    parser.add_argument("--opacity_blend_radius", type=float, default=0.1, help="Radius around translated gaussians to affect for blending")
    parser.add_argument("--keep_original_gaussians", action="store_true", help="Keep the original gaussians in place and duplicate them at the transformed location")
    parser.add_argument("--infer_video", action="store_true", help="Generate an interpolated video with smooth camera poses")
    parser.add_argument("--num_views", default=10, type=int, help="Number of keyframe views for interpolation")

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

    cfg_rotation = config.get("rotation", None)
    if cfg_rotation is not None:
        if isinstance(cfg_rotation, list):
            if len(cfg_rotation) != 3:
                raise ValueError("Config key 'rotation' must be a list [rx, ry, rz] or provide 'rotation_angle'.")
            args.rotation_angle = float(np.linalg.norm(np.asarray(cfg_rotation, dtype=np.float32)))
            print("[Reposition] Legacy 'rotation' vector detected; converted to scalar 'rotation_angle'.")
        else:
            args.rotation_angle = float(cfg_rotation)
    else:
        if "rotation_angle" in config:
            args.rotation_angle = float(config.get("rotation_angle", args.rotation_angle))
        elif any(k in config for k in ["rotation_rx", "rotation_ry", "rotation_rz"]):
            rx = float(config.get("rotation_rx", 0.0))
            ry = float(config.get("rotation_ry", 0.0))
            rz = float(config.get("rotation_rz", 0.0))
            args.rotation_angle = float(np.linalg.norm(np.asarray([rx, ry, rz], dtype=np.float32)))
            print("[Reposition] Legacy rotation_rx/ry/rz detected; converted to scalar 'rotation_angle'.")

    args.pseudo_gt_path = config.get("pseudo_gt_path", args.pseudo_gt_path)

    args.enable_opacity_blending = config.get("enable_opacity_blending", args.enable_opacity_blending)
    args.opacity_blend_target = config.get("opacity_blend_target", args.opacity_blend_target)
    args.opacity_blend_radius = config.get("opacity_blend_radius", args.opacity_blend_radius)
    args.keep_original_gaussians = config.get("keep_original_gaussians", args.keep_original_gaussians)
    args.infer_video = config.get("infer_video", args.infer_video)
    args.num_views = config.get("num_views", args.num_views)

    translation = [args.translation_dx, args.translation_dy, args.translation_dz]
    rotation = float(args.rotation_angle)
    print(f"Using translation: {translation}")
    print(f"Using rotation angle (degrees): {rotation}")
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
        rotation,
        args.pseudo_gt_path,
        enable_opacity_blending=args.enable_opacity_blending,
        opacity_blend_target=args.opacity_blend_target,
        opacity_blend_radius=args.opacity_blend_radius,
        keep_original_gaussians=args.keep_original_gaussians,
        infer_video=args.infer_video,
        num_views=args.num_views,
    )
