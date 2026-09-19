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
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
from time import time
from pathlib import Path
import imageio
from utils.camera_utils import generate_interpolated_path, visualizer
from scene.dataset_readers import loadCameras

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def confidence_to_heatmap(confidence_map):
    confidence_np = confidence_map.detach().cpu().numpy()
    confidence_np = np.clip(confidence_np, 0.0, 1.0)
    confidence_uint8 = (confidence_np * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(confidence_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def save_interpolate_pose(model_path, iter, num_views):
    """Generate smooth interpolated camera poses from optimized poses.
    
    Args:
        model_path: Path to model directory
        iter: Iteration number
        num_views: Number of keyframe views
    """
    org_pose = np.load(model_path / f"pose/ours_{iter}/pose_optimized.npy")
    visualizer(org_pose, ["green" for _ in org_pose], model_path / f"pose/ours_{iter}/poses_optimized.png")
    
    n_interp = int(10 * 30 / num_views)  # 10 seconds, fps=30
    all_inter_pose = []
    for i in range(num_views-1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.concatenate(all_inter_pose, axis=0)
    all_inter_pose = np.concatenate([all_inter_pose, org_pose[-1][:3, :].reshape(1, 3, 4)], axis=0)

    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)

    # # around y axis
    # R_y = np.array([
    #     [ c, 0, s, 0],
    #     [ 0, 1, 0, 0],
    #     [-s, 0, c, 0],
    #     [ 0, 0, 0, 1]
    # ])

    # around x axis
    R_x = np.array([
        [1,  0, 0, 0],
        [0,  c, -s, 0],
        [0,  s, c, 0],
        [0,  0, 0, 1]
    ])

    # # around z axis
    # R_z = np.array([
    #     [c, -s, 0, 0],
    #     [s,  c, 0, 0],
    #     [0,  0, 1, 0],
    #     [0,  0, 0, 1]
    # ])


    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        # tmp_view = tmp_view @ R_x

        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    visualizer(inter_pose, ["blue" for _ in inter_pose], model_path / f"pose/ours_{iter}/poses_interpolated.png")
    np.save(model_path / f"pose/ours_{iter}/pose_interpolated.npy", inter_pose)


def images_to_video(image_folder, output_video_path, fps=30):
    """Convert images in a folder to a video.

    Args:
        image_folder (str): Path to folder containing images.
        output_video_path (str): Path where output video will be saved.
        fps (int): Frames per second for output video.
    """
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, is_interpolated=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    confidence_heatmap_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_confidence_heatmap")
    makedirs(render_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(confidence_heatmap_path, exist_ok=True)
    
    # Only create GT-related directories for non-interpolated renders
    if not is_interpolated:
        makedirs(gts_path, exist_ok=True)
        makedirs(gt_colormask_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        
        logits = classifier(rendering_obj)
        probs = torch.softmax(logits, dim=0)
        confidence_map, pred_obj = torch.max(probs, dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
        confidence_heatmap = confidence_to_heatmap(confidence_map)
        
        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(confidence_heatmap).save(os.path.join(confidence_heatmap_path, '{0:05d}'.format(idx) + ".png"))
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
        # Save GT images only for non-interpolated renders
        if not is_interpolated:
            gt_objects = view.objects
            gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))
            Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    # Create concatenated output video only for non-interpolated renders
    if not is_interpolated:
        out_path = os.path.join(render_path[:-8],'concat')
        makedirs(out_path,exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
        size = (gt.shape[-1]*6,gt.shape[-2])
        fps = float(5) if 'train' in out_path else float(1)
        writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

        for file_name in sorted(os.listdir(gts_path)):
            gt = np.array(Image.open(os.path.join(gts_path,file_name)))
            rgb = np.array(Image.open(os.path.join(render_path,file_name)))
            gt_obj = np.array(Image.open(os.path.join(gt_colormask_path,file_name)))
            render_obj = np.array(Image.open(os.path.join(colormask_path,file_name)))
            pred_obj = np.array(Image.open(os.path.join(pred_obj_path,file_name)))
            conf_heat = np.array(Image.open(os.path.join(confidence_heatmap_path,file_name)))

            result = np.hstack([gt,rgb,gt_obj,pred_obj,render_obj,conf_heat])
            result = result.astype('uint8')

            Image.fromarray(result).save(os.path.join(out_path,file_name))
            writer.write(result[:,:,::-1])

        writer.release()
    else:
        # For interpolated renders, create a simple video from rendered images only
        images_to_video(render_path, os.path.join(render_path[:-8], f'{name}_interp_video.mp4'), fps=30)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier)

        # Render interpolated poses for smooth video generation if available
        if args and hasattr(args, 'infer_video') and args.infer_video and not dataset.eval:
            try:
                save_interpolate_pose(Path(dataset.model_path), iterations, args.num_views) # changed iterations to 7000
                interp_pose = np.load(Path(dataset.model_path) / 'pose' / f'ours_{iteration}' / 'pose_interpolated.npy')
                viewpoint_stack = loadCameras(interp_pose, scene.getTrainCameras())
                render_set(
                    dataset.model_path,
                    "interp",
                    scene.loaded_iter,
                    viewpoint_stack,
                    gaussians,
                    pipeline,
                    background,
                    classifier,
                    is_interpolated=True
                )
            except Exception as e:
                print(f"Warning: Could not render interpolated poses: {e}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--infer_video", action="store_true", help="Generate interpolated video with smooth camera poses")
    parser.add_argument("--num_views", default=10, type=int, help="Number of keyframe views for interpolation")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)