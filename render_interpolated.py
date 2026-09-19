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
from pathlib import Path
import imageio
from utils.camera_utils import generate_interpolated_path, visualizer
from scene.dataset_readers import loadCameras

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

    # around x axis
    R_x = np.array([
        [1,  0, 0, 0],
        [0,  c, -s, 0],
        [0,  s, c, 0],
        [0,  0, 0, 1]
    ])

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
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

def render_interpolated_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, f"ours_interp_{iteration}", "renders")
    makedirs(render_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering interpolated progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        
        # Save standard RGB render
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
    # Create the interpolated video from the saved frames
    output_video_name = f'{name}_interp_video.mp4'
    images_to_video(render_path, os.path.join(render_path[:-8], output_video_name), fps=30)
    print(f"Video saved to {os.path.join(render_path[:-8], output_video_name)}")

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, args=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print("Generating and rendering interpolated poses...")
        try:
            # Replaced the hardcoded '7000' with the dynamically loaded iteration from the scene
            save_interpolate_pose(Path(dataset.model_path), 7000, args.num_views) 
            interp_pose = np.load(Path(dataset.model_path) / 'pose' / f'ours_7000' / 'pose_interpolated.npy')
            viewpoint_stack = loadCameras(interp_pose, scene.getTrainCameras())
            
            render_interpolated_set(
                dataset.model_path,
                "interp",
                scene.loaded_iter,
                viewpoint_stack,
                gaussians,
                pipeline,
                background
            )
        except FileNotFoundError as e:
            print(f"Error: Missing pose files required for interpolation. {e}")
        except Exception as e:
            print(f"Warning: Could not render interpolated poses: {e}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Interpolated Video Rendering Script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num_views", default=10, type=int, help="Number of keyframe views for interpolation")
    args = get_combined_args(parser)
    print("Rendering Interpolated Video for: " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)