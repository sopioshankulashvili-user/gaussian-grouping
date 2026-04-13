# Copyright (C) 2024, Height Constraint Utilities for Gaussian-Grouping
# Utilities for constraining Gaussians to maintain flat surfaces (e.g., roads)
# AXIS-AGNOSTIC: Works with any axis orientation

import torch
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import least_squares


def _fit_plane_from_points_numpy(points):
    """Fit a plane to 3D points using covariance eigen-analysis.

    Returns:
        centroid (np.ndarray), normal (np.ndarray), d (float)
    """
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    cov_matrix = (centered_points.T @ centered_points) / max(len(points), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / normal_norm
    d = -float(np.dot(normal, centroid))
    return centroid, normal, d


def fit_plane_to_points_ransac(xyz, mask, return_full_info=False, n_iters=256, distance_threshold=0.02, min_inliers=16, random_state=0):
    """
    Robustly fit a plane to masked points using a simple RANSAC loop.

    Args:
        xyz: Tensor/array of shape (N, 3)
        mask: Boolean mask selecting candidate road points
        return_full_info: If True, return (centroid, normal, d)
        n_iters: Number of RANSAC iterations
        distance_threshold: Inlier threshold measured as absolute distance to plane
        min_inliers: Minimum inliers needed to accept a candidate plane
        random_state: Seed for deterministic sampling

    Returns:
        Same return shape as fit_plane_to_points_axis_agnostic() when successful.
    """
    if not mask.any():
        if return_full_info:
            return np.array([0, 0, 0]), np.array([0, 0, 1]), 0
        return {
            'centroid': np.array([0, 0, 0]),
            'normal': np.array([0, 0, 1]),
            'd': 0,
        }

    points = xyz[mask].detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else np.asarray(xyz[mask])
    if points.shape[0] < 3:
        return fit_plane_to_points_axis_agnostic(xyz, mask, return_full_info=return_full_info)

    rng = np.random.default_rng(random_state)
    best_inlier_count = -1
    best_inlier_mask = None
    best_model = None

    # Degenerate / tiny sets are handled by the fallback below.
    for _ in range(int(n_iters)):
        sample_idx = rng.choice(points.shape[0], size=3, replace=False)
        p1, p2, p3 = points[sample_idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p1))

        distances = np.abs(points @ normal + d)
        inlier_mask = distances < float(distance_threshold)
        inlier_count = int(inlier_mask.sum())
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_model = (normal, d)

    if best_model is None or best_inlier_mask is None or best_inlier_count < max(3, int(min_inliers)):
        return fit_plane_to_points_axis_agnostic(xyz, mask, return_full_info=return_full_info)

    inlier_points = points[best_inlier_mask]
    centroid, normal, d = _fit_plane_from_points_numpy(inlier_points)

    # Keep orientation stable.
    if best_model is not None and np.dot(normal, best_model[0]) < 0:
        normal = -normal
        d = -d

    if return_full_info:
        return centroid, normal, d
    return {
        'centroid': centroid,
        'normal': normal,
        'd': d,
    }


def detect_flat_surface_height(xyz, mask, outlier_percentile=5, verbose=False):
    """
    Detect the height of a flat surface by analyzing Z-coordinates of points in the masked region.
    DEPRECATED: Use fit_plane_to_points_axis_agnostic() for axis-agnostic constraints.
    
    Args:
        xyz: Tensor of shape (N, 3) containing Gaussian positions
        mask: Boolean tensor of shape (N,) indicating which points belong to the surface
        outlier_percentile: Percentile for outlier removal (removes top and bottom values)
        verbose: Whether to print debug information
    
    Returns:
        Estimated height (Z-coordinate) of the flat surface
    """
    if not mask.any():
        return 0.0
    
    surface_points = xyz[mask]
    z_coords = surface_points[:, 2]
    
    # Remove outliers based on percentile
    lower_bound = torch.quantile(z_coords, outlier_percentile / 100.0)
    upper_bound = torch.quantile(z_coords, 1 - outlier_percentile / 100.0)
    
    inliers = (z_coords >= lower_bound) & (z_coords <= upper_bound)
    estimated_height = z_coords[inliers].mean().item()
    
    if verbose:
        print(f"Surface Height Estimation:")
        print(f"  Range (Z): [{z_coords.min():.4f}, {z_coords.max():.4f}]")
        print(f"  Inliers after {outlier_percentile}% removal: {inliers.sum().item()}/{len(z_coords)}")
        print(f"  Estimated height: {estimated_height:.4f}")
    
    return estimated_height


def fit_plane_to_points(xyz, mask, return_plane_params=False):
    """
    Fit a plane to the masked points using least squares.
    DEPRECATED: Use fit_plane_to_points_axis_agnostic() for axis-agnostic constraints.
    
    Args:
        xyz: Tensor of shape (N, 3) containing Gaussian positions
        mask: Boolean tensor of shape (N,) indicating which points belong to the surface
        return_plane_params: If True, returns plane equation parameters (a, b, c, d) for ax+by+cz+d=0
    
    Returns:
        height: Float value representing the Z-coordinate of the fitted plane at centroid
        OR (height, plane_params) if return_plane_params=True
    """
    if not mask.any():
        return 0.0 if not return_plane_params else (0.0, None)
    
    points = xyz[mask].detach().cpu().numpy()
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    
    # Memory-efficient plane normal computation using covariance matrix
    # (instead of full SVD which would be O(N^2) memory for large clouds)
    cov_matrix = (centered_points.T @ centered_points) / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, np.argmin(eigenvalues)]  # Smallest eigenvalue = plane normal
    
    # Plane equation: normal · (p - centroid) = 0
    d = -np.dot(normal, centroid)
    
    # Height at centroid is centroid[2]
    height = centroid[2]
    
    if return_plane_params:
        return height, (normal, d)
    return height


def fit_plane_to_points_axis_agnostic(xyz, mask, return_full_info=False):
    """
    Fit a plane to masked points and return the plane normal (axis-agnostic).
    This function works with any axis orientation - determines the true vertical direction.
    
    Args:
        xyz: Tensor of shape (N, 3) containing Gaussian positions
        mask: Boolean tensor of shape (N,) indicating which points belong to the surface
        return_full_info: If True, returns (centroid, normal, d) else just returns dict
    
    Returns:
        If return_full_info=True:
            (centroid, normal, d) where normal is the plane normal vector
        Else:
            Dictionary with keys:
                'centroid': Center of the plane
                'normal': Unit normal vector to the plane (shape: 3,)
                'd': Plane equation constant (ax+by+cz+d=0)
    """
    if not mask.any():
        if return_full_info:
            return np.array([0, 0, 0]), np.array([0, 0, 1]), 0
        else:
            return {
                'centroid': np.array([0, 0, 0]),
                'normal': np.array([0, 0, 1]),
                'd': 0
            }
    
    points = xyz[mask].detach().cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz[mask]
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    
    # Memory-efficient plane normal computation using covariance matrix
    # (instead of full SVD which would be O(N^2) memory for large clouds)
    cov_matrix = (centered_points.T @ centered_points) / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, np.argmin(eigenvalues)]  # Smallest eigenvalue = plane normal
    
    # Ensure normal points in a consistent direction
    normal = normal / np.linalg.norm(normal)
    
    # Plane equation: normal · (p - centroid) = 0
    d = -np.dot(normal, centroid)
    
    if return_full_info:
        return centroid, normal, d
    else:
        return {
            'centroid': centroid,
            'normal': normal,
            'd': d
        }


def compute_constraint_values_along_normal(xyz, plane_info, mask):
    """
    Compute constraint values for each point projected onto the plane normal.
    This replaces the hardcoded Z-axis assumption.
    
    Args:
        xyz: Tensor of shape (N, 3) containing Gaussian positions
        plane_info: Dict from fit_plane_to_points_axis_agnostic() with 'centroid', 'normal', 'd'
        mask: Boolean tensor indicating which points are constrained
    
    Returns:
        constraint_values: Tensor of shape (N,) with distance along plane normal for each point
                          (or 0 for unconstrained points)
    """
    if isinstance(xyz, torch.Tensor):
        device = xyz.device
        dtype = xyz.dtype
        mask_t = mask.to(device=device, dtype=torch.bool)
        xyz_detached = xyz.detach()

        centroid_t = torch.as_tensor(plane_info['centroid'], dtype=dtype, device=device)
        normal_t = torch.as_tensor(plane_info['normal'], dtype=dtype, device=device)

        centered = xyz_detached - centroid_t.unsqueeze(0)
        projections = (centered * normal_t.unsqueeze(0)).sum(dim=1)

        constraint_values = torch.zeros(xyz.shape[0], dtype=dtype, device=device)
        constraint_values[mask_t] = projections[mask_t]
        return constraint_values.detach()

    centroid = plane_info['centroid']
    normal = plane_info['normal']
    centered = xyz - centroid
    projections = centered @ normal
    constraint_values = torch.zeros(xyz.shape[0], dtype=torch.float32)
    constraint_values[mask] = torch.from_numpy(projections[mask]).float()
    return constraint_values


def create_road_height_constraint(gaussians, mask, height_value=None, method='fit_plane_axis_agnostic', plane_info=None):
    """
    Create a height constraint for road Gaussians (AXIS-AGNOSTIC).
    
    Args:
        gaussians: GaussianModel instance
        mask: Boolean tensor indicating which Gaussians are part of the road
        height_value: Optional fixed height value. If None, computed from the data
        method: 'fit_plane_axis_agnostic' (recommended), 'mean', 'median', 'fit_plane'
        plane_info: Pre-computed plane info dict. If None, will be computed
    
    Returns:
        Tuple of (height_values, plane_info) or just the dict with both
    """
    xyz = gaussians.get_xyz
    
    if method in ('fit_plane_axis_agnostic', 'ransac'):
        # Compute plane and project points onto normal direction
        if plane_info is None:
            if method == 'ransac':
                plane_info = fit_plane_to_points_ransac(xyz, mask, return_full_info=False)
            else:
                plane_info = fit_plane_to_points_axis_agnostic(xyz, mask, return_full_info=False)

        if height_value is None:
            constraint_values = compute_constraint_values_along_normal(xyz, plane_info, mask).detach()
        else:
            mask_t = mask.to(device=xyz.device, dtype=torch.bool)
            constraint_values = torch.zeros(xyz.shape[0], dtype=xyz.dtype, device=xyz.device)
            constraint_values[mask_t] = float(height_value)
            constraint_values = constraint_values.detach()
        
        method_name = "RANSAC Plane Fitting" if method == 'ransac' else "Axis-Agnostic Plane Fitting"
        print(f"\n[RoadConstraint] {method_name}:")
        print(f"  Plane normal: [{plane_info['normal'][0]:.4f}, {plane_info['normal'][1]:.4f}, {plane_info['normal'][2]:.4f}]")
        print(f"  Plane centroid: [{plane_info['centroid'][0]:.4f}, {plane_info['centroid'][1]:.4f}, {plane_info['centroid'][2]:.4f}]")
        print(f"  Constrained Gaussians: {mask.sum().item()}")
        if height_value is not None:
            print(f"  Fixed normal-distance target: {float(height_value):.4f}")
        
        # Store both the constraint values and the plane info in the Gaussian model
        gaussians.set_height_constraint(mask, constraint_values, plane_info=plane_info)
        
        return constraint_values, plane_info
    
    elif method == 'mean':
        height_value = detect_flat_surface_height(xyz, mask, outlier_percentile=5, verbose=True)
    elif method == 'median':
        if mask.any():
            height_value = xyz[mask, 2].median().item()
    elif method == 'fit_plane':
        height_value = fit_plane_to_points(xyz, mask, return_plane_params=False)
    elif method == 'ransac':
        centroid, normal, d = fit_plane_to_points_ransac(xyz, mask, return_full_info=True)
        plane_info = {'centroid': centroid, 'normal': normal, 'd': d}
        height_value = compute_constraint_values_along_normal(xyz, plane_info, mask).detach()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # For legacy methods, set constraints the old way (Z-axis only)
    gaussians.set_height_constraint(mask, height_value)
    return height_value, None


def apply_height_constraint_during_training(gaussians, iteration, every_n_steps=1):
    """
    Apply height constraints during training to keep constrained Gaussians flat.
    Call this after each optimizer step.
    
    Args:
        gaussians: GaussianModel instance
        iteration: Current training iteration (for logging)
        every_n_steps: Apply constraint every N steps
    
    Returns:
        True if constraint was applied, False otherwise
    """
    if iteration % every_n_steps != 0:
        return False
    
    gaussians.apply_height_constraint_to_gradients()
    return True


def visualize_height_constraints(gaussians, output_path=None, verbose=False):
    """
    Analyze and optionally visualize the height constraints (AXIS-AGNOSTIC).
    
    Args:
        gaussians: GaussianModel instance
        output_path: Optional path to save visualization data
        verbose: Print detailed information
    
    Returns:
        Dictionary with constraint statistics
    """
    xyz = gaussians.get_xyz.detach()
    mask = gaussians.height_constrained_mask
    
    stats = {
        'num_constrained': mask.sum().item(),
        'total_points': mask.shape[0],
        'constrained_ratio': mask.sum().item() / max(mask.shape[0], 1)
    }
    
    if mask.any():
        constrained_xyz = xyz[mask]
        constraint_values = gaussians.height_constraint_values[mask]
        plane_normal = gaussians.plane_normal if hasattr(gaussians, 'plane_normal') else None
        
        # Compute statistics along constraint direction
        projection_errors = constrained_xyz - constraint_values.unsqueeze(1)  # Simple case
        if plane_normal is not None:
            # Project actual distance along the plane normal
            constraint_errors = constraint_values.abs()
        else:
            # Legacy Z-axis case
            constraint_errors = (constrained_xyz[:, 2] - constraint_values).abs()
        
        stats['constraint_z_min'] = constrained_xyz[:, 2].min().item()
        stats['constraint_z_max'] = constrained_xyz[:, 2].max().item()
        stats['constraint_z_std'] = constrained_xyz[:, 2].std().item()
        stats['constraint_value'] = constraint_values[0].item() if len(constraint_values) > 0 else None
        stats['constraint_error'] = constraint_errors.mean().item()
        
        if plane_normal is not None:
            stats['plane_normal'] = plane_normal.tolist() if isinstance(plane_normal, np.ndarray) else plane_normal
        
        if verbose:
            print("\nHeight Constraint Analysis (Axis-Agnostic):")
            print(f"  Total Gaussians: {stats['total_points']}")
            print(f"  Constrained Gaussians: {stats['num_constrained']}")
            print(f"  Ratio: {stats['constrained_ratio']:.2%}")
            print(f"  Constrained Z range: [{stats['constraint_z_min']:.4f}, {stats['constraint_z_max']:.4f}]")
            if plane_normal is not None:
                print(f"  Plane normal: {plane_normal}")
            print(f"  Constraint value: {stats['constraint_value']:.4f}")
            print(f"  Mean error from constraint: {stats['constraint_error']:.6f}")
    
    return stats


def get_road_mask_from_class(gaussians, classifier, class_id, probability_threshold=0.5):
    """
    Get a mask for road Gaussians using the classifier.
    
    Args:
        gaussians: GaussianModel instance
        classifier: Object classifier
        class_id: ID of the road class
        probability_threshold: Minimum probability for classification
    
    Returns:
        Boolean tensor indicating road Gaussians
    """
    with torch.no_grad():
        logits = classifier(gaussians._objects_dc.permute(2, 0, 1))
        probs = torch.softmax(logits, dim=0)
        mask = probs[class_id, :, :] > probability_threshold
        mask = mask.squeeze()
    
    return mask


def smooth_height_constraint(gaussians, kernel_size=3, iterations=1):
    """
    Smooth the height constraints spatially to reduce sharp transitions.
    
    Args:
        gaussians: GaussianModel instance
        kernel_size: Size of neighborhood for smoothing
        iterations: Number of smoothing iterations
    """
    if not gaussians.height_constrained_mask.any():
        return
    
    xyz = gaussians.get_xyz.detach()
    mask = gaussians.height_constrained_mask
    constrained_xyz = xyz[mask]
    
    # Build KD-tree for spatial smoothing
    kdtree = KDTree(constrained_xyz[:, :2].cpu().numpy())  # Only X, Y coordinates
    
    for _ in range(iterations):
        new_heights = gaussians.height_constraint_values[mask].clone()
        
        for i in range(len(constrained_xyz)):
            # Find neighbors
            _, neighbor_indices = kdtree.query(constrained_xyz[i, :2].unsqueeze(0).cpu().numpy(), k=kernel_size)
            neighbor_heights = gaussians.height_constraint_values[mask][torch.tensor(neighbor_indices)]
            new_heights[i] = neighbor_heights.mean()
        
        # Update constraint values
        full_indices = torch.where(mask)[0]
        gaussians.height_constraint_values[full_indices] = new_heights
