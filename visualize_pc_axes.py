#!/usr/bin/env python3
"""
Sanity test to visualize coordinate axes on input point clouds.
This helps verify which axis is vertical/height before setting up constraints.

Usage:
        python visualize_pc_axes.py <path_to_input.ply> [--scale 0.1] [--output output.ply]
"""

import numpy as np
import argparse
from pathlib import Path
from plyfile import PlyData, PlyElement


def load_ply(ply_path):
    """Load a PLY file and return vertices as numpy array."""
    ply = PlyData.read(ply_path)
    vertices = ply['vertex']
    points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    return points, vertices


def fit_plane_normal(points):
    """
    Compute the plane normal using covariance matrix eigendecomposition.
    This is memory-efficient even for very large point clouds (millions of points).
    
    Returns the true surface normal (perpendicular to the surface plane).
    
    Args:
        points: Numpy array of shape (N, 3)
    
    Returns:
        normal: Unit normal vector to the plane (shape: 3,)
        centroid: Centroid of the points (shape: 3,)
    """
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    
    # Compute covariance matrix (3x3, memory-efficient!)
    # cov = (1/N) * X^T @ X
    cov_matrix = (centered_points.T @ centered_points) / len(points)
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Eigenvector with SMALLEST eigenvalue = plane normal
    # (perpendicular to the surface)
    smallest_idx = np.argmin(eigenvalues)
    normal = eigenvectors[:, smallest_idx]
    normal = normal / np.linalg.norm(normal)
    
    return normal, centroid


def compute_axis_alignments(normal):
    """
    Compute how well each axis aligns with the plane normal.
    
    Args:
        normal: Unit normal vector (shape: 3,)
    
    Returns:
        Dictionary with alignment angles for each axis
    """
    standard_axes = {
        'X': np.array([1, 0, 0]),
        'Y': np.array([0, 1, 0]),
        'Z': np.array([0, 0, 1])
    }
    
    alignments = {}
    for axis_name, axis_vec in standard_axes.items():
        # Compute angle between normal and axis
        dot_product = np.abs(np.dot(normal, axis_vec))
        angle_rad = np.arccos(np.clip(dot_product, -1, 1))
        angle_deg = np.degrees(angle_rad)
        alignments[axis_name] = {
            'angle_deg': angle_deg,
            'alignment': dot_product  # 1 = perfect alignment, 0 = perpendicular
        }
    
    return alignments


def create_axis_lines(centroid, scale=0.1, num_samples=50):
    """
    Create dense coordinate axis lines from centroid.
    
    Returns:
        - X-axis line (red): centroid -> centroid + [scale, 0, 0]
        - Y-axis line (green): centroid -> centroid + [0, scale, 0]
        - Z-axis line (blue): centroid -> centroid + [0, 0, scale]
    
    Args:
        centroid: Center point
        scale: Length of each axis line
        num_samples: Number of points per axis (for dense visualization)
    """
    x_end = centroid + np.array([scale, 0, 0])
    y_end = centroid + np.array([0, scale, 0])
    z_end = centroid + np.array([0, 0, scale])
    
    # Dense sample along each axis
    x_line = np.linspace(centroid, x_end, num_samples)
    y_line = np.linspace(centroid, y_end, num_samples)
    z_line = np.linspace(centroid, z_end, num_samples)
    
    # Combine all axis points
    axis_points = np.vstack([x_line, y_line, z_line])
    
    return axis_points


def create_normal_line(centroid, normal, scale=0.1, num_samples=50):
    """
    Create a dense line visualizing the plane normal.
    
    Args:
        centroid: Center point
        normal: Unit normal vector
        scale: Length of normal line
        num_samples: Number of points for dense visualization
    
    Returns:
        Dense points along centroid -> centroid + normal*scale
    """
    normal_end = centroid + normal * scale
    normal_points = np.linspace(centroid, normal_end, num_samples)
    return normal_points


def create_output_ply(original_points, original_vertices, axis_points, normal_points, output_path):
    """
    Create a PLY file with original points and axis lines as different colors.
    
    Colors:
        - Original colors (from PLY if present, else gray)
        - Red: X-axis
        - Green: Y-axis
        - Blue: Z-axis
        - Yellow: TRUE plane normal (the actual vertical direction!)
    """
    # Preserve original colors if present, otherwise use gray
    if {'red', 'green', 'blue'}.issubset(set(original_vertices.data.dtype.names)):
        original_colors = np.stack([
            original_vertices['red'],
            original_vertices['green'],
            original_vertices['blue']
        ], axis=1).astype(np.uint8)
    else:
        original_colors = np.full((original_points.shape[0], 3), 128, dtype=np.uint8)
    
    # Color axis lines (standard X, Y, Z) - dense sampling
    num_axis_samples = axis_points.shape[0] // 3
    axis_colors = np.vstack([
        np.full((num_axis_samples, 3), [255, 0, 0], dtype=np.uint8),      # Red for X-axis
        np.full((num_axis_samples, 3), [0, 255, 0], dtype=np.uint8),      # Green for Y-axis
        np.full((num_axis_samples, 3), [0, 0, 255], dtype=np.uint8),      # Blue for Z-axis
    ])
    
    # Color normal line (yellow = the TRUE vertical direction)
    normal_colors = np.full((normal_points.shape[0], 3), [255, 255, 0], dtype=np.uint8)
    
    # Combine all points and colors
    all_points = np.vstack([original_points, axis_points, normal_points])
    all_colors = np.vstack([original_colors, axis_colors, normal_colors])

    # Preserve normals from input PLY if present
    normal_fields = None
    if {'nx', 'ny', 'nz'}.issubset(set(original_vertices.data.dtype.names)):
        normal_fields = ('nx', 'ny', 'nz')
    elif {'normal_x', 'normal_y', 'normal_z'}.issubset(set(original_vertices.data.dtype.names)):
        normal_fields = ('normal_x', 'normal_y', 'normal_z')

    num_original = original_points.shape[0]
    num_added = axis_points.shape[0] + normal_points.shape[0]
    
    # Create structured array for PLY
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if normal_fields is not None:
        vertex_dtype.extend([(normal_fields[0], 'f4'), (normal_fields[1], 'f4'), (normal_fields[2], 'f4')])

    vertex_array = np.zeros(all_points.shape[0], dtype=vertex_dtype)
    
    vertex_array['x'] = all_points[:, 0]
    vertex_array['y'] = all_points[:, 1]
    vertex_array['z'] = all_points[:, 2]
    vertex_array['red'] = all_colors[:, 0]
    vertex_array['green'] = all_colors[:, 1]
    vertex_array['blue'] = all_colors[:, 2]

    if normal_fields is not None:
        original_normals = np.stack([
            original_vertices[normal_fields[0]],
            original_vertices[normal_fields[1]],
            original_vertices[normal_fields[2]],
        ], axis=1).astype(np.float32)
        added_normals = np.zeros((num_added, 3), dtype=np.float32)
        all_normals = np.vstack([original_normals, added_normals])

        vertex_array[normal_fields[0]] = all_normals[:, 0]
        vertex_array[normal_fields[1]] = all_normals[:, 1]
        vertex_array[normal_fields[2]] = all_normals[:, 2]
    
    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el]).write(output_path)
    print(f"✓ Output saved to: {output_path}")


def analyze_axis_distributions(points):
    """Analyze point distribution along each axis AND compute true plane normal."""
    centroid = points.mean(axis=0)
    
    print(f"\n{'='*70}")
    print("POINT CLOUD ANALYSIS - AXIS DISTRIBUTION")
    print(f"{'='*70}")
    print(f"Total points: {points.shape[0]:,}")
    print(f"Centroid: X={centroid[0]:.4f}, Y={centroid[1]:.4f}, Z={centroid[2]:.4f}")
    
    print(f"\n{'Axis':<6} {'Min':<12} {'Max':<12} {'Range':<12} {'Std':<12}")
    print("-" * 54)
    
    ranges = []
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        coords = points[:, i]
        min_val = coords.min()
        max_val = coords.max()
        range_val = max_val - min_val
        std_val = coords.std()
        ranges.append(range_val)
        print(f"{axis_name:<6} {min_val:<12.4f} {max_val:<12.4f} {range_val:<12.4f} {std_val:<12.4f}")
    
    # IMPORTANT: Compute the TRUE plane normal from the data
    print(f"\n{'='*70}")
    print("SURFACE NORMAL COMPUTATION (via Eigendecomposition)")
    print(f"{'='*70}")
    
    # For very large point clouds, optionally sample
    if points.shape[0] > 1_000_000:
        print(f"⚠️  Large point cloud detected ({points.shape[0]:,} points)")
        print(f"   Computing normal on full dataset (memory-efficient method)...")
    
    print(f"   Computing covariance matrix...", end='', flush=True)
    normal, _ = fit_plane_normal(points)
    print(f" ✓")
    
    print(f"✓ TRUE Plane Normal: [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(f"  (This is the ACTUAL vertical direction for your surface)")
    
    # Compute alignment of each standard axis with the true normal
    alignments = compute_axis_alignments(normal)
    print(f"\n{'Axis-to-Normal Alignment:':<30}")
    print("-" * 60)
    print(f"{'Axis':<6} {'Angle (deg)':<15} {'Alignment':<15} {'Assessment':<15}")
    print("-" * 60)
    
    for axis_name in ['X', 'Y', 'Z']:
        angle = alignments[axis_name]['angle_deg']
        align = alignments[axis_name]['alignment']
        
        # Classify alignment
        if align > 0.9:
            assessment = "PERFECT ✓✓"
        elif align > 0.7:
            assessment = "GOOD ✓"
        elif align > 0.5:
            assessment = "MODERATE ~"
        else:
            assessment = "POOR ✗"
        
        print(f"{axis_name:<6} {angle:<15.2f} {align:<15.4f} {assessment:<15}")
    
    # Find which axis is closest to normal
    best_axis = max(alignments.keys(), key=lambda x: alignments[x]['alignment'])
    best_angle = alignments[best_axis]['angle_deg']
    best_align = alignments[best_axis]['alignment']
    
    print(f"\n{'='*70}")
    if best_align > 0.9:
        print(f"⚠️  AXIS {best_axis} aligns well with surface normal (angle: {best_angle:.2f}°)")
        print(f"✓ You can likely use axis {best_axis} for height constraints")
    elif best_align > 0.7:
        print(f"⚠️  AXIS {best_axis} somewhat aligns with surface normal (angle: {best_angle:.2f}°)")
        print(f"~ Consider checking orientation, axis-agnostic constraints recommended")
    else:
        print(f"⚠️  NO AXIS ALIGNS WELL with surface normal!")
        print(f"❌ Standard axes are NOT aligned with your surface")
        print(f"✓ AXIS-AGNOSTIC constraints are REQUIRED")
        print(f"   Best axis {best_axis} has only {best_align*100:.1f}% alignment (angle: {best_angle:.2f}°)")
    print(f"{'='*70}\n")
    
    return centroid, normal


def main():
    parser = argparse.ArgumentParser(description="Visualize coordinate axes and surface normal on point clouds")
    parser.add_argument('ply_file', help='Path to input PLY file')
    parser.add_argument('--scale', type=float, default=0.1, 
                       help='Scale of axis lines relative to point cloud (default: 0.1)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PLY file (default: input_filename_with_axes.ply)')
    parser.add_argument('--no-save', action='store_true',
                       help='Only analyze without saving output PLY')
    
    args = parser.parse_args()
    
    ply_path = Path(args.ply_file)
    if not ply_path.exists():
        print(f"✗ Error: File not found: {ply_path}")
        return 1
    
    print(f"Loading point cloud from: {ply_path}")
    points, vertices = load_ply(ply_path)
    
    # Analyze distributions AND compute plane normal
    centroid, plane_normal = analyze_axis_distributions(points)
    
    # Compute scale based on point cloud extent if not provided
    if args.scale is None:
        max_extent = np.max([points[:, i].max() - points[:, i].min() for i in range(3)])
        scale = max_extent * 0.1
    else:
        scale = args.scale

    scale = 3
    
    print(f"Using axis scale: {scale:.4f}")
    
    if not args.no_save:
        # Create axis lines (standard X, Y, Z)
        axis_points = create_axis_lines(centroid, scale=scale)
        
        # Create normal line (the ACTUAL vertical direction)
        normal_points = create_normal_line(centroid, plane_normal, scale=scale)
        
        # Determine output filename
        if args.output is None:
            output_path = ply_path.parent / f"{ply_path.stem}_with_axes.ply"
        else:
            output_path = Path(args.output)
        
        # Create and save output
        create_output_ply(points, vertices, axis_points, normal_points, str(output_path))
        print(f"\nVisualization Guide:")
        print(f"1. Open '{output_path}' in a 3D viewer (e.g., CloudCompare)")
        print(f"2. See axis colors:")
        print(f"   - RED:    X-axis (standard coordinate)")
        print(f"   - GREEN:  Y-axis (standard coordinate)")
        print(f"   - BLUE:   Z-axis (standard coordinate)")
        print(f"   - YELLOW: Surface Normal (TRUE vertical direction) ⭐")
        print(f"3. If YELLOW line doesn't align with any RED/GREEN/BLUE:")
        print(f"   → Your point cloud is NOT axis-aligned!")
        print(f"   → Axis-agnostic constraints will be used automatically ✓")
    
    return 0


if __name__ == '__main__':
    exit(main())
