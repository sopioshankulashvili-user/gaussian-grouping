# Axis-Agnostic Height Constraints for Gaussian-Grouping

This document explains the new axis-agnostic height constraint system that automatically detects and uses the true vertical direction of your point cloud, rather than hardcoding the Z-axis.

## Problem Solved

Previously, the height constraints assumed the Z-axis was always vertical. If your data had a different axis orientation, constraints would fail or produce incorrect results.

## Solution Overview

The system now:
1. **Fits a plane** to road/ground points using SVD decomposition
2. **Extracts the plane normal** - the true vertical direction
3. **Projects constraints** along the plane normal instead of just Z-axis
4. **Works with any axis orientation** - X, Y, Z, or any arbitrary orientation

## Quick Start

### Step 1: Verify Your Point Cloud Axes (Sanity Test)

Before training, visualize the coordinate axes on your point cloud:

```bash
python visualize_pc_axes.py <path_to_point_cloud.ply> --scale 0.1 --output output_with_axes.ply
```

**Output:**
- **RED line**: X-axis (left-right)
- **GREEN line**: Y-axis (front-back)  
- **BLUE line**: Z-axis (up-down)

**Example:**
```bash
python visualize_pc_axes.py /data/sopio/results/GaussianPro/2/input.ply --output check_axes.ply
```

Open `check_axes.ply` in a 3D viewer (CloudCompare, etc.) and observe:
- Which axis points upward (likely the height axis)
- Whether the points form a road-like surface

The script also prints statistics:
```
Axis    Min          Max          Range        Std
X       123.45       456.78       333.33       45.23
Y       -50.23       150.67       200.90       55.12
Z       0.12         0.89         0.77         0.15
```

The axis with the **smallest range** is likely the vertical/height axis.

### Step 2: Run Training with Axis-Agnostic Constraints

Update `config.json` if needed (defaults should work):

```json
{
  "flattened_road": true,
  "road_constraint_every": 500,
  "road_min_points": 500,
  "road_height_loss_weight": 1.0,
  "road_hole_loss_weight": 0.5,
  "road_min_opacity": 0.2,
  "road_height_blend": 0.8
}
```

Run training (the constraint method defaults to `fit_plane_axis_agnostic`):

```bash
python train.py --config_file config.json [other args...]
```

Training will print:
```
[RoadConstraint] Axis-Agnostic Plane Fitting:
  Plane normal: [0.0001, -0.0015, 0.9999]  <- The true vertical direction
  Plane centroid: [250.12, 75.46, 0.42]
  Constrained Gaussians: 12345
```

## How It Works

### Plane Fitting (SVD-based)

For each training iteration:
1. Extract visible Gaussians on the road mask
2. Center them around their centroid
3. Compute SVD decomposition
4. The smallest singular vector = plane normal (perpendicular to road surface)

### Constraint Application

Instead of constraining `z = constant`:
- Compute: `distance_along_normal = (point - centroid) · normal_vector`
- Constrain: `distance_along_normal = target_distance`
- Correction: Move point along `normal_vector` to match target

This works regardless of axis orientation!

## Key Functions

### New Functions in `height_constraint.py`

```python
# Fit plane and get the normal vector
plane_info = fit_plane_to_points_axis_agnostic(xyz, mask, return_full_info=False)
# Returns: {'centroid': array, 'normal': array, 'd': float}

# Compute per-point constraint values along the normal
constraint_values = compute_constraint_values_along_normal(xyz, plane_info, mask)

# Create constraints (NEW recommended method)
constraint_values, plane_info = create_road_height_constraint(
    gaussians,
    mask,
    method='fit_plane_axis_agnostic'  # Default method
)
```

### Updated GaussianModel

```python
# New attributes for storing plane info
gaussians.plane_normal       # Normal vector (3,) numpy array
gaussians.plane_centroid     # Centroid (3,) numpy array

# Updated method - now axis-agnostic
gaussians.apply_height_constraint_to_gradients(blend=0.8)
```

## Configuration Parameters

In `config.json` or command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `flattened_road` | false | Enable road height constraints |
| `road_constraint_every` | 500 | Apply constraint every N iterations |
| `road_min_points` | 500 | Min points to create constraint |
| `road_height_loss_weight` | 1.0 | Weight for height loss |
| `road_hole_loss_weight` | 0.5 | Weight for opacity loss (prevent holes) |
| `road_min_opacity` | 0.2 | Minimum opacity for road Gaussians |
| `road_height_blend` | 0.8 | Blending factor (0=no constraint, 1=full) |

## Backward Compatibility

- Old Z-axis only constraints still work via `method='fit_plane'` or `method='mean'`
- Existing code paths automatically use axis-agnostic if `plane_info` is provided
- Falls back to Z-axis if `plane_normal` is None

## Debugging & Visualization

### Print Constraint Statistics

```python
from height_constraint import visualize_height_constraints

stats = visualize_height_constraints(gaussians, verbose=True)
```

Output:
```
Height Constraint Analysis (Axis-Agnostic):
  Total Gaussians: 50000
  Constrained Gaussians: 12345
  Ratio: 24.69%
  Constrained Z range: [-0.1234, 0.3456]
  Plane normal: [0.0001, -0.0015, 0.9999]
  Constraint value: 0.0123
  Mean error from constraint: 0.0045
```

### Check Loss Values

Training logs will show:
```
train_loss_patches/total_loss: 0.1234
train_loss_patches/l1_loss: 0.0987
```

The `loss_road_height` component should decrease over time.

## Files Modified

1. **`visualize_pc_axes.py`** (NEW)
   - Sanity test to verify axis orientation
   - Visualizes axes as colored lines

2. **`height_constraint.py`** (UPDATED)
   - Added `fit_plane_to_points_axis_agnostic()`
   - Added `compute_constraint_values_along_normal()`
   - Updated `create_road_height_constraint()` with `method='fit_plane_axis_agnostic'`
   - Updated `visualize_height_constraints()` to show plane normal

3. **`scene/gaussian_model.py`** (UPDATED)
   - Added `plane_normal` and `plane_centroid` attributes
   - Updated `set_height_constraint()` to store plane info
   - Updated `apply_height_constraint_to_gradients()` to use plane normal
   - Updated `clear_height_constraint()` to clear plane info

4. **`train.py`** (UPDATED)
   - Updated imports
   - Changed constraint creation to use `method='fit_plane_axis_agnostic'`
   - Updated height loss computation to use plane normal

## Example Workflow

```bash
# 1. Check your point cloud axes
python visualize_pc_axes.py datasets/my_dataset/input.ply --output check.ply
# -> Open check.ply in CloudCompare, verify which axis is vertical

# 2. Update config.json
cat > config.json << EOF
{
  "flattened_road": true,
  "road_constraint_every": 500,
  "road_min_points": 300,
  "road_height_loss_weight": 1.5,
  "road_hole_loss_weight": 0.3,
  "road_min_opacity": 0.15,
  "road_height_blend": 0.9
}
EOF

# 3. Run training
python train.py \
  -s datasets/my_dataset \
  --config_file config.json \
  --iterations 30000 \
  [other params...]

# 4. Monitor output - look for:
# [RoadConstraint] Axis-Agnostic Plane Fitting:
# Plane normal: [...] <- Should point upward
```

## Troubleshooting

### "No usable mask for view 'X'"
- Check road mask files exist and format is correct
- Increase `road_min_points` threshold
- Verify camera calibration

### Plane normal looks wrong [0, 0, 1] when should be different
- Check your point cloud axes with `visualize_pc_axes.py`
- May need to rotate/transform point cloud first
- Verify road mask is correctly applied

### Height constraint not affecting optimization
- Check `road_height_blend` is not 0.0
- Verify `flattened_road: true` in config
- Check `road_height_loss_weight` is positive
- Ensure `road_constraint_every` is not too large

## References

- SVD-based plane fitting: Standard least-squares plane from point cloud
- Plane normal computation: Eigenvector with smallest eigenvalue
- Per-point projection: Dot product with plane normal vector
