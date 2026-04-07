# Axis-Agnostic Height Constraints - Implementation Summary

## Overview

Successfully implemented axis-agnostic height constraints for the Gaussian-Grouping codebase. The system now automatically detects the true vertical direction (plane normal) instead of hardcoding the Z-axis.

---

## Files Created

### 1. `visualize_pc_axes.py` (NEW)
**Purpose:** Sanity test to verify coordinate axes on input point clouds

**Features:**
- Loads any .ply file and analyzes point distribution along each axis
- Creates colored axis visualizations (RED=X, GREEN=Y, BLUE=Z)
- Outputs statistics showing which axis is likely vertical
- Saves visualization to new .ply file with axis lines

**Usage:**
```bash
python visualize_pc_axes.py <input.ply> [--scale 0.1] [--output output.ply]
```

**Example:**
```bash
python visualize_pc_axes.py /data/sopio/results/GaussianPro/2/input.ply \
  --output check_axes.ply
```

**Output in terminal:**
```
Axis    Min          Max          Range        Std
X       123.45       456.78       333.33       45.23
Y       -50.23       150.67       200.90       55.12
Z       0.12         0.89         0.77         0.15

INFERENCE: Axis Z has smallest range (0.77)
This suggests 'Z' might be the HEIGHT axis
```

---

## Files Modified

### 1. `height_constraint.py` (UPDATED)

**New Functions Added:**

#### `fit_plane_to_points_axis_agnostic(xyz, mask, return_full_info=False)`
- Fits plane to road Gaussians using SVD decomposition
- Returns plane normal vector (true vertical direction)
- Works with ANY axis orientation, not just Z

```python
plane_info = fit_plane_to_points_axis_agnostic(xyz, mask)
# Returns: {
#   'centroid': array([x, y, z]),
#   'normal': array([nx, ny, nz]),
#   'd': float
# }
```

#### `compute_constraint_values_along_normal(xyz, plane_info, mask)`
- Computes per-point constraint values projected onto plane normal
- Replaces hardcoded Z-axis constraint

```python
constraint_values = compute_constraint_values_along_normal(xyz, plane_info, mask)
# Returns: tensor of shape (N,) with distance along normal for each point
```

**Updated Functions:**

#### `create_road_height_constraint(gaussians, mask, method='fit_plane_axis_agnostic', plane_info=None)`
- New default method: `'fit_plane_axis_agnostic'` (axis-aware)
- Legacy methods still available: `'mean'`, `'median'`, `'fit_plane'`
- Returns tuple: `(constraint_values, plane_info)`

```python
constraint_values, plane_info = create_road_height_constraint(
    gaussians,
    road_mask,
    method='fit_plane_axis_agnostic'
)
```

#### `visualize_height_constraints(gaussians, ...)`
- Now prints plane normal information
- Works with both axis-agnostic and legacy constraints

**Old Functions (still available for backward compatibility):**
- `detect_flat_surface_height()` - Marked DEPRECATED
- `fit_plane_to_points()` - Marked DEPRECATED

---

### 2. `scene/gaussian_model.py` (UPDATED)

**New Attributes:**
```python
self.plane_normal = None      # Plane normal vector (3,) numpy array
self.plane_centroid = None    # Plane centroid (3,) numpy array
```

**Updated Methods:**

#### `set_height_constraint(mask, height_value, plane_info=None)`
- Now accepts optional `plane_info` parameter
- Stores plane normal and centroid when provided
- Maintains backward compatibility (plane_info can be None)

```python
# New way (axis-agnostic):
gaussians.set_height_constraint(mask, constraint_values, plane_info=plane_info)

# Old way (still works):
gaussians.set_height_constraint(mask, height_value)
```

#### `clear_height_constraint()`
- Now also clears `plane_normal` and `plane_centroid`

```python
gaussians.clear_height_constraint()
# Clears: height_constrained_mask, height_constraint_values, plane_normal, plane_centroid
```

#### `apply_height_constraint_to_gradients(blend=1.0)` - MAJOR UPDATE
- **NEW:** Uses plane normal if available (axis-agnostic)
- **FALLBACK:** Uses Z-axis if plane_normal is None (backward compatible)

```python
# Axis-agnostic application:
if self.plane_normal is not None:
    # Project gradient along plane normal
    plane_normal = torch.from_numpy(self.plane_normal).float()
    current_distance = (point - centroid) · normal
    correction = (current_distance - target) * normal
    point -= blend * correction
else:
    # Legacy Z-axis constraint
    z_new = (1 - blend) * z_current + blend * z_target
```

**Logic Flow:**
1. If `plane_normal` exists → Use axis-agnostic constraint
2. Otherwise → Fall back to Z-axis constraint

---

### 3. `train.py` (UPDATED)

**Import Changes:**
```python
from height_constraint import (
    detect_flat_surface_height,
    create_road_height_constraint,
    fit_plane_to_points,
    fit_plane_to_points_axis_agnostic,      # NEW
    compute_constraint_values_along_normal   # NEW
)
```

**Training Loop Changes:**

#### Road Constraint Creation (line ~167)
```python
# OLD (Z-axis only):
road_height = fit_plane_to_points(gaussians.get_xyz.detach(), road_gaussian_mask)
create_road_height_constraint(gaussians, road_gaussian_mask, height_value=road_height)

# NEW (axis-agnostic):
constraint_values, plane_info = create_road_height_constraint(
    gaussians,
    road_gaussian_mask,
    method='fit_plane_axis_agnostic'
)
```

#### Height Loss Computation (line ~197-206)
```python
# NEW: Checks for plane_normal before computing loss
if gaussians.plane_normal is not None:
    # AXIS-AGNOSTIC: Compute distance along plane normal
    plane_normal = torch.from_numpy(gaussians.plane_normal).float()
    plane_centroid = torch.from_numpy(gaussians.plane_centroid).float()
    current_vec = constrained_xyz - plane_centroid
    current_distance = (current_vec * plane_normal).sum(dim=1)
    loss_road_height = torch.mean((current_distance - constrained_targets) ** 2)
else:
    # LEGACY: Z-axis only
    loss_road_height = torch.mean((constrained_xyz[:, 2] - constrained_targets) ** 2)
```

**Print Output (NEW):**
```
[RoadConstraint] Axis-Agnostic Plane Fitting:
  Plane normal: [0.0001, -0.0015, 0.9999]
  Plane centroid: [250.12, 75.46, 0.42]
  Constrained Gaussians: 12345
```

---

## Documentation Added

### `AXIS_AGNOSTIC_CONSTRAINTS.md`
Comprehensive guide including:
- Problem solved
- How to use the visualization tool
- Step-by-step workflow
- Configuration parameters
- Troubleshooting tips
- Theory behind plane fitting

### `test_implementation.sh`
Script to verify all changes are properly implemented:
- Checks for new files and functions
- Verifies all imports are correct
- Confirms all methods are updated

---

## Backward Compatibility

✓ **Fully backward compatible**

1. **Legacy code still works:**
   - Old `fit_plane_to_points()` preserved
   - Old `detect_flat_surface_height()` preserved
   - Can still pass `method='fit_plane'` or `method='mean'`

2. **Existing constraints work:**
   - If `plane_normal = None`, falls back to Z-axis
   - Existing trained models won't break

3. **Gradual migration:**
   - Can mix old and new constraints in same codebase
   - Recommended: Use axis-agnostic (`method='fit_plane_axis_agnostic'`)

---

## How It Works: Technical Details

### Plane Fitting (SVD-based)

1. Extract road Gaussians from point cloud
2. Compute centroid: $c = \frac{1}{n} \sum_{i=1}^{n} p_i$
3. Center points: $p_i' = p_i - c$
4. SVD decomposition: $P' = U \Sigma V^T$
5. Plane normal = $V[-1]$ (eigenvector with smallest eigenvalue)

### Constraint Application

**Distance projection:**
$$d_i = (p_i - c) \cdot \mathbf{n}$$

**Constraint loss:**
$$L = \frac{1}{m} \sum_{i \in \text{constrained}} (d_i - d_{\text{target}})^2$$

**Gradient correction:**
$$\Delta p_i = -\alpha \cdot (d_i - d_{\text{target}}) \cdot \mathbf{n}$$

This works for ANY axis orientation!

---

## Usage Examples

### Example 1: Basic Workflow

```bash
# Step 1: Visualize axes
python visualize_pc_axes.py /data/my_dataset/input.ply --output check.ply

# Open check.ply in CloudCompare to verify which axis is vertical

# Step 2: Enable in config
cat > config.json << EOF
{
  "flattened_road": true,
  "road_constraint_every": 500,
  "road_height_loss_weight": 1.0
}
EOF

# Step 3: Train (automatically uses axis-agnostic constraints)
python train.py -s /data/my_dataset --config_file config.json --iterations 30000
```

### Example 2: Debug Constraints

```python
# In your training code or interactive session:
from height_constraint import visualize_height_constraints

# During training:
stats = visualize_height_constraints(gaussians, verbose=True)
print(f"Plane normal: {stats['plane_normal']}")
print(f"Constraint error: {stats['constraint_error']:.6f}")
```

### Example 3: Legacy Mode (if needed)

```python
# Use old Z-axis constraint method:
constraint_values, _ = create_road_height_constraint(
    gaussians,
    road_mask,
    method='fit_plane'  # or 'mean'
)
```

---

## Testing Checklist

Before using on a real dataset:

- [ ] Run `bash test_implementation.sh` - checks all components
- [ ] Test visualization: `python visualize_pc_axes.py <input.ply> --output test.ply`
- [ ] Verify axis visualization looks correct in 3D viewer
- [ ] Check plane_normal prints as expected during training
- [ ] Verify loss_road_height decreases over iterations
- [ ] Compare results with/without constraints

---

## Files and Locations

| File | Type | Change |
|------|------|--------|
| `visualize_pc_axes.py` | NEW | Axis visualization tool |
| `height_constraint.py` | UPDATED | Axis-agnostic plane fitting |
| `scene/gaussian_model.py` | UPDATED | Stores plane normal |
| `train.py` | UPDATED | Uses axis-agnostic methods |
| `AXIS_AGNOSTIC_CONSTRAINTS.md` | NEW | Comprehensive guide |
| `test_implementation.sh` | NEW | Verification script |

---

## Key Benefits

1. **Works with any axis orientation** - No manual axis swapping needed
2. **Automatic detection** - Finds true vertical direction from data
3. **Backward compatible** - Existing code unaffected
4. **Better accuracy** - Uses SVD plane fitting instead of percentile-based heuristics
5. **Flexible** - Can use per-point constraint values if needed

---

## Questions or Issues?

See `AXIS_AGNOSTIC_CONSTRAINTS.md` for:
- Troubleshooting
- Parameter tuning
- Advanced examples
- Theory details

