# Road Constraint Algorithm (`flattened_road`)

## Goal
Constrain Gaussians to road surface level and discourage holes on road regions during training.

## Inputs
- `flattened_road` (bool): enable/disable road constraints.
- `road_mask_path` (folder): per-view binary PNG masks (`0/1`).
- Camera/view data: `viewpoint_cam` (`image_name`, `image_width`, `image_height`, `full_proj_transform`).
- Renderer output: `visibility_filter`.
- Gaussian state: `gaussians.get_xyz`, `gaussians.get_opacity`.

## Mask Types
1. **2D road mask**: `road_mask_2d[H, W]` loaded from `<road_mask_path>/<image_name>.png`.
2. **Per-Gaussian mask**: `full_mask[N]` (bool), where `N = #gaussians`.
   - `True` means: visible in this view, projects inside image, and lands on road pixel (`mask==1`).

---

## Algorithm Implemented

### A) Build per-Gaussian road mask
For each training iteration (at interval `road_constraint_every`):

1. Load mask for current image:
   - `mask_file = road_mask_path / f"{image_name}.png"`.
   - Convert to bool with `mask_np > 0`.
   - Cache by `image_name`.

2. If mask size != camera size, resize by nearest-neighbor.

3. Get visible Gaussian indices:
   - `visible_indices = where(visibility_filter)`.

4. Project visible 3D means to NDC:
   - `projected_ndc = geom_transform_points(xyz_visible, full_proj_transform)`.

5. Convert NDC to pixel coordinates:
   - `px = ((x+1)*0.5*(W-1)).long()`
   - `py = ((1-y)*0.5*(H-1)).long()`

6. Keep only in-bounds projected points.

7. Sample 2D road mask at `(py, px)`:
   - `on_road = road_mask_2d[py, px]`.

8. Build global bool mask `full_mask[N]` and mark road Gaussians as `True`.

If no usable mask / no valid points / too few points (`< road_min_points`), skip constraint that iteration.

---

### B) Height target estimation and hard constraint setup
If `full_mask` is valid:
1. Estimate robust road height:
   - `road_height = detect_flat_surface_height(get_xyz, full_mask, outlier_percentile=5)`.
2. Register target:
   - `create_road_height_constraint(gaussians, full_mask, height_value=road_height, method='mean')`.

Else:
- `gaussians.clear_height_constraint()`.

---

### C) Losses added to training objective
If constraints exist (`height_constrained_mask.any()`):

1. **Road height loss**
   - `L_height = mean((z_constrained - z_target)^2)`
   - weighted by `road_height_loss_weight`.

2. **Road hole (opacity floor) loss**
   - `L_hole = mean(relu(road_min_opacity - opacity_constrained))`
   - weighted by `road_hole_loss_weight`.

Total:
- `L_total = base_losses + w_h * L_height + w_o * L_hole`.

---

### D) Gradient-time enforcement
After optimizer step:
- `gaussians.apply_height_constraint_to_gradients(blend=road_height_blend)`.

This keeps constrained Gaussians close to the road plane while still allowing controlled updates.

---

## Config/Args Used
- `flattened_road`
- `road_constraint_every`
- `road_min_points`
- `road_height_loss_weight`
- `road_hole_loss_weight`
- `road_min_opacity`
- `road_height_blend`
- `road_mask_path` (from dataset/model params)

---

## Notes
- The road PNG mask is **2D image-space**.
- The training-time selector is **1D per-Gaussian**, not a 3D voxel mask.
- Hole avoidance is implemented via opacity regularization on road-selected Gaussians.