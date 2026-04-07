# Quick Reference: Axis-Agnostic Constraints

## TL;DR - Get Started in 30 Seconds

```bash
# 1. Check your point cloud axes
python visualize_pc_axes.py /path/to/input.ply --output axes_check.ply

# 2. Enable in config.json
echo '{"flattened_road": true, "road_height_loss_weight": 1.0}' > config.json

# 3. Run training (automatically uses axis-aware constraints!)
python train.py -s /path/to/dataset --config_file config.json
```

Done! The system automatically detects which axis is vertical.

---

## 1-Minute Overview

| Aspect | Before | After |
|--------|--------|-------|
| **Vertical axis detection** | Hardcoded Z-axis ❌ | Auto-detected via plane normal ✓ |
| **Axis orientation required** | Must be Z-up | Any orientation works |
| **How constraints applied** | Only Z-coordination | Projection along plane normal |
| **Code complexity** | Simple but brittle | Robust & complex |

---

## Key Commands

### Visualize Axes (Sanity Check)
```bash
python visualize_pc_axes.py input.ply
# Prints axis statistics, creates output_with_axes.ply for 3D viewer
```

### Enable in Training
```json
{
  "flattened_road": true,              // Enable constraints
  "road_constraint_every": 500,        // How often to update
  "road_min_points": 500,              // Min points to create constraint
  "road_height_loss_weight": 1.0,      // Loss weight
  "road_height_blend": 0.8             // 0=no constraint, 1=full
}
```

### Check During Training
Watch terminal for:
```
[RoadConstraint] Axis-Agnostic Plane Fitting:
  Plane normal: [0.0001, -0.0015, 0.9999]     <- This is the vertical direction!
  Constrained Gaussians: 12345
```

---

## Main Functions Reference

### Find the plane (axis-aware):
```python
from height_constraint import fit_plane_to_points_axis_agnostic
plane_info = fit_plane_to_points_axis_agnostic(xyz_points, mask)
# Returns: normal, centroid, constant
```

### Create constraints:
```python
from height_constraint import create_road_height_constraint
values, plane_info = create_road_height_constraint(
    gaussians, mask, method='fit_plane_axis_agnostic'
)
```

### Apply constraints:
```python
# Automatically uses plane_normal if available
gaussians.apply_height_constraint_to_gradients(blend=0.8)
```

---

## What Changed?

| Component | Change | Impact |
|-----------|--------|--------|
| `visualize_pc_axes.py` | ✨ NEW | Verify axis orientation before training |
| `height_constraint.py` | 📝 UPDATED | New axis-agnostic functions |
| `gaussian_model.py` | 🔧 UPDATED | Stores plane_normal |
| `train.py` | 🔧 UPDATED | Uses new methods by default |

---

## Common Questions

**Q: How do I know which axis is vertical?**
A: Run `python visualize_pc_axes.py input.ply` and look at console:
```
Axis Z has smallest range (0.77)  <- This is likely vertical
```

**Q: Does this break my existing code?**
A: No! Fully backward compatible. If plane_normal=None, uses old Z-axis constraint.

**Q: Which method should I use?**
A: Use `method='fit_plane_axis_agnostic'` (default). It just works!

**Q: How do I debug if something's wrong?**
A: Check the plane normal printed during training:
```
Plane normal: [0.0001, -0.0015, 0.9999]
```
Should point roughly "up". If not, data might need rotation.

**Q: My point cloud has X-up, not Z-up. Does it matter?**
A: No! The system auto-detects it. Just need to verify with visualization tool.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No usable mask for view X` | Check road mask files exist |
| Plane normal wrong direction | Verify with `visualize_pc_axes.py` |
| Constraint not affecting result | Check `road_height_blend > 0` |
| Error: `plane_normal is None` | Constraint not created yet, check at iteration N |

---

## Key Files

```
gaussian-grouping/
├── visualize_pc_axes.py              ← NEW: Axis check tool
├── height_constraint.py              ← UPDATED: New functions
├── scene/gaussian_model.py           ← UPDATED: Stores plane info
├── train.py                          ← UPDATED: Uses new methods
├── AXIS_AGNOSTIC_CONSTRAINTS.md      ← NEW: Full guide
├── IMPLEMENTATION_SUMMARY.md         ← NEW: Detailed changes
└── test_implementation.sh            ← NEW: Verification script
```

---

## Theory in 30 Seconds

1. **SVD plane fitting**: Finds best-fit plane to road points
2. **Extract normal**: Smallest eigenvector = perpendicular to plane
3. **Project points**: Each point's distance along normal = constraint value
4. **Apply loss**: Minimize distance error along normal during training

Works for **ANY axis orientation**! 🎉

---

## Next Steps

1. ✅ Read this file (you are here!)
2. 🔍 Run `python visualize_pc_axes.py` on your data
3. 📋 Review `AXIS_AGNOSTIC_CONSTRAINTS.md` for details
4. 🚀 Enable in `config.json` and start training
5. 📊 Monitor plane normal and loss in terminal output

---

## Still Have Questions?

- Full guide: See `AXIS_AGNOSTIC_CONSTRAINTS.md`
- Implementation details: See `IMPLEMENTATION_SUMMARY.md`
- Verify setup: Run `bash test_implementation.sh`

