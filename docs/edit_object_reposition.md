# Gaussian Grouping: Repositioning Targeted Gaussians

This document describes the geometric editing pipeline implemented in `edit_object_reposition.py`, with emphasis on how translation and rotation are applied to a targeted subset of Gaussians. It also explains the two-pass rendering strategy used during visualization and the role of the additional opacity output returned by the renderer.

## 1. Overview of the repositioning pipeline

The repositioning script operates on a subset of Gaussians that has been selected from the scene using the learned object identity representation. In practice, the script first predicts a 3D mask for the selected object ids, and then either modifies those Gaussians in place or duplicates them, depending on whether the user wishes to preserve the original object in its initial position.

The editing process is therefore not a global rigid transformation of the entire scene. Instead, it is a local operation applied only to the Gaussians whose boolean mask is true. Let $\mathcal{G} = \{g_i\}_{i=1}^N$ denote all Gaussians in the scene and let $M \subset \{1,\dots,N\}$ denote the selected index set. The repositioning operation is applied only to $g_i$ for $i \in M$, while the remaining Gaussians are either left unchanged or, in the two-pass rendering stage, temporarily treated as background.

The script implements two related modes:

1. **In-place repositioning**, where the selected Gaussians are directly translated and/or rotated.
2. **Duplicate-and-reposition**, where the original Gaussians are retained and a transformed copy is created so that both the source object and the moved object can coexist.

The code paths for these modes are governed by the `keep_original_gaussians` option.

## 2. Translation of targeted Gaussians

### 2.1 Direct translation

The direct translation routine is implemented in `apply_translation_to_selected_gaussians`. Conceptually, this is the simplest possible rigid motion: every selected Gaussian center is shifted by the same 3D offset vector.

If $\mathbf{x}_i \in \mathbb{R}^3$ is the center of the $i$-th Gaussian and $\mathbf{t} \in \mathbb{R}^3$ is the requested translation, the updated center is

$$
\mathbf{x}'_i = \mathbf{x}_i + \mathbf{t}, \qquad i \in M.
$$

In the implementation, this update is performed only inside a `torch.no_grad()` block, because the edit is intended as a parameter mutation rather than a differentiable transformation.

A small but important detail is that the translation is applied to the internal parameter tensor `gaussians._xyz`, not to the read-only property returned by `gaussians.get_xyz`. The code also explicitly checks whether the translation vector is zero; if so, it returns immediately and avoids unnecessary tensor writes.

### 2.2 Duplicate-and-translate mode

When the user chooses to preserve the original object, the script uses `duplicate_and_translate_selected_gaussians`. In this case, the original subset remains unchanged, and a second set of Gaussians is appended to the model. The new copy receives the same appearance and shape parameters as the source Gaussians, but its spatial centers are shifted by the translation vector.

Formally, for each selected Gaussian $g_i$, the method creates a new Gaussian $\tilde{g}_i$ such that

$$
\tilde{\mathbf{x}}_i = \mathbf{x}_i + \mathbf{t},
$$

while the feature, opacity, scaling, rotation, and object-identity tensors are cloned from the source instance. This preserves the visual identity of the object in the new location.

The method then concatenates the original and duplicated tensors along the point dimension. By "point dimension," we refer to the first dimension (dimension 0) of the tensors, where each Gaussian point is indexed. For example, if the original position tensor has shape $[N, 3]$ where $N$ is the number of Gaussians, after concatenating with $|M|$ duplicates, the shape becomes $[N + |M|, 3]$. This concatenation is done for all attribute tensors: `_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`, `_rotation`, and `_objects_dc` (see the implementation in [edit_object_reposition.py](../edit_object_reposition.py#L243-L251), where each concatenation uses `torch.cat(..., dim=0)`). 

From this point on, the scene contains $N + |M|$ Gaussians, and the optimizer is configured with a correspondingly expanded boolean mask. This is why the function returns both `train_mask_expanded` and a mask that isolates only the newly created translated copies.

## 3. Rotation of targeted Gaussians

### 3.1 Euler-angle parameterization

The rotation logic is slightly more involved than the translation logic because the script must update both the Gaussian centers and the Gaussian orientation parameters. The user supplies rotation values in degrees along the three coordinate axes, conventionally written as $[r_x, r_y, r_z]$.

These Euler angles are converted to a quaternion in `_euler_degrees_to_quaternion`. The conversion is performed by first transforming degrees into radians, then forming three axis-aligned quaternions, and finally combining them in the order $X \rightarrow Y \rightarrow Z$. If the requested rotation is zero in all axes, the function returns the identity quaternion $(1, 0, 0, 0)$.

The use of quaternions is important because the Gaussian model stores orientation in quaternion form, and quaternion multiplication provides a stable way to compose rotations without introducing gimbal lock.

### 3.2 Rotation about a local center

The core rigid-body rotation is implemented in `_apply_rotation_to_gaussian_subset`. The key idea is to rotate the selected Gaussians not around a global origin, but around a local center point. This preserves the spatial relationships between the Gaussians and ensures they rotate as a cohesive object.

**Computing the rotation center:** The script first extracts the subset of selected Gaussian centers and computes a rotation center. By default, this is the arithmetic mean (centroid) of all selected Gaussian positions:

$$
\mathbf{c} = \frac{1}{|M|} \sum_{i \in M} \mathbf{x}_i.
$$

If a custom center is provided (such as a specific pivot point), that value is used instead.

**Applying the rotation:** To rotate each Gaussian around this center, the transformation follows a three-step process:

1. **Translate to origin:** Subtract the center from each Gaussian position: $(\mathbf{x}_i - \mathbf{c})$. This shifts all selected Gaussians so the rotation center is at the origin.

2. **Rotate:** Apply the rotation matrix $R \in \mathbb{R}^{3 \times 3}$ (derived from the quaternion $q_\Delta$) to the shifted positions: $R(\mathbf{x}_i - \mathbf{c})$.

3. **Translate back:** Add the center back: $R(\mathbf{x}_i - \mathbf{c}) + \mathbf{c}$.

Combining these steps, the new position of each selected Gaussian is:

$$
\mathbf{x}'_i = R(\mathbf{x}_i - \mathbf{c}) + \mathbf{c}, \qquad i \in M.
$$

**Why this preserves rigidity:** This transformation is a rigid motion because it maintains all pairwise distances between the Gaussians. The relative positions of all selected Gaussians remain unchanged—they all rotate together as if they were bolted to a rigid body that pivots around $\mathbf{c}$. This preserves the spatial arrangement and prevents distortion during editing.

### 3.3 Updating the Gaussian orientation

A Gaussian in this representation is not only positioned in space; it also has an orientation that affects its anisotropic covariance. After rotating the centers, the script also updates the quaternion stored in `gaussians._rotation`.

Let $q_i$ denote the original quaternion of the $i$-th selected Gaussian and let $q_\Delta$ denote the quaternion representing the user-requested edit. The new orientation is computed as quaternion composition:

$$
q'_i = q_\Delta \otimes q_i.
$$

The result is normalized before being written back to the model to preserve unit-length quaternion constraints.

This step is essential. If only the centers were rotated while the orientations stayed unchanged, the Gaussians would move rigidly but their ellipsoidal support regions would remain misaligned with the new geometry. Updating both position and orientation keeps the local covariance frame consistent with the edited object.

### 3.4 Duplicate-and-rotate mode

When `keep_original_gaussians` is enabled and a rotation is requested, the script can create a rotated duplicate instead of moving the original points. The function `duplicate_and_rotate_selected_gaussians` clones all geometry and appearance tensors for the selected subset, rotates the duplicated centers about the subset centroid, and composes the duplicated quaternions with the delta rotation.

The result is a scene containing both the source object and its rotated copy. As with translation, the method appends the transformed Gaussians to the end of the tensors and constructs a larger optimization mask so that the finetuning stage can act on the edited structure without destroying the original one.

## 4. Relationship between translation and rotation in the repositioning script

The script supports translation, rotation, or both. When both are requested, the order of application matters conceptually. In the in-place path, translation and rotation are applied directly to the same selected subset. In the duplicate-and-preserve path, the translation may be used to create the new copy, and the rotation may then be applied to the preserved or duplicated subset depending on the desired editing behavior.

From a geometric perspective, the operation is a rigid transform of the form

$$
\mathbf{x}' = R(\mathbf{x} - \mathbf{c}) + \mathbf{c} + \mathbf{t},
$$

where $\mathbf{c}$ is the rotation center and $\mathbf{t}$ is the translation vector. The implementation decomposes this into smaller routines so that each effect can be controlled independently and the same code can support both direct edits and copy-based edits.

The important modeling assumption is that the selected Gaussians are treated as a coherent object. Their positions, orientations, and appearance are duplicated or mutated together so that the object retains a consistent appearance after repositioning.

## 5. Two-pass rendering

### 5.1 Motivation

The repositioning script uses a two-pass rendering strategy when a foreground mask is available. The purpose of this strategy is to render the moved object and the remaining scene separately and then composite them in image space. This is useful because it allows the script to isolate the edited object visually while still preserving correct occlusion relationships with the background.

The mask used for this process is stored on the Gaussian model as `reposition_foreground_mask`. If the mask exists and has the same length as the current number of Gaussians, `render_set` enables the two-pass path. Otherwise, it falls back to the standard single-pass renderer.

### 5.2 Temporary masking through opacity suppression

The helper function `_render_with_active_mask` does not physically remove Gaussians from the model. Instead, it temporarily suppresses the opacity of all inactive Gaussians by overwriting their internal opacity parameters with a very small value. More precisely, the script stores the original opacity tensor, identifies inactive indices, and replaces them with the inverse-sigmoid of a tiny target value such as $10^{-6}$.

Because the Gaussian model stores opacity in an unconstrained parameter space and exposes the rendered opacity through a sigmoid activation, writing the inverse-sigmoid of a tiny number makes those Gaussians effectively transparent during the call to `render`. After rendering, the original opacity values are restored.

This design has two benefits. First, it avoids allocating a separate filtered model. Second, it preserves tensor shapes and optimizer state, which makes the masking operation lightweight and reversible.

### 5.3 Background and foreground passes

In the two-pass path, the scene is rendered twice:

1. **Background pass**: all foreground Gaussians are suppressed and the complementary set is rendered.
2. **Foreground pass**: all background Gaussians are suppressed and only the edited object is rendered.

Let $I_{\mathrm{bg}}$ denote the background image and $I_{\mathrm{fg}}$ denote the foreground image. Let $\alpha_{\mathrm{fg}}$ denote the foreground alpha map. The final composite is computed using the standard over operator:

$$
I = I_{\mathrm{fg}} + (1 - \alpha_{\mathrm{fg}}) I_{\mathrm{bg}}.
$$

This equation expresses the fact that the foreground contributes its own color, while the background contributes only where the foreground remains transparent.

The script applies the same idea to the object-id output. The object map is selected using the foreground alpha mask, with a hard threshold at $0.5$:

$$
M_{\mathrm{fg}} = [\alpha_{\mathrm{fg}} > 0.5].
$$

Pixels dominated by the foreground use the foreground object map, while the rest inherit the background object map.

### 5.4 Why this is useful for repositioning

Two-pass rendering is especially important in repositioning tasks because the edited object may overlap with the original background geometry from the new viewpoint. If the object were rendered in a single pass with all Gaussians active, the source and destination regions could interfere visually and complicate evaluation. Splitting the rendering into foreground and background passes provides a more controllable composition stage and makes the visual result easier to interpret.

## 6. The additional rasterizer output: opacity / accumulated alpha

### 6.1 Difference from the original 3DGS renderer

In the renderer defined in `gaussian_renderer/__init__.py`, the rasterizer returns four outputs rather than the three outputs commonly used in the original 3D Gaussian Splatting rendering pipeline. In addition to the rendered RGB image, screen-space radii, and rendered object features, the rasterizer also returns `rendered_alpha`, which is exposed by the wrapper under the key `opacity`.

The relevant return dictionary includes

$$
\texttt{"opacity"} : \texttt{rendered\_alpha}.
$$

This value is not the per-Gaussian opacity parameter itself. The per-Gaussian opacity parameter is stored inside the model and passed into the rasterizer as `opacities = pc.get_opacity`. By contrast, `rendered_alpha` is the **per-pixel accumulated alpha map** produced by rasterization after all visible Gaussians have been blended along each camera ray.

Thus, the model-level opacity controls how strongly each Gaussian contributes to the image, while the returned opacity map summarizes the final image-space coverage after compositing.

### 6.2 Meaning of accumulated alpha

For a given pixel, the rasterizer effectively accumulates contributions from multiple splatted Gaussians along the viewing ray. If $\alpha_i$ is the opacity contribution of the $i$-th contributing Gaussian after projection and blending, and $T_i$ is the transmittance of previously composited Gaussians, the final pixel alpha can be written conceptually as

$$
\alpha_{\mathrm{pixel}} = \sum_i T_i \alpha_i.
$$

This quantity measures how much of the pixel is covered by rendered matter. It is therefore suitable for compositing, visibility reasoning, and foreground-background separation.

### 6.3 Use of opacity in compositing

In `_composite_two_passes`, the foreground alpha map is used directly as a blending weight. The final image is assembled by treating the foreground as the front layer and the background as the back layer. Where the foreground alpha is close to one, the final pixel is taken almost entirely from the foreground. Where the foreground alpha is close to zero, the background dominates.

This is exactly why the renderer exposes the accumulated alpha map. Without it, the script would have no principled way to compute the per-pixel blending factor between the two rendered passes.

The opacity output is also useful for the object map composition. Since the foreground object is only trusted where the foreground actually contributes to the pixel, the script uses the alpha threshold to decide whether to take the foreground or background object prediction.

### 6.4 Practical interpretation in the repositioning pipeline

In summary, the `opacity` field returned by `render` should be interpreted as the rendered visibility map of the current pass. It tells the repositioning pipeline which pixels are truly occupied by the foreground pass and therefore should override the background. This differs from the Gaussian model's internal opacity parameters, which are per-point parameters optimized during training or finetuning.

The introduction of this field makes the renderer more informative than the vanilla 3DGS wrapper, because the repositioning script needs both color and coverage information to perform image-space compositing in a reliable way.

## 7. Summary

The repositioning procedure in `edit_object_reposition.py` is a local rigid-editing pipeline for selected Gaussians. Translation adds a constant vector to the selected centers, while rotation applies a rigid transform around a chosen center and updates both positions and quaternion orientations. When preservation of the original object is desired, the script duplicates the selected Gaussians before transforming them.

For visualization, the script can render the edited object and the remaining scene in two passes. The two images are combined with alpha compositing using the foreground accumulated opacity returned by the rasterizer. This additional `opacity` output is a pixel-level alpha map, distinct from the internal per-Gaussian opacity parameters, and it plays a central role in the final compositing step.
