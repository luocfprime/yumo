import os
import platform

import numpy as np
import pytest
from PIL import Image, ImageChops

from yumo.geometry_utils import bake_to_texture, map_to_uv, sample_surface, unwrap_uv
from yumo.utils import load_mesh


def test_e2e_texture_bake(test_data, tmp_path, refresh_golden):
    """
    End-to-end test of texture baking.

    Golden standard files are system-specific. Expected filenames:
      texture_bake_<System>_gt.png
      texture_bake_<System>_gt.npy
    """

    system = platform.system()  # e.g. "Darwin", "Linux", "Windows"

    # -- 1. Load mesh from STL file --
    mesh_path = os.path.join(test_data, "sample.STL")
    vertices, faces = load_mesh(mesh_path)

    # -- 2. Unwrap to UVs --
    (
        param_corner,
        H,
        W,
        vmapping,
        faces_unwrapped,
        uvs,
        vertices_unwrapped,
    ) = unwrap_uv(vertices, faces, brute_force=True)

    # -- 3. Sample surface --
    rng = np.random.default_rng(42)
    points, bary, indices = sample_surface(vertices_unwrapped, faces_unwrapped, points_per_area=500.0, rng=rng)

    # -- 4. Map samples to UV space --
    sample_uvs = map_to_uv(uvs, faces_unwrapped, bary, indices)

    # -- 5. Assign scalar values (all ones for coverage test) --
    values = np.ones(len(points), dtype=float)

    # -- 6. Bake to texture --
    tex = bake_to_texture(sample_uvs, values, H, W)

    # Always save to tmp_path for debugging
    out_npy = tmp_path / "texture_bake.npy"
    out_png = tmp_path / "texture_bake.png"
    np.save(out_npy, tex)

    tex_norm = (tex / tex.max() * 255).astype(np.uint8)
    img = Image.fromarray(tex_norm)
    img.save(out_png)

    # Compute golden file names
    gt_png = os.path.join(test_data, f"texture_bake_{system}_gt.png")
    gt_npy = os.path.join(test_data, f"texture_bake_{system}_gt.npy")

    if refresh_golden:
        # Overwrite system-specific goldens
        img.save(gt_png)
        np.save(gt_npy, tex)
        pytest.skip("Golden standards refreshed, skipping comparison.")

    # If no goldens for this system, skip
    if not (os.path.exists(gt_png) and os.path.exists(gt_npy)):
        pytest.skip(f"No golden standards available for system {system}")

    # --- Compare PNG golden ---
    gt_img = Image.open(gt_png).convert("L")
    assert img.size == gt_img.size, f"Texture size mismatch: got {img.size}, expected {gt_img.size}"

    diff = ImageChops.difference(img, gt_img)
    if diff.getbbox() is not None:
        diff_file = tmp_path / "texture_bake_diff.png"
        diff.save(diff_file)
        raise AssertionError(f"PNG mismatch.\nGenerated: {out_png}\nDiff: {diff_file}\nExpected: {gt_png}")

    # --- Compare NPY golden ---
    gt_tex = np.load(gt_npy)
    assert tex.shape == gt_tex.shape, f"Array shape mismatch: got {tex.shape}, expected {gt_tex.shape}"

    if not np.allclose(tex, gt_tex, atol=1e-6):
        diff_arr = tex - gt_tex
        diff_file = tmp_path / "texture_bake_diff.npy"
        np.save(diff_file, diff_arr)
        raise AssertionError(f"NPY mismatch.\nGenerated: {out_npy}\nDiff array: {diff_file}\nExpected: {gt_npy}")
