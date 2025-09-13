import os
import platform

import numpy as np
import pytest
from PIL import Image, ImageChops

from yumo.geometry_utils import unwrap_uv, uv_mask
from yumo.utils import load_mesh


@pytest.mark.usefixtures("refresh_golden")
def test_uv_binary_mask(test_data, tmp_path, refresh_golden):
    """
    End-to-end test of UV binary mask generation.

    Golden standard files are system-specific. Expected filenames:
      uv_mask_<System>_gt.png
      uv_mask_<System>_gt.npy
    """

    system = platform.system()  # e.g. "Darwin", "Linux", "Windows"

    # -- 1. Load mesh (vertices, faces) --
    mesh_path = os.path.join(test_data, "sample.STL")
    vertices, faces = load_mesh(mesh_path)

    # -- 2. Unwrap UVs --
    (
        param_corner,
        H,
        W,
        vmapping,
        faces_unwrapped,
        uvs,
        vertices_unwrapped,
    ) = unwrap_uv(vertices, faces, brute_force=True)

    # -- 3. Generate UV binary mask --
    mask = uv_mask(uvs, faces_unwrapped, W, H)

    # Convert to image for debugging or comparison
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    # Compute golden standard paths
    gt_png = os.path.join(test_data, f"uv_mask_{system}_gt.png")
    gt_npy = os.path.join(test_data, f"uv_mask_{system}_gt.npy")
    out_png = tmp_path / "uv_mask.png"
    out_npy = tmp_path / "uv_mask.npy"

    # Always save generated outputs for debugging
    mask_img.save(out_png)
    np.save(out_npy, mask)

    if refresh_golden:
        # Overwrite goldens for this system
        mask_img.save(gt_png)
        np.save(gt_npy, mask)
        pytest.skip("Golden standards refreshed, skipping comparison.")

    # If expected goldens for this system are missing -> skip test
    if not (os.path.exists(gt_png) and os.path.exists(gt_npy)):
        pytest.skip(f"No golden standards available for system {system}")

    # --- Compare PNG golden ---
    gt_img = Image.open(gt_png).convert("L")
    assert mask_img.size == gt_img.size, f"Image size mismatch: got {mask_img.size}, expected {gt_img.size}"

    diff = ImageChops.difference(mask_img, gt_img)
    if diff.getbbox() is not None:
        diff_file = tmp_path / "uv_mask_diff.png"
        diff.save(diff_file)
        raise AssertionError(f"PNG mismatch.\nGenerated: {out_png}\nDiff: {diff_file}\nExpected: {gt_png}")

    # --- Compare NPY golden ---
    gt_mask = np.load(gt_npy)
    assert mask.shape == gt_mask.shape, f"Array shape mismatch: got {mask.shape}, expected {gt_mask.shape}"

    if not np.array_equal(mask, gt_mask):
        diff_arr = mask.astype(int) - gt_mask.astype(int)
        diff_file = tmp_path / "uv_mask_diff.npy"
        np.save(diff_file, diff_arr)
        raise AssertionError(f"NPY mismatch.\nGenerated: {out_npy}\nDiff array: {diff_file}\nExpected: {gt_npy}")
