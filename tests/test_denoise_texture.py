import os
import platform

import numpy as np
import pytest
from PIL import Image, ImageChops

from yumo.constants import DENOISE_METHODS
from yumo.geometry_utils import denoise_texture


@pytest.mark.parametrize("method", DENOISE_METHODS)
def test_denoise_texture(test_data, tmp_path, refresh_golden, method):
    """
    End-to-end test of sparse texture denoising.

    Golden standard files are system-specific. Expected filenames:
      denoised_<method>_<System>_gt.png
      denoised_<method>_<System>_gt.npy
    """

    system = platform.system()

    # -- 1. Load inputs --
    # input is already sparse baked texture from golden .npy
    sparse_file_npy = os.path.join(test_data, f"texture_bake_{system}_gt.npy")
    uv_mask_file_npy = os.path.join(test_data, f"uv_mask_{system}_gt.npy")

    if not (os.path.exists(sparse_file_npy) and os.path.exists(uv_mask_file_npy)):
        pytest.skip(f"No sparse texture or UV mask available for system {system}")

    sparse_tex = np.load(sparse_file_npy)
    uv_mask = np.load(uv_mask_file_npy)

    # Randomly perturb the input a bit (so denoise must actually smooth it)
    rng = np.random.default_rng(42)
    sparse_tex = sparse_tex * (1 + rng.random(sparse_tex.shape) / 10)

    # -- 2. Run denoising --
    denoised_tex = denoise_texture(sparse_tex, method=method, mask=uv_mask)

    # -- 3. Save outputs for debug --
    out_npy = tmp_path / f"denoised_{method}.npy"
    out_png = tmp_path / f"denoised_{method}.png"
    np.save(out_npy, denoised_tex)

    denoised_img = Image.fromarray((denoised_tex / denoised_tex.max() * 255).astype(np.uint8))
    denoised_img.save(out_png)

    # -- 4. System-specific golden filenames --
    gt_png = os.path.join(test_data, f"denoised_{method}_{system}_gt.png")
    gt_npy = os.path.join(test_data, f"denoised_{method}_{system}_gt.npy")

    if refresh_golden:
        denoised_img.save(gt_png)
        np.save(gt_npy, denoised_tex)
        pytest.skip("Golden standards refreshed, skipping comparison.")

    if not (os.path.exists(gt_png) and os.path.exists(gt_npy)):
        pytest.skip(f"No golden standards available for method={method}, system={system}")

    # --- PNG comparison ---
    gt_img = Image.open(gt_png).convert("L")
    assert denoised_img.size == gt_img.size, "Image size mismatch"

    diff = ImageChops.difference(denoised_img, gt_img)
    if diff.getbbox() is not None:
        diff_file = tmp_path / f"denoised_{method}_diff.png"
        diff.save(diff_file)
        raise AssertionError(f"PNG mismatch.\nGenerated: {out_png}\nDiff: {diff_file}\nExpected: {gt_png}")

    # --- NPY comparison ---
    gt_tex = np.load(gt_npy)
    assert denoised_tex.shape == gt_tex.shape, "Array shape mismatch"
    if not np.allclose(denoised_tex, gt_tex, atol=1e-6):
        diff_arr = denoised_tex - gt_tex
        diff_file = tmp_path / f"denoised_{method}_diff.npy"
        np.save(diff_file, diff_arr)
        raise AssertionError(f"NPY mismatch.\nGenerated: {out_npy}\nDiff array: {diff_file}\nExpected: {gt_npy}")
