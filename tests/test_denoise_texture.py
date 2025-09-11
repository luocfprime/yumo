import os

import numpy as np
import pytest
from PIL import Image, ImageChops

from yumo.constants import DENOISE_METHODS
from yumo.geometry_utils import denoise_texture


@pytest.mark.parametrize("method", DENOISE_METHODS)
def test_denoise_texture(test_data, tmp_path, method):
    """
    End-to-end test of sparse texture denoising.

    Steps:
    1. Load sparse baked texture (texture_bake_gt.npy)
    2. Run denoise_texture with the chosen interpolation method
    3. Save result to disk (npy + png) for debugging
    4. Compare against golden denoised reference (texture_bake_gt.npy, texture_bake_gt.png)
    """

    # -- 1. Load input sparse texture --
    sparse_file_npy = os.path.join(test_data, "texture_bake_gt.npy")
    sparse_tex = np.load(sparse_file_npy)
    # random perturb the sparse_tex a bit
    rng = np.random.default_rng(42)
    sparse_tex *= 1 + rng.random(sparse_tex.shape) / 10

    # -- 2. Run denoising --
    denoised_tex = denoise_texture(sparse_tex, method=method)

    # -- 3. Save output --
    np.save(tmp_path / f"denoised_{method}.npy", denoised_tex)
    denoised_img = Image.fromarray((denoised_tex / denoised_tex.max() * 255).astype(np.uint8))
    out_file = tmp_path / f"denoised_{method}.png"
    denoised_img.save(out_file)

    # -- 4. Load golden reference --
    gt_file_png = os.path.join(test_data, f"denoised_{method}_gt.png")
    gt_img = Image.open(gt_file_png).convert("L")

    # Compare size
    assert denoised_img.size == gt_img.size, "Denoised texture size mismatch"

    # Pixel-wise difference
    diff = ImageChops.difference(denoised_img, gt_img)
    if diff.getbbox() is not None:
        diff_file = tmp_path / "diff.png"
        diff.save(diff_file)
        raise AssertionError(
            f"Denoised texture does not match golden reference.\n"
            f"Generated: {out_file}\n"
            f"Diff: {diff_file}\n"
            f"Expected: {gt_file_png}"
        )
