import os

import numpy as np
from PIL import Image, ImageChops

from yumo.geometry_utils import bake_to_texture, map_to_uv, sample_surface, unwrap_uv
from yumo.utils import load_mesh


def test_e2e_texture_bake(test_data, tmp_path):
    """
    End-to-end test of texture baking.

    Steps:
    1. Load a mesh model from test_data/sample.STL
    2. Perform UV unwrapping via xatlas
    3. Sample surface points with sample_surface
    4. Map sampled points to UV space
    5. Bake scalar values (all ones) into a texture map
    6. Save the baked texture image to disk
    7. Compare against golden reference image pixel-by-pixel
    """
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
    ) = unwrap_uv(vertices, faces)

    # -- 3. Sample surface --
    rng = np.random.default_rng(42)
    points, bary, indices = sample_surface(vertices_unwrapped, faces_unwrapped, points_per_area=500.0, rng=rng)

    # -- 4. Map samples to UV space --
    sample_uvs = map_to_uv(uvs, faces_unwrapped, bary, indices)

    # -- 5. Assign scalar values (all ones for coverage test) --
    values = np.ones(len(points), dtype=float)

    # -- 6. Bake to texture --
    tex = bake_to_texture(sample_uvs, values, H, W)

    np.save(tmp_path / "texture_bake.npy", tex)

    # Scale and convert to image
    tex_norm = (tex / tex.max() * 255).astype(np.uint8)
    img = Image.fromarray(tex_norm)

    # Save image for debug
    out_file = tmp_path / "texture_bake.png"
    img.save(out_file)

    # -- 7. Compare against golden reference --

    gt_file = os.path.join(test_data, "texture_bake_gt.png")
    gt_img = Image.open(gt_file).convert("L")

    assert img.size == gt_img.size, "Baked texture size mismatch"

    diff = ImageChops.difference(img, gt_img)
    if diff.getbbox() is not None:
        diff_file = tmp_path / "diff.png"
        diff.save(diff_file)
        raise AssertionError(
            f"Texture bake result does not match golden reference.\n"
            f"Generated: {out_file}\n"
            f"Diff: {diff_file}\n"
            f"Expected: {gt_file}"
        )

    # Also make sure the npy matches the golden reference
    gt_tex = np.load(os.path.join(test_data, "texture_bake_gt.npy"))
    assert np.allclose(tex, gt_tex, atol=1e-6), "Baked texture does not match golden reference"
