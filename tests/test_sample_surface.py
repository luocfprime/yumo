import os.path

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from yumo.geometry_utils import sample_surface


def test_sample_surface_triangle():
    # Define a single right triangle in XY plane
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = np.array([[0, 1, 2]])

    points, bary, indices = sample_surface(vertices, faces, points_per_area=10.0)

    # Basic sanity checks
    assert points.ndim == 2 and points.shape[1] == 3
    assert bary.ndim == 2 and bary.shape[1] == 3
    assert indices.ndim == 1
    assert len(points) == len(bary) == len(indices)

    # Barycentric coords must sum to ~1
    np.testing.assert_allclose(bary.sum(axis=1), 1.0, rtol=1e-6)

    # Points must lie in the triangle
    assert np.all((bary >= 0) & (bary <= 1))


def test_sample_surface_with_image(tmp_path, test_data):
    """Test if the sample_surface results looks right."""
    # Deterministic RNG to make results reproducible with the stored GT.
    rng = np.random.default_rng(42)

    # Arbitrary quad split into two triangles
    vertices = np.array([
        [23.3, 7.1, 0.0],
        [42.0, 6.5, 0.0],
        [40.2, 19.8, 0.0],
        [25.0, 18.0, 0.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])

    points, bary, indices = sample_surface(vertices, faces, points_per_area=500.0, rng=rng)

    # --- Draw image ---
    size = 500
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # Normalize to [0,1] for drawing only
    all_xy = np.vstack([vertices[:, :2], points[:, :2]])
    mins = all_xy.min(axis=0)
    maxs = all_xy.max(axis=0)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)

    def norm_xy(xy):
        return (xy - mins) / span

    def to_pix(v):
        x, y, _ = v
        x, y = norm_xy(np.array([x, y]))
        px = int(x * (size - 1))
        py = int((1 - y) * (size - 1))  # flip y-axis
        return px, py

    # Draw triangle edges
    for f in faces:
        tri = [to_pix(vertices[i]) for i in f] + [to_pix(vertices[f[0]])]
        draw.line(tri, fill="red", width=1)

    # Draw sampled points
    for v in points:
        draw.point(to_pix(v), fill="blue")

    img_file = tmp_path / "sample_surface_42.png"
    img.save(img_file)

    # --- Compare with golden image ---
    gt_path = os.path.join(test_data, "sample_surface_42_gt.png")
    gt_img = Image.open(gt_path).convert("RGB")

    if img.size != gt_img.size:
        raise AssertionError(f"Image size mismatch: {img.size} vs {gt_img.size}")

    diff = ImageChops.difference(img, gt_img)

    if diff.getbbox() is not None:
        diff_file = tmp_path / "diff.png"

        diff.save(diff_file)
        raise AssertionError(
            f"Image does not match golden. See generated={img_file}, diff={diff_file}, expected={gt_path}"
        )
