import logging
import re
import time
from contextlib import ContextDecorator
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import KDTree
from tqdm import tqdm

logger = logging.getLogger(__name__)


class profiler(ContextDecorator):
    def __init__(self, name=None, profiler_logger=None):
        self.name = name
        self.profiler_logger = profiler_logger or logger

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self._start
        self.profiler_logger.debug(f"{self.name} took {elapsed:.6f} seconds")

    def __call__(self, func):
        prof_name = self.name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with profiler(prof_name):
                return func(*args, **kwargs)

        return wrapper


def load_mesh(file_path: str | Path, return_trimesh: bool = False) -> trimesh.Trimesh | tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(file_path)
    if return_trimesh:
        return mesh
    return mesh.vertices, mesh.faces


def parse_plt_file(file_path: str | Path, skip_zeros: bool = False) -> np.ndarray:
    logger.info(f"Parsing file: {file_path}")
    points = []

    with open(file_path) as f:
        lines = f.readlines()

    data_pattern = re.compile(
        r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
    )

    for line in tqdm(lines, desc="Processing data"):
        match = data_pattern.match(line.strip())
        if not match:
            continue

        x, y, z, value = map(float, match.groups())
        if skip_zeros and value == 0.0:
            continue

        points.append([x, y, z, value])

    if skip_zeros:
        logger.info("Skipped points with value = 0.0")

    logger.info(f"Kept {len(points):,} points out of {len(lines):,}.")
    if len(points) == 0:
        raise ValueError("No points left after filtering")

    return np.array(points)


def write_plt_file(path: Path, points: np.ndarray):
    """
    Write points to a Tecplot ASCII .plt file (FEPOINT format).
    """
    n = len(points)
    with open(path, "w") as f:
        f.write("variables = x, y, z, Value(m-3)\n")
        f.write(f"zone N={n}, E=0, F=FEPOINT, ET=POINT\n")
        np.savetxt(f, points, fmt="%.6f")


def generate_colorbar_image(
    colorbar_height: int, colorbar_width: int, cmap: str, c_min: float, c_max: float
) -> np.ndarray:
    """
    Generate a colorbar image as a numpy array.

    Args:
        colorbar_height: Height of the colorbar image
        colorbar_width: Width of the colorbar image
        cmap: Matplotlib colormap name
        c_min: Minimum value for the colorbar
        c_max: Maximum value for the colorbar

    Returns:
        Numpy array of the colorbar image with values in [0, 1]
    """
    h, w = colorbar_height, colorbar_width
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("dejavusans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()  # type: ignore[assignment]

    bar_width = 25
    bar_x_pos = (w - bar_width) // 6
    text_padding = 15
    bar_start_y = text_padding
    bar_end_y = h - text_padding
    bar_height = bar_end_y - bar_start_y
    colormap = plt.get_cmap(cmap)
    gradient = np.linspace(1, 0, bar_height)
    bar_colors_rgba = colormap(gradient)
    bar_colors_rgb = (bar_colors_rgba[:, :3] * 255).astype(np.uint8)

    for i in range(bar_height):
        y_pos = bar_start_y + i
        draw.line(
            [(bar_x_pos, y_pos), (bar_x_pos + bar_width, y_pos)],
            fill=tuple(bar_colors_rgb[i]),
        )

    num_ticks = 7
    tick_values = np.linspace(c_max, c_min, num_ticks)
    tick_positions = np.linspace(bar_start_y, bar_end_y, num_ticks)
    text_x_pos = bar_x_pos + bar_width + 10

    for i, (val, pos) in enumerate(zip(tick_values, tick_positions, strict=False)):
        if i == 0:
            label = f">= {val:.2g}"
        elif i == len(tick_values) - 1:
            label = f"<= {val:.2g}"
        else:
            label = f"{val:.2g}"
        draw.line(
            [(bar_x_pos + bar_width, pos), (bar_x_pos + bar_width + 5, pos)],
            fill="black",
        )
        draw.text((text_x_pos, pos - 6), label, fill="black", font=font)

    return np.array(img) / 255.0


def estimate_densest_point_distance(points: np.ndarray, k: int = 1000, quantile: float = 0.01) -> float:
    """
    Estimate the densest distance between points and their nearest neighbors.

    This function samples k points from the input dataset, finds their nearest
    neighbors, and calculates the average distance after filtering outliers.

    Args:
        points: Array of shape (n, d) containing n points in d-dimensional space.
        k: Number of points to sample for the estimation. Default is 1000.
        quantile: Quantile threshold for outlier removal. Default is 0.01.
            Only distances in the range [min, quantile] are considered.

    Returns:
        float: Estimated densest distance to nearest neighbor after outlier filtering.

    Raises:
        ValueError: If points is empty or not a 2D array.
    """
    # Input validation
    if points.ndim != 2 or points.size == 0:
        raise ValueError("Input 'points' must be a non-empty 2D array")

    n = points.shape[0]

    # Handle case where number of points is less than k
    sample_size = min(n, k)
    sample_indices = np.random.choice(n, size=sample_size, replace=False) if n > 1 else np.array([0])
    sampled_points = points[sample_indices]

    # Handle edge case of a single point
    if n == 1:
        return 0.0

    # Build KD-tree for efficient nearest neighbor search
    kdtree = KDTree(points)

    # Find distance to nearest neighbor for each sampled point
    # k=2 returns the point itself (distance 0) and the nearest neighbor
    distances, _ = kdtree.query(sampled_points, k=2)

    # Take the second column (nearest non-self neighbor)
    nearest_distances = distances[:, 1]  # type: ignore[index]

    # Apply outlier filtering using the quantile parameter
    if len(nearest_distances) > 1:
        threshold = np.quantile(nearest_distances, quantile)
        filtered_distances = nearest_distances[nearest_distances <= threshold]
        # Use original distances if filtering removed everything
        if len(filtered_distances) == 0:
            filtered_distances = nearest_distances
    else:
        filtered_distances = nearest_distances

    return float(np.mean(filtered_distances))
