import json
import logging
import re
import time
from contextlib import ContextDecorator
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import Colormap, LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import KDTree
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CMAP_CACHE: dict[str, Colormap] = {}


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

        x, y, z, value = map(np.float64, match.groups())
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


def convert_power_of_10_to_scientific(x):
    """Convert 10^x to scientific notation a*10^b where b is an integer"""
    exponent = int(np.floor(x))
    coefficient = 10 ** (x - exponent)
    return coefficient, exponent


def format_scientific(x):
    """Format 10^x as aa.bbec where c is an integer"""
    coefficient, exponent = convert_power_of_10_to_scientific(x)
    return f"{coefficient:.2f}e{exponent}"


def _get_cmap(name: str, loaded_cmaps: dict[str, str] | None = None) -> Colormap:
    """
    Return a matplotlib colormap, optionally loading from image.
    Uses an internal global cache for performance.
    """
    global _CMAP_CACHE

    if name in _CMAP_CACHE:
        return _CMAP_CACHE[name]

    # Load from image if provided
    if loaded_cmaps and name in loaded_cmaps:
        path = loaded_cmaps[name]
        img = Image.open(path).convert("RGB")
        data = np.asarray(img) / 255.0
        center_row = data[data.shape[0] // 2, :, :]
        n_colors = center_row.shape[0]
        colors = [tuple(center_row[i]) for i in range(n_colors)]
        cmap = LinearSegmentedColormap.from_list(name, colors, N=n_colors)
    else:
        # fallback to standard matplotlib colormap
        cmap = plt.get_cmap(name)  # type: ignore[assignment]

    # Store in cache
    _CMAP_CACHE[name] = cmap
    return cmap


def generate_colorbar_image(
    colorbar_height: int,
    colorbar_width: int,
    cmap: str,
    c_min: float,
    c_max: float,
    method: str = "identity",
    loaded_cmaps: dict[str, str] | None = None,
) -> np.ndarray:
    """
    Generate a colorbar image as a numpy array with different labeling methods.

    Args:
        colorbar_height: Height of the colorbar image
        colorbar_width: Width of the colorbar image
        cmap: Matplotlib colormap name
        c_min: Minimum value for the colorbar
        c_max: Maximum value for the colorbar
        method: Display method for the colorbar values. Options:
            - "identity": Regular values (default)
            - "log_e": Label as e^value
        loaded_cmaps: An optional dict containing {name: cmap path}

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

    # --- get colormap ---
    colormap = _get_cmap(cmap, loaded_cmaps)
    gradient = np.linspace(1, 0, bar_height)
    bar_colors_rgba = colormap(gradient)
    bar_colors_rgb = (bar_colors_rgba[:, :3] * 255).astype(np.uint8)

    # --- draw vertical bar ---
    for i in range(bar_height):
        y_pos = bar_start_y + i
        draw.line(
            [(bar_x_pos, y_pos), (bar_x_pos + bar_width, y_pos)],
            fill=tuple(bar_colors_rgb[i]),
        )

    # --- ticks and labels ---
    num_ticks = 7
    tick_values = np.linspace(c_max, c_min, num_ticks)
    tick_positions = np.linspace(bar_start_y, bar_end_y, num_ticks)
    text_x_pos = bar_x_pos + bar_width + 10

    # Set the formatter based on method
    if method == "identity":
        # method_label = "Linear Scale"
        formatter = lambda x: f"{x:.2g}"
    elif method == "log_e":
        # method_label = "Natural Log Scale"
        formatter = lambda x: f"e^{x:.2g}"
    elif method == "log_10":
        # method_label = "Log10 Scale"
        formatter = format_scientific
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'identity' or 'log_e'")

    # draw tick marks and text
    for i, (val, pos) in enumerate(zip(tick_values, tick_positions, strict=False)):
        if i == 0:
            label = f">= {formatter(val)}"
        elif i == len(tick_values) - 1:
            label = f"<= {formatter(val)}"
        else:
            label = formatter(val)
        draw.line(
            [(bar_x_pos + bar_width, pos), (bar_x_pos + bar_width + 5, pos)],
            fill="black",
        )
        draw.text((text_x_pos, pos - 6), label, fill="black", font=font)

    return np.array(img) / 255.0


def estimate_densest_point_distance(points: np.ndarray, k: int = 1000, quantile: float = 0.01) -> np.float64:
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
        return np.float64(0.0)

    # Build KD-tree for efficient nearest neighbor search
    kdtree = KDTree(points)

    # Find distance to nearest neighbor for each sampled point
    # k=2 returns the point itself (distance 0) and the nearest neighbor
    distances, _ = kdtree.query(sampled_points, k=2)

    # Take the second column (nearest non-self neighbor)
    nearest_distances = distances[:, 1]

    # Apply outlier filtering using the quantile parameter
    if len(nearest_distances) > 1:
        threshold = np.quantile(nearest_distances, quantile)
        filtered_distances = nearest_distances[nearest_distances <= threshold]
        # Use original distances if filtering removed everything
        if len(filtered_distances) == 0:
            filtered_distances = nearest_distances
    else:
        filtered_distances = nearest_distances

    return np.float64(np.mean(filtered_distances))


def data_transform(points: np.ndarray, method: str) -> np.ndarray:
    """
    Preprocess the last column of (N, 4) points array using data transform.

    Args:
        points (np.ndarray): Array of shape (N, 4). The first three columns are coordinates,
                             the last column is the scalar value to be transformed.
        method (str): Preprocessing method. One of {"identity", "log_e", "log_10"}.

    Returns:
        np.ndarray: Preprocessed points with the same shape.
    """
    if method == "identity":
        return points

    elif method in ("log_e", "log_10"):
        transformed = points.copy()

        # 1. Select base
        log_fn = np.log if method == "log_e" else np.log10

        # 2. Mask positive values
        nonzero_mask = transformed[:, 3] > 0

        if not np.any(nonzero_mask):
            raise ValueError(f"No positive values found for {method} transform.")

        # 3. Apply log only to positive entries
        transformed[nonzero_mask, 3] = log_fn(transformed[nonzero_mask, 3])

        # 4. Find min among transformed positives
        min_value = np.min(transformed[nonzero_mask, 3])

        # 5. Replace originally non-positive values with min_value
        transformed[~nonzero_mask, 3] = min_value

        return transformed

    else:
        raise ValueError(f"Unknown data preprocess method: {method}")


def inverse_data_transform(points_or_values: np.ndarray | float, method: str) -> np.ndarray | float:
    """
    Invert the preprocessing applied by data_transform on the last column
    or on a single scalar value.

    Args:
        points_or_values (np.ndarray or float): Either
            - Array of shape (N, 4) that has been transformed, or
            - A single transformed scalar value.
        method (str): One of {"identity", "log_e", "log_10"}.

    Returns:
        np.ndarray or float: Inverse-transformed array or scalar.
    """
    if method == "identity":
        return points_or_values

    elif method in ("log_e", "log_10"):
        exp_fn = np.exp if method == "log_e" else (lambda x: np.power(10.0, x))

        if np.isscalar(points_or_values) or np.ndim(points_or_values) == 0:
            return exp_fn(points_or_values)

        inv = points_or_values.copy()  # type: ignore[union-attr]
        inv[:, 3] = exp_fn(inv[:, 3])
        return inv

    else:
        raise ValueError(f"Unknown data preprocess method: {method}")


def fmtn(vec: np.ndarray | list[float] | tuple[float, ...], n: int, precision: int = 2) -> str:
    """Format the first n components of a vector with fixed precision.

    Args:
        vec: Vector (numpy array, list, or tuple of floats).
        n: Number of components to format.
        precision: Decimal precision.

    Returns:
        A string like "(x1, x2, ..., xn)" with formatted floats.
    """
    formatted = [f"{vec[i]:.{precision}f}" for i in range(n)]
    return f"({', '.join(formatted)})"


def fmt2(vec: np.ndarray | list[float] | tuple[float, float], precision: int = 2) -> str:
    return fmtn(vec, 2, precision)


def fmt3(vec: np.ndarray | list[float] | tuple[float, float, float], precision: int = 2) -> str:
    return fmtn(vec, 3, precision)


def export_camera_view(view_mat: np.ndarray) -> str:
    """
    Export a 4x4 camera view matrix to a JSON string (list of lists).

    Args:
        view_mat (np.ndarray): A 4x4 numpy matrix.

    Returns:
        str: JSON-formatted string representing the matrix.
    """
    if view_mat.shape != (4, 4):
        raise ValueError(f"Expected (4,4) matrix, got {view_mat.shape}")

    # Convert numpy array to nested list
    mat_list = view_mat.tolist()
    return json.dumps(mat_list)


def load_camera_view(json_str: str) -> np.ndarray:
    """
    Load a 4x4 camera view matrix from a JSON string.

    Args:
        json_str (str): JSON string representing the matrix.

    Returns:
        np.ndarray: A 4x4 numpy matrix.
    """
    mat_list = json.loads(json_str)
    mat = np.array(mat_list, dtype=float)

    if mat.shape != (4, 4):
        raise ValueError(f"Expected (4,4) matrix, got {mat.shape}")

    return mat
