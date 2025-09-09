import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_plt_file(file_path: str | Path, skip_zeros: bool = True) -> np.ndarray[np.float64]:
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


def generate_colorbar_image(
    colorbar_height: int, colorbar_width: int, cmap: str, display_vmin: float, display_vmax: float
) -> np.ndarray:
    """
    Generate a colorbar image as a numpy array.

    Args:
        colorbar_height: Height of the colorbar image
        colorbar_width: Width of the colorbar image
        cmap: Matplotlib colormap name
        display_vmin: Minimum value for the colorbar
        display_vmax: Maximum value for the colorbar

    Returns:
        Numpy array of the colorbar image with values in [0, 1]
    """
    h, w = colorbar_height, colorbar_width
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("dejavusans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

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
    tick_values = np.linspace(display_vmax, display_vmin, num_ticks)
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
