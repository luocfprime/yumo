from dataclasses import dataclass

import numpy as np

from yumo.constants import CMAPS


# --- Context ---
@dataclass
class Context:
    # Data
    points: np.ndarray = None
    mesh_vertices: np.ndarray | None = None
    mesh_faces: np.ndarray | None = None

    # Statistics
    min_value: float = None
    max_value: float = None

    center: np.ndarray = None  # (3, )
    bbox_min: np.ndarray = None
    bbox_max: np.ndarray = None

    points_densest_distance: float = None

    # Settings
    cmap: str = CMAPS[0]
    color_min: float = None
    color_max: float = None
