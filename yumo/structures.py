import logging
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial import KDTree

from yumo.context import Context
from yumo.ui import ui_tree_node

logger = logging.getLogger(__name__)


class Structure(ABC):
    """Abstract base class for a visualizable structure in Polyscope."""

    def __init__(self, name: str, app_context: "Context", enabled: bool = True):
        self.name = name
        self.app_context = app_context
        self.enabled = enabled

        self._is_registered = False
        self._quantities_added = False
        self.prepared_quantities: dict[str, np.ndarray] = {}

    def register(self):
        """Registers the structure's geometry with Polyscope. (Called every frame, but runs once)."""
        if self._is_registered or not self.is_valid():
            return
        self._do_register()
        if self.polyscope_structure:
            self.polyscope_structure.set_enabled(self.enabled)
            self._is_registered = True

    def add_prepared_quantities(self):
        """Adds all prepared scalar quantities to the registered Polyscope structure."""
        if not self._is_registered:
            raise RuntimeError("Structure must be registered before adding quantities.")

        struct = self.polyscope_structure
        if not struct:
            return

        for name, values in self.prepared_quantities.items():
            struct.add_scalar_quantity(
                name,
                values,
                enabled=True,
                cmap=self.app_context.cmap,
                vminmax=(self.app_context.color_min, self.app_context.color_max),
            )
        self._quantities_added = True

    def update_all_quantities_colormap(self):
        """Updates the colormap and range for all managed quantities."""
        self.add_prepared_quantities()  # in Polyscope, re-adding quantities overwrites existing quantities

    def set_enabled(self, enabled: bool):
        """Enable or disable the structure in the UI."""
        self.enabled = enabled
        if self.polyscope_structure:
            self.polyscope_structure.set_enabled(self.enabled)

    @property
    @abstractmethod
    def polyscope_structure(self):
        """Get the underlying Polyscope structure object."""
        pass

    @abstractmethod
    def prepare_quantities(self):
        """Subclass-specific logic to calculate and prepare scalar data arrays."""
        pass

    @abstractmethod
    def _do_register(self):
        """Subclass-specific geometry registration logic."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if the structure has valid data to be registered."""
        pass

    @abstractmethod
    def ui(self):
        """Update structure related UI"""
        pass

    @abstractmethod
    def callback(self):
        """Update structure related callback"""
        pass


class PointCloudStructure(Structure):
    """Represents a point cloud structure."""

    QUANTITY_NAME: ClassVar[str] = "point_values"

    def __init__(self, name: str, app_context: "Context", points: np.ndarray, **kwargs):
        super().__init__(name, app_context, **kwargs)
        self.points = points

    @property
    def polyscope_structure(self):
        return ps.get_point_cloud(self.name)

    def is_valid(self) -> bool:
        return self.points is not None and self.points.shape[0] > 0

    def prepare_quantities(self):
        """Prepare the scalar data from the 4th column of the points array."""
        if self.is_valid():
            self.prepared_quantities[self.QUANTITY_NAME] = self.points[:, 3]

    def _do_register(self):
        """Register only the point cloud geometry (XYZ coordinates)."""
        logger.info(f"Registering point cloud geometry: '{self.name}'")
        p = ps.register_point_cloud(self.name, self.points[:, :3])
        p.set_radius(self.app_context.points_size, relative=False)
        p.set_point_render_mode(self.app_context.points_render_mode)

    def set_radius(self, radius: float, relative: bool = False):
        if self.polyscope_structure:
            self.polyscope_structure.set_radius(radius, relative=relative)

    def ui(self):
        """Points related UI"""
        with ui_tree_node("Points", open_first_time=True) as expanded:
            if not expanded:
                return

            changed, show = psim.Checkbox("Show", self.enabled)
            if changed:
                self.set_enabled(show)

            psim.SameLine()
            changed, radius = psim.SliderFloat(
                "Radius", self.app_context.points_size, v_min=1e-4, v_max=5e-2, format="%.4g"
            )
            if changed:
                self.app_context.points_size = radius
                self.set_radius(radius)

        psim.Separator()

    def callback(self):
        pass


class MeshStructure(Structure):
    """Represents a surface mesh structure."""

    QUANTITY_NAME: ClassVar[str] = "mesh_values"

    def __init__(self, name: str, app_context: "Context", vertices: np.ndarray, faces: np.ndarray, **kwargs):
        super().__init__(name, app_context, **kwargs)
        self.vertices = vertices
        self.faces = faces

    @property
    def polyscope_structure(self):
        return ps.get_surface_mesh(self.name)

    def is_valid(self) -> bool:
        return self.vertices is not None and self.faces is not None

    def prepare_quantities(self):
        """
        Interpolate scalar values from the source point cloud onto the mesh vertices
        using nearest-neighbor lookup. This is the expensive, one-time calculation.
        """
        source_points = self.app_context.points
        if not self.is_valid() or source_points is None or source_points.shape[0] == 0:
            return

        logger.info(f"Preparing nearest-neighbor scalar data for mesh '{self.name}'...")
        kdtree = KDTree(source_points[:, :3])
        _, nearest_indices = kdtree.query(
            self.vertices, k=1
        )  # TODO: check if using self.vertices make sense? Also distance_upper_bound
        interpolated_values = source_points[nearest_indices, 3]
        self.prepared_quantities[self.QUANTITY_NAME] = interpolated_values

    def _do_register(self):
        """Register only the mesh geometry."""
        logger.info(f"Registering mesh geometry: '{self.name}'")
        mesh = ps.register_surface_mesh(self.name, self.vertices, self.faces)
        mesh.set_material("clay")
        mesh.set_color([0.7, 0.7, 0.7])

    def ui(self):
        pass

    def callback(self):
        pass
