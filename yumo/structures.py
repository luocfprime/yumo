import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from yumo.constants import DENOISE_METHODS
from yumo.context import Context
from yumo.geometry_utils import (
    bake_to_texture,
    denoise_texture,
    map_to_uv,
    query_scalar_field,
    sample_surface,
    unwrap_uv,
)
from yumo.ui import ui_combo, ui_item_width, ui_tree_node

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

    def register(self, force: bool = False):
        """Registers the structure's geometry with Polyscope. (Called every frame, but runs once)."""
        if not self.is_valid():
            return
        if self._is_registered and not force:
            return
        self._do_register()
        if self.polyscope_structure:
            self.polyscope_structure.set_enabled(self.enabled)
            self._is_registered = True

    def add_prepared_quantities(self):
        """Adds all prepared scalar quantities to the registered Polyscope structure."""
        logger.debug(f"Updating quantities for structure: '{self.name}'")

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
        logger.debug(f"Registering point cloud geometry: '{self.name}'")
        p = ps.register_point_cloud(self.name, self.points[:, :3])
        p.set_radius(self.app_context.points_radius, relative=False)
        p.set_point_render_mode(self.app_context.points_render_mode)

    def set_point_render_mode(self, mode: str):
        if self.polyscope_structure:
            self.polyscope_structure.set_point_render_mode(mode)

    def set_radius(self, radius: float, relative: bool = False):
        if self.polyscope_structure:
            self.polyscope_structure.set_radius(radius, relative=relative)

    def ui(self):
        """Points related UI"""
        with ui_tree_node("Points", open_first_time=True) as expanded:
            if not expanded:
                return

            with ui_item_width(100):
                changed, show = psim.Checkbox("Show", self.enabled)
                if changed:
                    self.set_enabled(show)

                psim.SameLine()

                changed, radius = psim.SliderFloat(
                    "Radius",
                    self.app_context.points_radius,
                    v_min=self.app_context.points_densest_distance * 0.01,
                    v_max=self.app_context.points_densest_distance * 0.20,
                    format="%.4g",
                )
                if changed:
                    self.app_context.points_radius = radius
                    self.set_radius(radius)

                psim.SameLine()

                with ui_combo("Render Mode", self.app_context.points_render_mode) as expanded:
                    if expanded:
                        for mode in ["sphere", "quad"]:
                            selected, _ = psim.Selectable(mode, mode == self.app_context.points_render_mode)
                            if selected and mode != self.app_context.points_render_mode:
                                self.app_context.points_render_mode = mode
                                self.set_point_render_mode(mode)

            psim.Separator()

    def callback(self):
        pass


class MeshStructure(Structure):
    """Represents a surface mesh structure."""

    QUANTITY_NAME: ClassVar[str] = "mesh_texture_values"

    def __init__(self, name: str, app_context: "Context", vertices: np.ndarray, faces: np.ndarray, **kwargs):
        super().__init__(name, app_context, **kwargs)
        self.vertices = vertices
        self.faces = faces

        self.points_per_area = 1.0  # TODO: initialize with default

        # texture related
        self.param_corner = None
        self.texture_height = None
        self.texture_width = None
        self.vmapping = None
        self.faces_unwrapped = None
        self.uvs = None
        self.vertices_unwrapped = None

        self._resolution_changed = False
        self._should_bake = False
        self._display_mode = "preview"  # one of "preview", "baked"
        self._denoise_method = "linear"  # one of DENOISE_METHODS

    @property
    def polyscope_structure(self):
        return ps.get_surface_mesh(self.name)

    def is_valid(self) -> bool:
        return self.vertices is not None and self.faces is not None

    def prepare_quantities(self):
        """
        We preview the resolution first before proceeding to actual data sampling in update_data_texture.
        """
        self.update_resolution_preview()

    def update_resolution_preview(self):
        """Add a preview quantity of the surface sampling resolution using bake_texture"""
        self.prepared_quantities[self.QUANTITY_NAME] = self.bake_texture(
            sampler_func=lambda p: np.ones(p.shape[0]) * self.app_context.color_max,  # 0 (no fill); color_max (filled)
            denoise_func=lambda x: x,  # no denoise, as we want to visualize the resolution
        )

    def update_data_texture(self):
        """Sample the data points and update texture map"""
        self.prepared_quantities[self.QUANTITY_NAME] = self.bake_texture(
            sampler_func=functools.partial(query_scalar_field, data_points=self.app_context.points),
            denoise_func=functools.partial(denoise_texture, method=self._denoise_method),
        )

    def add_prepared_quantities(self):
        """Adds all prepared scalar quantities to the registered Polyscope structure."""
        logger.debug(f"Updating quantities for structure: '{self.name}'")

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
                defined_on="texture",  # Use texture coordinates
                param_name="uv",
                cmap=self.app_context.cmap,
                vminmax=(self.app_context.color_min, self.app_context.color_max),
            )
        self._quantities_added = True

    def _do_register(self):
        """Register only the mesh geometry."""
        logger.debug(f"Registering mesh geometry: '{self.name}'")
        mesh = ps.register_surface_mesh(self.name, self.vertices, self.faces)
        mesh.set_material("clay")
        mesh.set_color([0.7, 0.7, 0.7])

        # add uv parameterization
        (
            self.param_corner,
            self.texture_height,
            self.texture_width,
            self.vmapping,
            self.faces_unwrapped,
            self.uvs,
            self.vertices_unwrapped,
        ) = unwrap_uv(self.vertices, self.faces)

        mesh.add_parameterization_quantity("uv", self.param_corner, defined_on="corners", enabled=True)

    def ui(self):
        """Mesh related UI"""
        with ui_tree_node("Mesh", open_first_time=True) as expanded:
            if not expanded:
                return

            with ui_item_width(100):
                changed, show = psim.Checkbox("Show", self.enabled)
                if changed:
                    self.set_enabled(show)

                psim.SameLine()

                v_speed = 1.0
                changed, resolution = psim.DragFloat(
                    "Points / Unit Area", self.points_per_area, v_speed=v_speed, v_min=0.01, v_max=1000.0
                )
                if changed:
                    self.points_per_area = resolution
                    self._resolution_changed = True

                with ui_combo("Denoise Method", self._denoise_method) as expanded:
                    if expanded:
                        for method in DENOISE_METHODS:
                            selected, _ = psim.Selectable(method, method == self._denoise_method)
                            if selected and method != self._denoise_method:
                                self._denoise_method = method
                                self._should_bake = (
                                    self._display_mode == "baked"
                                )  # needs re-bake if denoise method changed

                psim.SameLine()

                if psim.Button("Bake"):
                    self._should_bake = True

            psim.Separator()

    def bake_texture(
        self, sampler_func: Callable[[np.ndarray], np.ndarray], denoise_func: Callable[[np.ndarray], np.ndarray]
    ):
        """

        Args:
            sampler_func: Takes in sample query points (N, 3), outputs values (N,)
            denoise_func: Smooth the texture result

        Returns:

        """
        # -- 1. Sample surface --
        points, bary, indices = sample_surface(
            self.vertices_unwrapped, self.faces_unwrapped, points_per_area=self.points_per_area
        )

        # -- 2. Map samples to UV space --
        sample_uvs = map_to_uv(self.uvs, self.faces_unwrapped, bary, indices)

        # -- 3. Sample scalar field --
        values = sampler_func(points)

        # -- 4. Bake to texture --
        tex = bake_to_texture(sample_uvs, values, self.texture_height, self.texture_width)

        # -- 5. Denoise --
        tex = denoise_func(tex)

        return tex

    def callback(self):
        if self._resolution_changed:
            self._resolution_changed = False
            self._display_mode = "preview"  # set preview resolution change before baking
            self.update_resolution_preview()  # update texture
            self.add_prepared_quantities()  # update polyscope quantity

        if self._should_bake:
            self._should_bake = False
            self._display_mode = "baked"
            self.update_data_texture()
            self.add_prepared_quantities()
