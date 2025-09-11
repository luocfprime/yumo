import logging
from typing import ClassVar

import numpy as np
import polyscope as ps
from polyscope import imgui as psim

from yumo.base_structure import Structure
from yumo.context import Context
from yumo.ui import ui_combo, ui_item_width, ui_tree_node

logger = logging.getLogger(__name__)


class PointCloudStructure(Structure):
    """Represents a point cloud structure."""

    QUANTITY_NAME: ClassVar[str] = "point_values"

    def __init__(self, name: str, app_context: "Context", points: np.ndarray, **kwargs):
        super().__init__(name, app_context, **kwargs)
        self.points = points

        # 10% of the densest point distance
        self._points_radius = 0.1 * self.app_context.points_densest_distance
        self._points_render_mode = "sphere"  # one of "sphere" or "quad", the latter is less costly

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
        p.set_radius(self._points_radius, relative=False)
        p.set_point_render_mode(self._points_render_mode)

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
                    self._points_radius,
                    v_min=self.app_context.points_densest_distance * 0.01,
                    v_max=self.app_context.points_densest_distance * 0.20,
                    format="%.4g",
                )
                if changed:
                    self._points_radius = radius
                    self.set_radius(radius)

                psim.SameLine()

                with ui_combo("Render Mode", self._points_render_mode) as expanded:
                    if expanded:
                        for mode in ["sphere", "quad"]:
                            selected, _ = psim.Selectable(mode, mode == self._points_render_mode)
                            if selected and mode != self._points_render_mode:
                                self._points_render_mode = mode
                                self.set_point_render_mode(mode)

            psim.Separator()

    def callback(self):
        pass
