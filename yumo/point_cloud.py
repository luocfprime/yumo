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

        # initialize threshold at top 10% of scalar values
        self.visualize_threshold_min = np.float64(np.percentile(points[:, 3], 90))
        self.visualize_threshold_max = points[:, 3].max()

        # 10% of the densest point distance
        self._points_radius = 0.1 * self.app_context.points_densest_distance
        self._points_render_mode = "sphere"  # one of "sphere" or "quad"

    @property
    def polyscope_structure(self):
        return ps.get_point_cloud(self.name)

    def is_valid(self) -> bool:
        return self.points is not None and self.points.shape[0] > 0

    def get_filtered_points(self):
        """Filter points above threshold."""
        if not self.is_valid():
            return np.empty((0, 4))
        return self.points[
            (self.points[:, 3] >= self.visualize_threshold_min) & (self.points[:, 3] <= self.visualize_threshold_max)
        ]

    def prepare_quantities(self):
        """Prepare scalar data only for filtered points."""
        filtered = self.get_filtered_points()
        if filtered.shape[0] > 0:
            self.prepared_quantities[self.QUANTITY_NAME] = filtered[:, 3]
        else:
            self.prepared_quantities[self.QUANTITY_NAME] = np.array([])

    def _do_register(self):
        """Register only the point cloud geometry (XYZ coordinates)."""
        logger.debug(
            f"Registering point cloud geometry (threshold [{self.visualize_threshold_min:.3f}, {self.visualize_threshold_max:.3f}]): '{self.name}'"
        )

        filtered = self.get_filtered_points()
        if filtered.shape[0] == 0:
            logger.debug("No points left after threshold filtering")
            return

        p = ps.register_point_cloud(self.name, filtered[:, :3])
        p.set_radius(self._points_radius, relative=False)
        p.set_point_render_mode(self._points_render_mode)

        # Register scalar values
        self.prepare_quantities()
        if len(self.prepared_quantities[self.QUANTITY_NAME]) > 0:
            p.add_scalar_quantity(
                self.QUANTITY_NAME,
                self.prepared_quantities[self.QUANTITY_NAME],
                enabled=True,
            )

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

            self._draw_basic_controls()

            if self.is_valid():
                self._draw_threshold_controls()

            psim.Separator()

    def _draw_basic_controls(self):
        with ui_item_width(100):
            thresh_min_changed, show = psim.Checkbox("Show", self.enabled)
            if thresh_min_changed:
                self.set_enabled(show)

            psim.SameLine()

            radius_changed, radius = psim.SliderFloat(
                "Radius",
                self._points_radius,
                v_min=self.app_context.points_densest_distance * 0.01,
                v_max=self.app_context.points_densest_distance * 0.20,
                format="%.4g",
            )
            if radius_changed:
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

    def _draw_threshold_controls(self):
        with ui_item_width(100):
            q_values = self.points[:, 3]
            min_val, max_val = np.float64(q_values.min()), np.float64(q_values.max())

            min_changed, new_min = psim.DragFloat(
                "Threshold Min",
                self.visualize_threshold_min,
                v_speed=(max_val - min_val) / 10000.0,
                v_min=min_val,
                v_max=self.visualize_threshold_max,
                format="%.4g",
            )
            if min_changed:
                self.visualize_threshold_min = new_min

            psim.SameLine()

            max_changed, new_max = psim.DragFloat(
                "Threshold Max",
                self.visualize_threshold_max,
                v_speed=(max_val - min_val) / 10000.0,
                v_min=self.visualize_threshold_min,
                v_max=max_val,
                format="%.4g",
            )
            if max_changed:
                self.visualize_threshold_max = new_max

            if min_changed or max_changed:
                self.register(force=True)
                self.update_all_quantities_colormap()

    def callback(self):
        pass
