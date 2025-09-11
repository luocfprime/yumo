import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from yumo.base_structure import Structure
from yumo.constants import CMAPS
from yumo.context import Context
from yumo.mesh import MeshStructure
from yumo.point_cloud import PointCloudStructure
from yumo.slices import Slices
from yumo.ui import ui_combo, ui_item_width, ui_tree_node
from yumo.utils import estimate_densest_point_distance, generate_colorbar_image, load_mesh, parse_plt_file

logger = logging.getLogger(__name__)


# --- Configs ---
@dataclass
class Config:
    data_path: Path
    mesh_path: Path | None
    sample_rate: float
    skip_zeros: bool


# --- Main Application ---
class PolyscopeApp:
    def __init__(self, config: Config):
        self.config = config
        self.context = Context()
        self.slices = Slices("slices", self.context)
        self.structures: dict[str, Structure] = {}
        self._should_init_quantities = True

        self.prepare_data_and_init_structures()

    def prepare_data_and_init_structures(self):
        """Load data from files, create structures."""
        # 1. Load raw data
        logger.info(f"Loading data from {self.config.data_path}")
        points = parse_plt_file(self.config.data_path, skip_zeros=self.config.skip_zeros)
        if self.config.sample_rate < 1.0:
            logger.info(
                f"Downsampling points from {points.shape[0]:,} to {int(points.shape[0] * self.config.sample_rate):,}"
            )
            indices = np.random.choice(
                points.shape[0], size=int(points.shape[0] * self.config.sample_rate), replace=False
            )
            points = points[indices]

        self.context.points = points

        self.context.center = np.mean(points[:, :3], axis=0)
        self.context.bbox_min = np.min(points[:, :3], axis=0)
        self.context.bbox_max = np.max(points[:, :3], axis=0)

        self.context.points_densest_distance = estimate_densest_point_distance(
            points[:, :3],
            k=5000,  # TODO: hard-coded
            quantile=0.01,
        )

        if self.config.mesh_path:
            logger.info(f"Loading mesh from {self.config.mesh_path}")
            self.context.mesh_vertices, self.context.mesh_faces = load_mesh(str(self.config.mesh_path))  # type: ignore[misc]

        # 2. Calculate statistics and set initial context
        self.context.min_value = np.min(points[:, 3])
        self.context.max_value = np.max(points[:, 3])
        self.context.color_min = self.context.min_value
        self.context.color_max = self.context.max_value

        # 3. Instantiate structures
        self.structures["points"] = PointCloudStructure("points", self.context, self.context.points, enabled=True)

        if self.context.mesh_vertices is not None and self.context.mesh_faces is not None:
            self.structures["mesh"] = MeshStructure(
                "mesh", self.context, self.context.mesh_vertices, self.context.mesh_faces, enabled=True
            )

    def update_all_scalar_quantities_colormap(self):
        """Update colormaps on all structures (including slices)."""
        for structure in self.structures.values():
            structure.update_all_quantities_colormap()

        self.slices.update_all_quantities_colormap()

    # --- UI Methods ---

    def _ui_top_text_brief(self):
        """A top text bar showing brief"""
        with ui_tree_node("Brief", open_first_time=True) as expanded:
            if not expanded:
                return
            psim.Text(f"Data: {self.config.data_path}")
            psim.Text(f"Mesh: {self.config.mesh_path if self.config.mesh_path else 'N/A'}")

            if self.config.mesh_path:  # the mesh should be loaded if the path is provided
                psim.Text(
                    f"Mesh vertices: {len(self.context.mesh_vertices):,}, faces: {len(self.context.mesh_faces):,}"  # type: ignore[arg-type]
                )

            psim.Text(f"Points: {self.context.points.shape[0]:,}")
            psim.SameLine()
            psim.Text(f"Points densest distance: {self.context.points_densest_distance:.4g}")

            psim.Text(
                f"Points center: ({self.context.center[0]:.2f},{self.context.center[1]:.2f},{self.context.center[2]:.2f})"
            )
            psim.Text(
                f"Bbox min: ({self.context.bbox_min[0]:.2f},{self.context.bbox_min[1]:.2f},{self.context.bbox_min[2]:.2f})"
            )
            psim.Text(
                f"Bbox max: ({self.context.bbox_max[0]:.2f},{self.context.bbox_max[1]:.2f},{self.context.bbox_max[2]:.2f})"
            )

            psim.Text(f"Data range: [{self.context.min_value:.2g}, {self.context.max_value:.2g}]")

        psim.Separator()

    def _ui_colorbar_controls(self):
        """Colorbar controls UI"""
        with ui_tree_node("Colormap Controls") as expanded:
            if not expanded:
                return

            needs_update = False

            # Colormap selection using the yumo helper
            with ui_combo("Colormap", self.context.cmap) as combo_expanded:
                if combo_expanded:
                    for cmap_name in CMAPS:
                        selected, _ = psim.Selectable(cmap_name, self.context.cmap == cmap_name)
                        if selected and cmap_name != self.context.cmap:
                            self.context.cmap = cmap_name
                            needs_update = True
                            logger.debug(f"Selected colormap: {cmap_name}")

            data_range = self.context.max_value - self.context.min_value
            v_speed = data_range / 1000.0 if data_range > 0 else 0.01

            with ui_item_width(100):
                # Min/Max value controls
                changed_min, new_min = psim.DragFloat(
                    "Min Value", self.context.color_min, v_speed, self.context.min_value, self.context.max_value, "%.4g"
                )

                psim.SameLine()

                if changed_min:
                    self.context.color_min = new_min
                    needs_update = True

                changed_max, new_max = psim.DragFloat(
                    "Max Value", self.context.color_max, v_speed, self.context.min_value, self.context.max_value, "%.4g"
                )

                psim.SameLine()

                if changed_max:
                    self.context.color_max = new_max
                    needs_update = True

                self.context.color_max = max(self.context.color_min, self.context.color_max)

                if psim.Button("Reset Range"):
                    self.context.color_min = self.context.min_value
                    self.context.color_max = self.context.max_value
                    needs_update = True

                if needs_update:
                    self.update_all_scalar_quantities_colormap()

        psim.Separator()

    def _ui_colorbar_display(self):
        """Add colorbar image"""
        colorbar_img = generate_colorbar_image(
            colorbar_height=300,
            colorbar_width=120,
            cmap=self.context.cmap,
            c_min=self.context.color_min,
            c_max=self.context.color_max,
        )
        ps.add_color_image_quantity(
            "colorbar",
            colorbar_img,
            image_origin="upper_left",
            show_in_imgui_window=True,
            enabled=True,
        )

    # --- Main Loop ---

    def callback(self) -> None:
        """The main callback loop for Polyscope."""
        # Phase 1: Register Geometries (runs only once internally per structure)
        for structure in self.structures.values():
            structure.register()

        # Phase 2: Add Scalar Quantities (runs only once via the flag)
        if self._should_init_quantities:
            # Prepare quantities (expensive, one-time calculations)
            for structure in self.structures.values():
                structure.prepare_quantities()

            # Add quantities to Polyscope structures
            for structure in self.structures.values():
                structure.add_prepared_quantities()
            self._should_init_quantities = False  # Prevent this from running again

        # Other callbacks
        for structure in self.structures.values():
            structure.callback()

        self.slices.callback()

        # Build the UI
        self._ui_top_text_brief()
        self._ui_colorbar_controls()
        self._ui_colorbar_display()

        for structure in self.structures.values():
            structure.ui()

        self.slices.ui()

    def run(self):
        """Initialize and run the Polyscope application."""
        ps.set_program_name("Yumo")
        ps.set_print_prefix("[Yumo][Polyscope] ")

        ps.init()
        ps.set_user_callback(self.callback)
        ps.show()
