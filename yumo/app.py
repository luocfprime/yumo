import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import potpourri3d as pp3d

from yumo.slices import Slices
from yumo.ui import ui_combo, ui_tree_node
from yumo.utils import generate_colorbar_image, parse_plt_file

logger = logging.getLogger(__name__)

# Constants
CMAPS = ("rainbow", "viridis", "coolwarm", "jet", "turbo", "magma")

# - structures
NAME_MESH = "mesh"
NAME_POINTS = "points"

# - quantities
NAME_MESH_QUANT = "mesh_values"
NAME_POINTS_QUANT = "points_values"

STRUCTURES = (NAME_MESH, NAME_POINTS)
STRUCTURES_WITH_QUANTITIES = (NAME_MESH, NAME_POINTS)


# Configs and State
@dataclass
class Config:
    plt_path: Path
    mesh_path: Path | None

    sample_rate: float


@dataclass
class Context:
    # Data
    points: np.ndarray = None
    mesh_vertices: np.ndarray | None = None
    mesh_faces: np.ndarray | None = None

    # Statistics
    min_value: float = None
    max_value: float = None

    bbox_min: np.ndarray = None
    bbox_max: np.ndarray = None

    # Settings
    visualize_points: bool = True
    visualize_mesh: bool = True
    visualize_slices: bool = True

    cmap: str = CMAPS[0]

    color_min: float = None
    color_max: float = None

    points_size: float = 0.005
    points_render_mode: str = "sphere"  # one of "sphere" or "quad", the latter is less costly

    should_register_structure: dict[str, bool] = field(default_factory=lambda: defaultdict(lambda: True))
    should_add_scalar_quantity: dict[str, bool] = field(default_factory=lambda: defaultdict(lambda: True))


# App
class PolyscopeApp:
    def __init__(self, config: Config):
        self.config = config
        self.context = Context()

        self.slices = Slices()

        self.load_data()

    def load_data(self):
        """Load plt file and mesh"""
        points = parse_plt_file(self.config.plt_path)
        if self.config.sample_rate < 1.0:  # downsample if necessary
            logger.info(
                f"Downsampling points from {points.shape[0]:,} to {int(points.shape[0] * self.config.sample_rate):,}"
            )
            indices = np.arange(0, points.shape[0])
            indices = np.random.choice(indices, size=int(points.shape[0] * self.config.sample_rate), replace=False)
            points = points[indices]

        self.context.points = points

        if self.config.mesh_path:
            logger.info(f"Loading mesh from {self.config.mesh_path}")
            self.context.mesh_vertices, self.context.mesh_faces = pp3d.read_polygon_mesh(str(self.config.mesh_path))

        self.context.min_value = np.min(self.context.points)
        self.context.max_value = np.max(self.context.points)

        # initialize color range
        self.context.color_min = self.context.min_value
        self.context.color_max = self.context.max_value

        self.context.bbox_min = np.min(self.context.points[:, :3], axis=0)
        self.context.bbox_max = np.max(self.context.points[:, :3], axis=0)

    def register(self, name):
        if name == NAME_MESH:
            self._register_mesh()
        elif name == NAME_POINTS:
            self._register_points()
        else:
            raise ValueError(f"Unknown structure name: {name}")

    def add_scalar_quantity(self, name):
        if name == NAME_POINTS:
            self._add_scalar_quantity_points()
        elif name == NAME_MESH:
            self._add_scalar_quantity_mesh()
        else:
            raise ValueError(f"Unknown structure name: {name}")

    def _register_mesh(self):
        if not self._mesh_loaded:
            return
        ps.register_surface_mesh(NAME_MESH, self.context.mesh_vertices, self.context.mesh_faces)
        mesh = ps.get_surface_mesh(NAME_MESH)
        mesh.set_material("clay")
        mesh.set_color([0.7, 0.7, 0.7])

        # TODO: check if mesh.set_enabled(self.options["show_mesh"])
        # TODO: check if upsampling mesh is needed

    def _add_scalar_quantity_mesh(self):
        if not self._mesh_loaded:
            return
        self.mesh_structure.add_scalar_quantity(...)

    def _register_points(self):
        p = ps.register_point_cloud(NAME_POINTS, self.context.points[:, :3])  # xyz
        p.set_radius(self.context.points_size, relative=False)
        p.set_point_render_mode(self.context.points_render_mode)
        p.set_enabled(self.context.visualize_points)

        self._add_scalar_quantity_points(p)

    def _add_scalar_quantity_points(self):
        self.points_structure.add_scalar_quantity(
            NAME_POINTS_QUANT,
            self.context.points[:, 3],
            enabled=True,
            cmap=self.context.cmap,
            vminmax=(self.context.color_min, self.context.color_max),
        )

    @property
    def points_structure(self):
        return ps.get_point_cloud(NAME_POINTS)

    @property
    def mesh_structure(self):
        return ps.get_surface_mesh(NAME_MESH) if self._mesh_loaded else None

    @property
    def _mesh_loaded(self):
        return self.context.mesh_vertices is not None and self.context.mesh_faces is not None

    def _maybe_register_structure(self, name):
        """Register the structure once."""
        register_funcs = {
            NAME_MESH: self._register_mesh,
            NAME_POINTS: self._register_points,
        }
        if self.context.should_register_structure[name]:
            register_funcs[name]()
            self.context.should_register_structure[name] = False  # refresh only once

    def _maybe_add_scalar_quantity(self, name):
        """Add scalar quantity to the structure once."""
        add_scalar_quantity_funcs = {
            NAME_MESH: self._add_scalar_quantity_mesh,
            NAME_POINTS: self._add_scalar_quantity_points,
        }
        if self.context.should_add_scalar_quantity[name]:
            add_scalar_quantity_funcs[name]()
            self.context.should_add_scalar_quantity[name] = False  # refresh only once

    def _ui_top_text_brief(self):
        """A top text bar showing brief"""
        with ui_tree_node("Brief", open_first_time=True) as expanded:
            if expanded:
                psim.Text(f"Data: {self.config.plt_path}")
                psim.Text(f"Mesh: {self.config.mesh_path}")
                psim.Text(
                    f"Mesh vertices: {len(self.context.mesh_vertices):,}, faces: {len(self.context.mesh_faces):,}"
                )
                psim.Text(f"Points: {self.context.points.shape[0]:,}")
                psim.Text(f"Data range: [{self.context.min_value:.4g}, {self.context.max_value:.4g}]")
                psim.Separator()

    def _ui_colorbar_controls(self):
        """Colorbar controls UI"""
        with ui_tree_node("Colormap Controls") as expanded:
            if not expanded:
                return

            needs_update = False

            # Colormap selection
            with ui_combo("Colormap", CMAPS) as combo_expanded:
                if combo_expanded:
                    for cmap_name in CMAPS:
                        selected, _ = psim.Selectable(cmap_name, self.context.cmap == cmap_name)
                        if selected and cmap_name != self.context.cmap:
                            self.context.cmap = cmap_name
                            needs_update = True

            # Calculate appropriate speed for dragging values
            data_range = self.context.max_value - self.context.min_value
            v_speed = data_range / 1000.0

            # Min value control
            changed_min, new_min = psim.DragFloat(
                "Min Value",
                self.context.color_min,
                v_speed=v_speed,
                v_min=self.context.min_value,
                v_max=self.context.max_value,
                format="%.4g",
            )
            if changed_min:
                self.context.color_min = new_min
                needs_update = True

            # Max value control
            changed_max, new_max = psim.DragFloat(
                "Max Value",
                self.context.max_value,
                v_speed=v_speed,
                v_min=self.context.min_value,
                v_max=self.context.max_value,
                format="%.4g",
            )
            if changed_max:
                self.context.color_max = new_max
                needs_update = True

            # Ensure max is always >= min
            self.context.color_max = max(self.context.color_min, self.context.color_max)

            # Reset range button
            if psim.Button("Reset Range"):
                self.context.color_min = self.value_min
                self.context.color_max = self.value_max
                needs_update = True

            # Update colorbar if needed
            if needs_update:
                self.update_all_scalar_quantities()

                colorbar_img = generate_colorbar_image(
                    self.colorbar_height,
                    self.colorbar_width,
                    self.context.cmap,
                    self.context.color_min,
                    self.context.color_max,
                )
                self.colorbar_img = colorbar_img

                ps.add_color_image_quantity(
                    "colorbar",
                    self.colorbar_img,
                    image_origin="upper_left",
                    show_in_imgui_window=True,
                    enabled=True,
                )

    def _ui_colorbar_display(self): ...

    def _ui_points(self):
        """Points related UI"""
        with ui_tree_node("Points", open_first_time=True) as expanded:
            if expanded:
                changed, show = psim.Checkbox("Show", self.context.visualize_points)
                if changed:
                    self.context.visualize_points = show
                    self.points_structure.set_enabled(show)

                psim.SameLine()
                changed, radius = psim.SliderFloat(
                    "Radius", self.context.points_size, v_min=1e-4, v_max=5e-2, format="%.4g"
                )
                if changed:
                    self.context.points_size = radius
                    self.points_structure.set_radius(radius, relative=False)  # update the point size

    def callback(self) -> None:
        # register structures
        for name in STRUCTURES:
            self._maybe_register_structure(name)
            self._maybe_add_scalar_quantity(name)

        if self._mesh_loaded:
            self._maybe_register_structure(NAME_MESH, self._register_mesh)

        # callback
        self.slices.callback()

        # ui
        self._ui_top_text_brief()
        self._ui_points()

    def run(self):
        ps.init()
        ps.set_user_callback(self.callback)
        ps.show()
