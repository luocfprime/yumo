import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

import einx
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from yumo.base_structure import Structure
from yumo.constants import CMAPS
from yumo.context import Context
from yumo.geometry_utils import map_to_uv, query_scalar_field
from yumo.mesh import MeshStructure
from yumo.point_cloud import PointCloudStructure
from yumo.slices import Slices
from yumo.ui import ui_combo, ui_item_width, ui_tree_node
from yumo.utils import (
    data_transform,
    estimate_densest_point_distance,
    export_camera_view,
    fmt2,
    fmt3,
    generate_colorbar_image,
    load_camera_view,
    load_mesh,
    parse_plt_file,
)

logger = logging.getLogger(__name__)


# --- Configs ---
@dataclass
class Config:
    data_path: Path
    mesh_path: Path | None
    camera_view_path: Path | None
    sample_rate: float
    skip_zeros: bool
    data_preprocess_method: str


# --- Main Application ---
class PolyscopeApp:
    def __init__(self, config: Config):
        self.config = config
        self.context = Context(data_preprocess_method=config.data_preprocess_method)
        self.slices = Slices("slices", self.context)
        self.structures: dict[str, Structure] = {}
        self._should_init_quantities = True

        self._view_controls_msgs: str = ""

        self._picker_should_query_field = False
        self._picker_msgs: list[str] = []

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

        points = data_transform(points, method=self.config.data_preprocess_method)

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
        self.structures["points"] = PointCloudStructure("points", self.context, self.context.points, enabled=False)

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
                    f"Mesh vertices: {len(self.context.mesh_vertices):,}, "  # type: ignore[arg-type]
                    f"faces: {len(self.context.mesh_faces):,}"  # type: ignore[arg-type]
                )

            psim.Text(f"Points: {self.context.points.shape[0]:,}")
            psim.SameLine()
            psim.Text(f"Points densest distance: {self.context.points_densest_distance:.4g}")

            psim.Text(f"Points center: {fmt3(self.context.center)}")
            psim.Text(f"Bbox min: {fmt3(self.context.bbox_min)}")
            psim.Text(f"Bbox max: {fmt3(self.context.bbox_max)}")

            psim.Text(f"Data preprocess: {self.context.data_preprocess_method}")
            psim.Text(
                f"{'Data' if self.context.data_preprocess_method == 'identity' else 'Preprocessed data'} range: "
                f"[{self.context.min_value:.2g}, {self.context.max_value:.2g}]"
            )

        psim.Separator()

    def _ui_view_controls(self):
        with ui_tree_node("View Controls", open_first_time=False) as expanded:
            if not expanded:
                return

            if psim.Button("Reset Camera View"):
                if self.context.default_view_mat is None:
                    msg = "Default camera view matrix is not set. Please set it first."
                    logger.warning(msg)
                    ps.warning(msg)
                else:
                    ps.set_camera_view_matrix(self.context.default_view_mat)

            psim.SameLine()

            if psim.Button("Export Camera View"):
                view_mat = ps.get_camera_view_matrix()
                camera_view_file = Path(datetime.datetime.now().strftime("%Y%m%d_%H%M%S_cam_view.json"))
                with open(camera_view_file, "w") as f:
                    f.write(export_camera_view(view_mat))

                msg = f"Camera view exported to \n{camera_view_file.absolute()}"
                logger.info(msg)
                self._view_controls_msgs = msg

            psim.Text(self._view_controls_msgs)

    def _ui_coord_picker(self):
        with ui_tree_node("Coord Picker", open_first_time=True) as expanded:
            if not expanded:
                return

        changed, query_field = psim.Checkbox("Query Field", self._picker_should_query_field)
        if changed:
            self._picker_should_query_field = query_field

        io = psim.GetIO()
        if io.MouseClicked[0]:  # left click
            self._picker_msgs = []

            screen_coords = io.MousePos
            world_coords = ps.screen_coords_to_world_position(screen_coords)

            msg = f"World coord picked: {world_coords}"
            logger.debug(msg)
            self._picker_msgs.append(msg)

            pick_result: ps.PickResult = ps.pick(screen_coords=screen_coords)  # type: ignore[no-any-unimported]

            if pick_result.is_hit:
                logger.debug(pick_result.structure_data)
                self._picker_msgs.append(f"Picked {pick_result.structure_name}: {pick_result.structure_data}")

                field_value = None
                if self._picker_should_query_field:
                    field_value = query_scalar_field(world_coords, self.context.points)
                    msg = f"Field value: {field_value:.4g}"
                    logger.debug(msg)
                    self._picker_msgs.append(msg)

                if pick_result.structure_name == "mesh":
                    texture_value = self._handle_mesh_pick(pick_result)
                    if texture_value and field_value:
                        rel_err = abs((texture_value - field_value) / field_value)
                        msg = f"Relative error: {rel_err * 100:,.2f}%"
                        logger.debug(msg)
                        self._picker_msgs.append(msg)

        for msg in self._picker_msgs:
            psim.Text(msg)

        psim.Separator()

    def _handle_mesh_pick(self, pick_result: ps.PickResult) -> np.float64 | None:  # type: ignore[no-any-unimported]
        """Handle mesh picking cases: face, vertex, corner."""
        logger.debug("Picked mesh")
        mesh: MeshStructure = self.structures["mesh"]  # type: ignore[assignment]

        data = pick_result.structure_data
        if "bary_coords" in data:  # ---- face hit
            barycentric_coord = data["bary_coords"].reshape(1, 3)
            indices = np.array([data["index"]], dtype=int)
            uv_coords = map_to_uv(
                uvs=mesh.uvs,
                faces_unwrapped=mesh.faces_unwrapped,
                barycentric_coord=barycentric_coord,
                indices=indices,
            )
        else:
            logger.warning(f"Unknown pick result: {data}, skipping")
            return None

        # ---- sample texture map
        texture_map = mesh.prepared_quantities[mesh.QUANTITY_NAME]
        h, w = texture_map.shape[:2]

        u, v = uv_coords[0]
        j = int(np.clip(u * (w - 1), 0, w - 1))  # x axis
        i = int(np.clip((1.0 - v) * (h - 1), 0, h - 1))  # y axis with flip

        # --- draw cross overlay for visualization (10px thick)
        indicator = einx.rearrange("h w -> h w 3", mesh.uv_mask).copy()

        thickness = 10  # pixels half-width of the line

        # horizontal band (thickness in vertical axis)
        i_min = max(i - thickness // 2, 0)
        i_max = min(i + thickness // 2 + 1, h)
        indicator[i_min:i_max, :, :] = [1, 0, 0]

        # vertical band (thickness in horizontal axis)
        j_min = max(j - thickness // 2, 0)
        j_max = min(j + thickness // 2 + 1, w)
        indicator[:, j_min:j_max, :] = [1, 0, 0]

        ps.add_color_image_quantity("Picked Cross", indicator)

        logger.debug(f"uv: {fmt2([u, v])}, hw: {h, w}, raster index: {i, j}")

        texture_value = texture_map[i, j]
        msg = f"Texture value: {texture_value:.4g}"

        logger.debug(msg)
        self._picker_msgs.append(msg)

        return np.float64(texture_value)

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
            method=self.context.data_preprocess_method,
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
        self._ui_view_controls()
        self._ui_colorbar_controls()
        self._ui_colorbar_display()
        self._ui_coord_picker()

        for structure in self.structures.values():
            structure.ui()

        self.slices.ui()

    def run(self):
        """Initialize and run the Polyscope application."""
        ps.set_program_name("Yumo")
        ps.set_print_prefix("[Yumo][Polyscope] ")
        ps.set_ground_plane_mode("shadow_only")
        ps.set_up_dir("z_up")
        ps.set_front_dir("x_front")

        ps.init()

        if self.config.camera_view_path:
            logger.info(f"Loading initial camera view from: {self.config.camera_view_path}")
            with open(self.config.camera_view_path) as f:
                lines = f.readlines()
            json_str = "".join(lines)
            view_mat = load_camera_view(json_str)
            ps.set_camera_view_matrix(view_mat)

            self.context.default_view_mat = view_mat

        ps.set_user_callback(self.callback)
        ps.show()
