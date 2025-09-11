import logging
import uuid
from typing import ClassVar

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from yumo.base_structure import Structure
from yumo.context import Context
from yumo.geometry_utils import generate_slice_mesh, query_scalar_field
from yumo.ui import ui_item_width, ui_tree_node

logger = logging.getLogger(__name__)


class Slice(Structure):
    QUANTITY_NAME: ClassVar[str] = "slice"

    def __init__(self, name: str, app_context: "Context", group: ps.Group, **kwargs):  # type: ignore[no-any-unimported]
        super().__init__(name, app_context, **kwargs)

        self.group = group
        self.app_context = app_context

        self._vertices, self.faces = None, None

        bbox_min, bbox_max = app_context.bbox_min, app_context.bbox_max

        diag = np.linalg.norm(bbox_max - bbox_min)

        self.width: float = diag / 1.73  # type: ignore[assignment]
        self.height: float = diag / 1.73  # type: ignore[assignment]

        self.resolution_w: int = int(self.width)
        self.resolution_h: int = int(self.height)

        self.plane_transform: np.ndarray = None  # type: ignore[assignment]

        self._need_update_quant = False
        self._live = False
        self._gizmo_enabled = True
        self._should_destroy = False

    @property
    def polyscope_structure(self):
        return ps.get_surface_mesh(self.name)

    def _do_register(self):
        """Register the slice mesh"""
        logger.debug(f"Registering slice mesh: '{self.name}'")
        self._vertices, self.faces = generate_slice_mesh(  # type: ignore[assignment]
            center=np.zeros(3),
            h=self.height,
            w=self.width,
            rh=self.resolution_h,
            rw=self.resolution_w,
        )

        p = ps.register_surface_mesh(self.name, self._vertices, self.faces)
        p.set_transparency(0.8)
        p.set_material("flat")
        p.set_transform_gizmo_enabled(True)
        p.add_to_group(self.group)

        self.plane_transform = p.get_transform()  # shape: (4, 4), should be an eye matrix, as it is being initialized

    def is_valid(self) -> bool:
        return not self._should_destroy

    def prepare_quantities(self):
        """Prepare the scalar data from the 4th column of the points array."""
        if self.is_valid():
            self.prepared_quantities[self.QUANTITY_NAME] = query_scalar_field(
                points_coord=self.vertices_transformed,
                data_points=self.app_context.points,
            )

    def callback(self):
        current_transform = self.polyscope_structure.get_transform()
        if (self._live or self._need_update_quant) and not np.allclose(
            current_transform,
            self.plane_transform,
        ):  # plane moved and should refresh
            self.plane_transform = current_transform
            self._need_update_quant = True

        if self._need_update_quant:
            self.prepare_quantities()  # query the field
            self.add_prepared_quantities()  # update the quantity
            self._need_update_quant = False  # reset

    @property
    def vertices_transformed(self):
        # Transform vertices
        local_verts = self._vertices  # shape: (N, 3)
        n_verts = local_verts.shape[0]  # type: ignore[attr-defined]

        # Convert to homogeneous coordinates: (N, 4)
        homogeneous_verts = np.hstack([local_verts, np.ones((n_verts, 1))])  # type: ignore[list-item]

        # Apply transform (assuming 4x4 matrix)
        transformed_homogeneous = (self.plane_transform @ homogeneous_verts.T).T

        # Divide by w to get back to [x, y, z]
        vertices = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3][:, np.newaxis]

        return vertices

    def ui(self):
        with ui_tree_node(f"Slice {self.name}") as expanded:
            if not expanded:
                return

            with ui_item_width(100):
                self._ui_visibility_controls()
                psim.SameLine()
                self._ui_live_mode_checkbox()
                psim.SameLine()
                self._ui_gizmo_controls()
                psim.SameLine()
                self._ui_action_buttons()
                self._ui_transparency_controls()

            with ui_item_width(120):
                need_update_structure = self._ui_dimension_inputs()

            if need_update_structure:
                self.register(force=True)

        psim.Separator()

    def _ui_visibility_controls(self):
        changed, show = psim.Checkbox("Show", self.enabled)
        if changed:
            self.set_enabled(show)

    def _ui_gizmo_controls(self):
        changed, enable = psim.Checkbox("Gizmo", self._gizmo_enabled)
        if changed:
            self._gizmo_enabled = enable
            self.polyscope_structure.set_transform_gizmo_enabled(enable)

    def _ui_transparency_controls(self):
        changed, transparency = psim.SliderFloat("Transparency", self.polyscope_structure.get_transparency(), 0.0, 1.0)
        if changed:
            self.polyscope_structure.set_transparency(transparency)

    def _ui_action_buttons(self):
        if psim.Button("Destroy"):
            self._should_destroy = True
        psim.SameLine()

        if psim.Button("Compute"):
            self._need_update_quant = True

    def _ui_live_mode_checkbox(self):
        changed, live = psim.Checkbox("Live", self._live)
        if changed:
            self._live = live
            self._need_update_quant = live is True

    def _ui_dimension_inputs(self):
        step = 1

        changed_h, new_h = psim.InputFloat("Height", self.height, step)
        psim.SameLine()
        changed_w, new_w = psim.InputFloat("Width", self.width, step)

        changed_rh, new_rh = psim.InputInt("Resolution H", self.resolution_h, step)
        psim.SameLine()
        changed_rw, new_rw = psim.InputInt("Resolution W", self.resolution_w, step)

        if changed_h:
            self.height = max(0.1, new_h)
        if changed_w:
            self.width = max(0.1, new_w)
        if changed_rh:
            self.resolution_h = max(4, new_rh)
        if changed_rw:
            self.resolution_w = max(4, new_rw)

        return changed_h or changed_w or changed_rh or changed_rw


class Slices:
    def __init__(self, name: str, app_context: "Context", enabled: bool = True):
        self.name = name
        self.app_context = app_context
        self.enabled = enabled

        self.group = None
        self.slices = {}  # type: ignore[var-annotated]

    def add_slice(self):
        name = None
        while name is None or name in self.slices:
            name = f"slice_{uuid.uuid4().hex[:4]}"  # use a short uuid (4 chars) as suffix

        s = Slice(name, self.app_context, self.group)
        self.slices[name] = s
        s.register()

    def remove_slice(self, name: str):
        ps.remove_surface_mesh(name, error_if_absent=False)
        self.slices.pop(name)

    def update_all_quantities_colormap(self):
        for slc in self.slices.values():
            slc.update_all_quantities_colormap()

    def callback(self) -> None:
        if self.group is None:
            self.group = ps.create_group("slices")

        to_be_removed = []
        for name, slc in self.slices.items():
            if slc.is_valid():
                slc.callback()
            else:
                to_be_removed.append(name)

        for name in to_be_removed:
            self.remove_slice(name)

    def ui(self):
        with ui_tree_node("Slices") as expanded:
            if not expanded:
                return
            if psim.Button("Add Slice"):
                self.add_slice()
            for slc in self.slices.values():
                slc.ui()
