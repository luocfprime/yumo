import logging
import uuid
from typing import ClassVar

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from yumo.context import Context
from yumo.geometry_utils import generate_slice_mesh
from yumo.structures import Structure
from yumo.ui import ui_item_width, ui_tree_node

logger = logging.getLogger(__name__)


class Slice(Structure):
    QUANTITY_NAME: ClassVar[str] = "slice"

    def __init__(self, name: str, app_context: "Context", group: ps.Group, **kwargs):
        super().__init__(name, app_context, **kwargs)

        self.group = group
        self.app_context = app_context

        bbox_min, bbox_max = app_context.bbox_min, app_context.bbox_max

        self.diag = int(np.linalg.norm(bbox_max - bbox_min))

        self.width = self.diag
        self.height = self.diag

        self.resolution_w = self.width
        self.resolution_h = self.height

        self.transforms = None

        self._need_update_quant = False
        self._live = False
        self._should_destroy = False

    @property
    def polyscope_structure(self):
        return ps.get_surface_mesh(self.name)

    def _do_register(self):
        """Register the slice mesh"""
        logger.info(f"Registering slice mesh: '{self.name}'")
        vertices, faces = generate_slice_mesh(
            center=self.app_context.center,
            h=self.height,
            w=self.width,
            rh=self.resolution_h,
            rw=self.resolution_w,
        )
        p = ps.register_surface_mesh(self.name, vertices, faces)
        p.set_transparency(0.8)
        p.set_material("flat")
        p.set_transform_gizmo_enabled(True)
        p.add_to_group(self.group)

    def is_valid(self) -> bool:
        return not self._should_destroy

    def prepare_quantities(self):
        pass  # TODO

    def callback(self):
        need_update_quant = self._need_update_quant or self._live  # noqa: F841
        pass  # TODO: implement transform callback

    def ui(self):
        with ui_tree_node(f"Slice {self.name}") as expanded:
            if not expanded:
                return

            with ui_item_width(100):
                self._ui_visibility_controls()
                psim.SameLine()
                self._ui_action_buttons()
                psim.SameLine()
                self._ui_live_mode_checkbox()
                need_update_structure = self._ui_dimension_inputs()

            if need_update_structure:
                self.register(force=True)

        psim.Separator()

    def _ui_visibility_controls(self):
        changed, show = psim.Checkbox("Show", self.enabled)
        if changed:
            self.set_enabled(show)

    def _ui_action_buttons(self):
        if psim.Button("Destroy"):
            self._should_destroy = True
        psim.SameLine()

        self._need_update_quant = False
        if psim.Button("Compute"):
            self._need_update_quant = True

    def _ui_live_mode_checkbox(self):
        changed, live = psim.Checkbox("Live", self._live)
        if changed:
            self._live = live

    def _ui_dimension_inputs(self):
        v_speed = 1.0

        changed_h, new_h = psim.DragInt("Height", self.height, v_speed, 10, self.diag)
        psim.SameLine()
        changed_w, new_w = psim.DragInt("Width", self.width, v_speed, 10, self.diag)

        changed_rh, new_rh = psim.DragInt("Resolution H", self.resolution_h, v_speed, 10, 99999999)
        psim.SameLine()
        changed_rw, new_rw = psim.DragInt("Resolution W", self.resolution_w, v_speed, 10, 99999999)

        if changed_h:
            self.height = new_h
        if changed_w:
            self.width = new_w
        if changed_rh:
            self.resolution_h = new_rh
        if changed_rw:
            self.resolution_w = new_rw

        return changed_h or changed_w or changed_rh or changed_rw


class Slices:
    def __init__(self, name: str, app_context: "Context", enabled: bool = True):
        self.name = name
        self.app_context = app_context
        self.enabled = enabled

        self.group = None
        self.slices = {}

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
