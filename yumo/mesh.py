import functools
import logging
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import polyscope as ps
from polyscope import imgui as psim

from yumo.base_structure import Structure
from yumo.constants import DENOISE_METHODS
from yumo.context import Context
from yumo.geometry_utils import (
    bake_to_texture,
    denoise_texture,
    map_to_uv,
    query_scalar_field,
    sample_surface,
    triangle_areas,
    unwrap_uv,
    uv_mask,
)
from yumo.ui import ui_combo, ui_item_width, ui_tree_node

logger = logging.getLogger(__name__)


class MeshStructure(Structure):
    """Represents a surface mesh structure."""

    QUANTITY_NAME: ClassVar[str] = "mesh_texture_values"

    def __init__(self, name: str, app_context: "Context", vertices: np.ndarray, faces: np.ndarray, **kwargs):
        super().__init__(name, app_context, **kwargs)
        self.vertices = vertices
        self.faces = faces

        self.points_per_area = 1000

        # texture related
        self.param_corner: np.ndarray = None  # type: ignore[assignment]
        self.texture_height: int = None  # type: ignore[assignment]
        self.texture_width: int = None  # type: ignore[assignment]
        self.vmapping: np.ndarray = None  # type: ignore[assignment]
        self.faces_unwrapped: np.ndarray = None  # type: ignore[assignment]
        self.uvs: np.ndarray = None  # type: ignore[assignment]
        self.vertices_unwrapped: np.ndarray = None  # type: ignore[assignment]

        # mask indicating which parts of the texture atlas are occupied by the mesh (1) and which are empty (0).
        self.uv_mask: np.ndarray = None  # type: ignore[assignment]

        self.raw_texture: np.ndarray = None  # type: ignore[assignment]

        self.inv_vmapping: np.ndarray = None  # type: ignore[assignment]

        self.mesh_surface_area = triangle_areas(self.vertices[self.faces]).sum()

        self._need_update = False
        self._need_rebake = True
        self._display_mode = "preview"  # one of "preview", "baked"

        self._enable_denoise = True
        self._denoise_method = DENOISE_METHODS[0]  # one of DENOISE_METHODS
        self._denoise_kwargs = {}  # type: ignore[var-annotated]

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

    def bake_texture(
        self,
        sampler_func: Callable[[np.ndarray], np.ndarray],
    ):
        """
        Args:
            sampler_func: Takes in sample query points (N, 3), outputs values (N,)

        Returns:

        """
        # -- 1. Sample surface --
        points, bary, indices = sample_surface(
            self.vertices_unwrapped,
            self.faces_unwrapped,
            points_per_area=self.points_per_area,
        )

        # -- 2. Map samples to UV space --
        sample_uvs = map_to_uv(self.uvs, self.faces_unwrapped, bary, indices)

        # -- 3. Sample scalar field --
        values = sampler_func(points)

        # -- 4. Bake to texture --
        tex = bake_to_texture(sample_uvs, values, self.texture_height, self.texture_width)

        return tex

    def update_resolution_preview(self):
        """Add a preview quantity of the surface sampling resolution using bake_texture"""
        self.raw_texture = self.bake_texture(
            sampler_func=lambda p: np.ones(p.shape[0]) * self.app_context.color_max,  # 0 (no fill); color_max (filled)
        )
        tex = self.raw_texture
        tex = tex * self.uv_mask
        self.prepared_quantities[self.QUANTITY_NAME] = tex

    def update_data_texture(self):
        """Sample the data points and update texture map"""
        if self._need_rebake:
            logger.debug(f"Rebaking data texture for structure: '{self.name}'")
            self._need_rebake = False
            self.raw_texture = self.bake_texture(
                sampler_func=functools.partial(query_scalar_field, data_points=self.app_context.points),
            )

        if self._enable_denoise:
            self._denoise_kwargs["mask"] = self.uv_mask
            tex = denoise_texture(self.raw_texture, method=self._denoise_method, **self._denoise_kwargs)
        else:
            tex = self.raw_texture.copy()

        tex = tex * self.uv_mask  # mask out unsampled areas
        self.prepared_quantities[self.QUANTITY_NAME] = tex

    def update_texture(self):
        if self._display_mode == "baked":
            self.update_data_texture()
        elif self._display_mode == "preview":
            self.update_resolution_preview()
        else:
            raise ValueError(f"Invalid display mode: {self._display_mode}")

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

        mesh = ps.register_surface_mesh(self.name, self.vertices_unwrapped, self.faces_unwrapped)
        mesh.set_material("clay")
        mesh.set_color([0.7, 0.7, 0.7])
        mesh.set_selection_mode(
            "faces_only"
        )  # only allow face selection (as the uv coord for face selection yields higher precision)

        self.uv_mask = uv_mask(
            uvs=self.uvs,
            faces_unwrapped=self.faces_unwrapped,
            texture_width=self.texture_width,
            texture_height=self.texture_height,
        )

        mesh.add_parameterization_quantity("uv", self.param_corner, defined_on="corners", enabled=True)

    def _ui_texture_map_display(self):
        """Add texture map image"""
        tex = self.prepared_quantities[self.QUANTITY_NAME]
        ps.add_scalar_image_quantity(
            "texture_map",
            tex,
            vminmax=(self.app_context.color_min, self.app_context.color_max),
            cmap=self.app_context.cmap,
            image_origin="upper_left",
            show_in_imgui_window=True,
            enabled=True,
        )

        ps.add_scalar_image_quantity(
            "raw_texture",
            self.raw_texture,
            vminmax=(self.app_context.color_min, self.app_context.color_max),
            cmap=self.app_context.cmap,
            image_origin="upper_left",
            show_in_imgui_window=True,
            enabled=True,
        )

    def _ui_texture_denoise_method(self):
        """Texture denoise method selection"""
        changed, denoise_enabled = psim.Checkbox("Enable Denoise", self._enable_denoise)
        if changed:
            self._enable_denoise = denoise_enabled
            self._need_update = self._display_mode == "baked"

        if self._enable_denoise:
            psim.SameLine()

            with ui_combo("Denoise Method", self._denoise_method) as expanded:
                if expanded:
                    for method in DENOISE_METHODS:
                        selected, _ = psim.Selectable(method, method == self._denoise_method)
                        if selected and method != self._denoise_method:
                            self._denoise_method = method
                            self._need_update = self._display_mode == "baked"

            changed, max_dist = psim.DragFloat(
                "Max Dist for Nearest Neighbour", self._denoise_kwargs.get("max_dist", 16), v_min=1, v_max=16
            )
            if changed:
                self._denoise_kwargs["max_dist"] = max_dist
                self._need_update = self._display_mode == "baked"

            if self._denoise_method in ["nearest_and_gaussian", "gaussian"]:
                changed, sigma = psim.DragFloat("Sigma", self._denoise_kwargs.get("sigma", 1.0), v_min=0.0, v_max=16.0)
                if changed:
                    self._denoise_kwargs["sigma"] = sigma
                    self._need_update = self._display_mode == "baked"

    def ui(self):
        """Mesh related UI"""
        with ui_tree_node("Mesh", open_first_time=True) as expanded:
            if not expanded:
                return

            psim.Text(f"Mesh Surface Area: {self.mesh_surface_area:.2f}")
            psim.Text(f"Texture Width: {self.texture_width:.2f}")
            psim.SameLine()
            psim.Text(f"Texture Height: {self.texture_height:.2f}")

            with ui_item_width(120):
                changed, show = psim.Checkbox("Show", self.enabled)
                if changed:
                    self.set_enabled(show)

                psim.SameLine()

                changed, resolution = psim.DragFloat(
                    "Points / Unit Area", self.points_per_area, v_min=1.0, v_max=10000.0
                )
                if changed:
                    self.points_per_area = resolution
                    self._display_mode = "preview"
                    self._need_update = True

                psim.SameLine()

                if psim.Button("Bake"):
                    self._need_update = True
                    self._need_rebake = True
                    self._display_mode = "baked"

                self._ui_texture_denoise_method()

        psim.Separator()

        self._ui_texture_map_display()

    def callback(self):
        if self._need_update:
            self._need_update = False
            self.update_texture()  # update texture
            self.add_prepared_quantities()  # update polyscope quantity
