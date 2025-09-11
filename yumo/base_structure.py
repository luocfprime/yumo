import logging
from abc import ABC, abstractmethod

import numpy as np

from yumo.context import Context

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
