from collections import defaultdict
from dataclasses import dataclass, field

import polyscope as ps


@dataclass
class Context:
    slices_names: set = field(default_factory=set)
    visualize_slices: dict[str, bool] = field(default_factory=lambda: defaultdict(lambda: True))
    should_refresh: dict[str, bool] = field(default_factory=lambda: defaultdict(lambda: True))


class Slices:
    def __init__(self):
        self.context = Context()

    def add_slice(self, h: float, w: float): ...

    def remove_slice(self, name):
        ps.remove_surface_mesh(name, error_if_absent=False)
        self.context.slices_names.remove(name)
        self.context.visualize_slices.pop(name)
        self.context.should_refresh.pop(name)

    def callback(self) -> None: ...
