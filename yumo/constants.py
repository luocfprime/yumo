from pathlib import Path

ASSETS_ROOT = Path(__file__).parent / "assets"

DATA_PREPROCESS_METHODS = ("identity", "log_e", "log_10")
DENOISE_METHODS = ("gaussian", "nearest_and_gaussian", "nearest")

_CMAPS = ["rainbow", "viridis", "coolwarm", "jet", "turbo", "magma"]
_MATERIALS = ["clay", "wax", "candy", "flat", "mud", "ceramic", "jade", "normal"]


def get_cmaps() -> list[str]:
    global _CMAPS
    return _CMAPS


def set_cmaps(new_cmaps: list[str]) -> None:
    global _CMAPS
    _CMAPS = new_cmaps


def get_materials() -> list[str]:
    global _MATERIALS
    return _MATERIALS


def set_materials(new_materials: list[str]) -> None:
    global _MATERIALS
    _MATERIALS = new_materials
