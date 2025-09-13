DATA_PREPROCESS_METHODS = ("identity", "log_e", "log_10")
DENOISE_METHODS = ("gaussian", "nearest_and_gaussian", "nearest")

_CMAPS = ["rainbow", "viridis", "coolwarm", "jet", "turbo", "magma"]


def get_cmaps() -> list[str]:
    global _CMAPS
    return _CMAPS


def set_cmaps(new_cmaps: list[str]) -> None:
    global _CMAPS
    _CMAPS = new_cmaps
