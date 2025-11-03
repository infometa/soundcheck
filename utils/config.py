from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"


@lru_cache(maxsize=1)
def _load_raw_config() -> dict:
    """Load and cache the raw YAML configuration."""
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a mapping, got {type(data).__name__}")
    return data


def load_config() -> dict:
    """Return a copy of the cached configuration to avoid accidental mutation."""
    return deepcopy(_load_raw_config())
