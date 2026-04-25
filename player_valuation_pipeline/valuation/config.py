"""Config loader voor valuation pipeline."""

import yaml
from pathlib import Path
from typing import Any


class Config:
    """Wrapper around nested dict voor dot-access."""
    
    def __init__(self, d: dict):
        self._d = d
        for key, val in d.items():
            if isinstance(val, dict):
                setattr(self, key, Config(val))
            else:
                setattr(self, key, val)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)
    
    def as_dict(self) -> dict:
        return self._d


def load_config(config_path: str = None) -> Config:
    """
    Laad config.yaml. Default: configs/default.yaml
    
    Parameters
    ----------
    config_path : optioneel pad naar andere config
    
    Returns
    -------
    Config object met nested dot-access
    """
    if config_path is None:
        # Default: configs/default.yaml relatief aan dit bestand
        pkg_root = Path(__file__).parent.parent
        config_path = pkg_root / "configs" / "default.yaml"
    
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    
    return Config(raw)
