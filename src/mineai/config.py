"""Configuration loading with defaults.

Every key has a default here, so a config file only needs to list what it changes.
Relative paths in the config are resolved against the config file's folder, so the
pipeline behaves the same no matter which directory it is launched from.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "paths": {
        "raw": "data/raw",
        "gcp": "data/gcp",
        "georef": "data/georef",
        "ocr": "data/ocr",
        "detect": "data/detect",
        "final": "data/final",
        "manifest": "data/manifest.json",
    },
    "georef": {
        # poly1 (affine), poly2 (2nd-order polynomial), or tps (thin-plate spline)
        "method": "poly1",
        # CRS for GCP CSV files and for scans that carry a world file but no CRS.
        # QGIS .points files carry their own CRS and ignore this.
        "default_crs": "EPSG:26917",
        "resampling": "nearest",
        "write_warped": True,
        # Flag a sheet in the summary when its GCP RMSE (map units) exceeds this.
        "max_rmse": None,
    },
    "ocr": {
        "enabled": True,
        "lang": "eng",
        "psm": 11,
        "tile_size": 2000,
        "overlap": 400,
        "min_conf": 50,
        "rotations": [0],
    },
    "detect": {
        "weights": "models/yolo/mine_entry.pt",
        "tile_size": 1024,
        "overlap": 256,
        "imgsz": 1024,
        "conf": 0.25,
        "iou": 0.5,
        "batch": 8,
        "device": None,
    },
    "export": {
        "crs": "EPSG:4269",
        "gpkg": "mineai_candidates.gpkg",
    },
}

PATH_KEYS = ("raw", "gcp", "georef", "ocr", "detect", "final", "manifest")


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = "config.yaml") -> dict[str, Any]:
    """Load a YAML config, fill in defaults, and resolve paths to absolute."""
    cfg_path = Path(path) if path else None
    user: dict[str, Any] = {}
    if cfg_path and cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        base_dir = cfg_path.resolve().parent
    elif cfg_path and str(path) != "config.yaml":
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    else:
        base_dir = Path.cwd()

    cfg = _merge(DEFAULTS, user)
    cfg["_base_dir"] = str(base_dir)
    for key in PATH_KEYS:
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = str(p if p.is_absolute() else (base_dir / p))
    weights = cfg["detect"].get("weights")
    if weights:
        w = Path(weights)
        cfg["detect"]["weights"] = str(w if w.is_absolute() else (base_dir / w))
    return cfg
