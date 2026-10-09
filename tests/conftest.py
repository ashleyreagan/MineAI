import shutil
from pathlib import Path

import pytest
import yaml

from mineai.config import load_config
from mineai.synthetic import make_sheet, write_gcps_csv


@pytest.fixture
def project(tmp_path: Path):
    """A throwaway project folder with one synthetic sheet, its GCPs and a config."""
    make_sheet(tmp_path / "data/raw/sample_sheet.png")
    write_gcps_csv(tmp_path / "data/gcp/sample_sheet.csv")
    cfg = {
        "export": {"crs": "EPSG:26917"},  # same as the GCPs, so expected coordinates are exact
        "detect": {"weights": "models/yolo/mine_entry.pt", "tile_size": 512, "overlap": 128},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return tmp_path


@pytest.fixture
def cfg(project):
    return load_config(project / "config.yaml")


@pytest.fixture
def fake_model(monkeypatch):
    """Make the detect stage use the red-blob stand-in instead of a YOLO model."""
    from mineai import detect
    from mineai.synthetic import RedBlobDetector

    monkeypatch.setattr(detect, "UltralyticsDetector", lambda *a, **k: RedBlobDetector())


needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
