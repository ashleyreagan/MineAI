import json
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from mineai.cli import main, run_stages
from mineai.manifest import Manifest
from mineai.synthetic import PORTALS, TRUE_AFFINE, make_sheet, true_xy

from conftest import needs_tesseract

ROOT = Path(__file__).resolve().parent.parent


def _gpkg(cfg):
    return Path(cfg["paths"]["final"]) / cfg["export"]["gpkg"]


def test_full_run_places_candidates_at_true_coordinates(cfg, fake_model):
    cfg["ocr"]["enabled"] = False
    assert run_stages(cfg, ["ingest", "georef", "ocr", "detect", "export"]) == 0
    c = gpd.read_file(_gpkg(cfg), layer="entry_candidates")
    assert len(c) == len(PORTALS)
    assert c.crs.to_epsg() == 26917
    for cx, cy in PORTALS:
        tx, ty = true_xy(cx + 0.5, cy + 0.5)
        d = np.hypot(c.geometry.x - tx, c.geometry.y - ty).min()
        assert d < 2.0  # GCPs carry ~0.5 m noise; pixel ≈ 1.2 m
    assert set(c["review_status"]) == {"unreviewed"}
    for col in ("sheet_sha16", "model", "georef_method", "georef_rmse", "pixel_x", "candidate_id"):
        assert col in c.columns
    s = gpd.read_file(_gpkg(cfg), layer="sheets")
    assert len(s) == 1 and s["georef_rmse"].iloc[0] < 2


def test_missing_model_fails_loudly_and_writes_no_candidates(cfg):
    cfg["ocr"]["enabled"] = False
    assert run_stages(cfg, ["ingest", "georef", "ocr", "detect", "export"]) == 1
    m = Manifest(cfg["paths"]["manifest"])
    assert m.status("sample_sheet", "detect") == "failed"
    layers = gpd.list_layers(_gpkg(cfg))["name"].tolist()
    assert "entry_candidates" not in layers


def test_placeholder_weights_fail(cfg, project):
    w = project / "models/yolo/mine_entry.pt"
    w.parent.mkdir(parents=True)
    w.write_text("DUMMY_WEIGHTS_HEADER_NOT_EMPTY")
    cfg["ocr"]["enabled"] = False
    assert run_stages(cfg, ["detect"]) == 1
    m = Manifest(cfg["paths"]["manifest"])
    assert "placeholder" in m.sheets["sample_sheet"]["stages"]["detect"]["message"]


def test_sheet_without_gcps_goes_to_unplaced_csv(cfg, project, fake_model):
    (project / "data/gcp/sample_sheet.csv").unlink()
    cfg["ocr"]["enabled"] = False
    assert run_stages(cfg, ["ingest", "georef", "ocr", "detect", "export"]) == 0
    m = Manifest(cfg["paths"]["manifest"])
    assert m.status("sample_sheet", "georef") == "skipped"
    unplaced = (Path(cfg["paths"]["final"]) / "unplaced_candidates.csv").read_text().splitlines()
    assert len(unplaced) == len(PORTALS) + 1
    assert not _gpkg(cfg).exists()


def test_bad_gcp_file_fails(cfg, project):
    (project / "data/gcp/sample_sheet.csv").write_text("# crs: EPSG:26917\ncol,row,x,y\n1,1,1,1\n")
    cfg["ocr"]["enabled"] = False
    assert run_stages(cfg, ["georef"]) == 1


def test_duplicate_sheet_names_fail(cfg, project):
    make_sheet(project / "data/raw/sample_sheet.tif", size=(400, 300))
    assert run_stages(cfg, ["ingest"]) == 1


def test_changed_scan_invalidates_old_outputs(cfg, project, fake_model):
    cfg["ocr"]["enabled"] = False
    assert run_stages(cfg, ["ingest", "georef", "ocr", "detect", "export"]) == 0
    # Replace the scan with one that has a single symbol, then export without re-detecting.
    make_sheet(project / "data/raw/sample_sheet.png", portals=[(100, 100)])
    os.utime(project / "data/raw/sample_sheet.png", (1, 1))
    assert run_stages(cfg, ["ingest", "georef", "export"]) == 0
    layers = gpd.list_layers(_gpkg(cfg))["name"].tolist()
    assert "entry_candidates" not in layers  # stale detections from the old scan are not exported


def test_already_georeferenced_geotiff_needs_no_gcps(cfg, project, fake_model):
    (project / "data/gcp/sample_sheet.csv").unlink()
    png = project / "data/raw/sample_sheet.png"
    with rasterio.open(png) as src:
        data = src.read()
    x0, a, b, y0, d, e = TRUE_AFFINE
    with rasterio.open(project / "data/raw/geo_sheet.tif", "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
                       count=3, dtype="uint8", crs="EPSG:26917", transform=Affine(a, b, x0, d, e, y0)) as dst:
        dst.write(data)
    png.unlink()
    cfg["ocr"]["enabled"] = False
    assert run_stages(cfg, ["ingest", "georef", "ocr", "detect", "export"]) == 0
    m = Manifest(cfg["paths"]["manifest"])
    assert m.sheets["geo_sheet"]["stages"]["georef"]["method"] == "embedded"
    c = gpd.read_file(_gpkg(cfg), layer="entry_candidates")
    tx, ty = true_xy(PORTALS[0][0] + 0.5, PORTALS[0][1] + 0.5)
    assert np.hypot(c.geometry.x - tx, c.geometry.y - ty).min() < 0.01


def test_export_reprojects_to_nad83_geographic(cfg, fake_model):
    cfg["ocr"]["enabled"] = False
    cfg["export"]["crs"] = "EPSG:4269"
    assert run_stages(cfg, ["ingest", "georef", "ocr", "detect", "export"]) == 0
    c = gpd.read_file(_gpkg(cfg), layer="entry_candidates")
    assert c.crs.to_epsg() == 4269
    assert ((c.geometry.x > -81) & (c.geometry.x < -79) & (c.geometry.y > 40) & (c.geometry.y < 41)).all()


@needs_tesseract
def test_ocr_text_is_exported_as_map_polygons(cfg, fake_model):
    assert run_stages(cfg, ["ingest", "georef", "ocr", "detect", "export"]) == 0
    t = gpd.read_file(_gpkg(cfg), layer="ocr_text")
    texts = " ".join(t["text"])
    assert "SAMPLE" in texts and "PITTSBURGH" in texts
    row = t[t["text"].str.contains("SAMPLE")].iloc[0]
    tx, ty = true_xy(300, 140)
    assert np.hypot(row.geometry.bounds[0] - tx, row.geometry.bounds[3] - ty) < 40


def test_cli_status_and_exit_codes(project):
    cfgp = str(project / "config.yaml")
    assert main(["--config", cfgp, "run", "--skip", "ocr", "detect"]) == 0
    assert main(["--config", cfgp, "detect"]) == 1
    assert main(["--config", cfgp, "status"]) == 0


def test_run_pipeline_shim_works_from_any_directory(tmp_path):
    env = {**os.environ, "PYTHONPATH": ""}
    r = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--skip", "ocr", "detect"],
                       cwd=tmp_path, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    # leave the repo's sample outputs as they were
