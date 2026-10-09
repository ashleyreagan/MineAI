"""Stage 5: project candidates and OCR text into map coordinates and write a GeoPackage.

Everything written here is a *candidate for analyst review*, never a finished
inventory: each feature carries review_status = "unreviewed" plus the provenance
needed to audit it (source sheet hash, model and hash, georef method and RMSE).

Layers in <final>/<export.gpkg>:
  entry_candidates  points (box centres) from detection
  ocr_text          polygons (text-line boxes) from OCR
  sheets            sheet footprints with georef method, GCP count and RMSE

Candidates on sheets that aren't georeferenced can't be placed on a map; they go to
<final>/unplaced_candidates.csv in pixel coordinates instead of being dropped.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from rich.console import Console
from shapely.geometry import Point, Polygon

from . import __version__
from .gcp import model_from_record
from .georef import load_record
from .manifest import FAILED, OK, SKIPPED, Manifest, now_iso

console = Console()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _current(manifest: Manifest, sheet_id: str, stage: str, data: dict | None) -> dict | None:
    """Only use a stage output that the manifest says succeeded for this exact scan."""
    if data is None or manifest.status(sheet_id, stage) != OK:
        return None
    if data.get("sheet_sha16") != manifest.sheets[sheet_id].get("sha16"):
        return None
    return data


def run(cfg: dict, manifest: Manifest) -> None:
    out_dir = Path(cfg["paths"]["final"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_crs = CRS.from_user_input(cfg["export"]["crs"])
    stamp = now_iso()
    cand_rows, text_rows, sheet_rows, unplaced = [], [], [], []

    for sheet_id, sheet in manifest.sheets.items():
        if manifest.status(sheet_id, "ingest") != OK:
            continue
        cands = _current(manifest, sheet_id, "detect", _load_json(Path(cfg["paths"]["detect"]) / f"{sheet_id}_candidates.json"))
        ocr = _current(manifest, sheet_id, "ocr", _load_json(Path(cfg["paths"]["ocr"]) / f"{sheet_id}_ocr.json"))
        rec = load_record(cfg, sheet_id) if manifest.status(sheet_id, "georef") == OK else None
        if rec is not None and rec.get("sheet_sha16") != sheet.get("sha16"):
            rec = None

        if rec is None:
            for c in (cands or {}).get("candidates", []):
                unplaced.append(
                    {"sheet": sheet_id, "candidate_id": f"{sheet_id}:{c['id']}", "cls": c["cls_name"], "score": c["score"],
                     "pixel_x": c["center"][0], "pixel_y": c["center"][1], "model": cands["model"]}
                )
            why = manifest.sheets[sheet_id]["stages"].get("georef", {}).get("message", "not georeferenced")
            manifest.set_status(sheet_id, "export", SKIPPED, f"not placed on map: {why}")
            continue

        try:
            model = model_from_record(rec)
            to_out = Transformer.from_crs(model.crs, out_crs, always_xy=True)

            def project(cols, rows):
                mx, my = model.forward(cols, rows)
                return to_out.transform(mx, my)

            common = {
                "sheet": sheet_id,
                "sheet_sha16": sheet.get("sha16"),
                "georef_method": rec["method"],
                "georef_rmse": rec.get("rmse"),
                "georef_rmse_kind": rec.get("rmse_kind"),
                "georef_crs": rec.get("crs_name"),
            }
            fx, fy = to_out.transform(*np.array(rec["footprint"]).T)
            sheet_rows.append({**common, "n_gcps": rec.get("n_gcps"), "gcp_file": rec.get("gcp_file"),
                               "geometry": Polygon(zip(fx, fy))})

            for c in (cands or {}).get("candidates", []):
                (x,), (y,) = project([c["center"][0]], [c["center"][1]])
                cand_rows.append({
                    "candidate_id": f"{sheet_id}:{c['id']}", **common,
                    "cls": c["cls_name"], "cls_id": c["cls_id"], "score": c["score"],
                    "model": cands["model"], "model_sha16": cands.get("model_sha16"),
                    "pixel_x": c["center"][0], "pixel_y": c["center"][1], "bbox_px": json.dumps(c["bbox"]),
                    "review_status": "unreviewed", "reviewer": "", "review_notes": "",
                    "exported_at": stamp, "mineai_version": __version__,
                    "geometry": Point(x, y),
                })
            for line in (ocr or {}).get("lines", []):
                x0, y0, x1, y1 = line["bbox"]
                px, py = project([x0, x1, x1, x0], [y0, y0, y1, y1])
                text_rows.append({
                    "text_id": f"{sheet_id}:{line['id']}", **common,
                    "text": line["text"], "conf": line["conf"], "rotation": line.get("rotation", 0),
                    "engine": ocr.get("engine"), "pixel_bbox": json.dumps(line["bbox"]),
                    "review_status": "unreviewed", "exported_at": stamp,
                    "geometry": Polygon(zip(px, py)),
                })
        except Exception as e:
            manifest.set_status(sheet_id, "export", FAILED, f"{type(e).__name__}: {e}")
            console.print(f"[red]✖ Export failed[/] {sheet_id}: {e}")
            continue
        n_c = len((cands or {}).get("candidates", []))
        n_t = len((ocr or {}).get("lines", []))
        manifest.set_status(sheet_id, "export", OK, f"{n_c} candidates, {n_t} text lines")

    gpkg = out_dir / cfg["export"]["gpkg"]
    gpkg.unlink(missing_ok=True)  # never append to (or mix with) a previous run
    written = []
    for layer, rows in (("entry_candidates", cand_rows), ("ocr_text", text_rows), ("sheets", sheet_rows)):
        if rows:
            gpd.GeoDataFrame(rows, geometry="geometry", crs=out_crs).to_file(gpkg, layer=layer, driver="GPKG")
            written.append(f"{layer} ({len(rows)})")
    unplaced_path = out_dir / "unplaced_candidates.csv"
    unplaced_path.unlink(missing_ok=True)
    if unplaced:
        with open(unplaced_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(unplaced[0]))
            w.writeheader()
            w.writerows(unplaced)
        console.print(f"[yellow]○ {len(unplaced)} candidates on un-georeferenced sheets →[/] {unplaced_path}")
    if written:
        console.print(f"[green]📦 Wrote[/] {gpkg}: {', '.join(written)}")
    else:
        console.print("[yellow]○ Nothing to export: no georeferenced sheets.[/]")
