"""MineAI QA dashboard (Streamlit).

Launch with `mineai gui` (or `streamlit run src/mineai/gui/app.py -- --config config.yaml`).

For each sheet it shows the stage statuses, the scan with detection and OCR boxes
drawn on it, the GCP residuals, and the exported candidates on a map (reprojected
to WGS84 for display) over USGS Topo or OpenStreetMap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from mineai.config import load_config
from mineai.georef import load_record
from mineai.manifest import Manifest
from mineai.raster import open_raster, read_overview

STAGES = ["ingest", "georef", "ocr", "detect", "export"]


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args


@st.cache_data(show_spinner=False)
def _overview(path: str, mtime: float, max_side: int = 1800):
    with open_raster(path) as ds:
        return read_overview(ds, max_side)


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _layer(gpkg: Path, layer: str) -> gpd.GeoDataFrame | None:
    if not gpkg.exists():
        return None
    try:
        return gpd.read_file(gpkg, layer=layer)
    except Exception:
        return None


st.set_page_config(page_title="MineAI QA", layout="wide")
cfg = load_config(_args().config)
manifest = Manifest(cfg["paths"]["manifest"])
st.title("MineAI — map QA")
st.caption("Everything here is a candidate for analyst review, not a verified feature.")

if not manifest.sheets:
    st.warning(f"No manifest at {cfg['paths']['manifest']}. Run `mineai run` first.")
    st.stop()

rows = []
for sid, sh in sorted(manifest.sheets.items()):
    row = {"sheet": sid, "MP": sh.get("megapixels")}
    for s in STAGES:
        e = sh.get("stages", {}).get(s, {})
        row[s] = e.get("status", "—")
        if s == "georef" and e.get("rmse") is not None:
            row["rmse"] = round(e["rmse"], 2)
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
if manifest.data.get("stage_errors"):
    for stage, err in manifest.data["stage_errors"].items():
        st.error(f"Last {stage} run failed: {err['message']}")

sheet_id = st.sidebar.selectbox("Sheet", sorted(manifest.sheets))
sheet = manifest.sheets[sheet_id]
show_det = st.sidebar.checkbox("Detection boxes", True)
show_ocr = st.sidebar.checkbox("OCR text boxes", True)
min_score = st.sidebar.slider("Minimum detection score", 0.0, 1.0, 0.25, 0.05)
for s in STAGES:
    e = sheet.get("stages", {}).get(s)
    if e:
        st.sidebar.write(f"**{s}**: {e['status']} — {e.get('message', '')}")

cands = _json(Path(cfg["paths"]["detect"]) / f"{sheet_id}_candidates.json")
ocr = _json(Path(cfg["paths"]["ocr"]) / f"{sheet_id}_ocr.json")
rec = load_record(cfg, sheet_id)

left, right = st.columns(2)
with left:
    st.subheader("Scan")
    img_arr, scale = _overview(sheet["path"], Path(sheet["path"]).stat().st_mtime)
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    if show_ocr and ocr:
        for line in ocr["lines"]:
            x0, y0, x1, y1 = [v * scale for v in line["bbox"]]
            draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=(30, 110, 220), width=2)
    if show_det and cands:
        for c in cands["candidates"]:
            if c["score"] < min_score:
                continue
            x0, y0, x1, y1 = [v * scale for v in c["bbox"]]
            pad = 4
            draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], outline=(220, 30, 30), width=2)
    if rec and rec.get("gcps"):
        for g in rec["gcps"]:
            cx, cy = g["col"] * scale, g["row"] * scale
            draw.line([cx - 12, cy, cx + 12, cy], fill=(0, 150, 60), width=3)
            draw.line([cx, cy - 12, cx, cy + 12], fill=(0, 150, 60), width=3)
    st.image(img, width="stretch", caption="red: detections · blue: OCR lines · green: GCPs")

with right:
    st.subheader("Map")
    gpkg = Path(cfg["paths"]["final"]) / cfg["export"]["gpkg"]
    sheets_gdf = _layer(gpkg, "sheets")
    cand_gdf = _layer(gpkg, "entry_candidates")
    text_gdf = _layer(gpkg, "ocr_text")
    fp = sheets_gdf[sheets_gdf["sheet"] == sheet_id].to_crs(4326) if sheets_gdf is not None else None
    if fp is None or fp.empty:
        st.info("This sheet isn't on the map: it has no georeferencing yet (add a GCP file and re-run), or export hasn't run.")
    else:
        minx, miny, maxx, maxy = fp.total_bounds
        m = folium.Map(tiles=None, control_scale=True)
        folium.TileLayer(
            "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
            attr="USGS The National Map", name="USGS Topo", max_zoom=16,
        ).add_to(m)
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
        folium.GeoJson(fp[["sheet", "geometry"]].to_json(), name="sheet footprint",
                       style_function=lambda _: {"color": "#555", "fill": False, "weight": 2}).add_to(m)
        if show_ocr and text_gdf is not None:
            t = text_gdf[text_gdf["sheet"] == sheet_id].to_crs(4326)
            if not t.empty:
                folium.GeoJson(t[["text", "conf", "geometry"]].to_json(), name="OCR text",
                               style_function=lambda _: {"color": "#1e6edc", "weight": 1},
                               tooltip=folium.GeoJsonTooltip(["text", "conf"])).add_to(m)
        if show_det and cand_gdf is not None:
            c = cand_gdf[(cand_gdf["sheet"] == sheet_id) & (cand_gdf["score"] >= min_score)].to_crs(4326)
            for _, r in c.iterrows():
                folium.CircleMarker([r.geometry.y, r.geometry.x], radius=6, color="#dc1e1e", fill=True,
                                    tooltip=f"{r['cls']} · score {r['score']:.2f} · {r['candidate_id']}").add_to(m)
        folium.LayerControl().add_to(m)
        m.fit_bounds([[miny, minx], [maxy, maxx]])
        page = m.get_root().render()
        if hasattr(st, "iframe"):  # Streamlit ≥ 1.5x
            st.iframe(page, height=620)
        else:
            from streamlit.components.v1 import html

            html(page, height=620)

if rec and rec.get("gcps"):
    st.subheader("Ground control points")
    rmse_txt = "n/a" if rec.get("rmse") is None else f"{rec['rmse']:.2f}"
    st.write(f"Method **{rec['method']}**, CRS {rec.get('crs_name')}, RMSE {rmse_txt} ({rec.get('rmse_kind')})")
    st.dataframe(pd.DataFrame(rec["gcps"]), hide_index=True, width="stretch")
if cands:
    st.subheader(f"Detection candidates — model {cands.get('model')}")
    st.dataframe(pd.DataFrame(cands["candidates"]), hide_index=True, width="stretch")
if ocr:
    st.subheader("OCR text")
    st.dataframe(pd.DataFrame([{k: l[k] for k in ("id", "text", "conf", "rotation", "bbox")} for l in ocr["lines"]]),
                 hide_index=True, width="stretch")
