<h1 align="center">🪨 MineAI</h1>
<p align="center">
  <em>AI-assisted geospatial intelligence for historic underground mine maps</em><br>
  <a href="https://github.com/ashleyreagan/MineAI/stargazers">
    <img src="https://img.shields.io/github/stars/ashleyreagan/MineAI?color=gold&style=flat-square" alt="stars">
  </a>
  <a href="https://github.com/ashleyreagan/MineAI/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="license">
  </a>
  <img src="https://img.shields.io/badge/python-3.11+-brightgreen.svg?style=flat-square" alt="python">
  <img src="https://img.shields.io/badge/streamlit-ready-FF4B4B.svg?style=flat-square" alt="streamlit">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg?style=flat-square" alt="platform">
</p>

<p align="center">
  <img src="assets/header_banner_v2.png" width="1000" alt="MineAI banner">
</p>

---

## 🧭 Overview

**MineAI** converts **historic underground mine maps** into **georeferenced, machine‑readable spatial datasets** using a pipeline of OCR, deep learning (YOLOv8), and AI‑assisted georeferencing. The goal is to make century‑old survey sheets usable in modern AML/OSMRE workflows.

---

## 🔬 Methodology — How It Works

1. **Ingestion & QC** — Load scanned rasters, read EXIF/DPI, build an ingest manifest.  
2. **AI‑Assisted Georeferencing** — Detect alignment cues (GCPs, section lines, rivers) and solve affine/projective warps to USGS/NAIP basemaps.  
3. **OCR & Layout Parsing** — Extract mine names, seam notes, T/R/S, and annotations using Tesseract + (optional) TrOCR for low‑quality handwriting.  
4. **Mine Entry Detection** — YOLOv8 identifies portal, adit, and shaft symbols; detections are projected into map coordinates.  
5. **Post‑processing & Export** — Clean duplicates, attach metadata, export to Shapefile/GeoJSON, optionally run PDAL normalization for LiDAR contexts.  
6. **QA Visualization** — Streamlit dashboard previews rasters/detections for analyst review and final export.

---

## 🧩 Architecture

```
                 ┌────────────┐
   Scans (.tif)  │  Ingest    │  EXIF/DPI, manifest
      ─────────► │ (ingest/)  │────────────────┐
                 └─────┬──────┘                │
                       │                       │
                       ▼                       │
                 ┌────────────┐                │
  GCPs / cues    │ Georef     │  affine/projective warp
 (optional) ───► │ (georef/)  │──────────────┐ │
                 └─────┬──────┘              │ │
                       │                     │ │
                       ├───────────────┐     │ │
                       │               │     │ │
                       ▼               ▼     ▼ ▼
                ┌───────────┐   ┌───────────┐ ┌───────────┐
                │   OCR     │   │  Detect   │ │  Streamlit│
                │ (ocr/)    │   │ (detect/) │ │  (gui/)   │
                └────┬──────┘   └────┬──────┘ └────┬──────┘
                     │              │              │
                     ▼              ▼              ▼
                 JSON text     GeoJSON boxes   QA preview
                     └──────────────┬──────────────┘
                                    ▼
                            Shapefile / GeoJSON
```

---

## 🚀 Quickstart

```bash
conda env create -f environment.yml
conda activate mineai

# End-to-end pipeline
python run_pipeline.py

# Or launch visualization
streamlit run src/gui/app_streamlit.py
```

**Inputs:** place `.tif/.jpg/.png` scans in `data/raw/`.  
**Outputs:** `data/final/` (shapefiles/geojson) and `data/detect/` (intermediate detections).

---

## 🧠 Models

### YOLOv8 Mine Entry Detector
- Trained on annotated portal/adit/shaft symbols from historic sheets.  
- Bounding boxes exported with confidence and class; coordinates mapped via georef transform.

Training command:
```bash
yolo task=detect mode=train data=data/mine_entries.yaml model=yolov8m.pt epochs=100 imgsz=640
```

### OCR (Tesseract / TrOCR)
- Baseline Tesseract for speed; TrOCR improves stylized handwriting.  
- Swap models by editing `trocr_id` in `config.yaml`.

---

## 🛰️ Georeferencing Details
- ORB/SIFT keypoints for coarse match; RANSAC for outlier‑robust alignment.  
- Affine or projective warp via rasterio/GDAL.  
- Planned: RMSE/reprojection error reporting for QA gates.

---

## 🧭 Integration
- **ArcGIS Pro / QGIS** — open shapefiles/GeoJSON directly.  
- **AGOL / PostGIS** — optional publishing via standard connectors.  
- **PDAL** — LiDAR normalization and DEM differencing hooks for elevation analysis.

---

## 🛠️ Development & Contributing
- Create env with `environment.yml`.  
- Keep PRs small and focused; include a short demo map or screenshot.  
- Linting/testing: `pytest` in `tests/` (add sample fixtures only, no giant rasters).

---

## 🪪 License
MIT © 2025 Ashley Mitchell — *Not an official DOI/OSMRE product.*