# 🪨 MineAI
**AI-assisted digitizing of historic mine maps**
*Part of the Appalachian Restoration Intelligence Ecosystem*
![MineAI banner](assets/header_banner_final_proxy.png)

---

### 📍 Status: alpha

| Part | State |
|---|---|
| Ingest (any size scan, manifest with hashes) | ✅ working, tested |
| Georeferencing from GCPs (QGIS `.points` or CSV; affine, 2nd-order, thin-plate spline) with per-GCP residuals and RMSE | ✅ working, tested |
| OCR of map lettering (Tesseract, tiled, optional rotated text) | ✅ working, tested on synthetic sheets |
| Mine-entry symbol detection (YOLO, tiled at native resolution) | ⚙️ pipeline working and tested; **no trained model yet** |
| Export to GeoPackage with provenance and review fields | ✅ working, tested |
| QA dashboard (Streamlit) | ✅ working |
| Training-set builder and held-out evaluation | ✅ working, tested |

**No accuracy numbers are published yet.** They will come from `mineai evaluate`
on held-out labeled sheets once a model is trained (see [docs/TRAINING.md](docs/TRAINING.md)).
Until then, `mineai run --skip detect` georeferences and OCRs sheets, and
detection fails with a clear message rather than inventing results.

Everything MineAI exports is a **candidate for analyst review**, never a finished
inventory. Nothing it produces should go into AMLIS or any other system of
record without review.

---

### 🧭 Overview
MineAI turns scanned underground mine maps into georeferenced, machine-readable
GIS layers for AML inventory and inspection work. It:

1. **georeferences** each sheet from ground control points, reporting the error;
2. **reads the lettering** (mine names, seam notes, labels) with OCR;
3. **finds mine-entry symbols** (portals, drifts, shafts, slopes) with a YOLO detector;
4. **projects** text and symbols into map coordinates and writes a GeoPackage
   whose features are tagged `review_status = unreviewed`, with full provenance.

It complements the [AMD Water Quality Pipeline](https://github.com/ashleyreagan/AMD_Water_Quality_Pipeline).

---

### 🔬 How it works

| Stage | Command | What it does |
|---|---|---|
| **1 Ingest** | `mineai ingest` | Inventories `data/raw/` and records size, bands, bit depth, palette, DPI and SHA-256 in `data/manifest.json`. Any size of sheet works, because nothing is ever loaded whole. |
| **2 Georef** | `mineai georef` | Fits a pixel→map transform from `data/gcp/<sheet>.points` (QGIS) or `.csv`, or uses an already-georeferenced GeoTIFF or world file. Writes the GCP residuals and RMSE, plus a warped GeoTIFF for viewing. See [docs/GEOREFERENCING.md](docs/GEOREFERENCING.md). |
| **3 OCR** | `mineai ocr` | Runs Tesseract in sparse-text mode over overlapping tiles of the **original** scan, removes duplicates between tiles, and can also read labels rotated 90°/270°. |
| **4 Detect** | `mineai detect` | Runs YOLO over overlapping native-resolution tiles of the original scan and merges duplicates across tiles. Fails if no trained weights are configured. |
| **5 Export** | `mineai export` | Projects detections and text through each sheet's GCP model into `data/final/mineai_candidates.gpkg`, with layers `entry_candidates`, `ocr_text` and `sheets`. Candidates on sheets that aren't georeferenced go to `unplaced_candidates.csv`. |
| **QA** | `mineai gui` | Dashboard showing the scan with boxes, GCP residuals, and candidates on USGS Topo. |

OCR and detection run on the original scan, not the warped one. Warping rotates
and resamples lettering and symbols. Positions stay in pixel space until export,
which projects them with the same model as the warp.

```text
 data/raw/*.tif|png|jpg ──► ingest ──► manifest.json (size, dpi, sha256, status per stage)
                              │
 data/gcp/<sheet>.points|csv ─┼──► georef ──► <sheet>_georef.json (GCPs, residuals, RMSE)
                              │                <sheet>_georef.tif  (viewing copy)
                              ├──► ocr ─────► <sheet>_ocr.json        (pixel coords)
                              └──► detect ──► <sheet>_candidates.json (pixel coords)
                                                   │
                                     export ◄──────┘  project through GCP model
                                       │
                                       ▼
                     mineai_candidates.gpkg  (entry_candidates · ocr_text · sheets)
```

Every stage records `ok`, `skipped` (with the reason) or `failed` (with the error) for
every sheet. `mineai` exits with status 1 if anything failed.

---

### 🚀 Quickstart

```bash
conda env create -f environment.yml      # or: pip install -e ".[detect,gui,dev]" (+ Tesseract)
conda activate mineai

mineai run --skip detect                 # georef + OCR + export on the sample sheet
mineai gui                               # QA dashboard
pytest                                   # the test suite (synthetic sheets, ~30 s)
```

The repo includes a synthetic sample sheet (`data/raw/sample_sheet.png`) and its
GCPs (`data/gcp/sample_sheet.csv`). `scripts/make_sample.py` regenerates them.

**For your own sheets:** put scans in `data/raw/`, create a GCP file for each in
`data/gcp/` ([how](docs/GEOREFERENCING.md)), and check `georef.default_crs` and
`export.crs` in `config.yaml`. Once you have trained weights
([how](docs/TRAINING.md)), set `detect.weights` and run `mineai run`.

`python run_pipeline.py` still works and is the same as `mineai run`.

---

### 📦 Output fields (`entry_candidates`)

| Field | Meaning |
|---|---|
| `candidate_id` | `<sheet>:<n>` |
| `cls`, `score` | detected symbol class and model confidence |
| `sheet`, `sheet_sha16` | source scan and its hash |
| `model`, `model_sha16` | weights file and hash that produced it |
| `pixel_x`, `pixel_y`, `bbox_px` | position on the original scan |
| `georef_method`, `georef_rmse`, `georef_rmse_kind`, `georef_crs` | how it was placed, and how well the GCPs agree |
| `review_status`, `reviewer`, `review_notes` | for the analyst; starts as `unreviewed` |
| `exported_at`, `mineai_version` | run provenance |

`ocr_text` carries the text, Tesseract confidence, rotation and the same
provenance. `sheets` carries each footprint with its GCP count and RMSE.

---

### ⚙️ Configuration

`config.yaml` lists every option with comments. The ones most often changed:

- `georef.method`: `poly1` | `poly2` | `tps`; `georef.default_crs`; `georef.max_rmse`
- `ocr.rotations`: `[0]` by default; `[0, 90, 270]` also reads labels running up and down the sheet
- `detect.weights`, `detect.conf` (lower it to raise recall), `detect.overlap` (must exceed the largest symbol)
- `export.crs`: `EPSG:4269` (NAD83 geographic) by default

---

### 🧩 Roadmap

- [ ] Label sheets from several states and companies; train and publish the first model **with held-out metrics**
- [ ] Review editing in the dashboard (accept / reject / reclassify, written back to the GeoPackage)
- [ ] Semi-automatic GCPs: detect coordinate ticks and grid intersections
- [ ] Handwriting OCR for annotations (e.g. TrOCR on line crops)
- [ ] Map export fields to AMLIS feature codes
- [ ] Publish to PostGIS / ArcGIS Online
- [ ] Mine pool and elevation analysis with 3DEP LiDAR

---

### 💬 Citation

> **Mitchell, A. (2025).** *MineAI: AI-Assisted Digitizing of Historic Underground Mine Maps.* GitHub.
> [https://github.com/ashleyreagan/MineAI](https://github.com/ashleyreagan/MineAI)

---

### 🛠️ Development & Contributing
1. Fork the repository and `pip install -e ".[detect,gui,dev]"`.
2. Keep pull requests small; add a test, using `mineai.synthetic` for sheets with known answers.
3. `pytest` must pass (`pytest -m "not slow"` skips the 276-megapixel sheet test).

---

### 🪪 License
MIT © 2025 Ashley Mitchell. *Not an official DOI/OSMRE product.*

---

### 🙏 Acknowledgments
MineAI builds on the mission of **OSMRE**, **state geological surveys**, and the
**Appalachian mine-mapping community**. Thanks to the open-data initiatives of
**PASDA**, **USGS**, and **EPA** for the geospatial layers that make this work possible.

### 🧠 Related Projects
- **[AMD Water Quality Pipeline](https://github.com/ashleyreagan/AMD_Water_Quality_Pipeline)**: chemistry and mine data for watershed restoration analytics.
