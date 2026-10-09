"""Stage 4: find mine-entry symbols (portals, shafts, slopes...) on the original scan.

Detection runs tile by tile at the scan's native resolution. Shrinking a whole
15,000-pixel sheet to the model's input size would reduce a 40-pixel portal
symbol to under 2 pixels, so it would never be found.

  * Tiles overlap by `detect.overlap` pixels, which must be larger than the
    biggest symbol. Boxes cut off by an interior tile edge are dropped because
    the neighbouring tile sees that symbol whole.
  * Duplicates from the overlaps are merged with per-class non-maximum suppression.
  * There is no fallback detector. If the weights are missing or can't be loaded,
    the stage fails and the pipeline exits non-zero.

Outputs <sheet>_candidates.json in pixel coordinates. Export projects them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from rich.console import Console

from .manifest import FAILED, OK, Manifest
from .raster import Tile, iter_boxes_nms, open_raster, read_tile, sha16, tile_grid

console = Console()


class DetectionModelError(RuntimeError):
    pass


class Detector(Protocol):
    """Anything that turns a batch of RGB uint8 tiles into boxes.

    predict() returns, for each tile, an (N, 6) array of
    x0, y0, x1, y1, score, class_id in that tile's pixel coordinates.
    """

    names: dict[int, str]
    model_id: str
    model_sha16: str | None

    def predict(self, tiles: Sequence[np.ndarray]) -> list[np.ndarray]: ...


class UltralyticsDetector:
    def __init__(self, weights: str, imgsz: int = 1024, conf: float = 0.25, device: str | None = None):
        path = Path(weights)
        if not path.exists():
            raise DetectionModelError(
                f"Model weights not found: {path}. MineAI does not ship a trained mine-entry model; "
                "train one (see docs/TRAINING.md) and set detect.weights, or run with --skip detect."
            )
        if path.stat().st_size < 100_000:
            raise DetectionModelError(f"{path} is only {path.stat().st_size} bytes; that is a placeholder, not a trained model.")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise DetectionModelError("ultralytics is not installed (pip install ultralytics).") from e
        try:
            self.model = YOLO(str(path))
        except Exception as e:
            raise DetectionModelError(f"Could not load {path}: {type(e).__name__}: {e}") from e
        self.names = {int(k): str(v) for k, v in dict(self.model.names).items()}
        self.model_id = path.name
        self.model_sha16 = sha16(path)
        self.imgsz, self.conf, self.device = imgsz, conf, device

    def predict(self, tiles: Sequence[np.ndarray]) -> list[np.ndarray]:
        # Ultralytics expects BGR for numpy input.
        batch = [np.ascontiguousarray(t[:, :, ::-1]) for t in tiles]
        results = self.model.predict(batch, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6)))
                continue
            xyxy = b.xyxy.cpu().numpy()
            out.append(np.column_stack([xyxy, b.conf.cpu().numpy(), b.cls.cpu().numpy()]))
        return out


def detect_sheet(path: str, detector: Detector, tile_size: int, overlap: int, iou: float, batch: int = 8) -> list[dict]:
    """Tiled inference over one sheet. Returns candidates in sheet pixel coordinates."""
    raw: list[np.ndarray] = []
    edge = 2.0
    with open_raster(path) as ds:
        W, H = ds.width, ds.height
        tiles = tile_grid(W, H, tile_size, overlap)
        for start in range(0, len(tiles), batch):
            chunk: list[Tile] = tiles[start : start + batch]
            imgs = [read_tile(ds, t) for t in chunk]
            preds = detector.predict(imgs)
            for t, p in zip(chunk, preds):
                p = np.asarray(p, float).reshape(-1, 6)
                if not len(p):
                    continue
                x0, y0, x1, y1 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
                cut = np.zeros(len(p), bool)
                if t.col_off > 0:
                    cut |= x0 <= edge
                if t.row_off > 0:
                    cut |= y0 <= edge
                if t.col_off + t.width < W:
                    cut |= x1 >= t.width - edge
                if t.row_off + t.height < H:
                    cut |= y1 >= t.height - edge
                p = p[~cut].copy()
                p[:, [0, 2]] += t.col_off
                p[:, [1, 3]] += t.row_off
                raw.append(p)
    if not raw:
        return []
    allp = np.vstack(raw)
    keep: list[int] = []
    for cls in np.unique(allp[:, 5]):
        idx = np.flatnonzero(allp[:, 5] == cls)
        keep.extend(idx[i] for i in iter_boxes_nms(allp[idx, :4], allp[idx, 4], iou))
    allp = allp[sorted(keep, key=lambda i: -allp[i, 4])]
    cands = []
    for n, (x0, y0, x1, y1, score, cls) in enumerate(allp, 1):
        cands.append(
            {
                "id": n,
                "cls_id": int(cls),
                "cls_name": detector.names.get(int(cls), str(int(cls))),
                "score": round(float(score), 4),
                "bbox": [round(float(v), 1) for v in (x0, y0, x1, y1)],
                "center": [round(float((x0 + x1) / 2), 1), round(float((y0 + y1) / 2), 1)],
            }
        )
    return cands


def run(cfg: dict, manifest: Manifest, detector: Detector | None = None) -> None:
    dcfg = cfg["detect"]
    out_dir = Path(cfg["paths"]["detect"])
    if detector is None:
        try:
            detector = UltralyticsDetector(dcfg["weights"], int(dcfg["imgsz"]), float(dcfg["conf"]), dcfg.get("device"))
        except DetectionModelError as e:
            manifest.fail_stage("detect", str(e))
            console.print(f"[red]✖ Detection unavailable:[/] {e}")
            return
    manifest.set_stage_error("detect", None)
    if int(dcfg["overlap"]) >= int(dcfg["tile_size"]):
        manifest.fail_stage("detect", "detect.overlap must be smaller than detect.tile_size")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for sheet_id, sheet in manifest.sheets.items():
        if manifest.status(sheet_id, "ingest") != OK:
            continue
        try:
            cands = detect_sheet(sheet["path"], detector, int(dcfg["tile_size"]), int(dcfg["overlap"]), float(dcfg["iou"]), int(dcfg.get("batch", 8)))
        except Exception as e:
            manifest.set_status(sheet_id, "detect", FAILED, f"{type(e).__name__}: {e}")
            console.print(f"[red]✖ Detection failed[/] {sheet_id}: {e}")
            continue
        out = {
            "sheet": sheet_id,
            "sheet_sha16": sheet.get("sha16"),
            "model": detector.model_id,
            "model_sha16": detector.model_sha16,
            "classes": detector.names,
            "params": {k: dcfg[k] for k in ("tile_size", "overlap", "imgsz", "conf", "iou")},
            "candidates": cands,
        }
        with open(out_dir / f"{sheet_id}_candidates.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        manifest.set_status(sheet_id, "detect", OK, f"{len(cands)} candidates", n_candidates=len(cands), model=detector.model_id)
        console.print(f"[green]🔎 Detect[/] {sheet_id}: {len(cands)} candidates")
