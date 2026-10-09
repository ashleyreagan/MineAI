"""Stage 3: OCR the original (unwarped) scan in overlapping tiles with Tesseract.

OCR runs on the original scan because warping rotates and resamples lettering.
Text positions are kept in pixel coordinates; export projects them through the
sheet's georef model.

Tiling details:
  * Tesseract runs in sparse-text mode (--psm 11 by default), which suits map
    labels scattered across a sheet far better than page-layout modes.
  * Lines that touch an interior tile edge are dropped: the overlap guarantees
    the same line appears whole in a neighbouring tile (for labels shorter than
    `ocr.overlap` pixels).
  * `ocr.rotations` lists text angles to look for, in degrees counter-clockwise.
    [0, 90, 270] also catches labels written up or down the sheet, at 3× the time.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rich.console import Console

from .manifest import FAILED, OK, SKIPPED, Manifest
from .raster import Tile, iter_boxes_nms, open_raster, read_tile, tile_grid

console = Console()


def _rotate_back(k: int, w: int, h: int, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map points from np.rot90(tile, k) back into the unrotated tile (w × h)."""
    k %= 4
    if k == 0:
        return xs, ys
    if k == 1:  # rot90 CCW: (x, y) -> (y, w - x)
        return w - ys, xs
    if k == 2:
        return w - xs, h - ys
    return ys, h - xs  # k == 3, CW: (x, y) -> (h - y, x)


def ocr_tile(img: np.ndarray, tile: Tile, rotation: int, ocfg: dict, sheet_w: int, sheet_h: int) -> list[dict]:
    import pytesseract

    k = (-(rotation // 90)) % 4  # rotate the image clockwise by `rotation` to make text upright
    rot = np.ascontiguousarray(np.rot90(img, k)) if k else img
    data = pytesseract.image_to_data(
        rot,
        lang=ocfg.get("lang", "eng"),
        config=f"--psm {int(ocfg.get('psm', 11))}",
        output_type=pytesseract.Output.DICT,
    )
    min_conf = float(ocfg.get("min_conf", 60))
    lines: dict[tuple, list[dict]] = {}
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < min_conf:
            continue
        if not any(ch.isalnum() for ch in text):
            continue  # map symbols misread as punctuation (e.g. a portal read as "@")
        x0, y0 = float(data["left"][i]), float(data["top"][i])
        x1, y1 = x0 + float(data["width"][i]), y0 + float(data["height"][i])
        cx, cy = _rotate_back(k, tile.width, tile.height, np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1]))
        box = [cx.min() + tile.col_off, cy.min() + tile.row_off, cx.max() + tile.col_off, cy.max() + tile.row_off]
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append({"text": text, "conf": conf, "bbox": [round(v, 1) for v in box]})

    out = []
    edge = 2.0
    for words in lines.values():
        b = np.array([w["bbox"] for w in words])
        box = [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]
        # Drop lines cut off by an interior tile edge; the overlapping tile has them whole.
        if (box[0] <= tile.col_off + edge and tile.col_off > 0) or (box[1] <= tile.row_off + edge and tile.row_off > 0):
            continue
        if (box[2] >= tile.col_off + tile.width - edge and tile.col_off + tile.width < sheet_w) or (
            box[3] >= tile.row_off + tile.height - edge and tile.row_off + tile.height < sheet_h
        ):
            continue
        out.append(
            {
                "text": " ".join(w["text"] for w in words),
                "conf": round(float(np.mean([w["conf"] for w in words])), 1),
                "bbox": [round(float(v), 1) for v in box],
                "rotation": rotation,
                "words": words,
            }
        )
    return out


def ocr_sheet(path: str, ocfg: dict) -> list[dict]:
    rotations = [int(r) % 360 for r in ocfg.get("rotations", [0])]
    bad = [r for r in rotations if r % 90]
    if bad:
        raise ValueError(f"ocr.rotations must be multiples of 90; got {bad}")
    workers = int(ocfg.get("workers") or max(1, (os.cpu_count() or 2) - 1))
    found: list[dict] = []
    with open_raster(path) as ds:
        tiles = tile_grid(ds.width, ds.height, int(ocfg["tile_size"]), int(ocfg["overlap"]))
        W, H = ds.width, ds.height
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for tile in tiles:
                img = read_tile(ds, tile)  # dataset reads stay on this thread
                for rot in rotations:
                    futures.append(pool.submit(ocr_tile, img, tile, rot, ocfg, W, H))
                if len(futures) >= workers * 4:
                    for f in futures:
                        found.extend(f.result())
                    futures = []
            for f in futures:
                found.extend(f.result())
    if not found:
        return []
    # Remove duplicates from overlapping tiles / rotations, keeping the most confident.
    boxes = np.array([l["bbox"] for l in found], float)
    scores = np.array([l["conf"] for l in found], float)
    keep = sorted(iter_boxes_nms(boxes, scores, 0.3), key=lambda i: (found[i]["bbox"][1], found[i]["bbox"][0]))
    lines = [found[i] for i in keep]
    for n, line in enumerate(lines, 1):
        line["id"] = n
    return lines


def run(cfg: dict, manifest: Manifest) -> None:
    ocfg = cfg["ocr"]
    out_dir = Path(cfg["paths"]["ocr"])
    if not ocfg.get("enabled", True):
        for sheet_id in manifest.sheets:
            if manifest.status(sheet_id, "ingest") == OK:
                manifest.set_status(sheet_id, "ocr", SKIPPED, "ocr.enabled is false")
        return
    try:
        import pytesseract

        version = str(pytesseract.get_tesseract_version())
    except Exception as e:
        msg = f"Tesseract is not available ({type(e).__name__}: {e}). Install it (conda install -c conda-forge tesseract) or set ocr.enabled: false."
        manifest.fail_stage("ocr", msg)
        console.print(f"[red]✖ OCR unavailable:[/] {msg}")
        return
    manifest.set_stage_error("ocr", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sheet_id, sheet in manifest.sheets.items():
        if manifest.status(sheet_id, "ingest") != OK:
            continue
        try:
            lines = ocr_sheet(sheet["path"], ocfg)
        except Exception as e:
            manifest.set_status(sheet_id, "ocr", FAILED, f"{type(e).__name__}: {e}")
            console.print(f"[red]✖ OCR failed[/] {sheet_id}: {e}")
            continue
        out = {
            "sheet": sheet_id,
            "sheet_sha16": sheet.get("sha16"),
            "engine": f"tesseract {version}",
            "params": {k: ocfg[k] for k in ("lang", "psm", "tile_size", "overlap", "min_conf", "rotations")},
            "lines": lines,
        }
        out_path = out_dir / f"{sheet_id}_ocr.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        manifest.set_status(sheet_id, "ocr", OK, f"{len(lines)} text lines", n_lines=len(lines))
        console.print(f"[green]📝 OCR[/] {sheet_id}: {len(lines)} text lines")
