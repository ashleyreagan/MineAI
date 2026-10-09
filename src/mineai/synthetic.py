"""Synthetic mine-map sheets with known answers, for the quickstart demo and the tests.

A sheet has off-white paper, a coordinate grid, lettering, and red "portal"
symbols at known pixel positions. Its true pixel→map transform is a known affine
(slightly rotated), so georeferencing, projection and detection can all be checked
against exact expected values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Pixel → NAD83 / UTM 17N, near Pittsburgh. ~1.2 m per pixel with a slight rotation.
TRUE_AFFINE = (585000.0, 1.2, 0.05, 4477000.0, -0.05, -1.2)
CRS = "EPSG:26917"

PORTALS = [  # (centre col, centre row) — two sit across detection tile seams
    (420, 380), (1010, 900), (1530, 760), (2210, 1480), (2600, 400), (760, 1780),
]
LABELS = [
    ((300, 120), "SAMPLE MINE No. 2"),
    ((470, 365), "DRIFT MOUTH"),
    ((1580, 745), "AIR SHAFT"),
    ((1900, 1900), "PITTSBURGH COAL"),
]


def true_xy(col, row):
    x0, a, b, y0, d, e = TRUE_AFFINE
    col, row = np.asarray(col, float), np.asarray(row, float)
    return x0 + a * col + b * row, y0 + d * col + e * row


def make_sheet(path: str | Path, size=(3000, 2200), symbol_radius: int = 14, font_size: int = 44,
               portals=None, labels=None) -> list[tuple[float, float, float, float]]:
    """Draw a sheet and save it. Returns the portal boxes (x0, y0, x1, y1) in pixels."""
    w, h = size
    img = Image.new("RGB", (w, h), (238, 232, 214))
    d = ImageDraw.Draw(img)
    for x in range(0, w, 500):
        d.line([(x, 0), (x, h)], fill=(150, 150, 150), width=2)
    for y in range(0, h, 500):
        d.line([(0, y), (w, y)], fill=(150, 150, 150), width=2)
    font = ImageFont.load_default(size=font_size)
    for (x, y), text in (LABELS if labels is None else labels):
        d.text((x, y), text, fill=(20, 20, 20), font=font)
    boxes = []
    for cx, cy in (PORTALS if portals is None else portals):
        r = symbol_radius
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 0, 0))
        boxes.append((cx - r, cy - r, cx + r + 1, cy + r + 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, dpi=(200, 200))
    return boxes


def write_yolo_labels(path: str | Path, boxes, size, cls: int = 0) -> None:
    """Full-sheet YOLO label file for `mineai dataset`."""
    w, h = size
    lines = [f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}" for x0, y0, x1, y1 in boxes]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_gcps_csv(path: str | Path, points=None, noise: float = 0.5, seed: int = 1) -> None:
    """GCPs at grid intersections, with a little noise so RMSE isn't exactly zero."""
    if points is None:
        points = [(500, 500), (2500, 500), (500, 1500), (2500, 1500), (1500, 1000), (1000, 2000), (2000, 0), (0, 1000)]
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# crs: {CRS}", "id,col,row,x,y"]
    for i, (c, r) in enumerate(points, 1):
        x, y = true_xy(c, r)
        x += rng.normal(0, noise)
        y += rng.normal(0, noise)
        lines.append(f"{i},{c},{r},{x:.3f},{y:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qgis_points(path: str | Path, points, crs_wkt: str) -> None:
    """Same GCPs in QGIS Georeferencer format (negative pixel Y, #CRS header)."""
    lines = [f"#CRS: {crs_wkt}", "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual"]
    for c, r in points:
        x, y = true_xy(c, r)
        lines.append(f"{x:.6f},{y:.6f},{c},{-r},1,0,0,0")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class RedBlobDetector:
    """Stand-in for a trained model in tests/demos: finds the red synthetic symbols.

    Implements the same Detector interface as UltralyticsDetector.
    """

    names = {0: "portal"}
    model_id = "red-blob-test-detector"
    model_sha16 = None

    def __init__(self, min_pixels: int = 30):
        self.min_pixels = min_pixels
        self.calls = 0

    def predict(self, tiles):
        from scipy import ndimage

        out = []
        for t in tiles:
            self.calls += 1
            t = t.astype(int)
            mask = (t[:, :, 0] > 150) & (t[:, :, 1] < 80) & (t[:, :, 2] < 80)
            lab, n = ndimage.label(mask)
            boxes = []
            for sl in ndimage.find_objects(lab):
                if sl is None:
                    continue
                ys, xs = sl
                if (ys.stop - ys.start) * (xs.stop - xs.start) < self.min_pixels:
                    continue
                boxes.append([xs.start, ys.start, xs.stop, ys.stop, 0.9, 0])
            out.append(np.array(boxes, float).reshape(-1, 6))
        return out
