"""Raster helpers shared by every stage.

All pixel reads go through rasterio windows, so sheets of any size (including
scans far above Pillow's ~179-megapixel decompression-bomb limit) are read one
tile at a time instead of loaded whole.
"""
from __future__ import annotations

import hashlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

SCAN_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".jp2")


@contextmanager
def open_raster(path: str | Path):
    """rasterio.open that doesn't warn about plain (not yet georeferenced) scans."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as ds:
            yield ds


def sha16(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    """First 16 hex digits of the file's SHA-256, read in chunks (scans can be GBs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


def has_georeferencing(ds) -> bool:
    """True when the file carries a real geotransform (GeoTIFF tags or a world file)."""
    return not ds.transform.is_identity or bool(ds.gcps[0])


def read_dpi(path: str | Path) -> tuple[float | None, float | None]:
    """Scan resolution from the file header, if recorded. Header-only; no pixel decode."""
    try:
        from PIL import Image

        old = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None  # header read only; pixels are never decoded here
        try:
            with Image.open(path) as im:
                dpi = im.info.get("dpi")
        finally:
            Image.MAX_IMAGE_PIXELS = old
        if dpi:
            return float(dpi[0]), float(dpi[1])
    except Exception:
        pass
    return None, None


@dataclass(frozen=True)
class Tile:
    col_off: int
    row_off: int
    width: int
    height: int

    @property
    def window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)


def tile_grid(width: int, height: int, tile: int, overlap: int) -> list[Tile]:
    """Cover the sheet with tiles of size `tile` that overlap by `overlap` pixels.

    The last tile in each row/column is shifted back to end exactly at the sheet
    edge, so no tile is padded and every pixel is covered.
    """
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile size")
    step = tile - overlap

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        s = list(range(0, n - tile, step))
        s.append(n - tile)
        return sorted(set(s))

    tiles = []
    for r in starts(height):
        for c in starts(width):
            tiles.append(Tile(c, r, min(tile, width - c), min(tile, height - r)))
    return tiles


def _to_uint8(arr: np.ndarray, nbits: int | None) -> np.ndarray:
    if arr.dtype == np.uint8:
        if nbits == 1:
            return (arr > 0).astype(np.uint8) * 255
        return arr
    a = arr.astype(np.float64)
    lo, hi = np.nanpercentile(a, 0.5), np.nanpercentile(a, 99.5)
    if hi <= lo:
        hi = lo + 1
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def read_rgb(ds, window: Window | None = None) -> np.ndarray:
    """Read a window as an H x W x 3 uint8 RGB array, whatever the source layout.

    Handles: 1-bit bilevel scans, paletted (colormap) images, grayscale,
    gray+alpha, RGB, RGBA (alpha dropped), and 16-bit data (stretched).
    """
    count = ds.count
    nbits = None
    try:
        nb = ds.tags(1, "IMAGE_STRUCTURE").get("NBITS")
        nbits = int(nb) if nb else None
    except Exception:
        pass

    colormap = None
    if count == 1:
        try:
            colormap = ds.colormap(1)
        except ValueError:
            colormap = None

    if count >= 3:
        data = ds.read([1, 2, 3], window=window)
        rgb = np.stack([_to_uint8(b, nbits) for b in data], axis=-1)
        return np.ascontiguousarray(rgb)

    band = ds.read(1, window=window)
    if colormap:
        lut = np.zeros((max(256, max(colormap) + 1), 3), dtype=np.uint8)
        for idx, rgba in colormap.items():
            lut[idx] = rgba[:3]
        return lut[band]
    gray = _to_uint8(band, nbits)
    return np.repeat(gray[:, :, None], 3, axis=2)


def read_tile(ds, tile: Tile) -> np.ndarray:
    return read_rgb(ds, tile.window)


def read_overview(ds, max_side: int = 2000) -> tuple[np.ndarray, float]:
    """Downsampled RGB preview of the whole sheet and the scale factor applied."""
    scale = min(1.0, max_side / max(ds.width, ds.height))
    out_h, out_w = max(1, int(ds.height * scale)), max(1, int(ds.width * scale))
    count = 3 if ds.count >= 3 else 1
    data = ds.read(
        list(range(1, count + 1)),
        out_shape=(count, out_h, out_w),
        resampling=rasterio.enums.Resampling.average if count == 3 else rasterio.enums.Resampling.nearest,
    )
    if count == 3:
        rgb = np.stack([_to_uint8(b, None) for b in data], axis=-1)
        return rgb, scale
    band = data[0]
    try:
        cmap = ds.colormap(1)
    except ValueError:
        cmap = None
    if cmap:
        lut = np.zeros((max(256, max(cmap) + 1), 3), dtype=np.uint8)
        for idx, rgba in cmap.items():
            lut[idx] = rgba[:3]
        return lut[band], scale
    nb = ds.tags(1, "IMAGE_STRUCTURE").get("NBITS")
    gray = _to_uint8(band, int(nb) if nb else None)
    return np.repeat(gray[:, :, None], 3, axis=2), scale


def list_scans(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SCAN_EXTENSIONS)


def iter_boxes_nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> Iterator[int]:
    """Greedy non-maximum suppression. Yields indices of kept boxes (x0, y0, x1, y1)."""
    order = np.argsort(-scores, kind="stable")
    x0, y0, x1, y1 = boxes.T
    areas = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    suppressed = np.zeros(len(boxes), dtype=bool)
    for pos, i in enumerate(order):
        if suppressed[i]:
            continue
        yield int(i)
        rest = order[pos + 1 :]
        rest = rest[~suppressed[rest]]
        if len(rest) == 0:
            continue
        ix0 = np.maximum(x0[i], x0[rest])
        iy0 = np.maximum(y0[i], y0[rest])
        ix1 = np.minimum(x1[i], x1[rest])
        iy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(0, ix1 - ix0) * np.maximum(0, iy1 - iy0)
        union = areas[i] + areas[rest] - inter
        ov = np.where(union > 0, inter / union, 0)
        suppressed[rest[ov > iou]] = True
