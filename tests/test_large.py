"""A sheet bigger than Pillow's decompression-bomb limit must ingest and run tile by tile."""
import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

from mineai.detect import detect_sheet
from mineai.ingest import scan_metadata
from mineai.synthetic import RedBlobDetector

pytestmark = pytest.mark.slow


def test_276_megapixel_sheet(tmp_path):
    # A 36 x 48 inch sheet scanned at 400 dpi.
    w, h = 19200, 14400
    p = tmp_path / "big.tif"
    profile = dict(driver="GTiff", width=w, height=h, count=3, dtype="uint8", tiled=True,
                   blockxsize=512, blockysize=512, compress="deflate")
    with rasterio.open(p, "w", **profile) as ds:
        block = np.full((3, 512, w), 235, np.uint8)
        for r in range(0, h, 512):
            n = min(512, h - r)
            ds.write(block[:, :n], window=Window(0, r, w, n))
        sym = np.zeros((3, 30, 30), np.uint8)
        sym[0] = 200
        ds.write(sym, window=Window(15000, 11000, 30, 30))
    m = scan_metadata(p)
    assert m["megapixels"] > 179
    cands = detect_sheet(str(p), RedBlobDetector(), 2048, 256, 0.5, batch=4)
    assert len(cands) == 1 and cands[0]["center"] == [15015.0, 11015.0]
