import numpy as np
import pytest

from mineai.detect import DetectionModelError, UltralyticsDetector, detect_sheet
from mineai.raster import tile_grid
from mineai.synthetic import PORTALS, RedBlobDetector, make_sheet


def test_tile_grid_covers_sheet_exactly():
    tiles = tile_grid(3000, 2200, 1024, 256)
    cover = np.zeros((2200, 3000), bool)
    for t in tiles:
        assert t.width == 1024 and t.height == 1024  # never padded
        cover[t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width] = True
    assert cover.all()
    assert tile_grid(500, 400, 1024, 256)[0].width == 500
    with pytest.raises(ValueError):
        tile_grid(100, 100, 256, 256)


@pytest.mark.parametrize("tile,overlap", [(512, 128), (700, 100), (1024, 256), (4000, 0)])
def test_tiled_detection_finds_each_symbol_once(tmp_path, tile, overlap):
    scan = tmp_path / "s.png"
    make_sheet(scan)
    cands = detect_sheet(str(scan), RedBlobDetector(), tile, overlap, iou=0.5)
    assert len(cands) == len(PORTALS)
    got = sorted(tuple(c["center"]) for c in cands)
    want = sorted((cx + 0.5, cy + 0.5) for cx, cy in PORTALS)
    for (gx, gy), (wx, wy) in zip(got, want):
        assert abs(gx - wx) <= 1 and abs(gy - wy) <= 1
    assert all(c["cls_name"] == "portal" for c in cands)


def test_symbol_on_a_tile_seam_is_not_split_or_doubled(tmp_path):
    scan = tmp_path / "s.png"
    # 512/128 tiles start at 0, 384, 768...; put symbols straddling seams and corners.
    pts = [(384, 200), (512, 512), (768, 768), (895, 1150)]
    make_sheet(scan, size=(1600, 1400), portals=pts, labels=[])
    cands = detect_sheet(str(scan), RedBlobDetector(), 512, 128, iou=0.5)
    assert len(cands) == len(pts)
    for c in cands:
        x0, y0, x1, y1 = c["bbox"]
        assert 28 <= x1 - x0 <= 30 and 28 <= y1 - y0 <= 30  # whole symbol, not a fragment


def test_no_dummy_fallback_missing_weights(tmp_path):
    with pytest.raises(DetectionModelError, match="not found"):
        UltralyticsDetector(str(tmp_path / "nope.pt"))


def test_placeholder_weights_rejected(tmp_path):
    p = tmp_path / "mine_entry_yolov8n.pt"
    p.write_text("DUMMY_WEIGHTS_HEADER_NOT_EMPTY")
    with pytest.raises(DetectionModelError, match="placeholder"):
        UltralyticsDetector(str(p))
