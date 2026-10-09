import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mineai.ocr import _rotate_back, ocr_sheet

from conftest import needs_tesseract

OCFG = {"lang": "eng", "psm": 11, "tile_size": 900, "overlap": 450, "min_conf": 40, "rotations": [0]}


def test_rotate_back_inverts_numpy_rot90():
    w, h = 7, 5
    img = np.zeros((h, w))
    img[1, 4] = 1  # pixel centre at x=4.5, y=1.5
    for k in range(4):
        rot = np.rot90(img, k)
        r, c = np.argwhere(rot == 1)[0]
        x, y = _rotate_back(k, w, h, np.array([c + 0.5]), np.array([r + 0.5]))
        assert (x[0], y[0]) == (4.5, 1.5)


def _sheet(path, rotated=False):
    img = Image.new("RGB", (2400, 1400), (240, 235, 220))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=48)
    d.text((820, 300), "UPPER FREEPORT", fill=(0, 0, 0), font=font)  # crosses the 900/450 tile seams
    d.text((300, 1100), "MAIN ENTRY", fill=(0, 0, 0), font=font)
    if rotated:
        lab = Image.new("RGB", (420, 70), (240, 235, 220))
        ImageDraw.Draw(lab).text((5, 5), "BACK HEADING", fill=(0, 0, 0), font=font)
        img.paste(lab.rotate(90, expand=True), (2000, 500))  # reads bottom-to-top
    img.save(path)


@needs_tesseract
def test_ocr_finds_labels_once_across_overlapping_tiles(tmp_path):
    p = tmp_path / "s.png"
    _sheet(p)
    lines = ocr_sheet(str(p), OCFG)
    texts = [l["text"] for l in lines]
    assert sum("FREEPORT" in t for t in texts) == 1, texts
    assert any("MAIN" in t and "ENTRY" in t for t in texts)
    up = next(l for l in lines if "FREEPORT" in l["text"])
    assert abs(up["bbox"][0] - 820) < 15 and abs(up["bbox"][1] - 310) < 25


@needs_tesseract
def test_rotated_labels_found_only_when_requested(tmp_path):
    p = tmp_path / "s.png"
    _sheet(p, rotated=True)
    plain = " ".join(l["text"] for l in ocr_sheet(str(p), {**OCFG, "tile_size": 2400, "overlap": 0}))
    assert "HEADING" not in plain
    lines = ocr_sheet(str(p), {**OCFG, "tile_size": 2400, "overlap": 0, "rotations": [0, 90]})
    hit = [l for l in lines if "HEADING" in l["text"]]
    assert hit and hit[0]["rotation"] == 90
    x0, y0, x1, y1 = hit[0]["bbox"]
    assert 1990 <= x0 and x1 <= 2080 and 490 <= y0 and y1 <= 930  # box is in sheet coordinates, upright
