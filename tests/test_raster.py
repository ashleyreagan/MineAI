import numpy as np
import pytest
import rasterio
from PIL import Image

from mineai.ingest import scan_metadata
from mineai.raster import open_raster, read_rgb


def test_paletted_png_is_expanded(tmp_path):
    p = tmp_path / "pal.png"
    im = Image.new("P", (50, 40), 0)
    im.putpalette([255, 255, 255, 200, 0, 0] + [0] * 762)
    im.paste(1, (10, 10, 20, 20))
    im.save(p)
    with open_raster(p) as ds:
        rgb = read_rgb(ds)
    assert rgb.shape == (40, 50, 3)
    assert tuple(rgb[15, 15]) == (200, 0, 0) and tuple(rgb[0, 0]) == (255, 255, 255)
    assert scan_metadata(p)["paletted"]


def test_one_bit_tiff_is_scaled_to_full_range(tmp_path):
    p = tmp_path / "bilevel.tif"
    a = np.zeros((32, 32), np.uint8)
    a[8:16, 8:16] = 1
    with rasterio.open(p, "w", driver="GTiff", width=32, height=32, count=1, dtype="uint8", nbits=1) as ds:
        ds.write(a, 1)
    with open_raster(p) as ds:
        rgb = read_rgb(ds)
    assert rgb.max() == 255 and rgb[10, 10, 0] == 255 and rgb[0, 0, 0] == 0
    assert scan_metadata(p)["nbits"] == 1


def test_16bit_and_rgba(tmp_path):
    p16 = tmp_path / "g16.tif"
    a = (np.arange(64 * 64).reshape(64, 64) * 10).astype(np.uint16)
    with rasterio.open(p16, "w", driver="GTiff", width=64, height=64, count=1, dtype="uint16") as ds:
        ds.write(a, 1)
    with open_raster(p16) as ds:
        rgb = read_rgb(ds)
    assert rgb.dtype == np.uint8 and rgb.max() == 255 and rgb.min() == 0

    p4 = tmp_path / "rgba.png"
    Image.new("RGBA", (10, 10), (10, 20, 30, 0)).save(p4)
    with open_raster(p4) as ds:
        rgb = read_rgb(ds)
    assert rgb.shape == (10, 10, 3) and tuple(rgb[0, 0]) == (10, 20, 30)  # alpha dropped, not read as a band


def test_metadata_reads_dpi_and_hash(tmp_path):
    p = tmp_path / "s.png"
    Image.new("L", (20, 10), 255).save(p, dpi=(400, 400))
    m = scan_metadata(p)
    assert m["dpi_x"] == pytest.approx(400, abs=1) and len(m["sha16"]) == 16
    assert m["width"] == 20 and m["height"] == 10 and not m["georeferenced_input"]
