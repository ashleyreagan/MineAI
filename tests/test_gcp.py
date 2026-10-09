import numpy as np
import pytest
import rasterio
from pyproj import CRS

from mineai.gcp import GCPError, GCPModel, GCPSet, model_from_record, read_gcp_file
from mineai.georef import warp_sheet
from mineai.synthetic import PORTALS, make_sheet, true_xy, write_qgis_points

UTM17 = CRS.from_epsg(26917)
PTS = [(500, 500), (2500, 500), (500, 1500), (2500, 1500), (1500, 1000), (1000, 2000), (2000, 0), (0, 1000)]


def gcpset(points=PTS, f=true_xy):
    c = np.array([p[0] for p in points], float)
    r = np.array([p[1] for p in points], float)
    x, y = f(c, r)
    return GCPSet([str(i) for i in range(len(c))], c, r, np.asarray(x), np.asarray(y), UTM17, "test")


def test_poly1_recovers_affine_exactly():
    m = GCPModel.fit(gcpset(), "poly1")
    x, y = m.forward([1234.5], [987.25])
    tx, ty = true_xy(1234.5, 987.25)
    assert abs(x[0] - tx) < 1e-6 and abs(y[0] - ty) < 1e-6
    assert m.rmse() < 1e-6


def test_rmse_is_none_with_minimum_points():
    m = GCPModel.fit(gcpset(PTS[:3]), "poly1")
    assert m.rmse() is None  # 3 points always fit an affine perfectly; zero error means nothing


def test_rmse_reflects_bad_gcp():
    g = gcpset()
    g.xs[0] += 30.0  # one GCP clicked 30 m off
    m = GCPModel.fit(g, "poly1")
    dx, dy, kind = m.residuals()
    assert kind == "fit"
    assert np.argmax(np.hypot(dx, dy)) == 0
    assert m.rmse() > 5


def test_poly2_fits_quadratic_warp():
    def quad(c, r):
        x, y = true_xy(c, r)
        return x + 2e-5 * np.asarray(c) ** 2, y - 1e-5 * np.asarray(r) ** 2

    m = GCPModel.fit(gcpset(f=quad), "poly2")
    x, y = m.forward([1800.0], [1300.0])
    tx, ty = quad(1800.0, 1300.0)
    assert abs(x[0] - tx) < 1e-4 and abs(y[0] - ty) < 1e-4


def test_tps_passes_through_gcps_and_reports_leave_one_out():
    m = GCPModel.fit(gcpset(), "tps")
    g = m.gcps
    x, y = m.forward(g.cols, g.rows)
    assert np.allclose(x, g.xs, atol=1e-3) and np.allclose(y, g.ys, atol=1e-3)
    _, _, kind = m.residuals()
    assert kind == "leave-one-out"
    assert m.rmse() is not None


def test_too_few_and_collinear_gcps_rejected():
    with pytest.raises(GCPError):
        GCPModel.fit(gcpset(PTS[:5]), "poly2")
    with pytest.raises(GCPError):
        GCPModel.fit(gcpset([(0, 0), (100, 100), (200, 200), (300, 300)]), "poly1")


def test_qgis_points_negative_y_crs_header_and_disabled_rows(tmp_path):
    p = tmp_path / "s.points"
    write_qgis_points(p, PTS, UTM17.to_wkt())
    lines = p.read_text().splitlines()
    lines.append("1.0,2.0,3.0,-4.0,0,0,0,0")  # disabled row must be ignored
    p.write_text("\n".join(lines) + "\n")
    g = read_gcp_file(p, default_crs=None)
    assert len(g) == len(PTS)
    assert (g.rows >= 0).all()
    assert g.crs.to_epsg() == 26917
    assert GCPModel.fit(g, "poly1").rmse() < 1e-3


def test_csv_needs_columns(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("x,y,z\n1,2,0\n")
    with pytest.raises(GCPError):
        read_gcp_file(p, "EPSG:26917")


def test_model_roundtrips_through_record():
    m = GCPModel.fit(gcpset(), "poly2")
    g = m.gcps
    rec = {"method": "poly2", "crs": UTM17.to_wkt(),
           "gcps": [{"id": i, "col": c, "row": r, "x": x, "y": y} for i, c, r, x, y in zip(g.ids, g.cols, g.rows, g.xs, g.ys)]}
    m2 = model_from_record(rec)
    assert np.allclose(m.forward([10, 2000], [30, 1500]), m2.forward([10, 2000], [30, 1500]))


@pytest.mark.parametrize("method", ["poly1", "poly2", "tps"])
def test_warped_raster_agrees_with_point_projection(tmp_path, method):
    """A symbol drawn on the scan must land where the GCP model says, in the warped GeoTIFF."""
    scan = tmp_path / "s.png"
    make_sheet(scan)
    m = GCPModel.fit(gcpset(), method)
    out = tmp_path / "s_georef.tif"
    info = warp_sheet(str(scan), m, out, "nearest")
    with rasterio.open(out) as ds:
        assert ds.count == 4 and ds.crs.to_epsg() == 26917
        px = abs(ds.transform.a)
        rgb = ds.read([1, 2, 3]).astype(int)
        red = (rgb[0] > 150) & (rgb[1] < 80) & (rgb[2] < 80)
        for cx, cy in PORTALS:
            ex, ey = m.forward([cx + 0.5], [cy + 0.5])
            col, row = rasterio.transform.rowcol(ds.transform, ex[0], ey[0], op=float)[::-1]
            r0, c0 = int(row), int(col)
            win = red[max(0, r0 - 30) : r0 + 30, max(0, c0 - 30) : c0 + 30]
            rr, cc = np.nonzero(win)
            assert len(rr), f"no symbol near expected location for {cx},{cy}"
            got_c = cc.mean() + max(0, c0 - 30) + 0.5
            got_r = rr.mean() + max(0, r0 - 30) + 0.5
            assert abs(got_c - col) < 1.5 and abs(got_r - row) < 1.5, (method, cx, cy, got_c - col, got_r - row)
        alpha = ds.read(4)
        assert alpha.min() == 0 and alpha.max() == 255  # collar is transparent
    assert info["warped_size"][0] > 0
