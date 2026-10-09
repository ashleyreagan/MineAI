"""Ground control points: reading them, fitting a pixel→map model, and reporting error.

Pixel convention (same as GDAL and QGIS): (col=0, row=0) is the top-left corner of
the top-left pixel; the centre of that pixel is (0.5, 0.5).

Supported GCP files, looked up by sheet name in the GCP folder:
  <sheet>.points  QGIS Georeferencer export. CRS read from its "#CRS:" header.
  <sheet>.csv     Columns col,row,x,y (optional id). CRS from georef.default_crs,
                  or from a "# crs: EPSG:xxxx" first line.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pyproj import CRS
from rasterio.control import GroundControlPoint
from rasterio.transform import Affine, GCPTransformer

METHODS = ("poly1", "poly2", "tps", "embedded")
MIN_GCPS = {"poly1": 3, "poly2": 6, "tps": 3}


class GCPError(ValueError):
    pass


@dataclass
class GCPSet:
    ids: list[str]
    cols: np.ndarray
    rows: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    crs: CRS
    source: str
    crs_source: str = ""

    def __len__(self) -> int:
        return len(self.ids)


def find_gcp_file(gcp_dir: str | Path, sheet_id: str) -> Path | None:
    d = Path(gcp_dir)
    for ext in (".points", ".csv"):
        p = d / f"{sheet_id}{ext}"
        if p.exists():
            return p
    # QGIS names exports after the full raster file name, e.g. sheet.tif.points
    for p in sorted(d.glob(f"{sheet_id}.*.points")):
        return p
    return None


def read_gcp_file(path: str | Path, default_crs: str | None) -> GCPSet:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".points":
        return _read_qgis_points(text, path, default_crs)
    return _read_csv(text, path, default_crs)


def _read_qgis_points(text: str, path: Path, default_crs: str | None) -> GCPSet:
    crs_txt, body = None, []
    for line in text.splitlines():
        if line.startswith("#CRS:"):
            crs_txt = line[len("#CRS:") :].strip()
        elif line.strip() and not line.startswith("#"):
            body.append(line)
    if not body:
        raise GCPError(f"{path.name}: no GCP rows")
    reader = csv.DictReader(io.StringIO("\n".join(body)))
    fields = {f.strip(): f for f in reader.fieldnames or []}
    sx = fields.get("sourceX") or fields.get("pixelX")
    sy = fields.get("sourceY") or fields.get("pixelY")
    if not sx or not sy or "mapX" not in fields or "mapY" not in fields:
        raise GCPError(f"{path.name}: expected QGIS columns mapX,mapY,sourceX,sourceY")
    ids, cols, rows, xs, ys = [], [], [], [], []
    for i, rec in enumerate(reader):
        enable = rec.get(fields.get("enable", "enable"), "1")
        if str(enable).strip() in ("0", "false", "False"):
            continue
        ids.append(str(i + 1))
        cols.append(float(rec[sx]))
        rows.append(float(rec[sy]))
        xs.append(float(rec[fields["mapX"]]))
        ys.append(float(rec[fields["mapY"]]))
    rows_arr = np.asarray(rows)
    # QGIS stores pixel Y as a negative number for rasters with no geotransform.
    if len(rows_arr) and np.all(rows_arr <= 0):
        rows_arr = -rows_arr
    if crs_txt:
        crs, crs_source = CRS.from_user_input(crs_txt), "QGIS .points header"
    elif default_crs:
        crs, crs_source = CRS.from_user_input(default_crs), "georef.default_crs"
    else:
        raise GCPError(f"{path.name}: no #CRS header and no georef.default_crs set")
    return GCPSet(ids, np.asarray(cols), rows_arr, np.asarray(xs), np.asarray(ys), crs, str(path), crs_source)


def _read_csv(text: str, path: Path, default_crs: str | None) -> GCPSet:
    lines = text.splitlines()
    crs_txt = None
    while lines and lines[0].lstrip().startswith("#"):
        head = lines.pop(0).lstrip("# ").strip()
        if head.lower().startswith("crs:"):
            crs_txt = head[4:].strip()
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    cols_l = {c.strip().lower(): c for c in reader.fieldnames or []}
    missing = [k for k in ("col", "row", "x", "y") if k not in cols_l]
    if missing:
        raise GCPError(f"{path.name}: missing column(s) {missing}; expected col,row,x,y")
    ids, cols, rows, xs, ys = [], [], [], [], []
    for i, rec in enumerate(reader):
        if not any((v or "").strip() for v in rec.values()):
            continue
        ids.append(str(rec.get(cols_l.get("id", ""), "") or i + 1))
        cols.append(float(rec[cols_l["col"]]))
        rows.append(float(rec[cols_l["row"]]))
        xs.append(float(rec[cols_l["x"]]))
        ys.append(float(rec[cols_l["y"]]))
    if crs_txt:
        crs, crs_source = CRS.from_user_input(crs_txt), "CSV header"
    elif default_crs:
        crs, crs_source = CRS.from_user_input(default_crs), "georef.default_crs"
    else:
        raise GCPError(f"{path.name}: no '# crs:' line and no georef.default_crs set")
    return GCPSet(ids, np.asarray(cols), np.asarray(rows), np.asarray(xs), np.asarray(ys), crs, str(path), crs_source)


def _terms(c: np.ndarray, r: np.ndarray, order: int) -> np.ndarray:
    if order == 1:
        return np.column_stack([np.ones_like(c), c, r])
    return np.column_stack([np.ones_like(c), c, r, c * c, c * r, r * r])


@dataclass
class GCPModel:
    """Pixel → map transform fitted from GCPs (or taken from an already-georeferenced file).

    poly1/poly2 are least-squares polynomials (the same models GDAL uses for
    MAX_GCP_ORDER=1/2). tps is GDAL's thin-plate spline, which passes exactly
    through every GCP, so its error is reported as leave-one-out RMSE instead.
    """

    method: str
    crs: CRS
    gcps: GCPSet | None = None
    affine: Affine | None = None
    _coef: tuple | None = field(default=None, repr=False)
    _norm: tuple | None = field(default=None, repr=False)
    _tps: GCPTransformer | None = field(default=None, repr=False)

    @classmethod
    def fit(cls, gcps: GCPSet, method: str) -> "GCPModel":
        if method not in MIN_GCPS:
            raise GCPError(f"Unknown georef method {method!r}; use poly1, poly2 or tps")
        n, need = len(gcps), MIN_GCPS[method]
        if n < need:
            raise GCPError(f"{method} needs at least {need} GCPs; {Path(gcps.source).name} has {n}")
        m = cls(method=method, crs=gcps.crs, gcps=gcps)
        if method == "tps":
            m._tps = GCPTransformer(_to_rio(gcps), tps=True)
        else:
            order = 1 if method == "poly1" else 2
            cm, cs = gcps.cols.mean(), gcps.cols.std() or 1.0
            rm, rs = gcps.rows.mean(), gcps.rows.std() or 1.0
            xm, ym = gcps.xs.mean(), gcps.ys.mean()
            A = _terms((gcps.cols - cm) / cs, (gcps.rows - rm) / rs, order)
            if np.linalg.matrix_rank(A) < A.shape[1]:
                raise GCPError("GCPs are collinear or duplicated; spread them across the sheet")
            cx, *_ = np.linalg.lstsq(A, gcps.xs - xm, rcond=None)
            cy, *_ = np.linalg.lstsq(A, gcps.ys - ym, rcond=None)
            m._coef = (order, cx, cy)
            m._norm = (cm, cs, rm, rs, xm, ym)
        return m

    @classmethod
    def from_affine(cls, affine: Affine, crs: CRS) -> "GCPModel":
        return cls(method="embedded", crs=crs, affine=affine)

    def forward(self, cols, rows) -> tuple[np.ndarray, np.ndarray]:
        cols = np.atleast_1d(np.asarray(cols, dtype=float))
        rows = np.atleast_1d(np.asarray(rows, dtype=float))
        if self.method == "embedded":
            t = self.affine
            return t.a * cols + t.b * rows + t.c, t.d * cols + t.e * rows + t.f
        if self.method == "tps":
            xs, ys = self._tps.xy(rows, cols, offset="ul")
            return np.atleast_1d(np.asarray(xs, float)), np.atleast_1d(np.asarray(ys, float))
        order, cx, cy = self._coef
        cm, cs, rm, rs, xm, ym = self._norm
        A = _terms((cols - cm) / cs, (rows - rm) / rs, order)
        return A @ cx + xm, A @ cy + ym

    def residuals(self) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Per-GCP (dx, dy) in map units and what kind of residual they are."""
        g = self.gcps
        if g is None:
            return None
        if self.method == "tps":
            if len(g) < 4:
                return None
            dx, dy = np.zeros(len(g)), np.zeros(len(g))
            for i in range(len(g)):
                keep = np.arange(len(g)) != i
                sub = GCPSet([g.ids[j] for j in np.flatnonzero(keep)], g.cols[keep], g.rows[keep], g.xs[keep], g.ys[keep], g.crs, g.source)
                t = GCPTransformer(_to_rio(sub), tps=True)
                px, py = t.xy([g.rows[i]], [g.cols[i]], offset="ul")
                dx[i] = np.asarray(px).ravel()[0] - g.xs[i]
                dy[i] = np.asarray(py).ravel()[0] - g.ys[i]
            return dx, dy, "leave-one-out"
        px, py = self.forward(g.cols, g.rows)
        return px - g.xs, py - g.ys, "fit"

    def rmse(self) -> float | None:
        res = self.residuals()
        if res is None:
            return None
        dx, dy, kind = res
        g = self.gcps
        # With exactly the minimum number of points a polynomial fits perfectly; the
        # zero residual says nothing about accuracy.
        if kind == "fit" and len(g) <= MIN_GCPS[self.method]:
            return None
        return float(np.sqrt(np.mean(dx**2 + dy**2)))

    def to_rasterio_gcps(self) -> list[GroundControlPoint]:
        return _to_rio(self.gcps)

    def warp_options(self) -> dict:
        if self.method == "poly1":
            return {"MAX_GCP_ORDER": 1}
        if self.method == "poly2":
            return {"MAX_GCP_ORDER": 2}
        if self.method == "tps":
            return {"SRC_METHOD": "GCP_TPS"}
        return {}


def _to_rio(g: GCPSet) -> list[GroundControlPoint]:
    # z must be a number: rasterio serialises z=None as the string "None", which
    # makes GDAL silently reject every GCP.
    return [
        GroundControlPoint(row=float(r), col=float(c), x=float(x), y=float(y), z=0.0, id=str(i))
        for i, c, r, x, y in zip(g.ids, g.cols, g.rows, g.xs, g.ys)
    ]


def model_from_record(rec: dict) -> GCPModel:
    """Rebuild the exact model a georef record describes (used by export and the GUI)."""
    crs = CRS.from_user_input(rec["crs"])
    if rec["method"] == "embedded":
        return GCPModel.from_affine(Affine(*rec["affine"]), crs)
    g = rec["gcps"]
    gs = GCPSet(
        [p["id"] for p in g],
        np.array([p["col"] for p in g], float),
        np.array([p["row"] for p in g], float),
        np.array([p["x"] for p in g], float),
        np.array([p["y"] for p in g], float),
        crs,
        rec.get("gcp_file", ""),
    )
    return GCPModel.fit(gs, rec["method"])
