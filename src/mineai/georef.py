"""Stage 2: georeference each sheet from its ground control points.

For each sheet this writes:
  <sheet>_georef.json  the GCPs, method, CRS, per-GCP residuals and RMSE. This
                       record is what export uses to place detections and text,
                       so OCR and detection never need the warped image.
  <sheet>_georef.tif   (optional) the sheet warped into map coordinates, with an
                       alpha band, for viewing in QGIS/ArcGIS.

A sheet with no GCP file is skipped, unless the scan is already georeferenced
(GeoTIFF tags or a world file), in which case its embedded transform is used.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rich.console import Console

from .gcp import GCPError, GCPModel, find_gcp_file, read_gcp_file
from .manifest import FAILED, OK, SKIPPED, Manifest
from .raster import has_georeferencing, open_raster

console = Console()


def record_path(cfg: dict, sheet_id: str) -> Path:
    return Path(cfg["paths"]["georef"]) / f"{sheet_id}_georef.json"


def load_record(cfg: dict, sheet_id: str) -> dict | None:
    p = record_path(cfg, sheet_id)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(cfg: dict, sheet_id: str, scan_path: str) -> tuple[GCPModel | None, str | None, str]:
    """Return (model, gcp_file, note). model is None when the sheet can't be georeferenced."""
    gcfg = cfg["georef"]
    gfile = find_gcp_file(cfg["paths"]["gcp"], sheet_id)
    if gfile:
        gcps = read_gcp_file(gfile, gcfg.get("default_crs"))
        model = GCPModel.fit(gcps, gcfg["method"])
        return model, str(gfile), f"{len(gcps)} GCPs from {gfile.name} (CRS from {gcps.crs_source})"
    with open_raster(scan_path) as ds:
        if has_georeferencing(ds) and not ds.gcps[0]:
            if ds.crs:
                return GCPModel.from_affine(ds.transform, CRS.from_user_input(ds.crs.to_wkt())), None, "embedded geotransform and CRS"
            if gcfg.get("default_crs"):
                crs = CRS.from_user_input(gcfg["default_crs"])
                return GCPModel.from_affine(ds.transform, crs), None, f"world file; CRS assumed from georef.default_crs ({gcfg['default_crs']})"
            raise GCPError("scan has a world file but no CRS; set georef.default_crs")
    return None, None, f"no GCP file ({sheet_id}.points or {sheet_id}.csv) in {cfg['paths']['gcp']}"


_GDAL_TYPES = {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16", "uint32": "UInt32",
               "int32": "Int32", "float32": "Float32", "float64": "Float64"}


def _source_vrt(src, scan_path: str, model: GCPModel) -> tuple[str, int]:
    """A VRT over the scan that carries its georeferencing and normalises the bands.

    * GDAL only applies GCPs that belong to the source dataset, so they're written
      into the VRT rather than passed to the warper.
    * Palettes are expanded to RGB, 1-bit scans are scaled 0/1 → 0/255, and an
      existing alpha band is dropped (the warp adds its own).
    Nothing is copied: the VRT points at the scan. Returns (xml, band count).
    """
    W, H = src.width, src.height
    fname = escape(str(Path(scan_path).resolve()))
    rect = f'<SrcRect xOff="0" yOff="0" xSize="{W}" ySize="{H}"/><DstRect xOff="0" yOff="0" xSize="{W}" ySize="{H}"/>'
    try:
        paletted = src.count == 1 and bool(src.colormap(1))
    except ValueError:
        paletted = False
    nb = src.tags(1, "IMAGE_STRUCTURE").get("NBITS")
    bilevel = nb is not None and int(nb) == 1 and not paletted

    def source(band: int, extra: str = "", kind: str = "SimpleSource") -> str:
        return (f"<{kind}><SourceFilename relativeToVRT=\"0\">{fname}</SourceFilename>"
                f"<SourceBand>{band}</SourceBand>{rect}{extra}</{kind}>")

    bands = []
    if paletted:
        for i, ci in enumerate(("Red", "Green", "Blue"), 1):
            bands.append(f'<VRTRasterBand dataType="Byte" band="{i}"><ColorInterp>{ci}</ColorInterp>'
                         f'{source(1, f"<ColorTableComponent>{i}</ColorTableComponent>", "ComplexSource")}</VRTRasterBand>')
    elif bilevel:
        bands.append('<VRTRasterBand dataType="Byte" band="1"><ColorInterp>Gray</ColorInterp>'
                     f'{source(1, "<ScaleOffset>0</ScaleOffset><ScaleRatio>255</ScaleRatio>", "ComplexSource")}</VRTRasterBand>')
    else:
        dtype = _GDAL_TYPES.get(src.dtypes[0])
        if dtype is None:
            raise ValueError(f"unsupported pixel type {src.dtypes[0]}")
        n = {2: 1, 4: 3}.get(src.count, src.count)  # drop gray+alpha / RGBA alpha
        for i in range(1, n + 1):
            bands.append(f'<VRTRasterBand dataType="{dtype}" band="{i}">{source(i)}</VRTRasterBand>')

    if model.method == "embedded":
        wkt = escape(model.crs.to_wkt())
        gt = model.affine.to_gdal()
        geo = f"<SRS>{wkt}</SRS><GeoTransform>{', '.join(repr(float(v)) for v in gt)}</GeoTransform>"
    else:
        wkt = escape(model.crs.to_wkt(), {'"': "&quot;"})
        gcps = "".join(
            f'<GCP Id="{escape(str(g.id), {chr(34): "&quot;"})}" Pixel="{g.col!r}" Line="{g.row!r}" X="{g.x!r}" Y="{g.y!r}" Z="0"/>'
            for g in model.to_rasterio_gcps()
        )
        geo = f'<GCPList Projection="{wkt}">{gcps}</GCPList>'
    return f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">{geo}{"".join(bands)}</VRTDataset>', len(bands)


def warp_sheet(scan_path: str, model: GCPModel, out_path: Path, resampling: str = "nearest") -> dict:
    """Warp the scan into map coordinates with GDAL, adding an alpha band for the map collar.

    The warped GeoTIFF is a viewing copy for QGIS/ArcGIS. OCR, detection and export
    all work from the original scan plus the GCP model, not from this file.
    """
    rs = Resampling[resampling]
    crs = rasterio.crs.CRS.from_wkt(model.crs.to_wkt())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vrt_path = out_path.with_suffix(".src.vrt")
    with open_raster(scan_path) as scan:
        xml, _ = _source_vrt(scan, scan_path, model)
    vrt_path.write_text(xml, encoding="utf-8")
    opts = model.warp_options()
    try:
        with open_raster(vrt_path) as src:
            count = src.count
            if model.method == "embedded":
                transform, width, height = calculate_default_transform(crs, crs, src.width, src.height, *src.bounds)
            else:
                transform, width, height = calculate_default_transform(
                    crs, crs, src.width, src.height, gcps=model.to_rasterio_gcps(), **opts)
            profile = {
                "driver": "GTiff", "width": width, "height": height, "count": count + 1,
                "dtype": src.dtypes[0], "crs": crs, "transform": transform, "tiled": True,
                "blockxsize": 512, "blockysize": 512, "compress": "deflate", "BIGTIFF": "IF_SAFER",
            }
            with rasterio.open(out_path, "w", **profile) as dst:
                reproject(
                    source=rasterio.band(src, list(range(1, count + 1))),
                    destination=rasterio.band(dst, list(range(1, count + 2))),
                    dst_transform=transform, dst_crs=crs, dst_alpha=count + 1,
                    resampling=rs, warp_mem_limit=256, **opts,
                )
                dst.colorinterp = _colorinterp(count + 1)
    finally:
        vrt_path.unlink(missing_ok=True)
    return {"warped_path": str(out_path), "warped_size": [width, height], "pixel_size": [transform.a, -transform.e]}


def _colorinterp(count: int):
    from rasterio.enums import ColorInterp as CI

    if count == 2:
        return [CI.gray, CI.alpha]
    if count == 4:
        return [CI.red, CI.green, CI.blue, CI.alpha]
    return [CI.undefined] * (count - 1) + [CI.alpha]


def run(cfg: dict, manifest: Manifest) -> None:
    out_dir = Path(cfg["paths"]["georef"])
    out_dir.mkdir(parents=True, exist_ok=True)
    max_rmse = cfg["georef"].get("max_rmse")
    for sheet_id, sheet in manifest.sheets.items():
        if manifest.status(sheet_id, "ingest") != OK:
            continue
        rec_path = record_path(cfg, sheet_id)
        try:
            model, gfile, note = build_model(cfg, sheet_id, sheet["path"])
        except (GCPError, ValueError, KeyError) as e:
            manifest.set_status(sheet_id, "georef", FAILED, str(e))
            rec_path.unlink(missing_ok=True)
            console.print(f"[red]✖ Georef failed[/] {sheet_id}: {e}")
            continue
        if model is None:
            manifest.set_status(sheet_id, "georef", SKIPPED, note)
            rec_path.unlink(missing_ok=True)
            console.print(f"[yellow]○ Georef skipped[/] {sheet_id}: {note}")
            continue

        rec: dict = {
            "sheet": sheet_id,
            "sheet_sha16": sheet.get("sha16"),
            "method": model.method,
            "crs": model.crs.to_wkt(),
            "crs_name": model.crs.name,
            "crs_epsg": model.crs.to_epsg(),
            "note": note,
            "gcp_file": gfile,
        }
        if model.method == "embedded":
            rec["affine"] = list(model.affine)[:6]
            rec["rmse"] = None
            rec["rmse_kind"] = "unknown (georeferenced outside MineAI)"
        else:
            g = model.gcps
            res = model.residuals()
            rec["n_gcps"] = len(g)
            rec["gcps"] = []
            for i in range(len(g)):
                p = {"id": g.ids[i], "col": float(g.cols[i]), "row": float(g.rows[i]), "x": float(g.xs[i]), "y": float(g.ys[i])}
                if res is not None:
                    dx, dy, _ = res
                    p.update(dx=float(dx[i]), dy=float(dy[i]), residual=float(np.hypot(dx[i], dy[i])))
                rec["gcps"].append(p)
            rec["rmse"] = model.rmse()
            rec["rmse_kind"] = res[2] if res is not None else None
            rec["map_units"] = model.crs.axis_info[0].unit_name if model.crs.axis_info else None

        # Sheet footprint in map coordinates (sampled along the border, so curved warps show).
        w, h = sheet["width"], sheet["height"]
        t = np.linspace(0, 1, 21)
        border_c = np.concatenate([t * w, np.full_like(t, w), (1 - t) * w, np.zeros_like(t)])
        border_r = np.concatenate([np.zeros_like(t), t * h, np.full_like(t, h), (1 - t) * h])
        fx, fy = model.forward(border_c, border_r)
        rec["footprint"] = [[float(x), float(y)] for x, y in zip(fx, fy)]

        try:
            if cfg["georef"].get("write_warped", True):
                rec.update(warp_sheet(sheet["path"], model, out_dir / f"{sheet_id}_georef.tif", cfg["georef"].get("resampling", "nearest")))
        except Exception as e:
            manifest.set_status(sheet_id, "georef", FAILED, f"warp failed: {type(e).__name__}: {e}")
            console.print(f"[red]✖ Warp failed[/] {sheet_id}: {e}")
            continue

        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        rmse = rec.get("rmse")
        rmse_txt = f"RMSE {rmse:.2f} {rec.get('map_units') or 'map units'} ({rec['rmse_kind']})" if rmse is not None else "RMSE n/a"
        flag = ""
        if max_rmse is not None and rmse is not None and rmse > max_rmse:
            flag = f" — exceeds georef.max_rmse ({max_rmse})"
        manifest.set_status(sheet_id, "georef", OK, note + flag, rmse=rmse, method=model.method)
        style = "yellow" if flag else "green"
        console.print(f"[{style}]✔ Georef[/] {sheet_id}: {model.method}, {rmse_txt}{flag}")
