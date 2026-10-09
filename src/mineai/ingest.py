"""Stage 1: inventory the scans in the raw folder and record their metadata."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from rich.console import Console

from .manifest import FAILED, OK, Manifest
from .raster import has_georeferencing, list_scans, open_raster, read_dpi, sha16

console = Console()


def scan_metadata(path: Path) -> dict:
    with open_raster(path) as ds:
        try:
            has_cmap = bool(ds.colormap(1))
        except ValueError:
            has_cmap = False
        nb = ds.tags(1, "IMAGE_STRUCTURE").get("NBITS")
        meta = {
            "path": str(path),
            "width": ds.width,
            "height": ds.height,
            "bands": ds.count,
            "dtype": ds.dtypes[0],
            "nbits": int(nb) if nb else None,
            "paletted": has_cmap,
            "driver": ds.driver,
            "georeferenced_input": has_georeferencing(ds),
            "input_crs": ds.crs.to_string() if ds.crs else None,
        }
    dpi_x, dpi_y = read_dpi(path)
    meta.update({"dpi_x": dpi_x, "dpi_y": dpi_y, "megapixels": round(meta["width"] * meta["height"] / 1e6, 1)})
    meta["sha16"] = sha16(path)
    meta["bytes"] = path.stat().st_size
    return meta


def run(cfg: dict, manifest: Manifest) -> None:
    raw = Path(cfg["paths"]["raw"])
    scans = list_scans(raw)
    if not scans:
        manifest.set_stage_error("ingest", f"No scans found in {raw}")
        console.print(f"[red]✖ No scans (.tif/.tiff/.jpg/.jpeg/.png/.jp2) found in[/] {raw}")
        return
    manifest.set_stage_error("ingest", None)

    # Sheet IDs are file stems; two files with the same stem would overwrite each other's outputs.
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for p in scans:
        by_stem[p.stem].append(p)

    unchanged = 0
    for stem, paths in by_stem.items():
        if len(paths) > 1:
            names = ", ".join(p.name for p in paths)
            manifest.set_status(stem, "ingest", FAILED, f"Duplicate sheet name; rename one of: {names}")
            console.print(f"[red]✖ Duplicate sheet name[/] {stem}: {names}")
            continue
        path = paths[0]
        prev = manifest.sheets.get(stem, {})
        st = path.stat()
        if (
            prev.get("path") == str(path)
            and prev.get("bytes") == st.st_size
            and prev.get("mtime") == st.st_mtime
            and manifest.status(stem, "ingest") == OK
        ):
            unchanged += 1
            continue  # unchanged since last run; skip re-hashing a large file
        try:
            meta = scan_metadata(path)
            meta["mtime"] = st.st_mtime
        except Exception as e:  # unreadable/corrupt file
            manifest.set_status(stem, "ingest", FAILED, f"{type(e).__name__}: {e}")
            console.print(f"[red]✖ Could not read[/] {path.name}: {e}")
            continue
        sheet = manifest.sheet(stem)
        if sheet.get("sha16") and sheet["sha16"] != meta["sha16"]:
            # The scan changed since the last run; earlier stage outputs are stale.
            sheet["stages"] = {}
        sheet.update(meta)
        manifest.set_status(stem, "ingest", OK)
        dpi = f" @ {meta['dpi_x']:.0f} dpi" if meta["dpi_x"] else ""
        console.print(f"[green]✔ Ingested[/] {path.name} {meta['width']}×{meta['height']} ({meta['megapixels']} MP){dpi}")

    if unchanged:
        console.print(f"[green]✔ {unchanged} sheet(s) unchanged since the last run[/]")

    # Forget sheets whose files were removed from the raw folder.
    present = set(by_stem)
    for stem in list(manifest.sheets):
        if stem not in present:
            del manifest.sheets[stem]
