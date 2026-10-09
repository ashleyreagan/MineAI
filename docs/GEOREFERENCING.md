# Georeferencing sheets

MineAI georeferences each sheet from **ground control points (GCPs)**: places
where you know both the pixel position on the scan and the real-world
coordinate. It fits a transform, reports how well the points agree, and
uses that same transform to put detections and OCR text on the map.

Automatic matching against modern basemaps is not implemented. Old mine maps
often use company or local grids, and they show underground workings that
have no counterpart on a topo map, so feature matching is unreliable.
Analyst-picked GCPs are the dependable starting point.

## Option A: QGIS Georeferencer (recommended)

1. In QGIS, open **Layer ▸ Georeferencer** and load the scan.
2. Add GCPs at features you can identify on both the scan and a basemap:
   property corners, grid or coordinate ticks, road and stream crossings,
   railroad junctions, survey monuments, and outcrop lines that match
   the geology.
   - Spread them across the whole sheet, not only one corner.
   - Use more than the minimum (see the table below), so the error estimate means something.
3. Set the target CRS in the transformation settings.
4. **File ▸ Save GCP Points as…** and save as `data/gcp/<sheet name>.points`.
   `<sheet name>` is the scan's file name without the extension.
   QGIS's default `sheet.tif.points` also works.
5. Run `mineai georef` (or `mineai run`).

The `.points` file carries its CRS in its `#CRS:` header, so `georef.default_crs`
is ignored for it. Disabled points (enable = 0) are skipped.

## Option B: a CSV of GCPs

Save as `data/gcp/<sheet name>.csv`. `col`/`row` are pixel positions measured
from the top-left corner of the scan:

```csv
# crs: EPSG:26917
id,col,row,x,y
1,512.5,488.0,585625.17,4476375.41
2,2490.0,505.5,588025.17,4476274.35
...
```

If the `# crs:` line is missing, `georef.default_crs` in `config.yaml` is used.

## Option C: scans that are already georeferenced

A GeoTIFF with a CRS, or a scan with a world file (`.tfw`, `.jgw`, `.pgw`), needs no
GCP file. MineAI uses the embedded transform and records the method as
`embedded`. If a world file has no CRS, MineAI uses `georef.default_crs` and says
so in the manifest. Its RMSE is unknown to MineAI, because the GCPs were picked elsewhere.

## Choosing a method (`georef.method`)

| method | minimum GCPs | use when | error reported |
|---|---|---|---|
| `poly1` (affine) | 3 (use ≥ 6) | most sheets: a flat scan of a sheet drawn to one scale | fit RMSE |
| `poly2` | 6 (use ≥ 10) | moderate paper shrinkage or a scan that isn't square-on | fit RMSE |
| `tps` (thin-plate spline) | 3 (use ≥ 15) | badly distorted linen or mylar, with many well-spread GCPs | leave-one-out RMSE |

A thin-plate spline passes exactly through every GCP, so its fit error is
always zero. MineAI reports **leave-one-out** RMSE instead: each GCP is
predicted from all the others. One bad GCP bends a spline locally without
any warning, so check the per-point residuals in the dashboard.

## Reading the results

- `data/georef/<sheet>_georef.json` holds every GCP with its residual (`dx`, `dy`,
  `residual`) and the overall RMSE in map units. The dashboard shows the same table.
  A single large residual usually means one mis-clicked point.
- `georef.max_rmse` (for example `15`) flags sheets above that RMSE in the run summary.
- `data/georef/<sheet>_georef.tif` is a warped viewing copy with a transparent
  collar, for checking the overlay in QGIS or ArcGIS. OCR and detection don't use it.
  They work on the original scan, and export projects their pixel positions
  through the GCP model.
