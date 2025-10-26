#!/usr/bin/env python3
import argparse, os, json, sys
from rich import print
try:
    from ultralytics import YOLO
    UL_OK = True
except Exception as e:
    print("[yellow]YOLO not available, using dummy detector[/]"); UL_OK = False

def dummy_detect(path):
    # Returns a trivial GeoJSON point at image center to prove wiring.
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    import numpy as np
    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        tfm: Affine = ds.transform
        cx, cy = tfm * (w/2, h/2)
        crs = ds.crs.to_string() if ds.crs else "EPSG:4326"
    feat = {"type":"Feature","properties":{"cls":"portal","score":1.0},"geometry":{"type":"Point","coordinates":[cx, cy]}}
    return {"type":"FeatureCollection","features":[feat],"crs":crs}

def yolo_detect(weights, path):
    model = YOLO(weights)
    res = model.predict(source=path, verbose=False, imgsz=640)
    # Convert first image prediction to simple GeoJSON in pixel space (placeholder).
    # Real implementation should map pixels → map coords from georef.
    feats = []
    for r in res:
        for b in r.boxes:
            x,y,w,h = b.xywh[0].tolist()
            feats.append({"type":"Feature","properties":{"cls":int(b.cls[0]),"score":float(b.conf[0])},
                          "geometry":{"type":"Point","coordinates":[x,y]}})
    return {"type":"FeatureCollection","features":feats,"crs":"pixel"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=False, default="models/yolo/mine_entry_yolov8n.pt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for fn in os.listdir(args.input):
        if fn.endswith("_georef.tif"):
            p = os.path.join(args.input, fn)
            out_geojson = os.path.join(args.out, os.path.splitext(fn)[0] + "_detect.geojson")
            try:
                gj = yolo_detect(args.weights, p) if UL_OK else dummy_detect(p)
                with open(out_geojson, "w") as f: json.dump(gj, f, indent=2)
                print(f"[green]🔎 Detect[/] {fn} → {out_geojson}")
            except Exception as e:
                print(f"[red]✖ Detect failed[/] {fn}: {e}")

if __name__ == "__main__":
    main()