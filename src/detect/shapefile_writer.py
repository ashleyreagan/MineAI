#!/usr/bin/env python3
import argparse, os, json, geopandas as gpd
from shapely.geometry import shape
from rich import print

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for fn in os.listdir(args.input):
        if fn.endswith("_detect.geojson"):
            p = os.path.join(args.input, fn)
            gj = json.load(open(p))
            feats = gj.get("features", [])
            if not feats:
                print(f"[yellow]No features in[/] {fn}")
                continue
            gdf = gpd.GeoDataFrame([f["properties"] for f in feats],
                                   geometry=[shape(f["geometry"]) for f in feats],
                                   crs="EPSG:26917" if gj.get("crs","").startswith("EPSG:26917") else None)
            shp = os.path.join(args.out, os.path.splitext(fn)[0] + ".shp")
            gdf.to_file(shp)
            print(f"[green]📦 Wrote[/] {shp}")

if __name__ == "__main__":
    main()