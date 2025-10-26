#!/usr/bin/env python3
import argparse, os, json, shutil
from rich import print

# Starter stub that "passes through" files to georef folder (no-op warp).
# Hook rasterio/pyproj warp here when GCPs available.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    count = 0
    for fn in os.listdir(args.input):
        if fn.lower().endswith((".tif",".tiff",".jpg",".jpeg",".png")):
            src = os.path.join(args.input, fn)
            dst = os.path.join(args.out, os.path.splitext(fn)[0] + "_georef.tif")
            shutil.copyfile(src, dst)
            count += 1
            print(f"[green]→[/] {dst}")
    print(f"[bold]{count}[/] files staged to georef (placeholder).")

if __name__ == "__main__":
    main()