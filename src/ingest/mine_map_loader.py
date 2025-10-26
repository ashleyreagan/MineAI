#!/usr/bin/env python3
import argparse, os, json, hashlib
from PIL import Image, ImageOps
from rich import print

def image_metadata(path):
    with Image.open(path) as im:
        dpi = im.info.get("dpi", (None, None))
        mode = im.mode; size = im.size; fmt = im.format
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    return {"path": path, "width": size[0], "height": size[1], "mode": mode, "format": fmt, "dpi_x": dpi[0], "dpi_y": dpi[1], "sha16": sha}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()
    in_dir = args.input
    os.makedirs("data/ocr", exist_ok=True)
    manifest = []
    for fn in os.listdir(in_dir):
        if fn.lower().endswith((".tif",".tiff",".jpg",".jpeg",".png")):
            p = os.path.join(in_dir, fn)
            try:
                meta = image_metadata(p)
                manifest.append(meta)
                print(f"[green]✔ Ingested[/] {fn} {meta['width']}x{meta['height']}")
            except Exception as e:
                print(f"[red]✖ Failed[/] {fn}: {e}")
    out = "data/ocr/ingest_manifest.json"
    with open(out, "w") as f: json.dump(manifest, f, indent=2)
    print("[bold]Manifest[/] →", out)

if __name__ == "__main__":
    main()