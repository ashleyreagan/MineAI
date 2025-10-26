#!/usr/bin/env python3
import argparse, os, json
from PIL import Image
import pytesseract
from rich import print

def ocr_image(path):
    img = Image.open(path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        if not txt: continue
        words.append({"text": txt, "conf": data["conf"][i], "bbox": [data["left"][i], data["top"][i], data["width"][i], data["height"][i]]})
    return {"path": path, "words": words}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = []
    for fn in os.listdir(args.input):
        if fn.endswith("_georef.tif"):
            p = os.path.join(args.input, fn)
            try:
                res = ocr_image(p)
                out_json = os.path.join(args.out, os.path.splitext(fn)[0] + "_ocr.json")
                with open(out_json, "w") as f: json.dump(res, f, indent=2)
                results.append(out_json)
                print(f"[green]📝 OCR[/] {fn} → {out_json}")
            except Exception as e:
                print(f"[red]✖ OCR failed[/] {fn}: {e}")
    print("[bold]Done[/]. Files:", len(results))

if __name__ == "__main__":
    main()