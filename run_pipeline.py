#!/usr/bin/env python3
from utils.config_loader import load_config
import subprocess, sys

def run(cmd):
    print("▶", " ".join(cmd)); sys.stdout.flush()
    return subprocess.run(cmd, check=False)

def main():
    cfg = load_config("config.yaml")
    print("🚀 MineAI pipeline starting")
    stages = [
        ["python", "src/ingest/mine_map_loader.py", "--input", cfg["paths"]["raw"]],
        ["python", "src/georef/georef_pipeline.py", "--input", cfg["paths"]["raw"], "--out", cfg["paths"]["georef"]],
        ["python", "src/ocr/text_extractor.py", "--input", cfg["paths"]["georef"], "--out", cfg["paths"]["ocr"]],
        ["python", "src/detect/yolo_detect_entries.py", "--input", cfg["paths"]["georef"], "--out", cfg["paths"]["detect"], "--weights", cfg["paths"]["yolo_weights"]],
        ["python", "src/detect/shapefile_writer.py", "--input", cfg["paths"]["detect"], "--out", cfg["paths"]["final"]],
    ]
    ok = True
    for step in stages:
        r = run(step)
        ok &= (r.returncode == 0)
    print("✅ Done" if ok else "❌ One or more stages failed")
    if ok:
        print("Outputs →", cfg["paths"]["final"])

if __name__ == "__main__":
    main()