#!/usr/bin/env python3
"""Regenerate the synthetic sample sheet and its GCP file used by the quickstart."""
from pathlib import Path

from mineai.synthetic import make_sheet, write_gcps_csv

root = Path(__file__).resolve().parent.parent
make_sheet(root / "data/raw/sample_sheet.png")
write_gcps_csv(root / "data/gcp/sample_sheet.csv")
print("Wrote data/raw/sample_sheet.png and data/gcp/sample_sheet.csv")
