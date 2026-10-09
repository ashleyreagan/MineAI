#!/usr/bin/env python3
"""Backward-compatible wrapper: same as `mineai run`. Prefer `pip install -e .` then `mineai run`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mineai.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["--config", str(Path(__file__).resolve().parent / "config.yaml"), "run", *sys.argv[1:]]))
