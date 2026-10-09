"""Command-line entry point: `mineai <command>`.

  mineai run [--skip ocr detect]   ingest → georef → ocr → detect → export
  mineai ingest|georef|ocr|detect|export
  mineai status                    show the per-sheet status table
  mineai gui                       launch the Streamlit QA dashboard
  mineai dataset ...               build a YOLO training set from labeled sheets
  mineai evaluate ...              score a model on held-out sheets

Exits with status 1 if any stage it ran recorded a failure.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .config import load_config
from .manifest import FAILED, OK, SKIPPED, Manifest

console = Console()
STAGES = ["ingest", "georef", "ocr", "detect", "export"]


def _stage_module(name: str):
    from . import detect, export, georef, ingest, ocr

    return {"ingest": ingest, "georef": georef, "ocr": ocr, "detect": detect, "export": export}[name]


def print_status(manifest: Manifest) -> None:
    table = Table(title="MineAI sheet status", show_lines=False)
    table.add_column("sheet")
    for s in STAGES:
        table.add_column(s)
    style = {OK: "green", SKIPPED: "yellow", FAILED: "red"}
    for sheet_id, sheet in sorted(manifest.sheets.items()):
        cells = []
        for s in STAGES:
            e = sheet.get("stages", {}).get(s)
            if not e:
                cells.append("—")
                continue
            txt = e["status"]
            if s == "georef" and e["status"] == OK and e.get("rmse") is not None:
                txt += f" ({e['rmse']:.2f})"
            if s == "detect" and e["status"] == OK:
                txt += f" ({e.get('n_candidates', 0)})"
            if s == "ocr" and e["status"] == OK:
                txt += f" ({e.get('n_lines', 0)})"
            cells.append(f"[{style.get(e['status'], 'white')}]{txt}[/]")
        table.add_row(sheet_id, *cells)
    console.print(table)


def run_stages(cfg: dict, stages: list[str]) -> int:
    manifest = Manifest(cfg["paths"]["manifest"])
    if stages and stages[0] != "ingest":
        stages = ["ingest"] + stages  # always refresh the inventory first (cheap for unchanged files)
    for name in stages:
        console.rule(f"[bold]{name}")
        _stage_module(name).run(cfg, manifest)
        manifest.save()
    print_status(manifest)
    failures = manifest.failures(stages)
    if failures:
        console.print(f"[red bold]✖ {len(failures)} failure(s):[/]")
        for sheet_id, stage, msg in failures:
            console.print(f"  [red]{stage}[/] {'(all sheets)' if sheet_id == '*' else sheet_id}: {msg}")
        return 1
    skipped = [(sid, s) for sid, sh in manifest.sheets.items() for s, e in sh.get("stages", {}).items() if e["status"] == SKIPPED and s in stages]
    if skipped:
        console.print(f"[yellow]Done with {len(skipped)} skipped step(s); see the table above.[/]")
    else:
        console.print("[green bold]✔ Done[/]")
    console.print(f"Outputs → {cfg['paths']['final']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mineai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default="config.yaml", help="config file (default: ./config.yaml)")
    ap.add_argument("--version", action="version", version=f"mineai {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the whole pipeline")
    p_run.add_argument("--skip", nargs="+", default=[], choices=["georef", "ocr", "detect", "export"], help="stages to skip")
    for s in STAGES:
        sub.add_parser(s, help=f"run the {s} stage (ingest runs first to refresh the inventory)")
    sub.add_parser("status", help="show the per-sheet status table")
    sub.add_parser("gui", help="launch the Streamlit QA dashboard")

    p_ds = sub.add_parser("dataset", help="tile labeled sheets into a YOLO dataset, split by sheet")
    p_ds.add_argument("--images", required=True, help="folder of full-sheet scans")
    p_ds.add_argument("--labels", required=True, help="folder of full-sheet YOLO .txt labels (same stem as scan) + classes.txt")
    p_ds.add_argument("--out", required=True, help="output dataset folder")
    p_ds.add_argument("--tile", type=int, default=1024)
    p_ds.add_argument("--overlap", type=int, default=256)
    p_ds.add_argument("--val", type=float, default=0.2, help="fraction of sheets for validation")
    p_ds.add_argument("--test", type=float, default=0.1, help="fraction of sheets held out for `mineai evaluate`")
    p_ds.add_argument("--empty-ratio", type=float, default=0.25, help="background tiles kept per labeled tile")
    p_ds.add_argument("--seed", type=int, default=0)

    p_ev = sub.add_parser("evaluate", help="per-class precision/recall on held-out sheets")
    p_ev.add_argument("--weights", required=True)
    p_ev.add_argument("--dataset", required=True, help="dataset folder made by `mineai dataset` (uses its test split)")
    p_ev.add_argument("--split", default="test", choices=["test", "val"])
    p_ev.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a match")
    p_ev.add_argument("--out", default=None, help="write the report JSON here")

    args = ap.parse_args(argv)

    if args.cmd == "gui":
        app = Path(__file__).parent / "gui" / "app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--", "--config", str(Path(args.config).resolve())])
    if args.cmd == "dataset":
        from .dataset import build_dataset

        build_dataset(args.images, args.labels, args.out, args.tile, args.overlap, args.val, args.test, args.empty_ratio, args.seed)
        return 0
    if args.cmd == "evaluate":
        from .dataset import evaluate_cli

        cfg = load_config(args.config)
        return evaluate_cli(cfg, args.weights, args.dataset, args.split, args.iou, args.out)

    cfg = load_config(args.config)
    if args.cmd == "status":
        print_status(Manifest(cfg["paths"]["manifest"]))
        return 0
    if args.cmd == "run":
        return run_stages(cfg, [s for s in STAGES if s not in args.skip])
    return run_stages(cfg, [args.cmd])


if __name__ == "__main__":
    sys.exit(main())
