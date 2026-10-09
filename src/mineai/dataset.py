"""Training-data and evaluation tools for the mine-entry detector.

`mineai dataset` turns full-sheet labels into a YOLO training set:
  * Labels are YOLO .txt files covering the WHOLE sheet (class cx cy w h, normalised
    0–1 to the sheet), as exported by CVAT, Label Studio, Roboflow, etc., with the
    same file stem as the scan, plus a classes.txt (one class name per line).
    An empty .txt means "labeled, nothing on this sheet"; a missing .txt means
    "not labeled yet" and the sheet is left out.
  * Sheets, not tiles, are split into train / val / test. Tiles from one sheet
    share paper, ink, drafting hand and symbol style, so splitting tiles at random
    would leak near-duplicates into validation and overstate accuracy.
  * Train/val sheets are cut into overlapping tiles at native resolution (the same
    tiling detection uses). Test sheets are left whole: `mineai evaluate` runs the
    real tiled pipeline on them, so its numbers describe what the pipeline does.

`mineai evaluate` reports per-class precision, recall and F1 on held-out sheets.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

from .raster import list_scans, open_raster, read_tile, tile_grid

console = Console()


def read_classes(labels_dir: Path) -> list[str]:
    p = labels_dir / "classes.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: list one class name per line, in class-id order")
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_sheet_labels(label_path: Path, width: int, height: int) -> np.ndarray:
    """Full-sheet YOLO labels → (N, 5) array of cls, x0, y0, x1, y1 in sheet pixels."""
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
        rows.append([c, (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return np.array(rows, float).reshape(-1, 5)


def split_sheets(stems: list[str], val: float, test: float, seed: int) -> dict[str, list[str]]:
    stems = sorted(stems)
    random.Random(seed).shuffle(stems)
    n = len(stems)
    n_test = int(round(n * test)) if test > 0 else 0
    n_val = int(round(n * val)) if val > 0 else 0
    if n >= 3:
        n_test = max(n_test, 1 if test > 0 else 0)
        n_val = max(n_val, 1 if val > 0 else 0)
    n_test = min(n_test, max(0, n - 1))
    n_val = min(n_val, max(0, n - 1 - n_test))
    return {"test": sorted(stems[:n_test]), "val": sorted(stems[n_test : n_test + n_val]), "train": sorted(stems[n_test + n_val :])}


def build_dataset(images: str, labels: str, out: str, tile: int = 1024, overlap: int = 256,
                  val: float = 0.2, test: float = 0.1, empty_ratio: float = 0.25, seed: int = 0,
                  min_visible: float = 0.6) -> dict:
    images_dir, labels_dir, out_dir = Path(images), Path(labels), Path(out)
    names = read_classes(labels_dir)
    scans = {p.stem: p for p in list_scans(images_dir)}
    labeled = {s: labels_dir / f"{s}.txt" for s in scans if (labels_dir / f"{s}.txt").exists()}
    unlabeled = sorted(set(scans) - set(labeled))
    if unlabeled:
        console.print(f"[yellow]○ {len(unlabeled)} sheet(s) have no label file and are left out:[/] {', '.join(unlabeled[:10])}{' …' if len(unlabeled) > 10 else ''}")
    if not labeled:
        raise SystemExit(f"No labeled sheets: put <sheet>.txt files in {labels_dir}")
    splits = split_sheets(list(labeled), val, test, seed)
    rng = random.Random(seed)
    counts = {k: {"sheets": len(v), "tiles": 0, "boxes": 0} for k, v in splits.items()}

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem in splits[split]:
            with open_raster(scans[stem]) as ds:
                gt = read_sheet_labels(labeled[stem], ds.width, ds.height)
                bad = gt[(gt[:, 0] < 0) | (gt[:, 0] >= len(names))]
                if len(bad):
                    raise SystemExit(f"{labeled[stem]}: class id(s) {sorted(set(bad[:, 0].astype(int)))} not in classes.txt")
                tiles = tile_grid(ds.width, ds.height, tile, overlap)
                with_boxes, empty = [], []
                for t in tiles:
                    tb = _boxes_in_tile(gt, t, min_visible)
                    (with_boxes if len(tb) else empty).append((t, tb))
                n_empty = min(len(empty), int(round(len(with_boxes) * empty_ratio))) if with_boxes else min(len(empty), 2)
                chosen = with_boxes + rng.sample(empty, n_empty)
                for t, tb in chosen:
                    name = f"{stem}_r{t.row_off}_c{t.col_off}"
                    Image.fromarray(read_tile(ds, t)).save(out_dir / "images" / split / f"{name}.png")
                    with open(out_dir / "labels" / split / f"{name}.txt", "w", encoding="utf-8") as f:
                        for c, x0, y0, x1, y1 in tb:
                            f.write(f"{int(c)} {(x0 + x1) / 2 / t.width:.6f} {(y0 + y1) / 2 / t.height:.6f} "
                                    f"{(x1 - x0) / t.width:.6f} {(y1 - y0) / t.height:.6f}\n")
                    counts[split]["tiles"] += 1
                    counts[split]["boxes"] += len(tb)

    val_dir = "images/val" if splits["val"] else "images/train"
    if not splits["val"]:
        console.print("[yellow]○ Too few sheets for a validation split; validating on train (numbers will be optimistic).[/]")
    data_yaml = out_dir / "data.yaml"
    lines = [f"path: {out_dir.resolve()}", "train: images/train", f"val: {val_dir}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "names": names,
        "splits": splits,
        "images_dir": str(images_dir.resolve()),
        "labels_dir": str(labels_dir.resolve()),
        "tile": tile,
        "overlap": overlap,
        "seed": seed,
        "counts": counts,
    }
    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    for k in ("train", "val", "test"):
        c = counts[k]
        extra = f", {c['tiles']} tiles, {c['boxes']} boxes" if k != "test" else " (kept whole for `mineai evaluate`)"
        console.print(f"[green]✔ {k}[/]: {c['sheets']} sheets{extra}")
    console.print(f"Train with: yolo detect train data={data_yaml} model=yolov8s.pt imgsz={tile}")
    return meta


def _boxes_in_tile(gt: np.ndarray, t, min_visible: float) -> np.ndarray:
    if not len(gt):
        return gt
    x0 = np.clip(gt[:, 1] - t.col_off, 0, t.width)
    y0 = np.clip(gt[:, 2] - t.row_off, 0, t.height)
    x1 = np.clip(gt[:, 3] - t.col_off, 0, t.width)
    y1 = np.clip(gt[:, 4] - t.row_off, 0, t.height)
    vis = (x1 - x0) * (y1 - y0)
    full = (gt[:, 3] - gt[:, 1]) * (gt[:, 4] - gt[:, 2])
    keep = (vis > 0) & (vis >= min_visible * np.maximum(full, 1e-9))
    return np.column_stack([gt[keep, 0], x0[keep], y0[keep], x1[keep], y1[keep]])


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def match(pred: list[dict], gt: np.ndarray, iou_thr: float) -> tuple[dict, dict, dict]:
    """Greedy one-to-one matching by score within each class. Returns per-class TP/FP/FN."""
    tp, fp, fn = {}, {}, {}
    classes = set(int(c) for c in gt[:, 0]) | {p["cls_id"] for p in pred}
    for c in classes:
        g = gt[gt[:, 0] == c][:, 1:5]
        ps = sorted([p for p in pred if p["cls_id"] == c], key=lambda p: -p["score"])
        used = np.zeros(len(g), bool)
        t = f = 0
        if len(ps) and len(g):
            ious = _iou(np.array([p["bbox"] for p in ps], float), g)
        for i in range(len(ps)):
            if len(g):
                cand = np.where(used, -1, ious[i])
                j = int(np.argmax(cand))
                if cand[j] >= iou_thr:
                    used[j] = True
                    t += 1
                    continue
            f += 1
        tp[c], fp[c], fn[c] = t, f, int((~used).sum())
    return tp, fp, fn


def evaluate(dataset: str, detector, tile: int, overlap: int, nms_iou: float, split: str = "test", iou_thr: float = 0.5) -> dict:
    from .detect import detect_sheet

    meta = json.loads((Path(dataset) / "splits.json").read_text(encoding="utf-8"))
    names = meta["names"]
    sheets = meta["splits"][split]
    if not sheets:
        raise SystemExit(f"The {split} split has no sheets; rebuild the dataset with more labeled sheets or a larger --{split} fraction.")
    images_dir, labels_dir = Path(meta["images_dir"]), Path(meta["labels_dir"])
    scans = {p.stem: p for p in list_scans(images_dir)}
    tot = {c: [0, 0, 0] for c in range(len(names))}
    per_sheet = {}
    for stem in sheets:
        with open_raster(scans[stem]) as ds:
            gt = read_sheet_labels(labels_dir / f"{stem}.txt", ds.width, ds.height)
        pred = detect_sheet(str(scans[stem]), detector, tile, overlap, nms_iou)
        tp, fp, fn = match(pred, gt, iou_thr)
        per_sheet[stem] = {"gt": int(len(gt)), "pred": len(pred), "tp": sum(tp.values()), "fp": sum(fp.values()), "fn": sum(fn.values())}
        for c in tp:
            tot.setdefault(c, [0, 0, 0])
            tot[c][0] += tp[c]; tot[c][1] += fp[c]; tot[c][2] += fn[c]

    def prf(t, f, n):
        p = t / (t + f) if t + f else None
        r = t / (t + n) if t + n else None
        f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
        return p, r, f1

    report = {"split": split, "sheets": sheets, "iou_threshold": iou_thr, "classes": {}, "per_sheet": per_sheet}
    T = F = N = 0
    for c, (t, f, n) in sorted(tot.items()):
        p, r, f1 = prf(t, f, n)
        report["classes"][names[c] if c < len(names) else str(c)] = {"tp": t, "fp": f, "fn": n, "precision": p, "recall": r, "f1": f1}
        T += t; F += f; N += n
    p, r, f1 = prf(T, F, N)
    report["overall"] = {"tp": T, "fp": F, "fn": N, "precision": p, "recall": r, "f1": f1}
    return report


def print_report(report: dict) -> None:
    table = Table(title=f"Held-out {report['split']} sheets ({len(report['sheets'])}), IoU ≥ {report['iou_threshold']}")
    for col in ("class", "TP", "FP", "FN", "precision", "recall", "F1"):
        table.add_column(col)
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    rows = list(report["classes"].items()) + [("ALL", report["overall"])]
    for name, m in rows:
        table.add_row(name, str(m["tp"]), str(m["fp"]), str(m["fn"]), fmt(m["precision"]), fmt(m["recall"]), fmt(m["f1"]))
    console.print(table)
    console.print("Recall is the number to watch for AML safety: a missed portal is an unmapped hazard.")


def evaluate_cli(cfg: dict, weights: str, dataset: str, split: str, iou: float, out: str | None) -> int:
    from .detect import UltralyticsDetector

    d = cfg["detect"]
    meta = json.loads((Path(dataset) / "splits.json").read_text(encoding="utf-8"))
    det = UltralyticsDetector(weights, int(d["imgsz"]), float(d["conf"]), d.get("device"))
    report = evaluate(dataset, det, int(meta.get("tile", d["tile_size"])), int(meta.get("overlap", d["overlap"])), float(d["iou"]), split, iou)
    report["model"] = det.model_id
    report["model_sha16"] = det.model_sha16
    report["conf_threshold"] = float(d["conf"])
    print_report(report)
    if out:
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"Report → {out}")
    return 0
