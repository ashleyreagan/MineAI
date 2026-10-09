import json
import random

import numpy as np

from mineai.dataset import build_dataset, evaluate, match, read_sheet_labels
from mineai.synthetic import RedBlobDetector, make_sheet, write_yolo_labels


def _labeled_sheets(root, n=7, size=(1600, 1200), seed=0):
    rng = random.Random(seed)
    (root / "labels").mkdir(parents=True)
    (root / "labels/classes.txt").write_text("portal\nshaft\n")
    for i in range(n):
        pts = [(rng.randint(40, size[0] - 40), rng.randint(40, size[1] - 40)) for _ in range(5)]
        boxes = make_sheet(root / f"images/sheet{i}.png", size=size, portals=pts, labels=[])
        write_yolo_labels(root / f"labels/sheet{i}.txt", boxes, size)
    make_sheet(root / "images/unlabeled.png", size=(300, 300), labels=[])


def test_dataset_splits_by_sheet_and_tiles_labels(tmp_path):
    _labeled_sheets(tmp_path)
    meta = build_dataset(str(tmp_path / "images"), str(tmp_path / "labels"), str(tmp_path / "ds"), tile=512, overlap=128, val=0.2, test=0.2)
    s = meta["splits"]
    all_sheets = s["train"] + s["val"] + s["test"]
    assert sorted(all_sheets) == sorted(f"sheet{i}" for i in range(7))  # unlabeled sheet left out
    assert len(set(all_sheets)) == len(all_sheets)  # no sheet in two splits
    assert s["test"] and s["val"]
    for split in ("train", "val"):
        imgs = list((tmp_path / "ds/images" / split).glob("*.png"))
        assert imgs
        stems = {p.stem.rsplit("_r", 1)[0] for p in imgs}
        assert stems <= set(s[split])  # tiles only come from that split's sheets
        for lab in (tmp_path / "ds/labels" / split).glob("*.txt"):
            for line in lab.read_text().splitlines():
                c, *v = line.split()
                assert c == "0" and all(0 <= float(x) <= 1 for x in v)
    assert not (tmp_path / "ds/images/test").exists()  # test sheets stay whole
    y = (tmp_path / "ds/data.yaml").read_text()
    assert "0: portal" in y and "1: shaft" in y


def test_evaluate_on_held_out_sheets(tmp_path):
    _labeled_sheets(tmp_path)
    build_dataset(str(tmp_path / "images"), str(tmp_path / "labels"), str(tmp_path / "ds"), tile=512, overlap=128, test=0.3)
    rep = evaluate(str(tmp_path / "ds"), RedBlobDetector(), 512, 128, 0.5)
    assert rep["overall"]["recall"] == 1.0 and rep["overall"]["precision"] == 1.0
    assert rep["sheets"] == json.loads((tmp_path / "ds/splits.json").read_text())["splits"]["test"]


def test_match_counts_misses_and_false_alarms():
    gt = np.array([[0, 0, 0, 10, 10], [0, 50, 50, 60, 60]], float)
    pred = [
        {"cls_id": 0, "score": 0.9, "bbox": [1, 1, 11, 11]},  # hit
        {"cls_id": 0, "score": 0.8, "bbox": [0, 0, 10, 10]},  # duplicate of the same symbol → FP
        {"cls_id": 0, "score": 0.7, "bbox": [200, 200, 210, 210]},  # nothing there → FP
    ]
    tp, fp, fn = match(pred, gt, 0.5)
    assert (tp[0], fp[0], fn[0]) == (1, 2, 1)


def test_read_labels_to_pixels(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("1 0.5 0.25 0.1 0.1\n")
    a = read_sheet_labels(p, 1000, 400)
    assert np.allclose(a, [[1, 450, 80, 550, 120]])
