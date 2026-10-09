# Training and evaluating the mine-entry detector

MineAI does not ship a trained model. This page covers how to build one and
how to measure it honestly.

## 1. Label sheets

Label **whole sheets**, not crops, in a tool that exports YOLO format: CVAT,
Label Studio, Roboflow, or similar.

- Draw a tight box around each symbol. Suggested classes are `portal`,
  `drift`, `shaft`, `slope`, and `air_shaft`; use whatever your sheets
  actually distinguish.
- Keep a `classes.txt` with one class name per line, in class-id order.
- Each sheet's label file is `<sheet name>.txt` in one folder, with lines of
  `class cx cy w h` normalized 0–1 to the full sheet.
- A sheet you checked that has no symbols still needs a label file, left
  empty. A missing file means "not labeled yet", and the sheet is left out.
- Symbol conventions differ by company, era and draftsman. Label sheets from
  many sources (different companies, states and decades) rather than many
  sheets from one source.

Possible sources include the National Mine Map Repository, PA's Mine Map Atlas,
and state mine-map collections. Check the reuse terms for each collection.

## 2. Build the dataset (split by sheet)

```bash
mineai dataset --images data/raw --labels data/labels --out data/yolo_dataset \
               --tile 1024 --overlap 256 --val 0.2 --test 0.1
```

- **Sheets** are split into train/val/test, not tiles. Tiles from one sheet share
  paper, ink, scan settings and drafting hand. A random tile split would leak
  near-duplicates into validation and overstate accuracy.
- Train and val sheets are cut into overlapping tiles at native resolution,
  the same tiling `mineai detect` uses. A box is kept in a tile when at least
  60% of it is visible there. Some symbol-free background tiles are kept
  (`--empty-ratio`) so the model learns what isn't a symbol.
- **Test sheets are kept whole** and are listed in `splits.json`. Don't train on them.
- Use the same `--tile` value as `detect.tile_size` in `config.yaml`.

## 3. Train

```bash
yolo detect train data=data/yolo_dataset/data.yaml model=yolov8s.pt imgsz=1024 epochs=150
cp runs/detect/train/weights/best.pt models/yolo/mine_entry.pt
```

A GPU helps a lot. Start from a pretrained checkpoint (`yolov8s.pt`), not from scratch.

## 4. Evaluate on the held-out sheets

```bash
mineai evaluate --weights models/yolo/mine_entry.pt --dataset data/yolo_dataset --out eval.json
```

This runs the real tiled pipeline on each whole test sheet. It matches
detections to labels one-to-one by class at IoU ≥ 0.5 and reports TP, FP, FN,
precision, recall and F1 for each class, plus counts for each sheet.

- **Recall matters most for AML work.** A missed portal is an unmapped hazard.
  A false positive costs a reviewer a few seconds.
  To raise recall, lower `detect.conf`.
- Report these numbers, with the number of test sheets, the sources they came
  from, and the model hash, wherever the model is described (for example in the
  README). Don't quote training or validation metrics as accuracy.

## 5. Use it

Set `detect.weights` to the model file and run `mineai run`. Every exported
candidate records the model file name and hash, so results can always be
traced back to the exact model that produced them.
