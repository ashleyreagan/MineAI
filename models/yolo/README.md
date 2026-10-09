# Detector weights

No trained mine-entry model is included in this repository yet.

Put a trained YOLO model here as `mine_entry.pt` (or point `detect.weights` in
`config.yaml` at it). `docs/TRAINING.md` explains how to label sheets, build a
dataset split by sheet, train, and measure precision and recall on held-out sheets.

Weights files are git-ignored. Share trained models through a GitHub release or
other storage, and record which model produced which outputs (every exported
candidate carries the model file name and hash).
