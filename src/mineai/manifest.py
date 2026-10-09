"""Per-sheet run manifest: what ran, what succeeded, what failed, and why.

Every stage records a status for every sheet it touches:
  ok       the stage produced its output
  skipped  the stage deliberately did nothing (reason given), e.g. no GCP file
  failed   the stage tried and failed (error given)

The CLI exits non-zero when any stage it ran recorded a failure.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OK, SKIPPED, FAILED = "ok", "skipped", "failed"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Manifest:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = {"sheets": {}, "stage_errors": {}}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.data.setdefault("sheets", {})
            self.data.setdefault("stage_errors", {})

    @property
    def sheets(self) -> dict[str, dict[str, Any]]:
        return self.data["sheets"]

    def sheet(self, sheet_id: str) -> dict[str, Any]:
        return self.sheets.setdefault(sheet_id, {"stages": {}})

    def set_status(self, sheet_id: str, stage: str, status: str, message: str = "", **extra: Any) -> None:
        entry = {"status": status, "message": message, "time": now_iso()}
        entry.update(extra)
        self.sheet(sheet_id).setdefault("stages", {})[stage] = entry

    def status(self, sheet_id: str, stage: str) -> str | None:
        return self.sheets.get(sheet_id, {}).get("stages", {}).get(stage, {}).get("status")

    def set_stage_error(self, stage: str, message: str | None) -> None:
        """Record (or clear) a failure that affected a whole stage, e.g. a missing model."""
        if message:
            self.data["stage_errors"][stage] = {"message": message, "time": now_iso()}
        else:
            self.data["stage_errors"].pop(stage, None)

    def fail_stage(self, stage: str, message: str) -> None:
        """A failure that affects the whole stage (e.g. no model): mark every ingested sheet."""
        self.set_stage_error(stage, message)
        for sheet_id in self.sheets:
            if self.status(sheet_id, "ingest") == OK:
                self.set_status(sheet_id, stage, FAILED, message)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)

    def failures(self, stages: list[str] | None = None) -> list[tuple[str, str, str]]:
        """List (sheet or '*', stage, message) for every failure in the given stages."""
        out = []
        stage_msgs = {}
        for stage, err in self.data["stage_errors"].items():
            if stages is None or stage in stages:
                out.append(("*", stage, err["message"]))
                stage_msgs[stage] = err["message"]
        for sheet_id, sheet in self.sheets.items():
            for stage, entry in sheet.get("stages", {}).items():
                if (stages is None or stage in stages) and entry.get("status") == FAILED:
                    if stage_msgs.get(stage) == entry.get("message"):
                        continue  # already reported once for the whole stage
                    out.append((sheet_id, stage, entry.get("message", "")))
        return out
