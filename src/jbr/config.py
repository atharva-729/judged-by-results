"""Paths and the fixed settings in config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
IMPORTED = DATA / "imported"
REPORTS = ROOT / "reports"


def load() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
