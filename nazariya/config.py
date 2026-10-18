"""Load the YAML config files."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


def load_outlets() -> list[dict]:
    return _load("outlets.yaml")["outlets"]


def load_feeds() -> dict[str, dict]:
    return _load("feeds.yaml")["feeds"]


def load_filters() -> dict:
    return _load("filters.yaml")
