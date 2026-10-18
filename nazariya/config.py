"""Load the YAML config files."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal KEY=VALUE loader for the gitignored .env (used by the home-PC task).
    Real environment variables win, so CI secrets are never overridden."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))


def load_outlets() -> list[dict]:
    return _load("outlets.yaml")["outlets"]


def runner_of(outlet: dict) -> str:
    """'cloud' (GitHub Actions) unless the outlet blocks datacenter IPs and runs from a home PC."""
    return outlet.get("runner", "cloud")


def load_feeds() -> dict[str, dict]:
    return _load("feeds.yaml")["feeds"]


def load_filters() -> dict:
    return _load("filters.yaml")
