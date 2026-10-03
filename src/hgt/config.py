"""Load the project config (config.yaml at the repo root)."""
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def site_years() -> list[tuple[str, int]]:
    """All (site, year) pairs with lidar."""
    cfg = load_config()
    return [(s, int(y)) for s, v in cfg["sites"].items() for y in v["years"]]
