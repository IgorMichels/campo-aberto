"""Shared fixtures: a generic 20-team table, used by tests that exercise the
simulation engine (standings, cascade, tabulation) through hand-rolled
configs rather than any specific real competition's YAML.
"""

from pathlib import Path

import numpy as np
import pytest

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def teams20() -> list[str]:
    return [f"T{i}" for i in range(1, 21)]


def make_order(teams: list[str], substitutions: dict[int, str] | None = None) -> list[str]:
    """teams with some 1-indexed positions swapped for named stand-ins, e.g.
    make_order(teams20, {8: "Champion"}) puts "Champion" in 8th place."""
    order = list(teams)
    for position, name in (substitutions or {}).items():
        order[position - 1] = name
    return order


@pytest.fixture
def rng():
    return np.random.default_rng(0)
