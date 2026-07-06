"""Shared fixtures: the real Serie A / Serie B configs (configs/*.yaml) and a
generic 20-team table, so tests exercise the exact position ranges, cascade
and aggregates the Brasileirao actually uses instead of a hand-rolled config.

The REC changes year to year (see configs/README.md's per-season note), so
there's one YAML per competition per season -- serie_a_config/serie_b_config
are the current default (2026) ruleset; serie_a_2025_config/serie_b_2025_config
are last season's, which some tests exercise specifically to cover the rule
differences (Serie A's extra pre-Libertadores slot, Serie B's access playoff
not existing yet).
"""

from pathlib import Path

import numpy as np
import pytest

from src.simulation.config import load_competition_config

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def serie_a_config():
    return load_competition_config(CONFIGS_DIR / "serie_a_2026.yaml")


@pytest.fixture
def serie_b_config():
    return load_competition_config(CONFIGS_DIR / "serie_b_2026.yaml")


@pytest.fixture
def serie_a_2025_config():
    return load_competition_config(CONFIGS_DIR / "serie_a_2025.yaml")


@pytest.fixture
def serie_b_2025_config():
    return load_competition_config(CONFIGS_DIR / "serie_b_2025.yaml")


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
