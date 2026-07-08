"""RoundRobinPhaseConfig.legs: the double round-robin (legs=2) every Brazilian
league config uses today, and the explicit guard against legs=1 (a single
round-robin, e.g. a World Cup-style group stage) -- see fixtures.py's module
docstring and configs/README.md for why the latter isn't implemented: this
engine has no source of a real remaining-fixture schedule for it, only a
team roster, and a double round-robin is the only shape derivable from that
alone.
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation.config import _parse_competition
from src.simulation.simulate import DrawParams, _run_round_robin_phase


def _round_robin_competition(legs=None):
    phase = {
        "id": "league",
        "type": "round_robin",
        "head_to_head_mode": "points_then_goal_diff",
        "spots": [{"name": "title", "positions": {"from": 1, "to": 1}}],
    }
    if legs is not None:
        phase["legs"] = legs
    return {"name": "Test", "n_teams": 20, "phases": [phase]}


def test_legs_defaults_to_double_round_robin():
    config = _parse_competition(_round_robin_competition(), source="test")
    assert config.phase("league").legs == 2


def test_legs_accepts_a_single_round_robin_value():
    config = _parse_competition(_round_robin_competition(legs=1), source="test")
    assert config.phase("league").legs == 1


def test_legs_rejects_anything_but_1_or_2():
    with pytest.raises(ValueError, match="'legs' must be 1 or 2"):
        _parse_competition(_round_robin_competition(legs=3), source="test")


def test_single_round_robin_phase_raises_not_implemented():
    """legs=1 has no combinatorial fixture derivation (see fixtures.py) --
    _run_round_robin_phase must refuse it outright rather than silently
    simulating a phantom double round-robin schedule."""
    config = _parse_competition(_round_robin_competition(legs=1), source="test")
    phase_cfg = config.phase("league")

    n_draws, n_teams = 3, 20
    draw_params: DrawParams = (
        np.zeros((n_draws, n_teams)),
        np.zeros((n_draws, n_teams)),
        np.zeros(n_draws),
        np.zeros(n_draws),
        np.zeros(n_draws),
        {f"T{i}": i for i in range(n_teams)},
    )

    with pytest.raises(NotImplementedError, match="legs=1"):
        _run_round_robin_phase(
            phase_cfg,
            competition="Test",
            season=2026,
            reference_date=pd.Timestamp("2026-01-01"),
            matches_df=pd.DataFrame(),
            draw_params=draw_params,
            rng=np.random.default_rng(0),
        )
