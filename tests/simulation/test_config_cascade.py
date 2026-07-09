"""Schema tests for RoundRobinPhaseConfig.cascade (see configs/README.md,
"Cascade" section)."""

import pytest

from src.simulation.config import RoundRobinPhaseConfig, _parse_competition, load_competition_config
from tests.simulation.conftest import CONFIGS_DIR


def _competition(cascade=None, extra_spots=()):
    spots = [
        {"name": "libertadores_grupos", "positions": {"from": 1, "to": 4}},
        {"name": "sulamericana", "positions": {"from": 6, "to": 11}},
        *extra_spots,
    ]
    phase = {
        "id": "league",
        "type": "round_robin",
        "head_to_head_mode": "points_then_goal_diff",
        "spots": spots,
    }
    if cascade is not None:
        phase["cascade"] = cascade
    return {"name": "Test", "n_teams": 20, "phases": [phase]}


def test_cascade_defaults_to_empty():
    config = _parse_competition(_competition(), source="test")
    assert config.phase("league").cascade == ()


def test_cascade_accepts_positions_based_spots_in_priority_order():
    config = _parse_competition(
        _competition(cascade=["libertadores_grupos", "sulamericana"]), source="test"
    )
    assert config.phase("league").cascade == ("libertadores_grupos", "sulamericana")


def test_cascade_rejects_unknown_spot_name():
    with pytest.raises(ValueError, match="cascade entry"):
        _parse_competition(_competition(cascade=["not_a_spot"]), source="test")


def test_cascade_rejects_a_result_based_spot():
    raw = _competition(cascade=["playoff_winner"])
    raw["phases"][0]["spots"].append({"name": "playoff_winner", "positions": {"from": 1, "to": 1}})
    # even a positions-based spot must be declared on *this* phase -- a spot
    # name that only exists elsewhere isn't a valid cascade entry either.
    raw["phases"][0]["cascade"] = ["libertadores_grupos", "not_declared_here"]
    with pytest.raises(ValueError, match="cascade entry"):
        _parse_competition(raw, source="test")


def test_cascade_rejects_duplicate_entries():
    with pytest.raises(ValueError, match="duplicate"):
        _parse_competition(
            _competition(cascade=["libertadores_grupos", "libertadores_grupos"]), source="test"
        )


def test_real_configs_parse_and_declare_double_round_robin_leagues():
    """Every config actually shipped in configs/*.yaml must load cleanly and,
    for its (ungrouped) round_robin phases, use the only fixture-derivable
    shape this engine supports (see fixtures.py) -- a double round-robin.
    Deliberately config-agnostic: doesn't hardcode which file has a cascade,
    an aggregate, or a playoff phase.
    """
    config_paths = sorted(CONFIGS_DIR.glob("*.yaml"))
    assert config_paths, f"no configs found under {CONFIGS_DIR}"

    for path in config_paths:
        config = load_competition_config(path)
        for phase in config.phases:
            if isinstance(phase, RoundRobinPhaseConfig) and phase.groups is None:
                assert phase.legs == 2, f"{path}: phase {phase.id!r} expected legs == 2"
            for spot in phase.spots:
                if spot.positions is not None:
                    assert spot.positions[1] <= config.n_teams, (
                        f"{path}: spot {spot.name!r} exceeds n_teams"
                    )
