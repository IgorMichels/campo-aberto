"""Schema tests for RoundRobinPhaseConfig.cascade and the real Serie A / Serie B
configs' use of it (see configs/README.md, "Cascade" section)."""

import pytest

from src.simulation.config import AggregateConfig, _parse_competition


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


def test_serie_a_config_declares_the_libertadores_cascade_and_aggregate(serie_a_config):
    league = serie_a_config.phase("league")
    assert league.cascade == ("libertadores_grupos", "libertadores_pre", "sulamericana")
    assert "title" not in league.cascade  # nested bonus, not a competing tier
    assert AggregateConfig(name="libertadores", of=("libertadores_grupos", "libertadores_pre")) in serie_a_config.aggregates


def test_serie_b_config_has_no_cascade(serie_b_config):
    assert serie_b_config.phase("league").cascade == ()
