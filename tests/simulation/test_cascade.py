"""Unit tests for _resolve_cascade, the Copa do Brasil guaranteed-slot
allocation used by Serie A's `league` phase (cascade: [libertadores_grupos,
libertadores_pre, sulamericana], capacities 4/1/6 -- see configs/serie_a_2026.yaml
and configs/README.md for the rules this implements).
"""

from src.simulation.config import SpotConfig
from src.simulation.simulate import _resolve_cascade
from tests.simulation.conftest import make_order

GRUPOS = SpotConfig(name="libertadores_grupos", positions=(1, 4))
PRE = SpotConfig(name="libertadores_pre", positions=(5, 5))
SULA = SpotConfig(name="sulamericana", positions=(6, 11))
CASCADE = [GRUPOS, PRE, SULA]


def test_no_guarantees_matches_plain_table_position(teams20):
    result = _resolve_cascade(make_order(teams20), CASCADE, {})

    assert result["libertadores_grupos"] == ["T1", "T2", "T3", "T4"]
    assert result["libertadores_pre"] == ["T5"]
    assert result["sulamericana"] == ["T6", "T7", "T8", "T9", "T10", "T11"]


def test_guarantee_better_than_table_position_bumps_team_up_and_backfills(teams20):
    """Copa do Brasil champion (Art. 6 par. 1) guaranteed libertadores_grupos,
    but finishes 8th (a sulamericana spot): skips sulamericana for an extra
    groups berth, and sulamericana backfills the vacated 8th seat from 12th."""
    order = make_order(teams20, {8: "Champion"})

    result = _resolve_cascade(order, CASCADE, {"Champion": ["libertadores_grupos"]})

    assert result["libertadores_grupos"] == ["Champion", "T1", "T2", "T3", "T4"]
    assert result["libertadores_pre"] == ["T5"]
    assert result["sulamericana"] == ["T6", "T7", "T9", "T10", "T11", "T12"]


def test_guarantee_worse_than_table_position_is_unused_and_cascades(teams20):
    """Copa do Brasil runner-up guaranteed libertadores_pre, but finishes 3rd
    (a libertadores_grupos spot): keeps the better table spot. Its unused pre
    guarantee opens an extra seat for 6th place (first team outside pre's
    5th-place window); 6th's own vacated sulamericana seat then backfills
    from 12th, same mechanic one level down."""
    order = make_order(teams20, {3: "RunnerUp"})

    result = _resolve_cascade(order, CASCADE, {"RunnerUp": ["libertadores_pre"]})

    assert result["libertadores_grupos"] == ["RunnerUp", "T1", "T2", "T4"]
    assert result["libertadores_pre"] == ["T5", "T6"]
    assert result["sulamericana"] == ["T7", "T8", "T9", "T10", "T11", "T12"]


def test_two_independent_guarantees_of_the_same_tier_both_cascade(teams20):
    """A team that's both this year's Libertadores champion and Copa do Brasil
    champion holds two separate libertadores_grupos guarantees, but still only
    fills one groups seat itself -- the second is a genuine extra berth, on
    top of the first."""
    order = make_order(teams20, {9: "DoubleChampion"})

    result = _resolve_cascade(
        order, CASCADE, {"DoubleChampion": ["libertadores_grupos", "libertadores_grupos"]}
    )

    assert result["libertadores_grupos"] == ["DoubleChampion", "T1", "T2", "T3", "T4", "T5"]
    assert result["libertadores_pre"] == ["T6"]
    assert result["sulamericana"] == ["T7", "T8", "T10", "T11", "T12", "T13"]


def test_two_different_guarantees_use_the_better_and_cascade_the_other(teams20):
    """A team guaranteed both libertadores_grupos and libertadores_pre uses the
    better one (grupos); the unused pre guarantee cascades independently, same
    as a lone pre guarantee would."""
    order = make_order(teams20, {9: "DoubleChampion"})

    result = _resolve_cascade(
        order, CASCADE, {"DoubleChampion": ["libertadores_grupos", "libertadores_pre"]}
    )

    assert result["libertadores_grupos"] == ["DoubleChampion", "T1", "T2", "T3", "T4"]
    assert result["libertadores_pre"] == ["T5", "T6"]
    assert result["sulamericana"] == ["T7", "T8", "T10", "T11", "T12", "T13"]


def test_guaranteed_team_absent_from_order_is_ignored(teams20):
    """A guarantee naming a team from a different competition/group (e.g. a
    Serie A guarantee passed in while simulating Serie B) has no effect."""
    result = _resolve_cascade(
        make_order(teams20), CASCADE, {"Not Playing / XX": ["libertadores_grupos"]}
    )

    assert result["libertadores_grupos"] == ["T1", "T2", "T3", "T4"]


def test_guarantee_for_a_spot_outside_cascade_is_ignored(teams20):
    """A guarantee naming a spot that isn't part of this phase's cascade (e.g.
    'title') is dropped -- the team falls back to its plain table position."""
    order = make_order(teams20, {8: "Z"})

    result = _resolve_cascade(order, CASCADE, {"Z": ["title"]})

    assert result["sulamericana"] == ["T6", "T7", "Z", "T9", "T10", "T11"]


def test_no_cascade_spots_returns_empty_result(teams20):
    assert _resolve_cascade(make_order(teams20), [], {"T1": ["libertadores_grupos"]}) == {}
