"""Integration tests: _tabulate against a hand-built round_robin-only config,
driving a single deterministic final table (n_draws=1) instead of simulating
scores, so each assertion is an exact 0.0/1.0 read of which zone a team lands
in. Generic over any config with position-based spots and a cascade -- not
tied to any specific real competition.
"""

import numpy as np

from src.simulation.config import _parse_competition
from src.simulation.simulate import RoundRobinResult, _tabulate
from tests.simulation.conftest import make_order


def _config():
    raw = {
        "name": "Test",
        "n_teams": 20,
        "phases": [
            {
                "id": "league",
                "type": "round_robin",
                "head_to_head_mode": "points_then_goal_diff",
                "spots": [
                    {"name": "title", "positions": {"from": 1, "to": 1}},
                    {"name": "tier_1", "positions": {"from": 1, "to": 4}},
                    {"name": "tier_2", "positions": {"from": 5, "to": 5}},
                    {"name": "tier_3", "positions": {"from": 6, "to": 11}},
                    {"name": "relegation", "positions": {"from": 17, "to": 20}},
                ],
                "cascade": ["tier_1", "tier_2", "tier_3"],
            }
        ],
        "aggregates": [{"name": "continental", "of": ["tier_1", "tier_2"]}],
    }
    return _parse_competition(raw, source="test")


def _run(config, order, guaranteed_slots=None):
    result = RoundRobinResult(group_orders=[{"_all": order}], group_all_results=None)
    df = _tabulate(
        config,
        {"league": result},
        n_draws=1,
        rng=np.random.default_rng(0),
        guaranteed_slots=guaranteed_slots,
    )
    return df.set_index("team")


def test_baseline_table_position_zones(teams20):
    df = _run(_config(), make_order(teams20))

    assert df.loc["T1", "prob_title"] == 1.0
    assert df.loc["T2", "prob_title"] == 0.0
    for team in ("T1", "T2", "T3", "T4"):
        assert df.loc[team, "prob_tier_1"] == 1.0
    assert df.loc["T5", "prob_tier_1"] == 0.0
    assert df.loc["T5", "prob_tier_2"] == 1.0
    for team in ("T6", "T7", "T8", "T9", "T10", "T11"):
        assert df.loc[team, "prob_tier_3"] == 1.0
    assert df.loc["T12", "prob_tier_3"] == 0.0
    for team in ("T17", "T18", "T19", "T20"):
        assert df.loc[team, "prob_relegation"] == 1.0
    assert df.loc["T16", "prob_relegation"] == 0.0

    # Aggregate: continental = tier_1 + tier_2
    assert df.loc["T1", "prob_continental"] == 1.0
    assert df.loc["T5", "prob_continental"] == 1.0
    assert df.loc["T6", "prob_continental"] == 0.0


def test_title_still_counts_as_a_tier_1_qualifier(teams20):
    """title is nested inside tier_1, not a competing cascade tier: the
    champion gets credit for both."""
    df = _run(_config(), make_order(teams20))

    assert df.loc["T1", "prob_title"] == 1.0
    assert df.loc["T1", "prob_tier_1"] == 1.0


def test_guaranteed_slot_outside_tier_1_gets_an_extra_berth(teams20):
    """A team guaranteed tier_1 but finishing 8th (a tier_3 spot) is credited
    tier_1, and tier_3 backfills from 12th."""
    order = make_order(teams20, {8: "Champion"})

    df = _run(_config(), order, guaranteed_slots={"Champion": ["tier_1"]})

    assert df.loc["Champion", "prob_tier_1"] == 1.0
    assert df.loc["Champion", "prob_tier_3"] == 0.0
    assert df.loc["T4", "prob_tier_1"] == 1.0  # table's natural top 4 unaffected
    assert df.loc["T12", "prob_tier_3"] == 1.0  # backfilled
    assert df.loc["T6", "prob_tier_3"] == 1.0


def test_guarantee_already_in_a_better_spot_frees_its_own_tier(teams20):
    """The guarantee is only used if it beats the team's table spot;
    finishing 3rd (already tier_1), the guarantee goes unused and cascades to
    6th place, which in turn frees its own tier_3 seat."""
    order = make_order(teams20, {3: "RunnerUp"})

    df = _run(_config(), order, guaranteed_slots={"RunnerUp": ["tier_2"]})

    assert df.loc["RunnerUp", "prob_tier_1"] == 1.0
    assert df.loc["RunnerUp", "prob_tier_2"] == 0.0
    assert df.loc["T5", "prob_tier_2"] == 1.0  # normal recipient, unaffected
    assert df.loc["T6", "prob_tier_2"] == 1.0  # inherits the freed guarantee
    assert df.loc["T6", "prob_tier_3"] == 0.0
    assert df.loc["T12", "prob_tier_3"] == 1.0  # backfills T6's vacated seat


def test_guarantee_for_a_team_not_in_this_table_is_ignored(teams20):
    df = _run(_config(), make_order(teams20), guaranteed_slots={"Not Playing / XX": ["tier_1"]})

    assert df.loc["T4", "prob_tier_1"] == 1.0
    assert "Not Playing / XX" not in df.index
