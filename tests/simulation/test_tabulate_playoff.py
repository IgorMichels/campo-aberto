"""Integration tests: _tabulate against a hand-built config with a round_robin
phase feeding a playoff phase (table_position pairing), driving deterministic
final tables/winners (n_draws=1). Generic over any such config -- not tied to
any specific real competition.
"""

import numpy as np

from src.simulation.config import _parse_competition
from src.simulation.simulate import PlayoffResult, RoundRobinResult, _tabulate
from tests.simulation.conftest import make_order


def _config():
    raw = {
        "name": "Test",
        "n_teams": 20,
        "phases": [
            {
                "id": "league",
                "type": "round_robin",
                "head_to_head_mode": "goal_diff_only",
                "spots": [
                    {"name": "title", "positions": {"from": 1, "to": 1}},
                    {"name": "direct_promotion", "positions": {"from": 1, "to": 2}},
                    {"name": "relegation", "positions": {"from": 17, "to": 20}},
                ],
            },
            {
                "id": "playoff",
                "type": "playoff",
                "source_phase": "league",
                "pairing": "table_position",
                "pairs": [[3, 6], [4, 5]],
                "legs": 2,
                "leg_order": "worse_seed_home_first",
                "tiebreak": "points_then_goal_diff",
                "spots": [{"name": "playoff_promotion", "result": "winner"}],
            },
        ],
        "aggregates": [{"name": "promotion", "of": ["direct_promotion", "playoff_promotion"]}],
    }
    return _parse_competition(raw, source="test")


def _run(config, order, playoff_winners, guaranteed_slots=None):
    league_result = RoundRobinResult(group_orders=[{"_all": order}], group_all_results=None)
    playoff_result = PlayoffResult(winners={i: np.array([winner]) for i, winner in enumerate(playoff_winners)})
    df = _tabulate(
        config,
        {"league": league_result, "playoff": playoff_result},
        n_draws=1,
        rng=np.random.default_rng(0),
        guaranteed_slots=guaranteed_slots,
    )
    return df.set_index("team")


def test_title_direct_promotion_and_relegation_zones(teams20):
    # playoff pairs are [3, 6] and [4, 5] (table_position pairing); 3rd and
    # 4th place (the better seeds) advance in this draw.
    df = _run(_config(), make_order(teams20), playoff_winners=["T3", "T4"])

    assert df.loc["T1", "prob_title"] == 1.0
    assert df.loc["T2", "prob_title"] == 0.0

    assert df.loc["T1", "prob_direct_promotion"] == 1.0
    assert df.loc["T2", "prob_direct_promotion"] == 1.0
    assert df.loc["T3", "prob_direct_promotion"] == 0.0

    for team in ("T17", "T18", "T19", "T20"):
        assert df.loc[team, "prob_relegation"] == 1.0
    assert df.loc["T16", "prob_relegation"] == 0.0


def test_playoff_promotion_and_promotion_aggregate(teams20):
    df = _run(_config(), make_order(teams20), playoff_winners=["T3", "T4"])

    assert df.loc["T3", "prob_playoff_promotion"] == 1.0
    assert df.loc["T4", "prob_playoff_promotion"] == 1.0
    assert df.loc["T5", "prob_playoff_promotion"] == 0.0
    assert df.loc["T6", "prob_playoff_promotion"] == 0.0

    # promotion = direct_promotion + playoff_promotion
    assert df.loc["T1", "prob_promotion"] == 1.0  # via direct_promotion
    assert df.loc["T3", "prob_promotion"] == 1.0  # via playoff_promotion
    assert df.loc["T5", "prob_promotion"] == 0.0  # lost the playoff


def test_worse_seed_can_still_win_the_playoff(teams20):
    """The playoff's winner isn't fixed to the better seed -- e.g. 6th
    upsetting 3rd -- so playoff_promotion tracks the actual simulated winner,
    not position."""
    df = _run(_config(), make_order(teams20), playoff_winners=["T6", "T5"])

    assert df.loc["T6", "prob_playoff_promotion"] == 1.0
    assert df.loc["T3", "prob_playoff_promotion"] == 0.0
    assert df.loc["T6", "prob_promotion"] == 1.0


def test_no_cascade_so_guaranteed_slots_have_no_effect(teams20):
    order = make_order(teams20)
    without = _run(_config(), order, playoff_winners=["T3", "T4"])
    with_guarantee = _run(
        _config(), order, playoff_winners=["T3", "T4"], guaranteed_slots={"T9": ["direct_promotion"]}
    )

    assert without.equals(with_guarantee)
