"""Integration tests: _tabulate against the real configs/serie_b.yaml.

Serie B zones (see configs/serie_b.yaml): title (1), direct_promotion (1-2),
rebaixamento (17-20), plus the "acesso" playoff (3v6, 4v5) feeding
playoff_promotion, aggregated into `promotion` alongside direct_promotion.
"""

import numpy as np

from src.simulation.simulate import PlayoffResult, RoundRobinResult, _tabulate
from tests.simulation.conftest import make_order


def _run(serie_b_config, order, playoff_winners):
    league_result = RoundRobinResult(group_orders=[{"_all": order}], group_all_results=None)
    acesso_result = PlayoffResult(
        winners={i: np.array([winner]) for i, winner in enumerate(playoff_winners)}
    )
    df = _tabulate(
        serie_b_config, {"league": league_result, "acesso": acesso_result}, n_draws=1, rng=np.random.default_rng(0)
    )
    return df.set_index("team")


def test_title_direct_promotion_and_relegation_zones(serie_b_config, teams20):
    # acesso pairs are [3, 6] and [4, 5] (table_position pairing); 3rd and 4th
    # place (the better seeds) advance in this draw.
    df = _run(serie_b_config, make_order(teams20), playoff_winners=["T3", "T4"])

    assert df.loc["T1", "prob_title"] == 1.0
    assert df.loc["T2", "prob_title"] == 0.0

    assert df.loc["T1", "prob_direct_promotion"] == 1.0
    assert df.loc["T2", "prob_direct_promotion"] == 1.0
    assert df.loc["T3", "prob_direct_promotion"] == 0.0

    for team in ("T17", "T18", "T19", "T20"):
        assert df.loc[team, "prob_rebaixamento"] == 1.0
    assert df.loc["T16", "prob_rebaixamento"] == 0.0


def test_playoff_promotion_and_promotion_aggregate(serie_b_config, teams20):
    df = _run(serie_b_config, make_order(teams20), playoff_winners=["T3", "T4"])

    assert df.loc["T3", "prob_playoff_promotion"] == 1.0
    assert df.loc["T4", "prob_playoff_promotion"] == 1.0
    assert df.loc["T5", "prob_playoff_promotion"] == 0.0
    assert df.loc["T6", "prob_playoff_promotion"] == 0.0

    # promotion = direct_promotion + playoff_promotion
    assert df.loc["T1", "prob_promotion"] == 1.0  # via direct_promotion
    assert df.loc["T3", "prob_promotion"] == 1.0  # via playoff_promotion
    assert df.loc["T5", "prob_promotion"] == 0.0  # lost the playoff


def test_worse_seed_can_still_win_the_playoff(serie_b_config, teams20):
    """acesso's winner isn't fixed to the better seed -- e.g. 6th upsetting 3rd
    -- so playoff_promotion tracks the actual simulated winner, not position."""
    df = _run(serie_b_config, make_order(teams20), playoff_winners=["T6", "T5"])

    assert df.loc["T6", "prob_playoff_promotion"] == 1.0
    assert df.loc["T3", "prob_playoff_promotion"] == 0.0
    assert df.loc["T6", "prob_promotion"] == 1.0
