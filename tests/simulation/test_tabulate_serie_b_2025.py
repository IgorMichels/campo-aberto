"""Integration tests: _tabulate against configs/serie_b_2025.yaml, which
differs from 2026's ruleset in one respect -- no access playoff existed yet,
so all 4 promotion spots are awarded directly by table position instead of
2 direct + 2 via a 3v6/4v5 playoff. See configs/README.md's "Per-season
configs" section.
"""

import numpy as np

from src.simulation.simulate import RoundRobinResult, _tabulate
from tests.simulation.conftest import make_order


def _run(serie_b_2025_config, order):
    result = RoundRobinResult(group_orders=[{"_all": order}], group_all_results=None)
    df = _tabulate(serie_b_2025_config, {"league": result}, n_draws=1, rng=np.random.default_rng(0))
    return df.set_index("team")


def test_direct_promotion_covers_all_four_spots(serie_b_2025_config, teams20):
    df = _run(serie_b_2025_config, make_order(teams20))

    for team in ("T1", "T2", "T3", "T4"):
        assert df.loc[team, "prob_direct_promotion"] == 1.0
    assert df.loc["T5", "prob_direct_promotion"] == 0.0


def test_no_access_playoff_phase_or_promotion_aggregate(serie_b_2025_config, teams20):
    df = _run(serie_b_2025_config, make_order(teams20))

    assert "prob_playoff_promotion" not in df.columns
    assert "prob_promotion" not in df.columns
    assert [p.id for p in serie_b_2025_config.phases] == ["league"]


def test_title_and_rebaixamento_unchanged_from_2026(serie_b_2025_config, teams20):
    df = _run(serie_b_2025_config, make_order(teams20))

    assert df.loc["T1", "prob_title"] == 1.0
    for team in ("T17", "T18", "T19", "T20"):
        assert df.loc[team, "prob_rebaixamento"] == 1.0
