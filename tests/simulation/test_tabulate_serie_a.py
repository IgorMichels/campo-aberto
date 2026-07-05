"""Integration tests: _tabulate against the real configs/serie_a.yaml, driving
a single deterministic final table (n_draws=1) instead of simulating scores,
so each assertion is an exact 0.0/1.0 read of which zone a team lands in.

Serie A zones (see configs/serie_a.yaml): title (1), libertadores_grupos
(1-4), libertadores_pre (5), sulamericana (6-11), rebaixamento (17-20).
"""

import numpy as np

from src.simulation.simulate import RoundRobinResult, _tabulate
from tests.simulation.conftest import make_order


def _run(serie_a_config, order):
    result = RoundRobinResult(group_orders=[{"_all": order}], group_all_results=None)
    df = _tabulate(serie_a_config, {"league": result}, n_draws=1, rng=np.random.default_rng(0))
    return df.set_index("team")


def test_baseline_table_position_zones(serie_a_config, teams20):
    df = _run(serie_a_config, make_order(teams20))

    assert df.loc["T1", "prob_title"] == 1.0
    assert df.loc["T2", "prob_title"] == 0.0
    for team in ("T1", "T2", "T3", "T4"):
        assert df.loc[team, "prob_libertadores_grupos"] == 1.0
    assert df.loc["T5", "prob_libertadores_grupos"] == 0.0
    assert df.loc["T5", "prob_libertadores_pre"] == 1.0
    for team in ("T6", "T7", "T8", "T9", "T10", "T11"):
        assert df.loc[team, "prob_sulamericana"] == 1.0
    assert df.loc["T12", "prob_sulamericana"] == 0.0
    for team in ("T17", "T18", "T19", "T20"):
        assert df.loc[team, "prob_rebaixamento"] == 1.0
    assert df.loc["T16", "prob_rebaixamento"] == 0.0

    # Aggregate: libertadores = grupos + pre
    assert df.loc["T1", "prob_libertadores"] == 1.0
    assert df.loc["T5", "prob_libertadores"] == 1.0
    assert df.loc["T6", "prob_libertadores"] == 0.0


def test_champion_still_counts_as_a_groups_qualifier(serie_a_config, teams20):
    """positions-based spots overlap freely: the champion (position 1) is
    counted for both title and libertadores_grupos, not just the narrowest
    one."""
    df = _run(serie_a_config, make_order(teams20))

    assert df.loc["T1", "prob_title"] == 1.0
    assert df.loc["T1", "prob_libertadores_grupos"] == 1.0
