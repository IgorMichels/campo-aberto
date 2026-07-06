"""Integration tests: _tabulate against configs/serie_a_2025.yaml, which
differs from 2026's ruleset in one respect -- an extra Libertadores
preliminary-phase slot (2 instead of 1), pushing sulamericana one position
later (7-12 instead of 6-11). See configs/README.md's "Per-season configs"
section.
"""

import numpy as np

from src.simulation.simulate import RoundRobinResult, _tabulate
from tests.simulation.conftest import make_order


def _run(serie_a_2025_config, order):
    result = RoundRobinResult(group_orders=[{"_all": order}], group_all_results=None)
    df = _tabulate(serie_a_2025_config, {"league": result}, n_draws=1, rng=np.random.default_rng(0))
    return df.set_index("team")


def test_pre_libertadores_has_two_slots_instead_of_one(serie_a_2025_config, teams20):
    df = _run(serie_a_2025_config, make_order(teams20))

    assert df.loc["T5", "prob_libertadores_pre"] == 1.0
    assert df.loc["T6", "prob_libertadores_pre"] == 1.0  # the extra 2025 slot
    assert df.loc["T7", "prob_libertadores_pre"] == 0.0

    # Aggregate still tracks grupos + pre correctly with the wider pre range.
    assert df.loc["T6", "prob_libertadores"] == 1.0


def test_sulamericana_starts_one_position_later_than_2026(serie_a_2025_config, teams20):
    df = _run(serie_a_2025_config, make_order(teams20))

    assert df.loc["T6", "prob_sulamericana"] == 0.0
    for team in ("T7", "T8", "T9", "T10", "T11", "T12"):
        assert df.loc[team, "prob_sulamericana"] == 1.0
    assert df.loc["T13", "prob_sulamericana"] == 0.0


def test_libertadores_grupos_title_and_rebaixamento_unchanged_from_2026(serie_a_2025_config, teams20):
    df = _run(serie_a_2025_config, make_order(teams20))

    assert df.loc["T1", "prob_title"] == 1.0
    for team in ("T1", "T2", "T3", "T4"):
        assert df.loc[team, "prob_libertadores_grupos"] == 1.0
    for team in ("T17", "T18", "T19", "T20"):
        assert df.loc[team, "prob_rebaixamento"] == 1.0
