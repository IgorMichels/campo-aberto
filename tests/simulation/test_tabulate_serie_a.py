"""Integration tests: _tabulate against the real configs/serie_a.yaml, driving
a single deterministic final table (n_draws=1) instead of simulating scores,
so each assertion is an exact 0.0/1.0 read of which zone a team lands in.

Serie A zones (see configs/serie_a.yaml): title (1), libertadores_grupos
(1-4), libertadores_pre (5), sulamericana (6-11), rebaixamento (17-20), plus
the guaranteed-slot cascade [libertadores_grupos, libertadores_pre,
sulamericana] for Copa do Brasil berths (REC Art. 6 par. 1).
"""

import numpy as np

from src.simulation.simulate import RoundRobinResult, _tabulate
from tests.simulation.conftest import make_order


def _run(serie_a_config, order, guaranteed_slots=None):
    result = RoundRobinResult(group_orders=[{"_all": order}], group_all_results=None)
    df = _tabulate(
        serie_a_config,
        {"league": result},
        n_draws=1,
        rng=np.random.default_rng(0),
        guaranteed_slots=guaranteed_slots,
    )
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
    """title is nested inside libertadores_grupos, not a competing cascade
    tier: the champion gets credit for both."""
    df = _run(serie_a_config, make_order(teams20))

    assert df.loc["T1", "prob_title"] == 1.0
    assert df.loc["T1", "prob_libertadores_grupos"] == 1.0


def test_copa_do_brasil_champion_outside_top5_gets_extra_groups_berth(serie_a_config, teams20):
    """REC Art. 6 par. 1: the Copa do Brasil champion is guaranteed a groups
    berth even finishing outside the table's continental places -- here 8th,
    a sulamericana spot -- and sulamericana backfills from 12th."""
    order = make_order(teams20, {8: "Champion"})

    df = _run(serie_a_config, order, guaranteed_slots={"Champion": ["libertadores_grupos"]})

    assert df.loc["Champion", "prob_libertadores_grupos"] == 1.0
    assert df.loc["Champion", "prob_sulamericana"] == 0.0
    assert df.loc["T4", "prob_libertadores_grupos"] == 1.0  # table's natural top 4 unaffected
    assert df.loc["T12", "prob_sulamericana"] == 1.0  # backfilled
    assert df.loc["T6", "prob_sulamericana"] == 1.0


def test_copa_do_brasil_runner_up_already_in_top4_frees_its_pre_slot(serie_a_config, teams20):
    """The runner-up's pre guarantee is only used if it beats the team's table
    spot; finishing 3rd (already groups), the guarantee goes unused and
    cascades to 6th place, which in turn frees its own sulamericana seat."""
    order = make_order(teams20, {3: "RunnerUp"})

    df = _run(serie_a_config, order, guaranteed_slots={"RunnerUp": ["libertadores_pre"]})

    assert df.loc["RunnerUp", "prob_libertadores_grupos"] == 1.0
    assert df.loc["RunnerUp", "prob_libertadores_pre"] == 0.0
    assert df.loc["T5", "prob_libertadores_pre"] == 1.0  # normal 5th-place recipient, unaffected
    assert df.loc["T6", "prob_libertadores_pre"] == 1.0  # inherits the freed guarantee
    assert df.loc["T6", "prob_sulamericana"] == 0.0
    assert df.loc["T12", "prob_sulamericana"] == 1.0  # backfills T6's vacated seat


def test_double_champion_libertadores_and_copa_do_brasil_gets_two_extra_berths(serie_a_config, teams20):
    """A team that wins both this year's Libertadores and the Copa do Brasil
    holds two independent groups guarantees; it fills one seat itself, and
    the second is a genuine extra berth cascading through pre into
    sulamericana."""
    order = make_order(teams20, {9: "DoubleChampion"})

    df = _run(
        serie_a_config,
        order,
        guaranteed_slots={"DoubleChampion": ["libertadores_grupos", "libertadores_grupos"]},
    )

    assert df.loc["DoubleChampion", "prob_libertadores_grupos"] == 1.0
    assert df.loc["T5", "prob_libertadores_grupos"] == 1.0  # pulled up from its natural pre spot
    assert df.loc["T5", "prob_libertadores_pre"] == 0.0
    assert df.loc["T6", "prob_libertadores_pre"] == 1.0  # inherits the freed pre slot
    assert df.loc["T6", "prob_sulamericana"] == 0.0
    assert df.loc["T13", "prob_sulamericana"] == 1.0  # double backfill (6th's and 9th's vacated seats)


def test_guarantee_for_a_team_not_in_this_table_is_ignored(serie_a_config, teams20):
    """Passing a guaranteed_slots entry for a team that isn't playing this
    competition (e.g. a Serie B team) has no effect."""
    df = _run(
        serie_a_config,
        make_order(teams20),
        guaranteed_slots={"Not Playing / XX": ["libertadores_grupos"]},
    )

    assert df.loc["T4", "prob_libertadores_grupos"] == 1.0
    assert "Not Playing / XX" not in df.index
