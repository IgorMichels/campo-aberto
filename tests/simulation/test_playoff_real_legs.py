"""Regression tests for the playoff real-legs bug: _simulate_playoff_pair used
to unconditionally sample BOTH legs of a two-legged playoff tie, even when
one or both legs had already been played for real in matches.csv as of
reference_date -- throwing away the real score for a fabricated one, instead
of mirroring round_robin's played/remaining split for that leg.

See src/simulation/fixtures.real_result (the single-pair lookup) and
src/simulation/simulate.py's _apply_real_leg_results/_simulate_playoff_pair
(where the override happens) for the fix.
"""

import numpy as np
import pandas as pd

from src.models.adapters.poisson_home import ADAPTER as POISSON_HOME
from src.simulation.config import PlayoffPhaseConfig, SpotConfig
from src.simulation.simulate import DrawParams, _apply_real_leg_results, _simulate_playoff_pair

TEAMS = ["TeamA", "TeamB"]  # TeamA: the better seed (idx_a). TeamB: the worse seed (idx_b).


def _row(home, away, home_goals, away_goals, date, competition="Test", season=2026):
    return {
        "competition": competition,
        "season": season,
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "match_datetime": pd.Timestamp(date),
    }


# The regular-season (round-robin) meetings between the two playoff teams --
# a double round-robin already plays both ordered directions of every pair
# once, before a playoff leg (see configs/serie_b_2026.yaml's access playoff)
# ever reuses one of those same directions -- see fixtures.real_result's
# docstring for why a single played row for a pair is deliberately NOT enough
# to count as that pair's playoff leg. Present in every scenario below to
# prove these two rows alone are never mistaken for a playoff leg's result.
BASE_ROWS = [
    _row("TeamB", "TeamA", 1, 1, "2026-06-01"),  # leg 1's ordered pair, regular season
    _row("TeamA", "TeamB", 2, 0, "2026-06-08"),  # leg 2's ordered pair, regular season
]


def _playoff_phase_cfg(legs=2, leg_order="worse_seed_home_first"):
    return PlayoffPhaseConfig(
        id="playoff",
        pairing="table_position",
        spots=(SpotConfig(name="playoff_promotion", result="winner"),),
        legs=legs,
        leg_order=leg_order,
        tiebreak="points_then_goal_diff",
    )


def _draw_params(n_draws, seed):
    # Deliberately tighter than a bare rng.normal(size=...) -- an unconstrained
    # N(0, 1) attack/defense/eta/beta_home sum can occasionally exp() into an
    # absurd scoring rate (e.g. a two-sigma tail on 4 summed normals already
    # exceeds exp(4)), which would make even a lopsided real leg 1 margin
    # spuriously overturnable by leg 2 -- unrepresentative of an actual fitted
    # posterior's typical spread (attack/defense within roughly +-1, beta_home
    # ~0.2-0.4). Scaling down keeps this a genuine (non-mocked) POISSON_HOME
    # sample while staying in a realistic scoring-rate range.
    rng = np.random.default_rng(seed)
    return DrawParams(
        adapter=POISSON_HOME,
        team_params={
            "attack": rng.normal(scale=0.3, size=(n_draws, 2)),
            "defense": rng.normal(scale=0.3, size=(n_draws, 2)),
        },
        shared_params={
            "eta": rng.normal(scale=0.1, size=n_draws),
            "beta_home": rng.normal(loc=0.3, scale=0.1, size=n_draws),
            "rho": np.zeros(n_draws),
        },
        team_index={"TeamA": 0, "TeamB": 1},
        n_draws=n_draws,
    )


def _simulate(
    matches_df, reference_date, n_draws=300, seed=0, legs=2, leg_order="worse_seed_home_first"
):
    draw_params = _draw_params(n_draws, seed)
    idx_a = np.zeros(n_draws, dtype=np.int64)
    idx_b = np.ones(n_draws, dtype=np.int64)
    phase_cfg = _playoff_phase_cfg(legs=legs, leg_order=leg_order)
    return _simulate_playoff_pair(
        idx_a,
        idx_b,
        draw_params,
        TEAMS,
        phase_cfg,
        np.random.default_rng(seed + 1),
        matches_df,
        "Test",
        2026,
        pd.Timestamp(reference_date),
    )


class TestApplyRealLegResults:
    """Direct unit coverage of the per-draw team-identity subtlety flagged in
    the task: idx_home/idx_away are (n_draws,) arrays, and which two teams
    occupy them can in principle vary per draw (table_position/
    bracket_adjacent pairings fed by a tiebreak coin flip) -- so overriding
    must group by unique (home_name, away_name) pair across all draws, not
    assume a single shared pair.
    """

    def test_overrides_only_draws_whose_specific_pair_is_really_played(self):
        matches_df = pd.DataFrame(
            [
                _row("X", "Y", 3, 1, "2026-06-01"),
                _row(
                    "X", "Y", 3, 1, "2026-11-15"
                ),  # 2nd row -> counts as "played" (see real_result)
                _row("Y", "X", 2, 2, "2026-06-08"),
                _row("Y", "X", 2, 2, "2026-11-15"),
            ]
        )
        teams = ["X", "Y", "Z"]
        # 4 draws, 4 different (home, away) pairs: (X,Y) real, (X,Z) not
        # played at all, (Y,Z) not played at all, (Y,X) real.
        idx_home = np.array([0, 0, 1, 1])
        idx_away = np.array([1, 2, 2, 0])
        g_home = np.array([9, 9, 9, 9])
        g_away = np.array([9, 9, 9, 9])

        out_home, out_away = _apply_real_leg_results(
            g_home,
            g_away,
            idx_home,
            idx_away,
            teams,
            matches_df,
            "Test",
            2026,
            pd.Timestamp("2026-12-01"),
        )

        np.testing.assert_array_equal(out_home, [3, 9, 9, 2])
        np.testing.assert_array_equal(out_away, [1, 9, 9, 2])
        # Sampled input arrays are untouched -- a defensive copy, not a mutation.
        np.testing.assert_array_equal(g_home, [9, 9, 9, 9])
        np.testing.assert_array_equal(g_away, [9, 9, 9, 9])


class TestSimulatePlayoffPairRealLegs:
    def test_neither_leg_played_stays_fully_simulated(self):
        """Regression guard: today's (correct) future-simulation behavior,
        with no real legs to apply, must be unchanged."""
        matches_df = pd.DataFrame(BASE_ROWS)
        winners = _simulate(matches_df, reference_date="2026-11-01")

        assert set(winners) <= {"TeamA", "TeamB"}
        assert len(set(winners)) == 2  # genuinely random across 300 draws, not deterministic

    def test_leg1_played_for_real_blowout_overwhelms_simulated_leg2(self):
        """Leg 1 played for real (a decisive blowout); leg 2 not yet played.
        A blowout margin no realistically-sampled leg 2 could overturn proves
        leg 1's real score, not a fabricated one, decided the outcome."""
        rows = [*BASE_ROWS, _row("TeamB", "TeamA", 0, 15, "2026-11-15")]
        winners = _simulate(pd.DataFrame(rows), reference_date="2026-11-20")

        assert set(winners) == {"TeamA"}

    def test_leg1_played_for_real_close_result_leg2_still_varies_by_draw(self):
        """Leg 1 played for real, but as an exact draw -- leaves the tie
        undecided, so the winner should still vary by draw, proving leg 2 is
        still being simulated (not also frozen)."""
        rows = [*BASE_ROWS, _row("TeamB", "TeamA", 1, 1, "2026-11-15")]
        winners = _simulate(pd.DataFrame(rows), reference_date="2026-11-20")

        assert len(set(winners)) == 2

    def test_both_legs_played_for_real_fully_deterministic(self):
        """Both legs played for real -- the winner must match the real
        aggregate exactly, with NO randomness left at all, regardless of
        n_draws/seed (every draw's sampled score is entirely overridden)."""
        rows = [
            *BASE_ROWS,
            _row("TeamB", "TeamA", 1, 3, "2026-11-15"),  # leg 1: TeamA wins 3-1
            _row("TeamA", "TeamB", 2, 0, "2026-11-22"),  # leg 2: TeamA wins 2-0
        ]
        matches_df = pd.DataFrame(rows)

        for seed in (0, 1, 2):
            winners = _simulate(matches_df, "2026-11-25", n_draws=50, seed=seed)
            assert set(winners) == {"TeamA"}

    def test_single_leg_decider_fully_real_bypasses_the_coin_flip(self):
        """legs=1's tiebreak coin flip only fires on a tie -- a decisive real
        single leg must produce a deterministic winner without ever
        consulting it."""
        rows = [*BASE_ROWS, _row("TeamB", "TeamA", 0, 4, "2026-11-15")]
        winners = _simulate(pd.DataFrame(rows), reference_date="2026-11-20", legs=1)

        assert set(winners) == {"TeamA"}
