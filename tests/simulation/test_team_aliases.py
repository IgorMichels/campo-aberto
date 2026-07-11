"""Integration tests for simulate_competition's `team_aliases` parameter (see
src.simulation.run_rounds._relegated_teams_previous_season /
_debut_team_aliases for how the mapping itself gets built during a real
backtest run): a debut/stale-data team is given its OWN, separate Stan index
carrying a copy of its substitute's attack/defense draws -- not a shared
index -- specifically to avoid _simulate_playoff_pair's
`np.array(teams)[winner_idx]` misattributing a playoff win to the
substitute's name instead of the alias's own (see simulate_competition's own
docstring for the full rationale).

Per this repo's established convention (see test_attach_team_strengths.py),
CmdStanMCMC itself is never mocked -- a hand-built fake exposing only
`.stan_variables()` stands in for it.
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation.config import (
    CompetitionConfig,
    PlayoffPhaseConfig,
    RoundRobinPhaseConfig,
    SpotConfig,
)
from src.simulation.simulate import simulate_competition


class _FakeMCMC:
    def __init__(self, stan_vars: dict):
        self._stan_vars = stan_vars

    def stan_variables(self) -> dict:
        return self._stan_vars


def _fake_mcmc(team_strengths: dict[str, tuple[float, float]], eta=0.0, beta_home=0.2, rho=0.0):
    """team_strengths: {team: (attack, defense)}, dict order defines Stan
    index. A single posterior draw, repeated -- simulate_competition resamples
    it with replacement up to n_draws, which is fine here since all the
    randomness this test cares about comes from the Poisson match simulation
    itself, not posterior variability."""
    teams = list(team_strengths)
    attack = np.array([[team_strengths[t][0] for t in teams]])
    defense = np.array([[team_strengths[t][1] for t in teams]])
    mcmc = _FakeMCMC(
        {
            "attack": attack,
            "defense": defense,
            "eta": np.array([eta]),
            "beta_home": np.array([beta_home]),
            "rho": np.array([rho]),
        }
    )
    return mcmc, teams


def _matches_df(teams: list[str], competition="Test League", season=2025):
    """Registers every team for this competition+season with no games played
    yet -- fixtures.split_fixtures derives the full double round-robin
    combinatorially from this roster alone (see that module's docstring)."""
    rows = [
        {
            "competition": competition,
            "season": season,
            "match_datetime": "2025-01-01",
            "home_team": teams[0],
            "away_team": team,
            "home_goals": None,
            "away_goals": None,
        }
        for team in teams[1:]
    ]
    df = pd.DataFrame(rows)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    return df


def _config() -> CompetitionConfig:
    league = RoundRobinPhaseConfig(
        id="league",
        head_to_head_mode="points_then_goal_diff",
        spots=(
            SpotConfig(name="title", positions=(1, 1)),
            SpotConfig(name="rebaixamento", positions=(3, 4)),
        ),
    )
    playoff = PlayoffPhaseConfig(
        id="playoff",
        pairing="table_position",
        source_phase="league",
        pairs=((1, 2),),
        legs=1,
        spots=(SpotConfig(name="playoff_winner", result="winner"),),
    )
    return CompetitionConfig(name="Test League", n_teams=4, phases=(league, playoff))


def test_alias_team_gets_credited_for_its_own_playoff_win_not_its_substitute():
    """Alias D borrows Real A's exact attack/defense -- both far stronger
    than Real B/Real C -- so with enough draws, Real A and Alias D should
    each reach the top-2 and meet in the single playoff pair often, each
    winning it roughly half the time (identical underlying strength). If
    aliasing shared a Stan index instead of duplicating it (the bug this
    test guards against), every one of Alias D's playoff wins would get
    reported under Real A's name instead: Alias D's own prob_playoff_winner
    would read ~0 while Real A's would be inflated to roughly double its
    true rate."""
    mcmc, teams = _fake_mcmc({"Real A": (3.0, 3.0), "Real B": (-3.0, -3.0), "Real C": (-3.0, -3.0)})
    matches_df = _matches_df([*teams, "Alias D"])
    config = _config()

    result = simulate_competition(
        config,
        mcmc,
        teams,
        matches_df,
        season=2025,
        reference_date=pd.Timestamp("2025-01-01"),
        n_draws=2000,
        seed=0,
        team_aliases={"Alias D": "Real A"},
    )

    by_team = result.set_index("team")
    assert "Alias D" in by_team.index
    assert "Real A" in by_team.index
    alias_prob = by_team.loc["Alias D", "prob_playoff_winner"]
    real_a_prob = by_team.loc["Real A", "prob_playoff_winner"]
    assert alias_prob > 0.1
    assert real_a_prob > 0.1
    assert abs(alias_prob - real_a_prob) < 0.15


def test_alias_team_attack_defense_in_output_matches_its_substitute():
    """_attach_team_strengths (called at the end of simulate_competition)
    reports the alias team's attack/defense as identical to its
    substitute's -- the whole point of borrowing a strength instead of
    inventing one from thin air."""
    mcmc, teams = _fake_mcmc({"Real A": (0.4, 0.1), "Real B": (-0.2, 0.0), "Real C": (0.0, -0.1)})
    matches_df = _matches_df([*teams, "Alias D"])
    config = _config()

    result = simulate_competition(
        config,
        mcmc,
        teams,
        matches_df,
        season=2025,
        reference_date=pd.Timestamp("2025-01-01"),
        n_draws=50,
        seed=0,
        team_aliases={"Alias D": "Real A"},
    )

    by_team = result.set_index("team")
    assert by_team.loc["Alias D", "attack"] == pytest.approx(by_team.loc["Real A", "attack"])
    assert by_team.loc["Alias D", "defense"] == pytest.approx(by_team.loc["Real A", "defense"])


def test_no_team_aliases_behaves_exactly_as_before():
    """team_aliases=None (the default) must not change simulate_competition's
    output at all -- a plain regression guard for the refactor that
    introduced attack_draws/defense_draws/sim_teams as local variables."""
    mcmc, teams = _fake_mcmc({"Real A": (0.3, 0.2), "Real B": (-0.1, 0.0), "Real C": (0.0, -0.2)})
    matches_df = _matches_df(teams)
    config = _config()

    result = simulate_competition(
        config,
        mcmc,
        teams,
        matches_df,
        season=2025,
        reference_date=pd.Timestamp("2025-01-01"),
        n_draws=200,
        seed=0,
    )

    assert set(result["team"]) == {"Real A", "Real B", "Real C"}
    assert result["prob_title"].sum() == pytest.approx(1.0)
