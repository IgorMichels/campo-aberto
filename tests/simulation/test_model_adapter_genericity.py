"""Proves src.simulation.simulate's orchestration (round-robin, playoff,
cascade, tabulate) is genuinely model-agnostic, not just poisson_home
running through an extra layer of indirection: this drives a full
round-robin + playoff competition through simulate_competition using
tests/models/_dummy_adapter.py's DummyAdapter -- one team param ("skill"),
one shared param ("home_boost"), no attack/defense/eta/beta_home/rho
anywhere, no Dixon-Coles-style low-score correction.

Per this repo's established convention (see test_attach_team_strengths.py),
CmdStanMCMC itself is never mocked -- a hand-built fake exposing only
`.stan_variables()` stands in for it.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.registry import MODEL_REGISTRY
from src.simulation.config import (
    CompetitionConfig,
    PlayoffPhaseConfig,
    RoundRobinPhaseConfig,
    SpotConfig,
)
from src.simulation.simulate import simulate_competition
from tests.models._dummy_adapter import ADAPTER as DUMMY


class _FakeMCMC:
    def __init__(self, stan_vars: dict):
        self._stan_vars = stan_vars

    def stan_variables(self) -> dict:
        return self._stan_vars


def _fake_mcmc(skills: dict[str, float], home_boost=0.2):
    """skills: {team: skill}, dict order defines Stan index. A single
    posterior draw, repeated -- simulate_competition resamples it with
    replacement up to n_draws."""
    teams = list(skills)
    skill = np.array([[skills[t] for t in teams]])
    mcmc = _FakeMCMC({"skill": skill, "home_boost": np.array([home_boost])})
    return mcmc, teams


def _matches_df(teams: list[str], competition="Test League", season=2025):
    """Registers every team for this competition+season with no games played
    yet -- fixtures.split_fixtures derives the full double round-robin
    combinatorially from this roster alone."""
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
        legs=2,
        spots=(SpotConfig(name="playoff_winner", result="winner"),),
    )
    return CompetitionConfig(name="Test League", n_teams=4, phases=(league, playoff))


def test_simulate_competition_runs_end_to_end_with_a_non_poisson_home_adapter(monkeypatch):
    monkeypatch.setitem(MODEL_REGISTRY, "dummy", DUMMY)
    mcmc, teams = _fake_mcmc({"A": 1.0, "B": 0.5, "C": -0.5, "D": -1.0})
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
        model="dummy",
    )

    assert set(result["team"]) == set(teams)
    assert result["prob_title"].sum() == pytest.approx(1.0)
    assert result["prob_playoff_winner"].sum() == pytest.approx(1.0)
    # the strongest team should win the title far more often than the weakest
    by_team = result.set_index("team")
    assert by_team.loc["A", "prob_title"] > by_team.loc["D", "prob_title"]
    # _attach_team_strengths reports exactly the dummy's own declared shape
    assert list(result.columns) == [
        "team",
        "expected_position",
        "prob_title",
        "prob_rebaixamento",
        "prob_playoff_winner",
        "model",
        "skill",
        "home_boost",
    ]
    assert (result["model"] == "dummy").all()
