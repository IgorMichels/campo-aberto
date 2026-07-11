"""Unit tests for simulate.py's private `_attach_team_strengths` helper: it
reads the FULL posterior (via `mcmc_fit.stan_variables()`), not the
n_draws-resampled subset `simulate_competition` uses elsewhere for season
odds, and broadcasts each posterior-mean team/shared param (as declared by
the given ModelAdapter) onto every row of the result DataFrame, plus a
`model` column naming the adapter.

Per this repo's established convention (see tests/models/test_fit.py's
`_FakeMCMC` and tests/simulation/test_round_robin_legs.py's `DrawParams`
fakes), `CmdStanMCMC` itself is never mocked -- a hand-built fake object
exposing only `.stan_variables()` (the one method this helper calls) stands
in for it instead.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.adapters.poisson_home import ADAPTER as POISSON_HOME
from src.simulation.simulate import _attach_team_strengths
from tests.models._dummy_adapter import ADAPTER as DUMMY


class _FakeMCMC:
    def __init__(self, stan_vars: dict):
        self._stan_vars = stan_vars

    def stan_variables(self) -> dict:
        return self._stan_vars


def _fake_mcmc(attack: list[list[float]], defense: list[list[float]], eta, beta_home, rho):
    """attack/defense: (n_draws, n_teams) nested lists. eta/beta_home/rho: (n_draws,) lists."""
    return _FakeMCMC(
        {
            "attack": np.array(attack),
            "defense": np.array(defense),
            "eta": np.array(eta),
            "beta_home": np.array(beta_home),
            "rho": np.array(rho),
        }
    )


def test_attaches_posterior_mean_attack_defense_per_team():
    teams = ["Alpha FC", "Beta FC", "Gamma FC"]
    mcmc = _fake_mcmc(
        attack=[[0.1, 0.4, -0.2], [0.3, 0.6, 0.0]],
        defense=[[0.05, -0.1, 0.2], [0.15, 0.1, 0.4]],
        eta=[0.2, 0.4],
        beta_home=[0.1, 0.3],
        rho=[-0.02, -0.06],
    )
    df = pd.DataFrame({"team": ["Beta FC", "Alpha FC", "Gamma FC"], "expected_position": [1, 2, 3]})

    out = _attach_team_strengths(df, mcmc, teams, POISSON_HOME)

    # Posterior mean across the two draws, per team, mapped by df["team"] --
    # row order (Beta, Alpha, Gamma) must not matter.
    assert out.set_index("team").loc["Alpha FC", "attack"] == pytest.approx(0.2)  # mean(0.1, 0.3)
    assert out.set_index("team").loc["Beta FC", "attack"] == pytest.approx(0.5)  # mean(0.4, 0.6)
    assert out.set_index("team").loc["Gamma FC", "attack"] == pytest.approx(-0.1)  # mean(-0.2, 0.0)
    assert out.set_index("team").loc["Alpha FC", "defense"] == pytest.approx(
        0.1
    )  # mean(0.05, 0.15)
    assert out.set_index("team").loc["Beta FC", "defense"] == pytest.approx(0.0)  # mean(-0.1, 0.1)
    assert out.set_index("team").loc["Gamma FC", "defense"] == pytest.approx(0.3)  # mean(0.2, 0.4)


def test_attaches_scalar_posterior_mean_eta_beta_home_rho_broadcast_on_every_row():
    teams = ["Alpha FC", "Beta FC"]
    mcmc = _fake_mcmc(
        attack=[[0.0, 0.0], [0.0, 0.0]],
        defense=[[0.0, 0.0], [0.0, 0.0]],
        eta=[0.2, 0.4],
        beta_home=[0.1, 0.3],
        rho=[-0.02, -0.06],
    )
    df = pd.DataFrame({"team": ["Alpha FC", "Beta FC"], "expected_position": [1, 2]})

    out = _attach_team_strengths(df, mcmc, teams, POISSON_HOME)

    assert np.allclose(out["eta"], 0.3)  # mean(0.2, 0.4)
    assert np.allclose(out["beta_home"], 0.2)  # mean(0.1, 0.3)
    assert np.allclose(out["rho"], -0.04)  # mean(-0.02, -0.06)


def test_original_result_columns_are_preserved_and_input_not_mutated():
    teams = ["Alpha FC", "Beta FC"]
    mcmc = _fake_mcmc(
        attack=[[0.1, 0.2]],
        defense=[[0.0, 0.0]],
        eta=[0.3],
        beta_home=[0.1],
        rho=[0.0],
    )
    df = pd.DataFrame(
        {"team": ["Alpha FC", "Beta FC"], "expected_position": [1.5, 1.5], "prob_title": [0.5, 0.5]}
    )

    out = _attach_team_strengths(df, mcmc, teams, POISSON_HOME)

    assert list(out.columns) == [
        "team",
        "expected_position",
        "prob_title",
        "model",
        "attack",
        "defense",
        "eta",
        "beta_home",
        "rho",
    ]
    assert (out["model"] == "poisson_home").all()
    assert "attack" not in df.columns  # input DataFrame left untouched


def test_a_differently_shaped_adapter_produces_exactly_its_own_declared_columns():
    """Proves the helper is genuinely generic, not just poisson_home with an
    unused adapter argument -- a one-team-param, one-shared-param adapter
    (see tests/models/_dummy_adapter.py) gets exactly that shape back, no
    attack/defense/eta/beta_home/rho anywhere."""
    teams = ["Alpha FC", "Beta FC"]
    mcmc = _FakeMCMC(
        {
            "skill": np.array([[0.2, 0.6], [0.4, 0.8]]),
            "home_boost": np.array([0.1, 0.3]),
        }
    )
    df = pd.DataFrame({"team": ["Alpha FC", "Beta FC"], "expected_position": [1, 2]})

    out = _attach_team_strengths(df, mcmc, teams, DUMMY)

    assert list(out.columns) == ["team", "expected_position", "model", "skill", "home_boost"]
    assert (out["model"] == "dummy").all()
    assert out.set_index("team").loc["Alpha FC", "skill"] == pytest.approx(0.3)  # mean(0.2, 0.4)
    assert out.set_index("team").loc["Beta FC", "skill"] == pytest.approx(0.7)  # mean(0.6, 0.8)
    assert np.allclose(out["home_boost"], 0.2)  # mean(0.1, 0.3)
