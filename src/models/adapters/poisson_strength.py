"""Candidate model: a single per-team strength parameter (attack == defense
in the Dixon-Coles-adjusted independent Poisson formula), instead of
poisson_home's separate attack/defense per team. Tests whether attack/defense
as independent random effects earns its keep out-of-sample, or whether a
single strength term predicts as well with half the free team-level
parameters. See src/models/stan_models/poisson_strength.stan for the Stan side, and
src/models/backtest.py for the out-of-sample comparison against poisson_home.

Never registered as DEFAULT_MODEL in src/models/registry.py -- this is a
tournament candidate (see plans/model_stats_page.md Step 0), not a
production model.
"""

import os

import numpy as np

from src.models.adapters._dixon_coles import simulate_scores

STAN_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "stan_models", "poisson_strength.stan")
)


def _match_rates(strength, eta, beta_home, home_idx, away_idx):
    """strength: (n_draws, T). eta, beta_home: (n_draws,). home_idx, away_idx: (n_matches,)."""
    mu_home = np.exp(
        strength[:, home_idx] - strength[:, away_idx] + eta[:, None] + beta_home[:, None]
    )
    mu_away = np.exp(strength[:, away_idx] - strength[:, home_idx] + eta[:, None])
    return mu_home, mu_away


def _match_rates_per_draw(strength, eta, beta_home, home_idx, away_idx):
    """Same as _match_rates, but home_idx/away_idx name a *different* single match per draw
    (shape (n_draws,)) instead of a shared batch of fixtures -- used for playoffs, where
    who plays whom depends on that draw's own outcome so far.
    """
    row = np.arange(strength.shape[0])
    mu_home = np.exp(strength[row, home_idx] - strength[row, away_idx] + eta + beta_home)
    mu_away = np.exp(strength[row, away_idx] - strength[row, home_idx] + eta)
    return mu_home, mu_away


class PoissonStrengthAdapter:
    name = "poisson_strength"
    stan_file = STAN_FILE
    team_param_names = ("strength",)
    shared_param_names = ("eta", "beta_home", "rho")

    def sample_scores(self, team_params, shared_params, home_idx, away_idx, rng):
        mu_home, mu_away = _match_rates(
            team_params["strength"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        return simulate_scores(mu_home, mu_away, shared_params["rho"], rng)

    def sample_scores_single(self, team_params, shared_params, home_idx, away_idx, rng):
        mu_home, mu_away = _match_rates_per_draw(
            team_params["strength"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        home_goals, away_goals = simulate_scores(
            mu_home[:, None], mu_away[:, None], shared_params["rho"], rng
        )
        return home_goals[:, 0], away_goals[:, 0]


ADAPTER = PoissonStrengthAdapter()
