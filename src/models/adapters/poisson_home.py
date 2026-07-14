"""Today's production model: independent Poisson attack/defense per team,
shared home-advantage (eta/beta_home), Dixon-Coles low-score correction
(rho). Math moved here unchanged from src/simulation/simulate.py -- see
src/models/stan_models/poisson_home.stan for the Stan side of the same formula.
"""

import os

import numpy as np

from src.models.adapters._dixon_coles import simulate_scores

STAN_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "stan_models", "poisson_home.stan")
)


def _match_rates(attack, defense, eta, beta_home, home_idx, away_idx):
    """attack, defense: (n_draws, T). eta, beta_home: (n_draws,). home_idx, away_idx: (n_matches,)."""
    mu_home = np.exp(attack[:, home_idx] - defense[:, away_idx] + eta[:, None] + beta_home[:, None])
    mu_away = np.exp(attack[:, away_idx] - defense[:, home_idx] + eta[:, None])
    return mu_home, mu_away


def _match_rates_per_draw(attack, defense, eta, beta_home, home_idx, away_idx):
    """Same as _match_rates, but home_idx/away_idx name a *different* single match per draw
    (shape (n_draws,)) instead of a shared batch of fixtures -- used for playoffs, where
    who plays whom depends on that draw's own outcome so far.
    """
    row = np.arange(attack.shape[0])
    mu_home = np.exp(attack[row, home_idx] - defense[row, away_idx] + eta + beta_home)
    mu_away = np.exp(attack[row, away_idx] - defense[row, home_idx] + eta)
    return mu_home, mu_away


class PoissonHomeAdapter:
    name = "poisson_home"
    stan_file = STAN_FILE
    team_param_names = ("attack", "defense")
    shared_param_names = ("eta", "beta_home", "rho")

    def sample_scores(self, team_params, shared_params, home_idx, away_idx, rng):
        mu_home, mu_away = _match_rates(
            team_params["attack"],
            team_params["defense"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        return simulate_scores(mu_home, mu_away, shared_params["rho"], rng)

    def sample_scores_single(self, team_params, shared_params, home_idx, away_idx, rng):
        mu_home, mu_away = _match_rates_per_draw(
            team_params["attack"],
            team_params["defense"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        home_goals, away_goals = simulate_scores(
            mu_home[:, None], mu_away[:, None], shared_params["rho"], rng
        )
        return home_goals[:, 0], away_goals[:, 0]


ADAPTER = PoissonHomeAdapter()
