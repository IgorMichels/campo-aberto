"""Candidate model: Bivariate Poisson (Karlis & Ntzoufras 2003) instead of
poisson_home's Dixon-Coles-adjusted independent Poisson. Home/away goals
share a latent Poisson component (home = X1 + X3, away = X2 + X3, X1/X2/X3
independent Poisson), inducing genuine scoreline covariance (Cov(home, away)
= lambda3) instead of patching 4 specific low-score cells. Tests whether
this more principled correlation mechanism out-predicts Dixon-Coles' ad hoc
patch. See src/models/stan_models/bivariate_poisson_home.stan for the Stan side, and
src/models/backtest.py for the out-of-sample comparison against poisson_home.

Never registered as DEFAULT_MODEL in src/models/registry.py -- this is a
tournament candidate (see plans/model_stats_page.md Step 0), not a
production model.
"""

import os

import numpy as np

STAN_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "stan_models", "bivariate_poisson_home.stan")
)


def _match_rates(attack, defense, eta, beta_home, home_idx, away_idx):
    """attack, defense: (n_draws, T). eta, beta_home: (n_draws,). home_idx, away_idx: (n_matches,).
    Returns (lambda1, lambda2), the home/away Poisson rates before the shared
    latent component lambda3 is added in."""
    lambda1 = np.exp(attack[:, home_idx] - defense[:, away_idx] + eta[:, None] + beta_home[:, None])
    lambda2 = np.exp(attack[:, away_idx] - defense[:, home_idx] + eta[:, None])
    return lambda1, lambda2


def _match_rates_per_draw(attack, defense, eta, beta_home, home_idx, away_idx):
    """Same as _match_rates, but home_idx/away_idx name a *different* single match per draw
    (shape (n_draws,)) instead of a shared batch of fixtures -- used for playoffs, where
    who plays whom depends on that draw's own outcome so far.
    """
    row = np.arange(attack.shape[0])
    lambda1 = np.exp(attack[row, home_idx] - defense[row, away_idx] + eta + beta_home)
    lambda2 = np.exp(attack[row, away_idx] - defense[row, home_idx] + eta)
    return lambda1, lambda2


def _simulate_scores(lambda1, lambda2, lambda3, rng):
    """Exact draw from the bivariate Poisson via its trivariate reduction:
    X = X1 + X3, Y = X2 + X3, X1/X2/X3 independent Poisson -- this
    construction *is* the distribution (Karlis & Ntzoufras 2003), not an
    approximation of it, so no rejection sampling is needed (unlike
    Dixon-Coles' multiplicative low-score patch).

    lambda1, lambda2: shape (n_draws, n_matches). lambda3: shape (n_draws,)
    -- a single shared-covariance rate per draw, broadcast to every match
    (each match still draws its own independent X3, just from the same rate).
    """
    x3 = rng.poisson(np.broadcast_to(lambda3[:, None], lambda1.shape))
    x1 = rng.poisson(lambda1)
    x2 = rng.poisson(lambda2)
    return (x1 + x3).astype(np.int64), (x2 + x3).astype(np.int64)


class BivariatePoissonHomeAdapter:
    name = "bivariate_poisson_home"
    stan_file = STAN_FILE
    team_param_names = ("attack", "defense")
    shared_param_names = ("eta", "beta_home", "lambda3")

    def sample_scores(self, team_params, shared_params, home_idx, away_idx, rng):
        lambda1, lambda2 = _match_rates(
            team_params["attack"],
            team_params["defense"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        return _simulate_scores(lambda1, lambda2, shared_params["lambda3"], rng)

    def sample_scores_single(self, team_params, shared_params, home_idx, away_idx, rng):
        lambda1, lambda2 = _match_rates_per_draw(
            team_params["attack"],
            team_params["defense"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        home_goals, away_goals = _simulate_scores(
            lambda1[:, None], lambda2[:, None], shared_params["lambda3"], rng
        )
        return home_goals[:, 0], away_goals[:, 0]


ADAPTER = BivariatePoissonHomeAdapter()
