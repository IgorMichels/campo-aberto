"""Candidate model: same as negbin_home (Negative-Binomial instead of
Dixon-Coles-adjusted Poisson), but with a single dispersion parameter `phi`
shared between home and away goals, instead of two independent
phi_home/phi_away. Tests whether letting home/away disperse differently
(negbin_home) is actually worth the extra free parameter, or whether one
shared overdispersion ratio predicts just as well. See
src/models/stan_models/negbin_home_shared_phi.stan for the Stan side, and
src/models/backtest.py for the out-of-sample comparison against negbin_home
and poisson_home.

Never registered as DEFAULT_MODEL in src/models/registry.py -- this is a
tournament candidate (see plans/model_stats_page.md Step 0), not a
production model.
"""

import os

import numpy as np

STAN_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "stan_models", "negbin_home_shared_phi.stan")
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


def _simulate_scores(mu_home, mu_away, phi, rng):
    """Independent Negative-Binomial(mu, phi) draws for home/away goals, both
    sides sharing the same phi -- no rejection sampling needed (unlike
    Dixon-Coles), since there's no score correlation term here, only
    overdispersion. Same Stan<->numpy conversion as negbin_home:
    neg_binomial_2(mu, phi) (mean mu, variance mu + mu^2/phi) equals
    negative_binomial(n=phi, p=phi/(phi+mu)).

    mu_home, mu_away: shape (n_draws, n_matches). phi: shape (n_draws,) --
    reshaped to (n_draws, 1) here since it doesn't broadcast against
    (n_draws, n_matches) on its own.
    """
    phi_col = phi[:, None]
    p_home = phi_col / (phi_col + mu_home)
    p_away = phi_col / (phi_col + mu_away)
    home_goals = rng.negative_binomial(phi_col, p_home)
    away_goals = rng.negative_binomial(phi_col, p_away)
    return home_goals.astype(np.int64), away_goals.astype(np.int64)


class NegBinHomeSharedPhiAdapter:
    name = "negbin_home_shared_phi"
    stan_file = STAN_FILE
    team_param_names = ("attack", "defense")
    shared_param_names = ("eta", "beta_home", "phi")

    def sample_scores(self, team_params, shared_params, home_idx, away_idx, rng):
        mu_home, mu_away = _match_rates(
            team_params["attack"],
            team_params["defense"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        return _simulate_scores(mu_home, mu_away, shared_params["phi"], rng)

    def sample_scores_single(self, team_params, shared_params, home_idx, away_idx, rng):
        mu_home, mu_away = _match_rates_per_draw(
            team_params["attack"],
            team_params["defense"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        home_goals, away_goals = _simulate_scores(
            mu_home[:, None], mu_away[:, None], shared_params["phi"], rng
        )
        return home_goals[:, 0], away_goals[:, 0]


ADAPTER = NegBinHomeSharedPhiAdapter()
