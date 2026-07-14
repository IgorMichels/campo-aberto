"""Candidate model: the same attack/defense/eta/beta_home structure as
poisson_home, but with the Dixon-Coles low-score correlation correction
replaced by Negative-Binomial overdispersion (one dispersion parameter per
side, phi_home/phi_away, since home and away scoring may be dispersed
differently rather than sharing one ratio). Tests the standard alternative
explanation for football's low-score correlation/overdispersion (a
dispersion parameter instead of an ad hoc 4-cell patch). See
src/models/stan_models/negbin_home.stan for the Stan side, and src/models/backtest.py
for the out-of-sample comparison against poisson_home.

Never registered as DEFAULT_MODEL in src/models/registry.py -- this is a
tournament candidate (see plans/model_stats_page.md Step 0), not a
production model.
"""

import os

import numpy as np

STAN_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "stan_models", "negbin_home.stan")
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


def _simulate_scores(mu_home, mu_away, phi_home, phi_away, rng):
    """Independent Negative-Binomial(mu, phi) draws for home/away goals -- no
    rejection sampling needed (unlike Dixon-Coles), since there's no score
    correlation term here, only overdispersion. Stan's neg_binomial_2(mu, phi)
    (mean mu, variance mu + mu^2/phi) is exactly numpy's
    negative_binomial(n=phi, p=phi/(phi+mu)) (mean n(1-p)/p, variance
    n(1-p)/p^2) -- both reduce to mean=mu, variance=mu+mu^2/phi.

    mu_home, mu_away: shape (n_draws, n_matches). phi_home, phi_away: shape
    (n_draws,) -- reshaped to (n_draws, 1) here since they don't broadcast
    against (n_draws, n_matches) on their own.
    """
    phi_home_col = phi_home[:, None]
    phi_away_col = phi_away[:, None]
    p_home = phi_home_col / (phi_home_col + mu_home)
    p_away = phi_away_col / (phi_away_col + mu_away)
    home_goals = rng.negative_binomial(phi_home_col, p_home)
    away_goals = rng.negative_binomial(phi_away_col, p_away)
    return home_goals.astype(np.int64), away_goals.astype(np.int64)


class NegBinHomeAdapter:
    name = "negbin_home"
    stan_file = STAN_FILE
    team_param_names = ("attack", "defense")
    shared_param_names = ("eta", "beta_home", "phi_home", "phi_away")

    def sample_scores(self, team_params, shared_params, home_idx, away_idx, rng):
        mu_home, mu_away = _match_rates(
            team_params["attack"],
            team_params["defense"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        return _simulate_scores(
            mu_home, mu_away, shared_params["phi_home"], shared_params["phi_away"], rng
        )

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
            mu_home[:, None],
            mu_away[:, None],
            shared_params["phi_home"],
            shared_params["phi_away"],
            rng,
        )
        return home_goals[:, 0], away_goals[:, 0]


ADAPTER = NegBinHomeAdapter()
