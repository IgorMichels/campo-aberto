"""Candidate model: same independent Poisson attack/defense per team,
shared home-advantage (eta/beta_home) as poisson_home_no_rho, but attack/
defense are drawn from one of 4 fixed hierarchical-prior groups (see
src.models.data._prior_groups) instead of a flat normal(0, 1) -- lets a
team with little history shrink toward its own group's learned mean (e.g.
a team new to the Serie A/B pyramid shrinks toward other historical
debutants, not toward the global zero or an established team's mean).
Each group's SCALE is fixed at 1 (same spread as poisson_home_no_rho for
every team); only the group MEAN is hierarchical. An earlier version also
estimated a per-group sigma_attack/sigma_defense, but that over-shrank the
small groups (elevador-A-B, elevador-B-C have few teams per checkpoint) and
backtested worse than poisson_home -- see src/models/stan_models/hierarchical_home.stan
for the full rationale. See src/models/backtest.py for the out-of-sample
comparison against the other registered models.

Since rho never enters this model's likelihood or score sampling, this
adapter's math is identical to poisson_home_no_rho's -- the hierarchy only
changes the Stan-side prior/fit structure, never how scores are drawn from
a posterior team_params dict, so simulate_scores is reused unchanged.

Never registered as DEFAULT_MODEL in src/models/registry.py -- this is a
tournament candidate (see plans/model_stats_page.md Step 0), not a
production model.
"""

import os

import numpy as np

from src.models.adapters._plain_poisson import simulate_scores

STAN_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "stan_models", "hierarchical_home.stan")
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


class HierarchicalHomeAdapter:
    name = "hierarchical_home"
    stan_file = STAN_FILE
    team_param_names = ("attack", "defense")
    shared_param_names = ("eta", "beta_home")

    def sample_scores(self, team_params, shared_params, home_idx, away_idx, rng):
        mu_home, mu_away = _match_rates(
            team_params["attack"],
            team_params["defense"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        return simulate_scores(mu_home, mu_away, rng)

    def sample_scores_single(self, team_params, shared_params, home_idx, away_idx, rng):
        mu_home, mu_away = _match_rates_per_draw(
            team_params["attack"],
            team_params["defense"],
            shared_params["eta"],
            shared_params["beta_home"],
            home_idx,
            away_idx,
        )
        home_goals, away_goals = simulate_scores(mu_home[:, None], mu_away[:, None], rng)
        return home_goals[:, 0], away_goals[:, 0]


ADAPTER = HierarchicalHomeAdapter()
