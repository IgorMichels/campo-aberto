"""Today's production model: independent Poisson attack/defense per team,
shared home-advantage (eta/beta_home), Dixon-Coles low-score correction
(rho). Math moved here unchanged from src/simulation/simulate.py -- see
src/models/poisson_home.stan for the Stan side of the same formula.
"""

import os

import numpy as np

STAN_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "poisson_home.stan"))


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


def _simulate_scores(mu_home, mu_away, rho, rng):
    """Batch Dixon-Coles-adjusted Poisson score sampling via rejection sampling.

    Args:
        mu_home, mu_away: shape (n_draws, n_matches).
        rho: shape (n_draws,).

    Returns:
        (home_goals, away_goals), each shape (n_draws, n_matches), int.

    Vectorized over (posterior draws x remaining fixtures): draw (x, y) from
    the two *independent* Poissons (numpy's native, vectorized, C-level
    rng.poisson), and accept/reject against the Dixon-Coles tau(x, y)
    correction (dc_log_prob's equivalent in poisson_home.stan), which only
    reweights the 4 cells x,y in {0,1}. This was tried two ways:
      - Gathering only the still-"pending" (draw, fixture) cells each round via
        boolean/fancy indexing (`arr[rows, cols]`). Fewer values processed per
        round, but fancy-indexed gather/scatter on a scattered subset of a
        large 2-D array is cache-hostile -- this was *slower* than the dense
        grid approach it replaced (measured: ~98s vs ~37s for 100k draws x
        ~200 fixtures).
      - Recomputing the *whole* (n_draws, n_fixtures) array every round with
        np.where, leaving already-accepted cells alone instead of compacting
        them out. Wastes some resampling on already-accepted cells, but every
        operation stays a plain elementwise, contiguous-memory numpy op --
        this is the one below, and it's ~5x faster than even the original
        dense (draws x fixtures x 11 x 11) grid + cumsum approach, a common
        way to vectorize this kind of score sampling, since it never
        materializes a per-score-pair grid at all. It also has no truncation
        (rng.poisson has no upper bound), unlike a fixed max-goals grid.
    Acceptance per round is 1/bound, and bound is close to 1 whenever rho is
    (our prior is ~N(0, 0.1)), so this converges in a handful of rounds in
    practice.
    """
    n_draws, n_matches = mu_home.shape
    rho = np.broadcast_to(rho[:, None], (n_draws, n_matches))

    tau00 = 1 - mu_home * mu_away * rho
    tau01 = 1 + mu_home * rho
    tau10 = 1 + mu_away * rho
    tau11 = 1 - rho
    bound = np.maximum.reduce([np.ones_like(mu_home), tau00, tau01, tau10, tau11])

    home_goals = np.zeros((n_draws, n_matches), dtype=np.int64)
    away_goals = np.zeros((n_draws, n_matches), dtype=np.int64)
    pending = np.ones((n_draws, n_matches), dtype=bool)

    while pending.any():
        x = rng.poisson(mu_home)
        y = rng.poisson(mu_away)

        tau = np.ones_like(mu_home)
        tau = np.where((x == 0) & (y == 0), np.maximum(tau00, 0), tau)
        tau = np.where((x == 0) & (y == 1), np.maximum(tau01, 0), tau)
        tau = np.where((x == 1) & (y == 0), np.maximum(tau10, 0), tau)
        tau = np.where((x == 1) & (y == 1), np.maximum(tau11, 0), tau)

        accept = pending & (rng.random((n_draws, n_matches)) < (tau / bound))
        home_goals = np.where(accept, x, home_goals)
        away_goals = np.where(accept, y, away_goals)
        pending &= ~accept

    return home_goals, away_goals


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
        return _simulate_scores(mu_home, mu_away, shared_params["rho"], rng)

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
            mu_home[:, None], mu_away[:, None], shared_params["rho"], rng
        )
        return home_goals[:, 0], away_goals[:, 0]


ADAPTER = PoissonHomeAdapter()
