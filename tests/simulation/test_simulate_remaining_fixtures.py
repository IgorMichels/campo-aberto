"""Regression test for _simulate_remaining_all_draws with an empty fixture list.

A round_robin phase that's already fully played as of reference_date (e.g. the
season's final backtest checkpoint, right after its last match) legitimately
has zero remaining fixtures. `np.array([])` with no explicit dtype defaults to
float64, and numpy's fancy indexing (`attack[:, home_idx]`) rejects a float
index array outright -- this used to crash run_rounds.py with
`IndexError: arrays used as indices must be of integer (or boolean) type`
whenever a competition's reference_dates walked past the round-robin phase's
true end (see src/simulation/simulate.py's dtype=np.int64 fix).
"""

import numpy as np

from src.simulation.simulate import _simulate_remaining_all_draws


def _dummy_posteriors(n_draws: int, n_teams: int):
    rng = np.random.default_rng(0)
    attack = rng.normal(size=(n_draws, n_teams))
    defense = rng.normal(size=(n_draws, n_teams))
    eta = rng.normal(size=n_draws)
    beta_home = rng.normal(size=n_draws)
    rho = np.zeros(n_draws)
    return attack, defense, eta, beta_home, rho, rng


def test_empty_remaining_fixtures_returns_empty_arrays_instead_of_crashing():
    attack, defense, eta, beta_home, rho, rng = _dummy_posteriors(n_draws=5, n_teams=3)
    team_index = {"A": 0, "B": 1, "C": 2}

    home_goals, away_goals = _simulate_remaining_all_draws(
        [], attack, defense, eta, beta_home, rho, team_index, rng
    )

    assert home_goals.shape == (5, 0)
    assert away_goals.shape == (5, 0)


def test_nonempty_remaining_fixtures_still_simulates_normally():
    attack, defense, eta, beta_home, rho, rng = _dummy_posteriors(n_draws=5, n_teams=3)
    team_index = {"A": 0, "B": 1, "C": 2}

    home_goals, away_goals = _simulate_remaining_all_draws(
        [("A", "B"), ("B", "C")], attack, defense, eta, beta_home, rho, team_index, rng
    )

    assert home_goals.shape == (5, 2)
    assert away_goals.shape == (5, 2)
    assert np.all(home_goals >= 0)
    assert np.all(away_goals >= 0)
