"""Unit tests for PoissonHomeAdapter.sample_scores/.sample_scores_single --
today's production Dixon-Coles-Poisson score sampling, moved out of
src/simulation/simulate.py into its own adapter (see
src/models/adapters/poisson_home.py). Previously this math was only
exercised indirectly through simulate.py's higher-level tests; this is its
first direct test.
"""

import os

import numpy as np
import pytest

from src.models.adapters.poisson_home import ADAPTER


def test_declares_its_own_name_and_parameter_shape():
    assert ADAPTER.name == "poisson_home"
    assert ADAPTER.team_param_names == ("attack", "defense")
    assert ADAPTER.shared_param_names == ("eta", "beta_home", "rho")
    assert os.path.isfile(ADAPTER.stan_file)
    assert ADAPTER.stan_file.endswith("poisson_home.stan")


def _random_team_params(n_draws, n_teams, rng):
    return {
        "attack": rng.normal(size=(n_draws, n_teams)),
        "defense": rng.normal(size=(n_draws, n_teams)),
    }


def test_sample_scores_returns_nonnegative_ints_with_the_right_shape():
    rng = np.random.default_rng(0)
    n_draws, n_teams, n_matches = 50, 4, 3
    team_params = _random_team_params(n_draws, n_teams, rng)
    shared_params = {
        "eta": rng.normal(size=n_draws),
        "beta_home": rng.normal(size=n_draws),
        "rho": np.full(n_draws, -0.05),
    }
    home_idx = np.array([0, 1, 2], dtype=np.int64)
    away_idx = np.array([1, 2, 3], dtype=np.int64)

    home_goals, away_goals = ADAPTER.sample_scores(
        team_params, shared_params, home_idx, away_idx, rng
    )

    assert home_goals.shape == (n_draws, n_matches)
    assert away_goals.shape == (n_draws, n_matches)
    assert home_goals.dtype == np.int64
    assert np.all(home_goals >= 0)
    assert np.all(away_goals >= 0)


def test_sample_scores_matches_the_poisson_means_when_rho_is_zero():
    """With rho=0 the Dixon-Coles correction is a no-op (tau is 1 everywhere,
    every candidate draw is accepted) -- home/away goals reduce to plain
    independent Poisson draws, so their empirical mean over enough draws
    should track mu_home/mu_away."""
    rng = np.random.default_rng(1)
    n_draws = 20000
    team_params = {
        "attack": np.full((n_draws, 2), [0.3, -0.1]),
        "defense": np.full((n_draws, 2), [0.0, 0.0]),
    }
    shared_params = {
        "eta": np.zeros(n_draws),
        "beta_home": np.full(n_draws, 0.2),
        "rho": np.zeros(n_draws),
    }
    home_idx = np.array([0], dtype=np.int64)
    away_idx = np.array([1], dtype=np.int64)
    expected_mu_home = np.exp(0.3 - 0.0 + 0.0 + 0.2)
    expected_mu_away = np.exp(-0.1 - 0.0 + 0.0)

    home_goals, away_goals = ADAPTER.sample_scores(
        team_params, shared_params, home_idx, away_idx, rng
    )

    assert home_goals.mean() == pytest.approx(expected_mu_home, rel=0.05)
    assert away_goals.mean() == pytest.approx(expected_mu_away, rel=0.05)


def test_sample_scores_rho_shifts_the_scoreless_draw_frequency():
    """A negative rho (this model's fitted regime, rho ~ N(0, 0.1)) inflates
    tau(0,0) above the uncorrected Poisson product, so 0-0 draws should be
    strictly more frequent than with rho=0 for the same mu_home/mu_away."""
    n_draws = 20000
    team_params = {
        "attack": np.zeros((n_draws, 2)),
        "defense": np.zeros((n_draws, 2)),
    }
    shared = {"eta": np.zeros(n_draws), "beta_home": np.zeros(n_draws)}
    home_idx = np.array([0], dtype=np.int64)
    away_idx = np.array([1], dtype=np.int64)

    home0, away0 = ADAPTER.sample_scores(
        team_params,
        {**shared, "rho": np.zeros(n_draws)},
        home_idx,
        away_idx,
        np.random.default_rng(2),
    )
    home_neg, away_neg = ADAPTER.sample_scores(
        team_params,
        {**shared, "rho": np.full(n_draws, -0.3)},
        home_idx,
        away_idx,
        np.random.default_rng(2),
    )

    freq_00_zero_rho = np.mean((home0 == 0) & (away0 == 0))
    freq_00_neg_rho = np.mean((home_neg == 0) & (away_neg == 0))
    assert freq_00_neg_rho > freq_00_zero_rho


def test_sample_scores_single_returns_one_score_per_draw():
    rng = np.random.default_rng(3)
    n_draws, n_teams = 30, 4
    team_params = _random_team_params(n_draws, n_teams, rng)
    shared_params = {
        "eta": rng.normal(size=n_draws),
        "beta_home": rng.normal(size=n_draws),
        "rho": np.full(n_draws, -0.05),
    }
    home_idx = rng.integers(0, n_teams, size=n_draws)
    away_idx = (home_idx + 1) % n_teams  # never the same team home vs away

    home_goals, away_goals = ADAPTER.sample_scores_single(
        team_params, shared_params, home_idx, away_idx, rng
    )

    assert home_goals.shape == (n_draws,)
    assert away_goals.shape == (n_draws,)
    assert np.all(home_goals >= 0)
    assert np.all(away_goals >= 0)


def test_sample_scores_and_sample_scores_single_agree_on_a_shared_fixture():
    """Broadcasting the same (home, away) pair to every draw via sample_scores
    vs. asking sample_scores_single for that exact pair on every draw draws
    from the identical per-draw mu_home/mu_away/rho -- just two different
    call shapes (round-robin batch vs. per-draw playoff) over the same
    underlying math, so their empirical means should agree."""
    n_draws, n_teams = 20000, 3
    team_params = {
        "attack": np.full((n_draws, n_teams), [0.4, -0.2, 0.1]),
        "defense": np.full((n_draws, n_teams), [0.0, 0.1, -0.1]),
    }
    shared_params = {
        "eta": np.zeros(n_draws),
        "beta_home": np.full(n_draws, 0.25),
        "rho": np.full(n_draws, -0.1),
    }

    home_batch, away_batch = ADAPTER.sample_scores(
        team_params,
        shared_params,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.random.default_rng(42),
    )
    home_single, away_single = ADAPTER.sample_scores_single(
        team_params,
        shared_params,
        np.zeros(n_draws, dtype=np.int64),
        np.ones(n_draws, dtype=np.int64),
        np.random.default_rng(42),
    )

    assert home_batch[:, 0].mean() == pytest.approx(home_single.mean(), rel=0.05)
    assert away_batch[:, 0].mean() == pytest.approx(away_single.mean(), rel=0.05)
