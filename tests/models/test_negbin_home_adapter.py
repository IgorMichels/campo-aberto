"""Unit tests for NegBinHomeAdapter.sample_scores/.sample_scores_single --
mirrors tests/models/test_poisson_home_adapter.py's shape/mean tests, plus
tests specific to Negative-Binomial overdispersion (variance inflated above
the Poisson case, and independently so for phi_home vs. phi_away -- see
src/models/adapters/negbin_home.py).
"""

import os

import numpy as np
import pytest

from src.models.adapters.negbin_home import ADAPTER


def test_declares_its_own_name_and_parameter_shape():
    assert ADAPTER.name == "negbin_home"
    assert ADAPTER.team_param_names == ("attack", "defense")
    assert ADAPTER.shared_param_names == ("eta", "beta_home", "phi_home", "phi_away")
    assert os.path.isfile(ADAPTER.stan_file)
    assert ADAPTER.stan_file.endswith("negbin_home.stan")


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
        "phi_home": np.full(n_draws, 10.0),
        "phi_away": np.full(n_draws, 10.0),
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


def test_sample_scores_matches_the_mean_regardless_of_phi():
    """neg_binomial_2(mu, phi) has mean mu no matter what phi is -- unlike
    variance, the mean shouldn't depend on the dispersion parameters at all."""
    rng = np.random.default_rng(1)
    n_draws = 50000
    team_params = {
        "attack": np.full((n_draws, 2), [0.3, -0.1]),
        "defense": np.full((n_draws, 2), [0.0, 0.0]),
    }
    shared_params = {
        "eta": np.zeros(n_draws),
        "beta_home": np.full(n_draws, 0.2),
        "phi_home": np.full(n_draws, 3.0),
        "phi_away": np.full(n_draws, 3.0),
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


def test_sample_scores_variance_is_inflated_and_differs_between_phi_home_and_phi_away():
    """With attack/defense/eta/beta_home all 0, mu_home == mu_away == 1, so
    any difference in variance is attributable purely to phi_home vs.
    phi_away. Expected variance = mu + mu^2/phi (Stan's neg_binomial_2):
    phi_home=2 -> 1 + 1/2 = 1.5 (clearly above the Poisson variance of 1),
    phi_away=50 -> 1 + 1/50 = 1.02 (close to Poisson) -- the two sides should
    disperse very differently even though their means are identical."""
    n_draws = 200000
    team_params = {"attack": np.zeros((n_draws, 2)), "defense": np.zeros((n_draws, 2))}
    shared_params = {
        "eta": np.zeros(n_draws),
        "beta_home": np.zeros(n_draws),
        "phi_home": np.full(n_draws, 2.0),
        "phi_away": np.full(n_draws, 50.0),
    }
    home_idx = np.array([0], dtype=np.int64)
    away_idx = np.array([1], dtype=np.int64)

    home_goals, away_goals = ADAPTER.sample_scores(
        team_params, shared_params, home_idx, away_idx, np.random.default_rng(2)
    )

    home_variance = home_goals[:, 0].var()
    away_variance = away_goals[:, 0].var()
    assert home_variance == pytest.approx(1.5, rel=0.1)
    assert away_variance == pytest.approx(1.02, abs=0.1)
    assert home_variance > away_variance


def test_sample_scores_single_returns_one_score_per_draw():
    rng = np.random.default_rng(3)
    n_draws, n_teams = 30, 4
    team_params = _random_team_params(n_draws, n_teams, rng)
    shared_params = {
        "eta": rng.normal(size=n_draws),
        "beta_home": rng.normal(size=n_draws),
        "phi_home": np.full(n_draws, 10.0),
        "phi_away": np.full(n_draws, 10.0),
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
    from the identical per-draw mu_home/mu_away/phi -- just two different
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
        "phi_home": np.full(n_draws, 8.0),
        "phi_away": np.full(n_draws, 8.0),
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
