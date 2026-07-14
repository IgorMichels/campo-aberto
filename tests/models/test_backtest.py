"""Unit tests for src/models/backtest.py's pure-Python pieces --
_checkpoint_dates, _score_match, aggregate_metrics, calibration_table --
without ever fitting real Stan (mirrors tests/simulation/test_run_rounds.py's
hand-built matches.csv fixture convention). walk_forward_backtest itself
(which does call Stan) is exercised manually, not here -- see
src/models/backtest.py's own docstring / Usage line.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.models.backtest import (
    _checkpoint_cache_path,
    _checkpoint_dates,
    _load_cached_checkpoint,
    _log_season_done,
    _save_checkpoint_cache,
    _score_match,
    aggregate_metrics,
    calibration_table,
)
from tests.models._dummy_adapter import ADAPTER as DUMMY_ADAPTER


def _matches_df(rows: list[dict]) -> pd.DataFrame:
    rows = [{"home_goals": 1, "away_goals": 0, **row} for row in rows]
    df = pd.DataFrame(rows)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    return df


# --- _checkpoint_dates ---


def test_checkpoint_dates_includes_a_pre_window_checkpoint_before_the_first_match():
    df = _matches_df([{"competition": "Serie A", "season": 2022, "match_datetime": "2022-04-10"}])

    dates = _checkpoint_dates(df, start_season=2022, cadence_days=7)

    assert dates[0] < pd.Timestamp("2022-04-10")
    assert dates[-1] >= pd.Timestamp("2022-04-10")


def test_checkpoint_dates_skips_a_candidate_with_no_new_match():
    """Two matches 8 weeks apart (2022-06-05 is exactly first_day + 8*7 days)
    produce only the 3 checkpoints that actually capture something -- the
    pre-window one, the one landing on the first match, and the one landing
    on the second -- skipping every intervening weekly candidate with no new
    match, same "only candidates with news" rule as
    src.simulation.run_rounds.reference_dates."""
    df = _matches_df(
        [
            {"competition": "Serie A", "season": 2022, "match_datetime": "2022-04-10"},
            {"competition": "Serie A", "season": 2022, "match_datetime": "2022-06-05"},
        ]
    )

    dates = _checkpoint_dates(df, start_season=2022, cadence_days=7)

    assert dates == [
        pd.Timestamp("2022-04-03"),
        pd.Timestamp("2022-04-10"),
        pd.Timestamp("2022-06-05"),
    ]


def test_checkpoint_dates_filters_by_start_season():
    df = _matches_df(
        [
            {"competition": "Serie A", "season": 2021, "match_datetime": "2021-04-10"},
            {"competition": "Serie A", "season": 2022, "match_datetime": "2022-05-01"},
        ]
    )

    dates = _checkpoint_dates(df, start_season=2022, cadence_days=7)

    assert dates == [pd.Timestamp("2022-04-24"), pd.Timestamp("2022-05-01")]


def test_checkpoint_dates_returns_empty_when_no_played_matches_in_start_season():
    df = _matches_df([{"competition": "Serie A", "season": 2021, "match_datetime": "2021-04-10"}])

    assert _checkpoint_dates(df, start_season=2022, cadence_days=7) == []


# --- _log_season_done ---


def test_log_season_done_prints_the_model_and_season_and_returns_a_float(capsys):
    returned = _log_season_done("negbin_home", 2022, time.monotonic(), time.monotonic())

    captured = capsys.readouterr()
    assert "negbin_home" in captured.out
    assert "season 2022" in captured.out
    assert isinstance(returned, float)


# --- checkpoint cache ---


def test_checkpoint_cache_path_is_namespaced_by_model_and_date(tmp_path):
    path = _checkpoint_cache_path(str(tmp_path), "negbin_home", pd.Timestamp("2026-07-08"))

    assert path == str(tmp_path / "negbin_home" / "2026_07_08.csv")


def test_save_and_load_checkpoint_cache_round_trips_a_record(tmp_path):
    path = str(tmp_path / "poisson_home" / "2026_07_08.csv")
    records = [
        {
            "home_team": "Alpha FC",
            "away_team": "Beta FC",
            "match_datetime": pd.Timestamp("2026-07-05"),
            "competition": "Serie A",
            "season": 2026,
            "actual_outcome": "home",
            "home": 0.6,
            "draw": 0.25,
            "away": 0.15,
            "reference_date": pd.Timestamp("2026-07-08"),
        }
    ]

    _save_checkpoint_cache(path, records)
    loaded = _load_cached_checkpoint(path)

    assert len(loaded) == 1
    assert loaded[0]["home_team"] == "Alpha FC"
    assert loaded[0]["home"] == pytest.approx(0.6)
    assert loaded[0]["match_datetime"] == pd.Timestamp("2026-07-05")
    assert loaded[0]["reference_date"] == pd.Timestamp("2026-07-08")


def test_save_and_load_checkpoint_cache_round_trips_an_empty_checkpoint(tmp_path):
    """A checkpoint whose window had no match to score is still a completed
    checkpoint -- it must cache (and reload) as an empty list, not blow up on
    a columnless CSV."""
    path = str(tmp_path / "poisson_home" / "2026_02_01.csv")

    _save_checkpoint_cache(path, [])
    loaded = _load_cached_checkpoint(path)

    assert loaded == []


# --- _score_match (uses DummyAdapter, no Stan) ---


def _match_row(home_team, away_team, home_goals, away_goals):
    return pd.Series(
        {
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "match_datetime": pd.Timestamp("2024-01-01"),
            "competition": "Serie A",
            "season": 2024,
        }
    )


def test_score_match_returns_none_when_a_team_is_unknown_to_this_fit():
    team_params = {"skill": np.zeros((10, 2))}
    shared_params = {"home_boost": np.zeros(10)}
    team_index = {"A": 0, "B": 1}

    result = _score_match(
        DUMMY_ADAPTER,
        team_params,
        shared_params,
        team_index,
        _match_row("A", "C", 1, 0),
        np.random.default_rng(0),
    )

    assert result is None


def test_score_match_home_team_much_stronger_gives_high_home_win_probability():
    n_draws = 5000
    team_params = {"skill": np.full((n_draws, 2), [3.0, -3.0])}
    shared_params = {"home_boost": np.zeros(n_draws)}
    team_index = {"A": 0, "B": 1}

    result = _score_match(
        DUMMY_ADAPTER,
        team_params,
        shared_params,
        team_index,
        _match_row("A", "B", 2, 0),
        np.random.default_rng(0),
    )

    assert result["home"] > 0.9
    assert result["actual_outcome"] == "home"


# --- aggregate_metrics / calibration_table (hand-built records) ---


def _record(home, draw, away, actual, competition="Serie A", season=2024):
    return {
        "home": home,
        "draw": draw,
        "away": away,
        "actual_outcome": actual,
        "competition": competition,
        "season": season,
    }


def test_aggregate_metrics_perfect_predictions_has_zero_brier():
    records = [_record(1.0, 0.0, 0.0, "home"), _record(0.0, 0.0, 1.0, "away")]

    metrics = aggregate_metrics(records)

    assert metrics["all"]["brier"] == pytest.approx(0.0)
    assert metrics["all"]["direction_accuracy"] == 1.0


def test_aggregate_metrics_confident_correct_model_beats_the_uniform_baseline():
    records = [_record(0.9, 0.05, 0.05, "home")] * 10

    metrics = aggregate_metrics(records)

    assert metrics["all"]["brier"] < metrics["all"]["brier_uniform_baseline"]


def test_aggregate_metrics_breaks_down_by_competition_and_season():
    records = [
        _record(1, 0, 0, "home", competition="Serie A", season=2023),
        _record(0, 0, 1, "away", competition="Serie B", season=2024),
    ]

    metrics = aggregate_metrics(records)

    assert set(metrics["by_competition"]) == {"Serie A", "Serie B"}
    assert set(metrics["by_competition_season"]) == {"Serie A 2023", "Serie B 2024"}


def test_calibration_table_bins_predicted_probabilities():
    records = [_record(0.95, 0.03, 0.02, "home")] * 20

    table = calibration_table(records, n_bins=10)

    top_bin = table[-1]
    assert top_bin["n"] == 20
    assert top_bin["observed_freq"] == pytest.approx(1.0)


def test_calibration_table_empty_bin_reports_none():
    """The 3 pooled (predicted, actual) pairs from this one match all land in
    the [0.0, 0.1) or [0.9, 1.0) bins -- every middle bin, e.g. [0.5, 0.6),
    stays empty."""
    records = [_record(0.05, 0.05, 0.9, "away")]

    table = calibration_table(records, n_bins=10)

    middle_bin = table[5]
    assert middle_bin["bin_lo"] == pytest.approx(0.5)
    assert middle_bin["n"] == 0
    assert middle_bin["mean_predicted"] is None
