"""Unit tests for src.simulation.run_rounds's pure-Python helpers
(round_reference_dates, load_configs_by_season) -- not main()'s Stan fit +
simulate_competition loop, which needs a real matches CSV and a compiled Stan
model to run end-to-end.
"""

import pandas as pd

from src.simulation.config import CompetitionConfig
from src.simulation.run_rounds import load_configs_by_season, round_reference_dates


def _matches_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    return df


def test_round_reference_dates_groups_consecutive_days_into_one_round():
    df = _matches_df(
        [
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-01"},
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-02"},
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-09"},
        ]
    )

    dates = round_reference_dates(df, "Serie A", 2025)

    assert dates == [pd.Timestamp("2025-03-02"), pd.Timestamp("2025-03-09")]


def test_round_reference_dates_uses_the_last_day_of_each_round():
    """A round spanning three consecutive days (gap of exactly 1 between each) is
    still one round, and its reference_date is the round's last day -- the point
    at which every one of that round's results is actually known."""
    df = _matches_df(
        [
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-01"},
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-02"},
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-03"},
        ]
    )

    assert round_reference_dates(df, "Serie A", 2025) == [pd.Timestamp("2025-03-03")]


def test_round_reference_dates_filters_by_competition_and_season():
    df = _matches_df(
        [
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-01"},
            {"competition": "Serie B", "season": 2025, "match_datetime": "2025-03-15"},
            {"competition": "Serie A", "season": 2026, "match_datetime": "2026-03-01"},
        ]
    )

    assert round_reference_dates(df, "Serie A", 2025) == [pd.Timestamp("2025-03-01")]


def test_round_reference_dates_returns_empty_list_when_no_matches():
    df = _matches_df([{"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-01"}])

    assert round_reference_dates(df, "Serie B", 2025) == []


def test_load_configs_by_season_groups_the_real_configs_by_filename_season_suffix():
    """Exercises the real configs/*.yaml walk (the same one src.site.export_site_data
    relies on for competition/season discovery) instead of a fixture directory --
    staying in sync with whatever configs/serie_*_<year>.yaml files actually exist
    is the point of this test."""
    configs_by_season = load_configs_by_season([2025, 2026])

    assert set(configs_by_season) == {2025, 2026}
    for season, configs in configs_by_season.items():
        assert configs, f"no configs found for {season}"
        assert all(isinstance(c, CompetitionConfig) for c in configs)
        assert {c.name for c in configs} == {"Serie A", "Serie B"}


def test_load_configs_by_season_ignores_seasons_not_requested():
    configs_by_season = load_configs_by_season([2025])

    assert 2025 in configs_by_season
    assert 2026 not in configs_by_season


def test_load_configs_by_season_returns_empty_dict_for_a_season_with_no_configs():
    assert load_configs_by_season([1999]) == {}
