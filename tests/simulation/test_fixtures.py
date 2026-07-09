"""Unit tests for src.simulation.fixtures's season_teams/split_fixtures --
the combinatorial fixture derivation described in that module's docstring.
"""

import pandas as pd

from src.simulation.fixtures import season_teams, split_fixtures


def _match(competition, season, dt, home, away, home_goals=1, away_goals=0):
    return {
        "competition": competition,
        "season": season,
        "match_datetime": pd.Timestamp(dt),
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def test_season_teams_is_the_sorted_union_of_home_and_away():
    df = pd.DataFrame(
        [
            _match("Serie A", 2026, "2026-01-01", "Zeta FC", "Alpha FC"),
            _match("Serie A", 2026, "2026-01-08", "Beta FC", "Zeta FC"),
        ]
    )

    assert season_teams(df, "Serie A", 2026) == ["Alpha FC", "Beta FC", "Zeta FC"]


def test_split_fixtures_splits_played_from_remaining_by_reference_date():
    df = pd.DataFrame(
        [
            _match("Serie A", 2026, "2026-01-01", "Alpha FC", "Beta FC", 2, 1),
            _match("Serie A", 2026, "2026-07-01", "Beta FC", "Alpha FC", 1, 1),
        ]
    )

    played, remaining, teams = split_fixtures(
        df, "Serie A", 2026, reference_date=pd.Timestamp("2026-02-01")
    )

    assert played == [("Alpha FC", "Beta FC", 2, 1)]
    assert ("Beta FC", "Alpha FC") in remaining
    assert teams == ["Alpha FC", "Beta FC"]


def test_split_fixtures_does_not_treat_a_stale_dated_unplayed_row_as_played():
    """A postponed match keeps its original scheduled match_datetime (see
    src/ingestion/brazil/build_treated_dataset.py) -- that date alone must
    never be enough to count it as played once it's on/before reference_date."""
    df = pd.DataFrame(
        [
            _match("Serie A", 2026, "2026-01-01", "Alpha FC", "Beta FC", 2, 1),
            {
                "competition": "Serie A",
                "season": 2026,
                "match_datetime": pd.Timestamp("2026-01-15"),  # stale, pre-postponement date
                "home_team": "Beta FC",
                "away_team": "Alpha FC",
                "home_goals": None,
                "away_goals": None,
            },
        ]
    )

    played, remaining, _ = split_fixtures(
        df, "Serie A", 2026, reference_date=pd.Timestamp("2026-02-01")
    )

    assert played == [("Alpha FC", "Beta FC", 2, 1)]
    assert ("Beta FC", "Alpha FC") in remaining


def test_split_fixtures_restricts_to_an_explicit_roster_when_given():
    df = pd.DataFrame(
        [
            _match("Serie A", 2026, "2026-01-01", "Alpha FC", "Beta FC", 2, 1),
            _match("Serie A", 2026, "2026-01-01", "Gamma FC", "Delta FC", 0, 0),
        ]
    )

    played, remaining, teams = split_fixtures(
        df,
        "Serie A",
        2026,
        reference_date=pd.Timestamp("2026-02-01"),
        teams=["Alpha FC", "Beta FC"],
    )

    assert teams == ["Alpha FC", "Beta FC"]
    assert played == [("Alpha FC", "Beta FC", 2, 1)]
    assert remaining == [("Beta FC", "Alpha FC")]
