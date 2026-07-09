"""Unit tests for build_stan_data/load_stan_data (src/models/data.py).

Deliberately uses a fictional league/teams, not Brazilian club names, to
prove the schema handling has no country-specific assumptions baked in --
see the module docstring: "Works with any CSV following the ... schema
shared across countries".
"""

import pandas as pd
import pytest

from src.models.data import _time_weight, build_stan_data, load_stan_data


def _match(competition, season, dt, home, away, home_goals, away_goals):
    return {
        "competition": competition,
        "season": season,
        "match_datetime": pd.Timestamp(dt),
        "venue": "Somewhere",
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def test_team_index_is_alphabetically_sorted_and_one_indexed():
    df = pd.DataFrame(
        [
            _match("Generic League", 2026, "2026-01-01", "Zeta FC", "Alpha FC", 1, 0),
            _match("Generic League", 2026, "2026-01-08", "Beta FC", "Zeta FC", 2, 2),
        ]
    )

    stan_data, teams = build_stan_data(df, reference_date=pd.Timestamp("2026-01-08"))

    assert teams == ["Alpha FC", "Beta FC", "Zeta FC"]
    assert stan_data["T"] == 3
    assert stan_data["N"] == 2
    # Zeta FC (index 3) hosts Alpha FC (index 1) in the first match.
    assert stan_data["team_i"] == [3, 2]
    assert stan_data["team_j"] == [1, 3]
    assert stan_data["y_i"] == [1, 2]
    assert stan_data["y_j"] == [0, 2]


def test_reference_date_defaults_to_the_dataframes_latest_match():
    df = pd.DataFrame(
        [
            _match("Generic League", 2026, "2026-01-01", "Zeta FC", "Alpha FC", 1, 0),
            _match("Generic League", 2026, "2026-01-08", "Beta FC", "Zeta FC", 2, 2),
        ]
    )

    stan_data, _ = build_stan_data(df)  # no reference_date passed

    # Same as passing reference_date=2026-01-08 explicitly: the later match is
    # 0 weeks ago (weight 1.0), the earlier one exactly 1 week ago.
    assert stan_data["game_weight"] == pytest.approx([0.5 ** (1 / 25), 1.0])


def test_decay_formula_is_pure_exponential_half_life():
    weeks_ago = pd.Series([0, 25, 50])

    weights = _time_weight(weeks_ago, half_life_weeks=25)

    assert weights.tolist() == pytest.approx([1.0, 0.5, 0.25])


def test_half_life_is_overridable():
    df = pd.DataFrame([_match("X League", 2026, "2025-01-01", "A FC", "B FC", 1, 1)])
    reference = pd.Timestamp("2025-01-01") + pd.Timedelta(weeks=10)

    stan_data, _ = build_stan_data(df, reference_date=reference, half_life_weeks=10)

    assert stan_data["game_weight"] == pytest.approx([0.5])  # exactly one half-life old


def test_matches_older_than_cutoff_are_dropped_entirely():
    df = pd.DataFrame(
        [
            _match("X League", 2026, "2024-01-01", "A FC", "B FC", 1, 0),  # ancient
            _match("X League", 2026, "2026-01-01", "B FC", "A FC", 2, 2),
        ]
    )
    reference = pd.Timestamp("2026-01-01")

    stan_data, teams = build_stan_data(df, reference_date=reference, max_weeks_ago=10)

    assert stan_data["N"] == 1
    assert teams == ["A FC", "B FC"]  # both still present via the remaining match


def test_cutoff_can_remove_a_team_from_the_roster_entirely():
    df = pd.DataFrame(
        [
            _match(
                "X League", 2026, "2020-01-01", "Ghost FC", "A FC", 1, 0
            ),  # Ghost FC's only match
            _match("X League", 2026, "2026-01-01", "A FC", "B FC", 2, 2),
        ]
    )
    reference = pd.Timestamp("2026-01-01")

    stan_data, teams = build_stan_data(df, reference_date=reference, max_weeks_ago=10)

    assert teams == ["A FC", "B FC"]
    assert stan_data["T"] == 2


def test_load_stan_data_reads_csv_and_parses_match_datetime(tmp_path):
    csv_path = tmp_path / "matches.csv"
    pd.DataFrame([_match("Generic League", 2026, "2026-01-01", "A FC", "B FC", 1, 0)]).to_csv(
        csv_path, index=False
    )

    stan_data, teams = load_stan_data(str(csv_path))

    assert teams == ["A FC", "B FC"]
    assert stan_data["N"] == 1
    assert stan_data["game_weight"] == pytest.approx([1.0])  # the file's own latest match


def test_load_stan_data_forwards_kwargs_to_build_stan_data(tmp_path):
    csv_path = tmp_path / "matches.csv"
    pd.DataFrame(
        [
            _match("Generic League", 2026, "2024-01-01", "A FC", "B FC", 1, 0),
            _match("Generic League", 2026, "2026-01-01", "B FC", "A FC", 2, 2),
        ]
    ).to_csv(csv_path, index=False)

    stan_data, teams = load_stan_data(
        str(csv_path), reference_date=pd.Timestamp("2026-01-01"), max_weeks_ago=10
    )

    assert stan_data["N"] == 1
    assert teams == ["A FC", "B FC"]
