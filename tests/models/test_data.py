"""Unit tests for build_stan_data/load_stan_data (src/models/data.py).

Deliberately uses a fictional league/teams, not Brazilian club names, to
prove the schema handling has no country-specific assumptions baked in --
see the module docstring: "Works with any CSV following the ... schema
shared across countries".
"""

import pandas as pd
import pytest

from src.models.data import (
    ARRIVED_FROM_BELOW,
    ELEVATOR,
    STAYED_SECOND,
    STAYED_TOP,
    _prior_groups,
    _time_weight,
    build_stan_data,
    load_stan_data,
)


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


def test_build_stan_data_drops_rows_with_no_result_yet():
    """matches.csv can now carry scheduled/postponed rows with no result yet
    (see src/ingestion/brazil/build_treated_dataset.py) -- only played
    matches ("para simular devemos considerar apenas os resultados dos jogos
    já disputados") ever inform the fit."""
    df = pd.DataFrame(
        [
            _match("Generic League", 2026, "2026-01-01", "Alpha FC", "Beta FC", 1, 0),
            _match("Generic League", 2026, "2026-07-01", "Beta FC", "Alpha FC", None, None),
        ]
    )

    stan_data, teams = build_stan_data(df, reference_date=pd.Timestamp("2026-01-01"))

    assert stan_data["N"] == 1
    assert teams == ["Alpha FC", "Beta FC"]  # both still known via the played match
    assert stan_data["y_i"] == [1]
    assert stan_data["y_j"] == [0]


def test_build_stan_data_casts_goals_back_to_int_after_filtering():
    """Filtering out the None-goals rows upcasts home_goals/away_goals to
    float64 (pandas' NaN handling) -- must be cast back to int."""
    df = pd.DataFrame(
        [
            _match("Generic League", 2026, "2026-01-01", "Alpha FC", "Beta FC", 2, 1),
            _match("Generic League", 2026, "2026-07-01", "Beta FC", "Alpha FC", None, None),
        ]
    )

    stan_data, _ = build_stan_data(df, reference_date=pd.Timestamp("2026-01-01"))

    assert stan_data["y_i"] == [2]
    assert isinstance(stan_data["y_i"][0], int)
    assert isinstance(stan_data["y_j"][0], int)


def test_reference_date_defaults_to_the_played_rows_latest_match():
    """A future scheduled fixture's date must never be picked as the default
    reference_date -- only a played match can define "as of now"."""
    df = pd.DataFrame(
        [
            _match("Generic League", 2026, "2026-01-01", "Zeta FC", "Alpha FC", 1, 0),
            _match("Generic League", 2026, "2026-01-08", "Beta FC", "Zeta FC", 2, 2),
            _match("Generic League", 2026, "2026-12-01", "Alpha FC", "Beta FC", None, None),
        ]
    )

    stan_data, _ = build_stan_data(df)  # no reference_date passed

    # Same as the all-played case: latest played match (2026-01-08) is 0
    # weeks ago, not the unplayed 2026-12-01 fixture.
    assert stan_data["game_weight"] == pytest.approx([0.5 ** (1 / 25), 1.0])


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


# --- "group" (hierarchical-prior classification, see _prior_groups) ---


def _mobility_df() -> pd.DataFrame:
    """5 teams covering all 4 groups at once:
    - Alpha FC: Serie A in both 2025 and 2026 -> STAYED_TOP.
    - Beta FC: Serie B in both 2025 and 2026 -> STAYED_SECOND.
    - Gamma FC: Serie B in 2025, Serie A in 2026 (promoted) -> ELEVATOR.
    - Delta FC: Serie A in 2025, Serie B in 2026 (relegated) -> ELEVATOR too
      (same group regardless of direction).
    - Epsilon FC: only plays in 2026 (Serie B), no 2025 data at all ->
      ARRIVED_FROM_BELOW.
    """
    return pd.DataFrame(
        [
            _match("Serie A", 2025, "2025-01-01", "Alpha FC", "Delta FC", 1, 0),
            _match("Serie B", 2025, "2025-01-08", "Beta FC", "Gamma FC", 2, 1),
            _match("Serie A", 2026, "2026-01-01", "Alpha FC", "Gamma FC", 1, 1),
            _match("Serie B", 2026, "2026-01-08", "Beta FC", "Delta FC", 0, 2),
            _match("Serie B", 2026, "2026-01-15", "Epsilon FC", "Beta FC", 1, 1),
        ]
    )


def test_prior_group_assigns_all_4_groups_by_season_transition():
    stan_data, teams = build_stan_data(
        _mobility_df(), reference_date=pd.Timestamp("2026-01-15"), max_weeks_ago=1000
    )

    assert teams == ["Alpha FC", "Beta FC", "Delta FC", "Epsilon FC", "Gamma FC"]
    group_by_team = dict(zip(teams, stan_data["group"]))
    assert group_by_team["Alpha FC"] == STAYED_TOP
    assert group_by_team["Beta FC"] == STAYED_SECOND
    assert group_by_team["Delta FC"] == ELEVATOR  # Serie A 2025 -> Serie B 2026
    assert group_by_team["Gamma FC"] == ELEVATOR  # Serie B 2025 -> Serie A 2026
    assert group_by_team["Epsilon FC"] == ARRIVED_FROM_BELOW


def test_prior_group_is_aligned_with_the_teams_list_order():
    stan_data, teams = build_stan_data(
        _mobility_df(), reference_date=pd.Timestamp("2026-01-15"), max_weeks_ago=1000
    )

    assert stan_data["group"] == [STAYED_TOP, STAYED_SECOND, ELEVATOR, ARRIVED_FROM_BELOW, ELEVATOR]
    assert len(stan_data["group"]) == len(teams)


def test_prior_group_raises_with_a_single_competition():
    df = pd.DataFrame(
        [
            _match("Generic League", 2025, "2025-01-01", "Alpha FC", "Beta FC", 1, 0),
            _match("Generic League", 2026, "2026-01-01", "Beta FC", "Alpha FC", 2, 2),
        ]
    )

    with pytest.raises(ValueError):
        _prior_groups(df, ["Alpha FC", "Beta FC"])


def test_prior_group_raises_with_more_than_2_competitions():
    df = pd.DataFrame(
        [
            _match("Serie A", 2026, "2026-01-01", "Alpha FC", "Beta FC", 1, 0),
            _match("Serie B", 2026, "2026-01-02", "Gamma FC", "Delta FC", 2, 2),
            _match("Serie C", 2026, "2026-01-03", "Epsilon FC", "Zeta FC", 0, 0),
        ]
    )

    with pytest.raises(ValueError):
        _prior_groups(df, ["Alpha FC", "Beta FC", "Gamma FC", "Delta FC", "Epsilon FC", "Zeta FC"])


def test_build_stan_data_omits_group_when_not_exactly_2_competitions():
    """build_stan_data itself must stay generic over any number of
    competitions (see module docstring) -- only hierarchical_home.stan
    actually declares "group" in its data block, so every other model must
    keep working on data _prior_groups can't classify. The ValueError is
    swallowed here and surfaces later only if hierarchical_home is fit on
    such data (at Stan's own data validation)."""
    df = pd.DataFrame(
        [
            _match("Generic League", 2025, "2025-01-01", "Alpha FC", "Beta FC", 1, 0),
            _match("Generic League", 2026, "2026-01-01", "Beta FC", "Alpha FC", 2, 2),
        ]
    )

    stan_data, teams = build_stan_data(df, reference_date=pd.Timestamp("2026-01-01"))

    assert teams == ["Alpha FC", "Beta FC"]
    assert "group" not in stan_data
