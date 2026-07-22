"""Unit tests for src.simulation.run_rounds's pure-Python helpers
(reference_dates, load_configs_by_season) -- not main()'s Stan fit +
simulate_competition loop, which needs a real matches CSV and a compiled Stan
model to run end-to-end.
"""

import numpy as np
import pandas as pd

from src.simulation.config import (
    CompetitionConfig,
    GuaranteedSlotConfig,
    RoundRobinPhaseConfig,
    SpotConfig,
)
from src.simulation.run_rounds import (
    _already_computed,
    _debut_team_aliases,
    _relegated_teams_previous_season,
    _write_log,
    latest_checkpoint_date,
    load_configs_by_season,
    reference_dates,
)


def _matches_df(rows: list[dict]) -> pd.DataFrame:
    # home_goals defaults to a played value (1) unless a row overrides it --
    # keeps every "all played" test below implicit while still letting the
    # unplayed-row test set it to None explicitly.
    rows = [{"home_goals": 1, **row} for row in rows]
    df = pd.DataFrame(rows)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    return df


def _config(guaranteed_slots: tuple[GuaranteedSlotConfig, ...] = ()) -> CompetitionConfig:
    """A minimal CompetitionConfig -- reference_dates only ever reads
    `config.guaranteed_slots`, so the single round-robin phase is just filler to
    satisfy the dataclass."""
    phase = RoundRobinPhaseConfig(
        id="league",
        head_to_head_mode="points_then_goal_diff",
        spots=(SpotConfig(name="title", positions=(1, 1)),),
    )
    return CompetitionConfig(
        name="Serie A",
        n_teams=20,
        phases=(phase,),
        guaranteed_slots=guaranteed_slots,
    )


def _config_with_relegation(positions: tuple[int, int] = (3, 4)) -> CompetitionConfig:
    """A CompetitionConfig with a real `rebaixamento` spot -- for
    _relegated_teams_previous_season, which reads it directly off the
    league phase (there's no configs/*_<season-1>.yaml to load a real
    historical config from, see that function's docstring)."""
    phase = RoundRobinPhaseConfig(
        id="league",
        head_to_head_mode="points_then_goal_diff",
        spots=(
            SpotConfig(name="title", positions=(1, 1)),
            SpotConfig(name="rebaixamento", positions=positions),
        ),
    )
    return CompetitionConfig(name="Serie B", n_teams=4, phases=(phase,))


def test_latest_checkpoint_date_returns_the_same_day_when_already_a_monday():
    assert latest_checkpoint_date(pd.Timestamp("2026-07-20")) == pd.Timestamp("2026-07-20")


def test_latest_checkpoint_date_returns_the_same_day_when_already_a_friday():
    assert latest_checkpoint_date(pd.Timestamp("2026-07-17")) == pd.Timestamp("2026-07-17")


def test_latest_checkpoint_date_walks_back_from_saturday_to_friday():
    assert latest_checkpoint_date(pd.Timestamp("2026-07-18")) == pd.Timestamp("2026-07-17")


def test_latest_checkpoint_date_walks_back_from_sunday_to_friday():
    assert latest_checkpoint_date(pd.Timestamp("2026-07-19")) == pd.Timestamp("2026-07-17")


def test_latest_checkpoint_date_walks_back_from_midweek_to_monday():
    # Tuesday, Wednesday, Thursday all fall back to the same preceding Monday.
    for day in ("2026-07-21", "2026-07-22", "2026-07-23"):
        assert latest_checkpoint_date(pd.Timestamp(day)) == pd.Timestamp("2026-07-20")


def test_latest_checkpoint_date_normalizes_away_the_time_of_day():
    assert latest_checkpoint_date(pd.Timestamp("2026-07-20 18:45:00")) == pd.Timestamp("2026-07-20")


def test_latest_checkpoint_date_defaults_to_a_checkpoint_on_or_before_now():
    result = latest_checkpoint_date()
    assert result.weekday() in (0, 4)
    assert result <= pd.Timestamp.now().normalize()


def test_reference_dates_lands_on_the_next_monday_or_friday_on_or_after_a_match():
    """A single played match produces two reference_dates: the unconditional
    pre-season checkpoint (see test_reference_dates_prepends_pre_season_checkpoint)
    plus the first Monday-or-Friday on/after the match's day, not an arbitrary
    round boundary. 2025-03-01 is a Saturday, so the checkpoint that captures
    it is the following Monday 2025-03-03."""
    df = _matches_df([{"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-01"}])

    dates = reference_dates(df, "Serie A", 2025, _config())

    # 2025-03-01 is Saturday; the pre-season checkpoint is Friday 2025-02-28,
    # and the first Monday/Friday on-or-after it is Monday 2025-03-03.
    assert dates == [pd.Timestamp("2025-02-28"), pd.Timestamp("2025-03-03")]
    assert dates[-1].day_name() == "Monday"


def test_reference_dates_collapses_two_matches_in_the_same_window_to_one_date():
    """Two matches falling between the same pair of Monday/Friday checkpoints
    collapse to a single reference_date -- no duplicate refit for the second."""
    df = _matches_df(
        [
            # Both land in the (Mon 2025-03-03, Fri 2025-03-07] window.
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-04"},
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-05"},
        ]
    )

    dates = reference_dates(df, "Serie A", 2025, _config())

    # Pre-season checkpoint (Monday 2025-03-03, the last Mon/Fri strictly
    # before the season's first match on Tuesday 2025-03-04) plus the single
    # checkpoint capturing both matches.
    assert dates == [pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-07")]
    assert dates[-1].day_name() == "Friday"


def test_reference_dates_includes_a_candidate_for_a_guaranteed_slot_crossing():
    """A guaranteed_slots berth whose known_from crosses between two candidate
    dates -- with no new match in between -- still produces a reference_date on
    its own, so the newly-known berth gets a refit."""
    slot = GuaranteedSlotConfig(
        team="Flamengo / RJ", spot="title", known_from=pd.Timestamp("2025-03-19")
    )
    df = _matches_df([{"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-01"}])

    dates = reference_dates(df, "Serie A", 2025, _config(guaranteed_slots=(slot,)))

    # Pre-season Friday 2025-02-28, then match -> Monday 2025-03-03.
    # known_from 2025-03-19 (Wed) -> next checkpoint Friday 2025-03-21, with
    # no match in between but the crossed berth.
    assert dates == [
        pd.Timestamp("2025-02-28"),
        pd.Timestamp("2025-03-03"),
        pd.Timestamp("2025-03-21"),
    ]


def test_reference_dates_skips_candidates_with_no_new_information():
    """Between two matches several weeks apart, every intervening Monday/Friday
    checkpoint with neither a new match nor a newly-crossed known_from is skipped
    entirely -- only the two checkpoints that capture the two matches survive
    (plus the unconditional pre-season one)."""
    df = _matches_df(
        [
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-01"},
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-04-05"},
        ]
    )

    dates = reference_dates(df, "Serie A", 2025, _config())

    # Pre-season checkpoint + the two capturing checkpoints; none of the
    # ~10 Mon/Fri in between.
    assert dates == [
        pd.Timestamp("2025-02-28"),
        pd.Timestamp("2025-03-03"),
        pd.Timestamp("2025-04-07"),
    ]


def test_reference_dates_filters_by_competition_and_season():
    df = _matches_df(
        [
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-01"},
            {"competition": "Serie B", "season": 2025, "match_datetime": "2025-03-15"},
            {"competition": "Serie A", "season": 2026, "match_datetime": "2026-03-01"},
        ]
    )

    assert reference_dates(df, "Serie A", 2025, _config()) == [
        pd.Timestamp("2025-02-28"),
        pd.Timestamp("2025-03-03"),
    ]


def test_reference_dates_returns_empty_list_when_no_played_matches():
    df = _matches_df([{"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-01"}])

    assert reference_dates(df, "Serie B", 2025, _config()) == []


def test_reference_dates_ignores_unplayed_rows():
    """matches.csv can now carry scheduled/postponed rows with no result yet
    (see src/ingestion/brazil/build_treated_dataset.py) -- a future fixture
    window must never spuriously produce a backtest checkpoint of its own."""
    df = _matches_df(
        [
            {"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-01"},
            {
                "competition": "Serie A",
                "season": 2025,
                "match_datetime": "2025-07-20",
                "home_goals": None,  # scheduled, not played yet
            },
        ]
    )

    assert reference_dates(df, "Serie A", 2025, _config()) == [
        pd.Timestamp("2025-02-28"),
        pd.Timestamp("2025-03-03"),
    ]


def test_reference_dates_prepends_pre_season_checkpoint_before_a_monday_first_match():
    """When the season's first match falls ON a Monday or Friday itself, the
    pre-season checkpoint must still be the previous Monday/Friday, not that
    same day -- otherwise the "pre-season" fit would train on round 1's own
    result, defeating its purpose (a real prediction for round 1 needs a
    snapshot strictly before it)."""
    # 2025-03-03 is a Monday.
    df = _matches_df([{"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-03"}])

    dates = reference_dates(df, "Serie A", 2025, _config())

    # Previous Friday (2025-02-28), then the match's own Monday checkpoint.
    assert dates == [pd.Timestamp("2025-02-28"), pd.Timestamp("2025-03-03")]


def test_reference_dates_prepends_pre_season_checkpoint_before_a_friday_first_match():
    # 2025-03-07 is a Friday.
    df = _matches_df([{"competition": "Serie A", "season": 2025, "match_datetime": "2025-03-07"}])

    dates = reference_dates(df, "Serie A", 2025, _config())

    # Previous Monday (2025-03-03), then the match's own Friday checkpoint.
    assert dates == [pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-07")]


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


# ---------------------------------------------------------------------------
# _write_log
# ---------------------------------------------------------------------------


def test_write_log_writes_every_row_with_the_expected_columns(tmp_path):
    log_path = tmp_path / "run_rounds_log.csv"
    rows = [
        {
            "reference_date": "2026-03-16",
            "competition": "Serie A",
            "season": 2026,
            "status": "skipped",
            "reason": "missing teams (haven't played yet): ['Team X']",
        },
        {
            "reference_date": "2026-04-20",
            "competition": "Serie B",
            "season": 2026,
            "status": "failed",
            "reason": "Stan fit failed: boom",
        },
    ]

    _write_log(rows, str(log_path))

    written = pd.read_csv(log_path)
    assert list(written.columns) == ["reference_date", "competition", "season", "status", "reason"]
    assert len(written) == 2
    assert written.iloc[0]["status"] == "skipped"
    assert written.iloc[1]["status"] == "failed"


def test_write_log_writes_a_header_only_csv_when_there_is_nothing_to_log(tmp_path):
    log_path = tmp_path / "run_rounds_log.csv"

    _write_log([], str(log_path))

    written = pd.read_csv(log_path)
    assert list(written.columns) == ["reference_date", "competition", "season", "status", "reason"]
    assert len(written) == 0


def test_write_log_creates_missing_parent_directories(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "run_rounds_log.csv"

    _write_log([], str(log_path))

    assert log_path.exists()


# ---------------------------------------------------------------------------
# _already_computed
# ---------------------------------------------------------------------------


def test_already_computed_true_when_the_results_file_exists(tmp_path):
    out_dir = tmp_path / "serie_a" / "2026"
    out_dir.mkdir(parents=True)
    (out_dir / "2026_05_04.csv").write_text("team,attack\n")

    assert _already_computed("Serie A", 2026, pd.Timestamp("2026-05-04"), str(tmp_path)) is True


def test_already_computed_false_when_the_results_file_is_missing(tmp_path):
    assert _already_computed("Serie A", 2026, pd.Timestamp("2026-05-04"), str(tmp_path)) is False


def test_already_computed_matches_save_results_own_slug_and_path_formula(tmp_path):
    """Same slug formula (lowercase, spaces -> underscores) and filename
    format (YYYY_MM_DD.csv) as src.simulation.results.save_results -- a
    mismatch here would make every resumed run recompute everything."""
    out_dir = tmp_path / "serie_b" / "2025"
    out_dir.mkdir(parents=True)
    (out_dir / "2025_04_18.csv").write_text("team,attack\n")

    assert _already_computed("Serie B", 2025, pd.Timestamp("2025-04-18"), str(tmp_path)) is True
    # A different date, or a name that doesn't slugify to "serie_b", must not match.
    assert _already_computed("Serie B", 2025, pd.Timestamp("2025-04-19"), str(tmp_path)) is False
    assert _already_computed("Serie C", 2025, pd.Timestamp("2025-04-18"), str(tmp_path)) is False


# ---------------------------------------------------------------------------
# _relegated_teams_previous_season
# ---------------------------------------------------------------------------


def test_relegated_teams_previous_season_returns_teams_in_the_relegation_range():
    """A, B, C, D finish 2025 in that exact order by points (A=9, B=6, C=3,
    D=0) -- with rebaixamento positions (3, 4), the relegated teams for a
    season=2026 lookup (previous season = 2025) are C then D, worst-first."""
    rows = [
        {
            "competition": "Serie B",
            "season": 2025,
            "match_datetime": "2025-01-01",
            "home_team": "A",
            "away_team": "B",
            "home_goals": 2,
            "away_goals": 0,
        },
        {
            "competition": "Serie B",
            "season": 2025,
            "match_datetime": "2025-01-02",
            "home_team": "A",
            "away_team": "C",
            "home_goals": 2,
            "away_goals": 0,
        },
        {
            "competition": "Serie B",
            "season": 2025,
            "match_datetime": "2025-01-03",
            "home_team": "A",
            "away_team": "D",
            "home_goals": 2,
            "away_goals": 0,
        },
        {
            "competition": "Serie B",
            "season": 2025,
            "match_datetime": "2025-01-04",
            "home_team": "B",
            "away_team": "C",
            "home_goals": 2,
            "away_goals": 0,
        },
        {
            "competition": "Serie B",
            "season": 2025,
            "match_datetime": "2025-01-05",
            "home_team": "B",
            "away_team": "D",
            "home_goals": 2,
            "away_goals": 0,
        },
        {
            "competition": "Serie B",
            "season": 2025,
            "match_datetime": "2025-01-06",
            "home_team": "C",
            "away_team": "D",
            "home_goals": 2,
            "away_goals": 0,
        },
    ]
    df = _matches_df(rows)
    config = _config_with_relegation(positions=(3, 4))

    relegated = _relegated_teams_previous_season(
        df, "Serie B", 2026, config, np.random.default_rng(0)
    )

    assert relegated == ["C", "D"]


def test_relegated_teams_previous_season_returns_empty_list_without_a_rebaixamento_spot():
    rows = [
        {
            "competition": "Serie A",
            "season": 2025,
            "match_datetime": "2025-01-01",
            "home_team": "A",
            "away_team": "B",
        }
    ]
    df = _matches_df(rows)

    relegated = _relegated_teams_previous_season(
        df, "Serie A", 2026, _config(), np.random.default_rng(0)
    )

    assert relegated == []


def test_relegated_teams_previous_season_returns_empty_list_when_previous_season_has_no_matches():
    df = _matches_df(
        [
            {
                "competition": "Serie B",
                "season": 2026,
                "match_datetime": "2026-01-01",
                "home_team": "A",
                "away_team": "B",
            }
        ]
    )
    config = _config_with_relegation()

    relegated = _relegated_teams_previous_season(
        df, "Serie B", 2026, config, np.random.default_rng(0)
    )

    assert relegated == []


# ---------------------------------------------------------------------------
# _debut_team_aliases
# ---------------------------------------------------------------------------


def test_debut_team_aliases_pairs_missing_and_relegated_alphabetically():
    missing = {"Zeta", "Alpha"}
    relegated = ["Delta", "Beta"]

    aliases = _debut_team_aliases(missing, relegated)

    assert aliases == {"Alpha": "Beta", "Zeta": "Delta"}


def test_debut_team_aliases_leaves_extra_missing_teams_unmapped():
    missing = {"Alpha", "Beta", "Gamma"}
    relegated = ["Zeta"]  # only one substitute available

    aliases = _debut_team_aliases(missing, relegated)

    assert aliases == {"Alpha": "Zeta"}
    assert "Beta" not in aliases
    assert "Gamma" not in aliases


def test_debut_team_aliases_returns_empty_dict_when_no_relegated_teams_available():
    assert _debut_team_aliases({"Alpha"}, []) == {}
