"""Unit + integration tests for build_treated_dataset.py. No network calls:
the raw-docket cache (CBF) and the raw ESPN cache are pre-seeded on disk, and
the team-name mapping (always manually curated -- see team_name_mapping.py)
is pre-seeded or omitted as each test needs.

NOTE: load_mapping defaults to `path=MAPPING_PATH`, a value bound into the
function's defaults at *def time* -- patching team_name_mapping's
MAPPING_PATH attribute afterward does NOT change what an already-imported
`load_mapping()` call (no explicit path) reads from. These tests always pass
`path=` explicitly (or patch build_treated_dataset.load_mapping with a bound
wrapper) instead, so nothing here can ever touch the real
data/processed/brazil/team_name_mapping.csv.

Every test that calls btd.main() also patches ESPN_CACHE_DIR (to an empty or
explicitly-seeded tmp_path dir) and DISCREPANCY_LOG_PATH, for the same
reason: nothing here should ever touch the real data/raw/brazil/espn/ or
data/processed/brazil/score_discrepancies.csv.
"""

import csv
import functools
import os
from unittest import mock

import pandas as pd

from src.ingestion.brazil import build_treated_dataset as btd
from src.ingestion.brazil import scrape_raw_matches as srm
from src.ingestion.brazil import team_name_mapping as tnm
from src.ingestion.brazil.espn_fixtures import FIELDNAMES as ESPN_FIELDNAMES


def _save_espn_games(cache_dir: str, competition_key: str, year: int, rows: list[dict]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{competition_key}_{year}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ESPN_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_parse_datetime_converts_cbf_format_to_iso_like():
    assert btd.parse_datetime("08/08/2020", "19:00") == "2020-08-08 19:00"


def test_result_pattern_extracts_goals_tolerating_spacing_and_case():
    assert btd.RESULT_PATTERN.match("2 X 1").groups() == ("2", "1")
    assert btd.RESULT_PATTERN.match("  10x 0 ").groups() == ("10", "0")


def test_result_pattern_does_not_match_garbage():
    assert btd.RESULT_PATTERN.match("abc") is None


def test_espn_datetime_to_brazil_local_applies_the_fixed_utc_minus_3_offset():
    """Brazil has used a fixed UTC-3 offset with no DST since 2019 -- a
    known case: 22:00 UTC is 19:00 in Sao Paulo, same calendar day."""
    assert btd.espn_datetime_to_brazil_local("2026-01-28T22:00Z") == "2026-01-28 19:00"


def test_espn_datetime_to_brazil_local_crosses_midnight():
    assert btd.espn_datetime_to_brazil_local("2026-07-17T02:00Z") == "2026-07-16 23:00"


def test_load_raw_games_reads_every_cached_season(tmp_path):
    game = {
        "Date": "08/08/2020",
        "Time": "19:00",
        "Stadium": "X",
        "Home": "A",
        "Away": "B",
        "Result": "1 X 0",
    }
    cache_dir = str(tmp_path)

    with (
        mock.patch.object(srm, "CBF_CACHE_DIR", cache_dir),
        mock.patch.object(btd, "CBF_CACHE_DIR", cache_dir),
    ):
        srm._save_games("Serie_A", 2020, {"001": game})
        srm._save_games("Serie_B", 2021, {"001": game})

        raw_games_by_season = btd.load_raw_games()

    assert set(raw_games_by_season) == {("Serie_A", 2020), ("Serie_B", 2021)}


def test_load_espn_games_reads_every_cached_season(tmp_path):
    row = {
        "date": "2026-07-16T22:30Z",
        "venue": "Maracana",
        "home_team_raw": "Flamengo",
        "away_team_raw": "Palmeiras",
        "home_goals": "",
        "away_goals": "",
        "status": "scheduled",
    }
    cache_dir = str(tmp_path)
    _save_espn_games(cache_dir, "Serie_A", 2026, [row])

    with mock.patch.object(btd, "ESPN_CACHE_DIR", cache_dir):
        espn_games_by_season = btd.load_espn_games()

    assert set(espn_games_by_season) == {("Serie_A", 2026)}
    loaded = espn_games_by_season[("Serie_A", 2026)][0]
    assert loaded["home_goals"] is None
    assert loaded["away_goals"] is None
    assert loaded["status"] == "scheduled"


def test_dedupe_espn_games_prefers_scheduled_row_over_stale_postponed_row():
    """The confirmed real-world case: a match gets postponed (ESPN keeps the
    stale original-date row with status="postponed") and later rescheduled
    (ESPN adds a second row, same team pair, status="scheduled", new date).
    Only the informative one should survive."""
    postponed_row = {
        "date": "2026-05-01T19:00Z",
        "venue": "Fonte Nova",
        "home_team_raw": "Bahia",
        "away_team_raw": "Chapecoense",
        "home_goals": None,
        "away_goals": None,
        "status": "postponed",
    }
    rescheduled_row = {
        "date": "2026-07-16T22:30Z",
        "venue": "Fonte Nova",
        "home_team_raw": "Bahia",
        "away_team_raw": "Chapecoense",
        "home_goals": None,
        "away_goals": None,
        "status": "scheduled",
    }

    deduped = btd._dedupe_espn_games([postponed_row, rescheduled_row])

    assert len(deduped) == 1
    assert deduped[0]["status"] == "scheduled"
    assert deduped[0]["date"] == "2026-07-16T22:30Z"


def test_dedupe_espn_games_order_does_not_matter():
    """Same pair as above, but postponed row seen second -- still only the
    scheduled row survives."""
    postponed_row = {
        "date": "2026-05-01T19:00Z",
        "venue": "Fonte Nova",
        "home_team_raw": "Bahia",
        "away_team_raw": "Chapecoense",
        "home_goals": None,
        "away_goals": None,
        "status": "postponed",
    }
    rescheduled_row = {
        "date": "2026-07-16T22:30Z",
        "venue": "Fonte Nova",
        "home_team_raw": "Bahia",
        "away_team_raw": "Chapecoense",
        "home_goals": None,
        "away_goals": None,
        "status": "scheduled",
    }

    deduped = btd._dedupe_espn_games([rescheduled_row, postponed_row])

    assert len(deduped) == 1
    assert deduped[0]["status"] == "scheduled"


def test_dedupe_espn_games_keeps_distinct_pairs():
    row_a = {
        "date": "2026-07-16T22:30Z",
        "venue": "X",
        "home_team_raw": "Bahia",
        "away_team_raw": "Chapecoense",
        "home_goals": None,
        "away_goals": None,
        "status": "scheduled",
    }
    row_b = {
        "date": "2026-07-17T22:30Z",
        "venue": "Y",
        "home_team_raw": "Botafogo",
        "away_team_raw": "Vitoria",
        "home_goals": None,
        "away_goals": None,
        "status": "scheduled",
    }

    deduped = btd._dedupe_espn_games([row_a, row_b])

    assert len(deduped) == 2


def test_main_writes_treated_matches_and_logs_unmapped_names(tmp_path):
    game1 = {
        "Date": "08/08/2020",
        "Time": "19:00",
        "Stadium": "Vila Belmiro",
        "Home": "Santos Fc / SP",  # resolvable via the pre-seeded mapping
        "Away": "Flamengo / RJ",  # never in the mapping -> stays unmapped
        "Result": "2 X 1",
    }
    cache_dir = str(tmp_path / "raw")
    espn_cache_dir = str(tmp_path / "espn")  # left empty -- no ESPN data this run
    output_path = str(tmp_path / "matches.csv")
    unmapped_path = str(tmp_path / "unmapped.csv")
    discrepancy_path = str(tmp_path / "discrepancies.csv")
    mapping_path = str(tmp_path / "mapping.csv")

    tnm.save_mapping({"Santos Fc / SP": "Santos / SP"}, path=mapping_path)

    with (
        mock.patch.object(srm, "CBF_CACHE_DIR", cache_dir),
        mock.patch.object(btd, "CBF_CACHE_DIR", cache_dir),
        mock.patch.object(btd, "ESPN_CACHE_DIR", espn_cache_dir),
        mock.patch.object(btd, "OUTPUT_PATH", output_path),
        mock.patch.object(btd, "UNMAPPED_LOG_PATH", unmapped_path),
        mock.patch.object(btd, "DISCREPANCY_LOG_PATH", discrepancy_path),
        mock.patch.object(
            btd, "load_mapping", functools.partial(tnm.load_mapping, path=mapping_path)
        ),
    ):
        srm._save_games("Serie_A", 2020, {"001": game1})

        btd.main()

    matches = pd.read_csv(output_path)
    assert len(matches) == 1
    row = matches.iloc[0]
    assert row["competition"] == "Serie A"
    assert row["season"] == 2020
    assert row["match_datetime"] == "2020-08-08 19:00"
    assert row["home_team"] == "Santos / SP"  # resolved via the mapping
    assert row["away_team"] == "Flamengo / RJ"  # left as-is, logged instead
    assert row["home_goals"] == 2
    assert row["away_goals"] == 1
    assert row["status"] == "played"

    unmapped = pd.read_csv(unmapped_path)
    assert (unmapped["raw_name"] == "Flamengo / RJ").any()

    discrepancies = pd.read_csv(discrepancy_path)
    assert len(discrepancies) == 0


def test_main_never_writes_to_the_mapping_file(tmp_path):
    """team_name_mapping.csv is entirely manually curated -- the pipeline only
    ever reads it, even when it doesn't exist yet (every name is just left
    unmapped and logged for a human to resolve)."""
    game = {
        "Date": "08/08/2020",
        "Time": "19:00",
        "Stadium": "X",
        "Home": "A FC / SP",
        "Away": "B FC / RJ",
        "Result": "1 X 0",
    }
    cache_dir = str(tmp_path / "raw")
    espn_cache_dir = str(tmp_path / "espn")
    output_path = str(tmp_path / "matches.csv")
    unmapped_path = str(tmp_path / "unmapped.csv")
    discrepancy_path = str(tmp_path / "discrepancies.csv")
    mapping_path = str(tmp_path / "mapping.csv")  # deliberately never created

    with (
        mock.patch.object(srm, "CBF_CACHE_DIR", cache_dir),
        mock.patch.object(btd, "CBF_CACHE_DIR", cache_dir),
        mock.patch.object(btd, "ESPN_CACHE_DIR", espn_cache_dir),
        mock.patch.object(btd, "OUTPUT_PATH", output_path),
        mock.patch.object(btd, "UNMAPPED_LOG_PATH", unmapped_path),
        mock.patch.object(btd, "DISCREPANCY_LOG_PATH", discrepancy_path),
        mock.patch.object(
            btd, "load_mapping", functools.partial(tnm.load_mapping, path=mapping_path)
        ),
    ):
        srm._save_games("Serie_A", 2020, {"001": game})

        btd.main()

    assert not os.path.exists(mapping_path)  # never created, let alone written to
    matches = pd.read_csv(output_path)
    assert matches.iloc[0]["home_team"] == "A FC / SP"  # left unresolved, as-is


def _run_main_with_espn(tmp_path, cbf_games, espn_rows, mapping):
    """Shared harness for the CBF+ESPN merge tests below: seeds one CBF game
    per (competition_key, year, game_code, game) in cbf_games and one ESPN
    raw row per (competition_key, year, row) in espn_rows, runs btd.main(),
    and returns (matches_df, unmapped_df, discrepancies_df)."""
    cache_dir = str(tmp_path / "raw")
    espn_cache_dir = str(tmp_path / "espn")
    output_path = str(tmp_path / "matches.csv")
    unmapped_path = str(tmp_path / "unmapped.csv")
    discrepancy_path = str(tmp_path / "discrepancies.csv")
    mapping_path = str(tmp_path / "mapping.csv")

    tnm.save_mapping(mapping, path=mapping_path)

    with (
        mock.patch.object(srm, "CBF_CACHE_DIR", cache_dir),
        mock.patch.object(btd, "CBF_CACHE_DIR", cache_dir),
        mock.patch.object(btd, "ESPN_CACHE_DIR", espn_cache_dir),
        mock.patch.object(btd, "OUTPUT_PATH", output_path),
        mock.patch.object(btd, "UNMAPPED_LOG_PATH", unmapped_path),
        mock.patch.object(btd, "DISCREPANCY_LOG_PATH", discrepancy_path),
        mock.patch.object(
            btd, "load_mapping", functools.partial(tnm.load_mapping, path=mapping_path)
        ),
    ):
        for competition_key, year, game_code, game in cbf_games:
            srm._save_games(competition_key, year, {game_code: game})
        for competition_key, year, row in espn_rows:
            _save_espn_games(espn_cache_dir, competition_key, year, [row])

        btd.main()

    return (
        pd.read_csv(output_path),
        pd.read_csv(unmapped_path),
        pd.read_csv(discrepancy_path),
    )


_MAPPING = {"Flamengo": "Flamengo / RJ", "Palmeiras": "Palmeiras / SP"}


def test_espn_row_matching_a_cbf_game_with_the_same_score_produces_no_duplicate_or_discrepancy(
    tmp_path,
):
    cbf_game = {
        "Date": "16/07/2026",
        "Time": "19:30",
        "Stadium": "Maracana",
        "Home": "Flamengo",
        "Away": "Palmeiras",
        "Result": "2 X 1",
    }
    espn_row = {
        "date": "2026-07-16T22:30Z",
        "venue": "Maracana",
        "home_team_raw": "Flamengo",
        "away_team_raw": "Palmeiras",
        "home_goals": "2",
        "away_goals": "1",
        "status": "played",
    }

    matches, _, discrepancies = _run_main_with_espn(
        tmp_path,
        [("Serie_A", 2026, "001", cbf_game)],
        [("Serie_A", 2026, espn_row)],
        _MAPPING,
    )

    assert len(matches) == 1  # no duplicate ESPN-only row
    assert len(discrepancies) == 0


def test_espn_row_matching_a_cbf_game_with_a_different_score_produces_a_discrepancy_row(
    tmp_path,
):
    cbf_game = {
        "Date": "16/07/2026",
        "Time": "19:30",
        "Stadium": "Maracana",
        "Home": "Flamengo",
        "Away": "Palmeiras",
        "Result": "2 X 1",
    }
    espn_row = {
        "date": "2026-07-16T22:30Z",
        "venue": "Maracana",
        "home_team_raw": "Flamengo",
        "away_team_raw": "Palmeiras",
        "home_goals": "3",  # disagrees with CBF's 2
        "away_goals": "1",
        "status": "played",
    }

    matches, _, discrepancies = _run_main_with_espn(
        tmp_path,
        [("Serie_A", 2026, "001", cbf_game)],
        [("Serie_A", 2026, espn_row)],
        _MAPPING,
    )

    assert len(matches) == 1
    assert matches.iloc[0]["home_goals"] == 2  # CBF's score still wins
    assert matches.iloc[0]["away_goals"] == 1

    assert len(discrepancies) == 1
    d = discrepancies.iloc[0]
    assert d["home_team"] == "Flamengo / RJ"
    assert d["away_team"] == "Palmeiras / SP"
    assert d["cbf_home_goals"] == 2
    assert d["cbf_away_goals"] == 1
    assert d["espn_home_goals"] == 3
    assert d["espn_away_goals"] == 1


def test_espn_only_row_appears_in_matches_with_empty_goals_and_the_right_status(tmp_path):
    espn_row = {
        "date": "2026-07-16T22:30Z",
        "venue": "Maracana",
        "home_team_raw": "Flamengo",
        "away_team_raw": "Palmeiras",
        "home_goals": "",
        "away_goals": "",
        "status": "scheduled",
    }

    matches, _, discrepancies = _run_main_with_espn(
        tmp_path, [], [("Serie_A", 2026, espn_row)], _MAPPING
    )

    assert len(matches) == 1
    row = matches.iloc[0]
    assert row["home_team"] == "Flamengo / RJ"
    assert row["away_team"] == "Palmeiras / SP"
    assert pd.isna(row["home_goals"])
    assert pd.isna(row["away_goals"])
    assert row["status"] == "scheduled"
    assert row["match_datetime"] == "2026-07-16 19:30"  # UTC -> Brazil local
    assert len(discrepancies) == 0


def test_espn_only_postponed_row_keeps_postponed_status(tmp_path):
    espn_row = {
        "date": "2026-07-16T22:30Z",
        "venue": "Maracana",
        "home_team_raw": "Flamengo",
        "away_team_raw": "Palmeiras",
        "home_goals": "",
        "away_goals": "",
        "status": "postponed",
    }

    matches, _, _ = _run_main_with_espn(tmp_path, [], [("Serie_A", 2026, espn_row)], _MAPPING)

    assert matches.iloc[0]["status"] == "postponed"


def test_espn_only_row_reported_played_with_no_cbf_match_still_gets_scheduled_status(tmp_path):
    """CBF's score is the only one this dataset ever trusts as final -- an
    ESPN row that reports a final score but has no matching CBF row yet
    (CBF's docket scraping hasn't caught up) is still written with
    home_goals/away_goals empty and status="scheduled" (not "played", and not
    a fourth status for this rare edge case)."""
    espn_row = {
        "date": "2026-07-16T22:30Z",
        "venue": "Maracana",
        "home_team_raw": "Flamengo",
        "away_team_raw": "Palmeiras",
        "home_goals": "2",  # ESPN says this already happened...
        "away_goals": "1",
        "status": "played",
    }

    matches, _, _ = _run_main_with_espn(tmp_path, [], [("Serie_A", 2026, espn_row)], _MAPPING)

    row = matches.iloc[0]
    assert row["status"] == "scheduled"  # ...but CBF hasn't confirmed it yet
    assert pd.isna(row["home_goals"])
    assert pd.isna(row["away_goals"])


def test_main_dedupes_espn_postponed_then_rescheduled_rows_into_a_single_match(tmp_path):
    """Integration-level version of the _dedupe_espn_games unit tests above:
    two ESPN rows for the same pair (a stale "postponed" row plus the
    rescheduled "scheduled" row, no CBF match yet to absorb either) must
    still produce exactly one row in matches.csv -- the real bug that
    inflated Serie A 2026's matches.csv to 382 rows instead of 380."""
    postponed_row = {
        "date": "2026-05-01T19:00Z",
        "venue": "Fonte Nova",
        "home_team_raw": "Bahia",
        "away_team_raw": "Chapecoense",
        "home_goals": "",
        "away_goals": "",
        "status": "postponed",
    }
    rescheduled_row = {
        "date": "2026-07-16T22:30Z",
        "venue": "Fonte Nova",
        "home_team_raw": "Bahia",
        "away_team_raw": "Chapecoense",
        "home_goals": "",
        "away_goals": "",
        "status": "scheduled",
    }
    mapping = {"Bahia": "Bahia / BA", "Chapecoense": "Chapecoense / SC"}

    cache_dir = str(tmp_path / "raw")
    espn_cache_dir = str(tmp_path / "espn")
    output_path = str(tmp_path / "matches.csv")
    unmapped_path = str(tmp_path / "unmapped.csv")
    discrepancy_path = str(tmp_path / "discrepancies.csv")
    mapping_path = str(tmp_path / "mapping.csv")

    tnm.save_mapping(mapping, path=mapping_path)
    _save_espn_games(espn_cache_dir, "Serie_A", 2026, [postponed_row, rescheduled_row])

    with (
        mock.patch.object(srm, "CBF_CACHE_DIR", cache_dir),
        mock.patch.object(btd, "CBF_CACHE_DIR", cache_dir),
        mock.patch.object(btd, "ESPN_CACHE_DIR", espn_cache_dir),
        mock.patch.object(btd, "OUTPUT_PATH", output_path),
        mock.patch.object(btd, "UNMAPPED_LOG_PATH", unmapped_path),
        mock.patch.object(btd, "DISCREPANCY_LOG_PATH", discrepancy_path),
        mock.patch.object(
            btd, "load_mapping", functools.partial(tnm.load_mapping, path=mapping_path)
        ),
    ):
        btd.main()

    matches = pd.read_csv(output_path)
    assert len(matches) == 1
    row = matches.iloc[0]
    assert row["status"] == "scheduled"
    assert row["match_datetime"] == "2026-07-16 19:30"  # UTC -> Brazil local


def test_unmapped_espn_name_lands_in_the_same_unmapped_log_as_an_unmapped_cbf_name(tmp_path):
    cbf_game = {
        "Date": "08/08/2020",
        "Time": "19:00",
        "Stadium": "X",
        "Home": "Flamengo",  # resolvable
        "Away": "Some Unmapped CBF Club / XX",  # unmapped
        "Result": "1 X 0",
    }
    espn_row = {
        "date": "2026-07-16T22:30Z",
        "venue": "Maracana",
        "home_team_raw": "Flamengo",  # resolvable
        "away_team_raw": "Some Unmapped ESPN Club",  # unmapped
        "home_goals": "",
        "away_goals": "",
        "status": "scheduled",
    }

    _, unmapped, _ = _run_main_with_espn(
        tmp_path,
        [("Serie_A", 2020, "001", cbf_game)],
        [("Serie_A", 2026, espn_row)],
        _MAPPING,
    )

    assert (unmapped["raw_name"] == "Some Unmapped CBF Club / XX").any()
    assert (unmapped["raw_name"] == "Some Unmapped ESPN Club").any()
