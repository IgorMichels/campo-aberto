"""Unit + integration tests for build_treated_dataset.py. No network calls:
the raw-docket cache is pre-seeded on disk, and the team-name mapping (always
manually curated -- see team_name_mapping.py) is pre-seeded or omitted as each
test needs.

NOTE: load_mapping defaults to `path=MAPPING_PATH`, a value bound into the
function's defaults at *def time* -- patching team_name_mapping's
MAPPING_PATH attribute afterward does NOT change what an already-imported
`load_mapping()` call (no explicit path) reads from. These tests always pass
`path=` explicitly (or patch build_treated_dataset.load_mapping with a bound
wrapper) instead, so nothing here can ever touch the real
data/processed/brazil/team_name_mapping.csv.
"""

import functools
import os
from unittest import mock

import pandas as pd

from src.ingestion.brazil import build_treated_dataset as btd
from src.ingestion.brazil import scrape_raw_matches as srm
from src.ingestion.brazil import team_name_mapping as tnm


def test_parse_datetime_converts_cbf_format_to_iso_like():
    assert btd.parse_datetime("08/08/2020", "19:00") == "2020-08-08 19:00"


def test_result_pattern_extracts_goals_tolerating_spacing_and_case():
    assert btd.RESULT_PATTERN.match("2 X 1").groups() == ("2", "1")
    assert btd.RESULT_PATTERN.match("  10x 0 ").groups() == ("10", "0")


def test_result_pattern_does_not_match_garbage():
    assert btd.RESULT_PATTERN.match("abc") is None


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
        mock.patch.object(srm, "CACHE_DIR", cache_dir),
        mock.patch.object(btd, "CACHE_DIR", cache_dir),
    ):
        srm._save_games("Serie_A", 2020, {"001": game})
        srm._save_games("Serie_B", 2021, {"001": game})

        raw_games_by_season = btd.load_raw_games()

    assert set(raw_games_by_season) == {("Serie_A", 2020), ("Serie_B", 2021)}


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
    output_path = str(tmp_path / "matches.csv")
    unmapped_path = str(tmp_path / "unmapped.csv")
    mapping_path = str(tmp_path / "mapping.csv")

    tnm.save_mapping({"Santos Fc / SP": "Santos / SP"}, path=mapping_path)

    with (
        mock.patch.object(srm, "CACHE_DIR", cache_dir),
        mock.patch.object(btd, "CACHE_DIR", cache_dir),
        mock.patch.object(btd, "OUTPUT_PATH", output_path),
        mock.patch.object(btd, "UNMAPPED_LOG_PATH", unmapped_path),
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

    unmapped = pd.read_csv(unmapped_path)
    assert (unmapped["raw_name"] == "Flamengo / RJ").any()


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
    output_path = str(tmp_path / "matches.csv")
    unmapped_path = str(tmp_path / "unmapped.csv")
    mapping_path = str(tmp_path / "mapping.csv")  # deliberately never created

    with (
        mock.patch.object(srm, "CACHE_DIR", cache_dir),
        mock.patch.object(btd, "CACHE_DIR", cache_dir),
        mock.patch.object(btd, "OUTPUT_PATH", output_path),
        mock.patch.object(btd, "UNMAPPED_LOG_PATH", unmapped_path),
        mock.patch.object(
            btd, "load_mapping", functools.partial(tnm.load_mapping, path=mapping_path)
        ),
    ):
        srm._save_games("Serie_A", 2020, {"001": game})

        btd.main()

    assert not os.path.exists(mapping_path)  # never created, let alone written to
    matches = pd.read_csv(output_path)
    assert matches.iloc[0]["home_team"] == "A FC / SP"  # left unresolved, as-is
