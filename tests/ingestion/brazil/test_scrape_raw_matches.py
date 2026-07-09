"""Unit tests for scrape_raw_matches.py's local CSV cache and incremental
resume logic. season_scraper.scrape_season is mocked -- these tests never hit
CBF's servers.
"""

from unittest import mock

from src.ingestion.brazil import scrape_raw_matches as srm

_GAME = {
    "Date": "08/08/2020",
    "Time": "19:00",
    "Stadium": "X",
    "Home": "A",
    "Away": "B",
    "Result": "1 X 0",
}


def test_save_and_load_season_csv_round_trip(tmp_path):
    games = {"001": _GAME}

    with mock.patch.object(srm, "CBF_CACHE_DIR", str(tmp_path)):
        srm._save_games("Serie_A", 2020, games)
        loaded = srm.load_season_csv(srm._cache_path("Serie_A", 2020))

    assert loaded == games


def test_cache_path_is_keyed_by_competition_and_year():
    with mock.patch.object(srm, "CBF_CACHE_DIR", "data/raw/brazil/cbf"):
        assert srm._cache_path("Serie_A", 2020) == "data/raw/brazil/cbf/Serie_A_2020.csv"


def test_a_complete_season_is_never_rescraped(tmp_path):
    """len(games) >= GAMES_PER_SEASON short-circuits before scrape_season is
    even called."""
    with (
        mock.patch.object(srm, "CBF_CACHE_DIR", str(tmp_path)),
        mock.patch.object(srm, "GAMES_PER_SEASON", 2),
    ):
        games = {"001": _GAME, "002": _GAME}
        srm._save_games("Serie_A", 2020, games)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("scrape_season should not run for an already-complete season")

        with mock.patch.object(srm, "scrape_season", fail_if_called):
            result = srm.scrape_season_games("Serie_A", 2020)

    assert sorted(result) == ["001", "002"]


def test_an_incomplete_season_resumes_from_its_highest_cached_game_id(tmp_path):
    with (
        mock.patch.object(srm, "CBF_CACHE_DIR", str(tmp_path)),
        mock.patch.object(srm, "GAMES_PER_SEASON", 10),
    ):
        srm._save_games("Serie_A", 2020, {"001": _GAME, "005": _GAME})

        captured = {}

        def fake_scrape_season(competition_key, year, games, resume_from):
            captured["resume_from"] = resume_from
            return games

        with mock.patch.object(srm, "scrape_season", fake_scrape_season):
            srm.scrape_season_games("Serie_A", 2020)

    assert captured["resume_from"] == 5


def test_resume_from_is_capped_at_games_per_season(tmp_path):
    """A stray game_id beyond GAMES_PER_SEASON (shouldn't normally happen)
    doesn't push resume_from past the season's real size."""
    with (
        mock.patch.object(srm, "CBF_CACHE_DIR", str(tmp_path)),
        mock.patch.object(srm, "GAMES_PER_SEASON", 5),
    ):
        srm._save_games("Serie_A", 2020, {"001": _GAME, "999": _GAME})

        captured = {}

        def fake_scrape_season(competition_key, year, games, resume_from):
            captured["resume_from"] = resume_from
            return games

        with mock.patch.object(srm, "scrape_season", fake_scrape_season):
            srm.scrape_season_games("Serie_A", 2020)

    assert captured["resume_from"] == 5


def test_a_never_scraped_season_resumes_from_zero(tmp_path):
    with (
        mock.patch.object(srm, "CBF_CACHE_DIR", str(tmp_path)),
        mock.patch.object(srm, "GAMES_PER_SEASON", 10),
    ):
        captured = {}

        def fake_scrape_season(competition_key, year, games, resume_from):
            captured["resume_from"] = resume_from
            return games

        with mock.patch.object(srm, "scrape_season", fake_scrape_season):
            srm.scrape_season_games("Serie_A", 2020)

    assert captured["resume_from"] == 0
