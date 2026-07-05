"""Unit tests for season_scraper.py's resumable season scrape. try_fetch_docket
(from cbf_docket.py) is mocked -- these tests never hit CBF's servers.
"""

from unittest import mock

from src.ingestion.brazil import season_scraper


def _run(games, resume_from, responses, miss_limit=2):
    """responses: {game_code: docket_or_None}. Any game_code not listed misses."""
    calls = []

    def fake_try_fetch(competition_key, year, game_code):
        calls.append(game_code)
        return responses.get(game_code)

    with (
        mock.patch.object(season_scraper, "try_fetch_docket", fake_try_fetch),
        mock.patch.object(season_scraper, "CONSECUTIVE_MISS_LIMIT", miss_limit),
    ):
        result = season_scraper.scrape_season("Serie_A", 2020, dict(games), resume_from)
    return result, calls


def test_gap_codes_below_resume_from_are_retried_first():
    """Codes between 1 and resume_from not already in games are gaps --
    postponed matches or games that hadn't been played yet last time."""
    games = {f"{i:03d}": {"Home": f"H{i}", "Away": f"A{i}"} for i in (1, 2, 4, 5)}

    result, calls = _run(games, resume_from=5, responses={3: {"Home": "H3", "Away": "A3"}, 6: None})

    assert calls[0] == 3  # the only gap in 1..5
    assert "003" in result


def test_a_game_already_present_is_never_refetched():
    games = {"001": {"Home": "H1", "Away": "A1"}}

    _result, calls = _run(games, resume_from=1, responses={})

    # no gap (1 is already present) and forward probing starts past resume_from
    assert calls == [2, 3]


def test_forward_probing_stops_after_consecutive_miss_limit():
    games = {}

    _result, calls = _run(games, resume_from=0, responses={}, miss_limit=3)

    # no gaps (resume_from=0), then 3 consecutive misses (games 1, 2, 3) stop probing
    assert calls == [1, 2, 3]


def test_forward_probing_resets_the_miss_counter_on_a_hit():
    games = {}
    responses = {1: None, 2: {"Home": "H2", "Away": "A2"}, 3: None, 4: None}

    result, calls = _run(games, resume_from=0, responses=responses, miss_limit=2)

    # 1 misses (counter at 1/2), 2 hits and resets it, then 3 and 4 miss to
    # reach the limit -- if the reset hadn't happened, probing would have
    # stopped one game earlier (miss 1 + miss 3 = 2).
    assert calls == [1, 2, 3, 4]
    assert "002" in result


def test_forward_probing_never_goes_past_games_per_season(monkeypatch):
    monkeypatch.setattr(season_scraper, "GAMES_PER_SEASON", 3)
    games = {}

    _result, calls = _run(games, resume_from=0, responses={1: {}, 2: {}, 3: {}}, miss_limit=100)

    assert calls == [1, 2, 3]  # stops at GAMES_PER_SEASON even though nothing missed
