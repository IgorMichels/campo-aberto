"""Unit tests for espn_fixtures.py's ESPN scoreboard fetch/parse. requests.get
is mocked one layer down (per test_cbf_docket.py's convention: a fake
response with .status_code/.json()), time.sleep no-op'd -- these tests never
hit the network.
"""

import requests

from src.ingestion.brazil import espn_fixtures


def _event(
    status_name,
    home_name="Flamengo",
    away_name="Palmeiras",
    home_score="2",
    away_score="1",
    date="2026-07-16T22:30Z",
    venue="Maracana",
):
    return {
        "date": date,
        "status": {"type": {"name": status_name}},
        "competitions": [
            {
                "venue": {"fullName": venue},
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": home_score,
                        "team": {"displayName": home_name},
                    },
                    {
                        "homeAway": "away",
                        "score": away_score,
                        "team": {"displayName": away_name},
                    },
                ],
            }
        ],
    }


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def test_fetch_scoreboard_without_dates_sends_no_params(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return _Response({"ok": True})

    monkeypatch.setattr(espn_fixtures.requests, "get", fake_get)

    result = espn_fixtures.fetch_scoreboard("bra.1")

    assert result == {"ok": True}
    assert len(calls) == 1
    url, params = calls[0]
    assert url == "https://site.api.espn.com/apis/site/v2/sports/soccer/bra.1/scoreboard"
    assert params is None


def test_fetch_scoreboard_with_dates_always_pairs_limit_1000(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return _Response({"ok": True})

    monkeypatch.setattr(espn_fixtures.requests, "get", fake_get)

    espn_fixtures.fetch_scoreboard("bra.2", dates="20260101-20261231")

    assert calls[0][1] == {"dates": "20260101-20261231", "limit": 1000}


def test_get_with_retries_retries_transient_network_errors(monkeypatch):
    monkeypatch.setattr(espn_fixtures.time, "sleep", lambda seconds: None)
    calls = {"count": 0}

    def flaky_get(url, params=None, timeout=None):
        calls["count"] += 1
        if calls["count"] < espn_fixtures.RETRY_ATTEMPTS:
            raise requests.exceptions.ConnectionError("boom")
        return _Response({"ok": True})

    monkeypatch.setattr(espn_fixtures.requests, "get", flaky_get)

    result = espn_fixtures._get_with_retries("http://example.test")

    assert result == {"ok": True}
    assert calls["count"] == espn_fixtures.RETRY_ATTEMPTS


def test_get_with_retries_gives_up_after_every_attempt_fails(monkeypatch):
    monkeypatch.setattr(espn_fixtures.time, "sleep", lambda seconds: None)
    calls = {"count": 0}

    def always_fail(url, params=None, timeout=None):
        calls["count"] += 1
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(espn_fixtures.requests, "get", always_fail)

    assert espn_fixtures._get_with_retries("http://example.test") is None
    assert calls["count"] == espn_fixtures.RETRY_ATTEMPTS


def test_get_with_retries_returns_none_on_a_non_200_status(monkeypatch):
    monkeypatch.setattr(espn_fixtures.time, "sleep", lambda seconds: None)

    def not_found(url, params=None, timeout=None):
        return _Response({}, status_code=404)

    monkeypatch.setattr(espn_fixtures.requests, "get", not_found)

    assert espn_fixtures._get_with_retries("http://example.test") is None


def test_event_to_row_maps_played_status_and_scores():
    row = espn_fixtures._event_to_row(_event("STATUS_FULL_TIME"))

    assert row == {
        "date": "2026-07-16T22:30Z",
        "venue": "Maracana",
        "home_team_raw": "Flamengo",
        "away_team_raw": "Palmeiras",
        "home_goals": 2,
        "away_goals": 1,
        "status": "played",
    }


def test_event_to_row_maps_scheduled_status_with_no_goals():
    row = espn_fixtures._event_to_row(_event("STATUS_SCHEDULED"))

    assert row["status"] == "scheduled"
    assert row["home_goals"] is None
    assert row["away_goals"] is None


def test_event_to_row_maps_postponed_status_with_no_goals():
    row = espn_fixtures._event_to_row(_event("STATUS_POSTPONED"))

    assert row["status"] == "postponed"
    assert row["home_goals"] is None
    assert row["away_goals"] is None


def test_event_to_row_drops_any_other_status():
    assert espn_fixtures._event_to_row(_event("STATUS_IN_PROGRESS")) is None


def test_fetch_season_matches_sizes_the_window_from_the_calendar_and_maps_every_status(
    monkeypatch,
):
    probe_payload = {
        "leagues": [
            {
                "season": {"year": 2026},
                "calendar": ["2026-01-28T08:00Z", "2026-12-02T08:00Z"],
            }
        ]
    }
    ranged_payload = {
        "events": [
            _event("STATUS_FULL_TIME", home_name="Bahia", away_name="Sport"),
            _event("STATUS_SCHEDULED", home_name="Santos", away_name="Gremio"),
            _event("STATUS_POSTPONED", home_name="Ceara", away_name="Vitoria"),
            _event("STATUS_CANCELED", home_name="X", away_name="Y"),
        ]
    }
    calls = []

    def fake_scoreboard(league_code, dates=None):
        calls.append(dates)
        return probe_payload if dates is None else ranged_payload

    monkeypatch.setattr(espn_fixtures, "fetch_scoreboard", fake_scoreboard)

    rows = espn_fixtures.fetch_season_matches("bra.1")

    assert calls == [None, "20260128-20261202"]
    assert [row["status"] for row in rows] == ["played", "scheduled", "postponed"]
    assert rows[0]["home_team_raw"] == "Bahia"


def test_fetch_season_matches_returns_empty_list_when_the_probe_fails(monkeypatch):
    monkeypatch.setattr(espn_fixtures, "fetch_scoreboard", lambda league_code, dates=None: None)

    assert espn_fixtures.fetch_season_matches("bra.1") == []


def test_main_writes_one_csv_per_competition(tmp_path, monkeypatch):
    cache_dir = str(tmp_path / "espn")
    monkeypatch.setattr(espn_fixtures, "ESPN_CACHE_DIR", cache_dir)

    def fake_window(league_code):
        year = 2026
        rows = [espn_fixtures._event_to_row(_event("STATUS_FULL_TIME"))]
        return year, rows

    monkeypatch.setattr(espn_fixtures, "_fetch_season_window", fake_window)

    espn_fixtures.main()

    import os

    assert set(os.listdir(cache_dir)) == {"Serie_A_2026.csv", "Serie_B_2026.csv"}
