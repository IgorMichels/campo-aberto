"""Fetches Brazilian Serie A / Serie B schedule + score data from ESPN's public
scoreboard API.

Only fetches and parses the raw scoreboard events, exactly as ESPN reports
them -- no team-name treatment happens here (see team_name_mapping.py and
build_treated_dataset.py for that, same division of labor as the existing CBF
raw layer in scrape_raw_matches.py / cbf_docket.py).

This is a deliberate, user-approved exception to this repo's established
"CBF is the only external data source" convention, scoped *only* to fixture
scheduling: CBF's docket score still wins whenever available (see
build_treated_dataset.py's merge logic and data/processed/brazil/README.md).

Endpoint, verified live (no auth required):
    GET https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/scoreboard
    GET .../scoreboard?dates=YYYYMMDD-YYYYMMDD&limit=1000

The no-param call returns only a couple of "today" events but includes
leagues[0].season.year and leagues[0].calendar (every ISO date the season
touches), used only to size the ranged call's date window. The ranged call
silently caps at 100 events without limit=1000; with it, it returns the
league's entire season, played and unplayed alike.

Runs are incremental, mirroring scrape_raw_matches.py's own rule: a cached
season with a full GAMES_PER_SEASON rows, all of them already "played", is
never re-fetched (its cheap no-param probe still runs every time, just to
learn the current season's year -- see _probe_season_year). Unlike CBF's
per-game_id resume, ESPN's ranged call is all-or-nothing, so an in-progress
season is refetched in full every run rather than resumed incrementally.
"""

import csv
import os
import time
from datetime import datetime
from typing import Optional

import requests

from src.ingestion.brazil.constants import (
    COMPETITIONS,
    ESPN_CACHE_DIR,
    ESPN_LEAGUE_CODES,
    ESPN_SCOREBOARD_URL,
    GAMES_PER_SEASON,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF_SECONDS,
)

FIELDNAMES = [
    "date",
    "venue",
    "home_team_raw",
    "away_team_raw",
    "home_goals",
    "away_goals",
    "status",
]

# ESPN's status.type.name -> this dataset's status vocabulary. Any other
# status (e.g. an in-progress match) is dropped entirely -- see
# fetch_season_matches.
_STATUS_MAP = {
    "STATUS_FULL_TIME": "played",
    "STATUS_SCHEDULED": "scheduled",
    "STATUS_POSTPONED": "postponed",
}


def _get_with_retries(url: str, params: Optional[dict] = None) -> Optional[dict]:
    """GETs a URL and parses its JSON body, retrying on transient network
    errors -- same linear-backoff pattern as cbf_docket._get_with_retries,
    adapted to return parsed JSON instead of the raw response.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code != 200:
                if attempt == RETRY_ATTEMPTS - 1:
                    return None
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            return response.json()
        except requests.exceptions.RequestException:
            if attempt == RETRY_ATTEMPTS - 1:
                return None
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    return None


def fetch_scoreboard(league_code: str, dates: Optional[str] = None) -> Optional[dict]:
    """One GET against ESPN's scoreboard endpoint for a league code (e.g. "bra.1").

    `dates`, when given, is a "YYYYMMDD-YYYYMMDD" range and is always paired
    with limit=1000 -- ESPN silently caps a ranged query at 100 events
    otherwise (confirmed live).
    """
    url = ESPN_SCOREBOARD_URL.format(league_code=league_code)
    params = {"dates": dates, "limit": 1000} if dates is not None else None
    return _get_with_retries(url, params=params)


def _event_to_row(event: dict) -> Optional[dict]:
    status_name = event["status"]["type"]["name"]
    status = _STATUS_MAP.get(status_name)
    if status is None:
        return None  # e.g. an in-progress match -- not a stable status to record

    competition = event["competitions"][0]
    competitors = {c["homeAway"]: c for c in competition["competitors"]}
    home, away = competitors["home"], competitors["away"]

    home_goals = int(home["score"]) if status == "played" else None
    away_goals = int(away["score"]) if status == "played" else None

    return {
        "date": event["date"],
        "venue": competition.get("venue", {}).get("fullName", ""),
        "home_team_raw": home["team"]["displayName"],
        "away_team_raw": away["team"]["displayName"],
        "home_goals": home_goals,
        "away_goals": away_goals,
        "status": status,
    }


def _fetch_season_window(league_code: str) -> tuple[Optional[int], list[dict]]:
    """Probes for the season's year + full calendar span, then makes one
    ranged call covering that whole window. Split out from
    fetch_season_matches so main() can grab the season year without a second,
    redundant probe call.
    """
    probe = fetch_scoreboard(league_code)
    if probe is None or not probe.get("leagues"):
        return None, []

    league = probe["leagues"][0]
    year = league["season"]["year"]
    calendar = league["calendar"]
    if not calendar:
        return year, []

    start = datetime.strptime(calendar[0], "%Y-%m-%dT%H:%MZ").strftime("%Y%m%d")
    end = datetime.strptime(calendar[-1], "%Y-%m-%dT%H:%MZ").strftime("%Y%m%d")

    payload = fetch_scoreboard(league_code, dates=f"{start}-{end}")
    if payload is None:
        return year, []

    rows = [row for event in payload.get("events", []) if (row := _event_to_row(event)) is not None]
    return year, rows


def fetch_season_matches(league_code: str) -> list[dict]:
    """Fetches the whole season's schedule for a league code (e.g. "bra.1"):
    played and unplayed matches alike -- ESPN's value-add here is the
    schedule, not just upcoming fixtures.

    Returns raw rows: {"date": iso_utc_str, "venue": str, "home_team_raw": str,
    "away_team_raw": str, "home_goals": int | None, "away_goals": int | None,
    "status": "played" | "scheduled" | "postponed"}.
    """
    _, rows = _fetch_season_window(league_code)
    return rows


def _cache_path(competition_key: str, year: int) -> str:
    return os.path.join(ESPN_CACHE_DIR, f"{competition_key}_{year}.csv")


def _save_rows(competition_key: str, year: int, rows: list[dict]) -> None:
    os.makedirs(ESPN_CACHE_DIR, exist_ok=True)
    with open(_cache_path(competition_key, year), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _is_season_complete(cache_path: str) -> bool:
    """True once a season's cache has a full GAMES_PER_SEASON rows and every
    one of them is already "played" -- mirrors scrape_raw_matches.py's "a
    finished season is never re-scraped" rule. A played match's score is
    final; only a scheduled/postponed row can ever change on a later run, so
    once none are left there's nothing ESPN's (expensive, ~380-event) ranged
    call could still teach us.
    """
    if not os.path.exists(cache_path):
        return False
    with open(cache_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return len(rows) >= GAMES_PER_SEASON and all(row["status"] == "played" for row in rows)


def _probe_season_year(league_code: str) -> Optional[int]:
    """The cheap, no-params probe call's season year alone -- used to locate
    the cache file *before* deciding whether the full ranged fetch (see
    _fetch_season_window) is even worth making."""
    probe = fetch_scoreboard(league_code)
    if probe is None or not probe.get("leagues"):
        return None
    return probe["leagues"][0]["season"]["year"]


def main() -> None:
    for competition_key in COMPETITIONS:
        league_code = ESPN_LEAGUE_CODES[competition_key]
        year = _probe_season_year(league_code)
        if year is None:
            print(f"{competition_key}: ESPN fetch failed, skipping")
            continue

        cache_path = _cache_path(competition_key, year)
        if _is_season_complete(cache_path):
            print(f"{competition_key} {year}: season already complete, skipping ESPN fetch")
            continue

        _, rows = _fetch_season_window(league_code)
        _save_rows(competition_key, year, rows)
        print(f"{competition_key} {year}: {len(rows)} ESPN rows fetched")


if __name__ == "__main__":
    main()
