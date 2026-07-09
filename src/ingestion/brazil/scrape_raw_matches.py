"""Scrapes Brazilian Serie A / Serie B match dockets directly from CBF since 2020.

Only fetches and parses the raw dockets, exactly as they come from CBF -- no
team-name treatment happens here (see team_name_mapping.py and
build_treated_dataset.py for that). Runs are incremental: each season is
cached as a CSV under CBF_CACHE_DIR, keyed by the source's own game_id. A
finished season (GAMES_PER_SEASON games found) is never re-scraped; an
in-progress one resumes from its own highest game_id instead of restarting,
after first re-checking whichever lower game_ids are still missing.
"""

import csv
import os

from src.ingestion.brazil.constants import (
    CBF_CACHE_DIR,
    COMPETITIONS,
    END_YEAR,
    GAMES_PER_SEASON,
    START_YEAR,
)
from src.ingestion.brazil.season_scraper import scrape_season

FIELDNAMES = ["game_id", "date", "time", "stadium", "home_team", "away_team", "result"]


def _cache_path(competition_key: str, year: int) -> str:
    return os.path.join(CBF_CACHE_DIR, f"{competition_key}_{year}.csv")


def load_season_csv(path: str) -> dict:
    """Loads a raw season CSV into {game_id: {Date, Time, Stadium, Home, Away, Result}}."""
    with open(path, encoding="utf-8", newline="") as f:
        return {
            row["game_id"]: {
                "Date": row["date"],
                "Time": row["time"],
                "Stadium": row["stadium"],
                "Home": row["home_team"],
                "Away": row["away_team"],
                "Result": row["result"],
            }
            for row in csv.DictReader(f)
        }


def _load_games(competition_key: str, year: int) -> dict:
    path = _cache_path(competition_key, year)
    if not os.path.exists(path):
        return {}
    return load_season_csv(path)


def _save_games(competition_key: str, year: int, games: dict) -> None:
    os.makedirs(CBF_CACHE_DIR, exist_ok=True)
    with open(_cache_path(competition_key, year), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for game_id in sorted(games, key=int):
            game = games[game_id]
            writer.writerow(
                {
                    "game_id": game_id,
                    "date": game["Date"],
                    "time": game["Time"],
                    "stadium": game["Stadium"],
                    "home_team": game["Home"],
                    "away_team": game["Away"],
                    "result": game["Result"],
                }
            )


def scrape_season_games(competition_key: str, year: int) -> dict:
    """Scrapes a season's raw dockets, reusing the local CSV cache whenever
    possible and only probing CBF for games it doesn't already have.
    """
    games = _load_games(competition_key, year)
    if len(games) >= GAMES_PER_SEASON:
        return games

    resume_from = min(max((int(game_id) for game_id in games), default=0), GAMES_PER_SEASON)
    games = scrape_season(competition_key, year, games, resume_from)
    _save_games(competition_key, year, games)
    return games


def main() -> None:
    for competition_key in COMPETITIONS:
        for year in range(START_YEAR, END_YEAR + 1):
            games = scrape_season_games(competition_key, year)
            print(f"{competition_key} {year}: {len(games)} games scraped")


if __name__ == "__main__":
    main()
