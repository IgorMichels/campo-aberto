"""Scrapes a season's match dockets directly from CBF, resuming from where a
previous run left off instead of starting over.
"""

from src.ingestion.brazil.cbf_docket import try_fetch_docket
from src.ingestion.brazil.constants import GAMES_PER_SEASON

CONSECUTIVE_MISS_LIMIT = 40


def scrape_season(competition_key: str, year: int, games: dict, resume_from: int) -> dict:
    """Fetches new dockets for a season, raw (no team-name treatment).

    `resume_from` is the highest game code already seen for this season
    (capped at GAMES_PER_SEASON by the caller). Codes between 1 and
    `resume_from` not already in `games` are gaps -- postponed matches, or
    games that simply hadn't been played yet last time -- and are retried
    first. Probing then continues forward from `resume_from` until either
    CONSECUTIVE_MISS_LIMIT dockets in a row come back missing (the live edge
    of an in-progress season) or GAMES_PER_SEASON is reached. A game already
    present in `games` is never re-fetched.
    """
    gap_codes = sorted(set(range(1, resume_from + 1)) - {int(code) for code in games})
    for game_code in gap_codes:
        docket = try_fetch_docket(competition_key, year, game_code)
        if docket is not None:
            games[f"{game_code:03d}"] = docket

    consecutive_misses = 0
    game_code = resume_from
    while consecutive_misses < CONSECUTIVE_MISS_LIMIT and game_code < GAMES_PER_SEASON:
        game_code += 1
        docket = try_fetch_docket(competition_key, year, game_code)
        if docket is not None:
            games[f"{game_code:03d}"] = docket
            consecutive_misses = 0
        else:
            consecutive_misses += 1

    return games
