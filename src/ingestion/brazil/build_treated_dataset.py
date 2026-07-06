"""Builds the treated matches CSV from the raw scraped dockets.

Reads every season cached under data/raw/brazil (see scrape_raw_matches.py),
maps each club's raw name to its normalized spelling using the manually
curated de-para table (see team_name_mapping.py and data/processed/brazil/
README.md's "Resolving unmapped team names" section), and writes the result
to data/processed/brazil/matches.csv. A raw name with no mapping yet is kept
as-is in the output and logged, with rapidfuzz suggestions, to
data/processed/brazil/unmapped_team_names_log.csv for manual review.
"""

import glob
import os
import re
from datetime import datetime

import pandas as pd

from src.ingestion.brazil.constants import COMPETITIONS
from src.ingestion.brazil.scrape_raw_matches import CACHE_DIR, load_season_csv
from src.ingestion.brazil.team_name_mapping import (
    build_lookup_tables,
    load_mapping,
    resolve_team_name,
    suggest_matches,
)

OUTPUT_PATH = "data/processed/brazil/matches.csv"
UNMAPPED_LOG_PATH = "data/processed/brazil/unmapped_team_names_log.csv"
RESULT_PATTERN = re.compile(r"^\s*(\d+)\s*[Xx]\s*(\d+)\s*$")


def load_raw_games() -> dict:
    """Loads every cached raw season as {(competition_key, year): games}."""
    raw_games_by_season = {}
    for path in glob.glob(os.path.join(CACHE_DIR, "*.csv")):
        file_name = os.path.splitext(os.path.basename(path))[0]
        competition_key, year = file_name.rsplit("_", 1)
        raw_games_by_season[(competition_key, int(year))] = load_season_csv(path)
    return raw_games_by_season


def parse_datetime(date: str, time: str) -> str:
    return datetime.strptime(f"{date} {time}", "%d/%m/%Y %H:%M").strftime("%Y-%m-%d %H:%M")


def main() -> None:
    raw_games_by_season = load_raw_games()

    mapping = load_mapping()
    lower_mapping, lower_known_names = build_lookup_tables(mapping)
    known_normalized_names = set(mapping.values())

    rows = []
    unmapped_rows = []
    for (competition_key, year), games in sorted(raw_games_by_season.items()):
        competition_label = COMPETITIONS[competition_key]
        for game_code, game in sorted(games.items()):
            home_goals, away_goals = RESULT_PATTERN.match(game["Result"]).groups()

            resolved_names = {}
            for side in ("Home", "Away"):
                raw_name = game[side]
                resolved_name, was_resolved = resolve_team_name(
                    raw_name, lower_mapping, lower_known_names
                )
                resolved_names[side] = resolved_name
                if not was_resolved:
                    suggestions = suggest_matches(raw_name, known_normalized_names)
                    unmapped_rows.append(
                        {
                            "raw_name": raw_name,
                            "competition": competition_label,
                            "season": year,
                            "game_code": game_code,
                            "suggestions": "; ".join(
                                f"{name} ({score:.0f})" for name, score in suggestions
                            ),
                        }
                    )

            rows.append(
                {
                    "competition": competition_label,
                    "season": year,
                    "match_datetime": parse_datetime(game["Date"], game["Time"]),
                    "venue": game["Stadium"],
                    "home_team": resolved_names["Home"],
                    "away_team": resolved_names["Away"],
                    "home_goals": int(home_goals),
                    "away_goals": int(away_goals),
                }
            )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["competition", "season", "match_datetime"])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} matches to {OUTPUT_PATH}")

    unmapped_df = pd.DataFrame(unmapped_rows)
    unmapped_df.to_csv(UNMAPPED_LOG_PATH, index=False)
    print(f"{len(unmapped_df)} unmapped team-name occurrences logged to {UNMAPPED_LOG_PATH}")


if __name__ == "__main__":
    main()
