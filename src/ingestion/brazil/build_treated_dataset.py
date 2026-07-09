"""Builds the treated matches CSV from the raw scraped dockets, merged with
ESPN's schedule.

Reads every season cached under data/raw/brazil/cbf (see scrape_raw_matches.py),
maps each club's raw name to its normalized spelling using the manually
curated de-para table (see team_name_mapping.py and data/processed/brazil/
README.md's "Resolving unmapped team names" section), and writes the result
to data/processed/brazil/matches.csv. A raw name with no mapping yet is kept
as-is in the output and logged, with rapidfuzz suggestions, to
data/processed/brazil/unmapped_team_names_log.csv for manual review.

CBF's docket score is always authoritative. ESPN (see espn_fixtures.py) is
merged in only to fill the schedule -- a not-yet-played match's date, or a
played match CBF's docket scraping hasn't caught up to yet. When both sources
have a played score for the same match and they disagree, CBF's score is kept
in matches.csv regardless, and the disagreement is flagged in
data/processed/brazil/score_discrepancies.csv for manual review.
"""

import csv
import glob
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.ingestion.brazil.constants import (
    CBF_CACHE_DIR,
    COMPETITIONS,
    DISCREPANCY_LOG_PATH,
    ESPN_CACHE_DIR,
    OUTPUT_PATH,
    UNMAPPED_LOG_PATH,
)
from src.ingestion.brazil.scrape_raw_matches import load_season_csv
from src.ingestion.brazil.team_name_mapping import (
    build_lookup_tables,
    load_mapping,
    resolve_team_name,
    suggest_matches,
)

RESULT_PATTERN = re.compile(r"^\s*(\d+)\s*[Xx]\s*(\d+)\s*$")

_UTC = ZoneInfo("UTC")
_BRAZIL_TZ = ZoneInfo("America/Sao_Paulo")


def load_raw_games() -> dict:
    """Loads every cached raw season as {(competition_key, year): games}."""
    raw_games_by_season = {}
    for path in glob.glob(os.path.join(CBF_CACHE_DIR, "*.csv")):
        file_name = os.path.splitext(os.path.basename(path))[0]
        competition_key, year = file_name.rsplit("_", 1)
        raw_games_by_season[(competition_key, int(year))] = load_season_csv(path)
    return raw_games_by_season


def load_espn_games() -> dict:
    """Loads every cached ESPN season as {(competition_key, year): [row, ...]}
    from data/raw/brazil/espn (see espn_fixtures.py), same
    file_name.rsplit("_", 1) convention load_raw_games() already uses.
    """
    espn_games_by_season = {}
    for path in glob.glob(os.path.join(ESPN_CACHE_DIR, "*.csv")):
        file_name = os.path.splitext(os.path.basename(path))[0]
        competition_key, year = file_name.rsplit("_", 1)
        with open(path, encoding="utf-8", newline="") as f:
            rows = [
                {
                    "date": row["date"],
                    "venue": row["venue"],
                    "home_team_raw": row["home_team_raw"],
                    "away_team_raw": row["away_team_raw"],
                    "home_goals": int(row["home_goals"]) if row["home_goals"] else None,
                    "away_goals": int(row["away_goals"]) if row["away_goals"] else None,
                    "status": row["status"],
                }
                for row in csv.DictReader(f)
            ]
        espn_games_by_season[(competition_key, int(year))] = rows
    return espn_games_by_season


def parse_datetime(date: str, time: str) -> str:
    return datetime.strptime(f"{date} {time}", "%d/%m/%Y %H:%M").strftime("%Y-%m-%d %H:%M")


def espn_datetime_to_brazil_local(iso_utc: str) -> str:
    """Converts an ESPN "YYYY-MM-DDTHH:MMZ" UTC datetime to Brazil local time
    ("YYYY-MM-DD HH:MM", matching parse_datetime's output format). Brazil has
    used a fixed UTC-3 offset with no DST since 2019, so this is a
    straightforward zoneinfo conversion, not a "which offset applies" problem.
    """
    dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=_UTC)
    return dt.astimezone(_BRAZIL_TZ).strftime("%Y-%m-%d %H:%M")


def _resolve_and_log(
    raw_name: str,
    competition_label: str,
    year: int,
    game_code,
    lower_mapping: dict,
    lower_known_names: dict,
    known_normalized_names: set,
    unmapped_rows: list,
) -> str:
    """Resolves one raw team name, appending to unmapped_rows (shared across
    both CBF and ESPN sources -- one unified log) when it can't be resolved.
    """
    resolved_name, was_resolved = resolve_team_name(raw_name, lower_mapping, lower_known_names)
    if not was_resolved:
        suggestions = suggest_matches(raw_name, known_normalized_names)
        unmapped_rows.append(
            {
                "raw_name": raw_name,
                "competition": competition_label,
                "season": year,
                "game_code": game_code,
                "suggestions": "; ".join(f"{name} ({score:.0f})" for name, score in suggestions),
            }
        )
    return resolved_name


def main() -> None:
    raw_games_by_season = load_raw_games()
    espn_games_by_season = load_espn_games()

    mapping = load_mapping()
    lower_mapping, lower_known_names = build_lookup_tables(mapping)
    known_normalized_names = set(mapping.values())

    rows = []
    unmapped_rows = []
    for (competition_key, year), games in sorted(raw_games_by_season.items()):
        competition_label = COMPETITIONS[competition_key]
        for game_code, game in sorted(games.items()):
            home_goals, away_goals = RESULT_PATTERN.match(game["Result"]).groups()

            resolved_names = {
                side: _resolve_and_log(
                    game[side],
                    competition_label,
                    year,
                    game_code,
                    lower_mapping,
                    lower_known_names,
                    known_normalized_names,
                    unmapped_rows,
                )
                for side in ("Home", "Away")
            }

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
                    "status": "played",
                }
            )

    # {(competition, season, home, away): cbf_row} -- round-robin means this
    # key is unique within a competition+season.
    cbf_lookup = {
        (row["competition"], row["season"], row["home_team"], row["away_team"]): row for row in rows
    }

    discrepancy_rows = []
    espn_only_rows = []
    for (competition_key, year), games in sorted(espn_games_by_season.items()):
        competition_label = COMPETITIONS[competition_key]
        for game in games:
            resolved_names = {
                side: _resolve_and_log(
                    game[side],
                    competition_label,
                    year,
                    "",  # ESPN rows have no CBF game code
                    lower_mapping,
                    lower_known_names,
                    known_normalized_names,
                    unmapped_rows,
                )
                for side in ("home_team_raw", "away_team_raw")
            }
            home_team = resolved_names["home_team_raw"]
            away_team = resolved_names["away_team_raw"]

            cbf_row = cbf_lookup.get((competition_label, year, home_team, away_team))
            if cbf_row is not None:
                # CBF's score (already in `rows`, status="played") is
                # authoritative -- never add a duplicate row for this match.
                if game["status"] == "played" and (
                    game["home_goals"] != cbf_row["home_goals"]
                    or game["away_goals"] != cbf_row["away_goals"]
                ):
                    discrepancy_rows.append(
                        {
                            "competition": competition_label,
                            "season": year,
                            "home_team": home_team,
                            "away_team": away_team,
                            "cbf_home_goals": cbf_row["home_goals"],
                            "cbf_away_goals": cbf_row["away_goals"],
                            "espn_home_goals": game["home_goals"],
                            "espn_away_goals": game["away_goals"],
                        }
                    )
                continue

            # CBF hasn't confirmed this game yet -- the normal case for a
            # genuinely future match, or a played match CBF's docket scraping
            # hasn't caught up to (a known pre-existing gap, see load_raw_games's
            # module docstring / README). Either way this dataset never trusts
            # a score it hasn't gotten from CBF, so home_goals/away_goals stay
            # empty; an ESPN row that *did* report a final score here still
            # gets "scheduled" (it did happen, just isn't CBF-confirmed yet --
            # not worth inventing a fourth status for this rare edge case).
            status = "postponed" if game["status"] == "postponed" else "scheduled"
            espn_only_rows.append(
                {
                    "competition": competition_label,
                    "season": year,
                    "match_datetime": espn_datetime_to_brazil_local(game["date"]),
                    "venue": game["venue"],
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": None,
                    "away_goals": None,
                    "status": status,
                }
            )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = pd.DataFrame(rows + espn_only_rows).sort_values(
        ["competition", "season", "match_datetime"]
    )
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} matches to {OUTPUT_PATH}")

    unmapped_df = pd.DataFrame(
        unmapped_rows, columns=["raw_name", "competition", "season", "game_code", "suggestions"]
    )
    unmapped_df.to_csv(UNMAPPED_LOG_PATH, index=False)
    print(f"{len(unmapped_df)} unmapped team-name occurrences logged to {UNMAPPED_LOG_PATH}")

    discrepancy_df = pd.DataFrame(
        discrepancy_rows,
        columns=[
            "competition",
            "season",
            "home_team",
            "away_team",
            "cbf_home_goals",
            "cbf_away_goals",
            "espn_home_goals",
            "espn_away_goals",
        ],
    )
    discrepancy_df.to_csv(DISCREPANCY_LOG_PATH, index=False)
    print(f"{len(discrepancy_df)} CBF/ESPN score discrepancies logged to {DISCREPANCY_LOG_PATH}")


if __name__ == "__main__":
    main()
