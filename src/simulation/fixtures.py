"""Derives a season's fixture set and splits it into played/remaining as of a reference date.

Only supports a double round-robin (RoundRobinPhaseConfig.legs == 2, the
default -- see configs/README.md): every ordered (home, away) pair among the
season's clubs occurs exactly once, so the complete fixture list is fully
determined combinatorially from the roster alone -- no actual round-by-round
schedule needed, just which of those pairs are already in matches.csv by the
reference date. Serie A and Serie B are both this shape (19 rounds home, 19
away -- REC Art. 14/11); a single round-robin (legs == 1, e.g. a World Cup
group stage) has no such derivation and isn't implemented (simulate.py raises
NotImplementedError for it before reaching this module).
"""

import pandas as pd


def season_teams(df: pd.DataFrame, competition: str, season: int) -> list[str]:
    season_df = df[(df["competition"] == competition) & (df["season"] == season)]
    return sorted(set(season_df["home_team"]) | set(season_df["away_team"]))


def split_fixtures(
    df: pd.DataFrame,
    competition: str,
    season: int,
    reference_date: pd.Timestamp,
    teams: list[str] | None = None,
) -> tuple[list[tuple], list[tuple[str, str]], list[str]]:
    """Splits a season into what's already played and what's left, as of reference_date.

    Args:
        teams: restricts the fixture set to this explicit roster instead of deriving
            the full season roster -- used to build a single group's fixtures within
            a grouped round_robin phase (see simulate.py).

    Returns:
        (played_results, remaining_fixtures, teams).
        played_results: list of (home, away, home_goals, away_goals).
        remaining_fixtures: list of (home, away) pairs not yet played.
    """
    if teams is None:
        teams = season_teams(df, competition, season)
    team_set = set(teams)
    all_fixtures = {(home, away) for home in teams for away in teams if home != away}

    season_df = df[(df["competition"] == competition) & (df["season"] == season)]
    played_df = season_df[season_df["match_datetime"] <= reference_date]

    played_results = [
        row
        for row in played_df[["home_team", "away_team", "home_goals", "away_goals"]].itertuples(
            index=False, name=None
        )
        if row[0] in team_set and row[1] in team_set
    ]
    played_pairs = {(home, away) for home, away, _, _ in played_results}
    remaining_fixtures = sorted(all_fixtures - played_pairs)

    return played_results, remaining_fixtures, teams
