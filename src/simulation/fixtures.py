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
    # matches.csv can now carry scheduled/postponed rows with no result (see
    # src/ingestion/brazil/build_treated_dataset.py) -- guard against a
    # postponed match's stale original date landing at/before reference_date
    # while it's still actually unplayed.
    played_df = season_df[
        (season_df["match_datetime"] <= reference_date) & season_df["home_goals"].notna()
    ].copy()
    # Filtering out the None-goals rows still leaves home_goals/away_goals
    # upcast to float64 (pandas' NaN handling on the full column) -- cast
    # back to int so played_results' goals are real ints, not e.g. 2.0,
    # matching build_stan_data's same fix in src/models/data.py.
    played_df["home_goals"] = played_df["home_goals"].astype(int)
    played_df["away_goals"] = played_df["away_goals"].astype(int)

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


def real_result(
    df: pd.DataFrame,
    competition: str,
    season: int,
    reference_date: pd.Timestamp,
    home: str,
    away: str,
) -> tuple[int, int] | None:
    """Looks up one exact already-played ordered (home, away) match's real
    score, as of reference_date, or None if it hasn't been played (or
    recorded) yet.

    Unlike split_fixtures, a playoff leg's (home, away) pair isn't derivable
    combinatorially -- who meets whom depends on standings/bracket results
    only known per Monte Carlo draw (see simulate.py's _simulate_playoff_pair)
    -- so this is a direct single-pair lookup instead of a whole-season split.

    Known limitation: matches.csv has no round/stage column, so a playoff leg
    that reuses an ordered pair its own round-robin source phase already
    played (true for every pair configs/serie_b_2026.yaml's access playoff can
    produce -- a double round-robin plays both directions of every pair once,
    before the playoff reuses one of those same directions) is only
    distinguishable from that earlier round-robin match by requiring at
    least *two* played rows for the pair and taking the latest: a single
    played row is treated as still the round-robin's own fixture, not yet
    this leg, which is the correct (if conservative) call while a real access
    playoff leg is still pending -- it just also means a genuinely
    first-ever-this-season meeting between two teams (a manual/bracket_adjacent
    playoff outside a shared round-robin, not used by any current config)
    would need a second data point before it's recognized as played. No
    current config needs that case; revisit if one ever does.
    """
    season_df = df[(df["competition"] == competition) & (df["season"] == season)]
    match_df = season_df[
        (season_df["home_team"] == home)
        & (season_df["away_team"] == away)
        & (season_df["match_datetime"] <= reference_date)
        & season_df["home_goals"].notna()
    ]
    if len(match_df) < 2:
        return None
    row = match_df.sort_values("match_datetime").iloc[-1]
    return int(row["home_goals"]), int(row["away_goals"])
