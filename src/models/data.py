"""Builds Stan data for poisson_home.stan from a matches CSV.

Works with any CSV following the `competition, season, match_datetime, venue,
home_team, away_team, home_goals, away_goals` schema shared across countries
under data/processed/ -- there's nothing Brazil-specific here.
"""

import pandas as pd

from src.constants import DEFAULT_HALF_LIFE_WEEKS, DEFAULT_MAX_WEEKS_AGO


def _time_weight(weeks_ago: pd.Series, half_life_weeks: float) -> pd.Series:
    """Exponential half-life decay (see src/constants.py)."""
    return 0.5 ** (weeks_ago / half_life_weeks)


def build_stan_data(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    half_life_weeks: float = DEFAULT_HALF_LIFE_WEEKS,
    max_weeks_ago: int = DEFAULT_MAX_WEEKS_AGO,
) -> tuple[dict, list[str]]:
    """Builds the data dict expected by poisson_home.stan from an in-memory matches DataFrame.

    Each match's game_weight decays with the number of complete weeks elapsed between
    its match_datetime and reference_date (see _time_weight); matches more than
    max_weeks_ago weeks old are dropped entirely rather than just down-weighted.

    Args:
        df: DataFrame with columns competition, season, match_datetime, venue,
            home_team, away_team, home_goals, away_goals.
        reference_date: the date weeks_ago is measured from. Defaults to df's
            latest match_datetime.

    Returns:
        (stan_data, teams), where teams[i - 1] is the team name for Stan index i.
    """
    # matches.csv can now contain scheduled/postponed rows with no result
    # (see src/ingestion/brazil/build_treated_dataset.py) -- only played
    # matches ever inform the fit. Filtering upcasts home_goals/away_goals
    # to float64 (pandas' NaN handling), so cast back to int once filtered.
    df = df[df["home_goals"].notna()].copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    if reference_date is None:
        reference_date = df["match_datetime"].max()

    weeks_ago = (reference_date - df["match_datetime"]).dt.days // 7
    recent_enough = weeks_ago <= max_weeks_ago
    df = df[recent_enough]
    weeks_ago = weeks_ago[recent_enough]

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_index = {team: i + 1 for i, team in enumerate(teams)}

    stan_data = {
        "N": len(df),
        "T": len(teams),
        "team_i": df["home_team"].map(team_index).tolist(),
        "team_j": df["away_team"].map(team_index).tolist(),
        "y_i": df["home_goals"].tolist(),
        "y_j": df["away_goals"].tolist(),
        "game_weight": _time_weight(weeks_ago, half_life_weeks).tolist(),
    }
    return stan_data, teams


def load_stan_data(
    matches_path: str, reference_date: pd.Timestamp | None = None, **kwargs
) -> tuple[dict, list[str]]:
    """Loads a matches CSV and builds the data dict expected by poisson_home.stan.

    Args:
        matches_path: path to a matches CSV (see build_stan_data for the schema).
        reference_date: forwarded to build_stan_data.
        **kwargs: forwarded to build_stan_data (half_life_weeks, max_weeks_ago).

    Returns:
        (stan_data, teams), where teams[i - 1] is the team name for Stan index i.
    """
    df = pd.read_csv(matches_path)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    return build_stan_data(df, reference_date=reference_date, **kwargs)
