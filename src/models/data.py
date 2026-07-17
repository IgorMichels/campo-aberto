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


STAYED_TOP = 1  # ficou-A
ELEVATOR = 2  # elevador-A-B (subiu ou desceu entre as 2 competições)
STAYED_SECOND = 3  # ficou-B
ARRIVED_FROM_BELOW = 4  # elevador-B-C (sem dado na temporada anterior)


def _prior_groups(df: pd.DataFrame, teams: list[str]) -> list[int]:
    """Classifies each team into 1 of 4 fixed hierarchical-prior groups
    (see src/models/stan_models/hierarchical_home.stan), based on which of the two
    tracked competitions it played in its own most recent season (within
    df) versus the season immediately before that -- fixed for a whole
    season (recomputed "at the turn of the year", not match by match):

      1. STAYED_TOP: played the alphabetically-first competition (Serie A)
         both last season and this one.
      2. ELEVATOR: played one tracked competition last season and the
         OTHER this season (promoted or relegated between them) -- one
         group regardless of direction.
      3. STAYED_SECOND: played the alphabetically-second competition
         (Serie B) both last season and this one.
      4. ARRIVED_FROM_BELOW: no data at all for the season before its most
         recent one in df -- a proxy for "promoted from a division this
         pipeline doesn't ingest" (e.g. Serie C, which has no rows in
         matches.csv at all), inferred from the ABSENCE of prior-season
         data, never from a literal competition name.

    Assumes df has EXACTLY 2 distinct `competition` values (true for
    today's Brazil pipeline: Serie A + Serie B) and that alphabetical order
    matches tier order (holds for "Serie A" < "Serie B") -- raises
    ValueError otherwise, rather than silently guessing an N-competition
    scheme nobody has designed or tested.

    Returns groups: groups[i] is teams[i]'s 1-based group (1-4).
    """
    competitions = sorted(df["competition"].unique())
    if len(competitions) != 2:
        raise ValueError(
            f"_prior_groups assumes exactly 2 tracked competitions, got {competitions}"
        )
    top, _second = competitions

    long = pd.concat(
        [
            df[["home_team", "season", "competition"]].rename(columns={"home_team": "team"}),
            df[["away_team", "season", "competition"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    by_team_season = long.groupby(["team", "season"])["competition"].apply(set)

    def _group_for(team: str) -> int:
        seasons = by_team_season.loc[team]
        latest_season = seasons.index.max()
        this_year = seasons[latest_season]
        last_year = seasons.get(latest_season - 1, set())
        if not last_year:
            return ARRIVED_FROM_BELOW
        if this_year == last_year:
            return STAYED_TOP if this_year == {top} else STAYED_SECOND
        return ELEVATOR

    return [_group_for(team) for team in teams]


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
        Also includes "group" (see _prior_groups), a fixed 4-way hierarchical-
        prior classification only src.models.hierarchical_home.stan reads --
        every other registered model's .stan file simply ignores it.
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
    try:
        stan_data["group"] = _prior_groups(df, teams)
    except ValueError:
        # hierarchical_home's grouping isn't computable for this data (e.g.
        # not exactly 2 tracked competitions) -- every other registered
        # model ignores "group" entirely, so simply omitting it here keeps
        # build_stan_data generic. Only fitting hierarchical_home on such
        # data would fail, at Stan's own data validation -- the right place
        # for that to surface, not here.
        pass
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
