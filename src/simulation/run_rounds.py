"""CLI: backtests every "round" of a season instead of a single --reference-date.

A round is a maximal run of consecutive calendar days (gap of at most 1 day)
that has at least one match for a given competition+season -- e.g. a normal
weekend or midweek round. Each round's reference_date is its *last* match day
(the point at which every one of that round's results is known), so running
this reproduces the timeseries of spot probabilities the way they actually
evolved across the season, one snapshot per round.

Loops over every configs/*.yaml whose filename's season suffix (see
configs/README.md's "Per-season configs" section) is in --seasons, grouping
by competition. Two competitions sharing a round's last day (common in
Brazil, where Serie A and Serie B are usually scheduled the same weekends)
reuse a single Stan fit instead of refitting twice for the same date.

    python -m src.simulation.run_rounds --seasons 2025 2026
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.constants import (
    DEFAULT_CHAINS,
    DEFAULT_ITER_WARMUP,
    DEFAULT_MATCHES_PATH,
    DEFAULT_N_DRAWS,
    DEFAULT_SEED,
)
from src.models.data import build_stan_data
from src.models.fit import fit_stan_data
from src.simulation import fixtures
from src.simulation.config import CompetitionConfig, load_competition_config
from src.simulation.results import save_results
from src.simulation.run import _parse_guaranteed_slots
from src.simulation.simulate import simulate_competition

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_SEASON_SUFFIX = re.compile(r"_(\d{4})\.yaml$")


def load_configs_by_season(seasons: list[int]) -> dict[int, list[CompetitionConfig]]:
    """{season: [config, ...]} for every configs/*.yaml whose filename's season
    suffix (e.g. serie_a_2025.yaml -> 2025) is in `seasons`."""
    configs_by_season: dict[int, list[CompetitionConfig]] = defaultdict(list)
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        match = _SEASON_SUFFIX.search(path.name)
        if match is None:
            continue
        season = int(match.group(1))
        if season in seasons:
            configs_by_season[season].append(load_competition_config(path))
    return configs_by_season


def round_reference_dates(df: pd.DataFrame, competition: str, season: int) -> list[pd.Timestamp]:
    """One reference_date per round -- the last calendar day of each maximal
    run of consecutive match days for this competition+season."""
    season_df = df[(df["competition"] == competition) & (df["season"] == season)]
    days = sorted(season_df["match_datetime"].dt.normalize().unique())
    if not days:
        return []

    rounds = [[days[0]]]
    for day in days[1:]:
        if (day - rounds[-1][-1]).days <= 1:
            rounds[-1].append(day)
        else:
            rounds.append([day])
    return [pd.Timestamp(round_days[-1]) for round_days in rounds]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matches", default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--iter-warmup", type=int, default=DEFAULT_ITER_WARMUP)
    parser.add_argument(
        "--guaranteed-slot",
        action="append",
        default=[],
        metavar="TEAM:SPOT",
        help="a team with an externally guaranteed spot, applied at every round regardless of date "
        "(use a config's guaranteed_slots list instead for date-gated guarantees); repeatable",
    )
    args = parser.parse_args()
    guaranteed_slots = _parse_guaranteed_slots(args.guaranteed_slot)

    df = pd.read_csv(args.matches)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])

    configs_by_season = load_configs_by_season(args.seasons)

    # {reference_date: [(config, season), ...]}, so two competitions whose
    # round happens to end the same day share one Stan fit.
    work: dict[pd.Timestamp, list[tuple[CompetitionConfig, int]]] = defaultdict(list)
    for season, configs in configs_by_season.items():
        for config in configs:
            for reference_date in round_reference_dates(df, config.name, season):
                work[reference_date].append((config, season))

    iter_sampling = -(-args.n_draws // args.chains)
    for reference_date in sorted(work):
        train_df = df[df["match_datetime"] <= reference_date]
        stan_data, teams = build_stan_data(train_df, reference_date=reference_date)
        mcmc_fit = fit_stan_data(
            stan_data, chains=args.chains, iter_warmup=args.iter_warmup, iter_sampling=iter_sampling
        )

        trained_teams = set(teams)
        for config, season in work[reference_date]:
            missing = set(fixtures.season_teams(df, config.name, season)) - trained_teams
            if missing:
                # A team with zero matches before reference_date has no Stan
                # attack/defense estimate at all (not just a stale one) --
                # simulating its remaining fixtures would need to invent a
                # strength for it out of thin air, so skip this round instead.
                print(
                    f"=== {config.name} {season} (as of {reference_date.date()}) ==="
                    f"\nSkipped: {sorted(missing)} haven't played yet\n"
                )
                continue

            result = simulate_competition(
                config,
                mcmc_fit,
                teams,
                df,
                season,
                reference_date,
                n_draws=args.n_draws,
                seed=args.seed,
                guaranteed_slots=guaranteed_slots,
            )
            print(f"=== {config.name} {season} (as of {reference_date.date()}) ===")
            print(result.to_string(index=False))

            results_path = save_results(result, config.name, season, reference_date)
            print(f"Saved results to {results_path}")
            print()


if __name__ == "__main__":
    main()
