"""Orchestrates the full campo-aberto pipeline end to end: scraping -> fit -> simulation -> site export.

1. Scrapes fresh Brazil match dockets, fetches ESPN's schedule, and rebuilds
   the treated matches CSV (mirrors src.ingestion.brazil.run_pipeline's
   stages -- see that module's docstring).
2. Fits poisson_home.stan on the full match history and saves posterior
   samples for historical tracking (src.models.fit).
3. Simulates the rest of the season as of the latest known match and reports
   title / continental / promotion / relegation probabilities
   (src.simulation.simulate).
4. Exports the fresh data/results/ snapshots into the static site's committed
   data (src.site.export_site_data). The site/ output still needs to be
   committed and pushed for a deploy to actually go out -- see site/README.md.
"""

import argparse

import pandas as pd

from src.constants import (
    DEFAULT_CHAINS,
    DEFAULT_CONFIGS,
    DEFAULT_ITER_WARMUP,
    DEFAULT_MATCHES_PATH,
    DEFAULT_N_DRAWS,
    DEFAULT_SEASON,
    DEFAULT_SEED,
    SITE_DIR,
)
from src.ingestion.brazil import build_treated_dataset, espn_fixtures, scrape_raw_matches
from src.models.fit import fit, save_samples
from src.simulation.config import load_competition_config
from src.simulation.results import save_results
from src.simulation.simulate import simulate_competition
from src.site.export_site_data import export_site_data


def _parse_guaranteed_slots(entries: list[str]) -> dict[str, list[str]]:
    """Repeat --guaranteed-slot TEAM:SPOT with the same team for independent
    guarantees of the same or different tiers (e.g. a team that's both this
    year's Libertadores champion and Copa do Brasil champion)."""
    guaranteed_slots: dict[str, list[str]] = {}
    for entry in entries:
        team, _, spot = entry.partition(":")
        if not spot:
            raise ValueError(f"--guaranteed-slot {entry!r} must be in TEAM:SPOT format")
        guaranteed_slots.setdefault(team, []).append(spot)
    return guaranteed_slots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--iter-warmup", type=int, default=DEFAULT_ITER_WARMUP)
    parser.add_argument(
        "--guaranteed-slot",
        action="append",
        default=[],
        metavar="TEAM:SPOT",
        help="a team with an externally guaranteed spot (e.g. a Copa do Brasil berth); repeatable",
    )
    args = parser.parse_args()
    guaranteed_slots = _parse_guaranteed_slots(args.guaranteed_slot)

    print("=== 1/4: scraping + building treated dataset ===")
    scrape_raw_matches.main()
    espn_fixtures.main()
    build_treated_dataset.main()

    df = pd.read_csv(args.matches)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    # matches.csv can now contain scheduled/postponed rows with no result
    # (see build_treated_dataset.py) -- reference_date must still land on the
    # latest *played* match, not a future fixture's date.
    reference_date = df[df["home_goals"].notna()]["match_datetime"].max()

    print("=== 2/4: fitting poisson_home.stan ===")
    # at least one posterior draw per requested Monte Carlo replicate
    iter_sampling = -(-args.n_draws // args.chains)
    mcmc_fit, teams = fit(
        args.matches,
        reference_date=reference_date,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=iter_sampling,
    )
    samples_path = save_samples(mcmc_fit, teams, args.matches)
    print(f"Saved samples to {samples_path}")

    print("=== 3/4: simulating the rest of the season ===")
    for config_path in args.configs:
        config = load_competition_config(config_path)
        result = simulate_competition(
            config,
            mcmc_fit,
            teams,
            df,
            args.season,
            reference_date,
            n_draws=args.n_draws,
            seed=args.seed,
            guaranteed_slots=guaranteed_slots,
        )
        print(f"\n=== {config.name} {args.season} (as of {reference_date.date()}) ===")
        print(result.to_string(index=False))

        results_path = save_results(result, config.name, args.season, reference_date)
        print(f"Saved results to {results_path}")

    print("=== 4/4: exporting site data ===")
    export_site_data()
    print(
        f"Site data refreshed under {SITE_DIR}/data -- review and commit that directory to deploy."
    )


if __name__ == "__main__":
    main()
