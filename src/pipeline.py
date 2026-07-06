"""Orchestrates the full campo-aberto pipeline end to end: scraping -> fit -> simulation.

1. Scrapes fresh Brazil match dockets and rebuilds the treated matches CSV
   (src.ingestion.brazil.run_pipeline).
2. Fits poisson_home.stan on the full match history and saves posterior
   samples for historical tracking (src.models.fit).
3. Simulates the rest of the season as of the latest known match and reports
   title / continental / promotion / relegation probabilities
   (src.simulation.simulate).
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
)
from src.ingestion.brazil import build_treated_dataset, scrape_raw_matches
from src.models.fit import fit, save_samples
from src.simulation.config import load_competition_config
from src.simulation.results import save_results
from src.simulation.simulate import simulate_competition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--iter-warmup", type=int, default=DEFAULT_ITER_WARMUP)
    args = parser.parse_args()

    print("=== 1/3: scraping + building treated dataset ===")
    scrape_raw_matches.main()
    build_treated_dataset.main()

    df = pd.read_csv(args.matches)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    reference_date = df["match_datetime"].max()

    print("=== 2/3: fitting poisson_home.stan ===")
    # at least one posterior draw per requested Monte Carlo replicate
    iter_sampling = -(-args.n_draws // args.chains)
    mcmc_fit, teams = fit(
        args.matches,
        weight_reference_date=reference_date,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=iter_sampling,
    )
    samples_path = save_samples(mcmc_fit, teams, args.matches)
    print(f"Saved samples to {samples_path}")

    print("=== 3/3: simulating the rest of the season ===")
    for config_path in args.configs:
        config = load_competition_config(config_path)
        result = simulate_competition(
            config, mcmc_fit, teams, df, args.season, reference_date, n_draws=args.n_draws, seed=args.seed
        )
        print(f"\n=== {config.name} {args.season} (as of {reference_date.date()}) ===")
        print(result.to_string(index=False))

        results_path = save_results(result, config.name, args.season, reference_date)
        print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
