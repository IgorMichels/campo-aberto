"""CLI: simulates the rest of a competition's season and reports spot probabilities.

Fits poisson_home.stan on every match up to --reference-date only (so results
after that date can never leak into the team-strength estimates), then Monte
Carlo simulates the remainder of --season to completion for every competition
config passed in, applying that competition's phases (configs/*.yaml -- see
configs/README.md for the schema) to turn final standings into spot
probabilities (title, promotion, relegation, etc).

Running this for a range of past --reference-date values is how you track
the evolution of these probabilities over the season:

    python -m src.simulation.run --reference-date 2026-04-01
    python -m src.simulation.run --reference-date 2026-07-01
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
from src.models.data import build_stan_data
from src.models.fit import fit_stan_data
from src.simulation.config import load_competition_config
from src.simulation.results import save_results
from src.simulation.simulate import simulate_competition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--reference-date", required=True, help="e.g. 2026-06-30")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--iter-warmup", type=int, default=DEFAULT_ITER_WARMUP)
    args = parser.parse_args()

    df = pd.read_csv(args.matches)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    reference_date = pd.Timestamp(args.reference_date)

    train_df = df[df["match_datetime"] <= reference_date]
    stan_data, teams = build_stan_data(train_df, reference_date=reference_date)
    # at least one posterior draw per requested Monte Carlo replicate
    iter_sampling = -(-args.n_draws // args.chains)
    mcmc_fit = fit_stan_data(
        stan_data, chains=args.chains, iter_warmup=args.iter_warmup, iter_sampling=iter_sampling
    )

    for config_path in args.configs:
        config = load_competition_config(config_path)
        result = simulate_competition(
            config, mcmc_fit, teams, df, args.season, reference_date, n_draws=args.n_draws, seed=args.seed
        )
        print(f"=== {config.name} {args.season} (as of {reference_date.date()}) ===")
        print(result.to_string(index=False))

        results_path = save_results(result, config.name, args.season, reference_date)
        print(f"Saved results to {results_path}")
        print()


if __name__ == "__main__":
    main()
