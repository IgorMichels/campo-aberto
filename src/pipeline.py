"""Orchestrates the full campo-aberto pipeline end to end: scraping -> fit -> simulation -> site export.

1. Scrapes fresh Brazil match dockets, fetches ESPN's schedule, and rebuilds
   the treated matches CSV (mirrors src.ingestion.brazil.run_pipeline's
   stages -- see that module's docstring).
2. Fits poisson_home.stan on the full match history and saves posterior
   samples for historical tracking (src.models.fit).
3. Simulates the rest of the season as of the most recent Monday or Friday
   (src.simulation.run_rounds.latest_checkpoint_date) and reports title /
   continental / promotion / relegation probabilities (src.simulation.simulate)
   -- the same twice-weekly cadence src.simulation.run_rounds backtests on,
   so this command's single ad-hoc "run for today" checkpoint always lands
   where a walk-forward backtest would too, instead of drifting onto
   whatever day of the week the latest match happened to be played.
4. Exports the fresh data into the static site's committed data: both the
   standings/odds export (src.site.export_site_data, data/results/ snapshots
   -> site/data/manifest.json + <slug>/<season>.json) and the Confrontos
   export (src.site.export_matches_data, matches.csv + the same results ->
   site/data/matches_manifest.json + <slug>/matches_<season>.json +
   params.json), both as of the same reference_date so they agree on
   "as of".
5. Recomputes the model-statistics page's aggregate metrics
   (src.site.model_stats, exact-scoreline/direction accuracy, Brier,
   calibration -- site/data/model_stats.json) from the played-match data
   stage 4 just wrote, since it depends on played_<season>.json already
   being on disk. The site/ output still needs to be reviewed and committed
   for a deploy to actually go out -- see site/README.md.
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
from src.models.registry import DEFAULT_MODEL, MODEL_REGISTRY
from src.simulation.config import load_competition_config
from src.simulation.results import save_results
from src.simulation.run_rounds import latest_checkpoint_date
from src.simulation.simulate import simulate_competition
from src.site.export_matches_data import export_matches_data
from src.site.export_site_data import export_site_data
from src.site.model_stats import export_model_stats


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
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_REGISTRY))
    parser.add_argument(
        "--guaranteed-slot",
        action="append",
        default=[],
        metavar="TEAM:SPOT",
        help="a team with an externally guaranteed spot (e.g. a Copa do Brasil berth); repeatable",
    )
    args = parser.parse_args()
    guaranteed_slots = _parse_guaranteed_slots(args.guaranteed_slot)

    print("=== 1/5: scraping + building treated dataset ===")
    scrape_raw_matches.main()
    espn_fixtures.main()
    build_treated_dataset.main()

    df = pd.read_csv(args.matches)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    # Same fixed Monday/Friday cadence src.simulation.run_rounds backtests on
    # -- this is meant to be the standard "update the site" command, so a
    # single ad-hoc run must land on a checkpoint the walk-forward backtest
    # would also produce, not an arbitrary result day (e.g. a Saturday) that
    # schedule can never recompute or line up with later.
    reference_date = latest_checkpoint_date()

    print(f"=== 2/5: fitting {args.model} ===")
    # at least one posterior draw per requested Monte Carlo replicate
    iter_sampling = -(-args.n_draws // args.chains)
    mcmc_fit, teams = fit(
        args.matches,
        reference_date=reference_date,
        model=args.model,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=iter_sampling,
    )
    samples_path = save_samples(mcmc_fit, teams, args.matches, model=args.model)
    print(f"Saved samples to {samples_path}")

    print("=== 3/5: simulating the rest of the season ===")
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
            model=args.model,
        )
        print(f"\n=== {config.name} {args.season} (as of {reference_date.date()}) ===")
        print(result.to_string(index=False))

        results_path = save_results(result, config.name, args.season, reference_date)
        print(f"Saved results to {results_path}")

    print("=== 4/5: exporting site data ===")
    export_site_data()
    export_matches_data(now=reference_date)

    print("=== 5/5: recomputing model-statistics page metrics ===")
    export_model_stats()

    print(
        f"Site data refreshed under {SITE_DIR}/data -- review and commit that directory to deploy."
    )


if __name__ == "__main__":
    main()
