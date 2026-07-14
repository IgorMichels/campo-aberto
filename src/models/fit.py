"""Fits a candidate model's .stan file on a matches CSV and reports posterior
team strengths.

Generic over both the matches CSV passed in -- pass any competition/country's
data/processed/.../matches.csv (see src/models/data.py for the expected
schema) -- and the model fit -- pass any src.models.registry.MODEL_REGISTRY
key (see src/models/adapter.py for what a model declares about its own
parameter names).
"""

import argparse
import os

import pandas as pd
from cmdstanpy import CmdStanMCMC, CmdStanModel

from src.constants import DEFAULT_MATCHES_PATH, SAMPLES_DIR
from src.models.data import load_stan_data
from src.models.registry import DEFAULT_MODEL, MODEL_REGISTRY


def fit_stan_data(stan_data: dict, model: str = DEFAULT_MODEL, **sample_kwargs) -> CmdStanMCMC:
    """Compiles `model`'s .stan file and samples it on an already-built stan_data dict."""
    stan_model = CmdStanModel(stan_file=MODEL_REGISTRY[model].stan_file)
    return stan_model.sample(data=stan_data, **sample_kwargs)


def fit(
    matches_path: str,
    reference_date: pd.Timestamp | None = None,
    model: str = DEFAULT_MODEL,
    **sample_kwargs,
) -> tuple[CmdStanMCMC, list[str]]:
    """Compiles `model`'s .stan file and samples it on the given matches CSV.

    Args:
        matches_path: path to a matches CSV (see load_stan_data).
        reference_date: forwarded to load_stan_data, i.e. the date each match's
            time-decay weight is measured from. Defaults to the matches CSV's
            latest match_datetime.
        model: a src.models.registry.MODEL_REGISTRY key.
        **sample_kwargs: forwarded to CmdStanModel.sample (e.g. chains, seed).

    Returns:
        (mcmc_fit, teams), where teams[i - 1] names Stan index i.
    """
    stan_data, teams = load_stan_data(matches_path, reference_date=reference_date)
    mcmc_fit = fit_stan_data(stan_data, model=model, **sample_kwargs)
    return mcmc_fit, teams


def summarize_teams(
    mcmc_fit: CmdStanMCMC, teams: list[str], model: str = DEFAULT_MODEL
) -> pd.DataFrame:
    """Builds a table of posterior mean team-strength params, one column per
    `model`'s declared team_param_names, sorted by the first of those
    descending (e.g. "attack" for poisson_home)."""
    adapter = MODEL_REGISTRY[model]
    draws = mcmc_fit.draws_pd()
    rows = [
        {
            "team": team,
            **{param: draws[f"{param}[{i}]"].mean() for param in adapter.team_param_names},
        }
        for i, team in enumerate(teams, start=1)
    ]
    return pd.DataFrame(rows).sort_values(adapter.team_param_names[0], ascending=False)


def samples_long(
    mcmc_fit: CmdStanMCMC, teams: list[str], model: str = DEFAULT_MODEL
) -> pd.DataFrame:
    """Builds a long-format DataFrame of posterior team-strength draws, one
    column per `model`'s declared team_param_names."""
    adapter = MODEL_REGISTRY[model]
    draws = mcmc_fit.draws_pd()
    frames = [
        pd.DataFrame(
            {
                "team": team,
                "draw": range(len(draws)),
                **{param: draws[f"{param}[{i}]"] for param in adapter.team_param_names},
            }
        )
        for i, team in enumerate(teams, start=1)
    ]
    return pd.concat(frames, ignore_index=True)


def latest_match_date(matches_path: str) -> str:
    """Returns the latest *played* match_datetime date (YYYY_MM_DD) in the matches CSV.

    Used to stamp a fit run's samples file, so successive runs on growing
    data can be tracked historically per club. matches.csv can now carry
    scheduled/postponed rows with no result yet (see
    src/ingestion/brazil/build_treated_dataset.py) -- those must be excluded
    so the stamp always reflects the data actually fit, not a future fixture.
    """
    df = pd.read_csv(matches_path, usecols=["match_datetime", "home_goals"])
    played = df[df["home_goals"].notna()]
    return pd.to_datetime(played["match_datetime"]).max().strftime("%Y_%m_%d")


def save_samples(
    mcmc_fit: CmdStanMCMC,
    teams: list[str],
    matches_path: str,
    samples_dir: str = SAMPLES_DIR,
    model: str = DEFAULT_MODEL,
) -> str:
    """Saves posterior team-strength draws for all teams to a dated CSV.

    The competition/country is inferred from matches_path's parent directory
    (mirroring data/processed/<country>/matches.csv), and the file is named
    after the latest match date in the input data plus `model` (e.g.
    "2026_07_08__poisson_home.csv"), so two candidate models fit on the same
    matches.csv never overwrite each other's samples.

    Returns:
        The path the samples were saved to.
    """
    country = os.path.basename(os.path.dirname(matches_path))
    out_dir = os.path.join(samples_dir, country)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{latest_match_date(matches_path)}__{model}.csv")
    samples_long(mcmc_fit, teams, model=model).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--output", default=None, help="optional CSV path to save team strengths")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_REGISTRY))
    args = parser.parse_args()

    mcmc_fit, teams = fit(args.matches, model=args.model)
    summary = summarize_teams(mcmc_fit, teams, model=args.model)
    print(summary.to_string(index=False))

    samples_path = save_samples(mcmc_fit, teams, args.matches, model=args.model)
    print(f"Saved samples to {samples_path}")

    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"Saved team strengths to {args.output}")


if __name__ == "__main__":
    main()
