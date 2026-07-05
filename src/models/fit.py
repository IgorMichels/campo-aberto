"""Fits poisson_home.stan on a matches CSV and reports posterior team strengths.

Generic over the matches CSV passed in -- pass any competition/country's
data/processed/.../matches.csv (see src/models/data.py for the expected schema).
"""

import argparse
import os

import pandas as pd
from cmdstanpy import CmdStanMCMC, CmdStanModel

from src.constants import DEFAULT_MATCHES_PATH, SAMPLES_DIR
from src.models.data import load_stan_data

STAN_FILE = os.path.join(os.path.dirname(__file__), "poisson_home.stan")


def fit_stan_data(stan_data: dict, **sample_kwargs) -> CmdStanMCMC:
    """Compiles poisson_home.stan and samples it on an already-built stan_data dict."""
    model = CmdStanModel(stan_file=STAN_FILE)
    return model.sample(data=stan_data, **sample_kwargs)


def fit(matches_path: str, weight_reference_date: pd.Timestamp | None = None, **sample_kwargs) -> tuple[CmdStanMCMC, list[str]]:
    """Compiles poisson_home.stan and samples it on the given matches CSV.

    Args:
        matches_path: path to a matches CSV (see load_stan_data).
        weight_reference_date: forwarded to load_stan_data as its reference_date, i.e.
            the date each match's time-decay weight is measured from. Defaults to the
            matches CSV's latest match_datetime.
        **sample_kwargs: forwarded to CmdStanModel.sample (e.g. chains, seed).

    Returns:
        (mcmc_fit, teams), where teams[i - 1] names Stan index i.
    """
    stan_data, teams = load_stan_data(matches_path, reference_date=weight_reference_date)
    mcmc_fit = fit_stan_data(stan_data, **sample_kwargs)
    return mcmc_fit, teams


def summarize_teams(mcmc_fit: CmdStanMCMC, teams: list[str]) -> pd.DataFrame:
    """Builds a table of posterior mean attack/defense strength per team."""
    draws = mcmc_fit.draws_pd()
    rows = [
        {
            "team": team,
            "attack": draws[f"attack[{i}]"].mean(),
            "defense": draws[f"defense[{i}]"].mean(),
        }
        for i, team in enumerate(teams, start=1)
    ]
    return pd.DataFrame(rows).sort_values("attack", ascending=False)


def samples_long(mcmc_fit: CmdStanMCMC, teams: list[str]) -> pd.DataFrame:
    """Builds a long-format DataFrame of posterior attack/defense draws per team."""
    draws = mcmc_fit.draws_pd()
    frames = [
        pd.DataFrame(
            {
                "team": team,
                "draw": range(len(draws)),
                "attack": draws[f"attack[{i}]"],
                "defense": draws[f"defense[{i}]"],
            }
        )
        for i, team in enumerate(teams, start=1)
    ]
    return pd.concat(frames, ignore_index=True)


def reference_date(matches_path: str) -> str:
    """Returns the latest match_datetime date (YYYY_MM_DD) in the matches CSV.

    Used to stamp a fit run's samples file, so successive runs on growing
    data can be tracked historically per club.
    """
    df = pd.read_csv(matches_path, usecols=["match_datetime"])
    return pd.to_datetime(df["match_datetime"]).max().strftime("%Y_%m_%d")


def save_samples(
    mcmc_fit: CmdStanMCMC,
    teams: list[str],
    matches_path: str,
    samples_dir: str = SAMPLES_DIR,
) -> str:
    """Saves posterior attack/defense draws for all teams to a dated CSV.

    The competition/country is inferred from matches_path's parent directory
    (mirroring data/processed/<country>/matches.csv), and the file is named
    after the latest match date in the input data.

    Returns:
        The path the samples were saved to.
    """
    country = os.path.basename(os.path.dirname(matches_path))
    out_dir = os.path.join(samples_dir, country)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{reference_date(matches_path)}.csv")
    samples_long(mcmc_fit, teams).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--output", default=None, help="optional CSV path to save team strengths")
    args = parser.parse_args()

    mcmc_fit, teams = fit(args.matches)
    summary = summarize_teams(mcmc_fit, teams)
    print(summary.to_string(index=False))

    samples_path = save_samples(mcmc_fit, teams, args.matches)
    print(f"Saved samples to {samples_path}")

    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"Saved team strengths to {args.output}")


if __name__ == "__main__":
    main()
