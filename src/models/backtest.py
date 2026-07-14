"""Reusable walk-forward, out-of-sample backtest harness for any
src.models.registry.MODEL_REGISTRY entry -- fits on a fixed cadence of
checkpoints and scores every match strictly between two consecutive
checkpoints against the EARLIER checkpoint's posterior, so no match is ever
scored using a fit that already saw its own result.

Deliberately model-agnostic: scoring goes through adapter.sample_scores_single
on the fit's own real posterior draws (Monte Carlo over the actual posterior,
not just its mean), so this file never hardcodes attack/defense/rho like
adhoc/model_backtest.py does -- it works for poisson_home, poisson_strength,
or any future adapter unchanged.

_checkpoint_dates below is a deliberately simplified sibling of
src.simulation.run_rounds.reference_dates: same "skip a candidate with no new
information since the last included one" idea, but with none of that
function's CompetitionConfig/guaranteed_slots coupling -- this harness scores
individual matches, not full-season standings, so it doesn't need
configs/serie_*_<year>.yaml to exist, and it operates over every competition
jointly (mirroring src.models.data.build_stan_data's own "fit on everything,
Serie A and B together" convention), not one competition+season at a time.

Each checkpoint's scored records are also cached to disk (see
_checkpoint_cache_path/_load_cached_checkpoint/_save_checkpoint_cache), one
CSV per (model, reference_date) under BACKTEST_CACHE_DIR -- a full run spans
dozens of Stan fits over hours (see this module's own progress logging), so a
second run (e.g. after adding a new checkpoint's worth of matches, or just
re-running after an interrupted attempt) skips every checkpoint whose cache
file already exists instead of re-fitting it, mirroring
src.simulation.run_rounds's own existence-based resumability
(_already_computed). Pass force=True (or --force) to ignore the cache and
recompute everything -- e.g. after changing window_weeks, chains, or any
other argument that would change what a checkpoint's fit actually is; the
cache has no way to detect that on its own, same tradeoff run_rounds already
accepts.

Usage: python -m src.models.backtest --model poisson_strength
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.constants import (
    BACKTEST_CACHE_DIR,
    DEFAULT_CHAINS,
    DEFAULT_ITER_WARMUP,
    DEFAULT_MATCHES_PATH,
    DEFAULT_SEED,
)
from src.models.data import build_stan_data
from src.models.fit import fit_stan_data
from src.models.registry import DEFAULT_MODEL, MODEL_REGISTRY

OUTCOMES = ("home", "draw", "away")
_RECORD_COLUMNS = [
    "home_team",
    "away_team",
    "match_datetime",
    "competition",
    "season",
    "actual_outcome",
    "home",
    "draw",
    "away",
    "reference_date",
]


def _checkpoint_dates(df: pd.DataFrame, start_season: int, cadence_days: int) -> list[pd.Timestamp]:
    """Every cadence_days-spaced calendar day from (the first start_season
    match's day, minus one cadence step) through today, keeping only a
    candidate at which at least one new match (across every competition/
    season jointly) was played since the previous included checkpoint. A
    checkpoint with no new match would refit on identical data and score no
    new matches, so skipping it is pure waste, not a behavior change.
    """
    played = df[df["home_goals"].notna() & (df["season"] >= start_season)]
    match_days = sorted(pd.to_datetime(played["match_datetime"]).dt.normalize().unique())
    if not match_days:
        return []
    match_days = [pd.Timestamp(day) for day in match_days]
    first_day = match_days[0]
    today = pd.Timestamp.now().normalize()

    pre_window = first_day - pd.Timedelta(days=cadence_days)
    candidates = pd.date_range(start=first_day, end=today, freq=f"{cadence_days}D")

    included = [pre_window]
    last_included = pre_window
    for candidate in candidates:
        if any(last_included < day <= candidate for day in match_days):
            included.append(candidate)
            last_included = candidate
    return included


def _actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def _predicted_outcome(record: dict) -> str:
    return max(OUTCOMES, key=lambda outcome: record[outcome])


def _checkpoint_cache_path(cache_dir: str, model: str, reference_date: pd.Timestamp) -> str:
    return os.path.join(cache_dir, model, f"{reference_date.strftime('%Y_%m_%d')}.csv")


def _load_cached_checkpoint(path: str) -> list[dict]:
    df = pd.read_csv(path)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    return df.to_dict("records")


def _save_checkpoint_cache(path: str, records: list[dict]) -> None:
    """Writes `records` (possibly empty -- a checkpoint whose window had no
    match to score is still a completed checkpoint) with an explicit column
    order, so an empty checkpoint round-trips through _load_cached_checkpoint
    just like a non-empty one, instead of producing a columnless CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(records, columns=_RECORD_COLUMNS).to_csv(path, index=False)


def _log_season_done(model: str, season: int, season_start: float, run_start: float) -> float:
    """Prints season-completion progress: time spent on that season and total
    elapsed time for this model's run so far. A full walk-forward backtest
    takes hours across dozens of Stan fits, so this is the only signal a user
    watching a background run gets between kicking it off and the final
    report. Returns the current time.monotonic() reading, to reset the
    caller's season_start.
    """
    now = time.monotonic()
    print(
        f"[{model}] season {season} done in {now - season_start:.1f}s (total elapsed {now - run_start:.1f}s)",
        flush=True,
    )
    return now


def _score_match(adapter, team_params, shared_params, team_index, match_row, rng) -> dict | None:
    """Monte Carlo-scores one played match against one fit's real posterior
    draws, via adapter.sample_scores_single -- genuinely model-agnostic (no
    attack/defense/rho referenced here), unlike adhoc/model_backtest.py's
    closed-form scoreline_grid. Returns None if either team has no posterior
    in this fit (a debut/long-absent team mid-window) -- unlike
    src.simulation.run_rounds, this harness has no relegated-team
    substitution; such matches are simply left unscored.
    """
    home_team, away_team = match_row["home_team"], match_row["away_team"]
    if home_team not in team_index or away_team not in team_index:
        return None

    n_draws = next(iter(team_params.values())).shape[0]
    home_idx = np.full(n_draws, team_index[home_team])
    away_idx = np.full(n_draws, team_index[away_team])
    home_goals, away_goals = adapter.sample_scores_single(
        team_params, shared_params, home_idx, away_idx, rng
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "match_datetime": match_row["match_datetime"],
        "competition": match_row["competition"],
        "season": match_row["season"],
        "actual_outcome": _actual_outcome(match_row["home_goals"], match_row["away_goals"]),
        "home": float(np.mean(home_goals > away_goals)),
        "draw": float(np.mean(home_goals == away_goals)),
        "away": float(np.mean(home_goals < away_goals)),
    }


def walk_forward_backtest(
    model: str = DEFAULT_MODEL,
    matches_path: str = DEFAULT_MATCHES_PATH,
    start_season: int = 2022,
    window_weeks: int = 104,
    cadence_days: int = 7,
    n_posterior_draws: int | None = None,
    seed: int = DEFAULT_SEED,
    cache_dir: str = BACKTEST_CACHE_DIR,
    force: bool = False,
    **sample_kwargs,
) -> list[dict]:
    """Fits `model` at every _checkpoint_dates checkpoint and scores every
    match strictly between it and the next checkpoint (or through today, for
    the last one) against that earlier fit's posterior -- see _score_match.

    Args:
        window_weeks: forwarded as build_stan_data's max_weeks_ago (how much
            history each checkpoint's fit trains on).
        n_posterior_draws: subsample this many draws from the fit's full
            posterior for Monte Carlo scoring (for speed); None (default)
            uses every draw the fit produced (chains * iter_sampling), i.e.
            genuine full-posterior Monte Carlo, not just the posterior mean.
        cache_dir: where each checkpoint's scored records are cached (see
            this module's docstring) -- one CSV per (model, reference_date).
        force: ignore any cached checkpoint and recompute it anyway. Needed
            after changing window_weeks/chains/iter_warmup/anything else that
            changes what a checkpoint's fit actually is -- the cache can't
            detect that on its own.
        **sample_kwargs: forwarded to fit_stan_data (chains, iter_warmup,
            iter_sampling, ...).

    Returns:
        A list of per-match score records (see _score_match) -- feed to
        aggregate_metrics / calibration_table.
    """
    df = pd.read_csv(matches_path)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])

    checkpoints = _checkpoint_dates(df, start_season, cadence_days)
    adapter = MODEL_REGISTRY[model]
    rng = np.random.default_rng(seed)

    records: list[dict] = []
    run_start = time.monotonic()
    season_start = run_start
    logged_season: int | None = None
    for i, reference_date in enumerate(checkpoints):
        window_end = (
            checkpoints[i + 1]
            if i + 1 < len(checkpoints)
            else pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        )
        to_score = df[
            (df["match_datetime"] > reference_date)
            & (df["match_datetime"] <= window_end)
            & df["home_goals"].notna()
            & (df["season"] >= start_season)
        ]

        cache_path = _checkpoint_cache_path(cache_dir, model, reference_date)
        if not force and os.path.isfile(cache_path):
            records.extend(_load_cached_checkpoint(cache_path))
        else:
            train_df = df[df["match_datetime"] <= reference_date]
            stan_data, teams = build_stan_data(
                train_df, reference_date=reference_date, max_weeks_ago=window_weeks
            )

            try:
                mcmc_fit = fit_stan_data(stan_data, model=model, **sample_kwargs)
            except Exception as exc:
                print(f"FAILED to fit checkpoint {reference_date.date()}: {exc}")
                continue

            stan_vars = mcmc_fit.stan_variables()
            team_params = {name: stan_vars[name] for name in adapter.team_param_names}
            shared_params = {name: stan_vars[name] for name in adapter.shared_param_names}
            if n_posterior_draws is not None:
                n_total = next(iter(team_params.values())).shape[0]
                keep = rng.choice(
                    n_total, size=n_posterior_draws, replace=n_posterior_draws > n_total
                )
                team_params = {name: values[keep] for name, values in team_params.items()}
                shared_params = {name: values[keep] for name, values in shared_params.items()}

            team_index = {team: t for t, team in enumerate(teams)}
            checkpoint_records = []
            for _, match_row in to_score.iterrows():
                record = _score_match(
                    adapter, team_params, shared_params, team_index, match_row, rng
                )
                if record is not None:
                    record["reference_date"] = reference_date
                    checkpoint_records.append(record)

            _save_checkpoint_cache(cache_path, checkpoint_records)
            records.extend(checkpoint_records)

        newest_season = to_score["season"].max() if not to_score.empty else None
        if newest_season is not None:
            if logged_season is None:
                logged_season = int(newest_season)
            elif newest_season > logged_season:
                season_start = _log_season_done(model, logged_season, season_start, run_start)
                logged_season = int(newest_season)

    if logged_season is not None:
        _log_season_done(model, logged_season, season_start, run_start)

    return records


def _brier(records: list[dict], probs_fn) -> float:
    """Mean multiclass Brier (Brier, 1950) over `records` using
    probs_fn(record) -> (p_home, p_draw, p_away) instead of the record's own
    predicted probabilities -- used both for the model's real Brier
    (probs_fn reads home/draw/away off the record) and for the baselines
    below."""
    total = 0.0
    for record in records:
        probs = dict(zip(OUTCOMES, probs_fn(record)))
        total += sum(
            (probs[outcome] - (1.0 if record["actual_outcome"] == outcome else 0.0)) ** 2
            for outcome in OUTCOMES
        )
    return total / len(records)


def _climatology_probs(records: list[dict]):
    n = len(records)
    rates = {
        outcome: sum(1 for r in records if r["actual_outcome"] == outcome) / n
        for outcome in OUTCOMES
    }
    return lambda record: (rates["home"], rates["draw"], rates["away"])


def aggregate_metrics(records: list[dict]) -> dict:
    """Mirrors adhoc/model_backtest.py's report() breakdown (ALL, by
    competition, by competition+season), plus a uniform (1/3 each) and an
    in-subset climatology Brier baseline -- "in-subset" meaning each
    baseline is computed against that same subset's own actual-outcome
    frequencies, so e.g. Serie B's climatology baseline isn't diluted by
    Serie A's draw rate."""

    def _summarize(subset: list[dict]) -> dict | None:
        if not subset:
            return None
        n = len(subset)
        correct_direction = sum(1 for r in subset if _predicted_outcome(r) == r["actual_outcome"])
        return {
            "n": n,
            "brier": _brier(subset, lambda r: (r["home"], r["draw"], r["away"])),
            "brier_uniform_baseline": _brier(subset, lambda r: (1 / 3, 1 / 3, 1 / 3)),
            "brier_climatology_baseline": _brier(subset, _climatology_probs(subset)),
            "direction_accuracy": correct_direction / n,
        }

    competitions = sorted({r["competition"] for r in records})
    comp_seasons = sorted({(r["competition"], r["season"]) for r in records})
    return {
        "all": _summarize(records),
        "by_competition": {
            c: _summarize([r for r in records if r["competition"] == c]) for c in competitions
        },
        "by_competition_season": {
            f"{c} {s}": _summarize(
                [r for r in records if r["competition"] == c and r["season"] == s]
            )
            for c, s in comp_seasons
        },
    }


def calibration_table(records: list[dict], n_bins: int = 10) -> list[dict]:
    """Reliability-diagram data: pools all 3 (predicted prob, did-it-happen)
    pairs per match, bins by predicted probability, reports mean predicted vs
    observed frequency per bin -- same idea as
    adhoc/model_backtest.py::calibration_table, parameterized by n_bins."""
    edges = [i / n_bins for i in range(n_bins + 1)]
    pairs = [
        (r[outcome], 1.0 if r["actual_outcome"] == outcome else 0.0)
        for r in records
        for outcome in OUTCOMES
    ]

    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = [(p, o) for p, o in pairs if (lo <= p < hi) or (hi == 1.0 and p == 1.0)]
        if not in_bin:
            bins.append(
                {"bin_lo": lo, "bin_hi": hi, "n": 0, "mean_predicted": None, "observed_freq": None}
            )
            continue
        bins.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n": len(in_bin),
                "mean_predicted": sum(p for p, _ in in_bin) / len(in_bin),
                "observed_freq": sum(o for _, o in in_bin) / len(in_bin),
            }
        )
    return bins


def _print_report(metrics: dict, calibration: list[dict]) -> None:
    def _line(label: str, m: dict | None) -> None:
        if m is None:
            return
        print(
            f"{label:<20} n={m['n']:<5} direction {m['direction_accuracy']:.1%}   "
            f"Brier={m['brier']:.4f}  (uniform={m['brier_uniform_baseline']:.4f}, "
            f"climatology={m['brier_climatology_baseline']:.4f})"
        )

    _line("ALL", metrics["all"])
    print()
    for label, m in metrics["by_competition"].items():
        _line(label, m)
    print()
    for label, m in metrics["by_competition_season"].items():
        _line(label, m)

    print("\nCalibration / reliability table:")
    for b in calibration:
        if b["n"] == 0:
            print(f"[{b['bin_lo']:.1f},{b['bin_hi']:.1f})   n=0")
            continue
        print(
            f"[{b['bin_lo']:.1f},{b['bin_hi']:.1f})  n={b['n']:>5}  "
            f"predicted={b['mean_predicted']:.3f}  observed={b['observed_freq']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--matches", default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--window-weeks", type=int, default=104)
    parser.add_argument("--cadence-days", type=int, default=7)
    parser.add_argument("--n-posterior-draws", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    parser.add_argument("--iter-warmup", type=int, default=DEFAULT_ITER_WARMUP)
    parser.add_argument("--iter-sampling", type=int, default=1000)
    parser.add_argument("--cache-dir", default=BACKTEST_CACHE_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute every checkpoint even if already cached under --cache-dir.",
    )
    args = parser.parse_args()

    records = walk_forward_backtest(
        model=args.model,
        matches_path=args.matches,
        start_season=args.start_season,
        window_weeks=args.window_weeks,
        cadence_days=args.cadence_days,
        n_posterior_draws=args.n_posterior_draws,
        seed=args.seed,
        cache_dir=args.cache_dir,
        force=args.force,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
    )
    print(f"Backtested {len(records)} matches.\n")
    _print_report(aggregate_metrics(records), calibration_table(records))


if __name__ == "__main__":
    main()
