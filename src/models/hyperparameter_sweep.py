"""Sequential (coordinate-wise) hyperparameter sweep for a single
src.models.registry.MODEL_REGISTRY model, over the data-weighting and
Stan-prior knobs that can plausibly change predictive quality --
half_life_weeks, window_weeks, and each model's own prior width(s) (see
src/models/backtest.py::walk_forward_backtest and src/models/data.py::
build_stan_data). Design doc: plans/hyperparameter_quality_sweep.md.

poisson_home and hierarchical_home were the first two run (2026-07-17/19,
see plans/hyperparameter_quality_sweep.md); negbin_home and
negbin_home_shared_phi are wired up too (2026-07-19, terrain prepared ahead
of actually running them) -- bivariate_poisson_home/poisson_strength/
poisson_home_no_rho remain deferred, same infrastructure applies whenever
they're picked back up.

A full factorial grid over every hyperparameter is tens of thousands of Stan
fits -- infeasible. Instead:
  1. evaluate() reuses walk_forward_backtest's own content-addressed
     checkpoint cache (src.models.backtest._param_hash /
     _checkpoint_cache_path), so re-evaluating an already-seen combination of
     (model, window_weeks, half_life_weeks, priors, chains, iter_warmup,
     iter_sampling, seed) costs one aggregate_metrics call, zero new Stan
     fits.
  2. coordinate_sweep() sweeps each hyperparameter independently (holding
     every other one at its default), rather than a full cross product --
     turning ~60-80 runs per model into ~13-14. This assumes the
     hyperparameters don't interact strongly; see the design doc for the
     caveat.
  3. neighborhood_check() is the (automated) fallback for that caveat: one
     local-search pass around coordinate_sweep's "best" combo, evaluating
     each parameter's immediate grid neighbor(s) with every OTHER parameter
     held at ITS best value (not at defaults, unlike coordinate_sweep's own
     sweep) -- this is what can actually see interaction between
     parameters. A single pass, not iterative descent to convergence.
  4. The search itself runs at a cheaper cadence (start_season=2024,
     cadence_days=14) than the final confirmation run of the winning combo
     (start_season=2022, cadence_days=7) -- see main().

Usage:
    python -m src.models.hyperparameter_sweep --model poisson_home
    python -m src.models.hyperparameter_sweep --model hierarchical_home
"""

import argparse
import glob
import os
import time

import pandas as pd

from src.constants import BACKTEST_CACHE_DIR, DEFAULT_HALF_LIFE_WEEKS, DEFAULT_MATCHES_PATH
from src.models.backtest import (
    _checkpoint_dates,
    _param_hash,
    aggregate_metrics,
    walk_forward_backtest,
)
from src.models.registry import MODEL_REGISTRY

SWEEP_RESULTS_PATH = os.path.join(BACKTEST_CACHE_DIR, "sweep_results.csv")

_SWEEP_RESULT_COLUMNS = [
    "hash",
    "model",
    "half_life_weeks",
    "window_weeks",
    "rho_prior_sd",
    "group_prior_mean",
    "group_prior_sd",
    "phi_prior",
    "n",
    "brier",
    "brier_uniform_baseline",
    "brier_climatology_baseline",
    "direction_accuracy",
]

# Today's production values (i.e. what was hardcoded in the .stan
# files/build_stan_data before priors moved to Stan `data`, see
# plans/hyperparameter_quality_sweep.md Step 0b) -- must stay in sync with
# src/models/data.py::build_stan_data's and
# src/models/backtest.py::walk_forward_backtest's own defaults. Used to fill
# in every sweep-relevant hyperparameter evaluate() doesn't receive (e.g. a
# poisson_home params dict has no group_prior_mean/group_prior_sd -- those
# still need a concrete value to pass to walk_forward_backtest and to record
# in sweep_results.csv).
_HYPERPARAM_DEFAULTS = {
    "half_life_weeks": DEFAULT_HALF_LIFE_WEEKS,
    "window_weeks": 104,
    "rho_prior_sd": 0.1,
    "group_prior_mean": (0.3, 0.1, -0.1, -0.3),
    "group_prior_sd": 1.0,
    "phi_prior": (2.0, 0.1),
}

# This round's param grids (plans/hyperparameter_quality_sweep.md Step 2).
# Defaults are today's production values (_HYPERPARAM_DEFAULTS above).
PARAM_GRIDS = {
    "poisson_home": {
        "half_life_weeks": [12, 18, 25, 35, 52],
        "window_weeks": [52, 78, 104, 130, 156, 182],
        "rho_prior_sd": [0.05, 0.1, 0.2],
    },
    "hierarchical_home": {
        "half_life_weeks": [12, 18, 25, 35, 52],
        "window_weeks": [52, 78, 104, 130, 156, 182],
        "group_prior_mean": [(0.3, 0.1, -0.1, -0.3), (0.15, 0.05, -0.05, -0.15)],
        "group_prior_sd": [1.0, 0.5],
    },
    "negbin_home": {
        "half_life_weeks": [12, 18, 25, 35, 52],
        "window_weeks": [52, 78, 104, 130, 156, 182],
        # (shape, rate) pairs -- default gamma(2, 0.1) (mean 20, close to
        # Poisson); variants gamma(2, 0.5) (mean 4, more overdispersion) and
        # gamma(4, 0.05) (mean 80, even closer to Poisson) -- swept as a pair
        # per B2 of the original plan draft, not shape/rate independently.
        "phi_prior": [(2.0, 0.1), (2.0, 0.5), (4.0, 0.05)],
    },
    "negbin_home_shared_phi": {
        "half_life_weeks": [12, 18, 25, 35, 52],
        "window_weeks": [52, 78, 104, 130, 156, 182],
        "phi_prior": [(2.0, 0.1), (2.0, 0.5), (4.0, 0.05)],
    },
}
DEFAULTS = {
    "poisson_home": {
        "half_life_weeks": _HYPERPARAM_DEFAULTS["half_life_weeks"],
        "window_weeks": _HYPERPARAM_DEFAULTS["window_weeks"],
        "rho_prior_sd": _HYPERPARAM_DEFAULTS["rho_prior_sd"],
    },
    "hierarchical_home": {
        "half_life_weeks": _HYPERPARAM_DEFAULTS["half_life_weeks"],
        "window_weeks": _HYPERPARAM_DEFAULTS["window_weeks"],
        "group_prior_mean": _HYPERPARAM_DEFAULTS["group_prior_mean"],
        "group_prior_sd": _HYPERPARAM_DEFAULTS["group_prior_sd"],
    },
    "negbin_home": {
        "half_life_weeks": _HYPERPARAM_DEFAULTS["half_life_weeks"],
        "window_weeks": _HYPERPARAM_DEFAULTS["window_weeks"],
        "phi_prior": _HYPERPARAM_DEFAULTS["phi_prior"],
    },
    "negbin_home_shared_phi": {
        "half_life_weeks": _HYPERPARAM_DEFAULTS["half_life_weeks"],
        "window_weeks": _HYPERPARAM_DEFAULTS["window_weeks"],
        "phi_prior": _HYPERPARAM_DEFAULTS["phi_prior"],
    },
}

# Step 3: cheaper cadence during the search itself, full weekly-since-2022
# backtest only for the one winning combination.
SEARCH_PHASE_KWARGS = {"start_season": 2024, "cadence_days": 14}
CONFIRMATION_PHASE_KWARGS = {"start_season": 2022, "cadence_days": 7}

# Every registered model's ORIGINAL (2026-07-13, pre-sweep) tournament
# backtest was cached under data/backtest_cache/<model>/*.csv -- the FLAT
# layout used before Step 1's hash-based cache directories existed (see
# src.models.backtest's docstring). Loaded lazily, once, and kept in memory
# for the rest of the process's run -- _report_progress re-slices this same
# in-memory data by season on every callback instead of re-reading disk each
# time.
_ORIGINAL_TOURNAMENT_RECORDS: dict[str, list[dict]] | None = None

# The current production model (src.models.registry.DEFAULT_MODEL) is
# poisson_home's ORIGINAL/untuned entry -- already one of the rows
# _load_original_tournament_records loads, called out by name below so a
# reader doesn't have to know which row is "production" versus just another
# candidate.
PRODUCTION_MODEL = "poisson_home"

# The current leader across BOTH tuned models (plans/hyperparameter_quality_sweep.md,
# 2026-07-19): poisson_home tuned at half_life=52/window=182/rho=0.05, Brier
# 0.6171 pooled 2022-2026 -- beats hierarchical_home tuned (0.6185) and every
# untuned tournament model. Update this (model + hash) if a later sweep ever
# finds something better -- it is NOT auto-detected.
_BEST_MODEL_REFERENCE = {"model": "poisson_home", "hash": "bc2b6127a18a"}
_BEST_MODEL_RECORDS: list[dict] | None = None


def _load_original_tournament_records() -> dict[str, list[dict]]:
    global _ORIGINAL_TOURNAMENT_RECORDS
    if _ORIGINAL_TOURNAMENT_RECORDS is not None:
        return _ORIGINAL_TOURNAMENT_RECORDS

    records_by_model: dict[str, list[dict]] = {}
    for other_model in MODEL_REGISTRY:
        files = glob.glob(os.path.join(BACKTEST_CACHE_DIR, other_model, "*.csv"))
        frames = [df for f in files if not (df := pd.read_csv(f)).empty]
        records_by_model[other_model] = (
            pd.concat(frames, ignore_index=True).to_dict("records") if frames else []
        )
    _ORIGINAL_TOURNAMENT_RECORDS = records_by_model
    return records_by_model


def _load_best_model_records() -> list[dict]:
    """Loads _BEST_MODEL_REFERENCE's own confirmation-phase-only checkpoints
    (start_season=2022, cadence_days=7) -- its hash directory also holds
    sparser search-phase (14-day cadence) checkpoints from its own sweep,
    which _checkpoint_dates lets us filter out so this never double-counts
    a match scored by both cadences."""
    global _BEST_MODEL_RECORDS
    if _BEST_MODEL_RECORDS is not None:
        return _BEST_MODEL_RECORDS

    df = pd.read_csv(DEFAULT_MATCHES_PATH)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])
    confirmation_dates = _checkpoint_dates(df, start_season=2022, cadence_days=7)
    expected_files = {f"{d.strftime('%Y_%m_%d')}.csv" for d in confirmation_dates}

    hash_dir = os.path.join(
        BACKTEST_CACHE_DIR, _BEST_MODEL_REFERENCE["model"], _BEST_MODEL_REFERENCE["hash"]
    )
    files = (
        [os.path.join(hash_dir, f) for f in os.listdir(hash_dir) if f in expected_files]
        if os.path.isdir(hash_dir)
        else []
    )
    frames = [df for f in files if not (df := pd.read_csv(f)).empty]
    _BEST_MODEL_RECORDS = pd.concat(frames, ignore_index=True).to_dict("records") if frames else []
    return _BEST_MODEL_RECORDS


# --- rough, always-approximate ETA tracking (per [[feedback_backtest_monitor_style]]) ---
#
# Reset whenever `model` changes. Only calls that take longer than
# _REAL_CALL_THRESHOLD_S count toward the timing average -- a cache-hit
# evaluate() call finishes in well under a second and would otherwise drag
# the average down to something meaningless. _expected_total_calls is a
# deterministic UPPER BOUND (not an exact count: neighborhood-check ties/
# edges mean fewer real neighbor calls than the 2-per-parameter bound, and
# many grid/neighborhood calls will hit cache and cost near-nothing) --
# treat the printed ETA as a rough anchor, not a promise.
_REAL_CALL_THRESHOLD_S = 5.0
_CONFIRMATION_CALL_WEIGHT = 3.3  # confirmation-phase has ~3.3x a search-phase run's checkpoints
_SWEEP_PROGRESS = {"model": None, "calls_done": 0.0, "real_time_total": 0.0, "real_calls_done": 0}


def _expected_total_calls(model: str) -> float:
    grid = PARAM_GRIDS[model]
    non_default_candidates = sum(len(candidates) - 1 for candidates in grid.values())
    neighborhood_upper_bound = 2 * len(grid)
    search_confirm = 1
    return (
        1  # baseline
        + non_default_candidates
        + neighborhood_upper_bound
        + search_confirm
        + _CONFIRMATION_CALL_WEIGHT
    )


def _record_call_timing(model: str, elapsed: float, is_confirmation_phase: bool) -> None:
    if _SWEEP_PROGRESS["model"] != model:
        _SWEEP_PROGRESS.update(model=model, calls_done=0.0, real_time_total=0.0, real_calls_done=0)
    _SWEEP_PROGRESS["calls_done"] += _CONFIRMATION_CALL_WEIGHT if is_confirmation_phase else 1
    if elapsed > _REAL_CALL_THRESHOLD_S:
        weight = _CONFIRMATION_CALL_WEIGHT if is_confirmation_phase else 1
        _SWEEP_PROGRESS["real_time_total"] += elapsed / weight
        _SWEEP_PROGRESS["real_calls_done"] += 1


def _print_eta(model: str) -> None:
    done = _SWEEP_PROGRESS["real_calls_done"]
    if done == 0:
        print(
            f"[{model}] ETA: not enough real (non-cached) fits yet in this run to estimate.",
            flush=True,
        )
        return
    avg_call_seconds = _SWEEP_PROGRESS["real_time_total"] / done
    remaining_calls = max(_expected_total_calls(model) - _SWEEP_PROGRESS["calls_done"], 0.0)
    eta_seconds = remaining_calls * avg_call_seconds
    print(
        f"[{model}] ETA (rough): ~{eta_seconds / 60:.0f} min remaining "
        f"(~{remaining_calls:.1f} search-phase-equivalent runs left, "
        f"averaging {avg_call_seconds:.0f}s/run over {done} real fit(s) so far this run)",
        flush=True,
    )


def _report_progress(model: str, season: int, records: list[dict]) -> None:
    """walk_forward_backtest's on_season_done callback: prints the
    cumulative Brier/LogScore/RPS over every record scored so far (across
    every season up to and including `season`), compared APPLES-TO-APPLES
    against: the current production model (PRODUCTION_MODEL), the uniform/
    climatology baselines, the current best known combo (_BEST_MODEL_REFERENCE),
    and every registered model's ORIGINAL tournament run -- each restricted
    to the EXACT SAME set of seasons this sweep run has covered so far (not
    their own full pooled number), per the user's 2026-07-17/19 requests.
    Also prints a rough remaining-time estimate (see _print_eta). Silent if
    nothing's been scored yet (e.g. a season with no debut-team-free
    matches)."""
    summary = aggregate_metrics(records)["all"]
    if summary is None:
        return

    included_seasons = {r["season"] for r in records}

    def _metrics_row(name: str, s: dict) -> tuple:
        return (name, s["n"], s["brier"], s["log_score"], s["rps"])

    rows = [_metrics_row(f"{model} (this run)", summary)]
    for other_model, other_records in _load_original_tournament_records().items():
        subset = [r for r in other_records if r["season"] in included_seasons]
        other_summary = aggregate_metrics(subset)["all"]
        if other_summary is not None:
            label = (
                f"{other_model} [PRODUCTION]" if other_model == PRODUCTION_MODEL else other_model
            )
            rows.append(_metrics_row(label, other_summary))

    best_subset = [r for r in _load_best_model_records() if r["season"] in included_seasons]
    best_summary = aggregate_metrics(best_subset)["all"]
    if best_summary is not None:
        best_label = f"{_BEST_MODEL_REFERENCE['model']} (tuned) [CURRENT BEST]"
        rows.append(_metrics_row(best_label, best_summary))

    rows.append(
        (
            "uniform baseline",
            summary["n"],
            summary["brier_uniform_baseline"],
            summary["log_score_uniform_baseline"],
            summary["rps_uniform_baseline"],
        )
    )
    rows.append(
        (
            "climatology baseline",
            summary["n"],
            summary["brier_climatology_baseline"],
            summary["log_score_climatology_baseline"],
            summary["rps_climatology_baseline"],
        )
    )
    rows.sort(key=lambda row: row[2])

    seasons_label = ",".join(str(s) for s in sorted(included_seasons))
    print(
        f"[{model}] cumulative through season {season} (seasons {seasons_label}), "
        "apples-to-apples (same seasons) vs. production/baselines/current best/tournament:",
        flush=True,
    )
    for name, n, brier, log_score, rps in rows:
        marker = "  <== this run" if name == f"{model} (this run)" else ""
        print(
            f"    {name:<34} n={n:<6} Brier={brier:.4f}  LogScore={log_score:.4f}  "
            f"RPS={rps:.4f}{marker}",
            flush=True,
        )
    _print_eta(model)


def _existing_hashes(path: str) -> set[str]:
    if not os.path.isfile(path):
        return set()
    return set(pd.read_csv(path, usecols=["hash"])["hash"].astype(str))


def _append_sweep_row_if_new(path: str, row_hash: str, result: dict) -> None:
    """Appends one row to sweep_results.csv the first time row_hash is seen
    -- mirrors the checkpoint cache's own resumability philosophy (Step 1):
    a sweep interrupted and restarted, or two sweeps that happen to evaluate
    the same combination, never produce duplicate rows for it."""
    if row_hash in _existing_hashes(path):
        return
    row = {
        **result,
        "group_prior_mean": str(tuple(result["group_prior_mean"])),
        "phi_prior": str(tuple(result["phi_prior"])),
    }
    row = {col: row[col] for col in _SWEEP_RESULT_COLUMNS}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.isfile(path)
    pd.DataFrame([row], columns=_SWEEP_RESULT_COLUMNS).to_csv(
        path, mode="a", header=write_header, index=False
    )


def evaluate(
    model: str, params: dict, results_path: str = SWEEP_RESULTS_PATH, **backtest_kwargs
) -> dict:
    """Runs a full walk_forward_backtest for `model` at one hyperparameter
    combination and aggregates it to a single row of metrics.

    Args:
        model: a src.models.registry.MODEL_REGISTRY key (poisson_home,
            hierarchical_home, negbin_home, or negbin_home_shared_phi).
        params: any subset of half_life_weeks/window_weeks/rho_prior_sd/
            group_prior_mean/group_prior_sd/phi_prior -- missing keys fall
            back to _HYPERPARAM_DEFAULTS (today's production values), so a
            poisson_home params dict never needs to mention
            group_prior_mean/group_prior_sd/phi_prior (and so on for every
            other model).
        results_path: sweep_results.csv path -- one row per distinct `params`
            (keyed by _param_hash(params), see _append_sweep_row_if_new).
        **backtest_kwargs: forwarded to walk_forward_backtest (start_season,
            cadence_days, chains, iter_warmup, iter_sampling, seed, ...).

    Returns:
        A dict with the evaluated hyperparameters, "hash", and every
        aggregate_metrics(records)["all"] key (n, brier, log_score, rps, and
        each's uniform/climatology baseline, direction_accuracy).
    """
    effective = {**_HYPERPARAM_DEFAULTS, **params}
    start = time.monotonic()
    records = walk_forward_backtest(
        model=model, on_season_done=_report_progress, **effective, **backtest_kwargs
    )
    is_confirmation_phase = (
        backtest_kwargs.get("start_season") == CONFIRMATION_PHASE_KWARGS["start_season"]
        and backtest_kwargs.get("cadence_days") == CONFIRMATION_PHASE_KWARGS["cadence_days"]
    )
    _record_call_timing(model, time.monotonic() - start, is_confirmation_phase)
    summary = aggregate_metrics(records)["all"]
    if summary is None:
        raise ValueError(
            f"No matches scored for model={model} params={params} "
            f"backtest_kwargs={backtest_kwargs}; nothing to evaluate."
        )

    row_hash = _param_hash(params)
    result = {
        "hash": row_hash,
        "model": model,
        "half_life_weeks": effective["half_life_weeks"],
        "window_weeks": effective["window_weeks"],
        "rho_prior_sd": effective["rho_prior_sd"],
        "group_prior_mean": tuple(effective["group_prior_mean"]),
        "group_prior_sd": effective["group_prior_sd"],
        "phi_prior": tuple(effective["phi_prior"]),
        **summary,
    }
    _append_sweep_row_if_new(results_path, row_hash, result)
    return result


def _log_candidate(model: str, param: str, value, row: dict, baseline: dict) -> None:
    delta = row["brier"] - baseline["brier"]
    print(
        f"[{model}] {param}={value} -> Brier {row['brier']:.4f} "
        f"(baseline {baseline['brier']:.4f}, delta {delta:+.4f})",
        flush=True,
    )


def coordinate_sweep(
    model: str, param_grid: dict[str, list], defaults: dict, **backtest_kwargs
) -> dict:
    """Sweeps each key of param_grid independently (holding every other
    parameter at its `defaults` value), then confirms the combination of
    every parameter's individually-best value in one final run.

    Args:
        model: a src.models.registry.MODEL_REGISTRY key.
        param_grid: {param_name: [candidate values]}, e.g.
            {"half_life_weeks": [12, 18, 25, 35, 52], ...}.
        defaults: today's production value for every key in param_grid --
            the shared baseline point, evaluated exactly once.
        **backtest_kwargs: forwarded to evaluate (typically
            SEARCH_PHASE_KWARGS during the sweep itself; the caller is
            responsible for re-running the returned "best" combo at
            CONFIRMATION_PHASE_KWARGS separately -- see main()).

    Returns:
        {"baseline": evaluate(defaults)'s result,
         "per_param": {param: [evaluate(...) result per candidate]},
         "best": {param: best-Brier candidate value},
         "confirm": evaluate(best)'s result (at the SAME backtest_kwargs as
         the sweep itself -- not yet the final confirmation-phase number)}.
    """
    baseline = evaluate(model, defaults, **backtest_kwargs)
    print(f"[{model}] baseline {defaults} -> Brier {baseline['brier']:.4f}", flush=True)

    per_param: dict[str, list[dict]] = {}
    best = dict(defaults)
    for param, candidates in param_grid.items():
        rows = []
        for value in candidates:
            if value == defaults[param]:
                # Already the baseline point -- reuse it, don't re-run.
                rows.append(baseline)
                continue
            row = evaluate(model, {**defaults, param: value}, **backtest_kwargs)
            _log_candidate(model, param, value, row, baseline)
            rows.append(row)
        per_param[param] = rows
        best_row = min(rows, key=lambda r: r["brier"])
        best[param] = candidates[rows.index(best_row)]
        print(f"[{model}] best {param} so far: {best[param]}", flush=True)

    print(f"[{model}] confirming best combo (at sweep-phase settings): {best}", flush=True)
    confirm = evaluate(model, best, **backtest_kwargs)
    print(f"[{model}] confirm -> Brier {confirm['brier']:.4f}", flush=True)

    return {"baseline": baseline, "per_param": per_param, "best": best, "confirm": confirm}


def _neighbor_values(candidates: list, value) -> list:
    """The immediate neighbor(s) of `value` inside `candidates` (a param
    grid's own already-tested list) -- one on each side, where one exists.
    An edge value (the grid's min or max) has only one neighbor. Only
    values already in the grid are considered "one step"; extrapolating
    beyond the tested range (e.g. trying something past window_weeks=182)
    is a separate decision, not this function's job."""
    idx = candidates.index(value)
    neighbors = []
    if idx > 0:
        neighbors.append(candidates[idx - 1])
    if idx < len(candidates) - 1:
        neighbors.append(candidates[idx + 1])
    return neighbors


def neighborhood_check(
    model: str, param_grid: dict[str, list], best: dict, best_result: dict, **backtest_kwargs
) -> dict:
    """One local-search pass around coordinate_sweep's `best` combo: for
    each parameter, evaluates its immediate grid neighbor(s) (see
    _neighbor_values) with every OTHER parameter held at its `best` value --
    unlike coordinate_sweep's own per-parameter sweep, which holds others at
    `defaults`. This is what can actually catch interaction between
    parameters (coordinate_sweep's independent, defaults-anchored sweep
    can't see it by construction -- see plans/hyperparameter_quality_sweep.md).

    Every trial here varies exactly one parameter away from `best` (a
    "plus-shaped" neighborhood, not a running hill-climb) -- a single pass,
    not iterative descent to convergence: if two different parameters' best
    neighbor would each individually improve on `best`, both are tried
    against the ORIGINAL `best`, not against each other's update.

    Args:
        best: coordinate_sweep's returned "best" combo.
        best_result: coordinate_sweep's returned "confirm" result (i.e.
            evaluate(best)'s own metrics) -- the point every neighbor is
            compared against; also returned unchanged if no neighbor wins.
        **backtest_kwargs: forwarded to evaluate (typically
            SEARCH_PHASE_KWARGS, same as the sweep that produced `best`).

    Returns:
        {"best": the (possibly updated) combo, "confirm": that combo's own
        evaluate() result -- best_result itself if nothing improved on it}.
    """
    current_best = dict(best)
    current_result = best_result

    for param, candidates in param_grid.items():
        for value in _neighbor_values(candidates, best[param]):
            trial = {**best, param: value}
            row = evaluate(model, trial, **backtest_kwargs)
            delta = row["brier"] - best_result["brier"]
            print(
                f"[{model}] neighborhood {param}={value} (others at best) -> "
                f"Brier {row['brier']:.4f} (best-combo {best_result['brier']:.4f}, "
                f"delta {delta:+.4f})",
                flush=True,
            )
            if row["brier"] < current_result["brier"]:
                current_result = row
                current_best = trial

    if current_best != best:
        print(f"[{model}] neighborhood check improved the combo: {current_best}", flush=True)
    else:
        print(f"[{model}] neighborhood check found nothing better than {best}", flush=True)

    return {"best": current_best, "confirm": current_result}


def _print_final_report(model: str, confirm: dict) -> None:
    print(f"\n=== {model}: confirmed result (full backtest, best hyperparameters) ===")
    print(
        "params: "
        f"half_life_weeks={confirm['half_life_weeks']}, "
        f"window_weeks={confirm['window_weeks']}, "
        f"rho_prior_sd={confirm['rho_prior_sd']}, "
        f"group_prior_mean={confirm['group_prior_mean']}, "
        f"group_prior_sd={confirm['group_prior_sd']}"
    )
    print()
    print(
        f"{model:<25} n={confirm['n']:<6} direction {confirm['direction_accuracy']:.1%}   "
        f"Brier={confirm['brier']:.4f}"
    )
    print(f"{'uniform baseline':<25} Brier={confirm['brier_uniform_baseline']:.4f}")
    print(f"{'climatology baseline':<25} Brier={confirm['brier_climatology_baseline']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(PARAM_GRIDS))
    args = parser.parse_args()
    model = args.model

    result = coordinate_sweep(model, PARAM_GRIDS[model], DEFAULTS[model], **SEARCH_PHASE_KWARGS)

    neighborhood = neighborhood_check(
        model, PARAM_GRIDS[model], result["best"], result["confirm"], **SEARCH_PHASE_KWARGS
    )
    final_best = neighborhood["best"]

    print(
        f"\n[{model}] re-running best combo {final_best} at full confirmation settings "
        f"{CONFIRMATION_PHASE_KWARGS}...",
        flush=True,
    )
    confirm = evaluate(model, final_best, **CONFIRMATION_PHASE_KWARGS)
    _print_final_report(model, confirm)


if __name__ == "__main__":
    main()
