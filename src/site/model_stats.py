"""Computes the model-statistics page's metrics (exact-scoreline/direction
accuracy, Brier, calibration) from the site's already-committed played-match
data, and exports them into site/data/model_stats.json for
site/model/stats.html to render -- no client-side aggregation, see
plans/model_stats_page.md's architecture-decision note.

Ports site/assets/js/poisson_home.js's closed-form Dixon-Coles scoreline
math to Python (same formula, same outcome_cap=10 default) and applies it to
each played match's own embedded historical params snapshot -- see
site/data/played_manifest.json + <slug>/played_<season>.json, written by
src.site.export_matches_data. This reproduces the exact same probabilities
matches/played.html's own cards already show (posterior-MEAN attack/defense,
zero Stan cost), not a fresh Monte Carlo re-fit like src.models.backtest's
walk-forward tournament harness -- a genuinely different concern (see that
module's own docstring for why it never hardcodes this math).

Reuses src.models.backtest's OUTCOMES/_brier/_climatology_probs/
calibration_table directly (same record shape: actual_outcome plus
home/draw/away probabilities) rather than duplicating those pure scoring-rule
helpers.

Run as part of `python -m src.pipeline` (after export_matches_data, which
this depends on for played_<season>.json), or standalone:

    python -m src.site.model_stats
"""

import argparse
import json
import math
import os

from src.constants import SITE_DIR
from src.models.backtest import OUTCOMES, _brier, _climatology_probs, calibration_table

OUTCOME_CAP = 10  # matches poisson_home.js's poissonPmfWithTail default
CALIBRATION_BINS = 10


def _poisson_pmf_with_tail(mu: float, max_exact: int) -> list[float]:
    """[P(0), ..., P(max_exact - 1), P(>=max_exact)], summing to exactly 1 --
    same as site/assets/js/poisson_home.js's poissonPmfWithTail."""
    probs = [0.0] * (max_exact + 1)
    p = math.exp(-mu)
    cumulative = 0.0
    for k in range(max_exact):
        probs[k] = p
        cumulative += p
        p = p * mu / (k + 1)
    probs[max_exact] = max(0.0, 1 - cumulative)
    return probs


def _dixon_coles_tau(x: int, y: int, mu_home: float, mu_away: float, rho: float) -> float:
    """Same as poisson_home.js's dixonColesTau -- only the four (x, y) in
    {0, 1}^2 cells are reweighted; every other cell keeps its
    independent-Poisson mass."""
    if x == 0 and y == 0:
        return max(0.0, 1 - mu_home * mu_away * rho)
    if x == 0 and y == 1:
        return max(0.0, 1 + mu_home * rho)
    if x == 1 and y == 0:
        return max(0.0, 1 + mu_away * rho)
    if x == 1 and y == 1:
        return max(0.0, 1 - rho)
    return 1.0


def _match_rates(
    attack_home: float,
    defense_home: float,
    attack_away: float,
    defense_away: float,
    eta: float,
    beta_home: float,
) -> tuple[float, float]:
    """Same as poisson_home.js's matchRates."""
    mu_home = math.exp(attack_home - defense_away + eta + beta_home)
    mu_away = math.exp(attack_away - defense_home + eta)
    return mu_home, mu_away


def compute_scoreline_grid(
    shared: dict, home_params: dict, away_params: dict, outcome_cap: int = OUTCOME_CAP
) -> dict:
    """Full (outcome_cap+1)x(outcome_cap+1) Dixon-Coles-corrected Poisson grid
    for one match, renormalized -- same math as poisson_home.js's
    scorelineProbabilities, just returning only what this page needs (no
    display-truncated 5x5 grid, this is an aggregate stats export, not a
    rendered heatmap).

    Returns {home_win, draw, away_win, best: {home, away, prob}}. `best` is
    the argmax over every REAL exact scoreline (x, y in 0..outcome_cap-1) --
    excludes the outcome_cap row/column, which is a ">=outcome_cap" tail
    bucket, not a single discrete score (in practice this exclusion never
    changes the argmax for a realistic mu, since that tail's probability is
    negligible for football scorelines, but it's the mathematically correct
    exclusion regardless).
    """
    mu_home, mu_away = _match_rates(
        home_params["attack"],
        home_params["defense"],
        away_params["attack"],
        away_params["defense"],
        shared["eta"],
        shared["beta_home"],
    )
    rho = shared["rho"]
    p_home = _poisson_pmf_with_tail(mu_home, outcome_cap)
    p_away = _poisson_pmf_with_tail(mu_away, outcome_cap)

    fine = [
        [
            _dixon_coles_tau(x, y, mu_home, mu_away, rho) * p_home[x] * p_away[y]
            for y in range(outcome_cap + 1)
        ]
        for x in range(outcome_cap + 1)
    ]
    total = sum(sum(row) for row in fine)

    home_win = draw = away_win = 0.0
    for x in range(outcome_cap + 1):
        for y in range(outcome_cap + 1):
            p = fine[x][y] / total
            if x > y:
                home_win += p
            elif x == y:
                draw += p
            else:
                away_win += p

    best_home = best_away = 0
    best_prob = -1.0
    for x in range(outcome_cap):
        for y in range(outcome_cap):
            p = fine[x][y] / total
            if p > best_prob:
                best_prob, best_home, best_away = p, x, y

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "best": {"home": best_home, "away": best_away, "prob": best_prob},
    }


def load_played_records(site_dir: str = SITE_DIR) -> list[dict]:
    """Every already-played match across every competition/season on the
    site that has a model snapshot, scored via compute_scoreline_grid --
    returns a flat list of {competition, season, home, draw, away,
    actual_outcome, exact_correct}, the exact record shape
    src.models.backtest's aggregate helpers (OUTCOMES/_brier/
    _climatology_probs/calibration_table) already expect (home/draw/away
    probabilities + actual_outcome), plus this module's own exact_correct
    field.
    """
    data_dir = os.path.join(site_dir, "data")
    with open(os.path.join(data_dir, "played_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    records = []
    for entry in manifest["competitions"]:
        slug = entry["slug"]
        for season in entry["seasons"]:
            path = os.path.join(data_dir, slug, f"played_{season}.json")
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            for match in payload["matches"]:
                if not match["has_model"]:
                    continue
                params = match["params"]
                if params["model"] != "poisson_home":
                    raise NotImplementedError(
                        f"compute_scoreline_grid only implements poisson_home's "
                        f"closed-form math (mirroring site/assets/js/poisson_home.js, "
                        f"the only model score_models.js registers today) -- got "
                        f"{params['model']!r} for {match['home_team']} vs "
                        f"{match['away_team']} ({entry['competition']} {season})"
                    )
                teams = params["teams"]
                home_team, away_team = match["home_team"], match["away_team"]
                if home_team not in teams or away_team not in teams:
                    continue

                grid = compute_scoreline_grid(params["shared"], teams[home_team], teams[away_team])
                home_goals, away_goals = match["home_goals"], match["away_goals"]
                actual_outcome = (
                    "home"
                    if home_goals > away_goals
                    else "away"
                    if home_goals < away_goals
                    else "draw"
                )
                records.append(
                    {
                        "competition": entry["competition"],
                        "season": season,
                        "home": grid["home_win"],
                        "draw": grid["draw"],
                        "away": grid["away_win"],
                        "actual_outcome": actual_outcome,
                        "exact_correct": (
                            grid["best"]["home"] == home_goals
                            and grid["best"]["away"] == away_goals
                        ),
                    }
                )
    return records


def _predicted_outcome(record: dict) -> str:
    probs = {outcome: record[outcome] for outcome in OUTCOMES}
    return max(probs, key=probs.get)


def aggregate_metrics(records: list[dict]) -> dict:
    """Groups by "all", each competition, each competition+season -- same
    breakdown shape as src.models.backtest.aggregate_metrics, plus this
    page's own exact_pct (not tracked there, since that harness scores via
    Monte Carlo home/draw/away sampling only, never a best-exact-score)."""

    def _summarize(subset: list[dict]) -> dict | None:
        if not subset:
            return None
        n = len(subset)
        model_probs = lambda r: (r["home"], r["draw"], r["away"])  # noqa: E731
        uniform_probs = lambda r: (1 / 3, 1 / 3, 1 / 3)  # noqa: E731
        climatology_probs = _climatology_probs(subset)
        correct_direction = sum(1 for r in subset if _predicted_outcome(r) == r["actual_outcome"])
        exact_correct = sum(1 for r in subset if r["exact_correct"])
        return {
            "n": n,
            "exact_pct": exact_correct / n,
            "direction_pct": correct_direction / n,
            "brier": _brier(subset, model_probs),
            "baseline_uniform_brier": _brier(subset, uniform_probs),
            "baseline_climatology_brier": _brier(subset, climatology_probs),
            "baseline_favorite_direction_pct": (
                sum(1 for r in subset if r["actual_outcome"] == "home") / n
            ),
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


def export_model_stats(site_dir: str = SITE_DIR) -> None:
    records = load_played_records(site_dir)
    payload = {
        "breakdown": aggregate_metrics(records),
        "calibration": calibration_table(records, n_bins=CALIBRATION_BINS),
    }
    output_path = os.path.join(site_dir, "data", "model_stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--site-dir", default=SITE_DIR)
    args = parser.parse_args()
    export_model_stats(args.site_dir)


if __name__ == "__main__":
    main()
