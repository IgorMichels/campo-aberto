"""CLI: backtests a season on a fixed twice-weekly cadence instead of a single
--reference-date.

Reference dates land on a fixed Monday/Friday schedule rather than being tied
to round boundaries, mirroring a real "run twice a week" cron. Every Monday and
Friday from a competition+season's first played match through today is a
*candidate*; a candidate only actually becomes a reference_date (triggering a
Stan refit and a saved snapshot) if something genuinely new happened since the
previous included reference_date -- either a newly played match, or a
config-level guaranteed_slots berth crossing its `known_from` date. Candidates
with no new information are skipped silently, so an inactive stretch of the
calendar (international breaks, off-season) produces no no-op snapshots that
would waste a Stan fit and clutter the site's evolution chart with flat points.
PLUS one unconditional pre-season checkpoint per competition+season (see
reference_dates), so round 1 itself gets a real prediction too, not just the
placeholder src.site.export_matches_data._played_cards renders for a match
with no prior snapshot at all.

A team with zero matches inside the training window as of that pre-season
checkpoint (a newly-promoted or long-absent club) has no Stan posterior of
its own -- rather than skip the whole competition's round for that reason,
it's given a stand-in: the attack/defense of a team relegated the previous
season, paired up by plain alphabetical order (see
_relegated_teams_previous_season / _debut_team_aliases). Logged as status
"substituted" (see DEFAULT_LOG_PATH below) whenever this happens. A
competition+season still falls back to skipping if there aren't enough
previous-season relegated teams to cover every debut/stale team.

Loops over every configs/*.yaml whose filename's season suffix (see
configs/README.md's "Per-season configs" section) is in --seasons, grouping
by competition. Two competitions whose checkpoints land on the same
Monday/Friday (common in Brazil, where Serie A and Serie B are usually
scheduled the same weekends) reuse a single Stan fit instead of refitting
twice for the same date.

Clears the terminal at the start of every reference_date iteration (when
stdout is a real terminal -- a no-op otherwise, e.g. redirected to a file),
then prints "Executing run X/Z" (or "Skipping run X/Z" when resumed, see
below) right after the clear -- see _clear_screen. X/Z counts every
reference date across every requested season combined (this is the outer
loop's own unit of work -- a fit shared by two competitions on the same
checkpoint still counts once). A Stan fit failing, or one competition's
simulate/save step failing, is logged and skipped rather than crashing the
whole backtest -- see DEFAULT_LOG_PATH / --log: every skipped or failed
round (reference_date, competition, season, status, reason) is written as
a fresh CSV once the run finishes, so a multi-hour unattended backtest
doesn't lose everything to one bad date.

**Resumable**: before fitting a reference_date, checks whether
save_results already wrote every (competition, season) sharing that date's
output file (data/results/<slug>/<season>/<date>.csv, see
src.simulation.results.save_results); if every one of them already exists,
the whole date is skipped without ever calling the (costly) Stan fit. If
only some already exist (e.g. Serie A's checkpoint was saved but Serie B's
wasn't), the fit still runs -- it's shared -- but only the missing ones are
re-simulated/saved. Re-running the exact same command after an interrupted
run therefore resumes roughly where it left off instead of redoing
everything. Pass --force to ignore existing files and recompute every date
regardless (e.g. after a model/code change invalidates old results).

    python -m src.simulation.run_rounds --seasons 2025 2026
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (
    DEFAULT_CHAINS,
    DEFAULT_ITER_WARMUP,
    DEFAULT_MATCHES_PATH,
    DEFAULT_N_DRAWS,
    DEFAULT_SEED,
    RESULTS_DIR,
)
from src.models.data import build_stan_data
from src.models.fit import fit_stan_data
from src.models.registry import DEFAULT_MODEL, MODEL_REGISTRY
from src.simulation import fixtures, standings
from src.simulation.config import CompetitionConfig, RoundRobinPhaseConfig, load_competition_config
from src.simulation.results import save_results
from src.simulation.run import _parse_guaranteed_slots
from src.simulation.simulate import simulate_competition

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_SEASON_SUFFIX = re.compile(r"_(\d{4})\.yaml$")
DEFAULT_LOG_PATH = "data/results/run_rounds_log.csv"  # skipped/failed rounds, see main()


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


def reference_dates(
    df: pd.DataFrame, competition: str, season: int, config: CompetitionConfig
) -> list[pd.Timestamp]:
    """Backtest reference dates on a fixed Monday/Friday cadence, gated on new
    information -- one snapshot per twice-weekly checkpoint at which something
    actually changed for this competition+season, PLUS one unconditional
    pre-season checkpoint (see below) so round 1 gets a real prediction too.

    The candidate set is every Monday and Friday (weekday 0 and 4) from the
    first played match's calendar day through today. Walking them in order, a
    candidate is *included* (and becomes the new baseline) only if, since the
    previous included reference_date, either:
      - a match was newly played (its calendar day falls in that window), or
      - a config `guaranteed_slots` berth crossed its `known_from` date.
    Candidates with neither are skipped silently -- refitting Stan and saving a
    snapshot for a checkpoint at which nothing changed would only waste a fit
    and add a flat, redundant point to the site's evolution chart.

    Reference dates are the plain Monday/Friday midnight, with *no* +1-day shift
    (unlike the old per-round rule). Filters elsewhere are
    `match_datetime <= reference_date`, so a reference_date of "this Monday
    00:00" means "as of the start of Monday, i.e. every match through last
    Sunday". A match kicking off later ON that Monday itself (rare in the
    Brazilian calendar) is simply picked up at the next Friday checkpoint --
    exactly the semantics of a cron that runs early Monday morning, before that
    evening's fixtures. Friday likewise captures everything through Thursday.

    Only genuinely played matches count (`home_goals.notna()`, the same guard
    the per-round rule used): scheduled/postponed rows with no result yet (see
    src/ingestion/brazil/build_treated_dataset.py) must never spuriously create
    a checkpoint for a fixture window whose results aren't known.

    Pre-season exception: under the "new information since last included" rule
    alone, round 1 could NEVER get a real prediction -- the very first
    candidate is already >= first_day, so its own fit already trains on round
    1's results (reference_date's filter is `match_datetime <= reference_date`).
    There is therefore no candidate that is genuinely "before round 1" unless
    one is added deliberately. So the last Monday/Friday strictly BEFORE
    first_day is always prepended, unconditionally (it has no "since last
    included" baseline to compare against -- it's the season's own starting
    point). Its Stan fit trains on whatever history already exists (previous
    seasons, other competitions sharing the joint fit) and produces the
    site's "sem modelo disponível" placeholder's replacement: a real snapshot
    round 1 matches can reference via
    src.site.export_matches_data._played_cards /
    src.site.export_site_data._snapshot_csv_before. If that history turns out
    to be empty or too thin to fit (e.g. a brand-new competition with no prior
    seasons at all), src.simulation.run_rounds.main already catches and logs a
    failed Stan fit rather than crashing, so this is safe to always attempt.
    """
    season_df = df[(df["competition"] == competition) & (df["season"] == season)]
    played_season_df = season_df[season_df["home_goals"].notna()]
    match_days = sorted(played_season_df["match_datetime"].dt.normalize().unique())
    if not match_days:
        return []

    first_day = pd.Timestamp(match_days[0])
    today = pd.Timestamp.now().normalize()
    candidates = [
        day
        for day in pd.date_range(start=first_day, end=today, freq="D")
        if day.weekday() in (0, 4)  # Monday, Friday
    ]

    match_days = [pd.Timestamp(day) for day in match_days]
    known_froms = sorted(slot.known_from.normalize() for slot in config.guaranteed_slots)

    pre_season = first_day - pd.Timedelta(days=1)
    while pre_season.weekday() not in (0, 4):
        pre_season -= pd.Timedelta(days=1)

    included: list[pd.Timestamp] = [pre_season]
    last_included = pre_season
    for candidate in candidates:
        new_match = any(last_included < day <= candidate for day in match_days)
        new_slot = any(last_included < known_from <= candidate for known_from in known_froms)
        if new_match or new_slot:
            included.append(candidate)
            last_included = candidate
    return included


def _relegated_teams_previous_season(
    df: pd.DataFrame,
    competition: str,
    season: int,
    config: CompetitionConfig,
    rng: np.random.Generator,
) -> list[str]:
    """The teams that finished in this competition's `rebaixamento` (relegation)
    table-position range at the end of `season - 1` -- candidate substitutes
    for a team `simulate_competition` has no posterior for at all (see
    reference_dates' pre-season checkpoint: a newly-promoted or long-absent
    team has zero matches inside the training window, so its own attack/
    defense can't be estimated). Uses `config` (the CURRENT season's config,
    the only one this repo keeps on disk -- there's no configs/*_<season-1>.yaml
    to load) for both the `rebaixamento` spot's position range and the
    tiebreak rule, on the assumption a competition's relegation-slot count and
    tiebreak method don't change from one season to the next (true for both
    Serie A and Serie B across every season currently tracked).

    Returns [] (no substitution possible, caller falls back to skipping) when
    there's no `rebaixamento` spot declared, or no played matches at all for
    the previous season (e.g. this competition's very first tracked season).
    """
    league_phase = next(
        (p for p in config.phases if isinstance(p, RoundRobinPhaseConfig) and p.id == "league"),
        None,
    )
    if league_phase is None:
        return []
    relegation_spot = next((s for s in league_phase.spots if s.name == "rebaixamento"), None)
    if relegation_spot is None or relegation_spot.positions is None:
        return []

    previous_season = season - 1
    teams = fixtures.season_teams(df, competition, previous_season)
    if not teams:
        return []
    played_results, _, _ = fixtures.split_fixtures(
        df, competition, previous_season, pd.Timestamp.now(), teams=teams
    )
    if not played_results:
        return []

    final_order = standings.rank_table(teams, played_results, rng, league_phase.head_to_head_mode)
    start, end = relegation_spot.positions  # 1-indexed, inclusive
    return final_order[start - 1 : end]


def _debut_team_aliases(missing: set[str], relegated: list[str]) -> dict[str, str]:
    """Pairs each debut/stale-data team with a previous-season relegated team
    by plain alphabetical order on both sides (missing[0] <-> relegated[0],
    missing[1] <-> relegated[1], ...) -- not an attempt to match a specific
    promoted team to the specific relegated team it "replaced" (Brazilian
    promotion/relegation has no such real 1:1 relationship, clubs are just
    swapped as two same-sized sets), just a simple, deterministic, reproducible
    convention. If there are more missing teams than relegated candidates, the
    extras are left unmapped (caller still skips them) rather than reusing a
    substitute for two different aliases."""
    return dict(zip(sorted(missing), sorted(relegated)))


def _already_computed(
    competition: str, season: int, reference_date: pd.Timestamp, results_dir: str
) -> bool:
    """True if src.simulation.results.save_results already wrote this exact
    (competition, season, reference_date)'s output file -- mirrors that
    function's own slug/path formula exactly, so a resumed run recognizes
    precisely the files a previous run (interrupted or not) already
    produced, without needing its own separate manifest of completed work."""
    slug = competition.lower().replace(" ", "_")
    out_path = Path(results_dir) / slug / str(season) / f"{reference_date.strftime('%Y_%m_%d')}.csv"
    return out_path.exists()


def _clear_screen() -> None:
    """Clears the terminal right before each reference_date's progress line,
    so a long unattended backtest reads as one line of "where are we now"
    instead of scrolling behind hundreds of lines of Stan sampler output per
    fit. No-op when stdout isn't a real terminal (e.g. redirected to a file
    or running in CI) -- clearing would just inject useless escape codes
    into a log, and DEFAULT_LOG_PATH/--log already covers "what happened"
    for that case."""
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")  # noqa: S605 -- fixed command, no user input


_LOG_COLUMNS = ["reference_date", "competition", "season", "status", "reason"]


def _write_log(log_rows: list[dict], log_path: str) -> None:
    """Writes every non-plain-saved round (status "skipped", "failed", or
    "substituted" -- see main()) to `log_path` as a fresh CSV -- always
    rewritten from scratch (never appended), matching this repo's existing
    unmapped_team_names_log.csv/score_discrepancies.csv convention for a
    regenerated-every-run diagnostic log. Explicit `columns=` so a completely
    clean run (zero skips/failures/substitutions) still writes a valid
    header-only CSV instead of an unparseable empty file
    (`pd.DataFrame([])` has no columns)."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows, columns=_LOG_COLUMNS).to_csv(log_path, index=False)


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
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_REGISTRY))
    parser.add_argument(
        "--guaranteed-slot",
        action="append",
        default=[],
        metavar="TEAM:SPOT",
        help="a team with an externally guaranteed spot, applied at every reference date regardless of date "
        "(use a config's guaranteed_slots list instead for date-gated guarantees); repeatable",
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_PATH,
        help="CSV path for the skipped/failed-round log, always rewritten fresh (default: %(default)s)",
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute every reference date even if its results file already exists "
        "(default: resume, skipping dates already fully saved by a previous run)",
    )
    args = parser.parse_args()
    guaranteed_slots = _parse_guaranteed_slots(args.guaranteed_slot)

    df = pd.read_csv(args.matches)
    df["match_datetime"] = pd.to_datetime(df["match_datetime"])

    configs_by_season = load_configs_by_season(args.seasons)

    # {reference_date: [(config, season), ...]}, so two competitions whose
    # Monday/Friday checkpoint lands on the same day share one Stan fit.
    work: dict[pd.Timestamp, list[tuple[CompetitionConfig, int]]] = defaultdict(list)
    for season, configs in configs_by_season.items():
        for config in configs:
            for reference_date in reference_dates(df, config.name, season, config):
                work[reference_date].append((config, season))

    log_rows: list[dict] = []
    total_runs = len(work)  # every reference date across every requested season combined
    iter_sampling = -(-args.n_draws // args.chains)
    relegation_rng = np.random.default_rng(
        args.seed
    )  # tie-breaks in _relegated_teams_previous_season
    for run_index, reference_date in enumerate(sorted(work), start=1):
        _clear_screen()

        # Resumability: a (config, season) whose results file already exists
        # doesn't need to be resimulated, and if NONE of the items sharing
        # this date need it, the (costly) Stan fit itself is skipped
        # entirely -- re-running the same command after an interruption
        # picks up close to where it left off instead of redoing everything.
        pending = (
            work[reference_date]
            if args.force
            else [
                (config, season)
                for config, season in work[reference_date]
                if not _already_computed(config.name, season, reference_date, args.results_dir)
            ]
        )
        if not pending:
            print(
                f"Skipping run {run_index}/{total_runs} (as of {reference_date.date()}): "
                "already computed"
            )
            continue

        print(f"Executing run {run_index}/{total_runs} (as of {reference_date.date()})")
        train_df = df[df["match_datetime"] <= reference_date]

        # Broad except is deliberate here (unlike this codebase's usual
        # let-it-crash style): this loop can run for hours across dozens of
        # dates, and one bad date (e.g. a Stan convergence failure) shouldn't
        # cost every other date's already-computed results. Every (config,
        # season) sharing this reference_date's fit is logged as failed and
        # the backtest moves on to the next date.
        try:
            stan_data, teams = build_stan_data(train_df, reference_date=reference_date)
            mcmc_fit = fit_stan_data(
                stan_data,
                model=args.model,
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=iter_sampling,
            )
        except Exception as exc:
            print(f"FAILED to fit reference_date {reference_date.date()}: {exc}\n")
            for config, season in pending:
                log_rows.append(
                    {
                        "reference_date": reference_date.date().isoformat(),
                        "competition": config.name,
                        "season": season,
                        "status": "failed",
                        "reason": f"Stan fit failed: {exc}",
                    }
                )
            continue

        trained_teams = set(teams)
        for config, season in pending:
            missing = set(fixtures.season_teams(df, config.name, season)) - trained_teams
            team_aliases: dict[str, str] = {}
            if missing:
                # A team with zero matches before reference_date has no Stan
                # attack/defense estimate at all (not just a stale one) --
                # simulating its remaining fixtures would need to invent a
                # strength for it out of thin air. Before giving up on the
                # whole round, try substituting each such team with a
                # previous-season relegated team (see
                # _relegated_teams_previous_season/_debut_team_aliases) --
                # typically true for the pre-season checkpoint, where a
                # newly-promoted or long-absent team is otherwise the only
                # thing blocking a real prediction for round 1.
                relegated = _relegated_teams_previous_season(
                    df, config.name, season, config, relegation_rng
                )
                team_aliases = _debut_team_aliases(missing, relegated)
                still_missing = missing - set(team_aliases)
                if still_missing:
                    print(
                        f"=== {config.name} {season} (as of {reference_date.date()}) ==="
                        f"\nSkipped: {sorted(still_missing)} haven't played yet\n"
                    )
                    log_rows.append(
                        {
                            "reference_date": reference_date.date().isoformat(),
                            "competition": config.name,
                            "season": season,
                            "status": "skipped",
                            "reason": f"missing teams (haven't played yet): {sorted(still_missing)}",
                        }
                    )
                    continue

            try:
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
                    team_aliases=team_aliases or None,
                    model=args.model,
                )
                print(f"=== {config.name} {season} (as of {reference_date.date()}) ===")
                print(result.to_string(index=False))

                results_path = save_results(
                    result, config.name, season, reference_date, results_dir=args.results_dir
                )
                print(f"Saved results to {results_path}")
                if team_aliases:
                    print(
                        f"Substituted debut/stale team(s) with last season's relegated "
                        f"team(s): {team_aliases}"
                    )
                    log_rows.append(
                        {
                            "reference_date": reference_date.date().isoformat(),
                            "competition": config.name,
                            "season": season,
                            "status": "substituted",
                            "reason": (
                                "debut/stale team(s) given a previous season's relegated "
                                f"team's attack/defense: {team_aliases}"
                            ),
                        }
                    )
                print()
            except Exception as exc:
                print(f"FAILED {config.name} {season} (as of {reference_date.date()}): {exc}\n")
                log_rows.append(
                    {
                        "reference_date": reference_date.date().isoformat(),
                        "competition": config.name,
                        "season": season,
                        "status": "failed",
                        "reason": str(exc),
                    }
                )

    _write_log(log_rows, args.log)
    skipped_count = sum(1 for row in log_rows if row["status"] == "skipped")
    failed_count = sum(1 for row in log_rows if row["status"] == "failed")
    substituted_count = sum(1 for row in log_rows if row["status"] == "substituted")
    print(
        f"Done: {total_runs} run(s) executed, {skipped_count} round(s) skipped, "
        f"{failed_count} round(s) failed, {substituted_count} round(s) used a "
        f"debut/stale-team substitute. Log written to {args.log}"
    )


if __name__ == "__main__":
    main()
