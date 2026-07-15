"""CLI: exports simulation results (data/results/*.csv, git-ignored) into the
static site's committed data + crest assets (site/data/*.json,
site/assets/crests/) -- see site/README.md for the exported JSON schema.

Reuses src.simulation.run_rounds.load_configs_by_season to discover which
competition+season combinations exist (the same configs/*.yaml the
simulation itself reads), then for each one exports every dated CSV under
data/results/<slug>/<season>/ (see src.simulation.results.save_results for
how that path and its dated filenames are produced) as one selectable
reference-date snapshot, each paired with a real (not simulated) standings
table as of that same date -- see src.simulation.fixtures.split_fixtures,
src.simulation.standings.team_records/rank_table/resolve_cascade.

Run as the last step of `python -m src.pipeline`; its site/ output still
needs to be reviewed and committed for a deploy to go out, since
data/results/ never reaches git/CI on its own -- see site/README.md.
Can also be run standalone, e.g. after editing configs/*.yaml or
data/club_infos.csv without rerunning the full pipeline:

    python -m src.site.export_site_data
"""

import argparse
import bisect
import filecmp
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from src.constants import (
    CLUB_INFOS_PATH,
    DEFAULT_MATCHES_PATH,
    DEFAULT_SEASON,
    RESULTS_DIR,
    SITE_DIR,
)
from src.simulation import fixtures, standings
from src.simulation.config import (
    AggregateConfig,
    CompetitionConfig,
    PlayoffPhaseConfig,
    RoundRobinPhaseConfig,
)
from src.simulation.run_rounds import load_configs_by_season

# Portuguese labels for every non-aggregate spot declared across
# configs/serie_*.yaml (see configs/README.md). Short on purpose: a spot that's
# part of an aggregate (see AGGREGATE_GROUP_LABELS) renders nested under its
# group header, which already gives the "Libertadores"/"Acesso" context, so
# repeating it here would just make columns wider for no reason.
SPOT_LABELS = {
    "title": "Título",
    "libertadores_grupos": "Fase de grupos",
    "libertadores_pre": "Pré-fase",
    "sulamericana": "Sul-Americana",
    "rebaixamento": "Rebaixamento",
    "direct_promotion": "Direto",
    "playoff_promotion": "Playoff",
}

# Aggregate name -> group header label. Grouping falls directly out of each
# config's own `aggregates: [{name, of: [...]}]` declaration (see
# configs/README.md) instead of hardcoding which spots belong together, so
# any future aggregate gets the same treatment as long as it's added here.
AGGREGATE_GROUP_LABELS = {
    "libertadores": "Libertadores",
    "promotion": "Acesso",
}
AGGREGATE_TOTAL_LABEL = (
    "Geral"  # the aggregate's own combined probability, nested as a group's last child
)

DEFAULT_SEASONS = [DEFAULT_SEASON - 1, DEFAULT_SEASON]  # current season + previous season


def _competition_slug(name: str) -> str:
    """Same slug save_results derives from a competition name (e.g. "Serie A" ->
    "serie_a") -- kept in sync by hand since results.py doesn't expose it."""
    return name.lower().replace(" ", "_")


def _all_results_csvs(slug: str, season: int, results_dir: str) -> list[str]:
    """Every dated CSV for this competition+season, oldest first -- each one
    becomes a selectable reference-date snapshot on the site."""
    return sorted(glob.glob(os.path.join(results_dir, slug, str(season), "*.csv")))


def _snapshot_csv_before(
    slug: str, season: int, before: pd.Timestamp, results_dir: str
) -> str | None:
    """The latest dated CSV for this competition+season whose embedded date is
    STRICTLY before `before`'s calendar date -- i.e. the most recent model
    snapshot fit before a given match was played. Filenames are YYYY_MM_DD.csv,
    so a plain string comparison against `before`'s own YYYY-MM-DD is already
    chronological (see _all_results_csvs). Returns None when no such snapshot
    exists yet -- a real, confirmed case for a season's earliest played
    matches, which predate that competition+season's very first backtest."""
    csv_paths = _all_results_csvs(slug, season, results_dir)
    dates = [os.path.splitext(os.path.basename(p))[0].replace("_", "-") for p in csv_paths]
    idx = bisect.bisect_left(dates, before.strftime("%Y-%m-%d"))
    return csv_paths[idx - 1] if idx > 0 else None


def _copy_crest(src_path: str, crests_dir: str) -> str:
    """Copies src_path into crests_dir, skipping the write if an identical file is
    already there. Returns the path the front-end should use for it, relative to
    site/."""
    os.makedirs(crests_dir, exist_ok=True)
    filename = os.path.basename(src_path)
    dest_path = os.path.join(crests_dir, filename)
    if not os.path.exists(dest_path) or not filecmp.cmp(src_path, dest_path, shallow=False):
        shutil.copyfile(src_path, dest_path)
    return f"assets/crests/{filename}"


def _build_columns(raw_names: list[str], aggregates: tuple[AggregateConfig, ...]) -> list[dict]:
    """Turns a flat list of raw spot/aggregate names (CSV column order) into a
    front-end-ready column tree: every aggregate's `of` spots + the aggregate's
    own total are nested under one group column, in the group's own `of` order,
    with the group placed where its first child would otherwise have appeared.
    Everything not part of any aggregate stays a flat, standalone column."""
    child_to_aggregate = {child: agg for agg in aggregates for child in agg.of}
    grouped_names = set(child_to_aggregate) | {agg.name for agg in aggregates}

    columns = []
    emitted_groups = set()
    for raw in raw_names:
        if raw not in grouped_names:
            columns.append({"key": raw, "label": SPOT_LABELS[raw]})
            continue

        agg = child_to_aggregate.get(raw) or next(a for a in aggregates if a.name == raw)
        if agg.name in emitted_groups:
            continue
        emitted_groups.add(agg.name)
        columns.append(
            {
                "key": agg.name,
                "label": AGGREGATE_GROUP_LABELS[agg.name],
                "children": [{"key": child, "label": SPOT_LABELS[child]} for child in agg.of]
                + [{"key": agg.name, "label": AGGREGATE_TOTAL_LABEL}],
            }
        )
    return columns


def _active_guaranteed_slots(
    config: CompetitionConfig, reference_date: pd.Timestamp
) -> dict[str, list[str]]:
    """{team: [spot_name, ...]} for every config.guaranteed_slots entry already
    known as of reference_date -- same known_from gating simulate_competition
    itself applies (see GuaranteedSlotConfig), so the real table's zone
    assignment reacts to a berth (e.g. a Libertadores champion decided
    mid-season) on the same date the probabilities do."""
    guaranteed_slots: dict[str, list[str]] = {}
    for entry in config.guaranteed_slots:
        if reference_date >= entry.known_from:
            guaranteed_slots.setdefault(entry.team, []).append(entry.spot)
    return guaranteed_slots


def _real_classification(
    teams: list[str],
    played_results: list[tuple],
    league_phase: RoundRobinPhaseConfig,
    playoff_phases: list[PlayoffPhaseConfig],
    guaranteed_slots: dict[str, list[str]],
) -> dict[str, dict]:
    """Real (not simulated) {team: {"rank", "zone"}} from already-played results:
    `rank` is the official classification position (standings.rank_table's full
    CBF tiebreak, not just points), `zone` is the positions-based spot (e.g.
    libertadores_grupos, rebaixamento) that position earns -- run through
    standings.resolve_cascade for any spot in league_phase.cascade, so an
    externally guaranteed berth (see _active_guaranteed_slots) shifts the real
    zone boundary exactly as it would the final table, not just table position.
    `zone` is None for a position outside every declared spot (e.g. mid-table).

    A `table_position`-paired playoff phase sourced from league_phase (e.g.
    Serie B's access playoff, "cruzamento olímpico") isn't itself a `positions`
    spot -- its own spot only fires once that playoff is actually played out
    (`result: winner`) -- but the real table can still show which teams
    currently occupy that bracket, same as any other zone, from the phase's
    own `pairs`.

    The random-draw tiebreak step (rank_table's last resort, matching REC's own
    "sorteio") uses a fixed seed: real disciplinary data isn't available to
    break a genuine last-resort tie, and a fixed seed keeps a given date's
    export reproducible across reruns.
    """
    rng = np.random.default_rng(0)
    order = standings.rank_table(teams, played_results, rng, league_phase.head_to_head_mode)
    rank_of = {team: position for position, team in enumerate(order, start=1)}

    zone_of: dict[str, str] = {}
    cascade_spots = [
        next(s for s in league_phase.spots if s.name == name) for name in league_phase.cascade
    ]
    if cascade_spots:
        credited = standings.resolve_cascade(order, cascade_spots, guaranteed_slots)
        for spot_name, recipients in credited.items():
            for team in recipients:
                zone_of[team] = spot_name

    # Widest range first: a spot fully nested inside a broader one (e.g. Serie
    # A/B's title, always positions (1, 1)) has no cascade entry of its own to
    # resolve the overlap the way libertadores_grupos does above, so without
    # this ordering it would win the position from the broader spot it's
    # actually part of just by coming first in the config's spot list.
    positional_spots = sorted(
        (s for s in league_phase.spots if s.name not in league_phase.cascade and s.positions),
        key=lambda s: s.positions[1] - s.positions[0],
        reverse=True,
    )
    for spot in positional_spots:
        for team in teams:
            if team in zone_of:
                continue
            if spot.positions[0] <= rank_of[team] <= spot.positions[1]:
                zone_of[team] = spot.name

    for phase in playoff_phases:
        if phase.pairing != "table_position" or phase.source_phase != league_phase.id:
            continue
        positions = {position for pair in phase.pairs for position in pair}
        spot_name = phase.spots[0].name
        for team in teams:
            if team not in zone_of and rank_of[team] in positions:
                zone_of[team] = spot_name

    return {team: {"rank": rank_of[team], "zone": zone_of.get(team)} for team in teams}


def _real_standings(
    matches_df: pd.DataFrame,
    competition: str,
    season: int,
    reference_date: pd.Timestamp,
    teams: list[str],
    league_phase: RoundRobinPhaseConfig,
    playoff_phases: list[PlayoffPhaseConfig],
    guaranteed_slots: dict[str, list[str]],
) -> dict[str, dict]:
    """Points/played/goals_for/goals_against/goal_diff/rank/zone per team from
    actually-played matches up to reference_date -- the real table as of that
    date, not a simulated one, so a reader can see what the probabilities are
    reacting to. `rank`/`zone` come from _real_classification (see there for
    the guaranteed-slot cascade this reuses from the simulation itself)."""
    played_results, _, _ = fixtures.split_fixtures(
        matches_df, competition, season, reference_date, teams=teams
    )
    records = standings.team_records(teams, played_results)
    classification = _real_classification(
        teams, played_results, league_phase, playoff_phases, guaranteed_slots
    )
    return {
        team: {
            "points": rec["points"],
            "played": rec["played"],
            "goals_for": rec["goals_for"],
            "goals_against": rec["goals_against"],
            "goal_diff": rec["goal_diff"],
            "rank": classification[team]["rank"],
            "zone": classification[team]["zone"],
        }
        for team, rec in records.items()
    }


def _export_snapshot(
    csv_path: str,
    crest_by_team: dict,
    color_by_team: dict,
    crests_dir: str,
    config: CompetitionConfig,
    matches_df: pd.DataFrame,
    season: int,
) -> tuple[str, list[dict], list[dict]]:
    """One dated CSV -> (reference_date, columns, teams) for a single snapshot.
    `columns` is only exposed so the caller can sanity-check it stays the same
    across every date in a season (it's config-driven, so it always should)."""
    aggregates = config.aggregates
    league_phase = next(p for p in config.phases if isinstance(p, RoundRobinPhaseConfig))
    playoff_phases = [p for p in config.phases if isinstance(p, PlayoffPhaseConfig)]

    df = pd.read_csv(csv_path).drop(columns=["expected_position"])
    prob_columns = [c for c in df.columns if c.startswith("prob_")]
    raw_names = [c.removeprefix("prob_") for c in prob_columns]

    aggregate_names = {agg.name for agg in aggregates}
    unknown = set(raw_names) - set(SPOT_LABELS) - aggregate_names
    if unknown:
        raise ValueError(
            f"{csv_path}: no Portuguese label for spot(s) {sorted(unknown)} -- add to SPOT_LABELS"
        )
    missing_group_labels = aggregate_names - AGGREGATE_GROUP_LABELS.keys()
    if missing_group_labels:
        raise ValueError(
            f"{csv_path}: no group label for aggregate(s) {sorted(missing_group_labels)} -- add to AGGREGATE_GROUP_LABELS"
        )
    columns = _build_columns(raw_names, aggregates)

    reference_date = os.path.splitext(os.path.basename(csv_path))[0].replace("_", "-")
    reference_timestamp = pd.Timestamp(reference_date)
    team_names = list(df["team"])
    guaranteed_slots = _active_guaranteed_slots(config, reference_timestamp)
    standings_by_team = _real_standings(
        matches_df,
        config.name,
        season,
        reference_timestamp,
        team_names,
        league_phase,
        playoff_phases,
        guaranteed_slots,
    )

    teams = []
    for _, row in df.iterrows():
        team = row["team"]
        crest_path = crest_by_team.get(team)
        if not crest_path or pd.isna(crest_path):
            raise ValueError(
                f"{csv_path}: team {team!r} has no crest_path in {CLUB_INFOS_PATH} -- add one before exporting"
            )
        teams.append(
            {
                "team": team,
                "crest": _copy_crest(crest_path, crests_dir),
                "color": color_by_team.get(team, "#4A5568"),
                "standings": standings_by_team[team],
                "probs": {
                    raw: round(float(row[col]), 4) for col, raw in zip(prob_columns, raw_names)
                },
            }
        )

    return reference_date, columns, teams


def _export_season(
    csv_paths: list[str],
    crest_by_team: dict,
    color_by_team: dict,
    crests_dir: str,
    config: CompetitionConfig,
    matches_df: pd.DataFrame,
    season: int,
) -> dict:
    dates = []
    snapshots = {}
    columns = None
    for csv_path in csv_paths:
        date, date_columns, teams = _export_snapshot(
            csv_path,
            crest_by_team,
            color_by_team,
            crests_dir,
            config,
            matches_df,
            season,
        )
        dates.append(date)
        snapshots[date] = {"teams": teams}
        columns = date_columns  # identical across dates in practice -- both config-driven
    return {"columns": columns, "dates": dates, "snapshots": snapshots}


def export_site_data(
    seasons: list[int] = DEFAULT_SEASONS,
    results_dir: str = RESULTS_DIR,
    club_infos_path: str = CLUB_INFOS_PATH,
    matches_path: str = DEFAULT_MATCHES_PATH,
    site_dir: str = SITE_DIR,
) -> None:
    club_infos = pd.read_csv(club_infos_path)
    crest_by_team = dict(zip(club_infos["club"], club_infos["crest_path"]))
    color_by_team = dict(zip(club_infos["club"], club_infos["primary_color"]))
    crests_dir = os.path.join(site_dir, "assets", "crests")
    data_dir = os.path.join(site_dir, "data")

    matches_df = pd.read_csv(matches_path)
    matches_df["match_datetime"] = pd.to_datetime(matches_df["match_datetime"])

    # {slug: (competition name, [season, ...])}, only seasons actually exported.
    exported: dict[str, tuple[str, list[int]]] = {}
    for season, configs in sorted(load_configs_by_season(seasons).items()):
        for config in configs:
            slug = _competition_slug(config.name)
            csv_paths = _all_results_csvs(slug, season, results_dir)
            if not csv_paths:
                print(
                    f"Skipped {config.name} {season}: no results under {results_dir}/{slug}/{season}/"
                )
                continue

            try:
                season_data = _export_season(
                    csv_paths,
                    crest_by_team,
                    color_by_team,
                    crests_dir,
                    config,
                    matches_df,
                    season,
                )
            except ValueError as exc:
                # A data-completeness gap (missing crest_path, undeclared spot label)
                # -- fail loudly for this one competition/season, but let every other
                # one still export instead of losing the whole run over it.
                print(f"ERROR: {exc}")
                continue

            season_dir = os.path.join(data_dir, slug)
            os.makedirs(season_dir, exist_ok=True)
            season_path = os.path.join(season_dir, f"{season}.json")
            with open(season_path, "w", encoding="utf-8") as f:
                json.dump(season_data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"Wrote {season_path} ({len(csv_paths)} dated snapshot(s))")

            exported.setdefault(slug, (config.name, []))[1].append(season)

    manifest = {
        "competitions": [
            {"competition": name, "slug": slug, "seasons": sorted(seasons_)}
            for slug, (name, seasons_) in exported.items()
        ]
    }
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--club-infos", default=CLUB_INFOS_PATH)
    parser.add_argument("--matches", default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--site-dir", default=SITE_DIR)
    args = parser.parse_args()
    export_site_data(args.seasons, args.results_dir, args.club_infos, args.matches, args.site_dir)


if __name__ == "__main__":
    main()
