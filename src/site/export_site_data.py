"""CLI: exports simulation results (data/results/*.csv, git-ignored) into the
static site's committed data + crest assets (site/data/*.json,
site/assets/crests/) -- see site/README.md for the exported JSON schema.

Reuses src.simulation.run_rounds.load_configs_by_season to discover which
competition+season combinations exist (the same configs/*.yaml the
simulation itself reads), then for each one exports every dated CSV under
data/results/<slug>/<season>/ (see src.simulation.results.save_results for
how that path and its dated filenames are produced) as one selectable
reference-date snapshot, each paired with a real (not simulated) standings
table as of that same date -- see src.simulation.fixtures.split_fixtures +
src.simulation.standings.team_records.

Must be re-run (and its site/ output committed) after every
`python -m src.pipeline`, since data/results/ never reaches git/CI on its
own -- see site/README.md.

    python -m src.site.export_site_data
"""

import argparse
import filecmp
import glob
import json
import os
import shutil

import pandas as pd

from src.constants import CLUB_INFOS_PATH, DEFAULT_MATCHES_PATH, RESULTS_DIR, SITE_DIR
from src.simulation import fixtures, standings
from src.simulation.config import AggregateConfig
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

DEFAULT_SEASONS = [2025, 2026]


def _competition_slug(name: str) -> str:
    """Same slug save_results derives from a competition name (e.g. "Serie A" ->
    "serie_a") -- kept in sync by hand since results.py doesn't expose it."""
    return name.lower().replace(" ", "_")


def _all_results_csvs(slug: str, season: int, results_dir: str) -> list[str]:
    """Every dated CSV for this competition+season, oldest first -- each one
    becomes a selectable reference-date snapshot on the site."""
    return sorted(glob.glob(os.path.join(results_dir, slug, str(season), "*.csv")))


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


def _real_standings(
    matches_df: pd.DataFrame,
    competition: str,
    season: int,
    reference_date: pd.Timestamp,
    teams: list[str],
) -> dict[str, dict]:
    """Points/played/goals_for/goals_against/goal_diff per team from
    actually-played matches up to reference_date -- the real table as of that
    date, not a simulated one, so a reader can see what the probabilities are
    reacting to."""
    played_results, _, _ = fixtures.split_fixtures(
        matches_df, competition, season, reference_date, teams=teams
    )
    records = standings.team_records(teams, played_results)
    return {
        team: {
            "points": rec["points"],
            "played": rec["played"],
            "goals_for": rec["goals_for"],
            "goals_against": rec["goals_against"],
            "goal_diff": rec["goal_diff"],
        }
        for team, rec in records.items()
    }


def _export_snapshot(
    csv_path: str,
    crest_by_team: dict,
    color_by_team: dict,
    crests_dir: str,
    aggregates: tuple[AggregateConfig, ...],
    matches_df: pd.DataFrame,
    competition: str,
    season: int,
) -> tuple[str, list[dict], list[dict]]:
    """One dated CSV -> (reference_date, columns, teams) for a single snapshot.
    `columns` is only exposed so the caller can sanity-check it stays the same
    across every date in a season (it's config-driven, so it always should)."""
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
    team_names = list(df["team"])
    standings_by_team = _real_standings(
        matches_df, competition, season, pd.Timestamp(reference_date), team_names
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
    aggregates: tuple[AggregateConfig, ...],
    matches_df: pd.DataFrame,
    competition: str,
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
            aggregates,
            matches_df,
            competition,
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
                    config.aggregates,
                    matches_df,
                    config.name,
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
