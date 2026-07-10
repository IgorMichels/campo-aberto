"""CLI: exports real, not-yet-played fixtures (from the unified
data/processed/brazil/matches.csv, git-ignored) plus real, posterior-mean
team strengths (from data/results/*.csv's extra columns, also git-ignored --
see src.simulation.simulate._attach_team_strengths) into the static site's
committed Confrontos data (site/data/matches_manifest.json,
site/data/<slug>/matches_<season>.json, site/data/params.json).

Deliberately a separate module from src.site.export_site_data (standings/
odds export): different inputs (matches.csv + data/results' new columns,
vs. every dated results CSV), different output files (matches_manifest.json
vs. manifest.json -- "has upcoming fixtures right now" is a different
question from "has ever been backtested", so a finished season can appear
in one and not the other), and no probabilities baked in at export time --
this file writes only real fixtures + real model parameters, never a
scoreline. Every scoreline probability the Confrontos page shows is computed
client-side, live, by site/assets/js/dixon_coles.js from params.json -- see
plans/confrontos_rework.md Step 3/4/5.

Run after both `python -m src.ingestion.brazil.run_pipeline` (for a fresh
matches.csv) and `python -m src.simulation.run_rounds` / `src.pipeline`
(for the team-strength columns on data/results/), same as
export_site_data.py:

    python -m src.site.export_matches_data
"""

import argparse
import glob
import json
import os
from zoneinfo import ZoneInfo

import pandas as pd

from src.constants import CLUB_INFOS_PATH, DEFAULT_MATCHES_PATH, RESULTS_DIR, SITE_DIR
from src.simulation.run_rounds import load_configs_by_season
from src.site.export_site_data import DEFAULT_SEASONS, _competition_slug, _copy_crest

# How far ahead a "scheduled" match still counts as an upcoming card, and how
# many to fall back to (soonest first, regardless of date) when that window
# is empty -- e.g. during a real-world break like a World Cup, where every
# scheduled match is further out than 14 days.
FIXTURE_WINDOW_DAYS = 14
FIXTURE_FALLBACK_COUNT = 10

# matches.csv's match_datetime is always Brazil local time (see
# src.ingestion.brazil.build_treated_dataset, both CBF's own local timestamps
# and ESPN's UTC ones converted to match) -- converted back to true UTC here
# so the exported "date" field is an unambiguous ISO-8601 instant a browser's
# `new Date(...)` can render in whatever timezone it's actually running in,
# not just Brazil's. Brazil has used a fixed UTC-3 offset with no DST since
# 2019, same rationale as build_treated_dataset.py's own conversion.
_BRAZIL_TZ = ZoneInfo("America/Sao_Paulo")


def _latest_results_csv(results_dir: str = RESULTS_DIR) -> str:
    """Every dated CSV across every competition/season under results_dir,
    picks the single globally latest one by the date embedded in its
    filename (e.g. "2026_07_09.csv"). src.simulation.simulate._attach_team_strengths
    duplicates attack/defense/eta/beta_home/rho identically across every
    competition/season sharing a joint Stan fit at THAT SAME reference_date,
    so this file's shared scalars are the freshest ones available -- used
    for eta/beta_home/rho, see _load_params. NOT used alone for the "teams"
    dict any more -- see _latest_results_csv_by_competition below for why."""
    csv_paths = glob.glob(os.path.join(results_dir, "*", "*", "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No results CSVs found under {results_dir}")
    return max(csv_paths, key=lambda path: os.path.splitext(os.path.basename(path))[0])


def _latest_results_csv_by_competition(results_dir: str = RESULTS_DIR) -> dict[str, str]:
    """{competition_slug: latest dated CSV path} -- one per competition, not
    one globally. Two competitions' latest completed rounds can land on
    different dates (e.g. one paused for a real-world break, like a World
    Cup, while the other keeps playing -- confirmed the real, current case:
    Serie A's latest round is 2026-06-01, Serie B's is 2026-07-09), in which
    case a single globally-latest file's own `team` column only ever lists
    ONE competition's roster, silently dropping the other's teams from
    params.json entirely. Every competition's own latest file still has
    real, valid, just slightly-less-fresh attack/defense estimates for its
    own roster -- the best available, since re-fitting is out of this
    module's scope (data export only)."""
    csv_paths = glob.glob(os.path.join(results_dir, "*", "*", "*.csv"))
    latest_by_slug: dict[str, str] = {}
    for path in csv_paths:
        slug = os.path.relpath(path, results_dir).split(os.sep)[0]
        date_key = os.path.splitext(os.path.basename(path))[0]
        current = latest_by_slug.get(slug)
        if current is None or date_key > os.path.splitext(os.path.basename(current))[0]:
            latest_by_slug[slug] = path
    return latest_by_slug


def _load_params(results_dir: str = RESULTS_DIR) -> dict:
    """Shared scalars (eta/beta_home/rho, plus the reference_date they were
    fit as of) from the single globally-latest results CSV -- the freshest
    fit available. `teams`, however, is the UNION of every competition's own
    latest file's attack/defense (see _latest_results_csv_by_competition):
    using the single globally-latest file alone would silently drop every
    team from a competition whose own latest round fell on an earlier date
    (a real, currently-occurring case -- see that helper's docstring)."""
    csv_path = _latest_results_csv(results_dir)
    df = pd.read_csv(csv_path)
    reference_date = os.path.splitext(os.path.basename(csv_path))[0].replace("_", "-")
    first = df.iloc[0]

    teams: dict[str, dict] = {}
    for slug_csv_path in _latest_results_csv_by_competition(results_dir).values():
        slug_df = pd.read_csv(slug_csv_path)
        for _, row in slug_df.iterrows():
            teams[row["team"]] = {"attack": float(row["attack"]), "defense": float(row["defense"])}

    return {
        "reference_date": reference_date,
        "eta": float(first["eta"]),
        "beta_home": float(first["beta_home"]),
        "rho": float(first["rho"]),
        "teams": teams,
    }


def _to_utc_iso(ts: pd.Timestamp) -> str:
    localized = ts.tz_localize(_BRAZIL_TZ) if ts.tzinfo is None else ts
    return localized.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _selected_rows(
    matches_df: pd.DataFrame, competition: str, season: int, now: pd.Timestamp
) -> pd.DataFrame:
    """Every not-yet-played row for this competition+season that survives the
    14-day-window-then-fallback-to-10 scheduled selection, plus every
    postponed row (always included, no date to filter by) -- in final card
    order (dated soonest-first, then postponed alphabetically by home team).
    Factored out of _upcoming_cards so export_matches_data can also call it,
    BEFORE any crest gets copied, to know exactly which teams a card might
    need a crest for -- copying site/assets/crests/ eagerly for every club
    in club_infos.csv (~57, most of them irrelevant to any upcoming fixture)
    would commit a pile of unreferenced PNGs to the deployed site for no
    reason."""
    rows = matches_df[
        (matches_df["competition"] == competition)
        & (matches_df["season"] == season)
        & (matches_df["status"] != "played")
    ]

    scheduled = rows[rows["status"] == "scheduled"].sort_values("match_datetime")
    window_end = now + pd.Timedelta(days=FIXTURE_WINDOW_DAYS)
    windowed = scheduled[
        (scheduled["match_datetime"] >= now) & (scheduled["match_datetime"] <= window_end)
    ]
    dated_rows = windowed if not windowed.empty else scheduled.head(FIXTURE_FALLBACK_COUNT)

    # No date to filter by -- always included in full, alphabetical by home
    # team, appended after the dated ones (see the module docstring / plan's
    # sort-order requirement).
    postponed_rows = rows[rows["status"] == "postponed"].sort_values("home_team")

    return pd.concat([dated_rows, postponed_rows])


def _upcoming_cards(
    matches_df: pd.DataFrame,
    competition: str,
    season: int,
    now: pd.Timestamp,
    crest_by_team: dict,
    color_by_team: dict,
    known_teams: set,
) -> list[dict]:
    """Every real, not-yet-played match for this competition+season, as a
    card dict ready for site/data/<slug>/matches_<season>.json -- no scores/
    probabilities (computed client-side, see dixon_coles.js), no `round`
    concept (matches.csv has no round column, just dates)."""
    cards = []
    for _, row in _selected_rows(matches_df, competition, season, now).iterrows():
        home_team, away_team = row["home_team"], row["away_team"]
        if (
            home_team not in known_teams
            or away_team not in known_teams
            or home_team not in crest_by_team
            or away_team not in crest_by_team
        ):
            # A team the latest Stan fit has never seen (or with no crest to
            # show) -- the JS side could never compute a grid for it, so
            # catch it here at export time instead of shipping a dead card.
            print(
                f"Skipped {home_team} x {away_team} ({competition} {season}): "
                "missing team strengths or crest"
            )
            continue

        cards.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_crest": crest_by_team[home_team],
                "away_crest": crest_by_team[away_team],
                "home_color": color_by_team.get(home_team, "#4A5568"),
                "away_color": color_by_team.get(away_team, "#4A5568"),
                "date": _to_utc_iso(row["match_datetime"])
                if row["status"] == "scheduled"
                else None,
                "status": row["status"],
            }
        )
    return cards


def export_matches_data(
    seasons: list[int] = DEFAULT_SEASONS,
    results_dir: str = RESULTS_DIR,
    matches_path: str = DEFAULT_MATCHES_PATH,
    club_infos_path: str = CLUB_INFOS_PATH,
    site_dir: str = SITE_DIR,
    now: pd.Timestamp | None = None,
) -> None:
    if now is None:
        now = pd.Timestamp.now()

    params = _load_params(results_dir)
    known_teams = set(params["teams"])

    club_infos = pd.read_csv(club_infos_path)
    raw_crest_path_by_team = dict(zip(club_infos["club"], club_infos["crest_path"]))
    color_by_team = dict(zip(club_infos["club"], club_infos["primary_color"]))
    crests_dir = os.path.join(site_dir, "assets", "crests")

    matches_df = pd.read_csv(matches_path)
    matches_df["match_datetime"] = pd.to_datetime(matches_df["match_datetime"])

    configs_by_season = load_configs_by_season(seasons)

    # Only copy a crest for a team that could actually end up in a card
    # (known to the latest Stan fit AND surviving the window/postponed
    # selection for some competition+season being exported) -- see
    # _selected_rows' docstring for why this matters (avoids committing
    # unreferenced crests for every one of club_infos.csv's ~57 clubs).
    needed_teams: set = set()
    for season, configs in configs_by_season.items():
        for config in configs:
            selected = _selected_rows(matches_df, config.name, season, now)
            needed_teams.update(selected["home_team"])
            needed_teams.update(selected["away_team"])
    needed_teams &= known_teams

    crest_by_team = {}
    for team in needed_teams:
        src_path = raw_crest_path_by_team.get(team)
        if src_path is not None and pd.notna(src_path):
            crest_by_team[team] = _copy_crest(src_path, crests_dir)

    data_dir = os.path.join(site_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # {slug: (competition name, [season, ...])}, only combos that actually
    # produced at least one card -- a finished season/competition naturally
    # disappears from both the manifest and its own JSON file.
    exported: dict[str, tuple[str, list[int]]] = {}
    for season, configs in sorted(configs_by_season.items()):
        for config in configs:
            slug = _competition_slug(config.name)
            cards = _upcoming_cards(
                matches_df, config.name, season, now, crest_by_team, color_by_team, known_teams
            )
            if not cards:
                print(f"Skipped {config.name} {season}: no upcoming cards")
                continue

            season_dir = os.path.join(data_dir, slug)
            os.makedirs(season_dir, exist_ok=True)
            season_path = os.path.join(season_dir, f"matches_{season}.json")
            with open(season_path, "w", encoding="utf-8") as f:
                json.dump({"matches": cards}, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"Wrote {season_path} ({len(cards)} card(s))")

            exported.setdefault(slug, (config.name, []))[1].append(season)

    manifest = {
        "competitions": [
            {"competition": name, "slug": slug, "seasons": sorted(seasons_)}
            for slug, (name, seasons_) in exported.items()
        ]
    }
    manifest_path = os.path.join(data_dir, "matches_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {manifest_path}")

    params_path = os.path.join(data_dir, "params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {params_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--matches", default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--club-infos", default=CLUB_INFOS_PATH)
    parser.add_argument("--site-dir", default=SITE_DIR)
    args = parser.parse_args()
    export_matches_data(
        args.seasons, args.results_dir, args.matches, args.club_infos, args.site_dir
    )


if __name__ == "__main__":
    main()
