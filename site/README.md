# Site

This directory is the deployable output of the results website: a static
page (plain HTML/CSS/vanilla JS, no build step) rendered from the simulation
outputs under `data/results/` (git-ignored). Unlike `data/results/`, this
directory _is_ tracked -- what's committed here is exactly what gets served.

A push to `main` that touches this directory triggers
`.github/workflows/deploy-site.yml`, which deploys it to GitHub Pages
(`actions/deploy-pages`, no `gh-pages` branch).

`index.html` is a static landing page (no data fetch of its own yet, just
navigation cards) -- the classification table that used to live at
`index.html` is now `probabilities.html`.

## Regenerating

`python -m src.pipeline` regenerates this directory's data as its last step,
calling both `src.site.export_site_data` (standings/odds:
`manifest.json`, `<slug>/<season>.json`) and `src.site.export_matches_data`
(the "Jogos" pages: `matches_manifest.json`, `<slug>/matches_<season>.json`,
`played_manifest.json`, `<slug>/played_<season>.json`, `params.json`),
passing both the same reference date so they agree on "as of". Review and
commit the result -- that commit is what actually triggers a new deploy,
since `data/results/` itself never reaches git or CI.

To regenerate just the site data (e.g. after editing `configs/*.yaml` or
`data/club_infos.csv` without rerunning the full pipeline), run the export
on its own:

```bash
python -m src.site.export_site_data
```

To regenerate just the "Jogos" data (e.g. after editing `matches.csv` or
`data/club_infos.csv` without rerunning the full pipeline), run:

```bash
python -m src.site.export_matches_data
```

This must run _after_ `python -m src.ingestion.brazil.run_pipeline` (which
produces the merged `matches.csv` both upcoming- and played-fixture cards
read from) and _after_ `python -m src.simulation.run_rounds` / `python -m
src.pipeline` (which produce the `model` column plus whichever team/shared
columns that model declares -- see `src/models/registry.py` -- on
`data/results/` that `params.json` -- and each played match's own embedded
historical params slice, see below -- are built from).

## Data schema

`data/manifest.json`: `{"competitions": [{"competition": "Serie A", "slug": "serie_a", "seasons": [2025, 2026]}, ...]}`.

`data/<slug>/<season>.json`, one file per competition/season:

```json
{
  "columns": [
    {"key": "title", "label": "Título"},
    {"key": "libertadores", "label": "Libertadores", "children": [
      {"key": "libertadores_grupos", "label": "Fase de grupos"},
      {"key": "libertadores_pre", "label": "Pré-fase"},
      {"key": "libertadores", "label": "Geral"}
    ]},
    {"key": "rebaixamento", "label": "Rebaixamento"}
  ],
  "dates": ["2025-03-31", "...", "2025-12-07"],
  "snapshots": {
    "2025-12-07": {
      "teams": [
        {"team": "Flamengo / RJ", "crest": "assets/crests/flarj.png", "color": "#C1121F",
         "acronym": "FLA",
         "standings": {"points": 79, "played": 38, "goals_for": 78, "goals_against": 27, "goal_diff": 51,
                       "rank": 1, "zone": "libertadores_grupos"},
         "probs": {"title": 1.0, "libertadores_grupos": 1.0, "libertadores_pre": 0.0,
                   "libertadores": 1.0, "rebaixamento": 0.0}},
        ...
      ]
    },
    ...
  }
}
```

- `columns` is a tree, identical across every date in a season (config-driven). A
  `{key, label}` entry is a standalone column; a `{key, label, children}` entry is a
  group (one per `aggregates:` entry in that competition's `configs/*.yaml`) whose
  `children` are its constituent spots plus the aggregate's own combined probability,
  labeled "Geral". Flatten with `columns.flatMap(c => c.children || [c])` to get the
  leaf render order (same order `app.js`'s `leafColumns()` uses).
- `dates` is every reference date this season has a simulated snapshot for (oldest
  first, one per round backtested by `src.simulation.run_rounds`).
- `snapshots[date].teams` is that date's team list in the row order the simulation
  produced (best expected position first) -- render as-is, don't re-sort.
- `probs` is keyed by raw spot/aggregate name (not the Portuguese label) -- a leaf
  column's own `key` is exactly what to look up.
- `standings` is real (not simulated) points/played/goals_for/goals_against/goal_diff
  as of that snapshot's date, computed from `data/processed/brazil/matches.csv`.
  `rank` is that date's official classification position (the full CBF tiebreak,
  via `src.simulation.standings.rank_table` -- not just points). `zone` is the
  `positions`-based spot name (e.g. `libertadores_grupos`, `direct_promotion`,
  `rebaixamento`) that rank currently earns, or `null` for a mid-table position
  outside every declared spot -- for any spot in that phase's `cascade` (see
  `configs/*.yaml`), this already accounts for externally guaranteed slots
  (e.g. a Copa do Brasil berth) via `src.simulation.standings.resolve_cascade`,
  so a guarantee can shift the real zone boundary exactly as it would the
  final table, not just raw table position. A `table_position`-paired playoff
  phase (e.g. Serie B's access playoff, "cruzamento olímpico") also gets a
  zone this way, from the playoff phase's own `pairs` (e.g.
  `playoff_promotion` for whichever positions currently seed it), even though
  that phase's own spot only resolves once the playoff is actually played.
  Meant to be rendered as-is (a rank number + a zone-based row color), not
  recomputed client-side.
- `crest`/`color`/`acronym` come straight from `data/assets/club_infos.csv`
  (`crest_path`/`primary_color`/`acronym` columns) -- `crest` is already
  rewritten to a `site`-root-relative path, `color` a hex string, `acronym`
  the hand-maintained broadcast-style 3-letter code (e.g. "FLA"), all used
  as-is.

## Jogos

Three pages under `matches/` share one match-score-probability "sticker"
card component (a 5x5 heatmap of scoreline probabilities plus a home/draw/
away win bar), inspired by the WorldCup2026 project's `previsoes.html`
sticker cards, and one shared modal (zoom/download-as-PNG/share-link):

1. **`matches/upcoming.html`** ("Próximos") -- every not-yet-played row in
   `data/processed/brazil/matches.csv` (the unified CBF+ESPN dataset, see
   below), soonest-first, plus a "Data a definir" bucket for postponed
   matches with no confirmed date, in a paginated grid (10 by default,
   "mostrar mais" reveals more from the same already-fetched list).
2. **`matches/played.html`** ("Passados") -- every already-played match for
   the same tracked seasons, most-recent-first, each sticker computed from
   the model snapshot most recently fit _before_ that specific match was
   played (not today's params) with the real final score highlighted; a
   match predating that competition/season's first backtest snapshot shows
   a "sem modelo disponível ainda" placeholder instead of a probability
   grid.
3. **`matches/simulate.html`** ("Simule") -- a FIFA-style free-pick builder:
   choose a competition then a team, for both home and away (any two teams,
   any competition/season, including cross-division matchups), and see a
   live scoreline grid for a matchup that may never actually be scheduled.

All three card types are computed the same way, by the same client-side JS
function (`computeCard(base, params)` in `assets/js/matches_shared.js`) --
there is no separate formula per page, only a different `params` object:
the current shared `params.json` for upcoming/free-pick cards, or a played
match's own embedded historical snapshot for past cards.

### Data lineage

`data/processed/brazil/matches.csv` is the single source of truth for every
match, played or not: CBF's official docket score wins whenever available;
ESPN's public scoreboard API fills in the schedule (and, for a match CBF
hasn't confirmed yet, the only date available at all). Produced by
`python -m src.ingestion.brazil.run_pipeline` (or `python -m src.pipeline`),
which now also fetches ESPN's fixture list and merges it into `matches.csv`
alongside CBF's results -- see `data/processed/brazil/README.md` for the
merge precedence and the `score_discrepancies.csv` sanity report.

`data/results/<slug>/<season>/<date>.csv` (produced by
`python -m src.simulation.run_rounds` / `python -m src.pipeline`) carries,
alongside its existing `team`/`expected_position`/`prob_*` columns, a
`model` column naming which `src/models/registry.py` adapter produced that
round's Stan fit, plus whichever team/shared columns that adapter declares
(posterior-**mean** `attack`/`defense` and shared `eta`/`beta_home`/`rho`,
for today's only registered model, `poisson_home`). `export_matches_data`
reads the single globally-latest such file for the shared scalars, and the
union of every competition's own latest file for the `teams` dict (Serie A
and Serie B are fit jointly, so their team-param values already live on one
shared scale, even when their schedules have drifted to different
latest-played dates) -- and raises loudly if two competitions' latest files
disagree on `model` (a reachable mid-migration state, not something to
silently merge).

### Scoreline math: client-side JS, one implementation per model

All scoreline probabilities -- for both real-fixture cards and free-pick
cards -- are computed live in the browser, dispatched through
`window.ScoreModels[params.model]` (`assets/js/score_models.js`) to
whichever model produced that `params` object's numbers. Today's only
registered model, `poisson_home` (`assets/js/poisson_home.js`,
`.matchRates` / `.scorelineProbabilities` / `.teamStrength`), is a
Dixon-Coles-adjusted independent-Poisson model: the same closed form
`src/models/poisson_home.stan` fits and
`src/models/adapters/poisson_home.py` already codes as a rejection-sampling
bound, evaluated here as an explicit probability instead. There is
deliberately no Python port of this formula for _rendering_ -- every
parameter it needs is already shipped to the browser in `params.json`
(`shared`/`teams`, see below), so a server-side implementation would be
pure duplication. A candidate model (see `src/models/registry.py`) needs
one new JS file implementing the same 3-function contract, one line
registering it in `score_models.js`'s style, and one `<script>` tag added
to each of `matches/{upcoming,played,simulate}.html` -- `matches_shared.js`
itself needs no change.

### Data schema

`data/matches_manifest.json` -- same shape as `manifest.json`, but scoped
to "has upcoming fixtures right now" rather than "has ever been
backtested" (a finished season/competition simply has no entry):

```json
{"competitions": [{"competition": "Serie A", "slug": "serie_a", "seasons": [2026]}, ...]}
```

`data/<slug>/matches_<season>.json` -- **every** remaining not-yet-played
match (no date window or count cap -- `matches/upcoming.html` paginates
client-side over the full list). No probabilities here; they're computed
client-side from `params.json`. Sorted scheduled-first-by-date, then
postponed last (alphabetical by home team):

```json
{
  "matches": [
    {
      "home_team": "Atlético Mineiro / MG",
      "away_team": "Flamengo / RJ",
      "home_crest": "assets/crests/atl_mg.png",
      "away_crest": "assets/crests/flarj.png",
      "home_color": "#000000",
      "away_color": "#C1121F",
      "date": "2026-07-21T22:30:00Z",
      "status": "scheduled"
    },
    {
      "home_team": "...",
      "away_team": "...",
      "home_crest": "...",
      "away_crest": "...",
      "home_color": "...",
      "away_color": "...",
      "date": null,
      "status": "postponed"
    }
  ]
}
```

`data/played_manifest.json` -- same shape as `matches_manifest.json`, but
scoped to "has at least one played card exported" (a season/competition can
appear here, in `matches_manifest.json`, in both, or in neither):

```json
{"competitions": [{"competition": "Serie A", "slug": "serie_a", "seasons": [2025, 2026]}, ...]}
```

`data/<slug>/played_<season>.json` -- every already-played match for this
competition/season, most-recent-first, each embedding its OWN 2-team params
slice (the model snapshot most recently fit _strictly before_ that match's
own date -- see `src.site.export_site_data._snapshot_csv_before` /
`src.site.export_matches_data._played_cards`) instead of reading the page's
shared `params.json`, since different matches reference different
historical dates:

```json
{
  "matches": [
    {
      "home_team": "Flamengo / RJ",
      "away_team": "Palmeiras / SP",
      "home_crest": "assets/crests/flarj.png",
      "away_crest": "assets/crests/palsp.png",
      "home_color": "#C1121F",
      "away_color": "#006437",
      "date": "2026-05-10T22:00:00Z",
      "home_goals": 2,
      "away_goals": 1,
      "has_model": true,
      "reference_date": "2026-05-04",
      "params": {
        "model": "poisson_home",
        "shared": { "eta": 0.021, "beta_home": 0.318, "rho": 0.026 },
        "teams": {
          "Flamengo / RJ": { "attack": 0.41, "defense": 0.18 },
          "Palmeiras / SP": { "attack": 0.52, "defense": 0.09 }
        }
      }
    },
    {
      "home_team": "...",
      "away_team": "...",
      "...": "...",
      "has_model": false,
      "reference_date": null,
      "params": null
    }
  ]
}
```

`has_model: false` (with `reference_date`/`params` both `null`) happens for
a match with no valid prior snapshot -- a real, recurring case for a
season's earliest played matches, which predate that competition/season's
very first backtest reference date. `matches/played.html` renders a "sem
modelo disponível ainda" placeholder for these instead of a probability
grid.

`data/params.json` -- the shared model parameters an upcoming-fixture or
free-pick card is computed from, entirely in the browser (a played card
uses its own embedded `params` above instead). `model` selects which
`window.ScoreModels` entry (`assets/js/score_models.js`) interprets
`shared`/`teams` -- see "Scoreline math" above:

```json
{
  "reference_date": "2026-07-09",
  "model": "poisson_home",
  "shared": { "eta": 0.021, "beta_home": 0.318, "rho": 0.026 },
  "teams": { "Flamengo / RJ": { "attack": 0.41, "defense": 0.18 }, "...": "..." }
}
```

The free-pick builder's team rosters (names/crests/colors, not strengths)
are read from the already-existing `data/<slug>/<season>.json` files above,
filtered to teams present in `params.json`'s `teams` dict.
