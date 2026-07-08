# Site

This directory is the deployable output of the results website: a static
page (plain HTML/CSS/vanilla JS, no build step) rendered from the simulation
outputs under `data/results/` (git-ignored). Unlike `data/results/`, this
directory *is* tracked -- what's committed here is exactly what gets served.

A push to `main` that touches this directory triggers
`.github/workflows/deploy-site.yml`, which deploys it to GitHub Pages
(`actions/deploy-pages`, no `gh-pages` branch).

## Regenerating

After running `python -m src.pipeline`, regenerate this directory's data and
commit the result -- that commit is what actually triggers a new deploy,
since `data/results/` itself never reaches git or CI:

```bash
python -m src.site.export_site_data
```

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
         "standings": {"points": 79, "played": 38, "goals_for": 78, "goals_against": 27, "goal_diff": 51},
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
- `crest`/`color` are `site`-root-relative image path / hex color, used as-is.
