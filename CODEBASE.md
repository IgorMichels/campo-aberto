# Codebase

Technical reference for how campo-aberto is built -- pipeline stages, data
schemas, the model tournament, the simulator, the site, and how CI is wired
up. For the project pitch, the live site link, and a quickstart, see
[README.md](README.md).

Forecasts Brazilian football competitions (Campeonato Brasileiro Serie A and
Serie B) end to end: scrapes official CBF match results, fits a Bayesian
team-strength model on them, Monte Carlo simulates the rest of the season to
report title / continental-berth / promotion / relegation probabilities per
club, and exports everything into the static results site.

```
CBF dockets  →  matches.csv  →  Stan fit  →  Monte Carlo simulation  →  site data export
  (scrape)      (treated,       (poisson_home.stan, or any            (site/data/*.json,
                 mapped names)   other MODEL_REGISTRY entry)            model_stats.json)
```

## Project layout

```
src/
  ingestion/brazil/     scrapes CBF dockets + ESPN's schedule, builds the treated matches CSV
  models/                model tournament: registry, fitting, walk-forward backtesting, hyperparameter sweeps
    stan_models/          one .stan file per registered model
    adapters/              one ModelAdapter per registered model (adapter.py defines the protocol)
  simulation/             phase-based competition simulator + CBF tiebreak rules
  site/                   exports simulation results into the static site's committed data
  pipeline.py              orchestrates every stage end to end (see "Running it" below)
configs/                  one YAML per competition/season (Serie A, Serie B, 2022-2026) -- see configs/README.md
data/
  raw/brazil/              raw scraped dockets, gitignored (reproducible from CBF)
  processed/brazil/        matches.csv + team-name mapping (tracked -- see its own README)
  assets/                  club_infos.csv (colors/acronyms) + crests/ (tracked)
  samples/brazil/           dated posterior-samples snapshots, gitignored
  results/                  dated simulated spot-probability snapshots per competition/season, gitignored
  backtest_cache/            per-model walk-forward backtest checkpoint cache, gitignored
site/                     the static results site deployed to GitHub Pages -- see site/README.md
tests/                    mirrors src/ at the package level -- see "Tests" below
scripts/                  dev tooling (the pre-commit related-tests runner)
```

## Running it

The whole pipeline, from the repo root:

```bash
python -m src.pipeline
```

Five stages, in order (see `src/pipeline.py`'s docstring for the full
rationale of each): scrape + rebuild the treated dataset, fit a model on the
full history, simulate the rest of the season as of the latest
Monday/Friday checkpoint, export that into the site's committed data
(`site/data/*.json`), then recompute the model-statistics page's aggregate
metrics from the played-match data the previous step just wrote. Pass
`--model <name>` to run a model other than the default (`poisson_home`) --
see "The model tournament" below.

Or any stage on its own:

```bash
# 1. scrape + rebuild the treated dataset
python -m src.ingestion.brazil.run_pipeline

# 2. fit a model and print each club's posterior attack/defense strength
python -m src.models.fit --model poisson_home

# 3. simulate the rest of the season as of a given date and report probabilities
python -m src.simulation.run --reference-date 2026-07-01

# 3b. or backtest a full season on a fixed twice-weekly cadence instead of
# one date -- this is what the site's "Evolução" and model-statistics pages
# are built from
python -m src.simulation.run_rounds

# 4. export the site's committed data (after 2/3 have produced results)
python -m src.site.export_site_data
python -m src.site.export_matches_data
python -m src.site.model_stats
```

Running step 3 for a range of past `--reference-date` values (or step 3b
directly) is how you track how these probabilities evolved over a season.

## The model tournament

`src/models/registry.py`'s `MODEL_REGISTRY` maps a name to a `ModelAdapter`
(`src/models/adapter.py`) -- the only place a model's parameter names and
score-sampling math live, so `src/simulation` never couples to one specific
model's parameterization. Adding candidate model N+1 means one new
`src/models/stan_models/<name>.stan` file, one new
`src/models/adapters/<name>.py`, and one line in the registry; nothing else
changes. Seven models are registered today, `poisson_home` (`DEFAULT_MODEL`)
among them:

- `poisson_home`, `poisson_strength`, `poisson_home_no_rho` -- independent-Poisson variants
- `negbin_home`, `negbin_home_shared_phi` -- negative-binomial (overdispersed) variants
- `bivariate_poisson_home` -- correlated home/away goals via a shared latent term
- `hierarchical_home` -- hierarchical shrinkage on club-level parameters

`poisson_home` fits one club-strength model across every match in
`matches.csv` (Serie A and Serie B jointly, so a club's strength estimate
carries over across promotion/relegation and between seasons): each match's
home/away goals are two Poisson draws whose rates come from the two clubs'
attack/defense parameters plus a home-advantage term, with a Dixon-Coles
low-score correlation correction (`rho`) for the 0-0/1-0/0-1/1-1 cells. See
`src/models/data.py` for the expected CSV schema and `src/models/fit.py` for
fitting/posterior-summary helpers.

Two tools compare candidates against each other, both model-agnostic (they
go through `ModelAdapter.sample_scores_single`, not per-model code):

- `src/models/backtest.py` -- a walk-forward, out-of-sample backtest harness:
  fits on a fixed cadence of checkpoints and scores every match strictly
  between two consecutive checkpoints against the _earlier_ checkpoint's
  posterior, so no match is ever scored using a fit that already saw its own
  result. Results cache under `data/backtest_cache/<model>/`.
- `src/models/hyperparameter_sweep.py` -- sequential (coordinate-wise) sweep
  over the data-weighting and Stan-prior knobs that can plausibly change
  predictive quality (`half_life_weeks`, `window_weeks`, each model's own
  prior widths). Design doc: `plans/hyperparameter_quality_sweep.md`.

## The simulator

A competition is a sequence of **phases** (`round_robin` and/or `playoff`)
defined declaratively in a YAML config under `configs/` -- Serie A is one
round-robin phase; Serie B is a round-robin phase feeding a two-legged
access-playoff phase. `src/simulation/simulate.py`'s `simulate_competition`
runs any such config against any registered model's adapter: it draws
posterior team-strength samples once, plays out each phase's remaining
fixtures with vectorized Dixon-Coles sampling, applies the CBF tiebreak
rules (`src/simulation/standings.py`), and reports what share of Monte Carlo
replicates each club lands in each declared "spot" (title, relegation,
promotion route, etc). `src/simulation/run_rounds.py` wraps this into the
fixed twice-weekly (Monday/Friday) backtest cadence the site's history and
the model-statistics page are both built from -- see its own docstring for
how it decides which candidate dates actually produce a snapshot.

The config schema supports shapes beyond Serie A/B (grouped round-robins,
arbitrary knockout brackets, cross-group pooling like "best 8 third-placed
teams") that aren't configured yet -- see `configs/README.md` for the full
reference and sketches for Copa do Brasil / Libertadores / World-Cup-style
competitions.

## The site

`src/site/export_site_data.py` and `src/site/export_matches_data.py` turn
`data/results/*.csv` (gitignored, one snapshot per reference date) and
`data/processed/brazil/matches.csv` into the static site's committed data
(`site/data/*.json`); `src/site/model_stats.py` then computes the
model-statistics page's exact-scoreline/direction accuracy, Brier score, and
calibration from that same committed played-match data (porting
`site/assets/js/poisson_home.js`'s closed-form scoreline math to Python, so
the number the page shows and the number the client-side "sticker" cards
render agree exactly). `site/` itself -- plain HTML/CSS/vanilla JS, no build
step -- is what's actually deployed; see [site/README.md](site/README.md)
for its data schemas, the "Jogos" match-card system, and the exported-JSON
contract every page reads.

## Tests

`tests/` mirrors `src/` at the package (directory) level, not file-for-file:
`tests/ingestion/`, `tests/models/`, `tests/simulation/`, `tests/site/`.

```bash
pytest                # full suite
```

The `pytest-related` pre-commit hook (`scripts/run_related_tests.py`) scopes
this down automatically to just the test package(s) touched by whatever
`src/`/`tests/` files are staged, so a normal commit only re-runs what's
relevant -- CI (`quality.yml`) still runs the full suite on every PR.

## CI/CD

Two GitHub Actions workflows, under `.github/workflows/`:

- **`quality.yml`** ("Quality checks") -- every PR into `main`: `pre-commit
run --all-files` (ruff, prettier, the housekeeping hooks) plus the full
  `pytest` suite.
- **`deploy-site.yml`** ("Deploy site") -- every push to `main` that touches
  `site/`: publishes `site/` to GitHub Pages via `actions/deploy-pages`, no
  `gh-pages` branch involved.

## Development

```bash
uv sync              # installs deps, including the dev group (pytest, ruff, pre-commit)
pre-commit run --all-files   # lint + format checks (same as CI)
pytest                # tests
```
