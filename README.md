# campo-aberto

Forecasts Brazilian football competitions (Campeonato Brasileiro Serie A and
Serie B) end to end: scrapes official CBF match results, fits a Bayesian
team-strength model on them, and Monte Carlo simulates the rest of the season
to report title / continental-berth / promotion / relegation probabilities
per club.

```
CBF dockets  →  data/raw/brazil/*.csv  →  data/processed/brazil/matches.csv  →  Stan fit  →  Monte Carlo simulation  →  probabilities
   (scrape)         raw, no name          treated, mapped names              (poisson_home.stan)   (configs/*.yaml)
                      treatment
```

## Project layout

```
src/
  ingestion/brazil/          scrapes CBF dockets, builds the treated matches CSV
  models/                   Dixon-Coles-adjusted Poisson model (Stan) fit on matches.csv
  simulation/                phase-based competition simulator + YAML competition configs
  pipeline.py                 orchestrates all three stages end to end
configs/                     one YAML per competition (Serie A, Serie B) -- see configs/README.md
data/
  raw/brazil/                 raw scraped dockets, gitignored (reproducible from CBF)
  processed/brazil/           matches.csv + team-name mapping (tracked -- see its own README)
  samples/brazil/              dated posterior-samples snapshots, gitignored
```

## Running it

The whole pipeline, from the repo root:

```bash
python -m src.pipeline
```

Or any stage on its own:

```bash
# 1. scrape + rebuild the treated dataset
python -m src.ingestion.brazil.run_pipeline

# 2. fit the model and print each club's posterior attack/defense strength
python -m src.models.fit

# 3. simulate the rest of the season as of a given date and report probabilities
python -m src.simulation.run --reference-date 2026-07-01
```

Running step 3 for a range of past `--reference-date` values is how you
track how these probabilities evolved over a season.

## The model

`src/models/poisson_home.stan` fits one club-strength model across every
match in `matches.csv` (Serie A and Serie B jointly, so a club's strength
estimate carries over across promotion/relegation and between seasons): each
match's home/away goals are two Poisson draws whose rates come from the two
clubs' attack/defense parameters plus a home-advantage term, with a
Dixon-Coles low-score correlation correction (`rho`) for the 0-0/1-0/0-1/1-1
cells. See `src/models/data.py` for the expected CSV schema and
`src/models/fit.py` for fitting/posterior-summary helpers.

## The simulator

A competition is a sequence of **phases** (`round_robin` and/or `playoff`)
defined declaratively in a YAML config under `configs/` -- Serie A is one
round-robin phase; Serie B is a round-robin phase feeding a two-legged
access-playoff phase. `src/simulation/simulate.py`'s `simulate_competition`
runs any such config: it draws posterior team-strength samples once, plays
out each phase's remaining fixtures with vectorized Dixon-Coles sampling,
applies the CBF tiebreak rules (`src/simulation/standings.py`), and reports
what share of Monte Carlo replicates each club lands in each declared "spot"
(title, relegation, promotion route, etc).

The schema supports shapes beyond Serie A/B (grouped round-robins, arbitrary
knockout brackets, cross-group pooling like "best 8 third-placed teams") that
aren't configured yet -- see `configs/README.md` for the full reference and
sketches for Copa do Brasil / Libertadores / World-Cup-style competitions.

## Development

```bash
uv sync              # installs deps, including the dev group (pytest, ruff)
ruff check .          # lint
pytest                # tests
```
