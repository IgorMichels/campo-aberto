# campo-aberto

Forecasts Brazilian football competitions (Campeonato Brasileiro Serie A and
Serie B) end to end: scrapes official CBF match results, fits a Bayesian
team-strength model on them, and Monte Carlo simulates the rest of the season
to report title / continental-berth / promotion / relegation probabilities
per club.

## Project layout

```
src/
  ingestion/brazil/          scrapes CBF dockets, builds the treated matches CSV
  models/                   Dixon-Coles-adjusted Poisson model (Stan) fit on matches.csv
data/
  raw/brazil/                 raw scraped dockets, gitignored (reproducible from CBF)
  processed/brazil/           matches.csv + team-name mapping (tracked -- see its own README)
  samples/brazil/              dated posterior-samples snapshots, gitignored
```

## Running it

```bash
# 1. scrape + rebuild the treated dataset
python -m src.ingestion.brazil.run_pipeline

# 2. fit the model and print each club's posterior attack/defense strength
python -m src.models.fit
```

## The model

`src/models/poisson_home.stan` fits one club-strength model across every
match in `matches.csv` (Serie A and Serie B jointly, so a club's strength
estimate carries over across promotion/relegation and between seasons): each
match's home/away goals are two Poisson draws whose rates come from the two
clubs' attack/defense parameters plus a home-advantage term, with a
Dixon-Coles low-score correlation correction (`rho`) for the 0-0/1-0/0-1/1-1
cells. See `src/models/data.py` for the expected CSV schema and
`src/models/fit.py` for fitting/posterior-summary helpers.

## Development

```bash
uv sync              # installs deps, including the dev group (pytest, ruff)
ruff check .          # lint
pytest                # tests
```
