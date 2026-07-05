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
data/
  raw/brazil/                 raw scraped dockets, gitignored (reproducible from CBF)
  processed/brazil/           matches.csv + team-name mapping (tracked -- see its own README)
```

## Running it

```bash
# scrape + rebuild the treated dataset
python -m src.ingestion.brazil.run_pipeline
```

## Development

```bash
uv sync              # installs deps, including the dev group (pytest, ruff)
ruff check .          # lint
pytest                # tests
```
