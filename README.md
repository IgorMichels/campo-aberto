# campo-aberto

Forecasts Brazilian football competitions (Campeonato Brasileiro Serie A and
Serie B) end to end: scrapes official CBF match results, fits a Bayesian
team-strength model on them, and Monte Carlo simulates the rest of the season
to report title / continental-berth / promotion / relegation probabilities
per club.

## Development

```bash
uv sync              # installs deps, including the dev group (pytest, ruff)
ruff check .          # lint
pytest                # tests
```
