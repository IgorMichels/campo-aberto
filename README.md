# campo-aberto

Probabilities for the Campeonato Brasileiro (Serie A and Serie B): title,
continental berths, promotion, and relegation, updated round by round from a
Bayesian team-strength model.

[![Quality checks](https://github.com/IgorMichels/campo-aberto/actions/workflows/quality.yml/badge.svg)](https://github.com/IgorMichels/campo-aberto/actions/workflows/quality.yml)
[![Deploy site](https://github.com/IgorMichels/campo-aberto/actions/workflows/deploy-site.yml/badge.svg)](https://github.com/IgorMichels/campo-aberto/actions/workflows/deploy-site.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[igormichels.github.io/campo-aberto](https://igormichels.github.io/campo-aberto/)**

## What it does

Official CBF match results are scraped, fed into a Dixon-Coles-adjusted
Poisson model (fit in Stan) that estimates each club's attack/defense
strength, and the rest of the season is Monte Carlo simulated thousands of
times to report, per club: title odds, Libertadores/Sul-Americana
qualification, promotion, and relegation probabilities. The site also carries
a scoreline probability grid for every match, played or upcoming, and how
every probability evolved round by round.

## Quickstart

```bash
uv sync                # install dependencies
python -m src.pipeline # scrape, fit, simulate, export the site's data
```

For the model's internals, the simulator, data schemas, and how to run each
pipeline stage on its own, see [CODEBASE.md](CODEBASE.md).

## Repo map

- [`src/`](src) -- ingestion, model, simulation, pipeline (see [CODEBASE.md](CODEBASE.md))
- [`configs/`](configs) -- one YAML per competition ([configs/README.md](configs/README.md))
- [`site/`](site) -- the static results site deployed to GitHub Pages ([site/README.md](site/README.md))
- [`data/processed/brazil/`](data/processed/brazil) -- the treated match dataset ([its README](data/processed/brazil/README.md))

## License

[MIT](LICENSE)
