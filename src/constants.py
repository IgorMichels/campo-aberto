"""Default parameter values shared by the fitting and simulation CLIs
(src.models.fit, src.simulation.run, src.pipeline).
"""

from datetime import datetime
import glob

DEFAULT_MATCHES_PATH = "data/processed/brazil/matches.csv"
# Competition configs are per-season: the REC changes year to year (e.g. Serie
# A's extra pre-Libertadores slot in 2025, Serie B's access playoff introduced
# in 2026 -- see configs/serie_*_2025.yaml vs configs/serie_*_2026.yaml). Pass
# --configs explicitly when simulating/backtesting a season other than the
# current default.
#
# DEFAULT_CONFIGS is derived from DEFAULT_SEASON rather than hardcoded, so it
# never needs a manual bump on its own -- but it degrades to an empty list if
# next year's configs/serie_*_<year>.yaml files haven't been authored yet.
# That's a real, expected manual step each year (the yaml encodes that
# season's competition rules, e.g. the REC changes above, which aren't
# inferable from anything) rather than something this glob can paper over by
# pointing at stale files.
DEFAULT_SEASON = datetime.now().year
DEFAULT_CONFIGS = sorted(glob.glob(f"configs/serie_*_{DEFAULT_SEASON}.yaml"))

DEFAULT_CHAINS = 4
DEFAULT_ITER_WARMUP = 1500
DEFAULT_N_DRAWS = 10000  # Monte Carlo simulation replicates (also drives Stan's iter_sampling)
DEFAULT_SEED = 0

# Time decay applied to each match's game_weight, based on complete weeks elapsed
# between the match and the fit's reference date: weight = 0.5 ** (weeks_ago /
# DEFAULT_HALF_LIFE_WEEKS). Matches older than DEFAULT_MAX_WEEKS_AGO are dropped
# from the Stan data entirely.
#
# Tuned via src.models.hyperparameter_sweep's poisson_home coordinate sweep
# (plans/hyperparameter_quality_sweep.md, 2026-07-19/20): half_life=52,
# window=182 was the winning combination pooled over 2022-2026, Brier 0.6171
# vs. 0.6200 at the previous defaults (25/100).
DEFAULT_HALF_LIFE_WEEKS = 52
DEFAULT_MAX_WEEKS_AGO = 182

# Seasons the site treats as real, browsable history (Confrontos "jogos
# passados", the evolution chart, standings, and the model-stats page) --
# not just current + previous year. 2022 is the floor because it's the
# first season with a genuine 2-year training window available (data back
# to 2020), matching src.models.backtest's walk-forward evaluation start.
SITE_SEASONS = [*range(2022, DEFAULT_SEASON + 1)]

SAMPLES_DIR = "data/samples"  # posterior attack/defense draws, saved by src.models.fit
RESULTS_DIR = "data/results"  # per-competition spot probabilities, saved by src.simulation.run
BACKTEST_CACHE_DIR = (
    "data/backtest_cache"  # per-checkpoint scored records, saved by src.models.backtest
)

CLUB_INFOS_PATH = (
    "data/assets/club_infos.csv"  # colors + crest_path per team, read by src.site.export_site_data
)
SITE_DIR = "site"  # deployable static site, written by src.site.export_site_data
