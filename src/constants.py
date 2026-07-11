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
DEFAULT_HALF_LIFE_WEEKS = 25
DEFAULT_MAX_WEEKS_AGO = 100

SAMPLES_DIR = "data/samples"  # posterior attack/defense draws, saved by src.models.fit
RESULTS_DIR = "data/results"  # per-competition spot probabilities, saved by src.simulation.run

CLUB_INFOS_PATH = (
    "data/assets/club_infos.csv"  # colors + crest_path per team, read by src.site.export_site_data
)
SITE_DIR = "site"  # deployable static site, written by src.site.export_site_data
