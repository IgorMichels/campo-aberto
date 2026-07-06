"""Default parameter values shared by the fitting and simulation CLIs
(src.models.fit, src.simulation.run, src.pipeline).
"""

DEFAULT_MATCHES_PATH = "data/processed/brazil/matches.csv"
DEFAULT_CONFIGS = ["configs/serie_a.yaml", "configs/serie_b.yaml"]
DEFAULT_SEASON = 2026

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
