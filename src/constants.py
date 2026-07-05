"""Default parameter values shared by the fitting and simulation CLIs
(src.models.fit, src.simulation.run, src.pipeline).
"""

DEFAULT_MATCHES_PATH = "data/processed/brazil/matches.csv"

# Time decay applied to each match's game_weight, based on complete weeks elapsed
# between the match and the fit's reference date: weight = 0.5 ** (weeks_ago /
# DEFAULT_HALF_LIFE_WEEKS). Matches older than DEFAULT_MAX_WEEKS_AGO are dropped
# from the Stan data entirely.
DEFAULT_HALF_LIFE_WEEKS = 25
DEFAULT_MAX_WEEKS_AGO = 100

SAMPLES_DIR = "data/samples"  # posterior attack/defense draws, saved by src.models.fit
