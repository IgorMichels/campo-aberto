"""Plain independent-Poisson score sampling (no low-score correlation
term), shared by every adapter whose Stan model has no rho/dispersion
correction -- poisson_home_no_rho and hierarchical_home today. This math
depends only on the two teams' scoring rates (mu_home, mu_away), never on
how those rates were computed or fit, so it lives here once.
"""

import numpy as np


def simulate_scores(mu_home, mu_away, rng):
    return rng.poisson(mu_home).astype(np.int64), rng.poisson(mu_away).astype(np.int64)
