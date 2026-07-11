"""A deliberately minimal, non-production ModelAdapter, used only to prove
src.simulation.simulate's orchestration (round-robin/playoff/cascade/
standings) and the fit/export boundary are genuinely model-agnostic --
shaped nothing like PoissonHomeAdapter on purpose (one team param, one
shared param, plain independent Poisson, no Dixon-Coles-style low-score
correction). Never registered in src.models.registry.MODEL_REGISTRY.
"""

import numpy as np


class DummyAdapter:
    name = "dummy"
    stan_file = (
        "unused.stan"  # never actually fit in tests -- team_params/shared_params are hand-built
    )
    team_param_names = ("skill",)
    shared_param_names = ("home_boost",)

    def sample_scores(self, team_params, shared_params, home_idx, away_idx, rng):
        skill = team_params["skill"]
        home_boost = shared_params["home_boost"]
        mu_home = np.exp(skill[:, home_idx] - skill[:, away_idx] + home_boost[:, None])
        mu_away = np.exp(skill[:, away_idx] - skill[:, home_idx])
        return rng.poisson(mu_home), rng.poisson(mu_away)

    def sample_scores_single(self, team_params, shared_params, home_idx, away_idx, rng):
        skill = team_params["skill"]
        home_boost = shared_params["home_boost"]
        row = np.arange(skill.shape[0])
        mu_home = np.exp(skill[row, home_idx] - skill[row, away_idx] + home_boost)
        mu_away = np.exp(skill[row, away_idx] - skill[row, home_idx])
        return rng.poisson(mu_home), rng.poisson(mu_away)


ADAPTER = DummyAdapter()
