"""The one interface every candidate scoring model implements in Python.

Threading `attack`/`defense`/`eta`/`beta_home`/`rho` by literal name through
src/simulation/simulate.py's round-robin/playoff/cascade orchestration would
couple that orchestration to one specific model's parameterization. A
ModelAdapter is the only place a model's parameter names and score-sampling
math live; everything else in src/simulation only ever calls the two methods
below, on whichever adapter src/models/registry.py resolves.
"""

from typing import Protocol

import numpy as np


class ModelAdapter(Protocol):
    name: str
    stan_file: str
    team_param_names: tuple[str, ...]
    shared_param_names: tuple[str, ...]

    def sample_scores(
        self,
        team_params: dict[str, np.ndarray],
        shared_params: dict[str, np.ndarray],
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched score sampling for a fixture list shared across every draw.

        team_params[name]: shape (n_draws, T). shared_params[name]: shape
        (n_draws,). home_idx/away_idx: shape (n_matches,). Returns
        (home_goals, away_goals), each shape (n_draws, n_matches), int.
        """
        ...

    def sample_scores_single(
        self,
        team_params: dict[str, np.ndarray],
        shared_params: dict[str, np.ndarray],
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-draw score sampling for one match per draw (playoffs, where who
        plays whom can differ by draw).

        home_idx/away_idx: shape (n_draws,). Returns (home_goals, away_goals),
        each shape (n_draws,), int.
        """
        ...
