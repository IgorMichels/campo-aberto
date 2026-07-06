"""Saves a competition's simulated spot probabilities to a dated CSV.

Mirrors the data/samples/<country>/<date>.csv convention used by
src.models.fit.save_samples, but keyed by competition and season instead of
country, since a single country can run several competitions (e.g. Serie A
and Serie B) concurrently.
"""

import os

import pandas as pd

from src.constants import RESULTS_DIR


def save_results(
    result: pd.DataFrame,
    competition: str,
    season: int,
    reference_date: pd.Timestamp,
    results_dir: str = RESULTS_DIR,
) -> str:
    """Saves `result` (see simulate.simulate_competition) to a dated CSV.

    Returns:
        The path the results were saved to.
    """
    slug = competition.lower().replace(" ", "_")
    out_dir = os.path.join(results_dir, slug, str(season))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{reference_date.strftime('%Y_%m_%d')}.csv")
    result.to_csv(out_path, index=False)
    return out_path
