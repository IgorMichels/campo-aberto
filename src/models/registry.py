"""The single place a model name resolves to its adapter.

Adding candidate model N+1 to the tournament (see plans/model_stats_page.md
Step 0) means: one new src/models/stan_models/<name>.stan file, one new
src/models/adapters/<name>.py implementing ModelAdapter, and one line here.
Nothing else in src/simulation or src/site needs to change.
"""

from src.models.adapter import ModelAdapter
from src.models.adapters.poisson_home import ADAPTER as _poisson_home
from src.models.adapters.poisson_home_no_rho import ADAPTER as _poisson_home_no_rho
from src.models.adapters.poisson_strength import ADAPTER as _poisson_strength

MODEL_REGISTRY: dict[str, ModelAdapter] = {
    "poisson_home": _poisson_home,
    "poisson_strength": _poisson_strength,
    "poisson_home_no_rho": _poisson_home_no_rho,
}

DEFAULT_MODEL = "poisson_home"
