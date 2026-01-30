from __future__ import annotations

from typing import Callable, Dict, List

# signature convention:
#   model_fn(curve: YieldCurve, **kwargs) -> pandas.DataFrame or (DataFrame, DataFrame)
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    """
    Decorator to register a model function under a string key.
    """
    def deco(fn: Callable):
        key = name.strip().lower()
        if key in MODEL_REGISTRY:
            raise ValueError(f"Model '{key}' already registered.")
        MODEL_REGISTRY[key] = fn
        return fn
    return deco


def get_model(name: str) -> Callable:
    key = name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY.keys())


# Import models so they auto-register when registry is imported.
# Add new models by creating a file in models/ and importing it here.
from .models import nelson_siegel  # noqa: F401
from .models import svensson       # noqa: F401
from .models import Rezende_2011       # noqa: F401


