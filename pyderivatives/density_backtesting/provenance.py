# provenance.py

from __future__ import annotations

import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy


DENSITY_BACKTESTING_VERSION = "0.1.0"


def runtime_metadata():
    return {
        "density_backtesting_version": DENSITY_BACKTESTING_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }


def describe_model(model):
    out = {
        "class": model.__class__.__name__,
        "name": getattr(model, "name", None),
    }

    for attr in ("rnd_key", "physical_key", "requires_fit", "clone_transform"):
        if hasattr(model, attr):
            out[attr] = getattr(model, attr)

    transform = getattr(model, "transform", None)
    if transform is not None:
        out["transform_class"] = transform.__class__.__name__
        out["transform_method"] = getattr(transform, "method_name", None)

    return out


def describe_models(models):
    return [describe_model(m) for m in models]