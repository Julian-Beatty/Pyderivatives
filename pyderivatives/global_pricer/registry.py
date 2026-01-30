# pyderivatives/global_pricer/registry.py
from __future__ import annotations

# expose registry functions
from .registration import register_model, get_model, available_models  # noqa: F401

# trigger model registration (imports modules with decorators)
from . import models  # noqa: F401
