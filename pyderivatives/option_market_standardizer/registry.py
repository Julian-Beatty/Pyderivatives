from __future__ import annotations

from typing import Callable, Dict
import pandas as pd

VendorAdapter = Callable[[pd.DataFrame], pd.DataFrame]

# Global registry for vendor adapters
VENDOR_REGISTRY: Dict[str, VendorAdapter] = {}


def register_vendor(name: str) -> Callable[[VendorAdapter], VendorAdapter]:
    """
    Decorator that registers an adapter under a vendor key.
    Usage:
        @register_vendor("optionmetrics")
        def adapt_optionmetrics(df): ...
    """
    key = name.strip().lower()

    def _decorator(fn: VendorAdapter) -> VendorAdapter:
        VENDOR_REGISTRY[key] = fn
        return fn

    return _decorator
