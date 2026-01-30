# pyderivatives/global_pricer/models/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Optional
import numpy as np


@dataclass
class FitResult:
    params: Dict[str, float]
    success: bool
    info: Optional[Dict[str, Any]] = None


class GlobalModel(Protocol):
    name: str

    def fit(
        self,
        K_obs: np.ndarray,
        T_obs: np.ndarray,
        C_obs: np.ndarray,
        x0: Dict[str, float],
        bounds: Dict[str, Dict[str, float]] | None = None,
        max_nfev: int = 200,
        **kwargs,
    ) -> FitResult: ...

    def price_surface(
        self,
        K_grid: np.ndarray,         # shape (nK,)
        T_grid: np.ndarray,         # shape (nT,)
        params: Dict[str, float],
        **kwargs,
    ) -> np.ndarray:                # shape (nT, nK)
        ...
