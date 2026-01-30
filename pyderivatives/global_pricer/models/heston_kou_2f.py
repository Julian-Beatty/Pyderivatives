from __future__ import annotations
import numpy as np

from .base import GlobalModel, FitResult
from ..registry import register_model
from .heston_kou_2f_core import HestonKou2FCore, HK2FParams


@register_model("heston_kou_2f")
class HestonKou2FModel(GlobalModel):
    def __init__(self, *, S0: float, r: float, q: float = 0.0, Umax: float = 200.0, n_quad: int = 96):
        self.core = HestonKou2FCore(S0=S0, r=r, q=q)
        self.Umax = float(Umax)
        self.n_quad = int(n_quad)

    def fit(self, K_obs, T_obs, C_obs, x0: dict | None = None, bounds=None, max_nfev: int = 200, **kwargs) -> FitResult:
        approx_order = int(kwargs.get("approx_order", 0))
        params = self.core.fit_to_calls(
            K_obs=np.asarray(K_obs, float).ravel(),
            T_obs=np.asarray(T_obs, float).ravel(),
            C_obs=np.asarray(C_obs, float).ravel(),
            x0=x0,
            bounds=bounds,
            Umax=self.Umax,
            n_quad=self.n_quad,
            max_nfev=int(max_nfev),
            approx_order=approx_order,
        )
        return FitResult(params=params, success=True)

    def call_prices(self, K: np.ndarray, T: float, params: HK2FParams | dict, **kwargs) -> np.ndarray:
        approx_order = int(kwargs.get("approx_order", 0))
        if isinstance(params, dict):
            params = HK2FParams.from_dict(params)
        return self.core.call_prices(np.asarray(K, float).ravel(), float(T), params, Umax=self.Umax, n_quad=self.n_quad, approx_order=approx_order)

    def price_surface(self, K_grid: np.ndarray, T_grid: np.ndarray, params: dict | HK2FParams, **kwargs) -> np.ndarray:
        approx_order = int(kwargs.get("approx_order", 0))
        if isinstance(params, HK2FParams):
            p = params
        else:
            p = HK2FParams.from_dict(params)
        return self.core.call_surface(K_grid, T_grid, p, Umax=self.Umax, n_quad=self.n_quad, approx_order=approx_order)
