from __future__ import annotations
import numpy as np

from .base import GlobalModel, FitResult
from ..registry import register_model
from .heston_kou_core import HKDEModel, HKDEParams  # (no need to import default_hkde_bounds here)


@register_model("heston_kou")
class HestonKouModel(GlobalModel):
    def __init__(self, *, S0: float, r: float, q: float = 0.0, Umax: float = 200.0, n_quad: int = 96):
        self.core = HKDEModel(S0=S0, r=r, q=q)
        self.Umax = float(Umax)
        self.n_quad = int(n_quad)

    def fit(self, K_obs, T_obs, C_obs, x0: dict | None = None, bounds=None, max_nfev: int = 200, **kwargs) -> FitResult:
        # Let core handle defaults: x0=None and bounds=None
        params = self.core.fit_to_calls(
            K_obs=np.asarray(K_obs, float).ravel(),
            T_obs=np.asarray(T_obs, float).ravel(),
            C_obs=np.asarray(C_obs, float).ravel(),
            x0=x0,
            bounds=bounds,
            Umax=self.Umax,
            n_quad=self.n_quad,
            max_nfev=int(max_nfev),
        )
        return FitResult(params=params, success=True)

    def call_prices(self, K: np.ndarray, T: float, params: HKDEParams | dict, **kwargs) -> np.ndarray:
        if isinstance(params, dict):
            params = HKDEParams.from_dict(params)
        K = np.asarray(K, float).ravel()
        return self.core.call_prices(K, float(T), params, Umax=self.Umax, n_quad=self.n_quad)

    def price_surface(self, K_grid: np.ndarray, T_grid: np.ndarray, params: dict | HKDEParams, **kwargs) -> np.ndarray:
        if isinstance(params, HKDEParams):
            p = params
        else:
            p = HKDEParams.from_dict(params)

        K_grid = np.asarray(K_grid, float).ravel()
        T_grid = np.asarray(T_grid, float).ravel()

        out = np.zeros((T_grid.size, K_grid.size), float)
        for i, t in enumerate(T_grid):
            out[i, :] = self.core.call_prices(K_grid, float(t), p, Umax=self.Umax, n_quad=self.n_quad)
        return out
