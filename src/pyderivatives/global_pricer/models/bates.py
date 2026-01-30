from __future__ import annotations
import numpy as np

from .base import GlobalModel, FitResult
from ..registry import register_model
from .bates_core import BatesCore, BatesParams  # default bounds handled inside core


@register_model("bates")
class BatesModel(GlobalModel):
    def __init__(
        self,
        *,
        S0: float,
        r: float,
        q: float = 0.0,
        Umax: float = 200.0,
        n_quad: int = 96,
    ):
        self.core = BatesCore(S0=S0, r=r, q=q)
        self.Umax = float(Umax)
        self.n_quad = int(n_quad)

    def fit(
        self,
        K_obs,
        T_obs,
        C_obs,
        x0: dict | None = None,     # <- allow None
        bounds=None,                # <- allow None
        max_nfev: int = 200,
        **kwargs,
    ):
        # Do NOT set defaults here; core handles x0=None and bounds=None
        params = self.core.fit_to_calls(
            K_obs=np.asarray(K_obs),
            T_obs=np.asarray(T_obs),
            C_obs=np.asarray(C_obs),
            x0=x0,                   # can be None
            bounds=bounds,           # can be None
            Umax=self.Umax,
            n_quad=self.n_quad,
            max_nfev=int(max_nfev),
        )
        return FitResult(params=params, success=True)

    def call_prices(self, K, T, params, **kwargs):
        if isinstance(params, dict):
            params = BatesParams.from_dict(params)
        return self.core.call_prices(K, T, params, Umax=self.Umax, n_quad=self.n_quad)

    def price_surface(self, K_grid, T_grid, params, **kwargs):
        # params might come in as BatesParams or dict depending on your pipeline
        if isinstance(params, BatesParams):
            p = params
        else:
            p = BatesParams.from_dict(params)

        K_grid = np.asarray(K_grid, float).ravel()
        T_grid = np.asarray(T_grid, float).ravel()

        out = np.zeros((T_grid.size, K_grid.size), dtype=float)
        for i, T in enumerate(T_grid):
            out[i] = self.core.call_prices(K_grid, float(T), p, Umax=self.Umax, n_quad=self.n_quad)
        return out
