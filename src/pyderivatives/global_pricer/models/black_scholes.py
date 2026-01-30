from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares

from .base import GlobalModel, FitResult
from ..registry import register_model


def _bs_call_price_vec(S0: float, K: np.ndarray, T: np.ndarray, r: float, q: float, sigma: float) -> np.ndarray:
    K = np.asarray(K, float)
    T = np.asarray(T, float)

    # intrinsic for T<=0
    out = np.maximum(S0 - K, 0.0)
    m = (T > 0) & (K > 0) & (S0 > 0) & (sigma > 0)
    if not np.any(m):
        return out

    Km = K[m]
    Tm = T[m]

    vol_sqrtT = sigma * np.sqrt(Tm)
    d1 = (np.log(S0 / Km) + (r - q + 0.5 * sigma * sigma) * Tm) / vol_sqrtT
    d2 = d1 - vol_sqrtT

    out[m] = S0 * np.exp(-q * Tm) * norm.cdf(d1) - Km * np.exp(-r * Tm) * norm.cdf(d2)
    return out


def _bs_arbitrage_bounds(S0: float, K: np.ndarray, T: np.ndarray, r: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    lower = np.maximum(S0 * disc_q - K * disc_r, 0.0)
    upper = S0 * disc_q
    return lower, upper


@dataclass(frozen=True)
class BSParams:
    sigma: float

    def to_dict(self) -> Dict[str, float]:
        return {"sigma": float(self.sigma)}


def default_bs_bounds() -> tuple[Dict[str, float], Dict[str, float]]:
    # You can tighten later; keep wide but sane for equity/crypto.
    return ({"sigma": 1e-4}, {"sigma": 6.0})


@register_model("black_scholes")
class BlackScholesModel(GlobalModel):
    """
    Global Black–Scholes with ONE volatility sigma across all quotes.

    Inputs are sparse quotes (K_obs, T_obs, C_obs). We fit sigma by least squares.
    """
    def __init__(self, *, S0: float, r: float, q: float = 0.0, **kwargs):
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)

    def fit(
        self,
        *,
        K_obs,
        T_obs,
        C_obs,
        x0: Dict[str, float],
        bounds: Optional[tuple[Dict[str, float], Dict[str, float]]] = None,
        max_nfev: int = 200,
        **kwargs,
    ) -> FitResult:
        K_obs = np.asarray(K_obs, float).ravel()
        T_obs = np.asarray(T_obs, float).ravel()
        C_obs = np.asarray(C_obs, float).ravel()
        if not (K_obs.size == T_obs.size == C_obs.size):
            raise ValueError("K_obs, T_obs, C_obs must have same length.")

        m = np.isfinite(K_obs) & np.isfinite(T_obs) & np.isfinite(C_obs) & (K_obs > 0) & (T_obs > 0) & (C_obs >= 0)
        K_obs, T_obs, C_obs = K_obs[m], T_obs[m], C_obs[m]
        if K_obs.size == 0:
            return FitResult(params=BSParams(sigma=float("nan")), success=False)

        # clip observations into no-arb bounds (optional but helps stability)
        lbC, ubC = _bs_arbitrage_bounds(self.S0, K_obs, T_obs, self.r, self.q)
        C_obs = np.minimum(np.maximum(C_obs, lbC), ubC)

        if bounds is None:
            bounds = default_bs_bounds()
        lb_d, ub_d = bounds
        if not (isinstance(lb_d, dict) and isinstance(ub_d, dict)):
            raise TypeError("bounds must be (lb_dict, ub_dict).")
        if "sigma" not in lb_d or "sigma" not in ub_d:
            raise KeyError("bounds dicts must include key 'sigma'.")

        sigma0 = float(x0.get("sigma", 0.5))
        lb = float(lb_d["sigma"])
        ub = float(ub_d["sigma"])

        def residuals(x: np.ndarray) -> np.ndarray:
            sig = float(x[0])
            if not (np.isfinite(sig) and sig > 0):
                return np.full_like(C_obs, 1e6)
            C_hat = _bs_call_price_vec(self.S0, K_obs, T_obs, self.r, self.q, sig)
            return C_hat - C_obs

        res = least_squares(
            residuals,
            x0=np.array([sigma0], float),
            bounds=(np.array([lb], float), np.array([ub], float)),
            method="trf",
            max_nfev=int(max_nfev),
            ftol=1e-12, xtol=1e-12, gtol=1e-12,
        )
        p = BSParams(sigma=float(res.x[0]))
        return FitResult(params=p, success=bool(res.success))

    def call_prices(self, K: np.ndarray, T: float, params, **kwargs) -> np.ndarray:
        # params can be dict or BSParams
        if isinstance(params, dict):
            sigma = float(params["sigma"])
        else:
            sigma = float(getattr(params, "sigma"))
        K = np.asarray(K, float).ravel()
        T_vec = np.full_like(K, float(T), dtype=float)
        return _bs_call_price_vec(self.S0, K, T_vec, self.r, self.q, sigma)

    def price_surface(self, K_grid: np.ndarray, T_grid: np.ndarray, params: Dict[str, float], **kwargs) -> np.ndarray:
        K_grid = np.asarray(K_grid, float).ravel()
        T_grid = np.asarray(T_grid, float).ravel()
        out = np.zeros((T_grid.size, K_grid.size), float)
        sigma = float(params["sigma"])
        for i, T in enumerate(T_grid):
            out[i, :] = _bs_call_price_vec(self.S0, K_grid, np.full_like(K_grid, float(T)), self.r, self.q, sigma)
        return out
