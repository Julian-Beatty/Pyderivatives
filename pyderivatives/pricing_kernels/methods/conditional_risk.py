from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..base import MeasureTransform
from ..config import CacheSpec, ConditionalRiskSpec, KeySpec
from ..registry import register_transform
from ..utils import (
    _as_1d,
    _cdf_from_density,
    _find_sigma,
    _safe_interp,
    _trapz_normalize_density,
)
@dataclass
class ConditionalRiskFitted:
    theta_hat: np.ndarray
    spec: ConditionalRiskSpec
    T: float
    loss: float
    success: bool
    message: str
    
@register_transform("conditional_risk")
class ConditionalRiskKernel(MeasureTransform):
    """
    Schreindorfer-style conditional-risk pricing kernel.

    Parameterization:

        M(r, sigma; theta)
            = exp(delta_t + sum_i c_i(sigma) r^i)

        c_i(sigma)
            = c_i / sigma^(b*i)

    where delta_t is solved per observation so that:

        ∫ M(r,sigma) f_Q(r) dr = 1.
    """

    def __init__(
        self,
        *,
        spec: ConditionalRiskSpec = ConditionalRiskSpec(),
        maxiter: int = 400,
        x0: Optional[np.ndarray] = None,
        key_spec: KeySpec = KeySpec(),
        fit_trim_alpha: Optional[Tuple[float, float]] = None,
        min_obs: int = 30,
        fit_maturities: Optional[List[float]] = None,
        maturity_match_tol: Optional[float] = None,
        eps: float = 1e-10,
        verbose: bool = True,
        penalty_value: float = 1e100,
        cache_spec: CacheSpec = CacheSpec(),
    ):
        super().__init__(
            key_spec=key_spec,
            fit_trim_alpha=fit_trim_alpha,
            min_obs=min_obs,
            fit_maturities=fit_maturities,
            maturity_match_tol=maturity_match_tol,
            eps=eps,
            verbose=verbose,
            penalty_value=penalty_value,
            cache_spec=cache_spec,
        )
        self.spec = spec
        self.maxiter = int(maxiter)
        self.x0 = None if x0 is None else _as_1d(x0)

    def _cache_params(self) -> dict:
        return {"spec": self.spec, "maxiter": self.maxiter, "x0": self.x0}

    def _theta_dim(self) -> int:
        # theta = [b, c1, c2, ..., cN]
        return int(self.spec.N) + 1

    def _unpack_theta(self, theta: np.ndarray):
        theta = _as_1d(theta)
        b = float(theta[0])
        c = theta[1:]
        return b, c

    def _bounds(self):
        b_lo, b_hi = self.spec.b_bounds
        c_lo, c_hi = self.spec.c_bounds

        bounds = [(float(b_lo), float(b_hi))]
        bounds.extend([(float(c_lo), float(c_hi)) for _ in range(int(self.spec.N))])
        return bounds

    def _g_no_delta(self, r: np.ndarray, sigma: float, theta: np.ndarray) -> np.ndarray:
        b, c = self._unpack_theta(theta)
        r = _as_1d(r)

        g = np.zeros_like(r, dtype=float)
        sigma = max(float(sigma), self.eps)

        for i in range(1, int(self.spec.N) + 1):
            coeff = c[i - 1] / (sigma ** (b * i))
            g += coeff * (r ** i)

        return g

    def _solve_delta(self, x_grid, f_q, sigma, theta):
        """
        Solve delta_t so that:

            ∫ exp(delta + g(r)) f_Q(r) dr = 1.

        Therefore:

            delta = -log ∫ exp(g(r)) f_Q(r) dr
        """
        g0 = self._g_no_delta(x_grid, sigma, theta)
        mass = np.trapezoid(
            np.exp(np.clip(g0, -700, 700)) * f_q,
            x_grid,
        )

        if not np.isfinite(mass) or mass <= self.eps:
            return np.nan

        return float(-np.log(mass))

    def _kernel(self, x_grid, f_q, sigma, theta):
        delta = self._solve_delta(x_grid, f_q, sigma, theta)
        if not np.isfinite(delta):
            return None, None

        g0 = self._g_no_delta(x_grid, sigma, theta)
        g = delta + g0

        M = np.exp(np.clip(g, -700, 700))
        return M, delta

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float):
        p = self._theta_dim()
        x0 = np.zeros(p, dtype=float) if self.x0 is None else self.x0.copy()
        if x0.size != p:
            raise ValueError(f"x0 has length {x0.size}, expected {p}.")

        bounds = self._bounds()
        eps = self.eps
        penalty = self.penalty_value

        def nll(theta):
            theta = _as_1d(theta)

            if theta.size != p or not np.all(np.isfinite(theta)):
                return penalty

            total_ll = 0.0

            for _, row in hist_T.iterrows():
                x_grid = _as_1d(row["x_grid"])
                f_q = _trapz_normalize_density(x_grid, row["f_q"], eps=eps)

                sigma = float(row.get("sigma", 1.0))
                realized = float(row["realized_return"])

                if not np.all(np.isfinite(f_q)):
                    return penalty

                M, delta = self._kernel(x_grid, f_q, sigma, theta)
                if M is None:
                    return penalty

                f_p = _trapz_normalize_density(
                    x_grid,
                    f_q / np.maximum(M, eps),
                    eps=eps,
                )

                if not np.all(np.isfinite(f_p)):
                    return penalty

                val = _safe_interp(realized, x_grid, f_p)
                if not np.isfinite(val) or val <= eps:
                    return penalty

                total_ll += np.log(val)

            out = -float(total_ll)
            return out if np.isfinite(out) else penalty

        res = minimize(
            nll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter},
        )

        fitted = ConditionalRiskFitted(
            theta_hat=np.asarray(res.x, dtype=float),
            spec=self.spec,
            T=float(T),
            loss=float(res.fun),
            success=bool(res.success),
            message=str(res.message),
        )

        if res.success:
            status = "success"
        
        elif (
            np.all(np.isfinite(res.x))
            and np.isfinite(res.fun)
        ):
            status = "questionable"
        
        else:
            status = "failed"
        
        diag = {
            "loss": float(res.fun),
            "loss_name": "negative_log_likelihood",
            "status": status,
            "message": str(res.message),
            "params": {
                "b": float(res.x[0]),
                **{f"c{i}": float(v) for i, v in enumerate(res.x[1:], start=1)},
            },
        }

        return fitted, diag

    def _transform_surface_with_model(
        self,
        fitted_model,
        x_grid,
        f_q,
        F_q,
        *,
        T,
        info,
    ):
        x_grid = _as_1d(x_grid)
        f_q = _trapz_normalize_density(x_grid, f_q, eps=self.eps)
        F_q = _as_1d(F_q)

        sigma = _find_sigma(info, self.key_spec.sigma_keys, default=1.0)

        M, delta = self._kernel(
            x_grid,
            f_q,
            sigma,
            fitted_model.theta_hat,
        )

        if M is None:
            raise RuntimeError("Failed to compute pricing kernel.")

        f_p_raw = f_q / np.maximum(M, self.eps)
        f_p = _trapz_normalize_density(
            x_grid,
            f_p_raw,
            eps=self.eps,
        )
        
        # recompute kernel so it is internally consistent
        M = f_q / np.maximum(f_p, self.eps)

        F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)
        weight = f_p / np.maximum(f_q, self.eps)

        return {
            "x_grid": x_grid,
            "f_q": f_q,
            "F_q": F_q,
            "f_p": f_p,
            "F_p": F_p,
            "weight": weight,
            "pricing_kernel": M,
            "delta": delta,
            "theta_hat": fitted_model.theta_hat,
            "T_fit": float(T),
        }

