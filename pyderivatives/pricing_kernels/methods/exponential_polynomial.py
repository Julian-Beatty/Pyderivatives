from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..base import MeasureTransform
from ..config import CacheSpec, KeySpec, ThetaSpec
from ..registry import register_transform
from ..utils import (
    _as_1d,
    _cdf_from_density,
    _find_sigma,
    _safe_interp,
    _trapz_normalize_density,
)
@dataclass
class ExponentialPolynomialFitted:
    theta_hat: np.ndarray
    theta_spec: ThetaSpec
    T: float
    loss: float
    success: bool
    message: str


@register_transform("exponential_polynomial")
class ExponentialPolynomialKernel(MeasureTransform):
    """
    Exponential-polynomial pricing-kernel transformation.

    Parameterization:
        M(r, sigma; theta) = exp(g(r, sigma; theta))

    Physical density:
        f_P(r) ∝ f_Q(r) / M(r, sigma; theta)
               = f_Q(r) exp(-g(r, sigma; theta))
    """

    def __init__(
        self,
        *,
        theta_spec: ThetaSpec = ThetaSpec(),
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
        self.theta_spec = theta_spec
        self.maxiter = int(maxiter)
        self.x0 = None if x0 is None else _as_1d(x0)

    def _cache_params(self) -> dict:
        return {"theta_spec": self.theta_spec, "maxiter": self.maxiter, "x0": self.x0}

    def _theta_dim(self) -> int:
        return int(self.theta_spec.N) * (int(self.theta_spec.Ksig) + 1)

    def _bounds(self):
        p = self._theta_dim()
        if self.theta_spec.bounds is None:
            return None
        lb, ub = self.theta_spec.bounds
        lb = _as_1d(lb)
        ub = _as_1d(ub)
        if lb.size != p or ub.size != p:
            raise ValueError("ThetaSpec.bounds must have lower and upper arrays of length N*(Ksig+1).")
        return list(zip(lb, ub))

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float) -> Tuple[ExponentialPolynomialFitted, Dict[str, Any]]:
        p = self._theta_dim()
        x0 = np.zeros(p, dtype=float) if self.x0 is None else self.x0.copy()
        if x0.size != p:
            raise ValueError(f"x0 has length {x0.size}, expected {p}.")

        bounds = self._bounds()
        N = int(self.theta_spec.N)
        Ksig = int(self.theta_spec.Ksig)
        eps = self.eps

        def nll(theta):
            theta = _as_1d(theta)
            total_ll = 0.0
            penalty = self.penalty_value

            if theta.size != p or not np.all(np.isfinite(theta)):
                return penalty

            for _, row in hist_T.iterrows():
                x_grid = _as_1d(row["x_grid"])

                # Invalid grid means this parameter vector/observation pair is unusable.
                if x_grid.size < 10 or np.any(~np.isfinite(x_grid)) or np.any(np.diff(x_grid) <= 0):
                    return penalty

                f_q = _trapz_normalize_density(x_grid, row["f_q"], eps=eps)
                if f_q.size != x_grid.size or not np.all(np.isfinite(f_q)):
                    return penalty

                sigma = float(row.get("sigma", 1.0))
                realized = float(row["realized_return"])

                if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(realized):
                    return penalty

                # If the realized return is outside the evaluation grid, the likelihood
                # contribution is not trustworthy. Penalize instead of allowing nan/zero.
                if realized < x_grid[0] or realized > x_grid[-1]:
                    return penalty

                g = g_r_sigma(x_grid, sigma, theta, N=N, Ksig=Ksig)
                if g.size != x_grid.size or not np.all(np.isfinite(g)):
                    return penalty

                # exp(-g) enters f_P. Clip to prevent overflow/underflow.
                weight = np.exp(np.clip(-g, -700, 700))
                raw = f_q * weight

                mass = float(np.trapezoid(raw, x_grid))
                if not np.isfinite(mass) or mass <= eps:
                    return penalty

                f_p = raw / mass
                if f_p.size != x_grid.size or not np.all(np.isfinite(f_p)):
                    return penalty

                f_at_realized = _safe_interp(realized, x_grid, f_p)
                if not np.isfinite(f_at_realized) or f_at_realized <= eps:
                    return penalty

                total_ll += np.log(f_at_realized)

            if not np.isfinite(total_ll):
                return penalty

            out = float(-total_ll)
            return out if np.isfinite(out) else penalty

        res = minimize(
            nll,
            x0,
            method="L-BFGS-B" if bounds is not None else "BFGS",
            bounds=bounds,
            options={"maxiter": self.maxiter},
        )

        fitted = ExponentialPolynomialFitted(
            theta_hat=np.asarray(res.x, dtype=float),
            theta_spec=self.theta_spec,
            T=float(T),
            loss=float(res.fun),
            success=bool(res.success),
            message=str(res.message),
        )

        # --------------------------------------------------------
        # Interpret optimizer result
        # --------------------------------------------------------

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
            "params": {f"theta_{i}": float(v) for i, v in enumerate(res.x)},
        }
        return fitted, diag

    def _transform_surface_with_model(
        self,
        fitted_model: ExponentialPolynomialFitted,
        x_grid: np.ndarray,
        f_q: np.ndarray,
        F_q: np.ndarray,
        *,
        T: float,
        info: dict,
    ) -> dict:
        x_grid = _as_1d(x_grid)
        f_q = _trapz_normalize_density(x_grid, f_q, eps=self.eps)
        F_q = _as_1d(F_q)

        sigma = _find_sigma(info, self.key_spec.sigma_keys, default=1.0)
        theta = fitted_model.theta_hat
        N = int(fitted_model.theta_spec.N)
        Ksig = int(fitted_model.theta_spec.Ksig)

        g = g_r_sigma(x_grid, sigma, theta, N=N, Ksig=Ksig)

        # Kernel is proportional to exp(g). Normalize so E_P[M] = 1 on the grid.
        raw_kernel = np.exp(np.clip(g, -700, 700))

        # Physical density is f_Q / M, normalized.
        raw_f_p = f_q / np.maximum(raw_kernel, self.eps)
        f_p = _trapz_normalize_density(x_grid, raw_f_p, eps=self.eps)
        F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)

        # dP/dQ
        weight = f_p / np.maximum(f_q, self.eps)

        # normalize M so ∫ M f_P dx = 1
        M = f_q / np.maximum(f_p, self.eps)
        Em = float(np.trapezoid(M * f_p, x_grid))
        if np.isfinite(Em) and Em > self.eps:
            M = M / Em

        return {
            "x_grid": x_grid,
            "f_q": f_q,
            "F_q": F_q,
            "f_p": f_p,
            "F_p": F_p,
            "weight": weight,
            "pricing_kernel": M,
            "g": g,
            "theta_hat": theta,
            "T_fit": float(T),
        }

def g_r_sigma(r: np.ndarray, sigma: float, theta: np.ndarray, *, N: int, Ksig: int) -> np.ndarray:
    theta_mat = unpack_theta(theta, N, Ksig)
    c = c_it(float(sigma), theta_mat)
    powers = np.vstack([_as_1d(r) ** i for i in range(1, N + 1)]).T
    return powers @ c

def unpack_theta(theta: np.ndarray, N: int, Ksig: int) -> np.ndarray:
    theta = _as_1d(theta)
    return theta.reshape((Ksig + 1, N), order="F")

def c_it(sigma: float, theta_mat: np.ndarray) -> np.ndarray:
    Ksig = theta_mat.shape[0] - 1
    sig_pow = float(sigma) ** np.arange(Ksig + 1, dtype=float)
    return sig_pow @ theta_mat
