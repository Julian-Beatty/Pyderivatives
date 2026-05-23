
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..base import MeasureTransform
from ..config import BetaCalibrationSpec, CacheSpec, KeySpec
from ..registry import register_transform
from ..utils import (
    _as_1d,
    _cdf_from_density,
    _trapz_normalize_density,
)

@dataclass
class BetaCalibrationFitted:
    a: float
    b: float
    T: float
    loss: float
    success: bool
    message: str
    
    
    
@register_transform("beta_calibration")
@register_transform("beta")
class BetaCalibration(MeasureTransform):
    """
    Beta calibration transformation.

    This estimates one beta calibration function per maturity slice.

    For each maturity T:

        u_t = F_Q,t(r_t^realized)

    Then estimate:

        u_t ~ Beta(a_T, b_T)

    Transformation:

        F_P(x) = B(F_Q(x); a_T, b_T)
        f_P(x) = b(F_Q(x); a_T, b_T) f_Q(x)

    The pricing kernel is recovered as:

        M(x) ∝ f_Q(x) / f_P(x).
    """

    def __init__(
        self,
        *,
        spec: BetaCalibrationSpec = BetaCalibrationSpec(),
        maxiter: int = 400,
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

    def _cache_params(self) -> dict:
        return {"spec": self.spec, "maxiter": self.maxiter}

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float):
        from scipy.stats import beta as beta_dist

        u = np.asarray(hist_T["pit"], dtype=float)
        u = np.clip(u[np.isfinite(u)], self.eps, 1.0 - self.eps)

        penalty = self.penalty_value

        if u.size < self.min_obs:
            fitted = BetaCalibrationFitted(
                a=np.nan,
                b=np.nan,
                T=float(T),
                loss=np.nan,
                success=False,
                message="Too few valid PIT observations.",
            )
            diag = {
                "loss": np.nan,
                "loss_name": "negative_log_likelihood",
                "status": "failed",
                "message": "Too few valid PIT observations.",
                "params": {"a": np.nan, "b": np.nan},
            }
            return fitted, diag

        def nll(params):
            a, b = map(float, params)
            if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
                return penalty
            ll = beta_dist.logpdf(u, a, b)
            if not np.all(np.isfinite(ll)):
                return penalty
            out = -float(np.sum(ll))
            return out if np.isfinite(out) else penalty

        bounds = [self.spec.a_bounds, self.spec.b_bounds]
        x0 = np.asarray(self.spec.x0, dtype=float)
        x0[0] = np.clip(x0[0], bounds[0][0], bounds[0][1])
        x0[1] = np.clip(x0[1], bounds[1][0], bounds[1][1])

        res = minimize(
            nll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter},
        )

        a_hat, b_hat = map(float, res.x)

        fitted = BetaCalibrationFitted(
            a=a_hat,
            b=b_hat,
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
        fitted_model: BetaCalibrationFitted,
        x_grid: np.ndarray,
        f_q: np.ndarray,
        F_q: np.ndarray,
        *,
        T: float,
        info: dict,
    ) -> dict:
        from scipy.stats import beta as beta_dist

        x_grid = _as_1d(x_grid)
        f_q = _trapz_normalize_density(x_grid, f_q, eps=self.eps)
        F_q = _as_1d(F_q)

        if not np.isfinite(fitted_model.a) or not np.isfinite(fitted_model.b):
            raise RuntimeError("Beta calibration model is not valid for this maturity.")

        u = np.clip(F_q, self.eps, 1.0 - self.eps)

        weight = beta_dist.pdf(u, fitted_model.a, fitted_model.b)
        weight = np.where(np.isfinite(weight) & (weight >= 0), weight, 0.0)

        f_p_raw = f_q * weight
        f_p = _trapz_normalize_density(x_grid, f_p_raw, eps=self.eps)

        # The analytical calibrated CDF is beta.cdf(F_q), but after grid
        # normalization of f_p, using numerical CDF keeps F_p consistent with f_p.
        F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)

        weight_final = f_p / np.maximum(f_q, self.eps)

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
            "weight": weight_final,
            "pricing_kernel": M,
            "a": fitted_model.a,
            "b": fitted_model.b,
            "T_fit": float(T),
        }


