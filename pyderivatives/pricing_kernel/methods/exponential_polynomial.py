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


def _physical_density_from_g(
    x_grid: np.ndarray,
    f_q: np.ndarray,
    g: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    """Compute f_P proportional to f_Q * exp(-g) in stable log space."""
    x_grid = _as_1d(x_grid)
    f_q = _as_1d(f_q)
    g = _as_1d(g)

    if not (x_grid.size == f_q.size == g.size):
        raise ValueError("x_grid, f_q, and g must have the same length.")
    if x_grid.size < 2:
        raise ValueError("At least two grid points are required.")
    if np.any(~np.isfinite(x_grid)) or np.any(np.diff(x_grid) <= 0):
        raise ValueError("x_grid must be finite and strictly increasing.")
    if np.any(~np.isfinite(f_q)) or np.any(f_q < 0):
        raise ValueError("f_q must be finite and nonnegative.")
    if np.any(~np.isfinite(g)):
        raise ValueError("g must be finite.")

    tiny = np.finfo(float).tiny
    log_raw = np.log(np.maximum(f_q, tiny)) - g
    log_raw -= float(np.max(log_raw))
    raw_f_p = np.exp(log_raw)

    f_p = _trapz_normalize_density(x_grid, raw_f_p, eps=eps)
    if (
        f_p.size != x_grid.size
        or np.any(~np.isfinite(f_p))
        or np.any(f_p < 0)
    ):
        raise ValueError("Could not construct a finite physical density.")
    return f_p



def _log_trapezoid_positive(
    x_grid: np.ndarray,
    log_values: np.ndarray,
) -> float:
    """Compute log(integral exp(log_values) dx) stably."""
    x = _as_1d(x_grid)
    log_y = _as_1d(log_values)

    if x.size != log_y.size:
        raise ValueError(
            "x_grid and log_values must have the same length."
        )

    if (
        x.size < 2
        or np.any(~np.isfinite(x))
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError(
            "x_grid must contain finite, strictly increasing values."
        )

    log_terms = (
        np.log(0.5 * np.diff(x))
        + np.logaddexp(
            log_y[:-1],
            log_y[1:],
        )
    )

    finite = np.isfinite(log_terms)

    if not np.any(finite):
        raise ValueError(
            "The log-space integral contains no positive mass."
        )

    maximum = float(np.max(log_terms[finite]))

    result = (
        maximum
        + np.log(
            np.sum(
                np.exp(
                    log_terms[finite] - maximum
                )
            )
        )
    )

    if not np.isfinite(result):
        raise ValueError(
            "The log-space integral is not finite."
        )

    return float(result)

def _transform_diagnostics(
    *,
    x_grid: np.ndarray,
    f_q: np.ndarray,
    f_p: np.ndarray,
    g: np.ndarray,
    theta: np.ndarray,
    effective_coefficients: np.ndarray,
) -> dict:
    """Diagnostics for concentrated or unstable transformed densities."""
    x_grid = _as_1d(x_grid)
    f_q = _as_1d(f_q)
    f_p = _as_1d(f_p)
    g = _as_1d(g)
    theta = _as_1d(theta)
    effective_coefficients = _as_1d(effective_coefficients)

    tiny = np.finfo(float).tiny
    peak = float(np.max(f_p))
    threshold = 1e-8 * peak if peak > 0 else np.inf
    support_fraction = float(np.mean(f_p > threshold))

    physical_mean = float(np.trapezoid(x_grid * f_p, x_grid))
    physical_variance = float(
        np.trapezoid((x_grid - physical_mean) ** 2 * f_p, x_grid)
    )
    physical_entropy = float(
        -np.trapezoid(f_p * np.log(np.maximum(f_p, tiny)), x_grid)
    )

    rnd_mean = float(np.trapezoid(x_grid * f_q, x_grid))
    rnd_variance = float(
        np.trapezoid((x_grid - rnd_mean) ** 2 * f_q, x_grid)
    )

    kl_pq = float(
        np.trapezoid(
            np.where(
                (f_p > tiny) & (f_q > tiny),
                f_p
                * (
                    np.log(np.maximum(f_p, tiny))
                    - np.log(np.maximum(f_q, tiny))
                ),
                0.0,
            ),
            x_grid,
        )
    )

    g_min = float(np.min(g))
    g_max = float(np.max(g))
    g_range = float(g_max - g_min)
    jump_ratio = (
        float(np.max(np.abs(np.diff(f_p))) / peak)
        if len(f_p) > 1 and peak > 0
        else np.nan
    )

    variance_ratio = (
        float(physical_variance / rnd_variance)
        if np.isfinite(rnd_variance) and rnd_variance > 0
        else np.nan
    )

    warnings = []
    if support_fraction < 0.10:
        warnings.append("physical_density_effective_support_extremely_narrow")
    elif support_fraction < 0.20:
        warnings.append("physical_density_effective_support_narrow")

    if g_range > 100:
        warnings.append("exponential_tilt_extremely_large_g_range")
    elif g_range > 50:
        warnings.append("exponential_tilt_large_g_range")

    if peak > 10:
        warnings.append("physical_density_high_peak")
    if np.isfinite(jump_ratio) and jump_ratio > 0.50:
        warnings.append("physical_density_large_internal_jump")
    if np.isfinite(variance_ratio) and variance_ratio < 0.05:
        warnings.append("physical_density_variance_collapsed_relative_to_rnd")

    return {
        "effective_support_fraction_1e-8": support_fraction,
        "physical_peak": peak,
        "physical_mean": physical_mean,
        "physical_variance": physical_variance,
        "physical_entropy": physical_entropy,
        "rnd_mean": rnd_mean,
        "rnd_variance": rnd_variance,
        "physical_to_rnd_variance_ratio": variance_ratio,
        "kl_p_to_q": kl_pq,
        "theta_l2_norm": float(np.linalg.norm(theta)),
        "effective_return_coefficients": effective_coefficients.tolist(),
        "g_min": g_min,
        "g_max": g_max,
        "g_range": g_range,
        "max_adjacent_jump_ratio": jump_ratio,
        "warnings": warnings,
        "anomaly": bool(warnings),
    }


@register_transform("exponential_polynomial")
class ExponentialPolynomialKernel(MeasureTransform):
    """
    Exponential-polynomial pricing-kernel transformation.

    M(r, sigma; theta) = exp(g(r, sigma; theta))
    f_P(r) proportional to f_Q(r) exp(-g(r, sigma; theta))
    """

    def __init__(
        self,
        *,
        theta_spec: ThetaSpec = ThetaSpec(),
        maxiter: int = 400,
        x0: Optional[np.ndarray] = None,
        ridge_penalty: float = 0.0,
        key_spec: KeySpec = KeySpec(),
        fit_trim_alpha: Optional[Tuple[float, float]] = None,
        min_obs: int = 30,
        fit_maturities: Optional[List[float]] = None,
        maturity_match_tol: Optional[float] = None,
        eps: float = 1e-10,
        verbose: bool = True,
        penalty_value: float = 1e100,
        cache_spec: CacheSpec = CacheSpec(),
        behavioral: bool = False,
        stock_df: Optional[pd.DataFrame] = None,
        stock_date_col: str = "date",
        volume_col: str = "volume",
        k1: float = 1.0,
        k2: float = 1.2,
        k3: float = 1.0,
        sentiment_alpha: float = 0.05,
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
            behavioral=behavioral,
            stock_df=stock_df,
            stock_date_col=stock_date_col,
            volume_col=volume_col,
            k1=k1,
            k2=k2,
            k3=k3,
            sentiment_alpha=sentiment_alpha,
        )
        if float(ridge_penalty) < 0:
            raise ValueError("ridge_penalty must be nonnegative.")

        self.theta_spec = theta_spec
        self.maxiter = int(maxiter)
        self.x0 = None if x0 is None else _as_1d(x0)
        self.ridge_penalty = float(ridge_penalty)

    def _cache_params(self) -> dict:
        return {
            "theta_spec": self.theta_spec,
            "maxiter": self.maxiter,
            "x0": self.x0,
            "ridge_penalty": self.ridge_penalty,
        }

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
            raise ValueError(
                "ThetaSpec.bounds must have lower and upper arrays "
                "of length N*(Ksig+1)."
            )
        return list(zip(lb, ub))

    def _fit_one_maturity(
        self,
        hist_T: pd.DataFrame,
        *,
        T: float,
    ) -> Tuple[ExponentialPolynomialFitted, Dict[str, Any]]:
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
            penalty = self.penalty_value
            if theta.size != p or not np.all(np.isfinite(theta)):
                return penalty

            total_ll = 0.0
            for _, row in hist_T.iterrows():
                x_grid = _as_1d(row["x_grid"])
                if (
                    x_grid.size < 10
                    or np.any(~np.isfinite(x_grid))
                    or np.any(np.diff(x_grid) <= 0)
                ):
                    return penalty

                f_q = _trapz_normalize_density(x_grid, row["f_q"], eps=eps)
                if f_q.size != x_grid.size or not np.all(np.isfinite(f_q)):
                    return penalty

                sigma = float(row.get("sigma", 1.0))
                realized = float(row["realized_return"])
                if (
                    not np.isfinite(sigma)
                    or sigma <= 0
                    or not np.isfinite(realized)
                ):
                    return penalty
                if realized < x_grid[0] or realized > x_grid[-1]:
                    return penalty

                g = g_r_sigma(x_grid, sigma, theta, N=N, Ksig=Ksig)
                if g.size != x_grid.size or not np.all(np.isfinite(g)):
                    return penalty

                try:
                    f_p = _physical_density_from_g(x_grid, f_q, g, eps=eps)
                except (ValueError, FloatingPointError):
                    return penalty

                f_at_realized = _safe_interp(realized, x_grid, f_p)
                if not np.isfinite(f_at_realized) or f_at_realized <= eps:
                    return penalty
                total_ll += np.log(f_at_realized)

            if not np.isfinite(total_ll):
                return penalty

            objective = (
                float(-total_ll)
                + self.ridge_penalty * float(np.dot(theta, theta))
            )
            return float(objective) if np.isfinite(objective) else penalty

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

        if res.success:
            status = "success"
        elif np.all(np.isfinite(res.x)) and np.isfinite(res.fun):
            status = "questionable"
        else:
            status = "failed"

        theta_hat = np.asarray(res.x, dtype=float)
        regularization_penalty = (
            self.ridge_penalty * float(np.dot(theta_hat, theta_hat))
            if np.all(np.isfinite(theta_hat))
            else np.nan
        )
        unpenalized_nll = (
            float(res.fun - regularization_penalty)
            if np.isfinite(res.fun) and np.isfinite(regularization_penalty)
            else np.nan
        )

        sigma_values = pd.to_numeric(
            hist_T.get(
                "sigma",
                pd.Series(np.ones(len(hist_T)), index=hist_T.index),
            ),
            errors="coerce",
        )
        sigma_values = sigma_values[
            np.isfinite(sigma_values) & (sigma_values > 0)
        ]
        representative_sigma = float(
            np.median(sigma_values) if len(sigma_values) else 1.0
        )
        effective_coefficients = c_it(
            representative_sigma,
            unpack_theta(theta_hat, N, Ksig),
        )

        diag = {
            "loss": float(res.fun),
            "loss_name": (
                "penalized_negative_log_likelihood"
                if self.ridge_penalty > 0
                else "negative_log_likelihood"
            ),
            "unpenalized_negative_log_likelihood": unpenalized_nll,
            "regularization_penalty": float(regularization_penalty),
            "ridge_penalty": float(self.ridge_penalty),
            "status": status,
            "message": str(res.message),
            "params": {
                f"theta_{i}": float(value)
                for i, value in enumerate(theta_hat)
            },
            "theta_l2_norm": float(np.linalg.norm(theta_hat)),
            "representative_sigma": representative_sigma,
            "effective_return_coefficients": {
                f"c_{i + 1}": float(value)
                for i, value in enumerate(effective_coefficients)
            },
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
        f_p = _physical_density_from_g(x_grid, f_q, g, eps=self.eps)
        F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)

        effective_coefficients = c_it(
            sigma,
            unpack_theta(theta, N, Ksig),
        )

        tiny = np.finfo(float).tiny
        weight = np.full_like(f_p, np.nan, dtype=float)
        valid_weight = np.isfinite(f_q) & np.isfinite(f_p) & (f_q > tiny)
        weight[valid_weight] = f_p[valid_weight] / f_q[valid_weight]

        # Construct and normalize M entirely in log space:
        #
        #   log E_P[exp(g)] = log integral exp(g) f_P dx
        #   log M = g - log E_P[exp(g)]
        #
        # This avoids dividing by an extremely small normalization
        # constant, which previously generated overflow warnings.
        log_f_p = np.log(
            np.maximum(
                f_p,
                np.finfo(float).tiny,
            )
        )

        log_Em = _log_trapezoid_positive(
            x_grid,
            log_f_p + g,
        )

        log_M = g - log_Em

        # Keep the stored kernel finite in float64. Any clipping is
        # explicitly recorded below as a transform anomaly.
        kernel_log_upper = 700.0
        kernel_log_lower = -745.0

        kernel_high_clip_count = int(
            np.sum(log_M > kernel_log_upper)
        )
        kernel_low_clip_count = int(
            np.sum(log_M < kernel_log_lower)
        )

        M = np.exp(
            np.clip(
                log_M,
                kernel_log_lower,
                kernel_log_upper,
            )
        )

        if not np.all(np.isfinite(M)):
            raise ValueError(
                "Could not construct a finite pricing kernel."
            )

        kernel_normalization_check = float(
            np.trapezoid(
                M * f_p,
                x_grid,
            )
        )

        diagnostics = _transform_diagnostics(
            x_grid=x_grid,
            f_q=f_q,
            f_p=f_p,
            g=g,
            theta=theta,
            effective_coefficients=effective_coefficients,
        )

        diagnostics.update({
            "pricing_kernel_log_normalizer": float(log_Em),
            "pricing_kernel_log_min": float(np.min(log_M)),
            "pricing_kernel_log_max": float(np.max(log_M)),
            "pricing_kernel_high_clip_count": kernel_high_clip_count,
            "pricing_kernel_low_clip_count": kernel_low_clip_count,
            "pricing_kernel_log_upper": kernel_log_upper,
            "pricing_kernel_log_lower": kernel_log_lower,
            "pricing_kernel_normalization_check": (
                kernel_normalization_check
            ),
        })

        if kernel_high_clip_count > 0:
            diagnostics["warnings"].append(
                "pricing_kernel_exceeds_float64_safe_range"
            )
            diagnostics["anomaly"] = True

        return {
            "x_grid": x_grid,
            "f_q": f_q,
            "F_q": F_q,
            "f_p": f_p,
            "F_p": F_p,
            "weight": weight,
            "pricing_kernel": M,
            "log_pricing_kernel": log_M,
            "g": g,
            "theta_hat": theta,
            "T_fit": float(T),
            "sigma": float(sigma),
            "effective_return_coefficients": effective_coefficients,
            "transform_diagnostics": diagnostics,
            "transform_anomaly": bool(diagnostics["anomaly"]),
            "transform_warnings": list(diagnostics["warnings"]),
        }


def g_r_sigma(
    r: np.ndarray,
    sigma: float,
    theta: np.ndarray,
    *,
    N: int,
    Ksig: int,
) -> np.ndarray:
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
