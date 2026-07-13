from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..base import MeasureTransform
from ..config import CacheSpec, ExponentialSpec, KeySpec
from ..registry import register_transform
from ..utils import (
    _as_1d,
    _cdf_from_density,
    _find_sigma,
    _safe_interp,
    _trapz_normalize_density,
)


@dataclass
class ExponentialFitted:
    theta_hat: np.ndarray
    spec: ExponentialSpec
    T: float
    loss: float
    success: bool
    message: str


@register_transform("exponential")
class ExponentialKernel(MeasureTransform):
    """
    Exponential pricing-kernel transformation.

    Uses implicit-delta normalization:

        g(r, sigma; theta) = sum_i c_i sigma^(-b i) r^i

        f_P(r) ∝ f_Q(r) exp(-g(r, sigma; theta))

    Then the pricing kernel is recovered after the physical density is built:

        M(r) = f_Q(r) / f_P(r)

    This is numerically more stable than constructing M first.
    """

    def __init__(
        self,
        *,
        spec: ExponentialSpec = ExponentialSpec(),
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

        self.spec = spec
        self.maxiter = int(maxiter)
        self.x0 = None if x0 is None else _as_1d(x0)

    def _cache_params(self) -> dict:
        return {
            "spec": self.spec,
            "maxiter": self.maxiter,
            "x0": self.x0,
        }

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
        bounds.extend(
            [(float(c_lo), float(c_hi)) for _ in range(int(self.spec.N))]
        )

        return bounds

    def _g_no_delta(
        self,
        r: np.ndarray,
        sigma: float,
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Polynomial part of the log pricing kernel without intercept delta.

        g(r, sigma)
            =
        sum_i c_i sigma^(-b i) r^i
        """
        b, c = self._unpack_theta(theta)

        r = _as_1d(r)
        sigma = max(float(sigma), self.eps)

        g = np.zeros_like(r, dtype=float)

        for i in range(1, int(self.spec.N) + 1):
            coeff = c[i - 1] / (sigma ** (b * i))
            g += coeff * (r ** i)

        return g

    def _physical_density_from_theta(
        self,
        x_grid,
        f_q,
        sigma,
        theta,
        *,
        eps: Optional[float] = None,
    ):
        """
        Build physical density directly using implicit-delta normalization.

        f_P(r)
            =
        f_Q(r) exp(-g(r)) / ∫ f_Q(u) exp(-g(u)) du

        Returns
        -------
        f_p:
            Normalized physical density.

        g:
            Unnormalized log-kernel polynomial.

        mass:
            Normalizing constant ∫ f_Q exp(-g).
        """
        if eps is None:
            eps = self.eps

        x_grid = _as_1d(x_grid)
        f_q = _trapz_normalize_density(x_grid, f_q, eps=eps)

        g = self._g_no_delta(x_grid, sigma, theta)

        if g.size != x_grid.size or not np.all(np.isfinite(g)):
            return None, None, np.nan

        weight_raw = np.exp(np.clip(-g, -700, 700))
        raw_f_p = f_q * weight_raw

        mass = float(np.trapezoid(raw_f_p, x_grid))

        if not np.isfinite(mass) or mass <= eps:
            return None, None, np.nan

        f_p = raw_f_p / mass

        if f_p.size != x_grid.size or not np.all(np.isfinite(f_p)):
            return None, None, np.nan

        return f_p, g, mass

    def _fit_one_maturity(
        self,
        hist_T: pd.DataFrame,
        *,
        T: float,
    ):
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

                if (
                    x_grid.size < 10
                    or np.any(~np.isfinite(x_grid))
                    or np.any(np.diff(x_grid) <= 0)
                ):
                    return penalty

                f_q = _trapz_normalize_density(
                    x_grid,
                    row["f_q"],
                    eps=eps,
                )

                sigma = float(row.get("sigma", 1.0))
                realized = float(row["realized_return"])

                if (
                    f_q.size != x_grid.size
                    or not np.all(np.isfinite(f_q))
                    or not np.isfinite(sigma)
                    or sigma <= 0
                    or not np.isfinite(realized)
                ):
                    return penalty

                if realized < x_grid[0] or realized > x_grid[-1]:
                    return penalty

                f_p, g, mass = self._physical_density_from_theta(
                    x_grid,
                    f_q,
                    sigma,
                    theta,
                    eps=eps,
                )

                if f_p is None:
                    return penalty

                f_at_realized = _safe_interp(realized, x_grid, f_p)

                if not np.isfinite(f_at_realized) or f_at_realized <= eps:
                    return penalty

                total_ll += np.log(f_at_realized)

            out = -float(total_ll)
            return out if np.isfinite(out) else penalty

        res = minimize(
            nll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter},
        )

        fitted = ExponentialFitted(
            theta_hat=np.asarray(res.x, dtype=float),
            spec=self.spec,
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

        diag = {
            "loss": float(res.fun),
            "loss_name": "negative_log_likelihood",
            "status": status,
            "message": str(res.message),
            "params": {
                "b": float(res.x[0]),
                **{
                    f"c{i}": float(v)
                    for i, v in enumerate(res.x[1:], start=1)
                },
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

        f_p, g, mass = self._physical_density_from_theta(
            x_grid,
            f_q,
            sigma,
            fitted_model.theta_hat,
            eps=self.eps,
        )

        if f_p is None:
            raise RuntimeError("Failed to compute physical density.")

        F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)

        # dP/dQ
        weight = f_p / np.maximum(f_q, self.eps)

        # Pricing kernel recovered after physical density is normalized.
        pricing_kernel = f_q / np.maximum(f_p, self.eps)

        # Normalize the recovered kernel so E_P[M] = 1 on the grid.
        Em = float(np.trapezoid(pricing_kernel * f_p, x_grid))
        if np.isfinite(Em) and Em > self.eps:
            pricing_kernel = pricing_kernel / Em

        return {
            "x_grid": x_grid,
            "f_q": f_q,
            "F_q": F_q,
            "f_p": f_p,
            "F_p": F_p,
            "weight": weight,
            "pricing_kernel": pricing_kernel,
            "g": g,
            "normalization_mass": mass,
            "theta_hat": fitted_model.theta_hat,
            "T_fit": float(T),
        }