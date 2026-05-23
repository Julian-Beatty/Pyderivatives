from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import MeasureTransform
from ..config import CacheSpec, KeySpec, NonparametricCalibrationSpec
from ..registry import register_transform
from ..utils import (
    _as_1d,
    _cdf_from_density,
    _trapz_normalize_density,
)

@dataclass
class NonparametricCalibrationFitted:
    z_obs: np.ndarray
    z_grid: np.ndarray
    h_grid: np.ndarray
    H_grid: np.ndarray
    bandwidth: float
    bandwidth_method: str
    T: float
    loss: float
    success: bool
    message: str


@register_transform("nonparametric")
@register_transform("nonparametric_calibration")
@register_transform("np_calibration")
class NonparametricCalibration(MeasureTransform):
    """
    Nonparametric PIT calibration using Gaussian KDE in normal-score space.

    This is the nonparametric analogue of beta calibration:

        F_P(x) = C(F_Q(x))

    but C is estimated from transformed PITs rather than forced to be beta.
    """

    def __init__(
        self,
        *,
        spec: NonparametricCalibrationSpec = NonparametricCalibrationSpec(),
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

    def _normal_pdf(self, z):
        z = np.asarray(z, dtype=float)
        return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

    def _normal_ppf(self, u):
        from scipy.stats import norm
        return norm.ppf(u)

    def _select_bandwidth(self, z: np.ndarray) -> Tuple[float, str]:
        z = _as_1d(z)
        n = z.size
        if n < 2:
            return float(self.spec.min_bandwidth), "fallback_min_bandwidth"

        bw = self.spec.bandwidth

        if isinstance(bw, str):
            method = bw.lower().strip()
            sd = float(np.std(z, ddof=1))
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0

            if method == "silverman":
                h = 1.06 * sd * (n ** (-1.0 / 5.0))
            elif method == "scott":
                h = sd * (n ** (-1.0 / 5.0))
            else:
                raise ValueError("bandwidth must be 'silverman', 'scott', or a positive float.")
        else:
            h = float(bw)
            method = "manual"

        if not np.isfinite(h) or h <= 0:
            h = float(self.spec.min_bandwidth)

        h = max(h, float(self.spec.min_bandwidth))
        if self.spec.max_bandwidth is not None:
            h = min(h, float(self.spec.max_bandwidth))

        return float(h), method

    def _kde_gaussian_on_grid(self, z_obs: np.ndarray, z_grid: np.ndarray, h: float) -> np.ndarray:
        z_obs = _as_1d(z_obs)
        z_grid = _as_1d(z_grid)
        h = float(h)

        # Matrix form is fine for moderate n/grid sizes. If n becomes huge,
        # this can be replaced by FFT/binning later.
        U = (z_grid[:, None] - z_obs[None, :]) / h
        h_grid = np.mean(np.exp(-0.5 * U**2) / np.sqrt(2.0 * np.pi), axis=1) / h
        h_grid = np.where(np.isfinite(h_grid) & (h_grid >= 0), h_grid, 0.0)
        return h_grid

    def _cache_params(self) -> dict:
        return {"spec": self.spec}

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float):
        u = np.asarray(hist_T["pit"], dtype=float)
        u = np.clip(u[np.isfinite(u)], self.eps, 1.0 - self.eps)

        if u.size < self.min_obs:
            fitted = NonparametricCalibrationFitted(
                z_obs=np.array([], dtype=float),
                z_grid=np.array([], dtype=float),
                h_grid=np.array([], dtype=float),
                H_grid=np.array([], dtype=float),
                bandwidth=np.nan,
                bandwidth_method="none",
                T=float(T),
                loss=np.nan,
                success=False,
                message="Too few valid PIT observations.",
            )
            diag = {
                "loss": np.nan,
                "loss_name": "negative_log_kde_likelihood",
                "status": "failed",
                "message": "Too few valid PIT observations.",
                "params": {"bandwidth": np.nan, "bandwidth_method": "none"},
            }
            return fitted, diag

        z_obs = self._normal_ppf(u)
        z_obs = z_obs[np.isfinite(z_obs)]

        if z_obs.size < self.min_obs:
            fitted = NonparametricCalibrationFitted(
                z_obs=np.array([], dtype=float),
                z_grid=np.array([], dtype=float),
                h_grid=np.array([], dtype=float),
                H_grid=np.array([], dtype=float),
                bandwidth=np.nan,
                bandwidth_method="none",
                T=float(T),
                loss=np.nan,
                success=False,
                message="Too few finite normal-score PIT observations.",
            )
            
            diag = {
                "loss": np.nan,
                "loss_name": "negative_log_kde_likelihood",
                "status": "failed",
                "message": "Too few finite normal-score PIT observations.",
                "params": {},
            }
            
            return fitted, diag
        h, method = self._select_bandwidth(z_obs)

        z_min = float(np.min(z_obs) - self.spec.z_grid_pad)
        z_max = float(np.max(z_obs) + self.spec.z_grid_pad)

        # Always include a reasonable normal-score region, because transformation
        # may be evaluated outside the observed z range.
        z_min = min(z_min, -5.0)
        z_max = max(z_max, 5.0)

        z_grid = np.linspace(z_min, z_max, int(self.spec.z_grid_size))
        h_grid = self._kde_gaussian_on_grid(z_obs, z_grid, h)
        h_grid = _trapz_normalize_density(z_grid, h_grid, eps=self.eps)
        H_grid = _cdf_from_density(z_grid, h_grid, eps=self.eps)

        # Pseudo likelihood: leave-in KDE likelihood on z_obs.
        h_at_obs = np.interp(z_obs, z_grid, h_grid, left=self.eps, right=self.eps)
        h_at_obs = np.maximum(h_at_obs, self.eps)
        loss = -float(np.sum(np.log(h_at_obs)))

        fitted = NonparametricCalibrationFitted(
            z_obs=z_obs,
            z_grid=z_grid,
            h_grid=h_grid,
            H_grid=H_grid,
            bandwidth=float(h),
            bandwidth_method=method,
            T=float(T),
            loss=loss,
            success=True,
            message="success",
        )

        diag = {
            "loss": loss,
            "loss_name": "negative_log_kde_likelihood",
            "status": "success",
            "message": "success",
            "params": {
                "bandwidth": float(h),
                "bandwidth_method": method,
                "n_z": int(z_obs.size),
                "z_min": float(z_min),
                "z_max": float(z_max),
            },
        }
        return fitted, diag

    def _transform_surface_with_model(
        self,
        fitted_model: NonparametricCalibrationFitted,
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

        if fitted_model.z_grid.size < 2:
            raise RuntimeError("Nonparametric calibration model is not valid for this maturity.")

        u = np.clip(F_q, self.eps, 1.0 - self.eps)
        z = self._normal_ppf(u)

        h_z = np.interp(
            z,
            fitted_model.z_grid,
            fitted_model.h_grid,
            left=self.eps,
            right=self.eps,
        )
        H_z = np.interp(
            z,
            fitted_model.z_grid,
            fitted_model.H_grid,
            left=0.0,
            right=1.0,
        )

        phi_z = self._normal_pdf(z)
        ratio = h_z / np.maximum(phi_z, self.eps)
        ratio = np.where(np.isfinite(ratio) & (ratio >= 0), ratio, 0.0)

        f_p_raw = f_q * ratio
        f_p = _trapz_normalize_density(x_grid, f_p_raw, eps=self.eps)

        # Use numerical CDF for consistency with normalized f_p.
        F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)

        weight = f_p / np.maximum(f_q, self.eps)

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
            "z": z,
            "h_z": h_z,
            "H_z": H_z,
            "bandwidth": fitted_model.bandwidth,
            "bandwidth_method": fitted_model.bandwidth_method,
            "T_fit": float(T),
        }
