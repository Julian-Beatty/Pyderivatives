# crra.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..base import MeasureTransform
from ..config import CacheSpec, KeySpec
from ..registry import register_transform
from ..utils import _as_1d, _cdf_from_density, _safe_interp, _trapz_normalize_density


@dataclass
class CRRAFitted:
    gamma: float
    T: float
    loss: float = 0.0
    success: bool = True
    message: str = "CRRA transform"


@register_transform("crra")
@register_transform("crra_kernel")
class CRRAKernel(MeasureTransform):
    """
    CRRA change of measure.

    Fixed gamma:

        f_P(r) ∝ f_Q(r) exp(gamma r)

    Fitted gamma:

        gamma is estimated by maximizing the historical log score of the
        transformed physical densities.
    """

    def __init__(
        self,
        *,
        gamma: float | str = 2.0,
        fit_gamma: bool = False,
        gamma_bounds: Tuple[float, float] = (-10.0, 20.0),
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

        if isinstance(gamma, str):
            if gamma.lower().strip() != "fit":
                raise ValueError("gamma must be a float or 'fit'.")
            fit_gamma = True
            gamma = 2.0

        self.gamma = float(gamma)
        self.fit_gamma = bool(fit_gamma)
        self.gamma_bounds = tuple(map(float, gamma_bounds))

    def _cache_params(self) -> dict:
        return {
            "gamma": self.gamma,
            "fit_gamma": self.fit_gamma,
            "gamma_bounds": self.gamma_bounds,
        }

    def _physical_pdf_from_gamma(self, x_grid, f_q, gamma):
        r = _as_1d(x_grid)
        q = _trapz_normalize_density(r, f_q, eps=self.eps)

        if not np.all(np.isfinite(q)):
            return None

        w = np.exp(np.clip(float(gamma) * r, -700, 700))
        raw = q * w
        f_p = _trapz_normalize_density(r, raw, eps=self.eps)

        if not np.all(np.isfinite(f_p)):
            return None

        return f_p

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float):
        if not self.fit_gamma:
            fitted = CRRAFitted(
                gamma=self.gamma,
                T=float(T),
                loss=0.0,
                success=True,
                message="fixed-gamma CRRA transform; no optimization performed.",
            )
            diag = {
                "loss": 0.0,
                "loss_name": "fixed_gamma",
                "status": "success",
                "message": "fixed-gamma CRRA transform; no optimization performed.",
                "params": {"gamma": self.gamma},
            }
            return fitted, diag

        rows = hist_T.copy()

        def nll(gamma):
            total = 0.0

            for _, row in rows.iterrows():
                x_grid = _as_1d(row["x_grid"])
                f_q = _as_1d(row["f_q"])
                realized = float(row["realized_return"])

                if x_grid.size < 10 or not np.isfinite(realized):
                    return self.penalty_value

                f_p = self._physical_pdf_from_gamma(x_grid, f_q, gamma)

                if f_p is None:
                    return self.penalty_value

                dens = float(_safe_interp(realized, x_grid, f_p))

                if not np.isfinite(dens) or dens <= self.eps:
                    return self.penalty_value

                total += np.log(dens)

            return -float(total)

        res = minimize_scalar(
            nll,
            bounds=self.gamma_bounds,
            method="bounded",
            options={"xatol": 1e-5},
        )

        gamma_hat = float(res.x)
        loss = float(res.fun)

        fitted = CRRAFitted(
            gamma=gamma_hat,
            T=float(T),
            loss=loss,
            success=bool(res.success),
            message=str(res.message),
        )

        diag = {
            "loss": loss,
            "loss_name": "negative_log_score",
            "status": "success" if res.success else "failed",
            "message": str(res.message),
            "params": {"gamma": gamma_hat},
        }

        return fitted, diag

    def _transform_surface_with_model(
        self,
        fitted_model: CRRAFitted,
        x_grid: np.ndarray,
        f_q: np.ndarray,
        F_q: np.ndarray,
        *,
        T: float,
        info: dict,
    ) -> Dict[str, Any]:
        r = _as_1d(x_grid)
        q = _trapz_normalize_density(r, f_q, eps=self.eps)

        gamma = float(fitted_model.gamma)
        f_p = self._physical_pdf_from_gamma(r, q, gamma)

        if f_p is None:
            raise ValueError("Could not construct CRRA physical density.")

        F_p = _cdf_from_density(r, f_p, eps=self.eps)

        pricing_kernel = q / np.maximum(f_p, self.eps)
        weight = f_p / np.maximum(q, self.eps)

        return {
            "f_p": f_p,
            "F_p": F_p,
            "weight": weight,
            "pricing_kernel": pricing_kernel,
            "gamma": gamma,
        }