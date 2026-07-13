from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import MeasureTransform
from ..config import CacheSpec, KeySpec
from ..registry import register_transform
from ..utils import _as_1d, _cdf_from_density, _trapz_normalize_density


@dataclass
class CRRAFitted:
    gamma: float
    T: float
    success: bool = True
    message: str = "closed-form CRRA transform"


@register_transform("crra")
@register_transform("crra_kernel")
class CRRAKernel(MeasureTransform):
    """
    Closed-form CRRA change of measure.

    On the log-return grid r = log(S_T/S_0), CRRA marginal utility is

        u'(R) = R^{-gamma} = exp(-gamma r),

    so the physical density is

        f_P(r) ∝ f_Q(r) / u'(R)
               ∝ f_Q(r) exp(gamma r).

    This method has no estimated parameters, but it follows the same
    fit/transform API as the other pricing-kernel transforms.
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        key_spec: KeySpec = KeySpec(),
        fit_trim_alpha: Optional[Tuple[float, float]] = None,
        min_obs: int = 1,
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
        self.gamma = float(gamma)

    def _cache_params(self) -> dict:
        return {"gamma": self.gamma}

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float):
        fitted = CRRAFitted(gamma=self.gamma, T=float(T))
        diag = {
            "loss": 0.0,
            "loss_name": "closed_form",
            "status": "success",
            "message": "closed-form CRRA transform; no optimization performed.",
            "params": {"gamma": self.gamma},
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
        w = np.exp(np.clip(gamma * r, -700, 700))
        raw = q * w
        f_p = _trapz_normalize_density(r, raw, eps=self.eps)
        F_p = _cdf_from_density(r, f_p, eps=self.eps)

        # Pricing kernel maps final P back to Q: M ∝ f_Q / f_P.
        pricing_kernel = q / np.maximum(f_p, self.eps)
        weight = f_p / np.maximum(q, self.eps)

        return {
            "f_p": f_p,
            "F_p": F_p,
            "weight": weight,
            "pricing_kernel": pricing_kernel,
            "gamma": gamma,
        }
