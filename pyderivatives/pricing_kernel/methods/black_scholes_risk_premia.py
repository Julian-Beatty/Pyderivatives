from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .stochastic_base import StochasticRiskPremiaTransform
from ..config import CacheSpec, KeySpec
from ..registry import register_transform


@dataclass
class BlackScholesRiskPremiaFitted:
    lambda_return: float
    T: float
    loss: float
    success: bool
    message: str


@register_transform("black_scholes_risk_premia")
@register_transform("bs_risk_premia")
class BlackScholesRiskPremia(StochasticRiskPremiaTransform):
    """Physical Black--Scholes density with an estimated return-risk premium.

    The option surface supplies the daily risk-neutral volatility ``sigma_t``.
    The physical dynamics are parameterized as

        dS_t / S_t = lambda_return * sigma_t**2 dt + sigma_t dW_t^P.

    Hence the conditional log-return density at horizon T is normal with

        mean = (lambda_return - 1/2) sigma_t**2 T,
        var  = sigma_t**2 T.

    A separate ``lambda_return`` is estimated for every fitted maturity by
    maximizing the historical log score of realized horizon returns.
    """

    supported_models = ("black_scholes", "blackscholes", "bs")

    def __init__(
        self,
        *,
        lambda_bounds: Tuple[float, float] = (-25.0, 25.0),
        xatol: float = 1e-6,
        maxiter: int = 500,
        key_spec: KeySpec = KeySpec(),
        fit_trim_alpha=None,
        min_obs: int = 30,
        fit_maturities: Optional[List[float]] = None,
        maturity_match_tol: Optional[float] = None,
        eps: float = 1e-10,
        verbose: bool = True,
        penalty_value: float = 1e100,
        cache_spec: CacheSpec = CacheSpec(),
        behavioral=False,
        stock_df=None,
        **kwargs,
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
            **kwargs,
        )
        lo, hi = map(float, lambda_bounds)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("lambda_bounds must be finite and increasing.")
        self.lambda_bounds = (lo, hi)
        self.xatol = float(xatol)
        self.maxiter = int(maxiter)

    def _cache_params(self) -> dict:
        return {
            "lambda_bounds": self.lambda_bounds,
            "xatol": self.xatol,
            "maxiter": self.maxiter,
        }

    @staticmethod
    def _sigma_from_q_params(q_params: Dict[str, float]) -> float:
        for key in ("sigma", "vol", "volatility"):
            if key in q_params:
                sigma = float(q_params[key])
                if np.isfinite(sigma) and sigma > 0:
                    return sigma
        raise KeyError("Black--Scholes Q parameters must contain a positive 'sigma'.")

    def _physical_pdf_from_q_state(
        self,
        *,
        x_grid: np.ndarray,
        T: float,
        q_params: Dict[str, float],
        risk_params: np.ndarray,
        S0: float,
        r: float,
        q: float,
        info: dict,
    ) -> np.ndarray:
        x = np.asarray(x_grid, dtype=float).ravel()
        sigma = self._sigma_from_q_params(q_params)
        lam = float(np.asarray(risk_params, dtype=float).ravel()[0])

        var = sigma * sigma * float(T)
        if not np.isfinite(var) or var <= 0:
            return np.full_like(x, np.nan)

        mean = (lam - 0.5) * sigma * sigma * float(T)
        z = (x - mean) / np.sqrt(var)
        return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi * var)

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float):
        def objective(lam: float) -> float:
            total = 0.0
            for _, row in hist_T.iterrows():
                ll = self._row_log_score(row, np.array([lam]), T=float(T))
                if not np.isfinite(ll):
                    return self.penalty_value
                total += ll
            return float(-total)

        res = minimize_scalar(
            objective,
            bounds=self.lambda_bounds,
            method="bounded",
            options={"xatol": self.xatol, "maxiter": self.maxiter},
        )

        fitted = BlackScholesRiskPremiaFitted(
            lambda_return=float(res.x),
            T=float(T),
            loss=float(res.fun),
            success=bool(res.success),
            message=str(res.message),
        )
        diag = {
            "loss": float(res.fun),
            "loss_name": "negative_log_likelihood",
            "status": "success" if res.success else "failed",
            "message": str(res.message),
            "params": {"lambda_return": float(res.x)},
        }
        return fitted, diag

    def _risk_vector_from_fitted(self, fitted_model: BlackScholesRiskPremiaFitted) -> np.ndarray:
        return np.array([float(fitted_model.lambda_return)], dtype=float)

    def _risk_params_dict(self, fitted_model: BlackScholesRiskPremiaFitted) -> Dict[str, float]:
        return {"lambda_return": float(fitted_model.lambda_return)}
