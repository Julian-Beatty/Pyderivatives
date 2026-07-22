from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .affine_heston_utils import (
    affine_heston_logreturn_cf,
    gauss_legendre_0U,
    invert_cf_to_pdf,
    translated_kou_jump_factor,
)
from .stochastic_base import StochasticRiskPremiaTransform
from ..config import CacheSpec, KeySpec
from ..registry import register_transform


@dataclass
class HestonKouRiskPremiaFitted:
    gamma_return: float
    gamma_variance: float
    T: float
    loss: float
    success: bool
    message: str


@register_transform("heston_kou_risk_premia")
@register_transform("hk_risk_premia")
class HestonKouRiskPremia(StochasticRiskPremiaTransform):
    """Two-parameter affine risk-premia transform for Heston--Kou.

    The transform estimates a return-risk premium and a variance-risk premium.
    The Kou jump law is left unchanged under P: jump intensity, up-jump
    probability, and both exponential decay rates remain at their daily Q
    estimates.
    """

    supported_models = ("heston_kou", "hestonkou", "hkde")

    def __init__(
        self,
        *,
        gamma_return_bounds: Tuple[float, float] = (-25.0, 25.0),
        gamma_variance_bounds: Tuple[float, float] = (-500.0, 10.0),
        x0: Tuple[float, float] = (2.0, -5.0),
        Umax: float = 250.0,
        n_quad: int = 256,
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
        self.bounds = [
            tuple(map(float, gamma_return_bounds)),
            tuple(map(float, gamma_variance_bounds)),
        ]
        self.x0 = np.asarray(x0, float)
        self.Umax = float(Umax)
        self.n_quad = int(n_quad)
        self.maxiter = int(maxiter)
        self._u, self._wu = gauss_legendre_0U(self.n_quad, self.Umax)

    def _cache_params(self) -> dict:
        return dict(
            bounds=self.bounds,
            x0=self.x0,
            Umax=self.Umax,
            n_quad=self.n_quad,
            maxiter=self.maxiter,
            jump_transform="unchanged",
            specification="diffusion_only",
        )

    @staticmethod
    def _q_params(q_params: Dict[str, float]):
        required = ("v0", "theta", "kappa", "sigma_v", "rho", "lam", "p_up", "eta1", "eta2")
        missing = [k for k in required if k not in q_params]
        if missing:
            raise KeyError(f"Heston--Kou Q parameter dictionary is missing {missing}.")
        vals = tuple(float(q_params[k]) for k in required)
        if vals[7] <= 1.0 or vals[8] <= 0.0:
            raise ValueError("Kou requires eta1 > 1 and eta2 > 0.")
        return vals

    def _physical_pdf_from_q_state(self, *, x_grid, T, q_params, risk_params, S0, r, q, info):
        v0, theta, kappa, sigma_v, rho, lam, p_up, eta1, eta2 = self._q_params(q_params)
        gamma_return, gamma_variance = map(float, np.asarray(risk_params).ravel())

        cf_h = affine_heston_logreturn_cf(
            self._u, T,
            v0=v0, theta_q=theta, kappa_q=kappa,
            sigma_v=sigma_v, rho=rho,
            gamma_return=gamma_return, gamma_variance=gamma_variance,
        )
        cf_j = translated_kou_jump_factor(
            self._u, T,
            lam=lam, p_up=p_up, eta1=eta1, eta2=eta2,
            jump_translation=0.0,
        )
        return invert_cf_to_pdf(x_grid, cf_h * cf_j, self._u, self._wu)

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float):
        def objective(z):
            z = np.asarray(z, float)
            total = 0.0
            for _, row in hist_T.iterrows():
                ll = self._row_log_score(row, z, T=float(T))
                if not np.isfinite(ll):
                    return self.penalty_value
                total += ll
            return -float(total)

        res = minimize(objective, self.x0, method="L-BFGS-B", bounds=self.bounds, options={"maxiter": self.maxiter})
        fitted = HestonKouRiskPremiaFitted(
            gamma_return=float(res.x[0]), gamma_variance=float(res.x[1]),
            T=float(T), loss=float(res.fun),
            success=bool(res.success), message=str(res.message),
        )
        diag = {
            "loss": float(res.fun), "loss_name": "negative_log_likelihood",
            "status": "success" if res.success else "failed", "message": str(res.message),
            "params": {
                "gamma_return": float(res.x[0]),
                "gamma_variance": float(res.x[1]),
                "jump_translation": 0.0,
            },
        }
        return fitted, diag

    def _risk_vector_from_fitted(self, fitted_model):
        return np.array([fitted_model.gamma_return, fitted_model.gamma_variance], float)

    def _risk_params_dict(self, fitted_model):
        return {
            "gamma_return": float(fitted_model.gamma_return),
            "gamma_variance": float(fitted_model.gamma_variance),
            "jump_translation": 0.0,
        }
