from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import MeasureTransform
from ..config import CacheSpec, KeySpec
from ..utils import _as_1d, _cdf_from_density, _safe_interp, _trapz_normalize_density


class StochasticRiskPremiaTransform(MeasureTransform):
    """Common base for Q-to-P transforms defined through stochastic dynamics.

    Subclasses estimate a small vector of risk-premium parameters from realized
    returns. For each historical date, the daily option-implied Q parameters are
    treated as observed conditioning information. A candidate risk-premium vector
    maps those Q parameters into a physical characteristic function or density.

    The public API remains the ordinary MeasureTransform API::

        transform.fit(rnd_history_dict, stock_df)
        out = transform.transform_rnd(rnd_history_dict[some_date])
    """

    supported_models: Tuple[str, ...] = ()

    def __init__(
        self,
        *,
        key_spec: KeySpec = KeySpec(),
        fit_trim_alpha=None,
        min_obs: int = 30,
        fit_maturities=None,
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

    @staticmethod
    def _normalize_model_name(name: Any) -> str:
        return str(name or "").strip().lower().replace("-", "_")

    def _validate_row_model(self, row: pd.Series) -> None:
        model = self._normalize_model_name(row.get("model", None))
        if self.supported_models and model not in self.supported_models:
            raise ValueError(
                f"{self.__class__.__name__} requires model in {self.supported_models}; "
                f"received {model!r}."
            )
        if not isinstance(row.get("params", None), dict):
            raise ValueError("Historical row is missing the daily Q parameter dictionary.")

    def _row_log_score(self, row: pd.Series, risk_params: np.ndarray, *, T: float) -> float:
        self._validate_row_model(row)
        x = _as_1d(row["x_grid"])
        f_p = self._physical_pdf_from_q_state(
            x_grid=x,
            T=float(T),
            q_params=dict(row["params"]),
            risk_params=_as_1d(risk_params),
            S0=float(row.get("S0", np.nan)),
            r=float(row.get("r", np.nan)),
            q=float(row.get("q", np.nan)),
            info={"model": row.get("model"), "meta": row.get("meta", {})},
        )
        f_p = _trapz_normalize_density(x, f_p, eps=self.eps)
        if not np.all(np.isfinite(f_p)):
            return -np.inf

        realized = float(row["realized_return"])
        dens = float(_safe_interp(realized, x, f_p))
        if not np.isfinite(dens) or dens <= self.eps:
            return -np.inf
        return float(np.log(dens))

    def _transform_surface_with_model(
        self,
        fitted_model: Any,
        x_grid: np.ndarray,
        f_q: np.ndarray,
        F_q: np.ndarray,
        *,
        T: float,
        info: dict,
    ) -> dict:
        model = self._normalize_model_name(info.get("model", None))
        if self.supported_models and model not in self.supported_models:
            raise ValueError(
                f"{self.__class__.__name__} requires model in {self.supported_models}; "
                f"received {model!r}."
            )

        q_params = info.get("params", None)
        if not isinstance(q_params, dict):
            raise ValueError("Input RND dictionary is missing info['params'].")

        S0 = info.get("S0", info.get("s0", None))
        if S0 is None:
            for key in self.key_spec.spot_keys:
                if key in info and info[key] is not None:
                    S0 = info[key]
                    break

        risk_params = self._risk_vector_from_fitted(fitted_model)
        f_p = self._physical_pdf_from_q_state(
            x_grid=_as_1d(x_grid),
            T=float(T),
            q_params=dict(q_params),
            risk_params=risk_params,
            S0=np.nan if S0 is None else float(S0),
            r=float(info.get("r", np.nan)),
            q=float(info.get("q", np.nan)),
            info=info,
        )
        f_p = _trapz_normalize_density(x_grid, f_p, eps=self.eps)
        if not np.all(np.isfinite(f_p)):
            raise ValueError("Physical density could not be normalized.")

        return {
            "f_p": f_p,
            "F_p": _cdf_from_density(x_grid, f_p, eps=self.eps),
            "risk_params": self._risk_params_dict(fitted_model),
        }

    @abstractmethod
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
        """Return a physical density in log-return space on ``x_grid``."""
        raise NotImplementedError

    @abstractmethod
    def _risk_vector_from_fitted(self, fitted_model: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _risk_params_dict(self, fitted_model: Any) -> Dict[str, float]:
        raise NotImplementedError
