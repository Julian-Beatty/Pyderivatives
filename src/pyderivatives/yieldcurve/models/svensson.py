import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
from scipy.optimize import minimize

from ..registry import register_model


def _L2(tau: np.ndarray, lam: float) -> np.ndarray:
    """
    L2(t) = (1 - exp(-lam*t)) / (lam*t), with stable limit at 0.
    """
    tau = np.asarray(tau, float)
    lam = float(lam)
    x = lam * tau
    return np.where(np.abs(x) > 1e-10, (1.0 - np.exp(-x)) / x, 1.0 - x / 2.0 + x * x / 6.0)


def _L3(tau: np.ndarray, lam: float) -> np.ndarray:
    """
    L3(t) = L2(t) - exp(-lam*t)
    """
    tau = np.asarray(tau, float)
    lam = float(lam)
    return _L2(tau, lam) - np.exp(-lam * tau)


def _nss_design(tau: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    """
    Svensson / NSS yield curve:

      y(t) = β0
           + β1 * L2(t, λ1)
           + β2 * L3(t, λ1)
           + β3 * L3(t, λ2)

    So the design matrix columns are:
      [1, L2(λ1), L3(λ1), L3(λ2)]
    """
    tau = np.asarray(tau, float)
    return np.column_stack([
        np.ones_like(tau),
        _L2(tau, lam1),
        _L3(tau, lam1),
        _L3(tau, lam2),
    ])


def _ols_beta(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _fit_lambdas_profile_ls(
    tau: np.ndarray,
    y: np.ndarray,
    lam1_bounds: Tuple[float, float] = (1e-4, 50.0),
    lam2_bounds: Tuple[float, float] = (1e-4, 50.0),
) -> Tuple[float, float, np.ndarray, float]:
    """
    Profile least squares:
      - optimize over (λ1, λ2) only (in log-space)
      - for each (λ1, λ2), solve β by OLS
    Returns (λ1_hat, λ2_hat, β_hat, SSE)
    """
    lo1, hi1 = lam1_bounds
    lo2, hi2 = lam2_bounds

    def obj(log_lams: np.ndarray) -> float:
        lam1 = float(np.exp(log_lams[0]))
        lam2 = float(np.exp(log_lams[1]))
        if not (lo1 <= lam1 <= hi1 and lo2 <= lam2 <= hi2):
            return 1e50
        X = _nss_design(tau, lam1, lam2)
        beta = _ols_beta(y, X)
        resid = y - X @ beta
        return float(np.sum(resid * resid))

    res = minimize(
        obj,
        x0=np.array([np.log(0.5), np.log(2.0)]),
        bounds=[(np.log(lo1), np.log(hi1)), (np.log(lo2), np.log(hi2))],
        method="L-BFGS-B",
    )

    lam1_hat = float(np.exp(res.x[0]))
    lam2_hat = float(np.exp(res.x[1]))
    X = _nss_design(tau, lam1_hat, lam2_hat)
    beta_hat = _ols_beta(y, X)
    resid = y - X @ beta_hat
    sse = float(np.sum(resid * resid))
    return lam1_hat, lam2_hat, beta_hat, sse


@register_model("svensson")
def fit_svensson(
    curve,  # YieldCurve container
    *,
    grid_days: Tuple[int, int],
    fit_days_window: Optional[Tuple[int, int]] = None,
    min_obs: int = 4,
    lam1_bounds: Tuple[float, float] = (1e-4, 50.0),
    lam2_bounds: Tuple[float, float] = (1e-4, 50.0),
    yields_in_percent: bool = True,
    return_params: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fit Svensson/NSS independently for each Date and evaluate on a day grid.

    Parameters
    ----------
    grid_days : (start,end) days inclusive for evaluation; output columns 'k/365'
    fit_days_window : optional (lo,hi) days for which maturities are used in fitting
    min_obs : minimum observed maturities after windowing (default 4)
    lam1_bounds, lam2_bounds : bounds for λ1, λ2 (>0)
    yields_in_percent : True if data are in percent units
    return_params : if True, also return per-date parameters

    Returns
    -------
    fitted_df : Date + grid columns
    params_df (optional)
    """
    start_d, end_d = grid_days
    if start_d < 1 or end_d < start_d:
        raise ValueError("Require 1 <= start_d <= end_d.")

    days = np.arange(start_d, end_d + 1, dtype=int)
    tau_grid = days / 365.0
    grid_cols = [f"{d}/365" for d in days]

    # Build fit-window mask
    if fit_days_window is not None:
        lo_d, hi_d = fit_days_window
        if lo_d < 1 or hi_d < lo_d:
            raise ValueError("fit_days_window must satisfy 1 <= lo <= hi.")
        lo_tau, hi_tau = lo_d / 365.0, hi_d / 365.0
        window_mask = (curve.taus_years >= lo_tau) & (curve.taus_years <= hi_tau)
    else:
        window_mask = np.ones_like(curve.taus_years, dtype=bool)

    if int(np.sum(window_mask)) == 0:
        raise ValueError("fit_days_window excludes all available maturity columns.")

    Ymat = curve.df[curve.maturity_cols].to_numpy(float)
    scale = 0.01 if yields_in_percent else 1.0

    fitted_rows = []
    date_rows = []
    params_rows = []

    for i in range(len(curve.df)):
        date_i = curve.df.loc[i, "Date"]
        y_row = Ymat[i, :]
        mask = np.isfinite(y_row) & window_mask
        n_obs = int(np.sum(mask))
        if n_obs < min_obs:
            continue

        tau_obs = curve.taus_years[mask]
        y_obs = (y_row[mask] * scale).astype(float)

        try:
            lam1_hat, lam2_hat, beta_hat, sse = _fit_lambdas_profile_ls(
                tau_obs, y_obs, lam1_bounds=lam1_bounds, lam2_bounds=lam2_bounds
            )
            y_fit = (_nss_design(tau_grid, lam1_hat, lam2_hat) @ beta_hat) / scale

            fitted_rows.append(y_fit.astype(float))
            date_rows.append(date_i)

            params_rows.append(
                [
                    date_i,
                    lam1_hat,
                    lam2_hat,
                    beta_hat[0] / scale,
                    beta_hat[1] / scale,
                    beta_hat[2] / scale,
                    beta_hat[3] / scale,
                    sse,
                    n_obs,
                ]
            )
        except Exception:
            continue

    fitted_df = pd.DataFrame(fitted_rows, columns=grid_cols)
    fitted_df.insert(0, "Date", date_rows)

    if not return_params:
        return fitted_df

    params_df = pd.DataFrame(
        params_rows,
        columns=["Date", "lambda1", "lambda2", "beta0", "beta1", "beta2", "beta3", "sse", "n_obs"],
    )
    return fitted_df, params_df
