from __future__ import annotations

from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..registry import register_model
from ..utils import build_day_grid, make_fit_window_mask


def _ns_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """
    Nelson–Siegel:
      y(t)=b1 + b2 * L2(t) + b3 * L3(t)
    """
    tau = np.asarray(tau, float)
    lam = float(lam)
    x = lam * tau
    L2 = np.where(np.abs(x) > 1e-10, (1.0 - np.exp(-x)) / x, 1.0 - x / 2.0 + x * x / 6.0)
    L3 = L2 - np.exp(-x)
    return np.column_stack([np.ones_like(tau), L2, L3])


def _ols_beta(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _fit_lambda_profile_ls(
    tau: np.ndarray,
    y: np.ndarray,
    lam_bounds: Tuple[float, float] = (1e-4, 50.0),
) -> tuple[float, np.ndarray, float]:
    lo, hi = lam_bounds

    def obj(log_lam: np.ndarray) -> float:
        lam = float(np.exp(log_lam[0]))
        if lam < lo or lam > hi:
            return 1e50
        X = _ns_loadings(tau, lam)
        beta = _ols_beta(y, X)
        resid = y - X @ beta
        return float(np.sum(resid * resid))

    res = minimize(
        obj,
        x0=np.array([np.log(0.5)]),
        bounds=[(np.log(lo), np.log(hi))],
        method="L-BFGS-B",
    )

    lam_hat = float(np.exp(res.x[0]))
    X = _ns_loadings(tau, lam_hat)
    beta_hat = _ols_beta(y, X)
    resid = y - X @ beta_hat
    sse = float(np.sum(resid * resid))
    return lam_hat, beta_hat, sse


@register_model("nelson_siegel")
def fit_nelson_siegel(
    curve,  # YieldCurve
    *,
    grid_days: Tuple[int, int],
    fit_days_window: Optional[Tuple[int, int]] = None,
    min_obs: int = 3,
    lam_bounds: Tuple[float, float] = (1e-4, 50.0),
    yields_in_percent: bool = True,
    return_params: bool = False,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fit Nelson–Siegel independently each date using available maturity columns,
    optionally restricting the fit to a days-window, then evaluate on a day-grid.

    Parameters
    ----------
    grid_days: (start,end) in days (inclusive). Output columns labeled 'k/365'.
    fit_days_window: (lo,hi) in days for which observed maturities are used to fit.
    min_obs: minimum # observed maturities after windowing.
    yields_in_percent: if True, inputs are like 4.25 (percent) and outputs are percent.
    return_params: if True, also returns per-date (lambda, betas, sse, n_obs).
    """
    tau_grid, grid_cols = build_day_grid(grid_days)
    window_mask = make_fit_window_mask(curve.taus_years, fit_days_window)

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
            lam_hat, beta_hat, sse = _fit_lambda_profile_ls(tau_obs, y_obs, lam_bounds=lam_bounds)
            y_fit = (_ns_loadings(tau_grid, lam_hat) @ beta_hat) / scale

            fitted_rows.append(y_fit.astype(float))
            date_rows.append(date_i)

            params_rows.append(
                [date_i, lam_hat, beta_hat[0] / scale, beta_hat[1] / scale, beta_hat[2] / scale, sse, n_obs]
            )
        except Exception:
            continue

    fitted_df = pd.DataFrame(fitted_rows, columns=grid_cols)
    fitted_df.insert(0, "Date", date_rows)

    if not return_params:
        return fitted_df

    params_df = pd.DataFrame(
        params_rows,
        columns=["Date", "lambda", "beta1", "beta2", "beta3", "sse", "n_obs"],
    )
    return fitted_df, params_df
