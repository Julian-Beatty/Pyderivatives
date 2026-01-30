# pyderivatives/yield_curve/models/rezende_2011.py
from __future__ import annotations

from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize


from ..registry import register_model
from ..utils import build_day_grid, make_fit_window_mask

# -----------------------------------------------------------------------------
# De Rezende & Ferreira (2011/2013) Five-Factor (FF) extended Nelson–Siegel model
#
# Spot/zero yield curve (their eq. (4)):
#   y(tau) = b1
#          + b2 * (1 - e^{-λ1 τ})/(λ1 τ)
#          + b3 * (1 - e^{-λ2 τ})/(λ2 τ)
#          + b4 * ((1 - e^{-λ1 τ})/(λ1 τ) - e^{-λ1 τ})
#          + b5 * ((1 - e^{-λ2 τ})/(λ2 τ) - e^{-λ2 τ})
# -----------------------------------------------------------------------------

def _safe_L2(x: np.ndarray) -> np.ndarray:
    """(1 - exp(-x))/x with a stable Taylor fallback near 0."""
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) <= 1e-10
    out[~small] = (1.0 - np.exp(-x[~small])) / x[~small]
    # 1 - x/2 + x^2/6 - x^3/24 + ...
    xs = x[small]
    out[small] = 1.0 - xs / 2.0 + (xs * xs) / 6.0 - (xs * xs * xs) / 24.0
    return out


def _ff_loadings(tau: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    """
    Factor loadings matrix X (N x 5) for the FF model.
    Columns correspond to [b1, b2, b3, b4, b5].
    """
    tau = np.asarray(tau, float)
    lam1 = float(lam1)
    lam2 = float(lam2)

    x1 = lam1 * tau
    x2 = lam2 * tau

    L2_1 = _safe_L2(x1)                 # (1 - e^{-λ1 τ})/(λ1 τ)
    L2_2 = _safe_L2(x2)                 # (1 - e^{-λ2 τ})/(λ2 τ)
    E1 = np.exp(-x1)
    E2 = np.exp(-x2)

    # Curvature-like terms (same structure as Svensson curvature loadings)
    C1 = L2_1 - E1                      # (1 - e^{-λ1 τ})/(λ1 τ) - e^{-λ1 τ}
    C2 = L2_2 - E2                      # (1 - e^{-λ2 τ})/(λ2 τ) - e^{-λ2 τ}

    return np.column_stack([np.ones_like(tau), L2_1, L2_2, C1, C2])


def _ols_beta(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _fit_lambdas_profile_ls(
    tau: np.ndarray,
    y: np.ndarray,
    lam_bounds: Tuple[float, float] = (1e-4, 50.0),
    # keep λ1 < λ2 to reduce label-switching / identifiability issues
    enforce_order: bool = True,
) -> tuple[float, float, np.ndarray, float]:
    lo, hi = lam_bounds

    def obj(log_lams: np.ndarray) -> float:
        lam1 = float(np.exp(log_lams[0]))
        lam2 = float(np.exp(log_lams[1]))

        if lam1 < lo or lam1 > hi or lam2 < lo or lam2 > hi:
            return 1e50
        if enforce_order and not (lam1 < lam2):
            return 1e50

        X = _ff_loadings(tau, lam1, lam2)
        beta = _ols_beta(y, X)
        resid = y - X @ beta
        return float(np.sum(resid * resid))

    # reasonable starting points
    x0 = np.array([np.log(0.10), np.log(0.30)], dtype=float)

    res = minimize(
        obj,
        x0=x0,
        bounds=[(np.log(lo), np.log(hi)), (np.log(lo), np.log(hi))],
        method="L-BFGS-B",
    )

    lam1_hat = float(np.exp(res.x[0]))
    lam2_hat = float(np.exp(res.x[1]))

    # if ordering not enforced (or optimizer found swapped), sort and refit
    if enforce_order and lam1_hat >= lam2_hat:
        lam1_hat, lam2_hat = sorted([lam1_hat, lam2_hat])

    X = _ff_loadings(tau, lam1_hat, lam2_hat)
    beta_hat = _ols_beta(y, X)
    resid = y - X @ beta_hat
    sse = float(np.sum(resid * resid))
    return lam1_hat, lam2_hat, beta_hat, sse


@register_model("rezende_2011")
def fit_rezende_2011(
    curve,  # YieldCurve initialized from a DataFrame with Date + maturity columns
    *,
    grid_days: Tuple[int, int],
    fit_days_window: Optional[Tuple[int, int]] = None,
    min_obs: int = 5,
    lam_bounds: Tuple[float, float] = (1e-4, 50.0),
    yields_in_percent: bool = True,
    return_params: bool = False,
    enforce_lambda_order: bool = True,
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fit the De Rezende & Ferreira five-factor (FF) model independently each date
    using available maturity columns, optionally restricting the fit to a days-window,
    then evaluate on a day-grid.

    Parameters
    ----------
    grid_days: (start,end) in days (inclusive). Output columns labeled 'k/365'.
    fit_days_window: (lo,hi) in days for which observed maturities are used to fit.
    min_obs: minimum # observed maturities after windowing.
    lam_bounds: bounds for (lambda1, lambda2) in 1/years (since tau is in years).
    yields_in_percent: if True, inputs are like 4.25 (percent) and outputs are percent.
    return_params: if True, also returns per-date (lambda1, lambda2, betas, sse, n_obs).
    enforce_lambda_order: if True, enforce lambda1 < lambda2.

    Returns
    -------
    fitted_df: Date + yields on grid (same units as input)
    params_df: (optional) Date, lambda1, lambda2, beta1..beta5, sse, n_obs
    """
    tau_grid, grid_cols = build_day_grid(grid_days)
    window_mask = make_fit_window_mask(curve.taus_years, fit_days_window)

    if int(np.sum(window_mask)) == 0:
        raise ValueError("fit_days_window excludes all available maturity columns.")

    Ymat = curve.df[curve.maturity_cols].to_numpy(float)
    scale = 0.01 if yields_in_percent else 1.0

    fitted_rows: list[np.ndarray] = []
    date_rows: list[pd.Timestamp] = []
    params_rows: list[list[object]] = []

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
                tau_obs,
                y_obs,
                lam_bounds=lam_bounds,
                enforce_order=enforce_lambda_order,
            )

            y_fit = (_ff_loadings(tau_grid, lam1_hat, lam2_hat) @ beta_hat) / scale

            fitted_rows.append(y_fit.astype(float))
            date_rows.append(date_i)

            if return_params:
                params_rows.append(
                    [
                        date_i,
                        lam1_hat,
                        lam2_hat,
                        beta_hat[0] / scale,
                        beta_hat[1] / scale,
                        beta_hat[2] / scale,
                        beta_hat[3] / scale,
                        beta_hat[4] / scale,
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
        columns=["Date", "lambda1", "lambda2", "beta1", "beta2", "beta3", "beta4", "beta5", "sse", "n_obs"],
    )
    return fitted_df, params_df
