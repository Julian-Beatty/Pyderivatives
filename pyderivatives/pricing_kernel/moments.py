from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import _as_1d, _cdf_from_density
def physical_moments_table(
    *,
    T_grid: np.ndarray,
    r_grid: np.ndarray,
    physical_lr_surface: np.ndarray,
) -> pd.DataFrame:
    """
    Compute moments and tail quantities from a log-return density surface.

    Inputs
    ------
    T_grid:
        Maturities, usually in years.
    r_grid:
        Log-return grid.
    physical_lr_surface:
        Density in log-return space with shape (len(T_grid), len(r_grid)).

    Returns
    -------
    DataFrame with mean, variance, volatility, annualized volatility,
    skewness, kurtosis, density area, and selected gross-return tail measures.
    """
    T = _as_1d(T_grid)
    r = _as_1d(r_grid)
    P = np.asarray(physical_lr_surface, dtype=float)

    if P.shape != (T.size, r.size):
        raise ValueError("physical_lr_surface must have shape (len(T_grid), len(r_grid)).")

    rows = []
    G = np.exp(r)

    for j, Tj in enumerate(T):
        f = np.asarray(P[j, :], dtype=float)
        m = np.isfinite(r) & np.isfinite(f) & (f >= 0)

        if m.sum() < 10:
            rows.append({"T": float(Tj), "area": np.nan})
            continue

        rr = r[m]
        ff = f[m]

        area = float(np.trapezoid(ff, rr))
        if not np.isfinite(area) or area <= 0:
            rows.append({"T": float(Tj), "area": area})
            continue

        ff = ff / area
        mu = float(np.trapezoid(rr * ff, rr))
        var = float(np.trapezoid(((rr - mu) ** 2) * ff, rr))
        vol = float(np.sqrt(var)) if var >= 0 else np.nan
        vol_ann = float(vol / np.sqrt(float(Tj))) if np.isfinite(vol) and Tj > 0 else np.nan

        if np.isfinite(var) and var > 0:
            sd = np.sqrt(var)
            skew = float(np.trapezoid(((rr - mu) / sd) ** 3 * ff, rr))
            kurt = float(np.trapezoid(((rr - mu) / sd) ** 4 * ff, rr))
        else:
            skew = np.nan
            kurt = np.nan

        cdf = _cdf_from_density(rr, ff, eps=1e-14)
        if np.all(np.isfinite(cdf)):
            q025_r = float(np.interp(0.025, cdf, rr))
            q05_r = float(np.interp(0.05, cdf, rr))
            q95_r = float(np.interp(0.95, cdf, rr))
            q975_r = float(np.interp(0.975, cdf, rr))

            var95 = 1.0 - float(np.exp(q05_r))
            tailgain95 = float(np.exp(q95_r) - 1.0)

            left = rr <= q025_r
            right = rr >= q975_r

            left_mass = float(np.trapezoid(ff[left], rr[left])) if left.sum() >= 2 else np.nan
            right_mass = float(np.trapezoid(ff[right], rr[right])) if right.sum() >= 2 else np.nan

            if np.isfinite(left_mass) and left_mass > 0:
                cvar97_5 = float(np.trapezoid((1.0 - np.exp(rr[left])) * ff[left], rr[left]) / left_mass)
            else:
                cvar97_5 = np.nan

            if np.isfinite(right_mass) and right_mass > 0:
                ctg97_5 = float(np.trapezoid((np.exp(rr[right]) - 1.0) * ff[right], rr[right]) / right_mass)
            else:
                ctg97_5 = np.nan
        else:
            var95 = cvar97_5 = tailgain95 = ctg97_5 = np.nan

        rows.append(
            {
                "T": float(Tj),
                "mean_r": mu,
                "var_r": var,
                "vol_r": vol,
                "vol_ann_r": vol_ann,
                "skew_r": skew,
                "kurt_r": kurt,
                "area": area,
                "var95": var95,
                "cvar97_5": cvar97_5,
                "tailgain95": tailgain95,
                "ctg97_5": ctg97_5,
            }
        )

    return pd.DataFrame(rows)