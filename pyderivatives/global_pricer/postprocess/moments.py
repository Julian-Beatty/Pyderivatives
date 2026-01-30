# pyderivatives/global_pricer/postprocess/moments.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MomentsConfig:
    """
    Moments of the LOG-RETURN density f_r(r), where r = log(K/S0).

    - If renormalize=True: we normalize f_r to integrate to 1 over r before moments.
      (Recommended, because BL can have small numerical area drift.)
    - If clip_negative=True: clamp negatives to 0 before renormalization.
    """
    renormalize: bool = True
    clip_negative: bool = True
    eps: float = 1e-30  # small floor to avoid divide-by-zero


def _central_moments_from_density(x: np.ndarray, f: np.ndarray) -> Dict[str, float]:
    """
    Compute mean/var/skew/kurtosis (non-excess) of x under density f on x-grid.
    Assumes f integrates to 1 over x.
    """
    mu = float(np.trapezoid(x * f, x))
    m2 = float(np.trapezoid(((x - mu) ** 2) * f, x))
    m3 = float(np.trapezoid(((x - mu) ** 3) * f, x))
    m4 = float(np.trapezoid(((x - mu) ** 4) * f, x))

    var = m2
    vol = float(np.sqrt(max(var, 0.0)))

    # Guard against division by 0 in skew/kurt
    if vol <= 0:
        skew = np.nan
        kurt = np.nan
    else:
        skew = float(m3 / (vol ** 3))
        kurt = float(m4 / (vol ** 4))

    return {"mean": mu, "var": var, "vol": vol, "skew": skew, "kurt": kurt}


def _cdf_from_density(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Numerically integrate f over x to get a CDF on the same grid.
    Assumes x strictly increasing and f >= 0 (not necessarily perfectly normalized).
    """
    x = np.asarray(x, float).ravel()
    f = np.asarray(f, float).ravel()
    if x.size != f.size:
        raise ValueError("x and f must have same length.")
    if x.size < 2:
        return np.array([np.nan], float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")

    dx = np.diff(x)
    c = np.empty_like(x)
    c[0] = 0.0
    c[1:] = np.cumsum(0.5 * (f[:-1] + f[1:]) * dx)
    return c


def _quantile_from_cdf(x: np.ndarray, cdf: np.ndarray, p: float) -> float:
    """
    Invert a (possibly not exactly normalized) CDF via interpolation.
    We scale by the final mass to handle tiny area drift.
    """
    x = np.asarray(x, float).ravel()
    cdf = np.asarray(cdf, float).ravel()
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    if x.size < 2 or cdf.size != x.size:
        return np.nan

    total = float(cdf[-1])
    if not np.isfinite(total) or total <= 0:
        return np.nan

    target = p * total
    cdf_m = np.maximum.accumulate(cdf)
    return float(np.interp(target, cdf_m, x))


def _tail_conditional_mean_of_gross(
    r: np.ndarray,
    fr: np.ndarray,
    *,
    tail: str,
    rq: float,
    eps: float = 1e-30,
) -> float:
    """
    Compute E[G | tail], where G = exp(r), using the log-return density fr(r).

    tail="left":  E[exp(r) | r <= rq]
    tail="right": E[exp(r) | r >= rq]
    """
    r = np.asarray(r, float).ravel()
    fr = np.asarray(fr, float).ravel()
    if r.size < 2 or r.size != fr.size:
        return np.nan

    if tail == "left":
        w = (r <= rq).astype(float)
    elif tail == "right":
        w = (r >= rq).astype(float)
    else:
        raise ValueError("tail must be 'left' or 'right'.")

    mass = float(np.trapezoid(fr * w, r))
    if not np.isfinite(mass) or mass <= eps:
        return np.nan

    num = float(np.trapezoid(np.exp(r) * fr * w, r))
    return float(num / mass)


def logreturn_moments_table(
    rnd_surface_K: np.ndarray,
    *,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    S0: float,
    cfg: Optional[MomentsConfig] = None,
) -> pd.DataFrame:
    """
    Build a per-maturity moments table for the LOG-RETURN density.

    Inputs
    ------
    rnd_surface_K : array (nT, nK)
        Risk-neutral density wrt strike, q_K(K|T). (This is what your BL produces.)
        It does NOT need to be perfectly normalized; we track area and can renormalize.
    K_grid : array (nK,)
        Increasing strike grid.
    T_grid : array (nT,)
        Maturities in years.
    S0 : float
        Spot/forward anchor used for r = log(K/S0).
    cfg : MomentsConfig
        Options for clipping/renormalization.

    Returns
    -------
    pandas.DataFrame with columns:
      T, mean, var, vol, vol_ann, skew, kurt, area_q, area_fr,
      var95, cvar97_5, tailgain95, ctg97_5

    where moments are for r = log(K/S0), but tail measures are computed in GROSS-return space
    G = exp(r) = S_T / S_0.

    Tail conventions (gross space):
      - Define gross shortfall loss L_G = 1 - G.
      - var95      = VaR_95(L_G)      = 1 - Q_G(0.05)
      - cvar97_5   = CVaR_97.5(L_G)   = E[1 - G | G <= Q_G(0.025)]
      - tailgain95 = Q_G(0.95) - 1
      - ctg97_5    = E[G - 1 | G >= Q_G(0.975)]
    """
    if cfg is None:
        cfg = MomentsConfig()

    qK = np.asarray(rnd_surface_K, float)
    K = np.asarray(K_grid, float).ravel()
    T = np.asarray(T_grid, float).ravel()
    S0 = float(S0)

    if qK.shape != (T.size, K.size):
        raise ValueError("rnd_surface_K must have shape (len(T_grid), len(K_grid)).")
    if np.any(np.diff(K) <= 0):
        raise ValueError("K_grid must be strictly increasing.")
    if S0 <= 0:
        raise ValueError("S0 must be > 0.")

    # log-return grid (aligned with K_grid)
    r_grid = np.log(K / S0)

    rows = []
    for i, Ti in enumerate(T):
        qi = qK[i, :].copy()

        # optional clamp negatives (BL numerical artifacts)
        if cfg.clip_negative:
            qi = np.maximum(qi, 0.0)

        # area under q_K(K) dK (diagnostic; should be near 1)
        area_q = float(np.trapezoid(qi, K)) if K.size >= 2 else np.nan

        # transform to log-return density: f_r(r) = q_K(K) * K  (since dK/dr = K)
        fr = qi * K

        # area under f_r(r) dr (should match area_q up to numerics)
        area_fr = float(np.trapezoid(fr, r_grid)) if r_grid.size >= 2 else np.nan

        if cfg.renormalize:
            denom = max(area_fr, cfg.eps) if np.isfinite(area_fr) else cfg.eps
            fr = fr / denom
            area_fr_norm = float(np.trapezoid(fr, r_grid))
        else:
            area_fr_norm = area_fr

        m = _central_moments_from_density(r_grid, fr)

        # annualize log-return vol: sqrt(var / T)  (standard for log-returns)
        if np.isfinite(Ti) and Ti > 0 and np.isfinite(m["var"]):
            vol_ann = float(np.sqrt(max(m["var"], 0.0) / Ti))
        else:
            vol_ann = np.nan

        # ---------- Tail measures (computed in GROSS-return space) ----------
        # Use r-quantiles since G=exp(r) is monotone; then transform.
        cdf_r = _cdf_from_density(r_grid, fr)

        qr_05 = _quantile_from_cdf(r_grid, cdf_r, 0.05)
        qr_025 = _quantile_from_cdf(r_grid, cdf_r, 0.025)
        qr_95 = _quantile_from_cdf(r_grid, cdf_r, 0.95)
        qr_975 = _quantile_from_cdf(r_grid, cdf_r, 0.975)

        qG_05 = float(np.exp(qr_05)) if np.isfinite(qr_05) else np.nan
        qG_025 = float(np.exp(qr_025)) if np.isfinite(qr_025) else np.nan
        qG_95 = float(np.exp(qr_95)) if np.isfinite(qr_95) else np.nan
        qG_975 = float(np.exp(qr_975)) if np.isfinite(qr_975) else np.nan

        # Loss is gross shortfall: L_G = 1 - G
        var95 = float(1.0 - qG_05) if np.isfinite(qG_05) else np.nan

        EG_left_025 = _tail_conditional_mean_of_gross(
            r_grid, fr, tail="left", rq=qr_025, eps=cfg.eps
        )
        cvar97_5 = float(1.0 - EG_left_025) if np.isfinite(EG_left_025) else np.nan

        # Upside analogs ("inverses")
        tailgain95 = float(qG_95 - 1.0) if np.isfinite(qG_95) else np.nan

        EG_right_975 = _tail_conditional_mean_of_gross(
            r_grid, fr, tail="right", rq=qr_975, eps=cfg.eps
        )
        ctg97_5 = float(EG_right_975 - 1.0) if np.isfinite(EG_right_975) else np.nan

        rows.append(
            dict(
                T=float(Ti),
                mean=m["mean"],
                var=m["var"],
                vol=m["vol"],
                vol_ann=vol_ann,
                skew=m["skew"],
                kurt=m["kurt"],          # non-excess kurtosis
                area_q=area_q,           # ∫ q_K dK
                area_fr=area_fr_norm,    # ∫ f_r dr (after optional renorm)
                var95=var95,
                cvar97_5=cvar97_5,
                tailgain95=tailgain95,
                ctg97_5=ctg97_5,
            )
        )

    return pd.DataFrame(rows)
