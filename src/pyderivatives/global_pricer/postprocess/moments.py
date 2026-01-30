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
    mu = float(np.trapz(x * f, x))
    m2 = float(np.trapz(((x - mu) ** 2) * f, x))
    m3 = float(np.trapz(((x - mu) ** 3) * f, x))
    m4 = float(np.trapz(((x - mu) ** 4) * f, x))

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
      T, mean, var, vol, vol_ann, skew, kurt, area_q, area_fr
    where moments are for r = log(K/S0).
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
        area_q = float(np.trapz(qi, K)) if K.size >= 2 else np.nan

        # transform to log-return density: f_r(r) = q_K(K) * K  (since dK/dr = K)
        fr = qi * K

        # area under f_r(r) dr (should match area_q up to numerics)
        area_fr = float(np.trapz(fr, r_grid)) if r_grid.size >= 2 else np.nan

        if cfg.renormalize:
            denom = max(area_fr, cfg.eps) if np.isfinite(area_fr) else cfg.eps
            fr = fr / denom
            area_fr_norm = float(np.trapz(fr, r_grid))
        else:
            area_fr_norm = area_fr

        m = _central_moments_from_density(r_grid, fr)

        # annualize log-return vol: sqrt(var / T)  (standard for log-returns)
        if np.isfinite(Ti) and Ti > 0 and np.isfinite(m["var"]):
            vol_ann = float(np.sqrt(max(m["var"], 0.0) / Ti))
        else:
            vol_ann = np.nan

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
            )
        )

    return pd.DataFrame(rows)
