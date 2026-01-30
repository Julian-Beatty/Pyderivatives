import numpy as np
import pandas as pd

def physical_moments_table(
    *, 
    T_grid: np.ndarray,
    r_common: np.ndarray,
    pr_surface: np.ndarray
) -> pd.DataFrame:
    """
    Moments of log return r under physical density p_r(r|T).

    Inputs
    ------
    T_grid     : (nT,) maturities in YEARS
    r_common   : (nr,) log-return grid
    pr_surface : (nT, nr) density in r-space evaluated on r_common

    Returns columns:
      T, mean_r, var_r, vol_r, vol_ann_r, skew_r, kurt_r, area_pr
    """
    T = np.asarray(T_grid, float).ravel()
    r = np.asarray(r_common, float).ravel()
    pr = np.asarray(pr_surface, float)

    rows = []
    for j in range(T.size):
        pj = pr[j, :]
        mask = np.isfinite(r) & np.isfinite(pj)
        if mask.sum() < 10:
            rows.append(dict(
                T=float(T[j]),
                mean_r=np.nan, var_r=np.nan, vol_r=np.nan, vol_ann_r=np.nan,
                skew_r=np.nan, kurt_r=np.nan, area_pr=np.nan
            ))
            continue

        rm = r[mask]
        pm = pj[mask]

        area = np.trapz(pm, rm)
        if not np.isfinite(area) or area <= 0:
            rows.append(dict(
                T=float(T[j]),
                mean_r=np.nan, var_r=np.nan, vol_r=np.nan, vol_ann_r=np.nan,
                skew_r=np.nan, kurt_r=np.nan, area_pr=area
            ))
            continue

        pm = pm / area

        mu = np.trapz(rm * pm, rm)
        m2 = np.trapz(((rm - mu) ** 2) * pm, rm)
        vol = np.sqrt(max(m2, 0.0))

        m3 = np.trapz(((rm - mu) ** 3) * pm, rm)
        m4 = np.trapz(((rm - mu) ** 4) * pm, rm)

        denom3 = (vol ** 3 + 1e-300)
        denom4 = (vol ** 4 + 1e-300)
        skew = m3 / denom3
        kurt = m4 / denom4   # raw kurtosis (not excess)

        # annualize: Var(r_T) ≈ T * Var_annual  =>  vol_ann = sqrt(var / T)
        Tj = float(T[j])
        vol_ann = np.sqrt(m2 / max(Tj, 1e-12))

        rows.append(dict(
            T=Tj,
            mean_r=float(mu),
            var_r=float(m2),
            vol_r=float(vol),
            vol_ann_r=float(vol_ann),
            skew_r=float(skew),
            kurt_r=float(kurt),
            area_pr=float(1.0),
        ))

    return pd.DataFrame(rows)
