import numpy as np
import pandas as pd


def _cdf_from_density(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Numerically integrate f over x to get a CDF on the same grid.
    Assumes x strictly increasing and f >= 0 (not necessarily normalized).
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
    Compute E[G | tail], where G = exp(r), using a log-return density fr(r).

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


def physical_moments_table(
    *,
    T_grid: np.ndarray,
    r_common: np.ndarray,
    pr_surface: np.ndarray,
) -> pd.DataFrame:
    """
    Moments of log return r under physical density p_r(r|T),
    PLUS tail risk/reward computed in GROSS-return space G = exp(r).

    Inputs
    ------
    T_grid     : (nT,) maturities in YEARS
    r_common   : (nr,) log-return grid
    pr_surface : (nT, nr) density in r-space evaluated on r_common

    Returns columns:
      T, mean_r, var_r, vol_r, vol_ann_r, skew_r, kurt_r, area_pr,
      var95, cvar97_5, tailgain95, ctg97_5

    Tail conventions (gross space):
      - Gross return: G = exp(r) = S_T / S_0
      - Define gross shortfall loss: L_G = 1 - G
      - var95      = VaR_95(L_G)      = 1 - Q_G(0.05)
      - cvar97_5   = CVaR_97.5(L_G)   = E[1 - G | G <= Q_G(0.025)]
      - tailgain95 = Q_G(0.95) - 1
      - ctg97_5    = E[G - 1 | G >= Q_G(0.975)]
    """
    T = np.asarray(T_grid, float).ravel()
    r = np.asarray(r_common, float).ravel()
    pr = np.asarray(pr_surface, float)

    if pr.shape != (T.size, r.size):
        raise ValueError("pr_surface must have shape (len(T_grid), len(r_common)).")
    if r.size >= 2 and np.any(np.diff(r) <= 0):
        raise ValueError("r_common must be strictly increasing.")

    rows = []
    for j in range(T.size):
        pj = pr[j, :]
        mask = np.isfinite(r) & np.isfinite(pj)
        if mask.sum() < 10:
            rows.append(
                dict(
                    T=float(T[j]),
                    mean_r=np.nan,
                    var_r=np.nan,
                    vol_r=np.nan,
                    vol_ann_r=np.nan,
                    skew_r=np.nan,
                    kurt_r=np.nan,
                    area_pr=np.nan,
                    var95=np.nan,
                    cvar97_5=np.nan,
                    tailgain95=np.nan,
                    ctg97_5=np.nan,
                )
            )
            continue

        rm = r[mask]
        pm = pj[mask]

        area = float(np.trapezoid(pm, rm))
        if not np.isfinite(area) or area <= 0:
            rows.append(
                dict(
                    T=float(T[j]),
                    mean_r=np.nan,
                    var_r=np.nan,
                    vol_r=np.nan,
                    vol_ann_r=np.nan,
                    skew_r=np.nan,
                    kurt_r=np.nan,
                    area_pr=area,
                    var95=np.nan,
                    cvar97_5=np.nan,
                    tailgain95=np.nan,
                    ctg97_5=np.nan,
                )
            )
            continue

        # normalize density in r-space
        pm = pm / area

        # ---- central moments in log-return space ----
        mu = float(np.trapezoid(rm * pm, rm))
        m2 = float(np.trapezoid(((rm - mu) ** 2) * pm, rm))
        vol = float(np.sqrt(max(m2, 0.0)))

        m3 = float(np.trapezoid(((rm - mu) ** 3) * pm, rm))
        m4 = float(np.trapezoid(((rm - mu) ** 4) * pm, rm))

        denom3 = vol**3 + 1e-300
        denom4 = vol**4 + 1e-300
        skew = float(m3 / denom3)
        kurt = float(m4 / denom4)  # raw kurtosis (not excess)

        # annualize: Var(r_T) ≈ T * Var_annual  =>  vol_ann = sqrt(var / T)
        Tj = float(T[j])
        vol_ann = float(np.sqrt(m2 / max(Tj, 1e-12)))

        # ---- tail measures in GROSS-return space ----
        # Compute r-quantiles and transform since G = exp(r) is monotone.
        cdf_r = _cdf_from_density(rm, pm)

        qr_05 = _quantile_from_cdf(rm, cdf_r, 0.05)
        qr_025 = _quantile_from_cdf(rm, cdf_r, 0.025)
        qr_95 = _quantile_from_cdf(rm, cdf_r, 0.95)
        qr_975 = _quantile_from_cdf(rm, cdf_r, 0.975)

        qG_05 = float(np.exp(qr_05)) if np.isfinite(qr_05) else np.nan
        qG_025 = float(np.exp(qr_025)) if np.isfinite(qr_025) else np.nan
        qG_95 = float(np.exp(qr_95)) if np.isfinite(qr_95) else np.nan
        qG_975 = float(np.exp(qr_975)) if np.isfinite(qr_975) else np.nan

        # VaR95 of gross shortfall loss L_G = 1 - G
        var95 = float(1.0 - qG_05) if np.isfinite(qG_05) else np.nan

        # CVaR97.5: 1 - E[G | G <= Q_G(0.025)] = 1 - E[exp(r) | r <= qr_025]
        EG_left_025 = _tail_conditional_mean_of_gross(
            rm, pm, tail="left", rq=qr_025, eps=1e-30
        )
        cvar97_5 = float(1.0 - EG_left_025) if np.isfinite(EG_left_025) else np.nan

        # Upside analogs
        tailgain95 = float(qG_95 - 1.0) if np.isfinite(qG_95) else np.nan

        EG_right_975 = _tail_conditional_mean_of_gross(
            rm, pm, tail="right", rq=qr_975, eps=1e-30
        )
        ctg97_5 = float(EG_right_975 - 1.0) if np.isfinite(EG_right_975) else np.nan

        rows.append(
            dict(
                T=Tj,
                mean_r=mu,
                var_r=m2,
                vol_r=vol,
                vol_ann_r=vol_ann,
                skew_r=skew,
                kurt_r=kurt,
                area_pr=1.0,  # after normalization
                var95=var95,
                cvar97_5=cvar97_5,
                tailgain95=tailgain95,
                ctg97_5=ctg97_5,
            )
        )

    return pd.DataFrame(rows)
