import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq
from scipy.stats import norm


# -------------------------
# Black–Scholes call + Vega
# -------------------------
def bs_call_price(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return float(S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_vega(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return float(S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


# -------------------------
# IV solver (Newton -> Brent)
# -------------------------
def implied_vol_call_newton_brent(
    C: float,
    *,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    # solver controls
    sigma_init: float = 0.25,
    sigma_lo: float = 1e-8,
    sigma_hi: float = 6.0,
    newton_max_iter: int = 50,
    newton_tol: float = 1e-8,
    vega_floor: float = 1e-10,
    brent_maxiter: int = 200,
    # OPTIONAL GUARDS (set to 0/None to disable)
    time_value_floor: float = 0.0,     # e.g. 1e-6
    reject_low_vega: float = 0.0,      # e.g. 1e-6 (post-solve)
) -> float:
    """
    Robust IV for CALL:
      1) Arbitrage bounds check
      2) Newton iterations (fast)
      3) Brent fallback (guaranteed if bracket exists)
    Returns np.nan if not solvable.

    Optional guards:
      - time_value_floor: if C - intrinsic <= floor -> return np.nan
      - reject_low_vega: after solving, if vega < reject_low_vega -> return np.nan
    """
    if T <= 0 or S0 <= 0 or K <= 0 or not np.isfinite(C):
        return np.nan

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    lower = max(S0 * disc_q - K * disc_r, 0.0)
    upper = S0 * disc_q
    if not (lower - 1e-10 <= C <= upper + 1e-10):
        return np.nan

    # optional: skip near-intrinsic prices (IV ill-posed)
    if time_value_floor > 0.0:
        if (C - lower) <= float(time_value_floor):
            return np.nan

    def f(sig: float) -> float:
        return bs_call_price(S0, K, T, r, q, sig) - C

    # ---- Newton
    sig = float(np.clip(sigma_init, sigma_lo, sigma_hi))
    for _ in range(int(newton_max_iter)):
        price = bs_call_price(S0, K, T, r, q, sig)
        diff = price - C
        if abs(diff) <= newton_tol:
            # optional: post-check vega
            if reject_low_vega > 0.0:
                v = _bs_vega(S0, K, T, r, q, sig)
                if v < float(reject_low_vega):
                    return np.nan
            return float(sig)

        v = _bs_vega(S0, K, T, r, q, sig)
        if v < vega_floor:
            break

        sig_new = sig - diff / v
        if not np.isfinite(sig_new):
            break
        sig = float(np.clip(sig_new, sigma_lo, sigma_hi))

    # ---- Brent fallback
    f_lo = f(sigma_lo)
    f_hi = f(sigma_hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return np.nan

    if f_lo * f_hi > 0:
        for hi in (10.0, 20.0):
            f_hi2 = f(hi)
            if f_lo * f_hi2 <= 0:
                sig = float(brentq(f, sigma_lo, hi, maxiter=int(brent_maxiter)))
                if reject_low_vega > 0.0:
                    v = _bs_vega(S0, K, T, r, q, sig)
                    if v < float(reject_low_vega):
                        return np.nan
                return sig
        return np.nan

    sig = float(brentq(f, sigma_lo, sigma_hi, maxiter=int(brent_maxiter)))

    if reject_low_vega > 0.0:
        v = _bs_vega(S0, K, T, r, q, sig)
        if v < float(reject_low_vega):
            return np.nan

    return sig


# -------------------------
# IV surface
# -------------------------
@dataclass(frozen=True)
class IVConfig:
    sigma_init: float = 0.5
    sigma_lo: float = 1e-8
    sigma_hi: float = 4.0
    newton_max_iter: int = 50
    newton_tol: float = 1e-5
    vega_floor: float = 1e-10
    brent_maxiter: int = 100
    # optional guards
    time_value_floor: float = 1e-4   # try 1e-6
    reject_low_vega: float = 1e-4    # try 1e-6


def iv_surface_from_calls(
    C_fit: np.ndarray,
    *,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    S0: float,
    r: float,
    q: float = 0.0,
    cfg: IVConfig = IVConfig(),
) -> np.ndarray:
    C_fit = np.asarray(C_fit, float)
    K_grid = np.asarray(K_grid, float).ravel()
    T_grid = np.asarray(T_grid, float).ravel()

    if C_fit.shape != (T_grid.size, K_grid.size):
        raise ValueError("C_fit must have shape (len(T_grid), len(K_grid)).")

    out = np.full_like(C_fit, np.nan, dtype=float)

    for i, T in enumerate(T_grid):
        Ti = float(T)
        for j, K in enumerate(K_grid):
            out[i, j] = implied_vol_call_newton_brent(
                float(C_fit[i, j]),
                S0=float(S0),
                K=float(K),
                T=Ti,
                r=float(r),
                q=float(q),
                sigma_init=float(cfg.sigma_init),
                sigma_lo=float(cfg.sigma_lo),
                sigma_hi=float(cfg.sigma_hi),
                newton_max_iter=int(cfg.newton_max_iter),
                newton_tol=float(cfg.newton_tol),
                vega_floor=float(cfg.vega_floor),
                brent_maxiter=int(cfg.brent_maxiter),
                time_value_floor=float(cfg.time_value_floor),
                reject_low_vega=float(cfg.reject_low_vega),
            )
    return out
import numpy as np
from typing import Dict, Optional


# def iv_surface_to_delta_surfaces(
#     iv_surface: np.ndarray,
#     *,
#     K_grid: np.ndarray,
#     T_grid: np.ndarray,
#     S0: float,
#     r: float,
#     q: float = 0.0,
#     delta_grid: Optional[np.ndarray] = None,   # e.g. np.linspace(0.05, 0.95, 61)
#     delta_eps: float = 1e-4,
#     min_points: int = 6,
# ) -> Dict[str, np.ndarray]:
#     """
#     Convert an IV surface σ(K,T) into delta-space surfaces and ALSO compute a "delta skew surface":
#         skew(T,Δ) = (σ_call(T,Δ) - σ_put(T,Δ)) / σ_ATM(T)

#     where σ_ATM(T) is taken as the **ATM-forward** vol, i.e. IV interpolated at K = F(T) = S0*exp((r-q)T).

#     Outputs (dict)
#     --------------
#       - "delta_axis"           : (nD,)
#       - "T_axis"               : (nT,)
#       - "iv_delta_call"        : (nT, nD)
#       - "iv_delta_put_abs"     : (nT, nD)   (absolute put delta axis)
#       - "atm_forward_K"        : (nT,)      forward strike F(T)
#       - "atm_vol"              : (nT,)      σ_ATM(T) (interpolated at K=F(T))
#       - "atm_delta_call"       : (nT,)      Δ_ATM,fwd(T) = e^{-qT} N(0.5 σ_ATM sqrt(T))
#       - "delta_skew_surface"   : (nT, nD)   (call-put)/atm_vol
#     """
#     iv = np.asarray(iv_surface, float)
#     K = np.asarray(K_grid, float).ravel()
#     T = np.asarray(T_grid, float).ravel()

#     if iv.shape != (T.size, K.size):
#         raise ValueError("iv_surface must have shape (len(T_grid), len(K_grid)).")
#     if np.any(K <= 0) or np.any(T <= 0):
#         raise ValueError("K_grid and T_grid must be strictly positive.")
#     if np.any(np.diff(K) <= 0):
#         raise ValueError("K_grid must be strictly increasing (needed for interpolation).")

#     S0 = float(S0)
#     r = float(r)
#     q = float(q)

#     if delta_grid is None:
#         delta_grid = np.linspace(0.05, 0.95, 61)
#     delta_axis = np.asarray(delta_grid, float).ravel()

#     # Normal CDF
#     try:
#         from scipy.stats import norm
#         cdf = norm.cdf
#     except Exception:
#         from math import erf, sqrt
#         def cdf(x):
#             x = np.asarray(x, float)
#             return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))

#     nT = T.size
#     nD = delta_axis.size

#     iv_delta_call = np.full((nT, nD), np.nan, float)
#     iv_delta_put_abs = np.full((nT, nD), np.nan, float)

#     # ATM-forward quantities per maturity
#     atm_forward_K = S0 * np.exp((r - q) * T)
#     atm_vol = np.full(nT, np.nan, float)
#     atm_delta_call = np.full(nT, np.nan, float)

#     # ---- Step 1: compute ATM-forward vol per T by interpolating strike-space IV at K=F(T)
#     for i, Ti in enumerate(T):
#         row = iv[i, :]
#         m = np.isfinite(row) & (row > 0)
#         if np.sum(m) < 2:
#             continue
#         Fi = float(atm_forward_K[i])
#         sig_atm = np.interp(Fi, K[m], row[m], left=np.nan, right=np.nan)
#         if np.isfinite(sig_atm) and sig_atm > 0:
#             atm_vol[i] = float(sig_atm)
#             # exact ATM-forward call delta under your BS delta convention:
#             # Δ_ATM,fwd = e^{-qT} N(0.5 σ_ATM sqrt(T))
#             atm_delta_call[i] = float(np.exp(-q * Ti) * cdf(0.5 * sig_atm * np.sqrt(Ti)))

#     # ---- Step 2: build delta-space surfaces by row-wise delta mapping + interpolation
#     for i, Ti in enumerate(T):
#         sig = iv[i, :]

#         m = np.isfinite(sig) & (sig > 0) & (K > 0)
#         if np.sum(m) < min_points:
#             continue

#         Km = K[m]
#         sigm = sig[m]
#         vol_sqrtT = sigm * np.sqrt(Ti)
#         ok = np.isfinite(vol_sqrtT) & (vol_sqrtT > 0)
#         if np.sum(ok) < min_points:
#             continue

#         Km = Km[ok]
#         sigm = sigm[ok]
#         vol_sqrtT = vol_sqrtT[ok]

#         d1 = (np.log(S0 / Km) + (r - q + 0.5 * sigm**2) * Ti) / vol_sqrtT
#         call_delta = np.exp(-q * Ti) * cdf(d1)  # (0, e^{-qT})

#         put_delta_abs = np.abs(call_delta - np.exp(-q * Ti))

#         # ---- call delta surface
#         keep_call = np.isfinite(call_delta) & (call_delta > delta_eps) & (call_delta < 1.0 - delta_eps)
#         if np.sum(keep_call) >= min_points:
#             dc = np.asarray(call_delta[keep_call], float)
#             vc = np.asarray(sigm[keep_call], float)
#             order = np.argsort(dc)
#             dc, vc = dc[order], vc[order]
#             dc_u, idx = np.unique(dc, return_index=True)
#             vc_u = vc[idx]
#             if dc_u.size >= min_points:
#                 iv_delta_call[i, :] = np.interp(delta_axis, dc_u, vc_u, left=np.nan, right=np.nan)

#         # ---- abs put delta surface
#         keep_put = np.isfinite(put_delta_abs) & (put_delta_abs > delta_eps) & (put_delta_abs < 1.0 - delta_eps)
#         if np.sum(keep_put) >= min_points:
#             dp = np.asarray(put_delta_abs[keep_put], float)
#             vp = np.asarray(sigm[keep_put], float)
#             order = np.argsort(dp)
#             dp, vp = dp[order], vp[order]
#             dp_u, idx = np.unique(dp, return_index=True)
#             vp_u = vp[idx]
#             if dp_u.size >= min_points:
#                 iv_delta_put_abs[i, :] = np.interp(delta_axis, dp_u, vp_u, left=np.nan, right=np.nan)

#     # ---- Step 3: delta skew surface (call-put)/ATM_vol, per maturity
#     # Note: ATM_vol is a scalar per T, broadcast across delta_axis
#     denom = atm_vol.reshape(-1, 1)  # (nT,1)
    
#     delta_skew_surface = np.full_like(iv_delta_call, np.nan, dtype=float)
#     valid_rows = np.isfinite(atm_vol) & (atm_vol > 0)
    
#     if np.any(valid_rows):
#         num = iv_delta_call[valid_rows, :] - iv_delta_put_abs[valid_rows, :]
#         skew = num / denom[valid_rows, :]
#         skew[~np.isfinite(num)] = np.nan
#         delta_skew_surface[valid_rows, :] = skew
#     return {
#         "delta_axis": delta_axis,
#         "T_axis": T,
#         "iv_delta_call": iv_delta_call,
#         "iv_delta_put_abs": iv_delta_put_abs,
#         "atm_forward_K": atm_forward_K,
#         "atm_vol": atm_vol,
#         "atm_delta_call": atm_delta_call,
#         "delta_skew_surface": delta_skew_surface,
#     }

def atm_summary_from_iv_surface(
    iv_surf: np.ndarray,
    *,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    S0: float,
) -> dict:
    """
    A simple ATM summary:
      - atm_K = closest strike to S0
      - atm_n = count of finite ATM vols across maturities
      - atm_vol = median of ATM vols across maturities
      - atm_T = maturity at which ATM vol is closest to the median (optional diagnostic)
    """
    iv_surf = np.asarray(iv_surf, float)
    K_grid = np.asarray(K_grid, float).ravel()
    T_grid = np.asarray(T_grid, float).ravel()

    if iv_surf.shape != (T_grid.size, K_grid.size):
        raise ValueError("iv_surf must have shape (len(T_grid), len(K_grid)).")

    j_atm = int(np.argmin(np.abs(K_grid - float(S0))))
    atm_iv = iv_surf[:, j_atm]
    m = np.isfinite(atm_iv) & (atm_iv > 0)

    if not np.any(m):
        return {"atm_K": float(K_grid[j_atm]), "atm_n": 0, "atm_vol": np.nan, "atm_T": np.nan}

    med = float(np.median(atm_iv[m]))
    # pick T whose atm_iv is closest to med
    idx = np.where(m)[0]
    k = idx[int(np.argmin(np.abs(atm_iv[m] - med)))]
    return {"atm_K": float(K_grid[j_atm]), "atm_n": int(np.sum(m)), "atm_vol": med, "atm_T": float(T_grid[k])}

import numpy as np
from typing import Dict, Optional, Tuple


def iv_surface_to_delta_surfaces(
    iv_surface: np.ndarray,
    *,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    S0: float,
    r: float,
    q: float = 0.0,
    delta_grid: Optional[np.ndarray] = None,
    delta_eps: float = 1e-4,
    min_points: int = 6,
) -> Dict[str, np.ndarray]:
    """
    Convert an IV surface σ(K,T) into delta-space surfaces:
      1) delta IV surfaces: σ_call(T,Δ), σ_put_abs(T,Δ)
      2) delta skew surface: (σ_call(T,Δ) - σ_put_abs(T,Δ)) / σ_ATM(T)

    Parameters
    ----------
    iv_surface : np.ndarray
        Implied volatility surface, shape (len(T_grid), len(K_grid))
    K_grid : np.ndarray
        Strike grid (strictly increasing)
    T_grid : np.ndarray
        Time-to-expiry grid
    S0 : float
        Current spot price
    r : float
        Risk-free rate
    q : float, optional
        Dividend yield (default 0.0)
    delta_grid : np.ndarray, optional
        Delta grid for output (default: np.linspace(0.05, 0.95, 61))
    delta_eps : float, optional
        Epsilon for delta bounds (default 1e-4)
    min_points : int, optional
        Minimum points needed for interpolation (default 6)

    Returns
    -------
    dict with keys:
        "delta_axis" : (nD,)
        "T_axis" : (nT,)
        "iv_delta_call" : (nT, nD)
        "iv_delta_put_abs" : (nT, nD)
        "atm_forward_K" : (nT,)
        "atm_vol" : (nT,)
        "atm_delta_call" : (nT,)
        "atm_delta_put_abs" : (nT,)
        "delta_skew_surface" : (nT, nD)
    """
    iv = np.asarray(iv_surface, float)
    K = np.asarray(K_grid, float).ravel()
    T = np.asarray(T_grid, float).ravel()

    if iv.shape != (T.size, K.size):
        raise ValueError("iv_surface must have shape (len(T_grid), len(K_grid)).")
    if np.any(K <= 0) or np.any(T <= 0):
        raise ValueError("K_grid and T_grid must be strictly positive.")
    if np.any(np.diff(K) <= 0):
        raise ValueError("K_grid must be strictly increasing (needed for interpolation).")

    S0 = float(S0)
    r = float(r)
    q = float(q)

    if delta_grid is None:
        delta_grid = np.linspace(0.05, 0.95, 61)
    delta_axis = np.asarray(delta_grid, float).ravel()
    nT, nD = T.size, delta_axis.size

    # Normal CDF
    try:
        from scipy.stats import norm
        cdf = norm.cdf
    except Exception:
        from math import erf, sqrt
        def cdf(x):
            x = np.asarray(x, float)
            return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))

    def _interp_1d_nan(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
        """Linear interp using finite points only; returns NaN outside support."""
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 2:
            return np.full_like(xq, np.nan, dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        # unique x for np.interp
        xu, idx = np.unique(x, return_index=True)
        yu = y[idx]
        out = np.interp(xq, xu, yu, left=np.nan, right=np.nan)
        # Force NaN outside bounds
        out = np.asarray(out, float)
        out[(xq < xu[0]) | (xq > xu[-1])] = np.nan
        return out

    # Initialize outputs
    iv_delta_call = np.full((nT, nD), np.nan, float)
    iv_delta_put_abs = np.full((nT, nD), np.nan, float)

    atm_forward_K = S0 * np.exp((r - q) * T)
    atm_vol = np.full(nT, np.nan, float)
    atm_delta_call = np.full(nT, np.nan, float)
    atm_delta_put_abs = np.full(nT, np.nan, float)

    # (1) ATM-forward vol + ATM-forward deltas
    for i, Ti in enumerate(T):
        row = iv[i, :]
        m = np.isfinite(row) & (row > 0)
        if np.sum(m) < 2:
            continue
        Fi = float(atm_forward_K[i])
        sig_atm = np.interp(Fi, K[m], row[m], left=np.nan, right=np.nan)
        if np.isfinite(sig_atm) and sig_atm > 0:
            atm_vol[i] = float(sig_atm)
            # ATM-forward call delta (BS with K=F): d1 = 0.5 σ√T
            atm_delta_call[i] = float(np.exp(-q * Ti) * cdf(0.5 * sig_atm * np.sqrt(Ti)))
            # abs put delta at ATM-forward: |Δ_put| = e^{-qT} - Δ_call
            atm_delta_put_abs[i] = float(np.exp(-q * Ti) - atm_delta_call[i])

    # (2) Delta-IV surfaces (call / abs put)
    for i, Ti in enumerate(T):
        sig = iv[i, :]

        m = np.isfinite(sig) & (sig > 0) & (K > 0)
        if np.sum(m) < min_points:
            continue

        Km = K[m]
        sigm = sig[m]
        vs = sigm * np.sqrt(Ti)
        ok = np.isfinite(vs) & (vs > 0)
        if np.sum(ok) < min_points:
            continue

        Km = Km[ok]
        sigm = sigm[ok]
        vs = vs[ok]

        d1 = (np.log(S0 / Km) + (r - q + 0.5 * sigm**2) * Ti) / vs
        call_delta = np.exp(-q * Ti) * cdf(d1)
        put_delta_abs = np.abs(call_delta - np.exp(-q * Ti))

        # Call side
        keep_call = np.isfinite(call_delta) & (call_delta > delta_eps) & (call_delta < 1.0 - delta_eps)
        if np.sum(keep_call) >= min_points:
            dc = np.asarray(call_delta[keep_call], float)
            vc = np.asarray(sigm[keep_call], float)
            iv_delta_call[i, :] = _interp_1d_nan(dc, vc, delta_axis)

        # Put side (abs)
        keep_put = np.isfinite(put_delta_abs) & (put_delta_abs > delta_eps) & (put_delta_abs < 1.0 - delta_eps)
        if np.sum(keep_put) >= min_points:
            dp = np.asarray(put_delta_abs[keep_put], float)
            vp = np.asarray(sigm[keep_put], float)
            iv_delta_put_abs[i, :] = _interp_1d_nan(dp, vp, delta_axis)

    # (3) Vol delta skew surface: (call-put)/ATM_vol
    delta_skew_surface = np.full_like(iv_delta_call, np.nan, dtype=float)
    valid_rows_vol = np.isfinite(atm_vol) & (atm_vol > 0)
    if np.any(valid_rows_vol):
        denom = atm_vol.reshape(-1, 1)
        num = iv_delta_call[valid_rows_vol, :] - iv_delta_put_abs[valid_rows_vol, :]
        skew = num / denom[valid_rows_vol, :]
        skew[~np.isfinite(num)] = np.nan
        delta_skew_surface[valid_rows_vol, :] = skew

    return {
        "delta_axis": delta_axis,
        "T_axis": T,
        "iv_delta_call": iv_delta_call,
        "iv_delta_put_abs": iv_delta_put_abs,
        "atm_forward_K": atm_forward_K,
        "atm_vol": atm_vol,
        "atm_delta_call": atm_delta_call,
        "atm_delta_put_abs": atm_delta_put_abs,
        "delta_skew_surface": delta_skew_surface,
    }

