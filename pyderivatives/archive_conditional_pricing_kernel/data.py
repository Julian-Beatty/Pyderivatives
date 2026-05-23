from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
from typing import Tuple, Optional

def _cdf_from_density_on_grid(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    CDF(x_i) = ∫ f(u) du from x_0 to x_i using trapezoids.
    Returns normalized CDF in [0,1] if total mass > 0 else all nan.
    """
    x = np.asarray(x, float).ravel()
    f = np.asarray(f, float).ravel()
    if x.size < 2:
        return np.full_like(x, np.nan)

    # assume x is increasing
    dx = np.diff(x)
    mid = 0.5 * (f[:-1] + f[1:])
    cdf = np.concatenate([[0.0], np.cumsum(mid * dx)])

    mass = cdf[-1]
    if not np.isfinite(mass) or mass <= 0:
        return np.full_like(x, np.nan)

    return cdf / mass


def _rnd_logreturn_quantile_bounds_from_qK(
    K_grid: np.ndarray,
    qK_row: np.ndarray,
    S0: float,
    alpha_lo: float,
    alpha_hi: float,
) -> tuple[float, float]:
    """
    Compute log-return bounds [r_lo, r_hi] implied by the RND q(K) at quantiles
    (alpha_lo, alpha_hi). Uses strike-space CDF and maps K -> r=log(K/S0).
    """
    K = np.asarray(K_grid, float).ravel()
    qK = np.asarray(qK_row, float).ravel()

    m = np.isfinite(K) & np.isfinite(qK) & (K > 0) & (qK >= 0)
    K = K[m]
    qK = qK[m]
    if K.size < 10:
        return -np.inf, np.inf

    idx = np.argsort(K)
    K = K[idx]
    qK = qK[idx]

    cdfK = _cdf_from_density_on_grid(K, qK)
    if not np.all(np.isfinite(cdfK)):
        return -np.inf, np.inf

    K_lo = float(np.interp(alpha_lo, cdfK, K))
    K_hi = float(np.interp(alpha_hi, cdfK, K))

    r_lo = float(np.log(K_lo / float(S0)))
    r_hi = float(np.log(K_hi / float(S0)))
    return r_lo, r_hi

def _ensure_dtindex(stock_df: pd.DataFrame, spot_col: str) -> pd.DataFrame:
    df = stock_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" not in df.columns:
            raise ValueError("stock_df must have DatetimeIndex or 'date' column.")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    if spot_col not in df.columns:
        raise KeyError(f"stock_df missing '{spot_col}', has {list(df.columns)}")
    return df

def build_obs_by_maturity(
    result_dict: Dict[str, dict],
    stock_df: pd.DataFrame,
    *,
    spot_col: str,
    align: str = "pad",
    max_gap_days: int = 3,
    min_horizon_days: int = 1,
    # ---- diagnostics ----
    diag_huge_returns: bool = False,
    huge_return_log_threshold: float = 1.0,
    drop_log: Optional[dict] = None,
    # ---- NEW: drop realized returns outside these RND quantiles ----
    rnd_tail_alpha: Optional[Tuple[float, float]] = None,  # e.g. (0.005, 0.995)
) -> tuple[np.ndarray, List[List[dict]]]:
    """
    Build observation lists by maturity for pricing-kernel theta fitting.

    Notes:
      - S0, K_grid, qK_row come from result_dict (option-surface basis).
      - Realized returns are computed from stock_df on split-adjusted basis if 'ajexdi' exists.
      - End-date alignment uses bfill but enforces max_gap_days tolerance.
      - R_real is a LOG RETURN.
      - Optional: drop obs if R_real is outside RND quantiles (rnd_tail_alpha).
    """
    stock = _ensure_dtindex(stock_df, spot_col).copy()

    # split-adjusted price for realized returns
    if "ajexdi" not in stock.columns:
        warnings.warn(
            "build_obs_by_maturity: No 'ajexdi' column found; realized returns are not split-adjusted.",
            RuntimeWarning,
            stacklevel=2,
        )
        stock["_price_adj"] = pd.to_numeric(stock[spot_col], errors="coerce").astype(float)
    else:
        aj = pd.to_numeric(stock["ajexdi"], errors="coerce").astype(float)
        px = pd.to_numeric(stock[spot_col], errors="coerce").astype(float)
        stock["_price_adj"] = px / aj

    keys = sorted(list(result_dict.keys()))
    if not keys:
        return np.asarray([], float), []

    first = result_dict[keys[0]]
    T_grid = np.asarray(first["T_grid"], float).ravel()
    nT = int(T_grid.size)

    def _log_drop(reason: str, payload: dict):
        if drop_log is None:
            return
        drop_log.setdefault("counts", {}).setdefault(reason, 0)
        drop_log["counts"][reason] += 1
        # keep a small sample
        drop_log.setdefault("examples", {}).setdefault(reason, [])
        if len(drop_log["examples"][reason]) < 10:
            drop_log["examples"][reason].append(payload)

    def align_date(d: pd.Timestamp) -> Optional[pd.Timestamp]:
        idx = stock.index.get_indexer([d], method=align)
        pos = int(idx[0])
        if pos < 0:
            return None
        d0 = stock.index[pos]
        if abs((d0 - d).days) > int(max_gap_days):
            return None
        return d0

    # validate rnd_tail_alpha
    if rnd_tail_alpha is not None:
        a_lo, a_hi = map(float, rnd_tail_alpha)
        if not (0.0 <= a_lo < a_hi <= 1.0):
            raise ValueError("rnd_tail_alpha must satisfy 0 <= lo < hi <= 1")

    obs_by_T: List[List[dict]] = [[] for _ in range(nT)]

    for k in keys:
        dd = result_dict[k]
        if not dd.get("success", True):
            _log_drop("surface_not_success", {"anchor_key": k})
            continue

        atm_vol = dd.get("atm_vol", np.nan)
        if (atm_vol is None) or (not np.isfinite(atm_vol)) or (float(atm_vol) <= 0):
            _log_drop("bad_atm_vol", {"anchor_key": k, "atm_vol": atm_vol})
            continue

        anchor_date = pd.Timestamp(k).tz_localize(None)
        d0 = align_date(anchor_date)
        if d0 is None:
            _log_drop("anchor_align_failed", {"anchor_key": k, "anchor_date": str(anchor_date)})
            continue

        # option-basis spot for mapping strikes -> r(K)
        S0 = float(dd.get("S0", np.nan))
        if not np.isfinite(S0) or S0 <= 0:
            # fallback only if dict missing/invalid
            S0 = float(stock.loc[d0, spot_col])
        if not np.isfinite(S0) or S0 <= 0:
            _log_drop("bad_S0", {"anchor_key": k, "S0": S0})
            continue

        K_grid = np.asarray(dd["K_grid"], float).ravel()
        qK = np.asarray(dd["rnd_surface"], float)  # (nT, nK)
        if qK.ndim != 2 or qK.shape[0] != nT:
            _log_drop("bad_qK_shape", {"anchor_key": k, "qK_shape": getattr(qK, "shape", None), "nT": nT})
            continue

        r_rf = float(dd.get("r", 0.0))
        sigma = float(atm_vol)

        # realized-return base price at anchor (split-adjusted)
        S0_ret = float(stock.loc[d0, "_price_adj"])
        if not np.isfinite(S0_ret) or S0_ret <= 0:
            _log_drop("bad_S0_ret", {"anchor_key": k, "d0": str(d0), "S0_ret": S0_ret})
            continue

        for j in range(nT):
            Tj = float(T_grid[j])
            if not np.isfinite(Tj) or Tj <= 0:
                continue

            horizon_days = int(np.round(Tj * 365.0))
            if horizon_days < int(min_horizon_days):
                continue

            end_date_target = d0 + pd.Timedelta(days=horizon_days)
            idx1 = stock.index.get_indexer([end_date_target], method="bfill")
            pos1 = int(idx1[0])
            if pos1 < 0:
                _log_drop("end_not_found", {"anchor_key": k, "j": j, "end_target": str(end_date_target)})
                continue

            d1 = stock.index[pos1]
            if abs((d1 - end_date_target).days) > int(max_gap_days):
                _log_drop(
                    "end_gap_too_large",
                    {"anchor_key": k, "j": j, "end_target": str(end_date_target), "end_matched": str(d1)},
                )
                continue

            S1_ret = float(stock.loc[d1, "_price_adj"])
            if not np.isfinite(S1_ret) or S1_ret <= 0:
                _log_drop("bad_S1_ret", {"anchor_key": k, "j": j, "d1": str(d1), "S1_ret": S1_ret})
                continue

            R_real = float(np.log(S1_ret / S0_ret))  # LOG return

            qK_row = np.asarray(qK[j, :], float)

            # ---- NEW: drop if outside RND quantile window ----
            if rnd_tail_alpha is not None:
                a_lo, a_hi = map(float, rnd_tail_alpha)
                r_lo, r_hi = _rnd_logreturn_quantile_bounds_from_qK(
                    K_grid=K_grid,
                    qK_row=qK_row,
                    S0=S0,
                    alpha_lo=a_lo,
                    alpha_hi=a_hi,
                )
                if (R_real < r_lo) or (R_real > r_hi):
                    _log_drop(
                        "outside_rnd_quantiles",
                        {
                            "anchor_key": k,
                            "j": j,
                            "Tj": Tj,
                            "d0": str(d0),
                            "d1": str(d1),
                            "R_real": R_real,
                            "gross": float(np.exp(R_real)),
                            "r_lo": r_lo,
                            "r_hi": r_hi,
                            "gross_lo": float(np.exp(r_lo)),
                            "gross_hi": float(np.exp(r_hi)),
                        },
                    )
                    continue

            if diag_huge_returns and (R_real > float(huge_return_log_threshold)):
                print("\n[HUGE RETURN CONFIRM]")
                print("anchor key:", k)
                print("Tj:", Tj, "horizon_days:", horizon_days)
                print("d0:", d0, "target end:", end_date_target, "matched d1:", d1)
                print("actual (d1 - d0) days:", (d1 - d0).days)
                print("adj prices S0_ret, S1_ret:", S0_ret, S1_ret)
                print("R_real:", R_real, "gross:", float(np.exp(R_real)))

            obs_by_T[j].append(
                dict(
                    anchor_key=str(k),
                    anchor_date=d0,
                    end_date_target=end_date_target,
                    end_date_matched=d1,
                    horizon_days=horizon_days,
                    # option-surface basis inputs:
                    S0=S0,
                    K_grid=K_grid,
                    qK_row=qK_row,
                    r=r_rf,
                    sigma=sigma,
                    # realized return:
                    R_real=R_real,
                    # debug fields:
                    S0_ret=S0_ret,
                    S1_ret=S1_ret,
                    aj0=float(stock.loc[d0, "ajexdi"]) if "ajexdi" in stock.columns else np.nan,
                    aj1=float(stock.loc[d1, "ajexdi"]) if "ajexdi" in stock.columns else np.nan,
                    px0=float(stock.loc[d0, spot_col]),
                    px1=float(stock.loc[d1, spot_col]),
                )
            )

    return T_grid, obs_by_T

