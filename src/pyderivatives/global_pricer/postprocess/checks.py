from __future__ import annotations

from typing import Any, Dict
import numpy as np


def reprice_observed_points(model, K_obs: np.ndarray, T_obs: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Reprice each observed quote using model.call_prices grouped by maturity.
    Requires: model.call_prices(K_vec, T, params_dict) -> np.ndarray
    """
    K_obs = np.asarray(K_obs, float).ravel()
    T_obs = np.asarray(T_obs, float).ravel()

    out = np.empty_like(K_obs, dtype=float)

    T_unique = np.unique(T_obs)
    for t in T_unique:
        idx = np.where(T_obs == t)[0]
        out[idx] = model.call_prices(K_obs[idx], float(t), params)

    return out


def error_summary(C_obs: np.ndarray, C_hat: np.ndarray) -> Dict[str, float]:
    C_obs = np.asarray(C_obs, float).ravel()
    C_hat = np.asarray(C_hat, float).ravel()
    err = C_hat - C_obs

    m = np.isfinite(err)
    if not np.any(m):
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "mape": np.nan, "max_abs": np.nan}

    e = err[m]
    y = C_obs[m]
    denom = np.maximum(np.abs(y), 1e-12)

    return {
        "n": int(e.size),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "mae": float(np.mean(np.abs(e))),
        "mape": float(np.mean(np.abs(e) / denom)),
        "max_abs": float(np.max(np.abs(e))),
    }


def error_by_maturity(
    K_obs: np.ndarray,
    T_obs: np.ndarray,
    C_obs: np.ndarray,
    C_hat: np.ndarray,
) -> Dict[float, Dict[str, float]]:
    """
    Returns a dict keyed by maturity T (float) -> error_summary for that maturity bucket.
    """
    K_obs = np.asarray(K_obs, float).ravel()
    T_obs = np.asarray(T_obs, float).ravel()
    C_obs = np.asarray(C_obs, float).ravel()
    C_hat = np.asarray(C_hat, float).ravel()

    if not (K_obs.size == T_obs.size == C_obs.size == C_hat.size):
        raise ValueError("Inputs must have same length.")

    out: Dict[float, Dict[str, float]] = {}
    for t in np.unique(T_obs):
        idx = np.where(T_obs == t)[0]
        out[float(t)] = error_summary(C_obs[idx], C_hat[idx])
    return out


def reprice_observed_points(
    model: Any,
    K_obs: np.ndarray,
    T_obs: np.ndarray,
    params: Dict[str, float],
) -> np.ndarray:
    """
    Reprice at the observed (K_obs, T_obs) points.

    Assumes your model implements:
      - call_prices(K: np.ndarray, T: float, params: dict) -> np.ndarray
    or equivalently something compatible.

    Returns
    -------
    C_hat_obs : np.ndarray shape (N,)
    """
    K_obs = np.asarray(K_obs, float).ravel()
    T_obs = np.asarray(T_obs, float).ravel()
    if K_obs.size != T_obs.size:
        raise ValueError("K_obs and T_obs must have same length.")

    C_hat = np.empty_like(K_obs, dtype=float)

    # Group by unique maturity to avoid calling the pricer N times
    T_unique = np.unique(T_obs)
    for t in T_unique:
        idx = np.where(T_obs == t)[0]
        if idx.size == 0:
            continue
        C_hat[idx] = model.call_prices(K_obs[idx], float(t), params)

    return C_hat


def error_summary(
    C_obs: np.ndarray,
    C_hat: np.ndarray,
    *,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Overall error diagnostics + top errors.

    Returns dict with:
      rmse, mae, mape (on positive C_obs), max_abs, bias,
      n, n_finite, top (table dict)
    """
    y = np.asarray(C_obs, float).ravel()
    yhat = np.asarray(C_hat, float).ravel()
    if y.size != yhat.size:
        raise ValueError("C_obs and C_hat must have same length.")

    m = np.isfinite(y) & np.isfinite(yhat)
    n = int(y.size)
    nf = int(np.sum(m))
    if nf == 0:
        return {
            "n": n,
            "n_finite": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "max_abs": np.nan,
            "bias": np.nan,
            "top": {"idx": [], "C_obs": [], "C_hat": [], "err": [], "abs_err": []},
        }

    err = (yhat[m] - y[m])
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(abs_err))
    bias = float(np.mean(err))
    max_abs = float(np.max(abs_err))

    # MAPE only where obs price is meaningfully > 0
    mp = y[m] > 1e-12
    mape = float(np.mean(abs_err[mp] / y[m][mp])) if np.any(mp) else np.nan

    # top errors (global indices)
    global_idx = np.where(m)[0]
    top_local = np.argsort(abs_err)[::-1][: int(top_n)]
    top_idx = global_idx[top_local]

    top = {
        "idx": top_idx.tolist(),
        "C_obs": y[top_idx].tolist(),
        "C_hat": yhat[top_idx].tolist(),
        "err": (yhat[top_idx] - y[top_idx]).tolist(),
        "abs_err": np.abs(yhat[top_idx] - y[top_idx]).tolist(),
    }

    return {
        "n": n,
        "n_finite": nf,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "max_abs": max_abs,
        "bias": bias,
        "top": top,
    }


def error_by_maturity(
    K_obs: np.ndarray,
    T_obs: np.ndarray,
    C_obs: np.ndarray,
    C_hat: np.ndarray,
) -> Dict[str, Any]:
    """
    Returns per-maturity MAE/RMSE and counts.

    Output:
      {
        "T": [...],
        "n": [...],
        "mae": [...],
        "rmse": [...],
        "bias": [...],
      }
    """
    K_obs = np.asarray(K_obs, float).ravel()
    T_obs = np.asarray(T_obs, float).ravel()
    y = np.asarray(C_obs, float).ravel()
    yhat = np.asarray(C_hat, float).ravel()

    if not (K_obs.size == T_obs.size == y.size == yhat.size):
        raise ValueError("K_obs, T_obs, C_obs, C_hat must all have same length.")

    m = np.isfinite(T_obs) & np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return {"T": [], "n": [], "mae": [], "rmse": [], "bias": []}

    Tm = T_obs[m]
    ym = y[m]
    yhm = yhat[m]

    T_unique = np.unique(Tm)
    T_unique.sort()

    out_T, out_n, out_mae, out_rmse, out_bias = [], [], [], [], []
    for t in T_unique:
        idx = np.where(Tm == t)[0]
        if idx.size == 0:
            continue
        e = yhm[idx] - ym[idx]
        out_T.append(float(t))
        out_n.append(int(idx.size))
        out_mae.append(float(np.mean(np.abs(e))))
        out_rmse.append(float(np.sqrt(np.mean(e**2))))
        out_bias.append(float(np.mean(e)))

    return {"T": out_T, "n": out_n, "mae": out_mae, "rmse": out_rmse, "bias": out_bias}


def top_error_table(
    K_obs: np.ndarray,
    T_obs: np.ndarray,
    C_obs: np.ndarray,
    C_hat: np.ndarray,
    *,
    top_n: int = 20,
) -> Dict[str, list]:
    """
    Returns a dict-of-lists table with K, T, C_obs, C_hat, err, abs_err for top abs errors.
    """
    K = np.asarray(K_obs, float).ravel()
    T = np.asarray(T_obs, float).ravel()
    y = np.asarray(C_obs, float).ravel()
    yhat = np.asarray(C_hat, float).ravel()

    if not (K.size == T.size == y.size == yhat.size):
        raise ValueError("All inputs must have same length.")

    m = np.isfinite(K) & np.isfinite(T) & np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return {"K": [], "T": [], "C_obs": [], "C_hat": [], "err": [], "abs_err": []}

    idx = np.where(m)[0]
    e = yhat[m] - y[m]
    ae = np.abs(e)

    j = np.argsort(ae)[::-1][: int(top_n)]
    jj = idx[j]

    return {
        "K": K[jj].tolist(),
        "T": T[jj].tolist(),
        "C_obs": y[jj].tolist(),
        "C_hat": yhat[jj].tolist(),
        "err": (yhat[jj] - y[jj]).tolist(),
        "abs_err": np.abs(yhat[jj] - y[jj]).tolist(),
    }
