from __future__ import annotations

import os
import json
import pickle
import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Tuple

import numpy as np

# ============================================================
# Numerical utilities
# ============================================================
def _validate_fit_trim_alpha(
    fit_trim_alpha: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """
    Validate trimming specification.

    Parameters
    ----------
    fit_trim_alpha:
        (left_alpha, right_alpha)

    Returns
    -------
    tuple or None
    """
    if fit_trim_alpha is None:
        return None

    if len(fit_trim_alpha) != 2:
        raise ValueError("fit_trim_alpha must have length 2.")

    a_left = float(fit_trim_alpha[0])
    a_right = float(fit_trim_alpha[1])

    if not (0.0 <= a_left < 1.0):
        raise ValueError("Left trim alpha must satisfy 0 <= alpha < 1.")

    if not (0.0 <= a_right < 1.0):
        raise ValueError("Right trim alpha must satisfy 0 <= alpha < 1.")

    if (a_left + a_right) >= 1.0:
        raise ValueError("fit_trim_alpha must satisfy left + right < 1.")

    return (a_left, a_right)
def _as_1d(x, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype).ravel()


def _trapz_normalize_density(x: np.ndarray, f: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    x = _as_1d(x)
    f = _as_1d(f)
    f = np.where(np.isfinite(f) & (f >= 0), f, 0.0)
    mass = float(np.trapezoid(f, x)) if x.size >= 2 else np.nan
    if not np.isfinite(mass) or mass <= eps:
        return np.full_like(f, np.nan, dtype=float)
    return f / mass

def _cdf_from_density(x: np.ndarray, f: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    x = _as_1d(x)
    f = _as_1d(f)
    if x.size != f.size or x.size < 2:
        return np.full_like(x, np.nan, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x grid must be strictly increasing.")
    f = np.where(np.isfinite(f) & (f >= 0), f, 0.0)
    dx = np.diff(x)
    area = 0.5 * (f[:-1] + f[1:]) * dx
    cdf = np.empty_like(x, dtype=float)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(area)
    total = float(cdf[-1])
    if not np.isfinite(total) or total <= eps:
        return np.full_like(x, np.nan, dtype=float)
    return cdf / total


def _safe_interp(x_new: float, x: np.ndarray, y: np.ndarray) -> float:
    x = _as_1d(x)
    y = _as_1d(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    xx = x[m]
    yy = y[m]
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    return float(np.interp(float(x_new), xx, yy, left=np.nan, right=np.nan))

def _find_spot(info: dict, spot_keys: Tuple[str, ...]) -> Optional[float]:
    for key in spot_keys:
        if key in info:
            val = info[key]
            try:
                val = float(val)
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    return None

def _find_sigma(info: dict, sigma_keys: Tuple[str, ...], default: float = 1.0) -> float:
    for key in sigma_keys:
        if key in info:
            val = info[key]
            try:
                val = float(val)
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    return float(default)

def _block_indices_circular(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap indices."""
    if n <= 0:
        return np.array([], dtype=int)
    L = max(1, int(block_length))
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=n_blocks)
    idx = []
    for s in starts:
        idx.extend([(int(s) + k) % n for k in range(L)])
    return np.asarray(idx[:n], dtype=int)




