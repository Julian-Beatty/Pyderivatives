from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
import numpy as np

try:
    import pandas as pd
except Exception:  # keep import-light if pandas isn't always installed
    pd = Any  # type: ignore


class SafetyClipResult(TypedDict):
    enabled: bool
    any_used: bool
    per_row: List[Dict[str, Any]]


class GlobalSurfaceResult(TypedDict):
    # identity
    date: Optional[str]
    model: str
    success: bool

    # inputs / scalars
    S0: float
    r: float
    q: float
    atm_K: Optional[float]
    atm_vol: Optional[float]   # "ATM vol (median across maturity)" per your note
    atm_n: Optional[int]
    atm_T: Optional[float]

    # observed (sparse) quotes
    K_obs: np.ndarray
    T_obs: np.ndarray
    C_obs: np.ndarray
    C_hat_obs: Optional[np.ndarray]

    # fitted params (keep both representations)
    params: Dict[str, float]        # always JSON-able
    params_obj: Optional[Any]       # e.g., HKDEParams instance

    # grids
    K_grid: np.ndarray
    T_grid: np.ndarray

    # call surfaces
    C_fit: np.ndarray               # (len(T_grid), len(K_grid))
    C_obs_surface: Optional[np.ndarray]  # optional rectangular "original call surface" if you build it
    C_fit_obs_surface: Optional[np.ndarray]  # optional fitted values on observed grid (if you build it)

    # implied vol surface on (T_grid, K_grid)
    iv_surface: Optional[np.ndarray]

    # RND + CDF
    rnd_surface: Optional[np.ndarray]
    rnd_surface_raw: Optional[np.ndarray]
    cdf_surface: Optional[np.ndarray]

    # moments
    rnd_moments_table: Optional["pd.DataFrame"]

    # safety clip
    safety_clip: Optional[SafetyClipResult]
    safety_clip_status: Optional[str]  # "Used"/"Unused" convenience

    # diagnostics
    errors: Optional[Dict[str, float]]
    errors_by_T: Optional[Dict[float, Dict[str, float]]]

    # misc
    meta: Dict[str, Any]
