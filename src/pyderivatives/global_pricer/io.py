from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd

from .data import CallSurfaceDay


def extract_call_surface_vectors(
    df: pd.DataFrame,
    *,
    strike_col: str = "strike",
    maturity_col: str = "rounded_maturity",
    price_col: str = "mid_price",
    rate_col: str = "risk_free_rate",
    spot_col: str = "stock_price",
    q_col: Optional[str] = None,  # optional if you have it
    option_type_col: str | None = "option_right",
    call_flag: str = "c",
    dropna: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Returns (K_obs, T_obs, C_obs, r_med, S0_med, q_med).
    q_med defaults to 0.0 if q_col is None.
    """
    d = df.copy()

    if option_type_col is not None:
        d = d[d[option_type_col].astype(str).str.lower() == call_flag]

    required_cols = [strike_col, maturity_col, price_col, rate_col, spot_col]
    if dropna:
        d = d.dropna(subset=required_cols)

    if d.empty:
        raise ValueError("No valid call options left after filtering.")

    K_obs = d[strike_col].to_numpy(dtype=float)
    T_obs = d[maturity_col].to_numpy(dtype=float)
    C_obs = d[price_col].to_numpy(dtype=float)

    r_med = float(np.median(d[rate_col].to_numpy(dtype=float)))
    S0_med = float(np.median(d[spot_col].to_numpy(dtype=float)))

    if q_col is None:
        q_med = 0.0
    else:
        q_med = float(np.median(d[q_col].to_numpy(dtype=float)))

    return K_obs, T_obs, C_obs, r_med, S0_med, q_med


def make_day_from_df(
    df: pd.DataFrame,
    *,
    price_col: str = "C_rep",
    strike_col: str = "strike",
    maturity_col: str = "rounded_maturity",
    rate_col: str = "risk_free_rate",
    spot_col: str = "stock_price",
    q_col: Optional[str] = None,
    option_type_col: str | None = "option_right",
    call_flag: str = "c",
) -> CallSurfaceDay:
    K, T, C, r, S0, q = extract_call_surface_vectors(
        df,
        strike_col=strike_col,
        maturity_col=maturity_col,
        price_col=price_col,
        rate_col=rate_col,
        spot_col=spot_col,
        q_col=q_col,
        option_type_col=option_type_col,
        call_flag=call_flag,
    )
    return CallSurfaceDay(S0=S0, r=r, q=q, K_obs=K, T_obs=T, C_obs=C)






