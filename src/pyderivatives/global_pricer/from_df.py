from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .data import CallSurfaceDay


def day_from_df(
    df: pd.DataFrame,
    *,
    strike_col: str = "strike",
    maturity_col: str = "rounded_maturity",
    price_col: str = "C_rep",
    rate_col: str = "risk_free_rate",
    spot_col: str = "stock_price",
    option_type_col: Optional[str] = "option_right",
    call_flag: str = "c",
    q: float = 0.0,
    dropna: bool = True,
) -> CallSurfaceDay:
    if option_type_col is not None:
        s = df[option_type_col].astype(str).str.lower()
        df = df.loc[s == call_flag]

    req = [strike_col, maturity_col, price_col, rate_col, spot_col]
    if dropna:
        df = df.dropna(subset=req)

    if df.empty:
        raise ValueError("No valid call quotes after filtering/dropna.")

    K = df[strike_col].to_numpy(float)
    T = df[maturity_col].to_numpy(float)
    C = df[price_col].to_numpy(float)

    r_med = float(np.median(df[rate_col].to_numpy(float)))
    S0_med = float(np.median(df[spot_col].to_numpy(float)))

    return CallSurfaceDay(S0=S0_med, r=r_med, q=float(q), K_obs=K, T_obs=T, C_obs=C)
