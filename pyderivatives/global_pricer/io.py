from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd

from .data import CallSurfaceDay

from typing import Optional, Tuple
import numpy as np
import pandas as pd

def extract_call_surface_vectors(
    df: pd.DataFrame,
    *,
    strike_col: str = "strike",
    maturity_col: str = "rounded_maturity",
    price_col: str = "mid_price",
    rate_col: str = "risk_free_rate",
    spot_col: str = "stock_price",
    q_col: Optional[str] = None,
    option_type_col: str | None = "option_right",
    call_flag: str = "c",
    dropna: bool = True,
    ticker_col: str = "ticker",                 # <- add this
    ticker_default: str = "Unknown",            # <- and this
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, str]:
    """
    Returns (K_obs, T_obs, C_obs, r_med, S0_med, q_med, ticker).
    q_med defaults to 0.0 if q_col is None.
    ticker defaults to ticker_default if ticker_col missing/empty.
    """
    d = df.copy()

    if option_type_col is not None:
        d = d[d[option_type_col].astype(str).str.lower().str[0] == call_flag]

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

    # --- ticker extraction ---
    if ticker_col in d.columns:
        tick = d[ticker_col].astype(str).replace("nan", np.nan).dropna()
        ticker = str(tick.iloc[0]) if len(tick) else ticker_default
    else:
        ticker = ticker_default

    return K_obs, T_obs, C_obs, r_med, S0_med, q_med, ticker




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
    ticker_col: str = "ticker",
    ticker_default: str = "Unknown",
) -> CallSurfaceDay:
    K, T, C, r, S0, q, ticker = extract_call_surface_vectors(
        df,
        strike_col=strike_col,
        maturity_col=maturity_col,
        price_col=price_col,
        rate_col=rate_col,
        spot_col=spot_col,
        q_col=q_col,
        option_type_col=option_type_col,
        call_flag=call_flag,
        ticker_col=ticker_col,
        ticker_default=ticker_default,
    )
    return CallSurfaceDay(S0=S0, r=r, q=q, K_obs=K, T_obs=T, C_obs=C, ticker=ticker)





