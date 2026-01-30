from __future__ import annotations

from typing import Sequence
import pandas as pd
import numpy as np
from typing import List, Optional
from typing import Dict, List, Optional
import pandas as pd

def ensure_timestamp(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c])
    return out

def extract_call_surface_from_df(
    option_df: pd.DataFrame,
    price_col: str = "mid_price",
    maturity_col: str = "rounded_maturity",
    right_col: str = "option_right",
    call_code: str = "c",
):
    """
    Converts pandas dataframe option df into call surface

    Parameters
    ----------
    option_df : pd.DataFrame
        DESCRIPTION.
    price_col : str, optional
        DESCRIPTION. The default is "mid_price".
    maturity_col : str, optional
        DESCRIPTION. The default is "rounded_maturity".
    right_col : str, optional
        DESCRIPTION. The default is "option_right".
    call_code : str, optional
        DESCRIPTION. The default is "c".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    strikes : TYPE
        DESCRIPTION.
    maturities : TYPE
        DESCRIPTION.
    C : TYPE
        DESCRIPTION.
    S0 : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.

    """
    df = option_df.copy()

    if right_col in df.columns:
        df = df[df[right_col] == call_code]

    df = df.dropna(subset=["strike", maturity_col, price_col])

    strikes = np.sort(df["strike"].unique())
    maturities = np.sort(df[maturity_col].unique())

    surface = df.pivot_table(
        index=maturity_col,
        columns="strike",
        values=price_col,
        aggfunc="mean",
    )
    surface = surface.reindex(index=maturities, columns=strikes)
    C = surface.values.astype(float)

    if "underlying_price" in df.columns:
        S0 = float(df["underlying_price"].median())
    elif "stock_price" in df.columns:
        S0 = float(df["stock_price"].median())
    else:
        raise ValueError("Need 'underlying_price' or 'stock_price' column for S0.")

    if "risk_free_rate" in df.columns:
        r = float(df["risk_free_rate"].median())
    elif "risk_f" in df.columns:
        r = float(df["risk_f"].median())
    else:
        r = 0.0

    return strikes, maturities, C, S0, r


def slice_call_surfaces_by_date(
    option_df: pd.DataFrame,
    *,
    date_col: str = "date",
    cols: Optional[List[str]] = None,
    sort_cols: Optional[List[str]] = None,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Slice an option dataframe into per-date call surfaces.

    Parameters
    ----------
    option_df : pd.DataFrame
        Full option dataframe containing multiple dates.
    date_col : str
        Column used to group by date.
    cols : list[str], optional
        Columns to keep in each slice.
    sort_cols : list[str], optional
        Columns to sort each slice by.

    Returns
    -------
    dict
        Dictionary mapping date -> sliced DataFrame.
    """
    if cols is None:
        cols = [
            "date",
            "exdate",
            "rounded_maturity",
            "stock_price",
            "risk_free_rate",
            "strike",
            "mid_price",
        ]

    if sort_cols is None:
        sort_cols = ["rounded_maturity", "strike"]

    return {
        d: g[cols].sort_values(sort_cols).reset_index(drop=True)
        for d, g in option_df.groupby(date_col, sort=False)
    }


def compute_mid(df: pd.DataFrame, bid_col: str = "bid", ask_col: str = "ask") -> pd.Series:
    bid = pd.to_numeric(df[bid_col], errors="coerce")
    ask = pd.to_numeric(df[ask_col], errors="coerce")
    return 0.5 * (bid + ask)


def yearfrac_act365(start: pd.Series, end: pd.Series) -> pd.Series:
    # start/end are Timestamp series
    return (end - start).dt.days / 365.0

def summarize_put_call_parity_diff(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    *,
    keys: Optional[List[str]] = None,
    price_cols: List[str] = ["best_bid", "mid_price", "best_offer"],
    keep_only_matches: bool = True,
    tol: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compare original options vs parity-transformed options and compute price differentials.

    Parameters
    ----------
    original_df : pd.DataFrame
        Original quotes (e.g., true puts or true calls).
    transformed_df : pd.DataFrame
        Parity-transformed quotes (e.g., calls->puts or puts->calls).
        IMPORTANT: option_right in transformed_df should already be flipped accordingly.
    keys : list[str], optional
        Columns used to match rows. If None, uses a robust default.
    price_cols : list[str]
        Price columns to compare.
    keep_only_matches : bool
        If True, drop rows that fail to match in either side.
    tol : float, optional
        If provided, adds a boolean column 'within_tol_mid' for |mid_diff| <= tol.

    Returns
    -------
    pd.DataFrame
        One row per matched contract with columns:
          - original_* prices
          - transformed_* prices
          - diff_* = transformed - original
          - abs_diff_*
    """
    if keys is None:
        # use your common identifiers
        keys = ["date", "exdate", "option_right", "strike"]

    o = original_df.copy()
    t = transformed_df.copy()

    # Ensure datetimes for merge keys if present
    for c in ["date", "exdate"]:
        if c in o.columns:
            o[c] = pd.to_datetime(o[c])
        if c in t.columns:
            t[c] = pd.to_datetime(t[c])

    # numericize price cols
    for c in price_cols:
        if c in o.columns:
            o[c] = pd.to_numeric(o[c], errors="coerce")
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    # Merge
    merged = o.merge(
        t,
        on=keys,
        how="inner" if keep_only_matches else "outer",
        suffixes=("_orig", "_xform"),
        indicator=not keep_only_matches,
    )

    # Compute diffs
    for c in price_cols:
        co = f"{c}_orig"
        cx = f"{c}_xform"
        if co in merged.columns and cx in merged.columns:
            merged[f"diff_{c}"] = merged[cx] - merged[co]
            merged[f"absdiff_{c}"] = merged[f"diff_{c}"].abs()

    # Add summary stats per row (nice quick diagnostic)
    # (uses mid if available else first available col)
    if "diff_mid_price" in merged.columns:
        merged["diff_used"] = merged["diff_mid_price"]
        merged["absdiff_used"] = merged["absdiff_mid_price"]
        used_col = "mid_price"
    else:
        # fallback: first price column that exists
        used_col = None
        for c in price_cols:
            if f"diff_{c}" in merged.columns:
                merged["diff_used"] = merged[f"diff_{c}"]
                merged["absdiff_used"] = merged[f"absdiff_{c}"]
                used_col = c
                break

    if tol is not None and "absdiff_used" in merged.columns:
        merged["within_tol_used"] = merged["absdiff_used"] <= tol

    # Optional: keep some context columns if they exist on either side
    context_cols = []
    for c in ["stock_price", "risk_free_rate", "rounded_maturity", "moneyness", "volume"]:
        if f"{c}_orig" in merged.columns:
            context_cols.append(f"{c}_orig")
        elif c in merged.columns:
            context_cols.append(c)

    # Arrange columns nicely
    base = keys.copy()
    prices = []
    diffs = []
    for c in price_cols:
        if f"{c}_orig" in merged.columns: prices.append(f"{c}_orig")
        if f"{c}_xform" in merged.columns: prices.append(f"{c}_xform")
        if f"diff_{c}" in merged.columns: diffs.append(f"diff_{c}")
        if f"absdiff_{c}" in merged.columns: diffs.append(f"absdiff_{c}")

    extras = []
    if not keep_only_matches and "_merge" in merged.columns:
        extras.append("_merge")
    if "diff_used" in merged.columns:
        extras += ["diff_used", "absdiff_used"]
    if tol is not None and "within_tol_used" in merged.columns:
        extras.append("within_tol_used")

    ordered = [c for c in base + context_cols + prices + diffs + extras if c in merged.columns]
    return merged[ordered].sort_values(base).reset_index(drop=True)
