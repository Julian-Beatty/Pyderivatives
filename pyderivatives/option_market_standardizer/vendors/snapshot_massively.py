# pyderivatives/option_market_standardizer/vendors/massively_snapshot.py

from __future__ import annotations

import pandas as pd

from ..registry import register_vendor


@register_vendor("massively_snapshot")
def adapt_massively_snapshot(raw: pd.DataFrame) -> pd.DataFrame:

    df = raw.copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "market_date": "date",
        "expiration_date": "exdate",
        "contract_type": "option_right",
        "strike_price": "strike",
        "bid": "best_bid",
        "ask": "best_offer",
        "day_volume": "volume",
        "underlying_ticker": "underlying",
        "underlying_price": "stock_price_snapshot",
        "implied_volatility": "impl_volatility",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # --------------------------------------------------
    # Required date from manually assigned market_date
    # --------------------------------------------------
    if "date" not in df.columns:
        raise ValueError(
            "massively_snapshot requires a market_date column. "
            "Do not use extracted_at_utc as the option date."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    # --------------------------------------------------
    # Expiration
    # --------------------------------------------------
    if "exdate" not in df.columns:
        raise ValueError("massively_snapshot requires expiration_date.")

    df["exdate"] = pd.to_datetime(df["exdate"], errors="coerce").dt.normalize()

    # --------------------------------------------------
    # Option right
    # --------------------------------------------------
    if "option_right" not in df.columns:
        raise ValueError("massively_snapshot requires contract_type.")

    s = df["option_right"].astype(str).str.strip().str.lower()

    df["option_right"] = s.map(
        {
            "call": "c",
            "put": "p",
            "c": "c",
            "p": "p",
        }
    )

    # --------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------
    numeric_cols = [
        "strike",
        "best_bid",
        "best_offer",
        "mid_price",
        "last_price",
        "day_close",
        "day_open",
        "day_high",
        "day_low",
        "volume",
        "open_interest",
        "stock_price_snapshot",
        "impl_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --------------------------------------------------
    # Build mid price robustly
    # --------------------------------------------------
    if "mid_price" not in df.columns:
        df["mid_price"] = pd.NA

    if {"best_bid", "best_offer"}.issubset(df.columns):
        quote_mid = 0.5 * (df["best_bid"] + df["best_offer"])
        df["mid_price"] = df["mid_price"].fillna(quote_mid)

    if "last_price" in df.columns:
        df["mid_price"] = df["mid_price"].fillna(df["last_price"])

    if "day_close" in df.columns:
        df["mid_price"] = df["mid_price"].fillna(df["day_close"])

    # --------------------------------------------------
    # Build bid/ask robustly
    # Your core.py currently requires best_bid > 0 and best_offer > 0.
    # So for snapshot rows without quotes, use mid as synthetic bid/ask.
    # --------------------------------------------------
    if "best_bid" not in df.columns:
        df["best_bid"] = pd.NA

    if "best_offer" not in df.columns:
        df["best_offer"] = pd.NA

    df["best_bid"] = df["best_bid"].fillna(df["mid_price"])
    df["best_offer"] = df["best_offer"].fillna(df["mid_price"])

    # If bid is zero but mid is positive, replace with mid to avoid core.py dropping it
    df.loc[
        (df["best_bid"] <= 0) & (df["mid_price"] > 0),
        "best_bid",
    ] = df["mid_price"]

    df.loc[
        (df["best_offer"] <= 0) & (df["mid_price"] > 0),
        "best_offer",
    ] = df["mid_price"]

    # Ensure bid <= ask
    bad_spread = df["best_bid"] > df["best_offer"]
    df.loc[bad_spread, "best_bid"] = df.loc[bad_spread, "mid_price"]
    df.loc[bad_spread, "best_offer"] = df.loc[bad_spread, "mid_price"]

    # --------------------------------------------------
    # Volume fallback
    # --------------------------------------------------
    if "volume" not in df.columns:
        df["volume"] = 0

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    # --------------------------------------------------
    # Drop unusable rows
    # --------------------------------------------------
    df = df.dropna(
        subset=[
            "date",
            "exdate",
            "option_right",
            "strike",
            "best_bid",
            "mid_price",
            "best_offer",
        ]
    )

    df = df[
        (df["strike"] > 0)
        & (df["mid_price"] > 0)
        & (df["best_bid"] > 0)
        & (df["best_offer"] > 0)
        & (df["best_bid"] <= df["best_offer"])
    ].copy()

    keep_cols = [
        "date",
        "exdate",
        "option_right",
        "strike",
        "best_bid",
        "mid_price",
        "best_offer",
        "volume",
    ]

    optional_cols = [
        "open_interest",
        "underlying",
        "ticker",
        "stock_price_snapshot",
        "impl_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "day_open",
        "day_high",
        "day_low",
        "day_close",
        "last_price",
        "extracted_at_utc",
        "extracted_at_et",
    ]

    keep_cols.extend([c for c in optional_cols if c in df.columns])

    return df[keep_cols].reset_index(drop=True)