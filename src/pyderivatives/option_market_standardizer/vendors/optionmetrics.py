from __future__ import annotations
import pandas as pd

from ..registry import register_vendor
from ..utils import ensure_timestamp, compute_mid


@register_vendor("optionmetrics")
def adapt_optionmetrics(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    rename_map = {
        "date": "date",
        "exdate": "exdate",
        "cp_flag": "option_right",
        "strike_price": "strike",
        "best_bid": "best_bid",
        "best_offer": "best_offer",
        "volume": "volume",
        "open_interest": "open_interest",
        "ticker": "underlying",
        "last": "last",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df = ensure_timestamp(df, ["date", "exdate"])
    df["strike"]=df["strike"]/1000
    if "option_right" in df.columns:
        s = (
            df["option_right"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        #df["option_right"] = s.map({"C": "c", "P": "p"}).fillna("c")

    for c in ("strike", "best_bid", "best_offer", "volume", "volume", "last"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if "best_bid" in df.columns and "best_offer" in df.columns:
        df["mid_price"] = compute_mid(df, "best_bid", "best_offer")

    return df
