##optionsdxfrom __future__ import annotations
import pandas as pd

from ..registry import register_vendor
from ..utils import ensure_timestamp, compute_mid


@register_vendor("optionsdx")
def adapt_optionsdx(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = df.columns.str.strip().str.strip("[]")

    rename_map = {
        "QUOTE_DATE": "date",
        "EXPIRY_DATE": "exdate",
        "OPTION_RIGHT": "option_right",
        "STRIKE": "strike",
        "BID_PRICE": "best_bid",
        "ASK_PRICE": "best_offer",
        "VOLUME": "volume",}

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df = ensure_timestamp(df, ["date", "exdate"])

    if "option_right" in df.columns:
        s = (
            df["option_right"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df["option_right"] = s.map({"call": "c", "put": "p"}).fillna("c")
    else:
        df["option_right"] = "c"
        print("No option type found, assuming all data are calls")
        
        
    for c in ("strike", "best_bid", "best_offer", "volume", "open_interest"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)

    if "best_bid" in df.columns and "best_offer" in df.columns:
        df["mid_price"] = compute_mid(df, "best_bid", "best_offer")
     
    return df

#df=opt_raw.copy()

