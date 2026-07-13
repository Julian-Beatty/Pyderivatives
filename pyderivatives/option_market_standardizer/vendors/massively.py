from __future__ import annotations

import re
import pandas as pd

from ..registry import register_vendor


_OCC_RE = re.compile(
    r"^O:(?P<root>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<right>[CP])(?P<strike>\d{8})$",
    re.IGNORECASE,
)


def _parse_occ_symbol(sym: str) -> pd.Series:
    m = _OCC_RE.match(str(sym).strip())

    if m is None:
        return pd.Series(
            {
                "underlying": None,
                "exdate": pd.NaT,
                "option_right": None,
                "strike": None,
            }
        )

    exdate = pd.Timestamp(
        year=2000 + int(m.group("yy")),
        month=int(m.group("mm")),
        day=int(m.group("dd")),
    )

    return pd.Series(
        {
            "underlying": m.group("root").upper(),
            "exdate": exdate.normalize(),
            "option_right": m.group("right").lower(),
            "strike": int(m.group("strike")) / 1000.0,
        }
    )


@register_vendor("massively")
def adapt_massively(raw: pd.DataFrame) -> pd.DataFrame:

    df = raw.copy()

    required = {"ticker"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"massively data missing required columns: {sorted(missing)}"
        )

    # ---------------------------------------
    # parse OCC ticker
    # ---------------------------------------
    parsed = df["ticker"].apply(_parse_occ_symbol)
    df = pd.concat([df, parsed], axis=1)

    # ---------------------------------------
    # normalize timestamps
    # ---------------------------------------
    if "window_start" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["window_start"],
            unit="ns",
            utc=True,
        )

        df["timestamp_et"] = (
            df["timestamp_utc"]
            .dt.tz_convert("America/New_York")
        )

        df["date"] = (
            df["timestamp_et"]
            .dt.tz_localize(None)
            .dt.normalize()
        )

    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    else:
        raise ValueError(
            "massively data requires either "
            "'window_start' or 'date'"
        )

    # ---------------------------------------
    # rename common polygon/massive columns
    # ---------------------------------------
    rename_map = {
        "close": "mid_price",
        "bid": "best_bid",
        "ask": "best_offer",
        "volume": "volume",
        "transactions": "transactions",
    }

    df = df.rename(
        columns={
            k: v for k, v in rename_map.items()
            if k in df.columns
        }
    )

    # ---------------------------------------
    # numeric conversion
    # ---------------------------------------
    for c in [
        "mid_price",
        "best_bid",
        "best_offer",
        "volume",
        "transactions",
        "strike",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------------------------------------
    # fallback quotes
    # ---------------------------------------
    if "mid_price" in df.columns:

        if "best_bid" not in df.columns:
            df["best_bid"] = df["mid_price"]

        if "best_offer" not in df.columns:
            df["best_offer"] = df["mid_price"]

    # ---------------------------------------
    # volume fallback
    # ---------------------------------------
    if "volume" not in df.columns:
        df["volume"] = 0

    # ---------------------------------------
    # cleanup
    # ---------------------------------------
    df = df.dropna(
        subset=[
            "date",
            "exdate",
            "option_right",
            "strike",
            "mid_price",
        ]
    )

    df = df[df["mid_price"] > 0]

    # ---------------------------------------
    # standardized output
    # ---------------------------------------
    keep_cols = [
        "date",
        "exdate",
        "option_right",
        "strike",
        "best_bid",
        "mid_price",
        "best_offer",
        "volume",
        "underlying",
        "ticker",
    ]

    optional_cols = [
        "transactions",
        "timestamp_et",
        "timestamp_utc",
    ]

    keep_cols.extend(
        [c for c in optional_cols if c in df.columns]
    )

    return df[keep_cols]