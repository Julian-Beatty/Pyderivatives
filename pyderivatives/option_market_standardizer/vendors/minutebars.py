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


@register_vendor("minute_bars")
def adapt_minute_bars(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    required = {"ticker", "start_nano", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"minute_bars data missing required columns: {sorted(missing)}")

    df["timestamp_utc"] = pd.to_datetime(df["start_nano"], unit="ns", utc=True)
    df["timestamp_et"] = df["timestamp_utc"].dt.tz_convert("America/New_York")

    df["date"] = df["timestamp_et"].dt.tz_localize(None).dt.normalize()

    parsed = df["ticker"].apply(_parse_occ_symbol)
    df = pd.concat([df, parsed], axis=1)

    for c in ["open", "close", "high", "low", "volume", "transactions"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0

    if "transactions" not in df.columns:
        df["transactions"] = 0

    df = df.dropna(
        subset=["date", "exdate", "option_right", "strike", "close"]
    )

    df = df[df["close"] > 0]

    df = df.sort_values(["date", "ticker", "timestamp_et"])

    eod = (
        df.groupby(["date", "ticker"], as_index=False)
          .tail(1)
          .copy()
          .reset_index(drop=True)
    )

    daily_stats = (
        df.groupby(["date", "ticker"], as_index=False)
          .agg(
              volume=("volume", "sum"),
              transactions=("transactions", "sum"),
          )
    )

    eod = eod.drop(columns=["volume", "transactions"], errors="ignore")
    eod = eod.merge(daily_stats, on=["date", "ticker"], how="left")

    eod["mid_price"] = eod["close"]
    eod["best_bid"] = eod["close"]
    eod["best_offer"] = eod["close"]

    eod["exdate"] = pd.to_datetime(eod["exdate"]).dt.normalize()

    return eod[
        [
            "date",
            "exdate",
            "option_right",
            "strike",
            "best_bid",
            "mid_price",
            "best_offer",
            "volume",
            "transactions",
            "underlying",
            "ticker",
            "timestamp_et",
        ]
    ]