from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Union, Tuple, Any, Dict, List
import numpy as np
import pandas as pd
from typing import Literal, Optional, Sequence, Dict

from .registry import VENDOR_REGISTRY
from .utils import ensure_timestamp, yearfrac_act365


from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Any

import pandas as pd

@dataclass
class OptionMarketStandardizer:
    """
    Loads option + stock data, applies vendor adapter to options,
    then merges spot (S0) and rates (r) and computes maturity (T).

    Eager-loads on initialization:
      - self.options_raw
      - self.stock_raw
      - self.opt_std
    """
    option_data_filename_prefix: str
    stock_data_filename: str
    rate_curve_df: pd.DataFrame

    data_directory_path: Optional[Union[str, Path]] = None
    vendor_name: str = "optionmetrics"

    # stock config
    stock_date_col: str = "date"
    stock_price_col: str = "close"

    # rate curve config
    rate_date_col: str = "date"
    default_rate_maturity_col: str = "3M"

    # outputs populated during __post_init__
    options_raw: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    stock_raw: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    opt_std: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # --- vendor selection ---
        self.vendor_key = self.vendor_name.strip().lower()
        if self.vendor_key not in VENDOR_REGISTRY:
            raise ValueError(
                f"Unsupported vendor_name='{self.vendor_name}'. "
                f"Supported: {sorted(VENDOR_REGISTRY.keys())}"
            )

        # --- base path ---
        self.base_path = Path(self.data_directory_path) if self.data_directory_path else Path(".")

        # --- normalize rate curve ---
        if not isinstance(self.rate_curve_df, pd.DataFrame):
            raise TypeError("rate_curve_df must be a pandas DataFrame.")
        self.rate_curve_df = self._normalize_rate_curve(self.rate_curve_df)

        # --- eager load raw + standardized ---
        opt_raw = self._read_many_csv(self.option_data_filename_prefix)
        stock_raw = pd.read_csv(self.base_path / self.stock_data_filename)

        self.options_raw = opt_raw.copy()
        self.stock_raw = stock_raw.copy()

        opt_std = self.standardize_frames(opt_raw, stock_raw)

        # keep your exact vendor-specific scaling behavior
        self.stock_raw["date"] = pd.to_datetime(self.stock_raw["date"])
        if self.vendor_name == "optionsdx":
            opt_std["mid_price"] = opt_std["mid_price"] * opt_std["stock_price"]
            opt_std["best_bid"] = opt_std["best_bid"] * opt_std["stock_price"]
            opt_std["best_offer"] = opt_std["best_offer"] * opt_std["stock_price"]

        self.opt_std = opt_std

    def _normalize_rate_curve(self, rc: pd.DataFrame) -> pd.DataFrame:
        out = rc.copy()
        if out.index.name == self.rate_date_col:
            out = out.reset_index()

        if self.rate_date_col not in out.columns:
            raise ValueError(f"rate_curve_df must have a '{self.rate_date_col}' column or index named that.")

        out[self.rate_date_col] = pd.to_datetime(out[self.rate_date_col])
        return out

    def _read_many_csv(self, prefix: str) -> pd.DataFrame:
        """
        Reads either:
          - exact file if prefix endswith .csv or .txt
          - otherwise all files in folder that start with prefix and end with .csv or .txt
        """
        p = Path(prefix)
        ext = p.suffix.lower()

        # --- exact file path ---
        if ext in {".csv", ".txt"}:
            full = (self.base_path / p) if not p.is_absolute() else p
            return pd.read_csv(full)

        # --- prefix search (multi-file) ---
        files = sorted(self.base_path.glob(f"{prefix}*.csv")) + sorted(self.base_path.glob(f"{prefix}*.txt"))
        if not files:
            raise FileNotFoundError(f"No files found for prefix='{prefix}' in {self.base_path.resolve()}")

        dfs = [pd.read_csv(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def standardize_frames(self, opt_raw: pd.DataFrame, stock_raw: pd.DataFrame) -> pd.DataFrame:
        # vendor adapter
        opt_std = VENDOR_REGISTRY[self.vendor_key](opt_raw)

        # stock -> (date, S0)
        stock = stock_raw.copy()

        if self.stock_date_col not in stock.columns:
            raise ValueError(f"stock file must contain '{self.stock_date_col}' column.")
        if self.stock_price_col not in stock.columns:
            raise ValueError(f"stock file must contain '{self.stock_price_col}' column.")

        stock[self.stock_date_col] = pd.to_datetime(stock[self.stock_date_col])
        stock = stock[[self.stock_date_col, self.stock_price_col]].rename(
            columns={self.stock_date_col: "date", self.stock_price_col: "stock_price"}
        )

        # merge S0
        opt_std = ensure_timestamp(opt_std, ["date", "exdate"])
        opt_std = opt_std.merge(stock, on="date", how="left")

        # compute T / DTE
        EPS = 2.05e-4  # ~0.000365 days ≈ 2.4 hours 
        opt_std["rounded_maturity"] = yearfrac_act365(opt_std["date"], opt_std["exdate"])
        opt_std["rounded_maturity"] = opt_std["rounded_maturity"].clip(lower=EPS) # 0 maturity clipped to tiny maturity
        opt_std["T"] = yearfrac_act365(opt_std["date"], opt_std["exdate"])
        opt_std["DTE"] = (pd.to_datetime(opt_std["exdate"]) - pd.to_datetime(opt_std["date"])).dt.days
        opt_std["DTE"] = opt_std["DTE"].clip(lower=1).astype(int)

        # attach rates
        rc = self.rate_curve_df
        opt_std = attach_rates_from_surface_nearest(
            opt_std,
            rc,
            opt_date_col="date",
            opt_exdate_col="exdate",
            opt_dte_col="DTE",
            rate_date_col=self.rate_date_col,
            force_min_dte=1,
            date_tolerance_days=None,
        )

        opt_std = opt_std.rename(columns={"r": "risk_free_rate"})
        opt_std["risk_free_rate"] = opt_std["risk_free_rate"] / 100
        opt_std["moneyness"] = (opt_std["strike"] / opt_std["stock_price"])

        opt_std = opt_std[
            (opt_std["best_bid"] <= opt_std["best_offer"]) &
            (opt_std["best_offer"] > 0) &
            (opt_std["mid_price"] > 0) &
            (opt_std["best_bid"] > 0)
        ]

        opt_std = collapse_duplicate_strikes(opt_std)

        needed = {
            "date", "exdate", "option_right", "strike", "moneyness",
            "best_bid", "best_offer", "mid_price",
            "rounded_maturity", "stock_price", "risk_free_rate", "volume"
        }
        missing = needed - set(opt_std.columns)
        if missing:
            raise ValueError(f"Standardization incomplete. Missing columns: {sorted(missing)}")

        opt_std = opt_std[
            ["date","exdate","rounded_maturity","moneyness","option_right","strike",
             "best_bid","mid_price","best_offer","stock_price","risk_free_rate","volume"]
        ]
        opt_std = opt_std.sort_values(by=["date", "rounded_maturity", "option_right", "strike"])
        return opt_std

    def keep_options(
        self,
        date_filter: Optional[Iterable[Union[str, pd.Timestamp]]] = None,
        maturity_filter: Optional[Iterable[Union[int, float]]] = None,
        moneyness_filter: Optional[Iterable[float]] = None,
        min_volume_filter: Optional[float] = None,
        min_price_filter: Optional[float] = None,
        option_right_filter: Optional[Union[str, Iterable[str]]] = None,
        junk_quotes: bool = False
    ) -> pd.DataFrame:
        """
        Keep only options that are within these filters (inclusive).
    
        Parameters
        ----------
        date_filter : iterable length-2, optional
            Keep options between [start_date, end_date] inclusive.
            Example: ["2021-06-01", "2022-06-01"].
        maturity_filter : iterable length-2, optional
            Keep options with rounded_maturity between [min_days, max_days] inclusive.
            Example: [30, 90].
        moneyness_filter : iterable length-2, optional
            Keep options with abs moneyness between [m_min, m_max] inclusive.
            moneyness = abs(K/S)
            Example: [0.8, 1.2].
        min_volume_filter : float, optional
            Keep options with volume >= min_volume_filter.
        min_price_filter : float, optional
            Keep options with mid_price >= min_price_filter (or you can switch to best_offer).
        option_right_filter : str or iterable[str], optional
            Keep only specified option types. Examples: "c", "p", ["c","p"].
    
        Returns
        -------
        pd.DataFrame
            Filtered options dataframe.
        """
        if not hasattr(self, "opt_std") or self.opt_std is None:
            raise ValueError("self.opt_std is not set. Call load() first.")
    
        df = self.opt_std.copy()
    
        # --- date filter ---
        if date_filter is not None:
            start, end = list(date_filter)
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            df = df[(df["date"] >= start) & (df["date"] <= end)]
    
        # --- maturity (days) filter ---
        if maturity_filter is not None:
            mmin, mmax = list(maturity_filter)
            df = df[(df["rounded_maturity"] >= mmin/365) & (df["rounded_maturity"] <= mmax/365)]
    
        # --- moneyness filter ---
        if moneyness_filter is not None:
            xmin, xmax = list(moneyness_filter)
            df = df[(df["moneyness"] >= xmin) & (df["moneyness"] <= xmax)]
    
        # --- volume filter ---
        if min_volume_filter is not None:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            df = df[df["volume"] >= float(min_volume_filter)]
    
        # --- price filter (mid) ---
        if min_price_filter is not None:
            df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce")
            df = df[df["mid_price"] >= float(min_price_filter)]
    
        # --- option type filter ---
        if option_right_filter is not None:
            if isinstance(option_right_filter, str):
                allowed = {option_right_filter.strip().lower()}
            else:
                allowed = {str(x).strip().lower() for x in option_right_filter}
    
            # normalize option_right to lowercase first char
            df["option_right"] = df["option_right"].astype(str).str.strip().str.lower().str[0]
            df = df[df["option_right"].isin(allowed)]
        if junk_quotes==False:
            df = df[
                (df["best_bid"] <= df["best_offer"]) &
                (df["best_offer"] > 0) &
                (df["mid_price"] > 0) & (df["best_bid"]>0)
            ]    
                    # keep consistent ordering
        df = df.sort_values(by=["date", "rounded_maturity", "option_right", "strike"]).reset_index(drop=True)
        return df
        


def put_call_parity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert calls <-> puts using put-call parity, updating quote columns and flipping option_right.

    Assumes:
      - rounded_maturity is already T in YEARS (e.g. days/365)
      - risk_free_rate is in DECIMAL units (e.g. 0.03, not 3)
      - no dividends (q=0)

    Parity:
      C - P = S - K*exp(-rT)
      => P = C - S + K*exp(-rT)
      => C = P + S - K*exp(-rT)
    """
    out = df.copy()

    required = {"option_right", "strike", "stock_price", "risk_free_rate", "rounded_maturity"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"_put_call_parity missing required columns: {sorted(missing)}")

    # normalize option_right
    out["option_right"] = out["option_right"].astype(str).str.strip().str.lower().str[0]

    # numeric columns
    for c in ["strike", "stock_price", "risk_free_rate", "rounded_maturity"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    S = out["stock_price"]
    K = out["strike"]
    r = out["risk_free_rate"]
    T = out["rounded_maturity"].clip(lower=0.0)  # years

    discK = K * np.exp(-r * T)

    def convert_price_col(col: str) -> None:
        if col not in out.columns:
            return

        px = pd.to_numeric(out[col], errors="coerce")

        is_call = out["option_right"].eq("c")
        is_put  = out["option_right"].eq("p")

        px_new = px.copy()
        # calls -> puts
        px_new[is_call] = px[is_call] - S[is_call] + discK[is_call]
        # puts -> calls
        px_new[is_put]  = px[is_put] + S[is_put] - discK[is_put]

        out[col] = px_new

    for col in ["best_bid", "mid_price", "best_offer"]:
        convert_price_col(col)

    # flip right
    out["option_right"] = out["option_right"].map({"c": "p", "p": "c"}).fillna(out["option_right"])

    return out




def _parse_day_from_col(col: str) -> int | None:
    # "90/365" -> 90
    if isinstance(col, str) and "/" in col:
        num, den = col.split("/", 1)
        num = num.strip()
        if num.isdigit():
            return int(num)
    return None

def attach_rates_from_surface_nearest(
    opt: pd.DataFrame,
    rate_surface: pd.DataFrame,
    *,
    opt_date_col: str = "date",
    opt_exdate_col: str = "exdate",
    opt_dte_col: str = "DTE",            # or "rounded_maturity" if that's your DTE-in-days
    rate_date_col: str = "Date",
    force_min_dte: int = 1,              # 0DTE -> 1
    date_tolerance_days: int | None = None,   # None = no limit
) -> pd.DataFrame:
    out = opt.copy()

    # --- normalize option date and DTE ---
    out[opt_date_col] = pd.to_datetime(out[opt_date_col]).dt.normalize()

    if opt_dte_col not in out.columns:
        out[opt_dte_col] = (
            pd.to_datetime(out[opt_exdate_col]) - pd.to_datetime(out[opt_date_col])
        ).dt.days

    dte = pd.to_numeric(out[opt_dte_col], errors="coerce")
    if dte.isna().any():
        bad = out.loc[dte.isna(), [opt_date_col, opt_exdate_col]].head(5)
        raise ValueError(f"Some rows have invalid DTE. Example rows:\n{bad}")

    dte = dte.clip(lower=force_min_dte).round().astype(int)
    out["_dte_int"] = dte

    # --- normalize rate surface date ---
    rc = rate_surface.copy()
    if rate_date_col in rc.columns:
        rc = rc.rename(columns={rate_date_col: "rate_date"})
    elif rc.index.name == rate_date_col:
        rc = rc.reset_index().rename(columns={rate_date_col: "rate_date"})
    else:
        raise ValueError(f"rate_surface must have '{rate_date_col}' column or index.")

    rc["rate_date"] = pd.to_datetime(rc["rate_date"]).dt.normalize()

    # --- build list of available maturity days and their columns ---
    mat_cols = [c for c in rc.columns if c != "rate_date"]
    day_map = {c: _parse_day_from_col(c) for c in mat_cols}
    keep = [(c, d) for c, d in day_map.items() if d is not None]
    if not keep:
        raise ValueError("No maturity columns like '90/365' found in rate_surface.")

    # sort maturities by day
    keep.sort(key=lambda x: x[1])
    cols_sorted = [c for c, _ in keep]
    days_sorted = np.array([d for _, d in keep], dtype=int)

    # keep only rate_date + maturity columns in sorted order
    rc = rc[["rate_date"] + cols_sorted].sort_values("rate_date").reset_index(drop=True)

    # --- Step A: map option date -> nearest available rate_date (weekends ok) ---
    unique_rate_dates = rc[["rate_date"]].drop_duplicates().sort_values("rate_date").reset_index(drop=True)
    out = out.sort_values(opt_date_col).reset_index(drop=True)

    tol = None if date_tolerance_days is None else pd.Timedelta(days=date_tolerance_days)

    out = pd.merge_asof(
        out,
        unique_rate_dates,
        left_on=opt_date_col,
        right_on="rate_date",
        direction="nearest",
        tolerance=tol,
    )

    if out["rate_date"].isna().any():
        bad_dates = out.loc[out["rate_date"].isna(), opt_date_col].head(5).tolist()
        raise ValueError(f"Some option dates could not be matched to a rate curve date. Examples: {bad_dates}")

    # --- Step B: nearest maturity day via searchsorted ---
    # index of nearest maturity day for each option
    idx = np.searchsorted(days_sorted, out["_dte_int"].to_numpy(), side="left")
    idx = np.clip(idx, 0, len(days_sorted) - 1)

    # compare left vs right neighbor to choose nearest
    # (only where there is a left neighbor)
    has_left = idx > 0
    left_idx = idx - 1
    right_idx = idx

    d = out["_dte_int"].to_numpy()
    choose_left = np.zeros_like(idx, dtype=bool)
    choose_left[has_left] = (d[has_left] - days_sorted[left_idx[has_left]]) <= (days_sorted[right_idx[has_left]] - d[has_left])
    nearest_idx = np.where(choose_left, left_idx, right_idx)

    out["r_dte_days"] = days_sorted[nearest_idx]

    # --- pull the rate values ---
    # reindex rc to align with out rows by rate_date
    rc_idxed = rc.set_index("rate_date")
    rows = out["rate_date"].to_numpy()
    # column name for each row
    col_for_row = np.array(cols_sorted, dtype=object)[nearest_idx]

    # vectorized gather: do it by splitting by column (fast enough and simple)
    out["r"] = np.nan
    for c in np.unique(col_for_row):
        mask = col_for_row == c
        out.loc[mask, "r"] = rc_idxed.loc[out.loc[mask, "rate_date"], c].to_numpy()

    out = out.drop(columns=["_dte_int"])
    return out

AggPrice = Literal["max", "min", "mean", "median", "last", "vwavg"]


def collapse_duplicate_strikes(
    df,
    *,
    date_col="date",
    T_col="rounded_maturity",
    right_col="option_right",
    K_col="strike",
    mid_col="mid_price",
    bid_col="best_bid",
    ask_col="best_offer",
    vol_col="volume",
    keep_cols_first=("exdate", "stock_price", "risk_free_rate", "moneyness"),
    sort_output=True,
):
    out = df.copy()

    # ensure clean grouping keys
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[T_col] = pd.to_numeric(out[T_col], errors="coerce")
    out[K_col] = pd.to_numeric(out[K_col], errors="coerce")
    out[right_col] = out[right_col].astype(str)

    agg = {}

    # carry-through columns
    for c in keep_cols_first:
        if c in out.columns:
            agg[c] = "first"

    # vectorized aggregations
    agg[mid_col] = "max"   # <- vectorized mean
    if bid_col in out.columns:
        agg[bid_col] = "max"
    if ask_col in out.columns:
        agg[ask_col] = "min"
    if vol_col in out.columns:
        agg[vol_col] = "sum"

    keys = [date_col, T_col, right_col, K_col]

    collapsed = (
        out.groupby(keys, sort=False, as_index=False)
           .agg(agg)
    )

    if sort_output:
        collapsed = collapsed.sort_values(keys).reset_index(drop=True)

    return collapsed
