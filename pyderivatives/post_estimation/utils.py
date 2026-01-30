from typing import Optional, Union, Literal
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from typing import Optional, Literal, Dict, Tuple
import numpy as npcompute_horizon_returns_backward
import pandas as pd


from typing import Optional, Literal
import numpy as np
import pandas as pd
from typing import Any
from typing import Dict, Tuple, Iterable, Union

HorizonMethod = Literal["nearest", "interp"]


def compute_horizon_returns_backward(
    stock_df: pd.DataFrame,
    *,
    horizon: int,
    price_col: str = "price",
    date_col: str = "date",
    split_col: str = "ajexdi",
    return_type: str = "log",     # {"log","simple"}
    group_col: str | None = "tic",
    sort: bool = True,
    return_dataframe: bool = True,
) -> pd.DataFrame | pd.Series:
    """
    Backward-looking (trailing) horizon returns:
        r_t(h) = log(P_t) - log(P_{t-h})   [log]
        r_t(h) = P_t / P_{t-h} - 1         [simple]

    Uses split-adjusted prices if `split_col` exists; otherwise uses raw prices.

    Returns
    -------
    pd.DataFrame (default)
        Columns: [date_col, f"ret_{horizon}"]
    or
    pd.Series
        If return_dataframe=False.
    """
    df = stock_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if group_col is not None:
        groups = df.groupby(group_col, sort=False)
    else:
        groups = [(None, df)]

    ret = pd.Series(index=df.index, dtype=float)

    for _, g in groups:
        g = g.copy()
        if sort:
            g = g.sort_values(date_col)

        # split-adjusted price if available
        if split_col in g.columns:
            adj_price = g[price_col].astype(float) / g[split_col].astype(float)
        else:
            adj_price = g[price_col].astype(float)

        p = adj_price.to_numpy(float)

        # backward-looking: compare to t-horizon
        p_lag = np.roll(p, horizon)  # p_{t-h} aligned with p_t

        if return_type == "log":
            r = np.log(p) - np.log(p_lag)
        elif return_type == "simple":
            r = (p / p_lag) - 1.0
        else:
            raise ValueError("return_type must be 'log' or 'simple'")

        # first `horizon` obs have no lagged price
        r[:horizon] = np.nan

        ret.loc[g.index] = r

    if not return_dataframe:
        return ret

    out = pd.DataFrame(
        {
            date_col: df[date_col],
            f"ret_{horizon}": ret,
        }
    )
    return out

from typing import Dict, Tuple
import pandas as pd

from typing import Dict, Tuple, Iterable, Union
import pandas as pd


def extract_moment_premia_timeseries(
    *,
    physical_dict: dict,
    rnd_dict: dict,
    moments: Tuple[str, ...] = ("var", "skew", "kurt"),
    T_days: Union[int, Iterable[int]],
    method: str = "nearest",
    tol_years: float = 5 / 365,
    physical_table_key: str = "physical_moments_table",
    rnd_table_key: str = "rnd_moments_table",
) -> Dict[int, Dict[str, pd.Series]]:
    """
    Extract physical, risk-neutral, and risk premia time series
    for selected moments at one or multiple horizons.

    Returns
    -------
    dict:
        {T_days: {
            phys_<moment>: Series,
            rnd_<moment>: Series,
            prem_<moment>: Series
        }}
    """

    # ---- normalize T_days to iterable ----
    if isinstance(T_days, int):
        T_list = [T_days]
    else:
        T_list = list(T_days)

    out: Dict[int, Dict[str, pd.Series]] = {}

    for T in T_list:
        res_T: Dict[str, pd.Series] = {}

        for m in moments:
            # --- physical ---
            phys = extract_physical_moment_timeseries(
                physical_dict,
                f"{m}_r" if m != "var" else "var_r",
                T_days=T,
                series_name=m,
                method=method,
                table_key=physical_table_key,
                tol_years=tol_years,
            )

            # --- risk-neutral ---
            rnd = extract_physical_moment_timeseries(
                rnd_dict,
                m,
                T_days=T,
                series_name=m,
                method=method,
                table_key=rnd_table_key,
                tol_years=tol_years,
            )

            # --- premia ---
            prem = rnd - phys

            res_T[f"phys_{m}"] = phys
            res_T[f"rnd_{m}"] = rnd
            res_T[f"prem_{m}"] = prem

        out[T] = res_T

    return out


def extract_physical_moment_timeseries(
    result_dict: dict,
    moment_col: str,
    *,
    # choose ONE
    T_years: Optional[float] = None,
    T_days: Optional[float] = None,

    # ✅ NEW
    series_name: Optional[str] = None,

    method: HorizonMethod = "nearest",
    tol_years: Optional[float] = None,     # e.g. 3/365 for +/- 3 days
    date_format: Optional[str] = None,     # if your keys need parsing; else None
    drop_missing: bool = True,
    table_key: str = "physical_moments_table",
) -> pd.Series:
    """
    Extract a time series of a physical moment from each day's physical_moments_table
    at a chosen horizon.

    Expects per-day dict contains `table_key` (default 'physical_moments_table'),
    a DataFrame with:
      - column 'T' in YEARS
      - column `moment_col`

    Returns a pd.Series indexed by date.
    """
    if (T_years is None) == (T_days is None):
        raise ValueError("Provide exactly one of T_years or T_days.")

    T_target = float(T_years) if T_years is not None else float(T_days) / 365.0
    out = {}

    for k, day in result_dict.items():
        # --- parse date key ---
        try:
            dt = pd.to_datetime(k, format=date_format) if date_format else pd.to_datetime(k)
        except Exception:
            dt = k  # fallback: keep original key

        # --- fetch table ---
        if not isinstance(day, dict) or table_key not in day or day[table_key] is None:
            out[dt] = np.nan
            continue

        tbl = day[table_key]
        if not isinstance(tbl, pd.DataFrame):
            out[dt] = np.nan
            continue

        if "T" not in tbl.columns:
            raise KeyError(f"{table_key} must contain a 'T' column (years).")
        if moment_col not in tbl.columns:
            raise KeyError(
                f"'{moment_col}' not found in {table_key}. "
                f"Available columns: {list(tbl.columns)}"
            )

        T = pd.to_numeric(tbl["T"], errors="coerce").to_numpy()
        y = pd.to_numeric(tbl[moment_col], errors="coerce").to_numpy()

        mask = np.isfinite(T) & np.isfinite(y)
        T = T[mask]
        y = y[mask]

        if T.size == 0:
            out[dt] = np.nan
            continue

        # sort by maturity
        order = np.argsort(T)
        T = T[order]
        y = y[order]

        if method == "nearest":
            j = int(np.argmin(np.abs(T - T_target)))
            val = float(y[j])
            if tol_years is not None and abs(T[j] - T_target) > float(tol_years):
                val = np.nan
            out[dt] = val

        elif method == "interp":
            if T_target <= T[0]:
                val = float(y[0])
                if tol_years is not None and abs(T[0] - T_target) > float(tol_years):
                    val = np.nan
                out[dt] = val
            elif T_target >= T[-1]:
                val = float(y[-1])
                if tol_years is not None and abs(T[-1] - T_target) > float(tol_years):
                    val = np.nan
                out[dt] = val
            else:
                j_hi = int(np.searchsorted(T, T_target, side="right"))
                j_lo = j_hi - 1
                T0, T1 = float(T[j_lo]), float(T[j_hi])
                y0, y1 = float(y[j_lo]), float(y[j_hi])
                w = (T_target - T0) / (T1 - T0) if T1 != T0 else 0.0
                out[dt] = float((1 - w) * y0 + w * y1)
        else:
            raise ValueError("method must be 'nearest' or 'interp'.")

    s = pd.Series(out).sort_index()
    if drop_missing:
        s = s.dropna()

    # ✅ naming logic
    if series_name is not None:
        s.name = series_name
    else:
        s.name = f"physical:{moment_col}@{T_target:.6f}y"

    return s

def summary_stat(
    moments_by_asset: dict,
    *,
    horizons: list[int] | None = None,
    moments: tuple[str, ...] = ("vol_ann", "skew", "kurt"),
    which: tuple[str, ...] = ("phys",),      # keep ("phys",) to match your paper-style tables
    digits: int = 4,
    # ---- ADF on first differences ----
    adf_regression: str = "c",               # "c", "ct", "ctt", "n"
    adf_autolag: str | None = "AIC",         # "AIC", "BIC", "t-stat", or None
    # ---- Ljung-Box on first differences ----
    lb_lags: int = 2,                       # common choice; set to 20 if you prefer
):
    """
    Create summary statistic tables (paper-style) from `moments_by_asset`.

    For each horizon, creates 3 panels (vol/skew/kurt). Rows are ASSETS ONLY
    (no "phys series" column). Stats include:
      Mean, Std. Dev., Min, Max, Skewness, Kurtosis (raw), p(J-B),
      ADF (on first differences, with stars), p(LB) (Ljung-Box on first differences).

    Expected structure:
      moments_by_asset[asset][horizon_days][key] -> pandas.Series
    where key is like "phys_vol_ann", "phys_skew", "phys_kurt", etc.
    """
    import numpy as np
    import pandas as pd

    from scipy.stats import jarque_bera, skew as _skew, kurtosis as _kurtosis
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox

    # ---- infer horizons if not provided ----
    if horizons is None:
        horizons = None
        for _, d in moments_by_asset.items():
            if isinstance(d, dict) and len(d) > 0:
                horizons = sorted([k for k in d.keys() if isinstance(k, (int, np.integer))])
                break
        if horizons is None or len(horizons) == 0:
            raise ValueError("Could not infer horizons from moments_by_asset keys. Pass `horizons=[...]`.")

    PANEL_LABELS = {
        "vol_ann": "Panel A: Volatility",
        "skew":    "Panel B: Skewness",
        "kurt":    "Panel C: Kurtosis",
    }

    def _adf_on_diff_with_stars(x: np.ndarray):
        x = x[np.isfinite(x)]
        dx = np.diff(x)
        if dx.size < 20:
            return np.nan, "", np.nan
        try:
            stat, pval, *_ = adfuller(dx, regression=adf_regression, autolag=adf_autolag)
            stat = float(stat)
            pval = float(pval)
            if pval <= 0.01:
                stars = "***"
            elif pval <= 0.05:
                stars = "**"
            elif pval <= 0.10:
                stars = "*"
            else:
                stars = ""
            return stat, stars, pval
        except Exception:
            return np.nan, "", np.nan

    def _lb_pvalue_on_diff(x: np.ndarray):
        x = x[np.isfinite(x)]
        dx = np.diff(x)
        if dx.size < (lb_lags + 5):
            return np.nan
        try:
            # returns DataFrame by default if return_df=True
            lb = acorr_ljungbox(dx, lags=[lb_lags], return_df=True)
            return float(lb["lb_pvalue"].iloc[0])
        except Exception:
            return np.nan

    def _series_stats(s: pd.Series):
        x = pd.to_numeric(s, errors="coerce").dropna().values
        if x.size == 0:
            return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
                        skew=np.nan, kurt=np.nan, jb_p=np.nan,
                        adf=np.nan, adf_stars="", lb_p=np.nan)

        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
        mn = float(np.min(x))
        mx = float(np.max(x))
        sk = float(_skew(x, bias=False)) if x.size > 2 else np.nan
        ku = float(_kurtosis(x, fisher=False, bias=False)) if x.size > 3 else np.nan  # raw kurtosis

        try:
            jb_p = float(jarque_bera(x).pvalue)
        except Exception:
            jb_p = np.nan

        adf_stat, stars, _ = _adf_on_diff_with_stars(x)
        lb_p = _lb_pvalue_on_diff(x)

        return dict(mean=mean, std=std, min=mn, max=mx, skew=sk, kurt=ku,
                    jb_p=jb_p, adf=adf_stat, adf_stars=stars, lb_p=lb_p)

    assets = list(moments_by_asset.keys())
    out = {}

    for H in horizons:
        panel_dfs = {}
        combined_parts = []

        for m in moments:
            panel = PANEL_LABELS.get(m, f"Panel: {m}")

            rows = []
            idx = []

            for asset in assets:
                hdict = moments_by_asset.get(asset, {})
                if H not in hdict:
                    continue
                if not isinstance(hdict[H], dict):
                    continue

                # If you pass multiple `which`, we will append them to the asset name.
                for w in which:
                    key = f"{w}_{m}"
                    s = hdict[H].get(key, None)
                    if s is None:
                        continue

                    stats = _series_stats(s)
                    asset_label = asset if len(which) == 1 else f"{asset}-{w}"
                    idx.append(asset_label)

                    rows.append([
                        stats["mean"],
                        stats["std"],
                        stats["min"],
                        stats["max"],
                        stats["skew"],
                        stats["kurt"],
                        stats["jb_p"],
                        stats["adf"],
                        stats["adf_stars"],
                        stats["lb_p"],
                    ])

            df = pd.DataFrame(
                rows,
                index=pd.Index(idx, name="Asset"),
                columns=["Mean", "Std. Dev.", "Min.", "Max.", "Skewness", "Kurtosis", "p (J-B)", "ADF", "_stars", "p (LB)"],
            )

            if not df.empty:
                df["ADF"] = df.apply(
                    lambda r: (np.nan if pd.isna(r["ADF"]) else f"{float(r['ADF']):.{digits}f}{r['_stars']}"),
                    axis=1,
                )
                df = df.drop(columns=["_stars"])

            panel_dfs[panel] = df

            if not df.empty:
                # panel header row for combined table
                header = pd.DataFrame(
                    [[np.nan] * df.shape[1]],
                    index=pd.Index([panel], name="Asset"),
                    columns=df.columns,
                )
                combined_parts.append(header)
                combined_parts.append(df)

        combined_df = pd.concat(combined_parts) if combined_parts else pd.DataFrame()

        # ---- LaTeX ----
        latex = ""
        if not combined_df.empty:
            df_ltx = combined_df.copy()

            num_cols = ["Mean", "Std. Dev.", "Min.", "Max.", "Skewness", "Kurtosis", "p (J-B)", "p (LB)"]
            for c in num_cols:
                df_ltx[c] = df_ltx[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.{digits}f}")
            df_ltx["ADF"] = df_ltx["ADF"].map(lambda v: "" if (v is np.nan or pd.isna(v)) else str(v))

            latex = df_ltx.to_latex(
                escape=False,
                index=True,
                caption=f"Summary statistics for moment series ({H}-day horizon). "
                        f"ADF and Ljung–Box tests are applied to first differences (lags={lb_lags} for Ljung–Box).",
                label=f"tab:summary_stats_{H}d",
            )

        out[H] = {"tables": panel_dfs, "df": combined_df, "latex": latex}

    return out


def call_fit_error_timeseries(
    result_dict: Dict[Any, dict],
    *,
    date_col: str = "date",
    metrics: Tuple[str, ...] = ("rmse", "mape"),
    require_success: bool = True,
    plot: bool = False,
    title: Optional[str] = None,
    save_csv: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a time series of fitted call price errors from a result_dict.

    Expected structure per day:
        day = result_dict[date]
        day["errors"] is a dict containing keys like "rmse", "mape"
        optionally day["success"] is bool

    Parameters
    ----------
    result_dict : dict
        date_key -> day dict
    metrics : tuple[str]
        which error metrics to extract from day["errors"]
    require_success : bool
        if True, skip days where day.get("success") is False
    plot : bool
        if True, plot the metric time series
    save_csv : str|Path|None
        if provided, save the time series DataFrame to CSV

    Returns
    -------
    ts_df : pd.DataFrame
        columns: [date_col] + metrics + optional 'success'
    summary_df : pd.DataFrame
        summary statistics for each metric (count, mean, std, min, p05, p25, median, p75, p95, max)
    """

    rows = []
    for k, day in result_dict.items():
        if not isinstance(day, dict):
            continue

        success = bool(day.get("success", True))
        if require_success and not success:
            continue

        errs = day.get("errors", None)
        if not isinstance(errs, dict):
            continue

        row = {date_col: k, "success": success}
        ok = True
        for m in metrics:
            v = errs.get(m, np.nan)
            try:
                row[m] = float(v)
            except Exception:
                row[m] = np.nan
            if np.isnan(row[m]):
                ok = False  # still keep row; you can choose to drop later
        rows.append(row)

    if len(rows) == 0:
        ts_df = pd.DataFrame(columns=[date_col, "success", *metrics])
        summary_df = pd.DataFrame(index=list(metrics))
        return ts_df, summary_df

    ts_df = pd.DataFrame(rows)

    # normalize/parse date column
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
    ts_df = ts_df.sort_values(date_col).reset_index(drop=True)

    # summary stats
    def _summary(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return pd.Series(
                {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
                 "p05": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan}
            )
        q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        return pd.Series(
            {
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
                "min": float(s.min()),
                "p05": float(q.loc[0.05]),
                "p25": float(q.loc[0.25]),
                "median": float(q.loc[0.50]),
                "p75": float(q.loc[0.75]),
                "p95": float(q.loc[0.95]),
                "max": float(s.max()),
            }
        )

    summary_df = pd.DataFrame({m: _summary(ts_df[m]) for m in metrics}).T

    # save
    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        ts_df.to_csv(save_csv, index=False)

    # plot
    if plot:
        ttl = title or "Call fit error time series"
        for m in metrics:
            plt.figure(figsize=(10, 3.2))
            plt.plot(ts_df[date_col], ts_df[m])
            plt.title(f"{ttl}: {m}")
            plt.xlabel("Date")
            plt.ylabel(m)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.show()

    return ts_df, summary_df