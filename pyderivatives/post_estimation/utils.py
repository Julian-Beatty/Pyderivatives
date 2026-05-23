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


from pathlib import Path

from pathlib import Path

def summary_stat(
    moments_by_asset: dict,
    *,
    horizons: list[int] | None = None,
    moments: tuple[str, ...] = ("vol_ann", "skew", "kurt"),
    which: tuple[str, ...] = ("phys",),
    digits: int = 3,
    adf_regression: str = "c",
    adf_autolag: str | None = "AIC",
    lb_lags: int = 2,
    save_dir: str | Path = ".",
    file_stub: str = "summary_stats",
    use_small_headers: bool = True,
):
    """
    Create summary-statistic tables from `moments_by_asset`.

    Features:
    - JB and LB report test statistics, not p-values
    - significance stars are based on p-values
    - ADF and Ljung-Box are applied to first differences
    - one LaTeX table per horizon
    - one combined LaTeX table across all horizons
    - no visible NaNs in header rows
    - inserts \\midrule between horizons in the all-horizons table

    Expected structure:
      moments_by_asset[asset][horizon_days][key] -> pandas.Series
    where key is like "phys_vol_ann", "phys_skew", "phys_kurt", etc.
    """
    import numpy as np
    import pandas as pd

    from scipy.stats import jarque_bera, skew as _skew, kurtosis as _kurtosis
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
        "vol_ann": "Panel A: Vol.",
        "skew":    "Panel B: Skew.",
        "kurt":    "Panel C: Kurt.",
    }

    if use_small_headers:
        COLS = ["Mean", "SD", "Min", "Max", "Skew", "Kurt", "JB", "ADF", f"LB({lb_lags})"]
    else:
        COLS = ["Mean", "Std. Dev.", "Min.", "Max.", "Skewness", "Kurtosis", "J-B", "ADF", f"LB({lb_lags})"]

    def _stars_from_pval(pval: float) -> str:
        if pd.isna(pval):
            return ""
        if pval <= 0.01:
            return "***"
        elif pval <= 0.05:
            return "**"
        elif pval <= 0.10:
            return "*"
        return ""

    def _fmt_num(val):
        return "" if pd.isna(val) else f"{float(val):.{digits}f}"

    def _fmt_stat(val, stars=""):
        return "" if pd.isna(val) else f"{float(val):.{digits}f}{stars}"

    def _adf_on_diff_with_stars(x: np.ndarray):
        x = x[np.isfinite(x)]
        dx = np.diff(x)
        if dx.size < 20:
            return np.nan, "", np.nan
        try:
            stat, pval, *_ = adfuller(dx, regression=adf_regression, autolag=adf_autolag)
            stat = float(stat)
            pval = float(pval)
            return stat, _stars_from_pval(pval), pval
        except Exception:
            return np.nan, "", np.nan

    def _lb_stat_on_diff_with_stars(x: np.ndarray):
        x = x[np.isfinite(x)]
        dx = np.diff(x)
        if dx.size < (lb_lags + 5):
            return np.nan, "", np.nan
        try:
            lb = acorr_ljungbox(dx, lags=[lb_lags], return_df=True)
            stat = float(lb["lb_stat"].iloc[0])
            pval = float(lb["lb_pvalue"].iloc[0])
            return stat, _stars_from_pval(pval), pval
        except Exception:
            return np.nan, "", np.nan

    def _jb_stat_with_stars(x: np.ndarray):
        x = x[np.isfinite(x)]
        if x.size < 5:
            return np.nan, "", np.nan
        try:
            jb = jarque_bera(x)
            stat = float(jb.statistic)
            pval = float(jb.pvalue)
            return stat, _stars_from_pval(pval), pval
        except Exception:
            return np.nan, "", np.nan

    def _series_stats(s):
        x = pd.to_numeric(s, errors="coerce").dropna().values
        if x.size == 0:
            return dict(
                mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
                skew=np.nan, kurt=np.nan,
                jb=np.nan, jb_stars="",
                adf=np.nan, adf_stars="",
                lb=np.nan, lb_stars=""
            )

        jb_stat, jb_stars, _ = _jb_stat_with_stars(x)
        adf_stat, adf_stars, _ = _adf_on_diff_with_stars(x)
        lb_stat, lb_stars, _ = _lb_stat_on_diff_with_stars(x)

        return dict(
            mean=float(np.mean(x)),
            std=float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
            min=float(np.min(x)),
            max=float(np.max(x)),
            skew=float(_skew(x, bias=False)) if x.size > 2 else np.nan,
            kurt=float(_kurtosis(x, fisher=False, bias=False)) if x.size > 3 else np.nan,
            jb=jb_stat,
            jb_stars=jb_stars,
            adf=adf_stat,
            adf_stars=adf_stars,
            lb=lb_stat,
            lb_stars=lb_stars,
        )

    def _make_header_row(label: str, columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [[""] * len(columns)],
            index=pd.Index([label], name="Asset"),
            columns=columns,
        )

    def _latex_wrap(tabular: str, caption: str, label: str) -> str:
        return (
            "\\begin{table}[!htbp]\n"
            "\\centering\n"
            "\\scriptsize\n"
            "\\setlength{\\tabcolsep}{4pt}\n"
            "\\renewcommand{\\arraystretch}{0.95}\n"
            f"{tabular}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{table}\n"
        )

    assets = list(moments_by_asset.keys())
    out = {}
    all_horizon_parts = []

    for H in horizons:
        panel_dfs = {}
        combined_parts = []

        for m in moments:
            panel = PANEL_LABELS.get(m, f"Panel: {m}")
            rows = []
            idx = []

            for asset in assets:
                hdict = moments_by_asset.get(asset, {})
                if H not in hdict or not isinstance(hdict[H], dict):
                    continue

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
                        _fmt_stat(stats["jb"], stats["jb_stars"]),
                        _fmt_stat(stats["adf"], stats["adf_stars"]),
                        _fmt_stat(stats["lb"], stats["lb_stars"]),
                    ])

            df = pd.DataFrame(
                rows,
                index=pd.Index(idx, name="Asset"),
                columns=COLS,
            )

            panel_dfs[panel] = df

            if not df.empty:
                combined_parts.append(_make_header_row(panel, COLS))
                combined_parts.append(df)

        combined_df = pd.concat(combined_parts) if combined_parts else pd.DataFrame()

        latex = ""
        tex_path = None

        if not combined_df.empty:
            df_ltx = combined_df.copy()

            numeric_cols = COLS[:6]
            stat_cols = COLS[6:]

            for c in numeric_cols:
                df_ltx[c] = df_ltx[c].map(lambda v: _fmt_num(v) if not isinstance(v, str) else v)

            for c in stat_cols:
                df_ltx[c] = df_ltx[c].map(lambda v: "" if pd.isna(v) else str(v))

            tabular = df_ltx.to_latex(
                escape=False,
                index=True,
                na_rep="",
                caption=None,
                label=None,
            )

            caption = (
                f"Summary statistics for moment series ({H}-day horizon). "
                f"JB is reported in levels. ADF and LB are applied to first differences. "
                f"Stars denote 1\\%, 5\\%, and 10\\% significance."
            )

            latex = _latex_wrap(
                tabular=tabular,
                caption=caption,
                label=f"tab:{file_stub}_{H}d",
            )

            tex_path = save_dir / f"{file_stub}_{H}d.tex"
            tex_path.write_text(latex, encoding="utf-8")

            all_horizon_parts.append(_make_header_row(f"{H}d", COLS))
            all_horizon_parts.append(df_ltx)

        out[H] = {
            "tables": panel_dfs,
            "df": combined_df,
            "latex": latex,
            "tex_path": tex_path,
        }

    all_horizons_df = pd.concat(all_horizon_parts) if all_horizon_parts else pd.DataFrame()
    all_horizons_latex = ""
    all_horizons_tex_path = None

    if not all_horizons_df.empty:
        tabular = all_horizons_df.to_latex(
            escape=False,
            index=True,
            na_rep="",
            caption=None,
            label=None,
        )

        # Insert a midrule before each horizon row except the first one
        for H in horizons[1:]:
            tabular = tabular.replace(f"\n{H}d ", f"\n\\midrule\n{H}d ")

        caption = (
            "Summary statistics for moment series across all horizons. "
            "JB is reported in levels. ADF and LB are applied to first differences. "
            "Stars denote 1\\%, 5\\%, and 10\\% significance."
        )

        all_horizons_latex = _latex_wrap(
            tabular=tabular,
            caption=caption,
            label=f"tab:{file_stub}_all_horizons",
        )

        all_horizons_tex_path = save_dir / f"{file_stub}_all_horizons.tex"
        all_horizons_tex_path.write_text(all_horizons_latex, encoding="utf-8")

    out["all_horizons"] = {
        "df": all_horizons_df,
        "latex": all_horizons_latex,
        "tex_path": all_horizons_tex_path,
    }

    return out



def call_fit_error_timeseries(
    result_dict: Dict[Any, dict],
    *,
    date_col: str = "date",
    metrics: Tuple[str, ...] = ("rmse", "mape"),
    require_success: bool = True,
    plot: bool = True,
    title: Optional[str] = None,
    latex_caption: str = "Calibration error summary statistics (daily call surface fits).",
    latex_label: str = "tab:call_fit_error_summary",
    latex_out: Optional[Union[str, Path]] = None,
    decimals: int = 4,
    include_percent_for: Tuple[str, ...] = ("mape",),
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Build a time series of fitted call price errors from a result_dict and return
    (ts_df, summary_df, latex_str).

    Expected structure per day:
        day = result_dict[date]
        day["errors"] is a dict containing keys like "rmse", "mape"
        optionally day["success"] is bool
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
        for m in metrics:
            v = errs.get(m, np.nan)
            try:
                row[m] = float(v)
            except Exception:
                row[m] = np.nan
        rows.append(row)

    if len(rows) == 0:
        ts_df = pd.DataFrame(columns=[date_col, "success", *metrics])
        summary_df = pd.DataFrame(index=list(metrics))
        return ts_df, summary_df, ""

    ts_df = pd.DataFrame(rows)

    # normalize/parse date column
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
    ts_df = ts_df.sort_values(date_col).reset_index(drop=True)

    # summary stats
    def _summary(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return pd.Series(
                {
                    "count": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
                    "p05": np.nan, "p25": np.nan, "median": np.nan,
                    "p75": np.nan, "p95": np.nan, "max": np.nan
                }
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

    # -------- LaTeX table generation --------
    # We want rows = metric, cols = summary stats
    col_order = ["count", "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max"]
    present_cols = [c for c in col_order if c in summary_df.columns]
    tab = summary_df.copy()

    # Pretty metric names
    def _pretty_metric(m: str) -> str:
        mm = str(m).lower()
        if mm == "rmse":
            return "RMSE"
        if mm == "mape":
            return "MAPE"
        return m

    # numeric formatting
    def _fmt_num(x: Any, *, is_count: bool = False) -> str:
        if x is None:
            return ""
        try:
            if is_count:
                if not np.isfinite(float(x)):
                    return ""
                return f"{int(round(float(x)))}"
            v = float(x)
        except Exception:
            return ""
        if not np.isfinite(v):
            return ""
        return f"{v:.{decimals}f}"

    # if you store MAPE in decimals (e.g., 0.12) and want percent (12.0),
    # we can multiply by 100 in the LaTeX output only
    percent_set = {str(x).lower() for x in include_percent_for}

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(rf"\caption{{{latex_caption}}}")
    lines.append(rf"\label{{{latex_label}}}")

    # column spec: l + one 'r' per stat
    colspec = "l" + "r" * len(present_cols)
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    # header row
    header_map = {
        "count": "N",
        "mean": "Mean",
        "std": "Std",
        "min": "Min",
        "p05": "P5",
        "p25": "P25",
        "median": "Median",
        "p75": "P75",
        "p95": "P95",
        "max": "Max",
    }
    header = ["Metric"] + [header_map.get(c, c) for c in present_cols]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # body
    for m in tab.index:
        mm = str(m)
        pretty = _pretty_metric(mm)

        row_cells = [pretty]
        for c in present_cols:
            x = tab.loc[m, c]
            if c == "count":
                row_cells.append(_fmt_num(x, is_count=True))
            else:
                v = x
                # percent formatting for selected metrics (e.g., MAPE)
                if str(m).lower() in percent_set:
                    try:
                        v = float(v) * 100.0
                    except Exception:
                        v = np.nan
                row_cells.append(_fmt_num(v, is_count=False))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # optional note for percent conversion
    if any(str(m).lower() in percent_set for m in tab.index):
        lines.append(r"\vspace{2pt}")
        lines.append(
            r"\begin{minipage}{0.95\linewidth}\footnotesize "
            r"\emph{Notes:} MAPE is reported in percent. "
            r"\end{minipage}"
        )

    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    # write .tex if requested
    if latex_out is not None:
        latex_out = Path(latex_out)
        latex_out.parent.mkdir(parents=True, exist_ok=True)
        latex_out.write_text(latex_str, encoding="utf-8")

    return ts_df, summary_df, latex_str


def export_physical_rnd_strike_percentiles(
    density_dict,
    output_csv="physical_rnd_strike_percentiles.csv",
    ticker=None,
    date=None,

    # NEW:
    target_maturities=None,
    maturity_tolerance_days=3,
):
    """
    Export strike-level percentiles under both the
    risk-neutral and physical measures to CSV.

    Parameters
    ----------
    target_maturities : list or None
        Example:
            [7, 14, 30]

        Keeps only maturities nearest to these
        target days-to-expiration values.

    maturity_tolerance_days : int
        Maximum allowed mismatch in days.
    """

    def get_key(d, names, required=True):
        for name in names:
            if name in d:
                return d[name]

        if required:
            raise KeyError(f"None of these keys found: {names}")

        return None

    # ============================================================
    # Load arrays
    # ============================================================

    grid_K = np.asarray(
        get_key(density_dict, ["grid_K", "grid_k"]),
        dtype=float
    )

    rnd_K_surface = np.asarray(
        get_key(density_dict, ["rnd_K_surface", "rnd_k_surface"]),
        dtype=float
    )

    physical_K_surface = np.asarray(
        get_key(density_dict, ["physical_K_surface", "physical_k_surface"]),
        dtype=float
    )

    # ============================================================
    # Maturity grid
    # ============================================================

    T_grid = get_key(
        density_dict,
        ["matched_T_grid", "T_grid", "t_grid"],
        required=False
    )

    if T_grid is None:
        T_grid = np.arange(rnd_K_surface.shape[0])

    else:
        T_grid = np.asarray(T_grid, dtype=float)

    # Convert to days
    maturity_days_grid = np.round(T_grid * 365).astype(int)

    # ============================================================
    # Expiration dates
    # ============================================================

    exp_dates = get_key(
        density_dict,
        ["exp_dates", "exp_date", "exdates", "exdate", "expiration_dates"],
        required=False
    )

    if exp_dates is not None:
        exp_dates = pd.to_datetime(exp_dates)

    # ============================================================
    # Valuation date
    # ============================================================

    valuation_date = date

    if valuation_date is None:
        valuation_date = get_key(
            density_dict,
            ["date", "valuation_date", "current_date"],
            required=False
        )

    if valuation_date is not None:
        valuation_date = pd.to_datetime(valuation_date)

    # ============================================================
    # Ticker
    # ============================================================

    if ticker is None:
        ticker = density_dict.get("ticker", None)

    # ============================================================
    # Determine which maturities to keep
    # ============================================================

    maturity_indices = np.arange(len(T_grid))

    if target_maturities is not None:

        selected_indices = []

        for target_day in target_maturities:

            closest_idx = np.argmin(
                np.abs(maturity_days_grid - target_day)
            )

            closest_day = maturity_days_grid[closest_idx]

            if abs(closest_day - target_day) <= maturity_tolerance_days:
                selected_indices.append(closest_idx)

        maturity_indices = np.unique(selected_indices)

    # ============================================================
    # Build rows
    # ============================================================

    rows = []

    for i in maturity_indices:

        T = T_grid[i]

        # Handle 1D vs 2D strike grids
        if grid_K.ndim == 1:
            K = grid_K.copy()

        else:
            K = grid_K[i].copy()

        rnd_pdf = np.asarray(
            rnd_K_surface[i],
            dtype=float
        ).copy()

        physical_pdf = np.asarray(
            physical_K_surface[i],
            dtype=float
        ).copy()

        # --------------------------------------------------------
        # Sort by strike
        # --------------------------------------------------------

        order = np.argsort(K)

        K = K[order]
        rnd_pdf = rnd_pdf[order]
        physical_pdf = physical_pdf[order]

        # --------------------------------------------------------
        # Remove invalid points
        # --------------------------------------------------------

        valid = (
            np.isfinite(K)
            & np.isfinite(rnd_pdf)
            & np.isfinite(physical_pdf)
            & (rnd_pdf >= 0)
            & (physical_pdf >= 0)
        )

        K = K[valid]
        rnd_pdf = rnd_pdf[valid]
        physical_pdf = physical_pdf[valid]

        if len(K) < 2:
            continue

        # --------------------------------------------------------
        # Normalize densities
        # --------------------------------------------------------

        rnd_area = np.trapezoid(rnd_pdf, K)

        physical_area = np.trapezoid(
            physical_pdf,
            K
        )

        rnd_pdf_norm = (
            rnd_pdf / rnd_area
            if rnd_area > 0 else rnd_pdf
        )

        physical_pdf_norm = (
            physical_pdf / physical_area
            if physical_area > 0 else physical_pdf
        )

        # --------------------------------------------------------
        # Compute CDFs
        # --------------------------------------------------------

        dK = np.diff(K)

        rnd_cdf = np.zeros_like(K)

        rnd_cdf[1:] = np.cumsum(
            0.5 * (
                rnd_pdf_norm[:-1]
                + rnd_pdf_norm[1:]
            ) * dK
        )

        physical_cdf = np.zeros_like(K)

        physical_cdf[1:] = np.cumsum(
            0.5 * (
                physical_pdf_norm[:-1]
                + physical_pdf_norm[1:]
            ) * dK
        )

        rnd_cdf = np.clip(rnd_cdf, 0, 1)

        physical_cdf = np.clip(
            physical_cdf,
            0,
            1
        )

        # --------------------------------------------------------
        # Expiration date
        # --------------------------------------------------------

        if (
            exp_dates is not None
            and len(exp_dates) == len(T_grid)
        ):

            exp_date_i = exp_dates[i]

        elif valuation_date is not None:

            exp_date_i = (
                valuation_date
                + pd.to_timedelta(
                    int(round(T * 365)),
                    unit="D"
                )
            )

        else:
            exp_date_i = pd.NaT

        # --------------------------------------------------------
        # Days until expiration
        # --------------------------------------------------------

        if (
            valuation_date is not None
            and pd.notna(exp_date_i)
        ):

            days_until_expiration = int(
                (exp_date_i - valuation_date).days
            )

        else:

            days_until_expiration = int(
                round(T * 365)
            )

        # --------------------------------------------------------
        # Save rows
        # --------------------------------------------------------

        for j in range(len(K)):

            rows.append({

                "ticker": ticker,

                "date": (
                    valuation_date.date()
                    if valuation_date is not None
                    else date
                ),

                "exp_date": (
                    exp_date_i.date()
                    if pd.notna(exp_date_i)
                    else None
                ),

                "days_until_expiration":
                    days_until_expiration,

                "maturity": T,

                "strike": K[j],

                "rnd_density": rnd_pdf[j],

                "physical_density":
                    physical_pdf[j],

                "rnd_percentile_pct":
                    100 * rnd_cdf[j],

                "physical_percentile_pct":
                    100 * physical_cdf[j],
            })

    # ============================================================
    # Create dataframe
    # ============================================================

    df = pd.DataFrame(rows)

    # ============================================================
    # Export CSV
    # ============================================================

    df.to_csv(output_csv, index=False)

    print(f"Saved percentile CSV to: {output_csv}")

    return df

def folder_to_optionmetrics_csv(
    input_folder,
    output_csv,
    ticker="NVDL",
    issuer="",
    index_flag=0,
    exercise_style="A",
    strike_multiplier=1000
):
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    csv_files = sorted(input_folder.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in: {input_folder}")

    all_option_rows = []

    for file in csv_files:
        df = pd.read_csv(file)

        required_cols = [
            "Today's Date",
            "Expire Date",
            "Strike",
            "Call Mark",
            "Put Mark"
        ]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Skipping {file.name}; missing columns: {missing}")
            continue

        trade_date = pd.to_datetime(df["Today's Date"].dropna().iloc[0])
        expire_date = pd.to_datetime(df["Expire Date"].dropna().iloc[0])

        for cp_flag, mark_col in [("C", "Call Mark"), ("P", "Put Mark")]:
            temp = df[["Strike", mark_col]].copy()
            temp = temp.rename(columns={mark_col: "mark"})
            temp = temp.dropna(subset=["Strike", "mark"])

            temp = temp[temp["mark"] > 0]

            temp["date"] = trade_date.strftime("%Y-%m-%d")
            temp["exdate"] = expire_date.strftime("%Y-%m-%d")
            temp["cp_flag"] = cp_flag
            temp["strike_price"] = (temp["Strike"] * strike_multiplier).round().astype(int)

            # Your source files only have mark prices, not bid/ask.
            temp["best_bid"] = temp["mark"]
            temp["best_offer"] = temp["mark"]

            temp["volume"] = 0
            temp["ticker"] = ticker
            temp["index_flag"] = index_flag
            temp["issuer"] = issuer
            temp["exercise_style"] = exercise_style

            temp = temp[
                [
                    "date",
                    "exdate",
                    "cp_flag",
                    "strike_price",
                    "best_bid",
                    "best_offer",
                    "volume",
                    "ticker",
                    "index_flag",
                    "issuer",
                    "exercise_style"
                ]
            ]

            all_option_rows.append(temp)

    if len(all_option_rows) == 0:
        raise ValueError("No usable option files were found.")

    optionmetrics_df = pd.concat(all_option_rows, ignore_index=True)

    optionmetrics_df = optionmetrics_df.sort_values(
        ["date", "exdate", "cp_flag", "strike_price"]
    ).reset_index(drop=True)

    optionmetrics_df.insert(
        7,
        "optionid",
        range(1, len(optionmetrics_df) + 1)
    )

    optionmetrics_df["best_bid"] = optionmetrics_df["best_bid"].round(4)
    optionmetrics_df["best_offer"] = optionmetrics_df["best_offer"].round(4)

    output_csv = Path(output_csv)

    if output_csv.suffix.lower() != ".csv":
        output_csv = output_csv / f"{ticker}_optionmetrics.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    optionmetrics_df.to_csv(output_csv, index=False)

    print(f"Saved OptionMetrics-style file to:")
    print(output_csv)
    print(f"Rows saved: {len(optionmetrics_df)}")

    return optionmetrics_df


# compiled = folder_to_optionmetrics_csv(
#     input_folder=r"C:\Users\beatt\Spyder directory\State Price Density\riskoptima\Riskoptima raw",
#     output_csv=r"C:\Users\beatt\Spyder directory\State Price Density\riskoptima\NVDL_optionmetrics.csv",
#     ticker="NVDL",
#     issuer="GRANITESHARES 2X LONG NVDA DAILY ETF"
# )