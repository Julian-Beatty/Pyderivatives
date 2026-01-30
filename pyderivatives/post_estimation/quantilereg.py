import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from typing import Dict, Sequence, Optional, Union, Tuple, Any


def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap indices of length n."""
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    if block_len == 1:
        return rng.integers(0, n, size=n)

    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx = []
    for s in starts:
        idx.extend([(s + k) % n for k in range(block_len)])
    return np.asarray(idx[:n], dtype=int)


def _quantreg_fit_bootstrap(
    y: np.ndarray,
    X: np.ndarray,
    *,
    taus: tuple[float, ...],
    B: int,
    bootstrap: str,
    block_len: int,
    seed: int | None,
    fit_kwargs: dict | None,
):
    """Internal: fit QuantReg for each tau + bootstrap SEs."""
    fit_kwargs = fit_kwargs or {}
    rng = np.random.default_rng(seed)
    n, p = X.shape

    colnames = getattr(X, "columns", None)

    params = {}
    se_boot = {}
    results = {}
    boot_params = {}

    for tau in taus:
        res = QuantReg(y, X).fit(q=tau, **fit_kwargs)
        results[tau] = res
        params[tau] = res.params

        b = np.zeros((B, p), float)
        for j in range(B):
            if bootstrap == "iid":
                idx = rng.integers(0, n, size=n)
            elif bootstrap == "block":
                idx = _block_bootstrap_indices(n, block_len=block_len, rng=rng)
            else:
                raise ValueError("bootstrap must be one of {'iid','block'}")

            yb = y[idx]
            Xb = X[idx, :]

            ok = False
            for _ in range(3):
                try:
                    rb = QuantReg(yb, Xb).fit(q=tau, **fit_kwargs)
                    b[j, :] = rb.params
                    ok = True
                    break
                except Exception:
                    if bootstrap == "iid":
                        idx = rng.integers(0, n, size=n)
                    else:
                        idx = _block_bootstrap_indices(n, block_len=block_len, rng=rng)
                    yb = y[idx]
                    Xb = X[idx, :]
            if not ok:
                b[j, :] = np.nan

        boot_params[tau] = b
        se_boot[tau] = np.nanstd(b, axis=0, ddof=1)

    return params, se_boot, results, boot_params


def run_asym_moment_quantile_regressions(
    *,
    r_df: pd.DataFrame,          # must have ['date','ret_30'] (or whatever ret_col is)
    var_s: pd.Series,            # Series named 'var' indexed by date OR with DatetimeIndex
    skew_s: pd.Series,           # Series named 'skew'
    kurt_s: pd.Series,           # Series named 'kurt'
    date_col: str = "date",
    ret_col: str = "ret_30",
    horizon_label: str = "30d",  # just for naming outputs
    n_ret_lags: int = 2,
    n_mom_lags: int = 2,
    taus=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    add_const: bool = True,
    # bootstrap controls
    bootstrap: str = "block",    # {"iid","block"}
    B: int = 1000,
    block_len: int = 10,
    seed: int | None = 123,
    fit_kwargs: dict | None = None,
    dropna: bool = True,
) -> dict:
    """
    Implements paper-style regressions (A) Var, (B) Skew, (C) Kurt using:
      - Koenker–Bassett linear quantile regression (statsmodels QuantReg)
      - Bootstrap standard errors
      - Separate regressions per tau

    Data requirements
    -----------------
    r_df: DataFrame with columns [date_col, ret_col] (backward-looking horizon returns)
    var_s, skew_s, kurt_s: Series indexed by date (DatetimeIndex) with names 'var','skew','kurt'
                           OR pass series with any name; they'll be renamed internally.

    Returns
    -------
    dict with keys:
      - "data": modeling DataFrame (merged + features)
      - "A_var": regression output dict
      - "B_skew": regression output dict
      - "C_kurt": regression output dict

    Each regression output dict contains:
      - params: DataFrame index=taus cols=regressors
      - se_boot: DataFrame
      - t_boot: DataFrame
      - p_boot_norm: DataFrame (normal approx)
      - results: dict[tau] -> statsmodels fit
      - boot_params: dict[tau] -> (B,p) bootstrap draws
      - X_cols: list of regressor names used
    """
    # --- normalize inputs ---
    r = r_df[[date_col, ret_col]].copy()
    r[date_col] = pd.to_datetime(r[date_col])

    def _series_to_df(s: pd.Series, name: str) -> pd.DataFrame:
        ss = s.copy()
        ss.name = name
        if not isinstance(ss.index, pd.DatetimeIndex):
            # allow series with date index stored as strings
            ss.index = pd.to_datetime(ss.index)
        return ss.to_frame().reset_index().rename(columns={"index": date_col})

    var_df = _series_to_df(var_s, "var")
    skew_df = _series_to_df(skew_s, "skew")
    kurt_df = _series_to_df(kurt_s, "kurt")

    # --- merge on date ---
    df = r.merge(var_df, on=date_col, how="inner") \
          .merge(skew_df, on=date_col, how="inner") \
          .merge(kurt_df, on=date_col, how="inner") \
          .sort_values(date_col) \
          .reset_index(drop=True)

    # --- first differences of moments ---
    df["d_var"] = df["var"].diff()
    df["d_skew"] = df["skew"].diff()
    df["d_kurt"] = df["kurt"].diff()

    # --- split returns into + / - parts (paper asymmetry) ---
    df["ret_pos"] = df[ret_col].clip(lower=0.0)
    df["ret_neg"] = df[ret_col].clip(upper=0.0)

    # --- return lags (include L=0 and 1..n_ret_lags) ---
    for L in range(1, n_ret_lags + 1):
        df[f"ret_pos_L{L}"] = df["ret_pos"].shift(L)
        df[f"ret_neg_L{L}"] = df["ret_neg"].shift(L)

    # --- moment-diff lags (usually 1..n_mom_lags) ---
    for L in range(1, n_mom_lags + 1):
        df[f"d_var_L{L}"] = df["d_var"].shift(L)
        df[f"d_skew_L{L}"] = df["d_skew"].shift(L)
        df[f"d_kurt_L{L}"] = df["d_kurt"].shift(L)

    if dropna:
        df_model = df.dropna().copy()
    else:
        df_model = df.copy()

    # --- helper to run one equation ---
    def _run_eq(dep: str, X_cols: list[str]) -> dict:
        y = df_model[dep].to_numpy(float)
        X = df_model[X_cols].astype(float)
        if add_const:
            X = sm.add_constant(X, has_constant="add")
        Xmat = X.to_numpy(float)
        colnames = list(X.columns)

        taus_t = tuple(float(t) for t in taus)
        params_dict, se_dict, results, boot_params = _quantreg_fit_bootstrap(
            y=y,
            X=Xmat,
            taus=taus_t,
            B=B,
            bootstrap=bootstrap,
            block_len=block_len,
            seed=seed,
            fit_kwargs=fit_kwargs,
        )

        params = pd.DataFrame([params_dict[t] for t in taus_t], index=taus_t, columns=colnames)
        se_boot = pd.DataFrame([se_dict[t] for t in taus_t], index=taus_t, columns=colnames)
        t_boot = params / se_boot

        # normal-approx p-values (two-sided); matches common applied reporting
        from math import erf, sqrt
        def pval_from_t(tt):
            Phi = 0.5 * (1.0 + erf(abs(float(tt)) / sqrt(2.0)))
            return 2.0 * (1.0 - Phi)

        p_boot_norm = t_boot.applymap(pval_from_t)

        return {
            "dep": dep,
            "X_cols": colnames,
            "params": params,
            "se_boot": se_boot,
            "t_boot": t_boot,
            "p_boot_norm": p_boot_norm,
            "results": results,
            "boot_params": boot_params,
            "meta": {
                "taus": taus_t,
                "bootstrap": bootstrap,
                "B": int(B),
                "block_len": int(block_len),
                "n_used": int(len(df_model)),
                "horizon_label": horizon_label,
                "n_ret_lags": int(n_ret_lags),
                "n_mom_lags": int(n_mom_lags),
                "ret_col": ret_col,
            },
        }

    # --- build regressor lists for (A), (B), (C) ---
    ret_terms = ["ret_pos", "ret_neg"] + \
                [f"ret_pos_L{L}" for L in range(1, n_ret_lags + 1)] + \
                [f"ret_neg_L{L}" for L in range(1, n_ret_lags + 1)]

    # (A) ΔVar_t
    X_A = ret_terms + \
          [f"d_var_L{L}" for L in range(1, n_mom_lags + 1)] + \
          [f"d_skew_L{L}" for L in range(1, n_mom_lags + 1)] + \
          [f"d_kurt_L{L}" for L in range(1, n_mom_lags + 1)]

    # (B) ΔSkew_t
    X_B = ret_terms + \
          [f"d_skew_L{L}" for L in range(1, n_mom_lags + 1)] + \
          [f"d_var_L{L}" for L in range(1, n_mom_lags + 1)] + \
          [f"d_kurt_L{L}" for L in range(1, n_mom_lags + 1)]

    # (C) ΔKurt_t
    X_C = ret_terms + \
          [f"d_kurt_L{L}" for L in range(1, n_mom_lags + 1)] + \
          [f"d_var_L{L}" for L in range(1, n_mom_lags + 1)] + \
          [f"d_skew_L{L}" for L in range(1, n_mom_lags + 1)]

    out = {
        "data": df_model,
        "A_var": _run_eq("d_var", X_A),
        "B_skew": _run_eq("d_skew", X_B),
        "C_kurt": _run_eq("d_kurt", X_C),
    }
    return out

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm


def plot_asym_moment_quantile_coeffs(
    out: dict,
    *,
    # which coefficient(s) to plot
    coef: str | list[str] = "ret_pos",
    # which equations to plot
    eq_keys: tuple[str, ...] = ("A_var", "B_skew", "C_kurt"),
    # plotting options
    ci: float | None = 0.95,          # None -> no CI; else e.g. 0.95
    use_boot_se: bool = True,         # use out[eq]["se_boot"] for CI
    show_ols: bool = True,            # add mean-regression horizontal line
    ols_hac_lags: int | None = None,  # if not None, use HAC(Newey-West) SEs for OLS line fit (slope unchanged)
    title_prefix: str = "",
    figsize: tuple[float, float] = (9.5, 5.0),
    panel: bool = False,              # True -> 3 stacked panels in one figure
    legend: bool = True,
) -> dict:
    """
    Plot quantile regression coefficient curves β(τ) for each moment equation from
    run_asym_moment_quantile_regressions output.

    Parameters
    ----------
    out : dict
        Output dict from run_asym_moment_quantile_regressions().
    coef : str or list[str]
        Regressor name(s) to plot (e.g. "ret_pos", "ret_neg", ["ret_pos","ret_neg"]).
        Must match column names in out[eq]["params"].
    eq_keys : tuple[str,...]
        Which equations to plot. Defaults to all three: ("A_var","B_skew","C_kurt").
    ci : float or None
        Confidence level for bands (e.g., 0.95). If None, no CI.
    use_boot_se : bool
        Use bootstrap SEs stored in out[eq]["se_boot"]. (This matches your estimator setup.)
    show_ols : bool
        Fit an OLS mean regression with same regressors and plot its slope as a horizontal line.
    ols_hac_lags : int or None
        If provided, uses HAC covariance for OLS fit (slope line same; mostly for reporting).
    panel : bool
        If True, returns a single figure with stacked panels (one per equation).
        If False, returns separate figures per equation.

    Returns
    -------
    dict with keys:
        - "fig" if panel=True else "figs" dict[eq_key] -> fig
        - "axes" similarly
    """
    coefs = [coef] if isinstance(coef, str) else list(coef)

    # mapping for nicer titles
    pretty = {
        "A_var":  "ΔVar",
        "B_skew": "ΔSkew",
        "C_kurt": "ΔKurt",
    }

    # z-value for normal approx CI (you used normal approx p-values already)
    if ci is not None:
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + ci / 2.0))
    else:
        z = None
    def _ols_line(eq_key: str, coef_name: str) -> float:
        """OLS slope for the same dependent and regressor set used in quantile reg."""
        df = out["data"]
        dep = out[eq_key]["dep"]
        X_cols = list(out[eq_key]["X_cols"])  # includes 'const' if you used add_const=True
    
        # remove const if it's in X_cols (it won't be a real column in df)
        has_const = ("const" in X_cols)
        if has_const:
            X_cols_wo = [c for c in X_cols if c != "const"]
        else:
            X_cols_wo = X_cols
    
        # build design matrix from df, then add constant the same way as in _run_eq
        X = df[X_cols_wo].astype(float).copy()
        if has_const:
            X = sm.add_constant(X, has_constant="add")
    
        y = df[dep].astype(float)
    
        model = sm.OLS(y.to_numpy(float), X.to_numpy(float))
        if ols_hac_lags is None:
            res = model.fit()
        else:
            res = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(ols_hac_lags)})
    
        # coef_name might be in columns as-is
        cols = list(X.columns)
        if coef_name not in cols:
            raise KeyError(f"coef '{coef_name}' not found in OLS X columns for {eq_key}. Columns: {cols}")
        j = cols.index(coef_name)
        return float(res.params[j])

    # ---- plotting ----
    if panel:
        fig, axes = plt.subplots(len(eq_keys), 1, figsize=(figsize[0], figsize[1] * len(eq_keys)), sharex=True)
        if len(eq_keys) == 1:
            axes = [axes]
        figs = {"fig": fig}
        axes_out = {"axes": dict(zip(eq_keys, axes))}
    else:
        figs = {}
        axes_out = {}

    for i, eq_key in enumerate(eq_keys):
        eq = out[eq_key]
        taus = np.array(eq["params"].index, dtype=float)

        if panel:
            ax = axes[i]
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            figs[eq_key] = fig
            axes_out[eq_key] = ax

        for coef_name in coefs:
            if coef_name not in eq["params"].columns:
                raise KeyError(
                    f"coef '{coef_name}' not in out['{eq_key}']['params'].columns. "
                    f"Available: {list(eq['params'].columns)}"
                )

            beta = eq["params"][coef_name].to_numpy(float)

            ax.plot(taus, beta, marker="o", linewidth=2, label=f"QRM: {coef_name}")

            # CI band
            if (ci is not None) and use_boot_se:
                se = eq["se_boot"][coef_name].to_numpy(float)
                lo = beta - z * se
                hi = beta + z * se
                ax.fill_between(taus, lo, hi, alpha=0.15)

            # OLS horizontal line
            if show_ols:
                b_ols = _ols_line(eq_key, coef_name)
                ax.axhline(b_ols, linestyle="--", linewidth=1.5, label=f"MRM (OLS): {coef_name}")

        ax.set_ylabel("Coefficient (β)")
        title = f"{title_prefix}{pretty.get(eq_key, eq_key)}: quantile slopes β(τ)"
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

        if legend:
            ax.legend(loc="best")

    if panel:
        axes[-1].set_xlabel("Quantile (τ)")
        fig.tight_layout()
        return {"fig": fig, "axes": axes_out["axes"]}

    else:
        # add x-label to each
        for eq_key in eq_keys:
            axes_out[eq_key].set_xlabel("Quantile (τ)")
            figs[eq_key].tight_layout()
        return {"figs": figs, "axes": axes_out}

import numpy as np
import matplotlib.pyplot as plt


def plot_qrm_across_quantiles(
    res_by_key,
    *,
    eq_key: str = "A_var",                 # "A_var", "B_skew", "C_kurt"
    coef_pos: str = "ret_pos",
    coef_neg: str = "ret_neg",
    keys_order=None,                       # optional explicit order for legend/lines
    ci: float | None = 0.95,               # None => no CI shading
    scale: float = 1.0,                    # use 100.0 if you want "Responses (%)"
    title: str | None = None,
    figsize=(10, 8),
    legend: bool = True,
):
    """
    Fig-3 style: For each moment equation, plot β(τ) across quantiles with one line per
    frequency/horizon. Two panels: positive-return coefficient and negative-return coefficient.

    Parameters
    ----------
    res_by_key : dict or list
        dict[label -> res_dict] OR list of (label, res_dict).
        Each res_dict is the output from run_asym_moment_quantile_regressions.
    eq_key : str
        "A_var", "B_skew", or "C_kurt".
    coef_pos / coef_neg : str
        Column names in res[eq_key]["params"] to plot.
    ci : float or None
        If not None, uses bootstrap SEs res[eq_key]["se_boot"] to shade +/- z*SE.
    scale : float
        Multiply coefficients/CI by this (100.0 => percent units).
    """
    # normalize input to list[(label,res)]
    if isinstance(res_by_key, dict):
        items = list(res_by_key.items())
    else:
        items = list(res_by_key)

    if keys_order is not None:
        order_map = {k: i for i, k in enumerate(keys_order)}
        items.sort(key=lambda kv: order_map.get(kv[0], 10**9))

    # find common taus across all results
    tau_sets = []
    for _, res in items:
        taus = np.array(res[eq_key]["params"].index, dtype=float)
        tau_sets.append(set(np.round(taus, 10)))
    common = sorted(set.intersection(*tau_sets))
    taus_common = np.array(common, dtype=float)

    # z for CI
    z = None
    if ci is not None:
        from math import erf, sqrt
        # approximate z via inverse error function is annoying; simplest: hardcode common 95%
        # but keep generic with scipy if available; else fallback
        try:
            from scipy.stats import norm
            z = float(norm.ppf(0.5 + ci / 2.0))
        except Exception:
            # fallback: 95% approx
            z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax_pos, ax_neg = axes

    for label, res in items:
        eq = res[eq_key]
        params = eq["params"].copy()
        se = eq.get("se_boot", None)

        # align to common taus
        params = params.loc[taus_common]
        bpos = params[coef_pos].to_numpy(float) * scale
        bneg = params[coef_neg].to_numpy(float) * scale

        ax_pos.plot(taus_common, bpos, marker="o", linewidth=2, label=str(label))
        ax_neg.plot(taus_common, bneg, marker="o", linewidth=2, label=str(label))

        if (ci is not None) and (se is not None) and (coef_pos in se.columns) and (coef_neg in se.columns):
            se_al = se.loc[taus_common]
            sepos = se_al[coef_pos].to_numpy(float) * scale
            seneg = se_al[coef_neg].to_numpy(float) * scale

            ax_pos.fill_between(taus_common, bpos - z * sepos, bpos + z * sepos, alpha=0.15)
            ax_neg.fill_between(taus_common, bneg - z * seneg, bneg + z * seneg, alpha=0.15)

    ax_pos.set_title(title or f"{eq_key}: coefficient curves across quantiles")
    ax_pos.set_ylabel(f"{coef_pos} β(τ)" + (" (scaled)" if scale != 1.0 else ""))
    ax_neg.set_ylabel(f"{coef_neg} β(τ)" + (" (scaled)" if scale != 1.0 else ""))
    ax_neg.set_xlabel("Quantile (τ)")

    ax_pos.grid(True, alpha=0.25)
    ax_neg.grid(True, alpha=0.25)

    if legend:
        ax_pos.legend(loc="best", title="Frequency / Horizon")
        ax_neg.legend(loc="best", title="Frequency / Horizon")

    fig.tight_layout()
    return fig, axes


def plot_qrm_by_quantile_across_frequencies(
    res_by_key,
    *,
    eq_key: str = "A_var",
    coef_pos: str = "ret_pos",
    coef_neg: str = "ret_neg",
    taus_to_plot=(0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95),
    keys_order=None,                       # x-axis order
    ncols: int = 2,
    ci: float | None = None,               # optional CI bars (uses +/- z*SE)
    scale: float = 1.0,                    # use 100.0 for percent units
    title: str | None = None,
    figsize_per_panel=(5.4, 3.2),
    legend: bool = True,
):
    """
    Fig-4 style: small multiples. Each subplot corresponds to a quantile τ.
    Within each subplot, x-axis is frequency/horizon, and you plot two lines:
    β_pos(τ) and β_neg(τ).

    Parameters
    ----------
    res_by_key : dict or list
        dict[label -> res_dict] OR list of (label, res_dict).
    taus_to_plot : iterable
        Quantiles to show as small-multiple panels.
    ci : float or None
        If provided, draws vertical error bars +/- z*SE for each point (bootstrap SE).
    """
    # normalize input
    if isinstance(res_by_key, dict):
        items = list(res_by_key.items())
    else:
        items = list(res_by_key)

    if keys_order is not None:
        order_map = {k: i for i, k in enumerate(keys_order)}
        items.sort(key=lambda kv: order_map.get(kv[0], 10**9))

    labels = [str(k) for k, _ in items]

    # choose available taus closest to requested (per result); enforce common set by rounding
    # We'll map each requested tau -> nearest tau in each result (within tolerance)
    def _nearest_tau_index(taus_arr, t):
        taus_arr = np.array(taus_arr, dtype=float)
        j = int(np.argmin(np.abs(taus_arr - float(t))))
        return float(taus_arr[j])

    # z for CI
    z = None
    if ci is not None:
        try:
            from scipy.stats import norm
            z = float(norm.ppf(0.5 + ci / 2.0))
        except Exception:
            z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96

    taus_panels = list(taus_to_plot)
    n_panels = len(taus_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # pre-extract for speed
    extracted = []
    for label, res in items:
        eq = res[eq_key]
        taus_av = np.array(eq["params"].index, dtype=float)
        params = eq["params"]
        se = eq.get("se_boot", None)
        extracted.append((str(label), taus_av, params, se))

    for i, t_req in enumerate(taus_panels):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        # x positions are 0..K-1, labeled by frequency/horizon label
        x = np.arange(len(labels), dtype=float)

        y_pos = np.zeros(len(labels), float)
        y_neg = np.zeros(len(labels), float)
        e_pos = np.zeros(len(labels), float) if (ci is not None) else None
        e_neg = np.zeros(len(labels), float) if (ci is not None) else None

        for j, (lab, taus_av, params, se) in enumerate(extracted):
            t_use = _nearest_tau_index(taus_av, t_req)

            y_pos[j] = float(params.loc[t_use, coef_pos]) * scale
            y_neg[j] = float(params.loc[t_use, coef_neg]) * scale

            if ci is not None and se is not None:
                e_pos[j] = float(se.loc[t_use, coef_pos]) * scale
                e_neg[j] = float(se.loc[t_use, coef_neg]) * scale

        # lines
        ax.plot(x, y_pos, marker="o", linewidth=2, label=f"{coef_pos}")
        ax.plot(x, y_neg, marker="s", linewidth=2, label=f"{coef_neg}")

        # optional error bars
        if ci is not None:
            ax.errorbar(x, y_pos, yerr=z * e_pos, fmt="none", capsize=3, alpha=0.8)
            ax.errorbar(x, y_neg, yerr=z * e_neg, fmt="none", capsize=3, alpha=0.8)

        ax.set_title(f"{eq_key}: τ≈{t_req:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("Response" + (" (%)" if scale == 100.0 else ""))
        ax.grid(True, alpha=0.25)

        if legend:
            ax.legend(loc="best")

    # hide unused axes
    for k in range(n_panels, nrows * ncols):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")

    if title is not None:
        fig.suptitle(title, y=0.995)

    fig.tight_layout()
    return fig, axes

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