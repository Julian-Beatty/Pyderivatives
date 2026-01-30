import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from typing import Optional, Sequence, Dict, Tuple, Union, Any


def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
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
    taus: Tuple[float, ...],
    B: int,
    bootstrap: str,
    block_len: int,
    seed: Optional[int],
    fit_kwargs: Optional[dict],
):
    fit_kwargs = fit_kwargs or {}
    rng = np.random.default_rng(seed)
    n, p = X.shape

    params, se_boot, results, boot_params = {}, {}, {}, {}

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


def run_asym_quantreg_with_controls(
    *,
    r_df: pd.DataFrame,
    var_s: pd.Series,
    skew_s: pd.Series,
    kurt_s: pd.Series,
    # optional controls (date-keyed)
    controls_df: Optional[pd.DataFrame] = None,       # must contain date_col + controls
    controls_cols: Union[str, Sequence[str]] = "all", # "all" or list of columns in controls_df
    controls_diff: bool = False,                      # if True: add d_<control> and its lags instead of levels
    n_controls_lags: int = 0,                          # lags for controls (levels or diffs)
    # base settings
    date_col: str = "date",
    ret_col: str = "ret_30",
    horizon_label: str = "30d",
    n_ret_lags: int = 2,
    n_mom_lags: int = 2,
    taus: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    add_const: bool = True,
    # bootstrap
    bootstrap: str = "block",
    B: int = 1000,
    block_len: int = 10,
    seed: Optional[int] = 123,
    fit_kwargs: Optional[dict] = None,
    dropna: bool = True,
    # advanced: allow custom X for each equation (overrides defaults if provided)
    X_A_extra: Optional[Sequence[str]] = None,
    X_B_extra: Optional[Sequence[str]] = None,
    X_C_extra: Optional[Sequence[str]] = None,
) -> dict:
    """
    Same spirit as your function, but adds optional date-keyed controls.

    controls_df:
        DataFrame with [date_col, <controls...>]. Example: covid deaths, stringency index, etc.
    controls_cols:
        "all" or list of control column names from controls_df.
    controls_diff:
        If True, uses first-differences of controls (and optional lags).
        If False, uses levels (and optional lags).

    X_*_extra:
        additional regressors to append to each equation design (after core terms).
        Use this if you want some controls in A but not B, etc.
    """
    # --- normalize inputs ---
    r = r_df[[date_col, ret_col]].copy()
    r[date_col] = pd.to_datetime(r[date_col])

    def _series_to_df(s: pd.Series, name: str) -> pd.DataFrame:
        ss = s.copy()
        ss.name = name
        if not isinstance(ss.index, pd.DatetimeIndex):
            ss.index = pd.to_datetime(ss.index)
        return ss.to_frame().reset_index().rename(columns={"index": date_col})

    var_df = _series_to_df(var_s, "var")
    skew_df = _series_to_df(skew_s, "skew")
    kurt_df = _series_to_df(kurt_s, "kurt")

    df = (
        r.merge(var_df, on=date_col, how="inner")
         .merge(skew_df, on=date_col, how="inner")
         .merge(kurt_df, on=date_col, how="inner")
         .sort_values(date_col)
         .reset_index(drop=True)
    )

    # --- merge controls (optional) ---
    control_names: list[str] = []
    if controls_df is not None:
        cdf = controls_df.copy()
        cdf[date_col] = pd.to_datetime(cdf[date_col])

        if controls_cols == "all":
            control_names = [c for c in cdf.columns if c != date_col]
        else:
            control_names = list(controls_cols)

        keep = [date_col] + control_names
        df = df.merge(cdf[keep], on=date_col, how="left")

    # --- moments diffs ---
    df["d_var"] = df["var"].diff()
    df["d_skew"] = df["skew"].diff()
    df["d_kurt"] = df["kurt"].diff()

    # --- asym returns ---
    df["ret_pos"] = df[ret_col].clip(lower=0.0)
    df["ret_neg"] = df[ret_col].clip(upper=0.0)

    for L in range(1, n_ret_lags + 1):
        df[f"ret_pos_L{L}"] = df["ret_pos"].shift(L)
        df[f"ret_neg_L{L}"] = df["ret_neg"].shift(L)

    for L in range(1, n_mom_lags + 1):
        df[f"d_var_L{L}"] = df["d_var"].shift(L)
        df[f"d_skew_L{L}"] = df["d_skew"].shift(L)
        df[f"d_kurt_L{L}"] = df["d_kurt"].shift(L)

    # --- controls transformations (optional) ---
    controls_terms: list[str] = []
    if control_names:
        if controls_diff:
            for c in control_names:
                dc = f"d_{c}"
                df[dc] = df[c].astype(float).diff()
                controls_terms.append(dc)
                for L in range(1, n_controls_lags + 1):
                    nm = f"{dc}_L{L}"
                    df[nm] = df[dc].shift(L)
                    controls_terms.append(nm)
        else:
            # levels + optional lags
            for c in control_names:
                controls_terms.append(c)
                for L in range(1, n_controls_lags + 1):
                    nm = f"{c}_L{L}"
                    df[nm] = df[c].shift(L)
                    controls_terms.append(nm)

    df_model = df.dropna().copy() if dropna else df.copy()

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

        # normal approx p-values (two-sided)
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
                "controls": control_names,
                "controls_diff": bool(controls_diff),
                "n_controls_lags": int(n_controls_lags),
                "ret_col": ret_col,
            },
        }

    # --- core regressors ---
    ret_terms = (
        ["ret_pos", "ret_neg"]
        + [f"ret_pos_L{L}" for L in range(1, n_ret_lags + 1)]
        + [f"ret_neg_L{L}" for L in range(1, n_ret_lags + 1)]
    )

    mom_terms_all = (
        [f"d_var_L{L}" for L in range(1, n_mom_lags + 1)]
        + [f"d_skew_L{L}" for L in range(1, n_mom_lags + 1)]
        + [f"d_kurt_L{L}" for L in range(1, n_mom_lags + 1)]
    )

    # paper-style defaults + your controls_terms
    X_A = ret_terms + mom_terms_all + controls_terms + (list(X_A_extra) if X_A_extra else [])
    X_B = ret_terms + (
        [f"d_skew_L{L}" for L in range(1, n_mom_lags + 1)]
        + [f"d_var_L{L}" for L in range(1, n_mom_lags + 1)]
        + [f"d_kurt_L{L}" for L in range(1, n_mom_lags + 1)]
    ) + controls_terms + (list(X_B_extra) if X_B_extra else [])
    X_C = ret_terms + (
        [f"d_kurt_L{L}" for L in range(1, n_mom_lags + 1)]
        + [f"d_var_L{L}" for L in range(1, n_mom_lags + 1)]
        + [f"d_skew_L{L}" for L in range(1, n_mom_lags + 1)]
    ) + controls_terms + (list(X_C_extra) if X_C_extra else [])

    out = {
        "data": df_model,
        "A_var": _run_eq("d_var", X_A),
        "B_skew": _run_eq("d_skew", X_B),
        "C_kurt": _run_eq("d_kurt", X_C),
        "controls_terms": controls_terms,
    }
    return out
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_qrm_across_quantiles_selectcoef(
    res_by_key,
    *,
    eq_key: str = "A_var",
    coefs=("ret_pos", "ret_neg"),          # <- choose any 1 or 2 coefficients (or more)
    keys_order=None,
    ci: float | None = 0.95,
    scale: float = 1.0,
    title: str | None = None,
    figsize=(10, 8),
    legend: bool = True,
    show_ols: bool = True,
    ols_hac_lags: int | None = None,
):
    """
    Generalized Fig-3 style:
      - x-axis: quantile τ
      - one line per horizon/label (res_by_key key)
      - one panel per coefficient in `coefs` (if len(coefs)=1 => 1 panel; else stacked)

    OLS:
      - dashed horizontal line per (label, coef, panel) using same X used in quantile reg
        stored in res[eq_key]["X_cols"].
    """
    # normalize input
    items = list(res_by_key.items()) if isinstance(res_by_key, dict) else list(res_by_key)
    if keys_order is not None:
        order_map = {k: i for i, k in enumerate(keys_order)}
        items.sort(key=lambda kv: order_map.get(kv[0], 10**9))

    # normalize coefs to list
    if isinstance(coefs, str):
        coefs = [coefs]
    else:
        coefs = list(coefs)
    n_panels = len(coefs)

    # common taus across results
    tau_sets = []
    for _, res in items:
        taus = np.array(res[eq_key]["params"].index, dtype=float)
        tau_sets.append(set(np.round(taus, 10)))
    common = sorted(set.intersection(*tau_sets))
    taus_common = np.array(common, dtype=float)

    # z for CI
    z = None
    if ci is not None:
        try:
            from scipy.stats import norm
            z = float(norm.ppf(0.5 + ci / 2.0))
        except Exception:
            z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96

    # helper: OLS slope for a given coef
    def _ols_slope_for_coef(res: dict, coef_name: str) -> float:
        df = res["data"]
        dep = res[eq_key]["dep"]
        X_cols = list(res[eq_key]["X_cols"])  # includes const if used

        has_const = "const" in X_cols
        X_cols_wo = [c for c in X_cols if c != "const"]

        X = df[X_cols_wo].astype(float)
        if has_const:
            X = sm.add_constant(X, has_constant="add")

        y = df[dep].astype(float).to_numpy(float)

        model = sm.OLS(y, X.to_numpy(float))
        if ols_hac_lags is None:
            fit = model.fit()
        else:
            fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(ols_hac_lags)})

        cols = list(X.columns)
        if coef_name not in cols:
            raise KeyError(f"coef '{coef_name}' not in OLS design columns: {cols}")
        return float(fit.params[cols.index(coef_name)])

    # figure / axes
    fig, axes = plt.subplots(n_panels, 1, figsize=(figsize[0], figsize[1] if n_panels > 1 else figsize[1]*0.6), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, coef_name in zip(axes, coefs):
        for label, res in items:
            eq = res[eq_key]
            params = eq["params"].copy()
            se = eq.get("se_boot", None)

            if coef_name not in params.columns:
                raise KeyError(f"'{coef_name}' not in res['{eq_key}']['params'].columns. Available: {list(params.columns)}")

            # align
            params = params.loc[taus_common]
            beta = params[coef_name].to_numpy(float) * scale

            ax.plot(taus_common, beta, marker="o", linewidth=2, label=str(label))

            # CI shading
            if (ci is not None) and (se is not None) and (coef_name in se.columns):
                se_al = se.loc[taus_common]
                se_vec = se_al[coef_name].to_numpy(float) * scale
                ax.fill_between(taus_common, beta - z * se_vec, beta + z * se_vec, alpha=0.15)

            # OLS line
            if show_ols:
                b_ols = _ols_slope_for_coef(res, coef_name) * scale
                ax.axhline(b_ols, linestyle="--", linewidth=1.5, alpha=0.9)

        ax.set_ylabel(f"{coef_name} β(τ)" + (" (scaled)" if scale != 1.0 else ""))
        ax.grid(True, alpha=0.25)

    axes[0].set_title(title or f"{eq_key}: coefficient curves across quantiles")
    axes[-1].set_xlabel("Quantile (τ)")

    if legend:
        axes[0].legend(loc="best", title="Frequency / Horizon")

    fig.tight_layout()
    return fig, axes
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_qrm_by_quantile_across_frequencies_selectcoef(
    res_by_key,
    *,
    eq_key: str = "A_var",
    coefs=("ret_pos", "ret_neg"),          # <- choose any list of coefs to plot in each panel
    taus_to_plot=(0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95),
    keys_order=None,
    ncols: int = 2,
    ci: float | None = None,              # error bars via +/- z*SE
    scale: float = 1.0,
    title: str | None = None,
    figsize_per_panel=(5.4, 3.2),
    legend: bool = True,
    show_ols: bool = True,
    ols_hac_lags: int | None = None,
):
    """
    Generalized Fig-4 style small multiples:
      - each subplot: a quantile τ
      - x-axis: horizon/frequency label (keys of res_by_key)
      - within each subplot: one line per coefficient in `coefs`

    OLS:
      - dashed horizontal line per coef (OLS slope), constant across horizons (computed per res)
        We compute OLS for each label separately; if you prefer a *single* OLS pooled across
        horizons, say so and I’ll adjust.
    """
    items = list(res_by_key.items()) if isinstance(res_by_key, dict) else list(res_by_key)
    if keys_order is not None:
        order_map = {k: i for i, k in enumerate(keys_order)}
        items.sort(key=lambda kv: order_map.get(kv[0], 10**9))
    labels = [str(k) for k, _ in items]

    if isinstance(coefs, str):
        coefs = [coefs]
    else:
        coefs = list(coefs)

    # nearest tau helper
    def _nearest_tau(taus_arr, t):
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

    # OLS slope helper (per res/label)
    def _ols_slope_for_coef(res: dict, coef_name: str) -> float:
        df = res["data"]
        dep = res[eq_key]["dep"]
        X_cols = list(res[eq_key]["X_cols"])

        has_const = "const" in X_cols
        X_cols_wo = [c for c in X_cols if c != "const"]

        X = df[X_cols_wo].astype(float)
        if has_const:
            X = sm.add_constant(X, has_constant="add")

        y = df[dep].astype(float).to_numpy(float)

        model = sm.OLS(y, X.to_numpy(float))
        if ols_hac_lags is None:
            fit = model.fit()
        else:
            fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(ols_hac_lags)})

        cols = list(X.columns)
        if coef_name not in cols:
            raise KeyError(f"coef '{coef_name}' not in OLS design columns: {cols}")
        return float(fit.params[cols.index(coef_name)])

    taus_panels = list(taus_to_plot)
    n_panels = len(taus_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # pre-extract objects for speed
    extracted = []
    for label, res in items:
        eq = res[eq_key]
        taus_av = np.array(eq["params"].index, dtype=float)
        params = eq["params"]
        se = eq.get("se_boot", None)
        extracted.append((str(label), res, taus_av, params, se))

    for i, t_req in enumerate(taus_panels):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        x = np.arange(len(labels), dtype=float)

        for coef_name in coefs:
            y = np.zeros(len(labels), float)
            e = np.zeros(len(labels), float) if (ci is not None) else None

            for j, (lab, res, taus_av, params, se) in enumerate(extracted):
                t_use = _nearest_tau(taus_av, t_req)

                if coef_name not in params.columns:
                    raise KeyError(
                        f"'{coef_name}' not in params columns for label={lab}. "
                        f"Available: {list(params.columns)}"
                    )

                y[j] = float(params.loc[t_use, coef_name]) * scale

                if ci is not None and se is not None and coef_name in se.columns:
                    e[j] = float(se.loc[t_use, coef_name]) * scale

            ax.plot(x, y, marker="o", linewidth=2, label=coef_name)

            if ci is not None:
                ax.errorbar(x, y, yerr=z * e, fmt="none", capsize=3, alpha=0.8)

            if show_ols:
                # show OLS per-label as dashed points connected (since each horizon has its own OLS fit)
                ols_vals = np.array([_ols_slope_for_coef(res, coef_name) for _, res, *_ in extracted], dtype=float) * scale
                ax.plot(x, ols_vals, linestyle="--", linewidth=1.5, alpha=0.9)

        ax.set_title(f"{eq_key}: τ≈{t_req:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("Response" + (" (%)" if scale == 100.0 else ""))
        ax.grid(True, alpha=0.25)
        if legend:
            ax.legend(loc="best")

    # hide unused axes
    for k in range(n_panels, nrows * ncols):
        rr = k // ncols
        cc = k % ncols
        axes[rr, cc].axis("off")

    if title is not None:
        fig.suptitle(title, y=0.995)

    fig.tight_layout()
    return fig, axes
