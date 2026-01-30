import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Sequence, Optional, Union, Tuple, Any
import pandas as pd
def plot_result_dict(
    plot_dict: Dict[str, dict],
    maturity_list: Sequence[Union[int, float]],
    date: Union[str, "np.datetime64", "object"],
    *,
    title: Optional[str] = None,
    panel_shape: Optional[Tuple[int, int]] = None,   # e.g. (2,3); default auto-ish
    figsize_per_panel: Tuple[float, float] = (4.0, 2.8),
    sharex: bool = True,
    sharey: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    legend: bool = True,
    legend_loc: str = "best",
    lw: float = 1.8,
    alpha: float = 0.95,
    grid: bool = True,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot a panel of log-return risk-neutral densities (RND) for a given date,
    overlaying the curves from each result_dict in plot_dict.

    Assumed per-model structure:
        day = result_dict[date]
        day["rnd_lr_surface"] : array shape (nT, nX)
        day["T_grid"]         : array shape (nT,)
        day["rnd_lr_grid"]    : array shape (nX,)

    maturity_list:
        - If values are ints (or int-like), treated as maturity indices into T_grid.
        - Otherwise treated as maturities (in same units as T_grid) and matched to nearest.

    Returns
    -------
    fig, axes
    """

    # ---- helpers ----
    def _date_key(res_dict: dict, date_in: Any):
        # allow passing date as str like "2021-06-01"
        if isinstance(date_in, str) and date_in in res_dict:
            return date_in
        # try pandas-ish Timestamp stringification without importing pandas
        s = str(date_in)
        if s in res_dict:
            return s
        # common case: "YYYY-MM-DD 00:00:00" -> "YYYY-MM-DD"
        s2 = s[:10]
        if s2 in res_dict:
            return s2
        raise KeyError(f"Date '{date_in}' not found in result_dict keys (tried '{s}' and '{s2}').")

    def _pick_T_index(T_grid: np.ndarray, m):
        # treat ints as indices if in range
        if isinstance(m, (int, np.integer)):
            j = int(m)
            if j < 0 or j >= len(T_grid):
                raise IndexError(f"maturity index {j} out of range for T_grid length {len(T_grid)}.")
            return j, float(T_grid[j])
        # if float is very close to int and within range, still interpret as index
        if isinstance(m, (float, np.floating)) and float(m).is_integer():
            j = int(m)
            if 0 <= j < len(T_grid):
                return j, float(T_grid[j])
        # otherwise nearest maturity value
        mval = float(m)
        j = int(np.argmin(np.abs(T_grid - mval)))
        return j, float(T_grid[j])

    # ---- extract one model to size panels / choose panel layout ----
    if len(plot_dict) == 0:
        raise ValueError("plot_dict is empty.")

    first_label = next(iter(plot_dict))
    first_res = plot_dict[first_label]
    date_key = _date_key(first_res, date)

    first_day = first_res[date_key]
    for k in ("rnd_lr_surface", "T_grid", "rnd_lr_grid"):
        if k not in first_day:
            raise KeyError(f"Missing '{k}' in plot_dict['{first_label}'][{date_key}].")

    T_grid0 = np.asarray(first_day["T_grid"], float)
    x_grid0 = np.asarray(first_day["rnd_lr_grid"], float)

    # ---- resolve maturity indices (using first model's T_grid as reference) ----
    mats = []
    for m in maturity_list:
        j, Tj = _pick_T_index(T_grid0, m)
        mats.append((j, Tj))

    n_panels = len(mats)

    # ---- panel shape ----
    if panel_shape is None:
        # simple auto: up to 3 columns
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        panel_shape = (nrows, ncols)

    nrows, ncols = panel_shape
    if nrows * ncols < n_panels:
        raise ValueError(f"panel_shape {panel_shape} too small for {n_panels} panels.")

    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.asarray(axes).reshape(nrows, ncols)

    # ---- plotting ----
    for p, (j, Tj) in enumerate(mats):
        ax = axes.flat[p]

        for model_label, res_dict in plot_dict.items():
            dkey = _date_key(res_dict, date)
            day = res_dict[dkey]

            # pull arrays
            T_grid = np.asarray(day["T_grid"], float)
            x_grid = np.asarray(day["rnd_lr_grid"], float)
            surf = np.asarray(day["rnd_lr_surface"], float)

            # if T grids differ slightly, remap by nearest T (safer than assuming same index)
            jj = int(np.argmin(np.abs(T_grid - Tj)))

            y = surf[jj, :]

            # handle mismatch in x grids by requiring same length; (better interpolation optional)
            if x_grid.shape != x_grid0.shape or not np.allclose(x_grid, x_grid0, atol=1e-12, rtol=1e-9):
                # fallback: interpolate to reference x_grid0
                y = np.interp(x_grid0, x_grid, y)
                x = x_grid0
            else:
                x = x_grid

            ax.plot(x, y, lw=lw, alpha=alpha, label=model_label)

        ax.set_title(f"T ≈ {Tj:.6g} (panel idx {p})")
        if grid:
            ax.grid(True, alpha=0.25)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # turn off unused axes
    for p in range(n_panels, nrows * ncols):
        axes.flat[p].axis("off")

    # labels
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel("RND (log-return)")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("log-return")

    if title is None:
        title = f"Log-return RND overlays on {str(date)[:10]}"
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # one legend (global)
    if legend:
        # grab handles/labels from first active axis
        handles, labels = None, None
        for ax in axes.flat:
            if ax.has_data():
                handles, labels = ax.get_legend_handles_labels()
                break
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)

            # make room for legend
            fig.subplots_adjust(bottom=0.08 + 0.03 * (len(labels) > 4))

    # save
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    return fig, axes


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


def P_Q_K_multipanel_multi(
    out_dict: Dict[str, dict],
    *,
    title: Optional[str] = None,
    n_panels: Optional[int] = None,                 # if None uses panel_shape product
    panel_shape: Tuple[int, int] = (2, 4),
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    # ----- truncation controls -----
    truncate: bool = True,
    ptail_alpha: Tuple[float, float] = (0.10, 0.0), # (alpha_left, alpha_right) for p-CDF tails
    trunc_mode: str = "cdf",                        # {"cdf","rbounds","none","cdf+rbounds"}
    r_bounds: Optional[Tuple[float, float]] = None,
    clip_trunc_to_support: bool = True,
    # ----- kernel axis controls -----
    kernel_linestyle: str = "--",
    kernel_yscale: str = "linear",                  # {"linear","log"}
    kernel_log_eps: float = 1e-300,
    # ----- display controls -----
    legend_loc: str = "upper center",
    lw_density: float = 1.6,
    lw_kernel: float = 1.4,
    alpha_density: float = 0.95,
    alpha_kernel: float = 0.90,
    # labeling style
    show_model_in_label: bool = True,               # labels like "q_R (ModelA)"
):
    """
    Multi-panel plot: q_R(R), p_R(R) and pricing kernel M(R) with dual y-axis,
    OVERLAYED across multiple out dictionaries.

    out_dict: {"label": out, ...}

    Assumed per-out structure:
      out["anchor_surfaces"]["qR_surface"], ["pR_surface"], ["M_surface"] with shape (nT, nR)
      out["T_anchor"] : (nT,)
      out["R_common"] : (nR,)
    """

    if not isinstance(out_dict, dict) or len(out_dict) == 0:
        raise ValueError("out_dict must be a non-empty dict like {'Model A': outA, ...}.")

    # -------------------------
    # parse truncation settings
    # -------------------------
    mode = str(trunc_mode).lower().strip()
    if not truncate:
        mode = "none"

    valid = {"none", "cdf", "rbounds", "cdf+rbounds"}
    if mode not in valid:
        raise ValueError(f"trunc_mode must be one of {valid}.")

    use_cdf = mode in {"cdf", "cdf+rbounds"}
    use_rbounds = mode in {"rbounds", "cdf+rbounds"}

    aL, aR = float(ptail_alpha[0]), float(ptail_alpha[1])
    if use_cdf:
        if not (0.0 <= aL < 1.0 and 0.0 <= aR < 1.0):
            raise ValueError("ptail_alpha must be in [0,1) for both tails.")
        if not (aL + aR < 1.0):
            raise ValueError("Require ptail_alpha[0] + ptail_alpha[1] < 1.")

    kernel_yscale = str(kernel_yscale).lower().strip()
    if kernel_yscale not in {"linear", "log"}:
        raise ValueError("kernel_yscale must be one of {'linear','log'}.")
    kernel_log_eps = float(kernel_log_eps)
    if kernel_yscale == "log" and not (kernel_log_eps > 0):
        raise ValueError("kernel_log_eps must be > 0 for log scale.")

    # -------------------------
    # reference grids from first model
    # -------------------------
    first_label = next(iter(out_dict))
    out0 = out_dict[first_label]
    if out0 is None or "anchor_surfaces" not in out0:
        raise KeyError(f"out_dict['{first_label}'] must contain out['anchor_surfaces'].")

    anchor0 = out0["anchor_surfaces"]
    T_ref = np.asarray(out0.get("T_anchor", []), float).ravel()
    R_ref = np.asarray(out0.get("R_common", []), float).ravel()

    if T_ref.size == 0 or R_ref.size == 0:
        raise ValueError("Missing T_anchor or R_common in the first out dict.")
    if R_ref.size >= 2 and np.any(np.diff(R_ref) <= 0):
        raise ValueError("R_common must be strictly increasing (reference model).")

    # rbounds range (if enabled) in reference coordinates
    R_min = R_max = None
    if use_rbounds:
        if r_bounds is None or len(r_bounds) != 2:
            raise ValueError("For rbounds truncation, provide r_bounds=(R_min, R_max).")
        R_min, R_max = float(r_bounds[0]), float(r_bounds[1])
        if clip_trunc_to_support:
            R_min = max(R_min, float(R_ref[0]))
            R_max = min(R_max, float(R_ref[-1]))
        if not (np.isfinite(R_min) and np.isfinite(R_max) and R_max > R_min):
            raise ValueError("Invalid r_bounds after clipping.")

    # -------------------------
    # choose maturities for panels
    # -------------------------
    nrows, ncols = panel_shape
    nT = T_ref.size
    idx_pool = np.arange(nT)

    n_pan = (nrows * ncols) if n_panels is None else int(n_panels)
    n_pan = max(1, min(n_pan, idx_pool.size))
    idxs = idx_pool[np.linspace(0, idx_pool.size - 1, n_pan, dtype=int)]

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.4 * ncols, 3.8 * nrows),
        sharex=True,
        constrained_layout=False
    )
    axes = np.array(axes).reshape(-1)
    ax2_list = []

    # -------------------------
    # helpers
    # -------------------------
    def _interp_to_ref(R_src, y_src):
        """Interpolate y(R_src) onto R_ref. Assumes R_src increasing."""
        return np.interp(R_ref, R_src, y_src)

    def _pcdf_cutoffs(R_src, p_src):
        """
        Compute p-CDF cutoffs on the model's full support (R_src),
        return (R_left, R_right) corresponding to alpha_left / 1-alpha_right.
        """
        pj = np.maximum(np.asarray(p_src, float), 0.0)
        dR = np.diff(R_src)
        inc = 0.5 * (pj[1:] + pj[:-1]) * dR
        cdf = np.empty_like(R_src)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(inc)
        total = float(cdf[-1])
        if not (total > 0 and np.isfinite(total)):
            return None

        cdf /= total

        # left cutoff
        if aL > 0:
            idxL = np.where(cdf >= aL)[0]
            iL = int(idxL[0]) if idxL.size else 0
        else:
            iL = 0

        # right cutoff
        if aR > 0:
            idxR = np.where(cdf <= (1.0 - aR))[0]
            iR = int(idxR[-1]) if idxR.size else (R_src.size - 1)
        else:
            iR = R_src.size - 1

        if iR <= iL:
            return None
        return float(R_src[iL]), float(R_src[iR])

    # -------------------------
    # main plot loop
    # -------------------------
    for k, j in enumerate(idxs):
        ax = axes[k]
        ax2 = ax.twinx()
        ax2_list.append(ax2)

        # base mask on reference grid (rbounds applies to everything)
        mask_all = np.isfinite(R_ref)
        if use_rbounds:
            mask_all &= (R_ref >= R_min) & (R_ref <= R_max)

        R_all = R_ref[mask_all]

        for model_label, out in out_dict.items():
            if out is None or "anchor_surfaces" not in out:
                continue

            anchor = out["anchor_surfaces"]
            T = np.asarray(out.get("T_anchor", []), float).ravel()
            R = np.asarray(out.get("R_common", []), float).ravel()
            qR = np.asarray(anchor.get("qR_surface", []), float)
            pR = np.asarray(anchor.get("pR_surface", []), float)
            M  = np.asarray(anchor.get("M_surface", []), float)

            if T.size == 0 or R.size == 0:
                continue
            if qR.shape != (T.size, R.size) or pR.shape != (T.size, R.size) or M.shape != (T.size, R.size):
                continue

            # match maturity by nearest T to reference T_ref[j]
            jj = int(np.argmin(np.abs(T - T_ref[j])))

            qj = qR[jj, :]
            pj = pR[jj, :]
            Mj = M[jj, :]

            # interpolate to reference R grid if needed
            if R.shape != R_ref.shape or (R.size > 1 and not np.allclose(R, R_ref, atol=1e-12, rtol=1e-9)):
                qj_ref = _interp_to_ref(R, qj)
                pj_ref = _interp_to_ref(R, pj)
                Mj_ref = _interp_to_ref(R, Mj)
            else:
                qj_ref, pj_ref, Mj_ref = qj, pj, Mj

            # apply rbounds mask to q/p/M for plotting on left axis and for kernel base
            q_plot = np.asarray(qj_ref, float)[mask_all]
            p_plot = np.asarray(pj_ref, float)[mask_all]
            M_plot = np.asarray(Mj_ref, float)[mask_all].copy()

            qlab = "q_R(R)"
            plab = "p_R(R)"
            mlab = "M(R)"
            if show_model_in_label:
                qlab += f" ({model_label})"
                plab += f" ({model_label})"
                mlab += f" ({model_label})"

            # --- plot densities and capture the model color (use q_R color as the model color) ---
            (q_line,) = ax.plot(
                R_all, q_plot,
                linewidth=lw_density,
                alpha=alpha_density,
                label=qlab,
            )
            model_color = q_line.get_color()
            
            ax.plot(
                R_all, p_plot,
                linewidth=lw_density,
                alpha=alpha_density,
                label=plab,
                color=model_color,   # keep p_R same color as q_R (optional but usually nicer)
            )
            
            # ---- kernel truncation by p-CDF tails on FULL model support ----
            R_k = R_all
            M_k = M_plot
            
            if use_cdf:
                cut = _pcdf_cutoffs(R, pR[jj, :])  # model-native grid for tail cutoffs
                if cut is None:
                    R_k = np.array([], float)
                    M_k = np.array([], float)
                else:
                    R_left, R_right = cut
                    keep = (R_k >= R_left) & (R_k <= R_right)
                    R_k = R_k[keep]
                    M_k = M_k[keep]
                    if show_model_in_label:
                        mlab = f"M(R) ({model_label}, p-tails ≥ {aL:.0%}/{aR:.0%})"
                    else:
                        mlab = f"M(R) (p-tails ≥ {aL:.0%}/{aR:.0%})"
            
            # log scale cleanup
            if kernel_yscale == "log" and M_k.size > 0:
                pos = np.isfinite(M_k) & (M_k > 0) & np.isfinite(R_k)
                R_k = np.asarray(R_k, float)[pos]
                M_k = np.asarray(M_k, float)[pos]
                M_k = np.maximum(M_k, kernel_log_eps)
            
            if R_k.size > 0:
                ax2.plot(
                    R_k, M_k,
                    label=mlab + ("" if kernel_yscale == "linear" else " (log y)"),
                    linestyle=kernel_linestyle,     # dotted/dashed as you like
                    linewidth=lw_kernel,
                    alpha=alpha_kernel,
                    color=model_color,              # <-- matches the model density color
                )


        # panel title from reference maturity
        T_days = float(T_ref[j] * 365.0)
        ax.set_title(f"T≈{T_days:.1f}d", fontsize=11)

        if (k % ncols) == 0:
            ax.set_ylabel("Density (R-space)")
        if k >= (n_pan - ncols):
            ax.set_xlabel("Gross return R")
        if (k % ncols) == (ncols - 1):
            ax2.set_ylabel("Pricing kernel M(R)")

        ax.grid(True, alpha=0.25)
        ax2.set_yscale(kernel_yscale)

        if use_rbounds:
            ax.set_xlim(R_min, R_max)

    # turn off unused axes
    for k in range(n_pan, axes.size):
        axes[k].axis("off")

    # legend from first active axes
    handles, labels = [], []
    for ax in [axes[0], ax2_list[0]]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    if title is None:
        title = "q_R vs p_R with Pricing Kernel (multi-model overlay)"

    fig.suptitle(title, y=0.995, fontsize=14)
    fig.legend(
        handles, labels,
        loc=legend_loc,
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.4
    )
    fig.subplots_adjust(top=0.88, wspace=0.28, hspace=0.30)

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")

    plt.show()
    return fig