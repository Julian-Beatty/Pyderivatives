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
                linewidth=lw_density + 0.3,
                alpha=alpha_density,
                label=qlab,
            )
            model_color = q_line.get_color()
            
            ax.plot(
                R_all, p_plot,
                linewidth=lw_density,
                alpha=0.60,                  # <-- lighter
                linestyle="-",               # same line, but subdued
                label=plab,
                color=model_color,
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
##############################################Multiday surface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _cdf_from_pdf_trapz(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """
    NaN-safe-ish CDF from a PDF on grid x using cumulative trapezoid integration.
    Normalizes to end at 1 when total area > 0.
    """
    x = np.asarray(x, float)
    f = np.asarray(pdf, float)

    n = x.size
    if n < 2:
        return np.zeros_like(x)

    dx = np.diff(x)
    trap = 0.5 * (f[:-1] + f[1:]) * dx

    cdf = np.empty(n, float)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(trap)

    total = cdf[-1]
    if np.isfinite(total) and total > 0:
        cdf /= total
    else:
        cdf[:] = 0.0

    return np.clip(cdf, 0.0, 1.0)


def plot_pricing_kernel_3d_surface_by_T(
    result_dict: dict,
    T_target: float,
    *,
    anchor_key: str = "anchor_surfaces",
    kernel_key: str = "mK_surface",
    fP_key: str = "fP_surface",        # physical density used for truncation
    r_key: str = "r_common",
    T_key: str = "T_anchor",

    # x-axis
    x_mode: str = "log",               # {"log","gross"}
    x_bounds: tuple[float, float] | None = None,

    # thinning
    max_dates: int = 250,
    stride: int | None = None,

    # z scaling
    z_mode: str = "log",               # {"level","log"}
    z_eps: float = 1e-300,

    # appearance
    cmap: str = "viridis",

    # NEW: truncate plotting window by physical tail mass (per date)
    truncate_by_ptails: bool = False,
    ptail_alphas: tuple[float, float] = (0.01, 0.01),  # (alpha_left, alpha_right)

    # output
    title: str | None = None,
    save: str | Path | None = None,
    dpi: int = 200,

    # interactive
    interactive: bool = False,
    interactive_engine: str = "plotly",
    save_html: str | Path | None = None,

    # NEW: allow multi-panel plotting
    ax=None,                           # pass an existing 3D Axes to draw on
    add_colorbar: bool = True,         # set False for multi-panel grids; add one global later
    clear_ax: bool = True,             # clear axes before plotting if reusing
):
    """
    3D surface: x=(log return r) OR x=(gross return exp(r)),
                y=time (date),
                z=pricing kernel at nearest maturity to T_target.

    If truncate_by_ptails=True:
      For EACH date-row, compute physical CDF from fP_surface at that T,
      then keep only x where CDF in [alpha_left, 1-alpha_right].
      Outside that interval, set Z to NaN (ragged edges in the surface).

    Notes for multi-panel figures
    -----------------------------
    - Pass `ax=<existing 3D axes>` to plot into a subplot.
    - For 2x2 grids, set `add_colorbar=False` for each call, then create a single
      shared colorbar using the returned `surf` (see example after function).
    """

    if x_mode not in {"log", "gross"}:
        raise ValueError("x_mode must be 'log' or 'gross'")
    if z_mode not in {"level", "log"}:
        raise ValueError("z_mode must be 'level' or 'log'")
    if truncate_by_ptails:
        aL, aR = float(ptail_alphas[0]), float(ptail_alphas[1])
        if not (0.0 <= aL < 1.0 and 0.0 <= aR < 1.0 and (aL + aR) < 1.0):
            raise ValueError("ptail_alphas must satisfy 0<=aL<1, 0<=aR<1, and aL+aR<1.")

    # --- sort dates ---
    keys = list(result_dict.keys())
    date_ts = pd.to_datetime(keys)
    order = np.argsort(date_ts.values)
    date_ts = date_ts[order]
    keys = [keys[i] for i in order]

    # --- thin dates ---
    n_all = len(keys)
    if stride is not None and stride > 1:
        keep = np.arange(0, n_all, stride, dtype=int)
    else:
        if n_all <= max_dates:
            keep = np.arange(n_all, dtype=int)
        else:
            keep = np.linspace(0, n_all - 1, max_dates).round().astype(int)

    keys = [keys[i] for i in keep]
    date_ts = date_ts[keep]

    # --- build Z(time, r) (and physical density rows if needed) ---
    r_ref = None
    T_used = None
    rows_Z = []
    rows_fP = []
    kept_dates = []

    for dk, dt in zip(keys, date_ts):
        day = result_dict[dk]
        if anchor_key not in day:
            continue
        if (T_key not in day) or (r_key not in day) or (kernel_key not in day[anchor_key]):
            continue
        if truncate_by_ptails and (fP_key not in day[anchor_key]):
            raise KeyError(f"truncate_by_ptails=True but '{fP_key}' missing in day[{anchor_key}].")

        T_grid = np.asarray(day[T_key], float).ravel()
        r_grid = np.asarray(day[r_key], float).ravel()

        Ksurf = np.asarray(day[anchor_key][kernel_key], float)
        if Ksurf.ndim != 2:
            continue

        j = int(np.nanargmin(np.abs(T_grid - float(T_target))))
        z_row = Ksurf[j, :].astype(float, copy=True)

        if r_ref is None:
            r_ref = r_grid
            T_used = float(T_grid[j])
            # CDF needs a monotone grid
            if not np.all(np.diff(r_ref) > 0):
                raise ValueError("r_common must be strictly increasing for CDF-based truncation.")
        else:
            if r_grid.shape != r_ref.shape or np.nanmax(np.abs(r_grid - r_ref)) > 1e-12:
                continue

        rows_Z.append(z_row)
        kept_dates.append(dt)

        if truncate_by_ptails:
            fPsurf = np.asarray(day[anchor_key][fP_key], float)
            if fPsurf.ndim != 2:
                raise ValueError(f"{fP_key} must be 2D (nT, nr).")
            rows_fP.append(fPsurf[j, :].astype(float, copy=True))

    if not rows_Z:
        raise ValueError(
            "No rows extracted. Check kernel_key/paths. "
            f"anchor_key='{anchor_key}', kernel_key='{kernel_key}', r_key='{r_key}', T_key='{T_key}'."
        )

    Z = np.vstack(rows_Z)  # (nt, nr)
    kept_dates = pd.to_datetime(kept_dates)

    # --- truncate each date-row by physical tail mass ---
    if truncate_by_ptails:
        FP = np.vstack(rows_fP)  # (nt, nr)
        aL, aR = float(ptail_alphas[0]), float(ptail_alphas[1])

        for i in range(Z.shape[0]):
            fp = FP[i, :]
            good = np.isfinite(fp) & np.isfinite(r_ref)
            if np.sum(good) < 5:
                Z[i, :] = np.nan
                continue

            rr = r_ref[good]
            ff = np.maximum(fp[good], 0.0)  # clip small negative noise

            cdf = _cdf_from_pdf_trapz(rr, ff)

            # keep rr where cdf in [aL, 1-aR]
            left_pos = int(np.searchsorted(cdf, aL, side="left"))
            right_pos = int(np.searchsorted(cdf, 1.0 - aR, side="right")) - 1

            if right_pos <= left_pos:
                Z[i, :] = np.nan
                continue

            keep_good = np.zeros(rr.size, dtype=bool)
            keep_good[left_pos: right_pos + 1] = True

            keep_full = np.zeros(r_ref.size, dtype=bool)
            keep_full[np.where(good)[0][keep_good]] = True

            Z[i, ~keep_full] = np.nan

    # --- choose x axis ---
    if x_mode == "log":
        x = r_ref.copy()
        xlab = "log return r"
    else:
        x = np.exp(r_ref)
        xlab = "gross return R = exp(r)"

    # --- x_bounds in chosen units ---
    if x_bounds is not None:
        lo, hi = float(x_bounds[0]), float(x_bounds[1])
        m = (x >= lo) & (x <= hi)
        x = x[m]
        Z = Z[:, m]

    # --- z transform ---
    if z_mode == "log":
        Z_plot = np.log(np.maximum(Z, z_eps))
        zlab = f"log({kernel_key})"
    else:
        Z_plot = Z
        zlab = kernel_key

    # --- title ---
    ttl = title if title is not None else (
        f"Pricing Kernel 3D Surface (T_target={T_target:.6f}y, used={T_used:.6f}y, x_mode={x_mode})"
        + (f", ptails={ptail_alphas}" if truncate_by_ptails else "")
    )

    # --- mesh ---
    y_num = mdates.date2num(kept_dates.to_pydatetime())
    X, Y = np.meshgrid(x, y_num)

    # ----------------------------
    # Interactive (Plotly)
    # ----------------------------
    if interactive:
        if interactive_engine != "plotly":
            raise ValueError("interactive_engine currently only supports 'plotly'.")
        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise ImportError("Plotly is required for interactive=True. Install with: pip install plotly") from e

        y_labels = [d.strftime("%Y-%m-%d") for d in kept_dates]

        fig = go.Figure(
            data=go.Surface(
                x=x, y=y_labels, z=Z_plot,
                colorscale=cmap,
                showscale=True
            )
        )
        fig.update_layout(
            title=ttl,
            scene=dict(xaxis_title=xlab, yaxis_title="date", zaxis_title=zlab),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        html_path = None
        if save_html is not None:
            html_path = Path(save_html)
        elif save is not None and str(save).lower().endswith(".html"):
            html_path = Path(save)

        if html_path is not None:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(html_path), include_plotlyjs="cdn")

        return fig

    # ----------------------------
    # Static (Matplotlib) - supports multi-panel
    # ----------------------------
    if ax is None:
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        if clear_ax:
            ax.cla()

    surf = ax.plot_surface(X, Y, Z_plot, linewidth=0, antialiased=True, cmap=cmap)

    ax.set_xlabel(xlab)
    ax.set_ylabel("date")
    ax.set_zlabel(zlab)
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_title(ttl)

    cbar = None
    if add_colorbar:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)

    # Only tighten if we're not in a grid, otherwise the caller can do it once.
    if ax is None:
        fig.tight_layout()

    # IO-friendly save (auto-create folder; auto-add .png)
    if save is not None:
        save_path = Path(save)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, surf, cbar


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _cdf_from_pdf_trapz(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    f = np.asarray(pdf, float)
    n = x.size
    if n < 2:
        return np.zeros_like(x)

    dx = np.diff(x)
    trap = 0.5 * (f[:-1] + f[1:]) * dx

    cdf = np.empty(n, float)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(trap)

    total = cdf[-1]
    if np.isfinite(total) and total > 0:
        cdf /= total
    else:
        cdf[:] = 0.0

    return np.clip(cdf, 0.0, 1.0)


def plot_rnd_3d_surface_by_T(
    result_dict: dict,
    T_target: float,
    *,
    # --- paths/keys (matches your screenshots) ---
    T_key: str = "T_grid",              # maturity grid (nT,)
    r_key: str = "rnd_lr_grid",         # log-return grid for RND (nr,)
    rnd_key: str = "rnd_lr_surface",    # risk-neutral density in log-return space (nT,nr)

    # --- x-axis choice ---
    x_mode: str = "log",                # {"log","gross"} where gross = exp(r)
    x_bounds: tuple[float, float] | None = None,  # bounds in x-units matching x_mode

    # --- thinning across dates ---
    max_dates: int = 250,
    stride: int | None = None,

    # --- z scaling ---
    z_mode: str = "log",                # {"level","log"}
    z_eps: float = 1e-300,

    # --- optional truncation by RND tail mass (per date) ---
    truncate_by_tails: bool = False,
    tail_alphas: tuple[float, float] = (0.01, 0.01),  # keep CDF in [aL, 1-aR]

    # --- appearance/output ---
    cmap: str = "viridis",
    title: str | None = None,
    save: str | Path | None = None,
    dpi: int = 200,

    # --- interactive ---
    interactive: bool = False,
    interactive_engine: str = "plotly",
    save_html: str | Path | None = None,
):
    """
    Multi-day 3D surface of the RISK-NEUTRAL DENSITY at a fixed maturity.

    Inputs expected (per date):
      day[T_key]       -> (nT,)
      day[r_key]       -> (nr,)
      day[rnd_key]     -> (nT, nr)

    Plots:
      x = r (log return) OR exp(r) (gross return)
      y = date
      z = q(r | T) (level or log)

    If truncate_by_tails=True:
      Per-date, compute CDF from q and keep only where CDF in [aL, 1-aR],
      setting the rest to NaN (ragged edges).
    """
    if x_mode not in {"log", "gross"}:
        raise ValueError("x_mode must be 'log' or 'gross'")
    if z_mode not in {"level", "log"}:
        raise ValueError("z_mode must be 'level' or 'log'")
    if truncate_by_tails:
        aL, aR = float(tail_alphas[0]), float(tail_alphas[1])
        if not (0.0 <= aL < 1.0 and 0.0 <= aR < 1.0 and (aL + aR) < 1.0):
            raise ValueError("tail_alphas must satisfy 0<=aL<1, 0<=aR<1, and aL+aR<1.")

    # --- sort dates ---
    keys = list(result_dict.keys())
    date_ts = pd.to_datetime(keys)
    order = np.argsort(date_ts.values)
    date_ts = date_ts[order]
    keys = [keys[i] for i in order]

    # --- thin dates ---
    n_all = len(keys)
    if stride is not None and stride > 1:
        keep = np.arange(0, n_all, stride, dtype=int)
    else:
        if n_all <= max_dates:
            keep = np.arange(n_all, dtype=int)
        else:
            keep = np.linspace(0, n_all - 1, max_dates).round().astype(int)

    keys = [keys[i] for i in keep]
    date_ts = date_ts[keep]

    # --- collect rows ---
    r_ref = None
    T_used = None
    rows = []
    kept_dates = []

    for dk, dt in zip(keys, date_ts):
        day = result_dict[dk]

        if (T_key not in day) or (r_key not in day) or (rnd_key not in day):
            continue

        T_grid = np.asarray(day[T_key], float).ravel()
        r_grid = np.asarray(day[r_key], float).ravel()
        Qsurf = np.asarray(day[rnd_key], float)

        if Qsurf.ndim != 2:
            continue

        j = int(np.nanargmin(np.abs(T_grid - float(T_target))))
        q_row = Qsurf[j, :].astype(float, copy=True)

        if r_ref is None:
            r_ref = r_grid
            T_used = float(T_grid[j])
            if truncate_by_tails and not np.all(np.diff(r_ref) > 0):
                raise ValueError("rnd_lr_grid must be strictly increasing for CDF-based truncation.")
        else:
            # require consistent r-grid across dates
            if r_grid.shape != r_ref.shape or np.nanmax(np.abs(r_grid - r_ref)) > 1e-12:
                continue

        rows.append(q_row)
        kept_dates.append(dt)

    if not rows:
        raise ValueError(
            "No rows extracted. Check keys or data availability. "
            f"Expected day['{T_key}'], day['{r_key}'], day['{rnd_key}']."
        )

    Z = np.vstack(rows)  # (nt, nr)
    kept_dates = pd.to_datetime(kept_dates)

    # --- optional per-row truncation by tail mass of q ---
    if truncate_by_tails:
        aL, aR = float(tail_alphas[0]), float(tail_alphas[1])
        for i in range(Z.shape[0]):
            q = Z[i, :]
            good = np.isfinite(q) & np.isfinite(r_ref)
            if np.sum(good) < 5:
                Z[i, :] = np.nan
                continue

            rr = r_ref[good]
            qq = np.maximum(q[good], 0.0)  # clip tiny negatives

            cdf = _cdf_from_pdf_trapz(rr, qq)
            left_pos = int(np.searchsorted(cdf, aL, side="left"))
            right_pos = int(np.searchsorted(cdf, 1.0 - aR, side="right")) - 1

            if right_pos <= left_pos:
                Z[i, :] = np.nan
                continue

            keep_good = np.zeros(rr.size, dtype=bool)
            keep_good[left_pos : right_pos + 1] = True

            keep_full = np.zeros(r_ref.size, dtype=bool)
            keep_full[np.where(good)[0][keep_good]] = True

            Z[i, ~keep_full] = np.nan

    # --- choose x axis ---
    if x_mode == "log":
        x = r_ref.copy()
        xlab = "log return r"
    else:
        x = np.exp(r_ref)
        xlab = "gross return R = exp(r)"

    # --- x bounds in chosen units ---
    if x_bounds is not None:
        lo, hi = float(x_bounds[0]), float(x_bounds[1])
        m = (x >= lo) & (x <= hi)
        x = x[m]
        Z = Z[:, m]

    # --- z transform ---
    if z_mode == "log":
        Z_plot = np.log(np.maximum(Z, z_eps))
        zlab = f"log({rnd_key})"
    else:
        Z_plot = Z
        zlab = rnd_key

    # --- title ---
    ttl = title if title is not None else (
        f"Risk-Neutral Density 3D Surface (T_target={T_target:.6f}y, used={T_used:.6f}y, x_mode={x_mode})"
        + (f", tails={tail_alphas}" if truncate_by_tails else "")
    )

    # --- mesh ---
    y_num = mdates.date2num(kept_dates.to_pydatetime())
    X, Y = np.meshgrid(x, y_num)

    # ----------------------------
    # Interactive (Plotly)
    # ----------------------------
    if interactive:
        if interactive_engine != "plotly":
            raise ValueError("interactive_engine currently only supports 'plotly'.")
        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise ImportError("Plotly is required for interactive=True. Install with: pip install plotly") from e

        y_labels = [d.strftime("%Y-%m-%d") for d in kept_dates]

        fig = go.Figure(
            data=go.Surface(
                x=x,
                y=y_labels,
                z=Z_plot,
                colorscale=cmap,  # e.g. "Viridis", "Inferno"
                showscale=True,
            )
        )
        fig.update_layout(
            title=ttl,
            scene=dict(
                xaxis_title=xlab,
                yaxis_title="date",
                zaxis_title=zlab,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        html_path = None
        if save_html is not None:
            html_path = Path(save_html)
        elif save is not None and str(save).lower().endswith(".html"):
            html_path = Path(save)

        if html_path is not None:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(html_path), include_plotlyjs="cdn")

        return fig, None  # (fig, ax) style, ax=None for interactive

    # ----------------------------
    # Static (Matplotlib)
    # ----------------------------
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z_plot, linewidth=0, antialiased=True, cmap=cmap)

    ax.set_xlabel(xlab)
    ax.set_ylabel("date")
    ax.set_zlabel(zlab)
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_title(ttl)

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
    fig.tight_layout()

    if save is not None:
        save_path = Path(save)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _cdf_from_pdf_trapz(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    f = np.asarray(pdf, float)
    n = x.size
    if n < 2:
        return np.zeros_like(x)

    dx = np.diff(x)
    trap = 0.5 * (f[:-1] + f[1:]) * dx

    cdf = np.empty(n, float)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(trap)

    total = cdf[-1]
    if np.isfinite(total) and total > 0:
        cdf /= total
    else:
        cdf[:] = 0.0

    return np.clip(cdf, 0.0, 1.0)


def plot_physical_density_3d_surface_by_T(
    result_dict: dict,
    T_target: float,
    *,
    anchor_key: str = "anchor_surfaces",
    fP_key: str = "fP_surface",        # physical density surface (nT, nr)
    r_key: str = "r_common",
    T_key: str = "T_anchor",

    # x-axis
    x_mode: str = "log",               # {"log","gross"}
    x_bounds: tuple[float, float] | None = None,

    # thinning
    max_dates: int = 250,
    stride: int | None = None,

    # z scaling
    z_mode: str = "log",               # {"level","log"}
    z_eps: float = 1e-300,

    # optional truncation by physical tails (uses the density itself)
    truncate_by_tails: bool = False,
    tail_alphas: tuple[float, float] = (0.01, 0.01),  # keep CDF in [aL, 1-aR]

    # appearance
    cmap: str = "viridis",

    # output
    title: str | None = None,
    save: str | Path | None = None,
    dpi: int = 200,

    # interactive
    interactive: bool = False,
    interactive_engine: str = "plotly",
    save_html: str | Path | None = None,
):
    """
    3D surface for PHYSICAL density at nearest maturity to T_target:

      x = log return r  OR  gross return exp(r)
      y = time (date)
      z = fP(r | T)  (level or log)

    If truncate_by_tails=True:
      Per-date, compute CDF from fP and keep only where CDF in [aL, 1-aR],
      setting the rest to NaN (ragged edges).
    """
    if x_mode not in {"log", "gross"}:
        raise ValueError("x_mode must be 'log' or 'gross'")
    if z_mode not in {"level", "log"}:
        raise ValueError("z_mode must be 'level' or 'log'")
    if truncate_by_tails:
        aL, aR = float(tail_alphas[0]), float(tail_alphas[1])
        if not (0.0 <= aL < 1.0 and 0.0 <= aR < 1.0 and (aL + aR) < 1.0):
            raise ValueError("tail_alphas must satisfy 0<=aL<1, 0<=aR<1, and aL+aR<1.")

    # --- sort dates ---
    keys = list(result_dict.keys())
    date_ts = pd.to_datetime(keys)
    order = np.argsort(date_ts.values)
    date_ts = date_ts[order]
    keys = [keys[i] for i in order]

    # --- thin dates ---
    n_all = len(keys)
    if stride is not None and stride > 1:
        keep = np.arange(0, n_all, stride, dtype=int)
    else:
        if n_all <= max_dates:
            keep = np.arange(n_all, dtype=int)
        else:
            keep = np.linspace(0, n_all - 1, max_dates).round().astype(int)

    keys = [keys[i] for i in keep]
    date_ts = date_ts[keep]

    # --- collect rows ---
    r_ref = None
    T_used = None
    rows = []
    kept_dates = []

    for dk, dt in zip(keys, date_ts):
        day = result_dict[dk]
        if anchor_key not in day:
            continue
        if (T_key not in day) or (r_key not in day) or (fP_key not in day[anchor_key]):
            continue

        T_grid = np.asarray(day[T_key], float).ravel()
        r_grid = np.asarray(day[r_key], float).ravel()
        fPsurf = np.asarray(day[anchor_key][fP_key], float)
        if fPsurf.ndim != 2:
            continue

        j = int(np.nanargmin(np.abs(T_grid - float(T_target))))
        fp = fPsurf[j, :].astype(float, copy=True)

        if r_ref is None:
            r_ref = r_grid
            T_used = float(T_grid[j])
            if truncate_by_tails and not np.all(np.diff(r_ref) > 0):
                raise ValueError("r_common must be strictly increasing for CDF-based truncation.")
        else:
            if r_grid.shape != r_ref.shape or np.nanmax(np.abs(r_grid - r_ref)) > 1e-12:
                continue

        rows.append(fp)
        kept_dates.append(dt)

    if not rows:
        raise ValueError(
            "No rows extracted. Check fP_key/paths. "
            f"anchor_key='{anchor_key}', fP_key='{fP_key}', r_key='{r_key}', T_key='{T_key}'."
        )

    Z = np.vstack(rows)  # (nt, nr)
    kept_dates = pd.to_datetime(kept_dates)

    # --- optional truncation by tails (per row) ---
    if truncate_by_tails:
        aL, aR = float(tail_alphas[0]), float(tail_alphas[1])
        for i in range(Z.shape[0]):
            fp = Z[i, :]
            good = np.isfinite(fp) & np.isfinite(r_ref)
            if np.sum(good) < 5:
                Z[i, :] = np.nan
                continue

            rr = r_ref[good]
            ff = np.maximum(fp[good], 0.0)
            cdf = _cdf_from_pdf_trapz(rr, ff)

            left_pos = int(np.searchsorted(cdf, aL, side="left"))
            right_pos = int(np.searchsorted(cdf, 1.0 - aR, side="right")) - 1
            if right_pos <= left_pos:
                Z[i, :] = np.nan
                continue

            keep_good = np.zeros(rr.size, dtype=bool)
            keep_good[left_pos : right_pos + 1] = True

            keep_full = np.zeros(r_ref.size, dtype=bool)
            keep_full[np.where(good)[0][keep_good]] = True
            Z[i, ~keep_full] = np.nan

    # --- choose x axis ---
    if x_mode == "log":
        x = r_ref.copy()
        xlab = "log return r"
    else:
        x = np.exp(r_ref)
        xlab = "gross return R = exp(r)"

    # --- apply x_bounds ---
    if x_bounds is not None:
        lo, hi = float(x_bounds[0]), float(x_bounds[1])
        m = (x >= lo) & (x <= hi)
        x = x[m]
        Z = Z[:, m]

    # --- z transform ---
    if z_mode == "log":
        Z_plot = np.log(np.maximum(Z, z_eps))
        zlab = f"log({fP_key})"
    else:
        Z_plot = Z
        zlab = fP_key

    # --- title ---
    ttl = title if title is not None else (
        f"Physical Density 3D Surface (T_target={T_target:.6f}y, used={T_used:.6f}y, x_mode={x_mode})"
        + (f", tails={tail_alphas}" if truncate_by_tails else "")
    )

    # --- mesh ---
    y_num = mdates.date2num(kept_dates.to_pydatetime())
    X, Y = np.meshgrid(x, y_num)

    # ----------------------------
    # Interactive (Plotly)
    # ----------------------------
    if interactive:
        if interactive_engine != "plotly":
            raise ValueError("interactive_engine currently only supports 'plotly'.")
        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise ImportError("Plotly is required for interactive=True. Install with: pip install plotly") from e

        y_labels = [d.strftime("%Y-%m-%d") for d in kept_dates]

        fig = go.Figure(
            data=go.Surface(
                x=x, y=y_labels, z=Z_plot,
                colorscale=cmap,
                showscale=True
            )
        )
        fig.update_layout(
            title=ttl,
            scene=dict(xaxis_title=xlab, yaxis_title="date", zaxis_title=zlab),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        html_path = None
        if save_html is not None:
            html_path = Path(save_html)
        elif save is not None and str(save).lower().endswith(".html"):
            html_path = Path(save)

        if html_path is not None:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(html_path), include_plotlyjs="cdn")

        return fig

    # ----------------------------
    # Static (Matplotlib)
    # ----------------------------
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z_plot, linewidth=0, antialiased=True, cmap=cmap)

    ax.set_xlabel(xlab)
    ax.set_ylabel("date")
    ax.set_zlabel(zlab)
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_title(ttl)

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
    fig.tight_layout()

    if save is not None:
        save_path = Path(save)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax