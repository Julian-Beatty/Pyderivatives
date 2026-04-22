import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Sequence, Optional, Union, Tuple, Any
import pandas as pd
from typing import Literal
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

    # ----- x-axis controls -----
    x_axis: str = "R",                             # {"R","logR"}

    # ----- truncation controls -----
    truncate: bool = True,
    ptail_alpha: Tuple[float, float] = (0.10, 0.0), # (alpha_left, alpha_right) for p-CDF tails
    trunc_mode: str = "cdf",                        # {"cdf","xbounds","none","cdf+xbounds"}
    x_bounds: Optional[Tuple[float, float]] = None, # bounds in displayed x-axis coordinates
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
    Multi-panel plot: q and p densities plus pricing kernel M with dual y-axis,
    overlaid across multiple out dictionaries.

    Parameters
    ----------
    out_dict : dict
        Dictionary like {"label": out, ...}

    Required per-out structure
    --------------------------
    out["anchor_surfaces"]["qR_surface"], ["pR_surface"], ["M_surface"] with shape (nT, nR)
    out["T_anchor"] : (nT,)
    out["R_common"] : (nR,)

    x_axis
    ------
    "R"    : plot in gross return space
    "logR" : plot in log-return space r = log(R)

    Notes
    -----
    If x_axis="logR", densities are transformed by the Jacobian:
        q_r(r) = q_R(R) * R
        p_r(r) = p_R(R) * R
    where R = exp(r). The pricing kernel M is not Jacobian-adjusted.
    """

    # -------------------------
    # basic validation
    # -------------------------
    if not isinstance(out_dict, dict) or len(out_dict) == 0:
        raise ValueError("out_dict must be a non-empty dict like {'Model A': outA, ...}.")

    x_axis = str(x_axis).strip().lower()
    if x_axis not in {"r", "logr"}:
        if x_axis != "r_common" and x_axis != "gross" and x_axis != "return":
            pass
    if x_axis not in {"r", "logr", "r_common", "gross", "return", "rspace", "log-return", "log_return", "R".lower()}:
        raise ValueError("x_axis must be either 'R' or 'logR'.")

    use_log_x = x_axis in {"logr", "r_common", "rspace", "log-return", "log_return"}

    # normalize display name
    x_axis_name = "logR" if use_log_x else "R"

    mode = str(trunc_mode).lower().strip()
    if mode == "rbounds":
        mode = "xbounds"
    if mode == "cdf+rbounds":
        mode = "cdf+xbounds"
    if not truncate:
        mode = "none"

    valid_modes = {"none", "cdf", "xbounds", "cdf+xbounds"}
    if mode not in valid_modes:
        raise ValueError(f"trunc_mode must be one of {valid_modes}.")

    use_cdf = mode in {"cdf", "cdf+xbounds"}
    use_xbounds = mode in {"xbounds", "cdf+xbounds"}

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
    # helpers
    # -------------------------
    def _extract_transformed_surfaces(out):
        """
        Return (T, X, qX, pX, M) where X is either R or log(R),
        and qX / pX are the correctly transformed densities for that x-axis.
        """
        if out is None or "anchor_surfaces" not in out:
            raise KeyError("Each out dict must contain 'anchor_surfaces'.")

        anchor = out["anchor_surfaces"]

        T = np.asarray(out.get("T_anchor", []), float).ravel()
        R = np.asarray(out.get("R_common", []), float).ravel()

        qR = np.asarray(anchor.get("qR_surface", []), float)
        pR = np.asarray(anchor.get("pR_surface", []), float)
        M  = np.asarray(anchor.get("M_surface", []), float)

        if T.size == 0 or R.size == 0:
            raise ValueError("Missing T_anchor or R_common.")
        if qR.shape != (T.size, R.size) or pR.shape != (T.size, R.size) or M.shape != (T.size, R.size):
            raise ValueError("anchor_surfaces arrays do not match dimensions of T_anchor and R_common.")
        if np.any(~np.isfinite(R)):
            raise ValueError("R_common contains non-finite values.")
        if R.size >= 2 and np.any(np.diff(R) <= 0):
            raise ValueError("R_common must be strictly increasing.")
        if use_log_x:
            if np.any(R <= 0):
                raise ValueError("All R_common values must be > 0 when x_axis='logR'.")
            X = np.log(R)
            jac = R[None, :]   # dR/dr = R
            qX = qR * jac
            pX = pR * jac
        else:
            X = R
            qX = qR
            pX = pR

        return T, X, qX, pX, M

    def _interp_to_ref(X_src, y_src, X_ref):
        """Interpolate y(X_src) onto X_ref. Assumes X_src increasing."""
        return np.interp(X_ref, X_src, y_src)

    def _cdf_cutoffs(X_src, p_src, alpha_left, alpha_right):
        """
        Compute density-CDF cutoffs on the displayed x-axis support (X_src),
        return (X_left, X_right) corresponding to alpha_left / 1-alpha_right.
        """
        px = np.maximum(np.asarray(p_src, float), 0.0)
        if X_src.size < 2:
            return None

        dX = np.diff(X_src)
        inc = 0.5 * (px[1:] + px[:-1]) * dX

        cdf = np.empty_like(X_src)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(inc)

        total = float(cdf[-1])
        if not (np.isfinite(total) and total > 0):
            return None

        cdf /= total

        # left cutoff
        if alpha_left > 0:
            idxL = np.where(cdf >= alpha_left)[0]
            iL = int(idxL[0]) if idxL.size else 0
        else:
            iL = 0

        # right cutoff
        if alpha_right > 0:
            idxR = np.where(cdf <= (1.0 - alpha_right))[0]
            iR = int(idxR[-1]) if idxR.size else (X_src.size - 1)
        else:
            iR = X_src.size - 1

        if iR <= iL:
            return None

        return float(X_src[iL]), float(X_src[iR])

    # -------------------------
    # reference grids from first model
    # -------------------------
    first_label = next(iter(out_dict))
    T_ref, X_ref, q_ref0, p_ref0, M_ref0 = _extract_transformed_surfaces(out_dict[first_label])

    # x-bounds range (if enabled) in displayed x coordinates
    X_min = X_max = None
    if use_xbounds:
        if x_bounds is None or len(x_bounds) != 2:
            raise ValueError("For xbounds truncation, provide x_bounds=(xmin, xmax).")
        X_min, X_max = float(x_bounds[0]), float(x_bounds[1])

        if clip_trunc_to_support:
            X_min = max(X_min, float(X_ref[0]))
            X_max = min(X_max, float(X_ref[-1]))

        if not (np.isfinite(X_min) and np.isfinite(X_max) and X_max > X_min):
            raise ValueError("Invalid x_bounds after clipping.")

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
    # main plot loop
    # -------------------------
    for k, j in enumerate(idxs):
        ax = axes[k]
        ax2 = ax.twinx()
        ax2_list.append(ax2)

        # base mask on reference x grid
        mask_all = np.isfinite(X_ref)
        if use_xbounds:
            mask_all &= (X_ref >= X_min) & (X_ref <= X_max)

        X_all = X_ref[mask_all]

        for model_label, out in out_dict.items():
            try:
                T, X, qX, pX, M = _extract_transformed_surfaces(out)
            except Exception:
                continue

            # match maturity by nearest T to reference T_ref[j]
            jj = int(np.argmin(np.abs(T - T_ref[j])))

            qj = qX[jj, :]
            pj = pX[jj, :]
            Mj = M[jj, :]

            # interpolate to reference X grid if needed
            same_grid = (X.shape == X_ref.shape) and (
                X.size <= 1 or np.allclose(X, X_ref, atol=1e-12, rtol=1e-9)
            )

            if not same_grid:
                qj_ref = _interp_to_ref(X, qj, X_ref)
                pj_ref = _interp_to_ref(X, pj, X_ref)
                Mj_ref = _interp_to_ref(X, Mj, X_ref)
            else:
                qj_ref, pj_ref, Mj_ref = qj, pj, Mj

            # apply x-bounds mask to q/p/M for plotting
            q_plot = np.asarray(qj_ref, float)[mask_all]
            p_plot = np.asarray(pj_ref, float)[mask_all]
            M_plot = np.asarray(Mj_ref, float)[mask_all].copy()

            # labels
            if use_log_x:
                qlab = "q_r(r)"
                plab = "p_r(r)"
                mlab = "M(r)"
            else:
                qlab = "q_R(R)"
                plab = "p_R(R)"
                mlab = "M(R)"

            if show_model_in_label:
                qlab += f" ({model_label})"
                plab += f" ({model_label})"
                mlab += f" ({model_label})"

            # plot densities and capture model color
            (q_line,) = ax.plot(
                X_all, q_plot,
                linewidth=lw_density + 0.3,
                alpha=alpha_density,
                label=qlab,
            )
            model_color = q_line.get_color()

            ax.plot(
                X_all, p_plot,
                linewidth=lw_density,
                alpha=0.60,
                linestyle="-",
                label=plab,
                color=model_color,
            )

            # kernel truncation by p-CDF tails on FULL model support in displayed x-space
            X_k = X_all
            M_k = M_plot

            if use_cdf:
                cut = _cdf_cutoffs(X, pX[jj, :], aL, aR)
                if cut is None:
                    X_k = np.array([], float)
                    M_k = np.array([], float)
                else:
                    X_left, X_right = cut
                    keep = (X_k >= X_left) & (X_k <= X_right)
                    X_k = X_k[keep]
                    M_k = M_k[keep]

                    if use_log_x:
                        tail_text = f"p-tails ≥ {aL:.0%}/{aR:.0%}"
                        mlab = f"M(r) ({model_label}, {tail_text})" if show_model_in_label else f"M(r) ({tail_text})"
                    else:
                        tail_text = f"p-tails ≥ {aL:.0%}/{aR:.0%}"
                        mlab = f"M(R) ({model_label}, {tail_text})" if show_model_in_label else f"M(R) ({tail_text})"

            # log-scale cleanup for kernel y-axis
            if kernel_yscale == "log" and M_k.size > 0:
                pos = np.isfinite(M_k) & (M_k > 0) & np.isfinite(X_k)
                X_k = np.asarray(X_k, float)[pos]
                M_k = np.asarray(M_k, float)[pos]
                M_k = np.maximum(M_k, kernel_log_eps)

            if X_k.size > 0:
                ax2.plot(
                    X_k, M_k,
                    label=mlab + ("" if kernel_yscale == "linear" else " (log y)"),
                    linestyle=kernel_linestyle,
                    linewidth=lw_kernel,
                    alpha=alpha_kernel,
                    color=model_color,
                )

        # panel title from reference maturity
        T_days = float(T_ref[j] * 365.0)
        ax.set_title(f"T≈{T_days:.1f}d", fontsize=11)

        if (k % ncols) == 0:
            ax.set_ylabel("Density")
        if k >= (n_pan - ncols):
            ax.set_xlabel("Log return r = log(R)" if use_log_x else "Gross return R")
        if (k % ncols) == (ncols - 1):
            ax2.set_ylabel("Pricing kernel")

        ax.grid(True, alpha=0.25)
        ax2.set_yscale(kernel_yscale)

        if use_xbounds:
            ax.set_xlim(X_min, X_max)

    # turn off unused axes
    for k in range(n_pan, axes.size):
        axes[k].axis("off")

    # legend from first active axes
    handles, labels = [], []
    for ax_ in [axes[0], ax2_list[0]]:
        h, l = ax_.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    if title is None:
        if use_log_x:
            title = "q_r vs p_r with Pricing Kernel (multi-model overlay)"
        else:
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
    fP_key: str = "fP_surface",
    r_key: str = "r_common",
    T_key: str = "T_anchor",

    # axis controls
    x_axis: str = "r",                            # {"R","r"}
    x_bounds: tuple[float, float] | None = None,

    # thinning
    max_dates: int = 250,
    stride: int | None = None,

    # z scaling
    zscale: Literal["linear", "log"] = "log",
    z_eps: float = 1e-300,

    # appearance
    title: str = "Pricing Kernel Surface",
    cmap: str = "viridis",
    dpi: int = 200,
    interactive: bool = False,
    show: bool = True,
    alpha: float = 0.9,
    elev: float = 28,
    azim: float = -60,

    # optional truncation by physical tails
    truncate_by_ptails: bool = False,
    ptail_alphas: tuple[float, float] = (0.01, 0.01),

    # saving
    save: str | Path | None = None,

    # subplot support
    ax=None,
    add_colorbar: bool = True,
    clear_ax: bool = True,
):
    """
    3D surface:
      x = chosen return axis ("r" or "R")
      y = calendar date
      z = pricing kernel at maturity nearest to T_target

    Returns
    -------
    fig, ax, surf     (static matplotlib case)
    fig               (interactive plotly case)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pathlib import Path

    if x_axis not in {"R", "r"}:
        raise ValueError("x_axis must be one of {'R', 'r'}.")
    if zscale not in {"linear", "log"}:
        raise ValueError("zscale must be 'linear' or 'log'.")
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

    # --- build Z(date, r) ---
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
            if not np.all(np.diff(r_ref) > 0):
                raise ValueError("r_common must be strictly increasing.")
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

    Z = np.vstack(rows_Z)
    kept_dates = pd.to_datetime(kept_dates)

    # --- truncate by physical tail mass, row by row ---
    if truncate_by_ptails:
        FP = np.vstack(rows_fP)
        aL, aR = float(ptail_alphas[0]), float(ptail_alphas[1])

        for i in range(Z.shape[0]):
            fp = FP[i, :]
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
            keep_good[left_pos:right_pos + 1] = True

            keep_full = np.zeros(r_ref.size, dtype=bool)
            keep_full[np.where(good)[0][keep_good]] = True
            Z[i, ~keep_full] = np.nan

    # --- choose x axis like your other functions ---
    if x_axis == "r":
        x = r_ref.copy()
        xlabel = "Log return r = log(S_T/S0)"
    else:  # "R"
        x = np.exp(r_ref)
        xlabel = "Gross return R"

    # --- x bounds ---
    if x_bounds is not None:
        lo, hi = float(x_bounds[0]), float(x_bounds[1])
        if lo >= hi:
            raise ValueError("x_bounds must be (lo, hi) with lo < hi.")
        m = (x >= lo) & (x <= hi)
        if not np.any(m):
            raise ValueError("x_bounds produced an empty plotting window.")
        x = x[m]
        Z = Z[:, m]

    # --- z transform ---
    if zscale == "log":
        Z_plot = np.where(np.isfinite(Z) & (Z > 0), np.log(np.maximum(Z, z_eps)), np.nan)
        zlabel = "log M(T,r)" if x_axis == "r" else "log M(T,R)"
    else:
        Z_plot = Z
        zlabel = "M(T,r)" if x_axis == "r" else "M(T,R)"

    # --- mesh ---
    y_num = mdates.date2num(kept_dates.to_pydatetime())
    X, Y = np.meshgrid(x, y_num)

    # =========================
    # INTERACTIVE (Plotly)
    # =========================
    if interactive:
        import plotly.graph_objects as go

        y_labels = [d.strftime("%Y-%m-%d") for d in kept_dates]

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=np.array(y_labels)[:, None].repeat(len(x), axis=1),
                    z=Z_plot,
                    colorscale=cmap,
                    colorbar=dict(title=zlabel),
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title="Date",
                zaxis_title=zlabel,
            ),
            margin=dict(l=0, r=0, t=55, b=0),
        )

        if save is not None:
            save = Path(save)
            save.parent.mkdir(parents=True, exist_ok=True)
            if save.suffix.lower() == ".html":
                fig.write_html(save)
            else:
                fig.write_image(save)
            print(f"[saved] {save}")

        if show:
            fig.show()

        return fig

    # =========================
    # STATIC (Matplotlib)
    # =========================
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure
        if clear_ax:
            ax.cla()

    surf = ax.plot_surface(
        X,
        Y,
        Z_plot,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=alpha,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")  # avoid collision with date tick labels
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)

    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    try:
        ax.set_yticks(y_num[::max(1, len(y_num)//6)])
    except Exception:
        pass

    ax.xaxis.labelpad = 8
    ax.yaxis.labelpad = 8
    ax.zaxis.labelpad = 10

    if add_colorbar and created_fig:
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label=zlabel)

    if created_fig:
        plt.tight_layout()

    if save is not None:
        save_path = Path(save)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save_path}")

    if show and created_fig:
        plt.show()
    elif created_fig and not show:
        plt.close(fig)

    return fig, ax, surf


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