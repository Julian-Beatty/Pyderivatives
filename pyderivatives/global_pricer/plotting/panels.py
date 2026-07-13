from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
def _maturity_label_days(T_years: float) -> str:
    days = float(T_years) * 365.0

    if days < 1.0:
        return "Same day"

    return f"{days:.0f} days"

def _pick_panel_indices(T_grid: np.ndarray, n_panels: int) -> np.ndarray:
    T = np.asarray(T_grid, float).ravel()
    if T.size == 0:
        return np.array([], dtype=int)

    n = min(max(int(n_panels), 1), T.size)
    idx = np.unique(np.round(np.linspace(0, T.size - 1, n)).astype(int))
    return idx


def _make_axes(
    n_panels: int,
    *,
    panel_shape: Optional[tuple[int, int]] = None,
    figsize_per_panel: float = 2.2,
    base_width: float = 8.5,
    sharex: bool = True,
):
    if panel_shape is None:
        nrows, ncols = n_panels, 1
    else:
        nrows, ncols = int(panel_shape[0]), int(panel_shape[1])
        if nrows * ncols < n_panels:
            raise ValueError(
                f"panel_shape={panel_shape} has {nrows * ncols} slots, "
                f"but need {n_panels}."
            )

    fig_w = base_width * ncols
    fig_h = max(4.0, figsize_per_panel * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=sharex,
    )

    axes = np.atleast_1d(axes).ravel()
    return fig, axes, nrows, ncols


def _save_show(
    fig,
    *,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    show: bool = True,
):
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        if save.suffix == "":
            save = save.with_suffix(".png")
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _get_grid(res: dict, x_axis: Literal["k", "lr", "r"]) -> np.ndarray:
    key = {
        "k": "grid_k",
        "lr": "grid_lr",
        "r": "grid_r",
    }[x_axis]

    if key not in res:
        raise KeyError(f"Missing required grid: res['{key}'].")

    return np.asarray(res[key], float).ravel()


def _get_rnd_surface(res: dict, x_axis: Literal["k", "lr", "r"]) -> np.ndarray:
    key = {
        "k": "rnd_k_surface",
        "lr": "rnd_lr_surface",
        "r": "rnd_r_surface",
    }[x_axis]

    if key not in res:
        raise KeyError(f"Missing required RND surface: res['{key}'].")

    return np.asarray(res[key], float)


def _axis_labels(x_axis: Literal["k", "lr", "r"]) -> tuple[str, str]:
    if x_axis == "k":
        return "Strike K", r"$q_K(K)$"
    if x_axis == "lr":
        return "Log return lr = log(K/S0)", r"$q_{lr}(lr)$"
    if x_axis == "r":
        return "Gross return R = K/S0", r"$q_R(R)$"
    raise ValueError("x_axis must be one of {'k', 'lr', 'r'}.")


def _x_mask(x: np.ndarray, x_bounds: Optional[tuple[float, float]]) -> np.ndarray:
    x = np.asarray(x, float).ravel()

    if x_bounds is None:
        mask = np.isfinite(x)
    else:
        lo, hi = map(float, x_bounds)
        if lo >= hi:
            raise ValueError("x_bounds must satisfy lo < hi.")
        mask = np.isfinite(x) & (x >= lo) & (x <= hi)

    if not np.any(mask):
        raise ValueError("x_bounds produced an empty plotting window.")

    return mask


from typing import Optional, Union, Literal
from pathlib import Path
import numpy as np

from typing import Optional, Union, Literal
from pathlib import Path
import numpy as np


from typing import Optional, Union, Literal
from pathlib import Path
import numpy as np


def call_panels(
    res: dict,
    *,
    day,
    n_panels: int = 6,
    title: str = "Observed vs Fitted Curves",
    date_str: Optional[str] = None,
    spot: Optional[float] = None,
    T_cluster_tol: float = 1.0 / 365.0,
    legend_loc: str = "upper right",
    figsize_per_panel: float = 2.2,
    K_pad_frac: float = 0.05,
    K_pad_abs: float = 0.0,
    x_axis: Literal["k", "r", "lr"] = "k",
    x_bounds: Optional[tuple[float, float]] = None,
    auto_ylim_visible: bool = True,
    y_pad_frac: float = 0.05,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    show: bool = True,
):
    """
    x_axis convention:
        "k"  = strike K
        "r"  = gross return / moneyness K/S
        "lr" = log return log(K/S)

    Examples:
        x_axis="k",  x_bounds=(6700, 7050)
        x_axis="r",  x_bounds=(0.98, 1.02)
        x_axis="lr", x_bounds=(-0.02, 0.02)
    """

    K_grid = np.asarray(res["grid_k"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()
    C_fit = np.asarray(res["C_fit"], float)

    if C_fit.shape != (T_grid.size, K_grid.size):
        raise ValueError(
            f"C_fit has shape {C_fit.shape}, expected {(T_grid.size, K_grid.size)}."
        )

    K_obs = np.asarray(day.K_obs, float).ravel()
    T_obs = np.asarray(day.T_obs, float).ravel()
    C_obs = np.asarray(day.C_obs, float).ravel()

    valid = (
        np.isfinite(K_obs)
        & np.isfinite(T_obs)
        & np.isfinite(C_obs)
        & (K_obs > 0)
        & (T_obs >= 0)
        & (C_obs >= 0)
    )

    K_obs, T_obs, C_obs = K_obs[valid], T_obs[valid], C_obs[valid]

    if K_obs.size == 0:
        raise ValueError("No valid observed quotes to plot.")

    if spot is None:
        spot = res.get("S0", None) or getattr(day, "S0", None) or getattr(day, "spot", None)

    spot = float(spot) if spot is not None else None

    if x_axis in ("r", "lr") and spot is None:
        raise ValueError("spot must be provided when x_axis is 'r' or 'lr'.")

    if x_axis == "k":
        X_grid = K_grid
        X_obs = K_obs
        x_label = "Strike K"
        spot_x = spot

    elif x_axis == "r":
        X_grid = K_grid / spot
        X_obs = K_obs / spot
        x_label = "Gross return / moneyness K/S"
        spot_x = 1.0

    elif x_axis == "lr":
        X_grid = np.log(K_grid / spot)
        X_obs = np.log(K_obs / spot)
        x_label = "Log return log(K/S)"
        spot_x = 0.0

    else:
        raise ValueError("x_axis must be one of {'k', 'r', 'lr'}.")

    if x_bounds is None:
        Xmin_obs = float(np.nanmin(X_obs))
        Xmax_obs = float(np.nanmax(X_obs))
        obs_range = max(Xmax_obs - Xmin_obs, 1e-12)

        if x_axis == "k":
            pad = float(K_pad_abs) + float(K_pad_frac) * obs_range
        else:
            pad = float(K_pad_frac) * obs_range

        Xmin_plot = Xmin_obs - pad
        Xmax_plot = Xmax_obs + pad
    else:
        Xmin_plot, Xmax_plot = map(float, x_bounds)

    x_mask = (X_grid >= Xmin_plot) & (X_grid <= Xmax_plot)

    if np.sum(x_mask) < 2:
        raise ValueError(
            f"X-axis window has too few grid points to plot. "
            f"x_axis={x_axis}, x_bounds=({Xmin_plot}, {Xmax_plot}), "
            f"grid range=({np.nanmin(X_grid)}, {np.nanmax(X_grid)})."
        )

    T_sorted = np.sort(np.unique(T_obs))

    clusters = []
    cur = [T_sorted[0]]

    for t in T_sorted[1:]:
        if abs(t - cur[-1]) <= T_cluster_tol:
            cur.append(t)
        else:
            clusters.append(float(np.mean(cur)))
            cur = [t]

    clusters.append(float(np.mean(cur)))
    T_centers = np.asarray(clusters, float)

    if T_centers.size <= n_panels:
        T_panels = T_centers
    else:
        idx = np.unique(
            np.round(np.linspace(0, T_centers.size - 1, n_panels)).astype(int)
        )
        T_panels = T_centers[idx]

    fig, axes, nrows, ncols = _make_axes(
        len(T_panels),
        panel_shape=None,
        figsize_per_panel=figsize_per_panel,
        sharex=True,
    )

    X_plot = X_grid[x_mask]

    for ax, T_panel in zip(axes[: len(T_panels)], T_panels):
        qmask = np.abs(T_obs - T_panel) <= T_cluster_tol
        fit_idx = int(np.argmin(np.abs(T_grid - T_panel)))

        visible_obs = qmask & (X_obs >= Xmin_plot) & (X_obs <= Xmax_plot)

        if np.any(qmask):
            ax.scatter(
                X_obs[qmask],
                C_obs[qmask],
                s=22,
                alpha=0.9,
                label="Observed",
            )

        fit_y = C_fit[fit_idx, x_mask]

        ax.plot(
            X_plot,
            fit_y,
            linewidth=2.2,
            label="Global model",
        )

        if spot_x is not None and Xmin_plot <= spot_x <= Xmax_plot:
            ax.axvline(spot_x, linestyle="--", linewidth=1.5, label="Spot")

        ax.set_xlim(Xmin_plot, Xmax_plot)

        if auto_ylim_visible:
            y_candidates = []

            if np.any(visible_obs):
                y_candidates.append(np.nanmax(C_obs[visible_obs]))

            if fit_y.size > 0 and np.any(np.isfinite(fit_y)):
                y_candidates.append(np.nanmax(fit_y))

            if len(y_candidates) > 0:
                ymax = float(np.nanmax(y_candidates))
                if np.isfinite(ymax):
                    ax.set_ylim(0, ymax * (1.0 + y_pad_frac))

        d = f"{date_str} " if date_str else ""
        ax.set_title(f"{d}T ≈ {_maturity_label_days(T_panel)}")
        ax.set_ylabel("Call price")
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)

    axes[len(T_panels) - 1].set_xlabel(x_label)

    fig.suptitle(title)
    fig.tight_layout()

    return _save_show(fig, save=save, dpi=dpi, show=show)


def iv_panels(
    res: dict,
    *,
    n_panels: int = 6,
    title: str = "IV Panels",
    x_axis: Literal["k", "lr", "r"] = "k",
    x_bounds: Optional[tuple[float, float]] = None,
    panel_shape: Optional[tuple[int, int]] = None,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = True,
    figsize_per_panel: float = 2.2,
):
    if "iv_surface" not in res:
        raise KeyError("Missing IV surface. Expected res['iv_surface'].")

    iv = np.asarray(res["iv_surface"], float)
    T_grid = np.asarray(res["T_grid"], float).ravel()

    if x_axis == "k":
        x = np.asarray(res["grid_k"], float).ravel()
    elif x_axis == "lr":
        x = np.asarray(res["grid_lr"], float).ravel()
    elif x_axis == "r":
        x = np.asarray(res["grid_r"], float).ravel()
    else:
        raise ValueError("x_axis must be one of {'k', 'lr', 'r'}.")

    if iv.shape != (T_grid.size, x.size):
        raise ValueError(f"iv_surface has shape {iv.shape}, expected {(T_grid.size, x.size)}.")

    xmask = _x_mask(x, x_bounds)
    x_plot = x[xmask]

    idxT = _pick_panel_indices(T_grid, n_panels)
    n_actual = len(idxT)

    fig, axes, nrows, ncols = _make_axes(
        n_actual,
        panel_shape=panel_shape,
        figsize_per_panel=figsize_per_panel,
        sharex=True,
    )

    xlabel = {
        "k": "Strike K",
        "lr": "Log return lr = log(K/S0)",
        "r": "Gross return R = K/S0",
    }[x_axis]

    for ax, j in zip(axes[:n_actual], idxT):
        ax.plot(x_plot, iv[j, xmask], linewidth=2.0)
        ax.set_title(f"T = {float(T_grid[j]):.5g} yr")
        ax.set_ylabel("Implied vol")
        ax.grid(True, alpha=0.25)

        if x_axis == "lr" and x_plot.min() <= 0 <= x_plot.max():
            ax.axvline(0.0, linestyle="--", linewidth=1.3)
        elif x_axis == "r" and x_plot.min() <= 1 <= x_plot.max():
            ax.axvline(1.0, linestyle="--", linewidth=1.3)

    for ax in axes[n_actual:]:
        ax.axis("off")

    for ax in axes[(nrows - 1) * ncols : nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel(xlabel)

    fig.suptitle(title)
    fig.tight_layout()

    return _save_show(fig, save=save, dpi=dpi, show=show)

def rnd_panels(
    res: dict,
    *,
    x_axis: Literal["k", "lr", "r"] = "k",
    n_panels: int = 6,
    title: str = "RND Panels",
    date_str: Optional[str] = None,
    x_bounds: Optional[tuple[float, float]] = None,
    panel_shape: Optional[tuple[int, int]] = None,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    show: bool = True,
    legend_loc: str = "upper right",
    figsize_per_panel: float = 2.2,
    show_spot: bool = True,

    pct_lower: Optional[float] = None,
    pct_upper: Optional[float] = None,
    mark_percentiles: bool = True,
    pct_line_style: str = ":",
    pct_line_width: float = 2.0,

    show_mode: bool = True,
    show_median: bool = True,
    mode_line_style: str = "-.",
    median_line_style: str = "--",
    mode_line_width: float = 2.0,
    median_line_width: float = 2.0,
    mode_color: str = "tab:purple",
    median_color: str = "tab:orange",
    target_maturities: Optional[Sequence[float]] = None,
    
    rnd_dict: Optional[dict] = None,
    snap_percentiles_to_traded_strikes: bool = False,
    snap_maturity_tol: float = 1.0 / 365.0,

    only_plot_traded_maturities: bool = False,
    traded_maturity_tol: float = 1.5 / 365.0,

    x_tick_nbins: int = 10,
    show_mean: bool = True,
    mean_line_style: str = "-",
    mean_line_width: float = 2.0,
    mean_color: str = "tab:brown",
):
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    def _cdf_prob_at_x(xv, pdf, xmark):
        xv = np.asarray(xv, float)
        pdf = np.clip(np.asarray(pdf, float), 0.0, np.inf)

        good = np.isfinite(xv) & np.isfinite(pdf)

        xv = xv[good]
        pdf = pdf[good]

        if xv.size < 2:
            return np.nan

        order = np.argsort(xv)
        xv = xv[order]
        pdf = pdf[order]

        area = np.trapezoid(pdf, xv)

        if not np.isfinite(area) or area <= 0:
            return np.nan

        pdf = pdf / area

        cdf = np.concatenate([
            [0.0],
            np.cumsum((pdf[1:] + pdf[:-1]) * 0.5 * np.diff(xv))
        ])

        cdf = np.clip(cdf, 0.0, 1.0)

        return float(np.interp(xmark, xv, cdf))

    def _to_prob(p):
        if p is None:
            return None

        p = float(p)

        if p > 1.0:
            p = p / 100.0

        if not (0.0 < p < 1.0):
            raise ValueError("Percentiles must be in (0,1) or (0,100).")

        return p

    def _quantile_from_pdf(xv, pdf, qprob):
        xv = np.asarray(xv, float)
        pdf = np.clip(np.asarray(pdf, float), 0.0, np.inf)

        good = np.isfinite(xv) & np.isfinite(pdf)

        xv = xv[good]
        pdf = pdf[good]

        if xv.size < 2:
            return np.nan

        order = np.argsort(xv)
        xv = xv[order]
        pdf = pdf[order]

        area = np.trapezoid(pdf, xv)

        if not np.isfinite(area) or area <= 0:
            return np.nan

        pdf = pdf / area

        cdf = np.concatenate([
            [0.0],
            np.cumsum((pdf[1:] + pdf[:-1]) * 0.5 * np.diff(xv))
        ])

        cdf = np.clip(cdf, 0.0, 1.0)

        return float(np.interp(qprob, cdf, xv))

    def _get_single_rnd_day():
        if rnd_dict is None:
            return None

        if isinstance(rnd_dict, dict) and "day" in rnd_dict:
            return rnd_dict["day"]

        if isinstance(rnd_dict, dict) and len(rnd_dict) == 1:
            entry = next(iter(rnd_dict.values()))

            if isinstance(entry, dict) and "day" in entry:
                return entry["day"]

        return None

    def _get_day_arrays(rnd_day):
        if rnd_day is None:
            return None, None

        if isinstance(rnd_day, dict):
            K_obs = np.asarray(rnd_day["K_obs"], float).ravel()
            T_obs = np.asarray(rnd_day["T_obs"], float).ravel()
        else:
            K_obs = np.asarray(rnd_day.K_obs, float).ravel()
            T_obs = np.asarray(rnd_day.T_obs, float).ravel()

        valid = (
            np.isfinite(K_obs)
            & np.isfinite(T_obs)
            & (K_obs > 0)
            & (T_obs >= 0)
        )

        return K_obs[valid], T_obs[valid]

    def _snap_to_traded_x(xmark, T_target):
        if rnd_day is None or not np.isfinite(xmark):
            return xmark

        K_obs, T_obs = _get_day_arrays(rnd_day)

        if K_obs is None or K_obs.size == 0:
            return xmark

        tmask = np.abs(T_obs - float(T_target)) <= float(snap_maturity_tol)

        if np.any(tmask):
            K_use = K_obs[tmask]
        else:
            K_use = K_obs

        if x_axis == "k":
            K_raw = float(xmark)
            S0 = None

        else:
            S0 = res.get("S0", None)

            if S0 is None and isinstance(rnd_day, dict):
                S0 = rnd_day.get("S0", None)

            if S0 is None:
                S0 = getattr(rnd_day, "S0", None)

            if S0 is None:
                return xmark

            S0 = float(S0)

            if x_axis == "r":
                K_raw = float(xmark) * S0
            elif x_axis == "lr":
                K_raw = float(np.exp(xmark) * S0)
            else:
                return xmark

        K_snap = float(K_use[np.argmin(np.abs(K_use - K_raw))])

        if x_axis == "k":
            return K_snap
        elif x_axis == "r":
            return K_snap / S0
        elif x_axis == "lr":
            return float(np.log(K_snap / S0))

        return xmark

    def _pick_traded_maturity_indices(T_grid, rnd_day, max_panels):
        K_obs, T_obs = _get_day_arrays(rnd_day)

        if T_obs is None or T_obs.size == 0:
            raise ValueError(
                "only_plot_traded_maturities=True requires rnd_dict={'day': day}."
            )

        traded_T = np.sort(np.unique(np.round(T_obs, 10)))

        idxs = []

        for T_traded in traded_T:
            j = int(np.argmin(np.abs(T_grid - T_traded)))
            dist = abs(float(T_grid[j]) - float(T_traded))

            if dist <= float(traded_maturity_tol):
                if j not in idxs:
                    idxs.append(j)

        if len(idxs) == 0:
            raise ValueError(
                "No traded maturities matched T_grid. Increase traded_maturity_tol "
                "or include the traded maturities in T_grid."
            )

        return np.asarray(idxs[:max_panels], dtype=int)

    x_axis = str(x_axis).lower().strip()

    if x_axis in {"strike", "k"}:
        x_axis = "k"
    elif x_axis in {"logreturn", "log_return", "lr"}:
        x_axis = "lr"
    elif x_axis in {"r", "return", "gross_return"}:
        x_axis = "r"
    else:
        raise ValueError("x_axis must be 'k', 'r', or 'lr'.")

    x = _get_grid(res, x_axis)
    qsurf = _get_rnd_surface(res, x_axis)
    T_grid = np.asarray(res["T_grid"], float).ravel()

    if qsurf.shape != (T_grid.size, x.size):
        raise ValueError(
            f"RND surface has shape {qsurf.shape}, "
            f"expected {(T_grid.size, x.size)}."
        )

    xmask = _x_mask(x, x_bounds)
    x_plot = x[xmask]

    rnd_day = _get_single_rnd_day()

    if target_maturities is not None:
    
        targets = np.asarray(target_maturities, float)
    
        # assume user passes days
        targets = targets / 365.0
    
        idxT = []
    
        for target in targets:
            j = int(np.argmin(np.abs(T_grid - target)))
    
            if j not in idxT:
                idxT.append(j)
    
        idxT = np.asarray(idxT, dtype=int)
    
    elif only_plot_traded_maturities:
    
        idxT = _pick_traded_maturity_indices(
            T_grid=T_grid,
            rnd_day=rnd_day,
            max_panels=int(n_panels),
        )
    
    else:
    
        idxT = _pick_panel_indices(T_grid, n_panels)
    n_actual = len(idxT)

    fig, axes, nrows, ncols = _make_axes(
        n_actual,
        panel_shape=panel_shape,
        figsize_per_panel=figsize_per_panel,
        sharex=True,
    )

    xlabel, ylabel = _axis_labels(x_axis)

    qL = _to_prob(pct_lower)
    qU = _to_prob(pct_upper)

    want_pct = (
        bool(mark_percentiles)
        and ((qL is not None) or (qU is not None))
    )

    for ax, j in zip(axes[:n_actual], idxT):
        pdf_row = qsurf[j, xmask]

        ax.plot(
            x_plot,
            pdf_row,
            linewidth=2.2,
            label="RND",
        )

        # --------------------------------------------------
        # Literal mode: never snapped to traded strikes
        # --------------------------------------------------
        if show_mode:
            good_mode = np.isfinite(x_plot) & np.isfinite(pdf_row)

            if np.any(good_mode):
                x_mode = float(x_plot[good_mode][np.argmax(pdf_row[good_mode])])

                ax.axvline(
                    x_mode,
                    linestyle=mode_line_style,
                    linewidth=mode_line_width,
                    color=mode_color,
                    label=f"Mode: K={x_mode:.2f}" if x_axis == "k" else f"Mode: {x_mode:.4f}",
                )

        # --------------------------------------------------
        # Literal median: never snapped to traded strikes
        # --------------------------------------------------
        if show_median:
            x_med = _quantile_from_pdf(x_plot, pdf_row, 0.50)

            if np.isfinite(x_med):
                ax.axvline(
                    x_med,
                    linestyle=median_line_style,
                    linewidth=median_line_width,
                    color=median_color,
                    label=f"Median: K={x_med:.2f}" if x_axis == "k" else f"Median: {x_med:.4f}",
                )
        # --------------------------------------------------
        # Mean: literal density mean
        # --------------------------------------------------
        if show_mean:
        
            pdf_mean = np.clip(np.asarray(pdf_row, float), 0.0, np.inf)
        
            area = np.trapezoid(pdf_mean, x_plot)
        
            if np.isfinite(area) and area > 0:
        
                pdf_mean = pdf_mean / area
        
                x_mean = float(
                    np.trapezoid(
                        x_plot * pdf_mean,
                        x_plot
                    )
                )
        
                ax.axvline(
                    x_mean,
                    linestyle=mean_line_style,
                    linewidth=mean_line_width,
                    color=mean_color,
                    label=(
                        f"Mean: K={x_mean:.2f}"
                        if x_axis == "k"
                        else f"Mean: {x_mean:.4f}"
                    ),
                )
        # --------------------------------------------------
        # Tail percentiles: optionally snapped to traded strikes
        # --------------------------------------------------
        if want_pct:
            if qL is not None:
                xL = _quantile_from_pdf(x_plot, pdf_row, qL)

                if snap_percentiles_to_traded_strikes:
                    xL = _snap_to_traded_x(xL, T_grid[j])

                if np.isfinite(xL):
                    actual_pct = 100 * _cdf_prob_at_x(x_plot, pdf_row, xL)

                    if x_axis == "k":
                        lab = f"{actual_pct:.1f}% (K={xL:.2f})"
                    elif x_axis == "r":
                        lab = f"{actual_pct:.1f}% (R={xL:.4f})"
                    else:
                        lab = f"{actual_pct:.1f}% (lr={xL:.4f})"

                    ax.axvline(
                        xL,
                        linestyle=pct_line_style,
                        linewidth=pct_line_width,
                        color="tab:red",
                        label=lab,
                    )

            if qU is not None:
                xU = _quantile_from_pdf(x_plot, pdf_row, qU)

                if snap_percentiles_to_traded_strikes:
                    xU = _snap_to_traded_x(xU, T_grid[j])

                if np.isfinite(xU):
                    actual_pct = 100 * _cdf_prob_at_x(x_plot, pdf_row, xU)

                    if x_axis == "k":
                        lab = f"{actual_pct:.1f}% (K={xU:.2f})"
                    elif x_axis == "r":
                        lab = f"{actual_pct:.1f}% (R={xU:.4f})"
                    else:
                        lab = f"{actual_pct:.1f}% (lr={xU:.4f})"

                    ax.axvline(
                        xU,
                        linestyle=pct_line_style,
                        linewidth=pct_line_width,
                        color="tab:green",
                        label=lab,
                    )

        if show_spot:
            if x_axis == "lr" and x_plot.min() <= 0 <= x_plot.max():
                ax.axvline(
                    0.0,
                    linestyle="--",
                    linewidth=1.3,
                    label="Spot: lr=0",
                )

            elif x_axis == "r" and x_plot.min() <= 1 <= x_plot.max():
                ax.axvline(
                    1.0,
                    linestyle="--",
                    linewidth=1.3,
                    label="Spot: R=1",
                )

            elif x_axis == "k":
                spot = res.get("S0", None)

                if spot is not None:
                    spot = float(spot)

                    if x_plot.min() <= spot <= x_plot.max():
                        ax.axvline(
                            spot,
                            linestyle="--",
                            linewidth=1.3,
                            label=f"Spot: K={spot:.2f}",
                        )

        d = f"{date_str} " if date_str else ""

        ax.set_title(f"{d}T = {_maturity_label_days(T_grid[j])}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_tick_nbins))
        ax.legend(loc=legend_loc)

    for ax in axes[n_actual:]:
        ax.axis("off")

    for ax in axes[(nrows - 1) * ncols : nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel(xlabel)

    fig.suptitle(title)
    fig.tight_layout()

    return _save_show(fig, save=save, dpi=dpi, show=show)


def cdf_panels(
    res: dict,
    *,
    n_panels: int = 6,
    title: str = "CDF Panels",
    date_str: Optional[str] = None,
    x_bounds: Optional[tuple[float, float]] = None,
    panel_shape: Optional[tuple[int, int]] = None,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    show: bool = True,
    legend_loc: str = "upper right",
    figsize_per_panel: float = 2.2,
):
    if "rnd_cdf_surface" not in res:
        raise KeyError("Missing CDF. Expected res['rnd_cdf_surface'].")

    F = np.asarray(res["rnd_cdf_surface"], float)
    K_grid = np.asarray(res["grid_k"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()

    if F.shape != (T_grid.size, K_grid.size):
        raise ValueError(f"CDF surface has shape {F.shape}, expected {(T_grid.size, K_grid.size)}.")

    k_mask = _x_mask(K_grid, x_bounds)
    K_plot = K_grid[k_mask]

    idxT = _pick_panel_indices(T_grid, n_panels)
    n_actual = len(idxT)

    fig, axes, nrows, ncols = _make_axes(
        n_actual,
        panel_shape=panel_shape,
        figsize_per_panel=figsize_per_panel,
        sharex=True,
    )

    for ax, j in zip(axes[:n_actual], idxT):
        ax.plot(K_plot, F[j, k_mask], linewidth=2.2, label="CDF")

        d = f"{date_str} " if date_str else ""
        ax.set_title(f"{d}T = {float(T_grid[j]):.5g} yr")
        ax.set_ylabel("CDF")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)

    for ax in axes[n_actual:]:
        ax.axis("off")

    for ax in axes[(nrows - 1) * ncols : nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel("Strike K")

    fig.suptitle(title)
    fig.tight_layout()

    return _save_show(fig, save=save, dpi=dpi, show=show)


def delta_panels(
    res: dict,
    *,
    which: Literal["skew", "call", "put"] = "skew",
    n_panels: int = 6,
    title: str = "Delta Panels",
    date_str: Optional[str] = None,
    spot: Optional[float] = None,
    legend_loc: str = "upper right",
    figsize_per_panel: float = 2.2,
    panel_shape: Optional[tuple[int, int]] = None,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    show: bool = True,
):
    if "delta_dict" not in res or res["delta_dict"] is None:
        raise KeyError("Missing delta_dict. Expected res['delta_dict'].")

    d = res["delta_dict"]

    delta = np.asarray(d["delta_axis"], float).ravel()
    T = np.asarray(d["T_axis"], float).ravel()

    if which == "call":
        Z_key = "iv_delta_call"
        ylabel = "IV (call delta)"
        label = "IV call"
    elif which == "put":
        Z_key = "iv_delta_put_abs"
        ylabel = "IV (|put delta|)"
        label = "IV put"
    elif which == "skew":
        Z_key = "delta_skew_surface"
        ylabel = "Normalized delta skew"
        label = "Skew"
    else:
        raise ValueError("which must be one of {'skew', 'call', 'put'}.")

    if Z_key not in d:
        raise KeyError(f"delta_dict is missing '{Z_key}'.")

    Z = np.asarray(d[Z_key], float)

    if Z.shape != (T.size, delta.size):
        raise ValueError(f"{Z_key} has shape {Z.shape}, expected {(T.size, delta.size)}.")

    valid_rows = np.where(np.any(np.isfinite(Z), axis=1))[0]

    if valid_rows.size == 0:
        raise ValueError("No finite rows to plot.")

    if valid_rows.size <= n_panels:
        idxT = valid_rows
    else:
        idx = np.unique(np.round(np.linspace(0, valid_rows.size - 1, n_panels)).astype(int))
        idxT = valid_rows[idx]

    n_actual = len(idxT)

    fig, axes, nrows, ncols = _make_axes(
        n_actual,
        panel_shape=panel_shape,
        figsize_per_panel=figsize_per_panel,
        sharex=True,
    )

    spot_delta = float(spot) if spot is not None else None

    for ax, j in zip(axes[:n_actual], idxT):
        y = Z[j, :]
        mask = np.isfinite(delta) & np.isfinite(y)

        if np.sum(mask) < 2:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.grid(True, alpha=0.25)
            continue

        ax.plot(delta[mask], y[mask], linewidth=2.2, label=label)

        if spot_delta is not None:
            ax.axvline(spot_delta, linestyle="--", linewidth=1.5, label="Spot delta")

        dstr = f"{date_str} " if date_str else ""
        ax.set_title(f"{dstr}T = {float(T[j]):.5g} yr")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)

    for ax in axes[n_actual:]:
        ax.axis("off")

    for ax in axes[(nrows - 1) * ncols : nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel("Delta")

    fig.suptitle(title)
    fig.tight_layout()

    return _save_show(fig, save=save, dpi=dpi, show=show)
def gamma_panels(
    res: dict,
    *,
    n_panels: int = 6,
    title: str = "Gamma Panels",
    x_axis: Literal["k", "lr", "r"] = "r",
    x_bounds: Optional[tuple[float, float]] = None,
    panel_shape: Optional[tuple[int, int]] = None,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = True,
    figsize_per_panel: float = 2.2,
):
    if "gamma_surface" not in res:
        raise KeyError("Missing gamma surface. Expected res['gamma_surface'].")

    gamma = np.asarray(res["gamma_surface"], float)
    T_grid = np.asarray(res["T_grid"], float).ravel()

    if x_axis == "k":
        x = np.asarray(res["grid_k"], float).ravel()
        xlabel = "Strike K"
        spot_x = res.get("S0", None)

    elif x_axis == "lr":
        x = np.asarray(res["grid_lr"], float).ravel()
        xlabel = "Log return lr = log(K/S0)"
        spot_x = 0.0

    elif x_axis == "r":
        x = np.asarray(res["grid_r"], float).ravel()
        xlabel = "Gross return R = K/S0"
        spot_x = 1.0

    else:
        raise ValueError("x_axis must be one of {'k', 'lr', 'r'}.")

    if gamma.shape != (T_grid.size, x.size):
        raise ValueError(
            f"gamma_surface has shape {gamma.shape}, expected {(T_grid.size, x.size)}."
        )

    xmask = _x_mask(x, x_bounds)
    x_plot = x[xmask]

    idxT = _pick_panel_indices(T_grid, n_panels)
    n_actual = len(idxT)

    fig, axes, nrows, ncols = _make_axes(
        n_actual,
        panel_shape=panel_shape,
        figsize_per_panel=figsize_per_panel,
        sharex=True,
    )

    for ax, j in zip(axes[:n_actual], idxT):
        y = gamma[j, xmask]

        ax.plot(x_plot, y, linewidth=2.0)
        ax.set_title(f"T = {float(T_grid[j]):.5g} yr")
        ax.set_ylabel("BS gamma")
        ax.grid(True, alpha=0.25)

        if spot_x is not None:
            spot_x = float(spot_x)
            if x_plot.min() <= spot_x <= x_plot.max():
                ax.axvline(spot_x, linestyle="--", linewidth=1.3, label="Spot")
                ax.legend(loc="upper right")

    for ax in axes[n_actual:]:
        ax.axis("off")

    for ax in axes[(nrows - 1) * ncols : nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel(xlabel)

    fig.suptitle(title)
    fig.tight_layout()

    return _save_show(fig, save=save, dpi=dpi, show=show)

def plot_overlay_rnd_lr_slices_subplots(
    results_dict: dict,
    maturities=None,
    maturity_indices=None,
    n_select: int = 5,
    figsize: tuple[float, float] = (10, 3),
    main_title: str = "Risk-Neutral Density Slices in Log-Return Space",
    xlabel: str = "Log return lr = log(K/S0)",
    ylabel: str = "Density",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    linewidth: float = 2.0,
    grid: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = True,
):
    if not results_dict:
        raise ValueError("results_dict is empty.")

    required_keys = ["rnd_lr_surface", "grid_lr", "T_grid"]

    asset_names = list(results_dict.keys())
    first_name = asset_names[0]
    first_result = results_dict[first_name]

    for key in required_keys:
        if key not in first_result:
            raise KeyError(f"{first_name} is missing required key: '{key}'.")

    T_grid_ref = np.asarray(first_result["T_grid"], float).ravel()
    x_ref = np.asarray(first_result["grid_lr"], float).ravel()
    surface_ref = np.asarray(first_result["rnd_lr_surface"], float)

    if surface_ref.shape != (T_grid_ref.size, x_ref.size):
        raise ValueError(
            f"{first_name}: rnd_lr_surface has shape {surface_ref.shape}, "
            f"expected {(T_grid_ref.size, x_ref.size)}."
        )

    if maturities is not None and np.isscalar(maturities):
        maturities = [maturities]

    if maturity_indices is not None and np.isscalar(maturity_indices):
        maturity_indices = [maturity_indices]

    if maturity_indices is not None:
        selected_idx = np.asarray(maturity_indices, dtype=int)
    elif maturities is not None:
        maturities = np.asarray(maturities, float)
        selected_idx = np.asarray(
            [np.argmin(np.abs(T_grid_ref - m)) for m in maturities],
            dtype=int,
        )
    else:
        if n_select < 1:
            raise ValueError("n_select must be at least 1.")
        selected_idx = np.linspace(
            0,
            T_grid_ref.size - 1,
            min(n_select, T_grid_ref.size),
            dtype=int,
        )

    selected_idx = np.asarray(list(dict.fromkeys(selected_idx.tolist())), dtype=int)

    if np.any(selected_idx < 0) or np.any(selected_idx >= T_grid_ref.size):
        raise IndexError("One or more maturity indices are out of bounds.")

    selected_T_ref = T_grid_ref[selected_idx]
    n_panels = len(selected_T_ref)

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(figsize[0], figsize[1] * n_panels),
        sharex=True,
    )

    axes = np.atleast_1d(axes).ravel()

    for p, T_target in enumerate(selected_T_ref):
        ax = axes[p]

        for asset_name, res in results_dict.items():
            for key in required_keys:
                if key not in res:
                    raise KeyError(f"{asset_name} is missing required key: '{key}'.")

            x = np.asarray(res["grid_lr"], float).ravel()
            T_grid = np.asarray(res["T_grid"], float).ravel()
            surface = np.asarray(res["rnd_lr_surface"], float)

            if surface.shape != (T_grid.size, x.size):
                raise ValueError(
                    f"{asset_name}: rnd_lr_surface has shape {surface.shape}, "
                    f"expected {(T_grid.size, x.size)}."
                )

            idx = int(np.argmin(np.abs(T_grid - T_target)))

            ax.plot(
                x,
                surface[idx, :],
                linewidth=linewidth,
                label=f"{asset_name} T={T_grid[idx]:.5g}",
            )

        if xlim is None or (xlim[0] <= 0.0 <= xlim[1]):
            ax.axvline(0.0, linestyle="--", linewidth=1.5)

        ax.set_title(f"T ≈ {T_target:.6f} yr")
        ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        if grid:
            ax.grid(True, alpha=0.3)

        ax.legend()

    axes[-1].set_xlabel(xlabel)

    fig.suptitle(main_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes