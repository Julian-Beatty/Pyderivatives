from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import matplotlib.pyplot as plt


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
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    show: bool = True,
):
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

    Kmin_obs = float(np.nanmin(K_obs))
    Kmax_obs = float(np.nanmax(K_obs))
    obs_range = max(Kmax_obs - Kmin_obs, 1e-12)

    pad = float(K_pad_abs) + float(K_pad_frac) * obs_range

    Kmin_plot = max(Kmin_obs - pad, float(np.nanmin(K_grid)))
    Kmax_plot = min(Kmax_obs + pad, float(np.nanmax(K_grid)))

    k_mask = (K_grid >= Kmin_plot) & (K_grid <= Kmax_plot)

    if np.sum(k_mask) < 2:
        raise ValueError("Strike window has too few grid points to plot.")

    if spot is None:
        spot = res.get("S0", None) or getattr(day, "S0", None) or getattr(day, "spot", None)

    spot = float(spot) if spot is not None else None

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
        idx = np.unique(np.round(np.linspace(0, T_centers.size - 1, n_panels)).astype(int))
        T_panels = T_centers[idx]

    fig, axes, nrows, ncols = _make_axes(
        len(T_panels),
        panel_shape=None,
        figsize_per_panel=figsize_per_panel,
        sharex=True,
    )

    K_plot = K_grid[k_mask]

    for ax, T_panel in zip(axes[: len(T_panels)], T_panels):
        qmask = np.abs(T_obs - T_panel) <= T_cluster_tol
        fit_idx = int(np.argmin(np.abs(T_grid - T_panel)))

        if np.any(qmask):
            ax.scatter(
                K_obs[qmask],
                C_obs[qmask],
                s=22,
                alpha=0.9,
                label="Observed",
            )

        ax.plot(
            K_plot,
            C_fit[fit_idx, k_mask],
            linewidth=2.2,
            label="Global model",
        )

        if spot is not None and Kmin_plot <= spot <= Kmax_plot:
            ax.axvline(spot, linestyle="--", linewidth=1.5, label="Spot")

        d = f"{date_str} " if date_str else ""
        ax.set_title(f"{d}T ≈ {T_panel:.5g} yr")
        ax.set_ylabel("Call price")
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)
        ax.set_xlim(Kmin_plot, Kmax_plot)

    axes[len(T_panels) - 1].set_xlabel("Strike K")

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

    # ---- ADD THESE BACK ----
    pct_lower: Optional[float] = None,
    pct_upper: Optional[float] = None,
    mark_percentiles: bool = True,
    pct_line_style: str = ":",
    pct_line_width: float = 2.0,
):
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
        pdf = np.clip(np.asarray(pdf, float), 0.0, np.inf)

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
    x_axis = str(x_axis).lower()

    x = _get_grid(res, x_axis)
    qsurf = _get_rnd_surface(res, x_axis)
    T_grid = np.asarray(res["T_grid"], float).ravel()

    if qsurf.shape != (T_grid.size, x.size):
        raise ValueError(f"RND surface has shape {qsurf.shape}, expected {(T_grid.size, x.size)}.")

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

    xlabel, ylabel = _axis_labels(x_axis)
    qL = _to_prob(pct_lower)
    qU = _to_prob(pct_upper)
    
    want_pct = (
        bool(mark_percentiles)
        and ((qL is not None) or (qU is not None))
    )

    for ax, j in zip(axes[:n_actual], idxT):
        ax.plot(x_plot, qsurf[j, xmask], linewidth=2.2, label="RND")
        if want_pct:

            pdf_row = qsurf[j, xmask]

            if qL is not None:
                xL = _quantile_from_pdf(x_plot, pdf_row, qL)

                if np.isfinite(xL):
                    ax.axvline(
                        xL,
                        linestyle=pct_line_style,
                        linewidth=pct_line_width,
                        color="tab:red",
                        label=f"p{int(round(qL * 100))}",
                    )

            if qU is not None:
                xU = _quantile_from_pdf(x_plot, pdf_row, qU)

                if np.isfinite(xU):
                    ax.axvline(
                        xU,
                        linestyle=pct_line_style,
                        linewidth=pct_line_width,
                        color="tab:green",
                        label=f"p{int(round(qU * 100))}",
                    )

        if show_spot:
            if x_axis == "lr" and x_plot.min() <= 0 <= x_plot.max():
                ax.axvline(0.0, linestyle="--", linewidth=1.3, label="Spot")
            elif x_axis == "r" and x_plot.min() <= 1 <= x_plot.max():
                ax.axvline(1.0, linestyle="--", linewidth=1.3, label="Spot")
            elif x_axis == "k":
                spot = res.get("S0", None)
                if spot is not None:
                    spot = float(spot)
                    if x_plot.min() <= spot <= x_plot.max():
                        ax.axvline(spot, linestyle="--", linewidth=1.3, label="Spot")

        d = f"{date_str} " if date_str else ""
        ax.set_title(f"{d}T = {float(T_grid[j]):.5g} yr")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
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