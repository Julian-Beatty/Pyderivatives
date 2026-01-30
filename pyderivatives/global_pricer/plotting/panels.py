from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Literal


# -------------------------
# small helpers (OK to keep)
# -------------------------
def _pick_panel_T_indices(T_grid: np.ndarray, n_panels: int) -> np.ndarray:
    T_grid = np.asarray(T_grid, float).ravel()
    n = T_grid.size
    if n == 0:
        return np.array([], dtype=int)
    if n_panels <= 0:
        return np.arange(n, dtype=int)

    k = min(int(n_panels), n)
    idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    return idx


def _make_panel_axes(n_panels: int, *, base_w: float = 4.5, base_h: float = 3.6) -> Tuple[plt.Figure, np.ndarray]:
    """
    Creates a near-square grid of subplots for n_panels panels.
    Returns (fig, axes_flat).
    """
    n = int(n_panels)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(base_w * ncols, base_h * nrows), squeeze=False)
    return fig, axes.ravel()

def _pick_panel_indices(T_grid: np.ndarray, n_panels: int) -> np.ndarray:
    T = np.asarray(T_grid, float).ravel()
    n = int(max(1, n_panels))
    if T.size == 0:
        return np.array([], dtype=int)
    if T.size <= n:
        return np.arange(T.size, dtype=int)
    idx = np.linspace(0, T.size - 1, n)
    return np.unique(np.round(idx).astype(int))


# -------------------------
# 1) Calls: observed vs fitted (your original)
# -------------------------
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
    # ---- NEW ----
    K_pad_frac: float = 0.05,          # 5% of observed strike range on each side
    K_pad_abs: float = 0.0,            # optional absolute padding (same units as K)
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
):
    K_grid = np.asarray(res["K_grid"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()
    C_fit  = np.asarray(res["C_fit"], float)

    if C_fit.shape != (T_grid.size, K_grid.size):
        raise ValueError("res['C_fit'] must have shape (len(T_grid), len(K_grid)).")

    # observed
    K_obs = np.asarray(day.K_obs, float).ravel()
    T_obs = np.asarray(day.T_obs, float).ravel()
    C_obs = np.asarray(day.C_obs, float).ravel()
    m = (
        np.isfinite(K_obs) & np.isfinite(T_obs) & np.isfinite(C_obs)
        & (K_obs > 0) & (T_obs >= 0) & (C_obs >= 0)
    )
    K_obs, T_obs, C_obs = K_obs[m], T_obs[m], C_obs[m]
    if K_obs.size == 0:
        raise ValueError("No valid observed quotes to plot.")

    # shared x-range (observed)
    Kmin_obs = float(np.nanmin(K_obs))
    Kmax_obs = float(np.nanmax(K_obs))

    # ---- NEW: padded plot window ----
    obs_range = max(Kmax_obs - Kmin_obs, 1e-12)
    pad = float(K_pad_abs) + float(K_pad_frac) * obs_range
    Kmin_plot = Kmin_obs - pad
    Kmax_plot = Kmax_obs + pad

    # also clip to available fitted grid so mask isn't empty
    Kmin_plot = max(Kmin_plot, float(np.nanmin(K_grid)))
    Kmax_plot = min(Kmax_plot, float(np.nanmax(K_grid)))

    # spot
    if spot is None:
        spot = res.get("s0", None) or getattr(day, "S0", None) or getattr(day, "spot", None)
    spot = float(spot) if spot is not None else None

    # ---- cluster observed maturities
    T_sorted = np.sort(np.unique(T_obs))
    clusters, cur = [], [T_sorted[0]]
    for t in T_sorted[1:]:
        if abs(t - cur[-1]) <= T_cluster_tol:
            cur.append(t)
        else:
            clusters.append(float(np.mean(cur)))
            cur = [t]
    clusters.append(float(np.mean(cur)))
    T_centers = np.array(clusters, float)

    # pick panels
    if T_centers.size <= n_panels:
        T_panels = T_centers
    else:
        idx = np.unique(np.round(np.linspace(0, T_centers.size - 1, n_panels)).astype(int))
        T_panels = T_centers[idx]

    nrows = T_panels.size
    fig_h = max(4.0, figsize_per_panel * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8.5, fig_h), sharex=True)
    if nrows == 1:
        axes = np.array([axes])

    # ---- NEW: restrict fitted line to padded strike window (not observed window) ----
    k_mask = (K_grid >= Kmin_plot) & (K_grid <= Kmax_plot)
    K_grid_plot = K_grid[k_mask]
    if K_grid_plot.size < 2:
        raise ValueError("Padded strike window has too few points in K_grid to plot.")

    for ax, T_panel in zip(axes, T_panels):
        qmask = np.abs(T_obs - T_panel) <= T_cluster_tol
        fit_idx = int(np.argmin(np.abs(T_grid - T_panel)))

        if np.any(qmask):
            ax.scatter(
                K_obs[qmask], C_obs[qmask],
                s=22, alpha=0.9, color="tab:blue", label="Observed"
            )

        ax.plot(
            K_grid_plot, C_fit[fit_idx, k_mask],
            linewidth=2.2, color="tab:blue", label="Global model (fit surface)"
        )

        if spot is not None:
            ax.axvline(spot, linestyle="--", linewidth=1.5, color="black", label="Spot")

        d = f"{date_str} " if date_str else ""
        ax.set_title(f"{d}(T ≈ {T_panel:.5g} yr)")
        ax.set_ylabel("Call price")
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)

        # ---- NEW: xlim uses padded window ----
        ax.set_xlim(Kmin_plot, Kmax_plot)

    axes[-1].set_xlabel("Strike K")
    fig.suptitle(title)
    fig.tight_layout()

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")

    plt.show()
    return fig



import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional



# -------------------------
# 2) IV panels
# -------------------------
def iv_panels(
    res: dict,
    *,
    n_panels: int = 6,
    title: str = "IV Panels",
    # ---- layout ----
    panel_shape: tuple[int, int] | None = None,   # e.g. (3,2); default keeps old behavior
    save: str | None = None,
    dpi: int = 300,
    show: bool = True,
    # ---- NEW: x-axis / bounds ----
    x_axis: str = "K",                            # {"K","R","r"} strike / gross return / log return
    x_bounds: tuple[float, float] | None = None,  # bounds in chosen x-axis units
    spot: float | None = None,                    # S0; required for x_axis in {"R","r"} unless in res
):
    """
    Panel plots of IV across maturities.

    Requires:
      res["iv_surface"] OR res["vol_surface"]
      res["K_grid"], res["T_grid"]

    x_axis:
      "K": x = K (strike)
      "R": x = R = K/S0 (gross return / moneyness)
      "r": x = r = log(K/S0) (log return)

    x_bounds are applied in the chosen x units.

    Note: For x_axis in {"R","r"} spot/S0 must be available (passed or in res).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # ---- load surface ----
    if "iv_surface" in res:
        iv = np.asarray(res["iv_surface"], float)
    elif "vol_surface" in res:
        iv = np.asarray(res["vol_surface"], float)
    else:
        raise KeyError("Missing iv surface. Expected res['iv_surface'] or res['vol_surface'].")

    K_grid = np.asarray(res["K_grid"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()

    if iv.shape != (T_grid.size, K_grid.size):
        raise ValueError("IV surface must have shape (len(T_grid), len(K_grid)).")

    # ---- spot/S0 ----
    if spot is None:
        spot = res.get("S0", None) or res.get("s0", None)
    spot = float(spot) if spot is not None else None

    x_axis = str(x_axis).strip()
    if x_axis not in {"K", "R", "r"}:
        raise ValueError("x_axis must be one of {'K','R','r'}.")

    if x_axis in {"R", "r"} and spot is None:
        raise ValueError("spot/S0 is required when x_axis is 'R' or 'r'.")

    # -------------------------
    # pick maturities
    # -------------------------
    idxT = _pick_panel_T_indices(T_grid, n_panels)
    n_panels_actual = int(idxT.size)

    # -------------------------
    # NEW: x-mask from bounds
    # -------------------------
    if x_bounds is None:
        k_mask = np.isfinite(K_grid)
    else:
        lo, hi = float(x_bounds[0]), float(x_bounds[1])
        if lo >= hi:
            raise ValueError("x_bounds must be (lo, hi) with lo < hi.")

        if x_axis == "K":
            k_mask = (K_grid >= lo) & (K_grid <= hi)
        elif x_axis == "R":
            k_mask = (K_grid >= lo * spot) & (K_grid <= hi * spot)
        else:  # x_axis == "r"
            k_mask = (K_grid >= spot * np.exp(lo)) & (K_grid <= spot * np.exp(hi))

    if not np.any(k_mask):
        raise ValueError("x_bounds produced an empty plotting window on K_grid.")

    K_plot = K_grid[k_mask]

    # ---- x for plotting ----
    if x_axis == "K":
        x_plot = K_plot
        xlabel = "Strike K"
    elif x_axis == "R":
        x_plot = K_plot / spot
        xlabel = "Gross return R = K/S0"
    else:
        x_plot = np.log(K_plot / spot)
        xlabel = "Log return r = log(K/S0)"

    # -------------------------
    # panel layout
    # -------------------------
    if panel_shape is None:
        # original behavior
        fig, axes = _make_panel_axes(n_panels_actual)
        axes = np.atleast_1d(axes)
    else:
        nrows, ncols = int(panel_shape[0]), int(panel_shape[1])
        if nrows * ncols < n_panels_actual:
            raise ValueError(
                f"panel_shape={panel_shape} has {nrows*ncols} slots, but need {n_panels_actual}."
            )

        fig_h = max(4.0, 2.2 * nrows)
        fig_w = 8.5 * ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True)
        axes = np.atleast_1d(axes).ravel()

    # -------------------------
    # plot
    # -------------------------
    for ax, j in zip(axes[:n_panels_actual], idxT):
        ax.plot(x_plot, iv[j, k_mask], linewidth=2.0)
        ax.set_title(f"T = {float(T_grid[j]):.4g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Implied vol")
        ax.grid(True, alpha=0.25)

        # x-limits in chosen units
        ax.set_xlim(float(x_plot.min()), float(x_plot.max()))

    # Turn off unused panels
    for ax in axes[n_panels_actual:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig





# -------------------------
# 3) RND panels
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from pathlib import Path
def rnd_panels(
    res: dict,
    *,
    n_panels: int = 6,
    title: str = "RND Panels",
    date_str: Optional[str] = None,
    day: Optional[object] = None,
    xlim_obs: bool = True,
    spot: Optional[float] = None,
    show_spot: bool = True,
    spot_label: str = "Spot",
    # --- layout ---
    panel_shape: tuple[int, int] | None = None,   # e.g. (3,2); default => stacked (n,1)
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    legend_loc: str = "upper right",
    figsize_per_panel: float = 2.2,
    # ---- NEW: x-axis / bounds ----
    x_axis: str = "K",                            # {"K","R","r"}  strike / gross return / log return
    x_bounds: tuple[float, float] | None = None,  # bounds in chosen x_axis units
):
    """
    Plot RND panels with optional x-axis transform:

    x_axis="K": x = K (strike)
    x_axis="R": x = R = K / S0 (gross return / moneyness)
    x_axis="r": x = r = log(K / S0) (log return)

    Notes:
    - When using x_axis in {"R","r"}, spot (S0) must be available.
    - This function plots the same y-values (q on K-grid). It does not apply Jacobian
      to convert to a density in R or r. If you want a true density on R or r, say so
      and I’ll add the Jacobian option.
    """
    if "rnd_surface" not in res:
        raise KeyError("Missing RND. Expected res['rnd_surface'].")

    q = np.asarray(res["rnd_surface"], float)
    K_grid = np.asarray(res["K_grid"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()

    if q.shape != (T_grid.size, K_grid.size):
        raise ValueError("RND surface must have shape (len(T_grid), len(K_grid)).")
    if K_grid.size < 3:
        raise ValueError("K_grid too small to plot meaningfully.")

    # ----- resolve spot -----
    if spot is None:
        spot = (
            res.get("S0", None)
            or res.get("s0", None)
            or getattr(day, "S0", None)
            or getattr(day, "spot", None)
        )
    spot = float(spot) if spot is not None else None

    x_axis = str(x_axis).strip()
    if x_axis not in {"K", "R", "r"}:
        raise ValueError("x_axis must be one of {'K','R','r'}.")

    if x_axis in {"R", "r"} and spot is None:
        raise ValueError("spot/S0 is required when x_axis is 'R' or 'r'.")

    # ----- choose maturities -----
    n_pan = int(min(max(n_panels, 1), T_grid.size))
    idxT = np.unique(np.round(np.linspace(0, T_grid.size - 1, n_pan)).astype(int))
    n_panels_actual = int(idxT.size)

    # ----- determine default K-window (from observed strikes if requested) -----
    if (day is not None) and xlim_obs:
        K_obs = np.asarray(getattr(day, "K_obs"), float).ravel()
        K_obs = K_obs[np.isfinite(K_obs) & (K_obs > 0)]
        if K_obs.size > 0:
            Kmin0, Kmax0 = float(np.nanmin(K_obs)), float(np.nanmax(K_obs))
        else:
            Kmin0, Kmax0 = float(K_grid.min()), float(K_grid.max())
    else:
        Kmin0, Kmax0 = float(K_grid.min()), float(K_grid.max())

    # ----- map bounds in chosen axis back to a K-mask -----
    # We always mask on K_grid indices, then compute x for plotting.
    if x_bounds is None:
        # use default K-window
        k_mask = (K_grid >= Kmin0) & (K_grid <= Kmax0)
    else:
        lo, hi = float(x_bounds[0]), float(x_bounds[1])
        if lo >= hi:
            raise ValueError("x_bounds must be (lo, hi) with lo < hi.")

        if x_axis == "K":
            k_mask = (K_grid >= lo) & (K_grid <= hi)
        elif x_axis == "R":
            # R = K/spot => K = R*spot
            k_mask = (K_grid >= lo * spot) & (K_grid <= hi * spot)
        else:  # x_axis == "r"
            # r = log(K/spot) => K = spot*exp(r)
            k_mask = (K_grid >= spot * np.exp(lo)) & (K_grid <= spot * np.exp(hi))

    # fall back if mask is empty
    if not np.any(k_mask):
        raise ValueError("x_bounds produced an empty plotting window on K_grid.")

    K_plot = K_grid[k_mask]

    # ----- compute x for plotting -----
    if x_axis == "K":
        x_plot = K_plot
        xlabel = "Strike K"
        x_spot = spot
        show_spot_line = bool(show_spot) and (spot is not None) and (x_plot.min() <= x_spot <= x_plot.max())
        spot_line_x = x_spot
    elif x_axis == "R":
        x_plot = K_plot / spot
        xlabel = "Gross return R = K/S0"
        show_spot_line = bool(show_spot)  # spot maps to R=1
        spot_line_x = 1.0
        show_spot_line = show_spot_line and (x_plot.min() <= spot_line_x <= x_plot.max())
    else:  # x_axis == "r"
        x_plot = np.log(K_plot / spot)
        xlabel = "Log return r = log(K/S0)"
        show_spot_line = bool(show_spot)  # spot maps to r=0
        spot_line_x = 0.0
        show_spot_line = show_spot_line and (x_plot.min() <= spot_line_x <= x_plot.max())

    # ----- panel shape -----
    if panel_shape is None:
        nrows, ncols = n_panels_actual, 1
    else:
        nrows, ncols = int(panel_shape[0]), int(panel_shape[1])
        if nrows * ncols < n_panels_actual:
            raise ValueError(f"panel_shape={panel_shape} has {nrows*ncols} slots, but need {n_panels_actual}.")

    fig_h = max(4.0, figsize_per_panel * nrows)
    fig_w = 8.5 if ncols == 1 else 8.5 * ncols / 1.0
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    # ----- plot -----
    for ax, j in zip(axes[:n_panels_actual], idxT):
        ax.plot(x_plot, q[j, k_mask], linewidth=2.2, color="tab:blue", label="RND")

        if show_spot_line:
            ax.axvline(spot_line_x, linestyle="--", linewidth=1.5, color="black", label=spot_label)

        d = f"{date_str} " if date_str else ""
        ax.set_title(f"{d}(T = {float(T_grid[j]):.5g} yr)")
        ax.set_ylabel("q(K|T)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)

        # apply x-limits directly in chosen axis units
        ax.set_xlim(float(x_plot.min()), float(x_plot.max()))

    # turn off unused axes
    for ax in axes[n_panels_actual:]:
        ax.axis("off")

    # x-label on bottom row
    for ax in axes[(nrows - 1) * ncols : nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel(xlabel)

    fig.suptitle(title)
    fig.tight_layout()

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")

    plt.show()
    return fig
def cdf_panels(
    res: dict,
    *,
    n_panels: int = 6,
    title: str = "CDF Panels",
    date_str: Optional[str] = None,
    day: Optional[object] = None,
    xlim_obs: bool = True,
    # ---- NEW ----
    panel_shape: tuple[int, int] | None = None,   # e.g. (3,2); default => stacked (n,1)
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    legend_loc: str = "upper right",
    figsize_per_panel: float = 2.2,
):
    if "cdf_surface" not in res:
        raise KeyError("Missing CDF. Expected res['cdf_surface'].")

    F = np.asarray(res["cdf_surface"], float)
    K_grid = np.asarray(res["K_grid"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()

    if F.shape != (T_grid.size, K_grid.size):
        raise ValueError("CDF surface must have shape (len(T_grid), len(K_grid)).")
    if K_grid.size < 3:
        raise ValueError("K_grid too small to plot meaningfully.")

    n_pan = int(min(max(n_panels, 1), T_grid.size))
    idxT = np.unique(np.round(np.linspace(0, T_grid.size - 1, n_pan)).astype(int))

    if (day is not None) and xlim_obs:
        K_obs = np.asarray(getattr(day, "K_obs"), float).ravel()
        K_obs = K_obs[np.isfinite(K_obs) & (K_obs > 0)]
        if K_obs.size > 0:
            Kmin, Kmax = float(np.nanmin(K_obs)), float(np.nanmax(K_obs))
        else:
            Kmin, Kmax = float(K_grid.min()), float(K_grid.max())
    else:
        Kmin, Kmax = float(K_grid.min()), float(K_grid.max())

    k_mask = (K_grid >= Kmin) & (K_grid <= Kmax)
    K_plot = K_grid[k_mask]

    n_panels_actual = int(idxT.size)

    if panel_shape is None:
        nrows, ncols = n_panels_actual, 1
    else:
        nrows, ncols = int(panel_shape[0]), int(panel_shape[1])
        if nrows * ncols < n_panels_actual:
            raise ValueError(f"panel_shape={panel_shape} has {nrows*ncols} slots, but need {n_panels_actual}.")

    fig_h = max(4.0, figsize_per_panel * nrows)
    fig_w = 8.5 if ncols == 1 else 8.5 * ncols / 1.0
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, j in zip(axes[:n_panels_actual], idxT):
        ax.plot(K_plot, F[j, k_mask], linewidth=2.2, color="tab:blue", label="CDF")
        d = f"{date_str} " if date_str else ""
        ax.set_title(f"{d}(T = {float(T_grid[j]):.5g} yr)")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc=legend_loc)
        ax.set_xlim(Kmin, Kmax)

    for ax in axes[n_panels_actual:]:
        ax.axis("off")

    for ax in axes[(nrows-1)*ncols : nrows*ncols]:
        if ax.has_data():
            ax.set_xlabel("Strike K")

    fig.suptitle(title)
    fig.tight_layout()

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")

    plt.show()
    return fig
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
    # ---- NEW ----
    panel_shape: tuple[int, int] | None = None,   # e.g. (3,2); default => stacked (n,1)
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
):
    if "delta_dict" not in res or res["delta_dict"] is None:
        raise KeyError("Missing delta_dict. Expected res['delta_dict'].")

    d = res["delta_dict"]
    delta = np.asarray(d["delta_axis"], float).ravel()
    T = np.asarray(d["T_axis"], float).ravel()

    if which == "call":
        Z_key = "iv_delta_call"
        ylab = "IV (call delta)"
        line_label = "IV call"
        line_color = "black"
    elif which == "put":
        Z_key = "iv_delta_put_abs"
        ylab = "IV (|put delta|)"
        line_label = "IV put"
        line_color = "black"
    elif which == "skew":
        Z_key = "delta_skew_surface"
        ylab = "Normalized Delta skew"
        line_label = "Skew"
        line_color = "black"
    else:
        raise ValueError("which must be one of {'skew', 'call', 'put'}.")

    if Z_key not in d:
        raise KeyError(f"delta_dict is missing '{Z_key}'.")

    Z = np.asarray(d[Z_key], float)
    if Z.shape != (T.size, delta.size):
        raise ValueError(f"Surface {Z_key} has shape {Z.shape} but expected {(T.size, delta.size)}.")

    row_ok = np.array([np.any(np.isfinite(Z[i, :])) for i in range(T.size)], dtype=bool)
    valid_rows = np.where(row_ok)[0]
    if valid_rows.size == 0:
        raise ValueError("No finite rows to plot in the requested surface.")

    if valid_rows.size <= n_panels:
        idxT = valid_rows
    else:
        idx = np.unique(np.round(np.linspace(0, valid_rows.size - 1, n_panels)).astype(int))
        idxT = valid_rows[idx]

    n_panels_actual = int(idxT.size)

    if panel_shape is None:
        nrows, ncols = n_panels_actual, 1
    else:
        nrows, ncols = int(panel_shape[0]), int(panel_shape[1])
        if nrows * ncols < n_panels_actual:
            raise ValueError(f"panel_shape={panel_shape} has {nrows*ncols} slots, but need {n_panels_actual}.")

    fig_h = max(4.0, figsize_per_panel * nrows)
    fig_w = 8.5 if ncols == 1 else 8.5 * ncols / 1.0
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    spot_delta = float(spot) if spot is not None else None

    for ax, j in zip(axes[:n_panels_actual], idxT):
        y = Z[j, :]
        m = np.isfinite(delta) & np.isfinite(y)
        if np.sum(m) < 2:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.grid(True, alpha=0.25)
            continue

        ax.plot(delta[m], y[m], linewidth=2.2, color=line_color, label=line_label)

        if spot_delta is not None:
            ax.axvline(spot_delta, linestyle="--", linewidth=1.5, color="black", label="Spot delta")

        dstr = f"{date_str} " if date_str else ""
        ax.set_title(f"{dstr}T = {float(T[j]):.5g} yr")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)

    for ax in axes[n_panels_actual:]:
        ax.axis("off")

    for ax in axes[(nrows-1)*ncols : nrows*ncols]:
        if ax.has_data():
            ax.set_xlabel("Delta")

    fig.suptitle(title)
    fig.tight_layout()

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        if save.suffix == "":
            save = save.with_suffix(".png")
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")

    plt.show()
    return fig
