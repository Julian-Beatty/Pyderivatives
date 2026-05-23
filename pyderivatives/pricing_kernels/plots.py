from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Literal

import numpy as np


# ============================================================
# Core utilities
# ============================================================

def _as_1d(x) -> np.ndarray:
    """Convert input to a flattened float array."""
    return np.asarray(x, dtype=float).ravel()


def _maybe_save(fig, save=None, dpi: int = 200) -> None:
    """Save a matplotlib figure if a path is supplied."""
    if save is None:
        return

    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=int(dpi), bbox_inches="tight")
    print(f"[saved] {save}")


def _get_meta_value(result: dict, *keys, default=None):
    """Look for metadata at the top level first, then inside result['meta'].""" 
    meta = result.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    for key in keys:
        if key in result and result[key] is not None:
            return result[key]
        if key in meta and meta[key] is not None:
            return meta[key]

    return default


def _title_suffix(result: dict) -> str:
    """Create a useful suffix such as ticker/date for titles."""
    ticker = _get_meta_value(result, "ticker", default="")
    date = _get_meta_value(
        result,
        "date",
        "valuation_date",
        "anchor_key_used",
        "anchor_date_used",
        default="",
    )

    pieces = [str(x) for x in (ticker, date) if x not in ("", None)]
    return " ".join(pieces)


def _standardize_x_axis(x_axis: str) -> str:
    """
    Canonicalize the x-axis choice.

    Returns:
        "r"      log return
        "R"      gross return
        "return" simple return R - 1
        "K"      strike / terminal price
    """
    key = str(x_axis).strip()

    aliases = {
        "r": "r",
        "log": "r",
        "lr": "r",
        "log_return": "r",
        "log-return": "r",

        "R": "R",
        "gross": "R",
        "gross_return": "R",

        "return": "return",
        "simple": "return",
        "simple_return": "return",

        "K": "K",
        "k": "K",
        "strike": "K",
        "terminal": "K",
        "terminal_price": "K",
        "ST": "K",
        "st": "K",
    }

    if key not in aliases:
        raise ValueError("x_axis must be one of {'r', 'R', 'return', 'K'}.")

    return aliases[key]


def _get_plot_x_grid(result: dict, x_axis: str = "R"):
    """
    Return the requested x-grid and an axis label.
    """
    x_axis = _standardize_x_axis(x_axis)

    if x_axis == "r":
        return _as_1d(result["grid_lr"]), r"Log return $r=\log(S_T/S_0)$"

    if x_axis == "R":
        return _as_1d(result["grid_r"]), r"Gross return $R=S_T/S_0$"

    if x_axis == "return":
        return _as_1d(result["grid_r"]) - 1.0, r"Simple return $R-1$"

    if x_axis == "K":
        return _as_1d(result["grid_k"]), r"Terminal price / strike $K=S_T$"

    raise RuntimeError("Unreachable x_axis.")


def _surface_key(kind: str, x_axis: str) -> tuple[str, str, str]:
    """
    Map a surface type and x-axis to the correct result dictionary key.

    Returns:
        surface_key, zlabel, default_title
    """
    kind = str(kind).lower().strip()
    x_axis = _standardize_x_axis(x_axis)

    suffix = {
        "r": "lr",
        "R": "r",
        "return": "r",
        "K": "k",
    }[x_axis]

    if kind in {"physical", "p"}:
        return (
            f"physical_{suffix}_surface",
            {
                "r": r"$f_P(r)$",
                "R": r"$p_P(R)$",
                "return": r"$p_P(R)$",
                "K": r"$p_P(K)$",
            }[x_axis],
            "Physical Density Surface",
        )

    if kind in {"rnd", "risk_neutral", "risk-neutral", "q"}:
        return (
            f"rnd_{suffix}_surface",
            {
                "r": r"$f_Q(r)$",
                "R": r"$p_Q(R)$",
                "return": r"$p_Q(R)$",
                "K": r"$p_Q(K)$",
            }[x_axis],
            "Risk-Neutral Density Surface",
        )

    if kind in {"pricing_kernel", "kernel", "m"}:
        return (
            "pricing_kernel_surface",
            r"$M$",
            "Pricing Kernel Surface",
        )

    if kind in {"rra", "relative_risk_aversion", "risk_aversion"}:
        return (
            "relative_risk_aversion_surface",
            "Relative risk aversion",
            "Relative Risk Aversion Surface",
        )

    raise ValueError("kind must be one of {'physical', 'rnd', 'pricing_kernel', 'rra'}.")


def _cdf_key(kind: str, x_axis: str) -> str:
    """
    Return the CDF key corresponding to a density surface.
    """
    kind = str(kind).lower().strip()
    x_axis = _standardize_x_axis(x_axis)

    suffix = {
        "r": "lr",
        "R": "r",
        "return": "r",
        "K": "k",
    }[x_axis]

    if kind in {"physical", "p"}:
        return f"physical_cdf_{suffix}_surface"

    if kind in {"rnd", "risk_neutral", "risk-neutral", "q"}:
        return f"cdf_{suffix}_surface"

    raise ValueError("CDF is only available for physical or risk-neutral density.")


def _cdf_from_density(x, f, eps: float = 1e-14) -> np.ndarray:
    """
    Compute a normalized CDF from a density using trapezoidal integration.
    """
    x = _as_1d(x)
    f = _as_1d(f)

    out = np.full_like(x, np.nan, dtype=float)

    good = np.isfinite(x) & np.isfinite(f) & (f >= 0)
    if good.sum() < 3:
        return out

    xx = x[good]
    ff = f[good]

    order = np.argsort(xx)
    xx = xx[order]
    ff = ff[order]

    inc = 0.5 * (ff[1:] + ff[:-1]) * np.diff(xx)

    cdf = np.empty_like(xx)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(inc)

    total = cdf[-1]
    if not np.isfinite(total) or total <= eps:
        return out

    cdf = cdf / total
    out[good] = np.interp(x[good], xx, cdf, left=np.nan, right=np.nan)

    return out


def _get_cdf_surface(result: dict, *, kind: str, x_axis: str) -> np.ndarray:
    """
    Fetch a stored CDF surface, or compute it row-by-row from the density surface.
    """
    cdf_key = _cdf_key(kind, x_axis)

    if cdf_key in result:
        return np.asarray(result[cdf_key], dtype=float)

    density_key, _, _ = _surface_key(kind, x_axis)

    if density_key not in result:
        raise KeyError(f"Need result['{cdf_key}'] or result['{density_key}'].")

    x_grid, _ = _get_plot_x_grid(result, x_axis=x_axis)
    density = np.asarray(result[density_key], dtype=float)

    return np.vstack([
        _cdf_from_density(x_grid, density[j, :])
        for j in range(density.shape[0])
    ])


def _slice_surface_bounds(
    T_grid,
    x_grid,
    Z,
    *,
    T_bounds=None,
    x_bounds=None,
):
    """
    Slice a surface by maturity and x-axis bounds.

    Assumes:
        Z.shape == (len(T_grid), len(x_grid))
    """
    T_grid = _as_1d(T_grid)
    x_grid = _as_1d(x_grid)
    Z = np.asarray(Z, dtype=float)

    if Z.shape != (T_grid.size, x_grid.size):
        raise ValueError(
            f"Surface shape mismatch. Expected {(T_grid.size, x_grid.size)}, got {Z.shape}."
        )

    tmask = np.isfinite(T_grid)
    xmask = np.isfinite(x_grid)

    if T_bounds is not None:
        lo, hi = sorted(map(float, T_bounds))
        tmask &= (T_grid >= lo) & (T_grid <= hi)

    if x_bounds is not None:
        lo, hi = sorted(map(float, x_bounds))
        xmask &= (x_grid >= lo) & (x_grid <= hi)

    if not np.any(tmask):
        raise ValueError("T_bounds produced an empty maturity grid.")

    if not np.any(xmask):
        raise ValueError("x_bounds produced an empty x-grid.")

    return T_grid[tmask], x_grid[xmask], Z[np.ix_(tmask, xmask)], tmask, xmask


def _pick_panel_indices(T_grid, n_panels: int) -> np.ndarray:
    """
    Pick approximately evenly spaced maturity indices.
    """
    T_grid = _as_1d(T_grid)

    if T_grid.size == 0:
        raise ValueError("Cannot select panels from an empty T_grid.")

    n_panels = int(min(max(1, n_panels), T_grid.size))
    return np.unique(np.linspace(0, T_grid.size - 1, n_panels).round().astype(int))


def _quantiles_from_cdf(X, F, probs):
    """
    Convert a CDF row into x-axis quantile locations.
    """
    X = np.asarray(X, float)
    F = np.asarray(F, float)
    probs = np.asarray(probs, float)

    good = np.isfinite(X) & np.isfinite(F)

    if good.sum() < 3:
        return np.full(probs.shape, np.nan)

    Xg = X[good]
    Fg = F[good]

    order = np.argsort(Xg)
    Xg = Xg[order]
    Fg = Fg[order]

    Fg = np.maximum.accumulate(Fg)

    unique_F, unique_idx = np.unique(Fg, return_index=True)
    unique_X = Xg[unique_idx]

    return np.interp(probs, unique_F, unique_X, left=np.nan, right=np.nan)


# ============================================================
# Generic 3D surface plot
# ============================================================

def plot_surface(
    result: dict,
    *,
    kind: str,
    x_axis: str = "R",
    T_bounds: Optional[Tuple[float, float]] = None,
    x_bounds: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    elev: float = 28,
    azim: float = -60,
    alpha: float = 0.9,
    log_z: bool = False,
    truncate_by_alpha: bool = False,
    ptail_alpha: Tuple[float, float] = (0.05, 0.05),
    truncation_measure: str = "physical",
    min_points_per_row: int = 5,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    show: bool = True,
):
    """
    Generic surface plot for physical density, RND, pricing kernel, or RRA.

    Parameters
    ----------
    kind:
        'physical', 'rnd', 'pricing_kernel', or 'rra'.

    truncate_by_alpha:
        If True, masks the surface outside the inner probability mass
        of either the physical or risk-neutral CDF.
    """
    import matplotlib.pyplot as plt

    T_grid = _as_1d(result["T_grid"])
    x_grid, xlabel = _get_plot_x_grid(result, x_axis=x_axis)

    surface_key, zlabel_base, default_title = _surface_key(kind, x_axis)

    if surface_key not in result:
        raise KeyError(f"Expected result['{surface_key}'].")

    Z = np.asarray(result[surface_key], dtype=float).copy()

    if Z.shape != (T_grid.size, x_grid.size):
        raise ValueError(
            f"{surface_key} must have shape (len(T_grid), len(x_grid))."
        )

    measure_label = None

    if truncate_by_alpha:
        a_left, a_right = map(float, ptail_alpha)

        if not (
            0.0 <= a_left < 1.0
            and 0.0 <= a_right < 1.0
            and a_left + a_right < 1.0
        ):
            raise ValueError("ptail_alpha must satisfy 0 <= left,right < 1 and left+right < 1.")

        truncation_measure = str(truncation_measure).lower().strip()

        if truncation_measure in {"physical", "p"}:
            F_trunc = _get_cdf_surface(result, kind="physical", x_axis=x_axis)
            measure_label = "physical"
        elif truncation_measure in {"risk_neutral", "risk-neutral", "q", "rnd"}:
            F_trunc = _get_cdf_surface(result, kind="rnd", x_axis=x_axis)
            measure_label = "risk-neutral"
        else:
            raise ValueError("truncation_measure must be 'physical' or 'risk_neutral'.")

        if F_trunc.shape != Z.shape:
            raise ValueError("Truncation CDF surface must have the same shape as the plotted surface.")

        keep = (F_trunc >= a_left) & (F_trunc <= 1.0 - a_right)

        for j in range(Z.shape[0]):
            if int(np.sum(keep[j, :])) >= int(min_points_per_row):
                Z[j, ~keep[j, :]] = np.nan

    T_plot, x_plot, Z_plot, _, _ = _slice_surface_bounds(
        T_grid,
        x_grid,
        Z,
        T_bounds=T_bounds,
        x_bounds=x_bounds,
    )

    if T_plot.size < 2 or x_plot.size < 2:
        raise ValueError("Not enough points to plot after bounds/truncation.")

    if log_z:
        Z_plot = np.where(np.isfinite(Z_plot) & (Z_plot > 0), np.log(Z_plot), np.nan)
        zlabel = rf"$\log$ {zlabel_base}"
    else:
        zlabel = zlabel_base

    X, T = np.meshgrid(x_plot, T_plot)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        T,
        Z_plot,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=alpha,
    )

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label=zlabel)

    if title is None:
        title = default_title

    suffix = _title_suffix(result)
    if suffix:
        title = f"{title} — {suffix}"

    if truncate_by_alpha:
        keep_pct = 100.0 * (1.0 - float(ptail_alpha[0]) - float(ptail_alpha[1]))
        title = f"{title} — inner {keep_pct:.0f}% by {measure_label} CDF"

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maturity $T$ (years)")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    _maybe_save(fig, save, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, surf


def plot_pricing_kernel_surface(result: dict, **kwargs):
    """3D pricing-kernel surface."""
    return plot_surface(result, kind="pricing_kernel", **kwargs)


def plot_physical_density_surface(result: dict, **kwargs):
    """3D physical-density surface."""
    return plot_surface(result, kind="physical", **kwargs)


def plot_rnd_surface(result: dict, **kwargs):
    """3D risk-neutral-density surface."""
    return plot_surface(result, kind="rnd", **kwargs)


def plot_rra_surface(result: dict, **kwargs):
    """3D relative-risk-aversion surface."""
    return plot_surface(result, kind="rra", **kwargs)


# ============================================================
# Generic maturity panels
# ============================================================

def plot_surface_panels(
    result: dict,
    *,
    kind: str,
    x_axis: str = "R",
    n_panels: int = 6,
    T_bounds: Optional[Tuple[float, float]] = None,
    x_bounds: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    yscale: Literal["linear", "log"] = "linear",
    ylog_eps: float = 1e-300,
    show_ref: bool = True,
    ref_x: Optional[float] = None,
    ref_label: Optional[str] = None,
    truncate_by_alpha: bool = False,
    ptail_alpha: Tuple[float, float] = (0.05, 0.05),
    truncation_measure: str = "physical",
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    show: bool = True,
    legend_loc: str = "best",
    figsize_per_panel: float = 2.4,
):
    """
    Generic stacked maturity panels.

    Works for:
        physical density
        RND
        pricing kernel
        RRA
    """
    import matplotlib.pyplot as plt

    T_grid = _as_1d(result["T_grid"])
    x_grid, xlabel = _get_plot_x_grid(result, x_axis=x_axis)

    surface_key, y_label, default_title = _surface_key(kind, x_axis)

    if surface_key not in result:
        raise KeyError(f"Expected result['{surface_key}'].")

    Z = np.asarray(result[surface_key], dtype=float)

    T2, x2, Z2, tmask, xmask = _slice_surface_bounds(
        T_grid,
        x_grid,
        Z,
        T_bounds=T_bounds,
        x_bounds=x_bounds,
    )

    F2 = None

    if truncate_by_alpha:
        a_left, a_right = map(float, ptail_alpha)

        if truncation_measure.lower() in {"physical", "p"}:
            F = _get_cdf_surface(result, kind="physical", x_axis=x_axis)
        elif truncation_measure.lower() in {"risk_neutral", "risk-neutral", "q", "rnd"}:
            F = _get_cdf_surface(result, kind="rnd", x_axis=x_axis)
        else:
            raise ValueError("truncation_measure must be 'physical' or 'risk_neutral'.")

        F2 = F[np.ix_(tmask, xmask)]

    idxs = _pick_panel_indices(T2, n_panels)
    nrows = len(idxs)

    fig_h = max(4.0, figsize_per_panel * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8.5, fig_h), sharex=True)

    if nrows == 1:
        axes = np.array([axes])

    x_axis_std = _standardize_x_axis(x_axis)

    if ref_x is None:
        if x_axis_std == "r":
            ref_x = 0.0
            ref_label = ref_label or r"$r=0$"
        elif x_axis_std == "R":
            ref_x = 1.0
            ref_label = ref_label or r"$R=1$"
        elif x_axis_std == "return":
            ref_x = 0.0
            ref_label = ref_label or r"$R-1=0$"
        elif x_axis_std == "K":
            ref_x = _get_meta_value(result, "S0", "s0", "spot", "spot_price", default=None)
            ref_label = ref_label or r"$K=S_0$"

    show_ref_eff = (
        show_ref
        and ref_x is not None
        and np.isfinite(float(ref_x))
        and x2[0] <= float(ref_x) <= x2[-1]
    )

    for ax, j in zip(axes, idxs):
        y = Z2[j, :].copy()
        x = x2.copy()

        label = y_label

        if truncate_by_alpha and F2 is not None:
            keep = (
                np.isfinite(F2[j, :])
                & (F2[j, :] >= a_left)
                & (F2[j, :] <= 1.0 - a_right)
            )
            y[~keep] = np.nan
            label += f" inner {100*(1-a_left-a_right):.0f}%"

        if yscale == "log":
            keep = np.isfinite(x) & np.isfinite(y) & (y > 0)
            x = x[keep]
            y = np.maximum(y[keep], ylog_eps)
        else:
            keep = np.isfinite(x) & np.isfinite(y)
            x = x[keep]
            y = y[keep]

        ax.plot(x, y, linewidth=2.1, label=label)

        if show_ref_eff:
            ax.axvline(
                float(ref_x),
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
                label=ref_label,
            )

        ax.set_title(f"T ≈ {365 * T2[j]:.1f}d")
        ax.set_ylabel(y_label)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)

    axes[-1].set_xlabel(xlabel)

    if title is None:
        title = default_title.replace("Surface", "Panels")

    suffix = _title_suffix(result)
    if suffix:
        title = f"{title} — {suffix}"

    fig.suptitle(title)
    fig.tight_layout()

    _maybe_save(fig, save, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_surface_3d_by_T(
    result_dict: dict,
    T_target: float,
    *,
    kind: str = "pricing_kernel",          # {"pricing_kernel", "physical", "rnd", "rra"}
    x_axis: str = "r",                     # {"r", "R", "return", "K"}
    x_bounds: tuple[float, float] | None = None,
    max_dates: int = 250,
    stride: int | None = None,
    zscale: str = "log",                   # {"linear", "log"}
    z_eps: float = 1e-300,
    title: str | None = None,
    cmap: str = "viridis",
    dpi: int = 200,
    interactive: bool = False,
    show: bool = True,
    alpha: float = 0.9,
    elev: float = 28,
    azim: float = -60,
    truncate_by_ptails: bool = False,
    ptail_alphas: tuple[float, float] = (0.01, 0.01),
    truncation_measure: str = "physical",  # {"physical", "rnd"}
    save=None,
    ax=None,
    add_colorbar: bool = True,
    clear_ax: bool = True,
):
    """
    Plot a 3D surface through calendar time at a fixed target maturity.

    New-interface dictionary expected for each date:
        result["T_grid"]
        result["grid_lr"]
        result["grid_r"]
        result["grid_k"]

        result["pricing_kernel_surface"]
        result["relative_risk_aversion_surface"]

        result["physical_lr_surface"], result["physical_r_surface"], result["physical_k_surface"]
        result["rnd_lr_surface"],      result["rnd_r_surface"],      result["rnd_k_surface"]

        Optional CDFs:
        result["physical_cdf_lr_surface"], result["physical_cdf_r_surface"], result["physical_cdf_k_surface"]
        result["cdf_lr_surface"],          result["cdf_r_surface"],          result["cdf_k_surface"]

    Axes:
        x = selected return / price axis
        y = calendar date
        z = chosen surface at maturity nearest to T_target
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pathlib import Path

    # ------------------------------------------------------------
    # Local helpers
    # ------------------------------------------------------------

    def _std_x_axis(x_axis):
        x = str(x_axis).strip()
        aliases = {
            "r": "r",
            "log": "r",
            "lr": "r",
            "log_return": "r",
            "log-return": "r",

            "R": "R",
            "gross": "R",
            "gross_return": "R",

            "return": "return",
            "simple": "return",
            "simple_return": "return",

            "K": "K",
            "k": "K",
            "strike": "K",
            "terminal": "K",
            "terminal_price": "K",
            "ST": "K",
            "st": "K",
        }
        if x not in aliases:
            raise ValueError("x_axis must be one of {'r', 'R', 'return', 'K'}.")
        return aliases[x]

    def _std_kind(kind):
        k = str(kind).lower().strip()
        aliases = {
            "pricing_kernel": "pricing_kernel",
            "kernel": "pricing_kernel",
            "pk": "pricing_kernel",
            "m": "pricing_kernel",

            "physical": "physical",
            "p": "physical",

            "rnd": "rnd",
            "risk_neutral": "rnd",
            "risk-neutral": "rnd",
            "q": "rnd",

            "rra": "rra",
            "relative_risk_aversion": "rra",
            "risk_aversion": "rra",
        }
        if k not in aliases:
            raise ValueError("kind must be one of {'pricing_kernel', 'physical', 'rnd', 'rra'}.")
        return aliases[k]

    def _axis_info(day, x_axis):
        x_axis = _std_x_axis(x_axis)

        if x_axis == "r":
            x = np.asarray(day["grid_lr"], float).ravel()
            xlabel = r"Log return $r=\log(S_T/S_0)$"
            suffix = "lr"

        elif x_axis == "R":
            x = np.asarray(day["grid_r"], float).ravel()
            xlabel = r"Gross return $R=S_T/S_0$"
            suffix = "r"

        elif x_axis == "return":
            x = np.asarray(day["grid_r"], float).ravel() - 1.0
            xlabel = r"Simple return $R-1$"
            suffix = "r"

        else:
            x = np.asarray(day["grid_k"], float).ravel()
            xlabel = r"Terminal price / strike $K=S_T$"
            suffix = "k"

        return x, xlabel, suffix

    def _surface_key_and_label(kind, x_axis):
        kind = _std_kind(kind)
        x_axis = _std_x_axis(x_axis)

        suffix = {
            "r": "lr",
            "R": "r",
            "return": "r",
            "K": "k",
        }[x_axis]

        if kind == "pricing_kernel":
            return "pricing_kernel_surface", r"$M$", "Pricing Kernel Surface"

        if kind == "rra":
            return "relative_risk_aversion_surface", "Relative risk aversion", "Relative Risk Aversion Surface"

        if kind == "physical":
            return f"physical_{suffix}_surface", {
                "r": r"$f_P(r)$",
                "R": r"$p_P(R)$",
                "return": r"$p_P(R)$",
                "K": r"$p_P(K)$",
            }[x_axis], "Physical Density Surface"

        if kind == "rnd":
            return f"rnd_{suffix}_surface", {
                "r": r"$f_Q(r)$",
                "R": r"$p_Q(R)$",
                "return": r"$p_Q(R)$",
                "K": r"$p_Q(K)$",
            }[x_axis], "Risk-Neutral Density Surface"

        raise RuntimeError("Unreachable kind.")

    def _cdf_from_density_local(x, f):
        x = np.asarray(x, float).ravel()
        f = np.asarray(f, float).ravel()

        out = np.full_like(x, np.nan, dtype=float)
        good = np.isfinite(x) & np.isfinite(f) & (f >= 0)

        if good.sum() < 3:
            return out

        xx = x[good]
        ff = f[good]

        order = np.argsort(xx)
        xx = xx[order]
        ff = ff[order]

        inc = 0.5 * (ff[1:] + ff[:-1]) * np.diff(xx)

        cdf = np.empty_like(xx)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(inc)

        total = cdf[-1]
        if not np.isfinite(total) or total <= 0:
            return out

        cdf /= total
        out[good] = np.interp(x[good], xx, cdf, left=np.nan, right=np.nan)
        return out

    def _cdf_key_and_density_key(measure, x_axis):
        measure = str(measure).lower().strip()
        x_axis = _std_x_axis(x_axis)

        suffix = {
            "r": "lr",
            "R": "r",
            "return": "r",
            "K": "k",
        }[x_axis]

        if measure in {"physical", "p"}:
            return f"physical_cdf_{suffix}_surface", f"physical_{suffix}_surface", "physical"

        if measure in {"rnd", "q", "risk_neutral", "risk-neutral"}:
            return f"cdf_{suffix}_surface", f"rnd_{suffix}_surface", "risk-neutral"

        raise ValueError("truncation_measure must be 'physical' or 'rnd'.")

    # ------------------------------------------------------------
    # Validate options
    # ------------------------------------------------------------

    kind = _std_kind(kind)
    x_axis = _std_x_axis(x_axis)

    if zscale not in {"linear", "log"}:
        raise ValueError("zscale must be 'linear' or 'log'.")

    if truncate_by_ptails:
        aL, aR = map(float, ptail_alphas)
        if not (0 <= aL < 1 and 0 <= aR < 1 and aL + aR < 1):
            raise ValueError("ptail_alphas must satisfy 0 <= left,right < 1 and left+right < 1.")
    else:
        aL, aR = 0.0, 0.0

    surface_key, zlabel_base, default_title = _surface_key_and_label(kind, x_axis)
    cdf_key, density_key, measure_label = _cdf_key_and_density_key(truncation_measure, x_axis)

    # ------------------------------------------------------------
    # Sort and thin dates
    # ------------------------------------------------------------

    keys = list(result_dict.keys())
    if len(keys) == 0:
        raise ValueError("result_dict is empty.")

    date_ts = pd.to_datetime(keys)
    order = np.argsort(date_ts.values)

    keys = [keys[i] for i in order]
    date_ts = date_ts[order]

    n_all = len(keys)

    if stride is not None and stride > 1:
        keep = np.arange(0, n_all, int(stride), dtype=int)
    elif n_all <= max_dates:
        keep = np.arange(n_all, dtype=int)
    else:
        keep = np.linspace(0, n_all - 1, int(max_dates)).round().astype(int)

    keys = [keys[i] for i in keep]
    date_ts = date_ts[keep]

    # ------------------------------------------------------------
    # Build Z rows through time
    # ------------------------------------------------------------

    x_ref = None
    xlabel = None

    rows_Z = []
    kept_dates = []
    T_used_list = []

    for dk, dt in zip(keys, date_ts):
        day = result_dict[dk]

        try:
            T_grid = np.asarray(day["T_grid"], float).ravel()
            x_grid, xlabel_i, _suffix = _axis_info(day, x_axis)

            if surface_key not in day:
                raise KeyError(f"Missing {surface_key}")

            Z_surface = np.asarray(day[surface_key], float)

            if Z_surface.shape != (T_grid.size, x_grid.size):
                raise ValueError(
                    f"{surface_key} shape {Z_surface.shape} does not match "
                    f"(len(T_grid), len(x_grid)) = {(T_grid.size, x_grid.size)}."
                )

            j = int(np.nanargmin(np.abs(T_grid - float(T_target))))
            z_row = Z_surface[j, :].astype(float, copy=True)

            if truncate_by_ptails:
                if cdf_key in day:
                    F_surface = np.asarray(day[cdf_key], float)

                    if F_surface.shape != Z_surface.shape:
                        raise ValueError(f"{cdf_key} has wrong shape.")

                    F_row = F_surface[j, :]

                else:
                    if density_key not in day:
                        raise KeyError(f"Need {cdf_key} or {density_key} for tail truncation.")

                    f_surface = np.asarray(day[density_key], float)

                    if f_surface.shape != Z_surface.shape:
                        raise ValueError(f"{density_key} has wrong shape.")

                    F_row = _cdf_from_density_local(x_grid, f_surface[j, :])

                keep_tail = (
                    np.isfinite(F_row)
                    & (F_row >= aL)
                    & (F_row <= 1.0 - aR)
                )

                z_row[~keep_tail] = np.nan

            if x_ref is None:
                x_ref = x_grid.copy()
                xlabel = xlabel_i
            else:
                same_grid = (
                    x_grid.shape == x_ref.shape
                    and np.allclose(x_grid, x_ref, atol=1e-12, rtol=1e-9, equal_nan=True)
                )

                if not same_grid:
                    good = np.isfinite(x_grid) & np.isfinite(z_row)
                    if good.sum() < 2:
                        raise ValueError("Not enough valid points to interpolate.")
                    z_row = np.interp(x_ref, x_grid[good], z_row[good], left=np.nan, right=np.nan)

            rows_Z.append(z_row)
            kept_dates.append(dt)
            T_used_list.append(float(T_grid[j]))

        except Exception as exc:
            print(f"[skip] {dk}: {exc}")
            continue

    if not rows_Z:
        raise ValueError("No valid rows extracted. Check result_dict structure and T_target.")

    Z = np.vstack(rows_Z)
    kept_dates = pd.to_datetime(kept_dates)
    x = np.asarray(x_ref, float).ravel()

    # ------------------------------------------------------------
    # Apply x bounds
    # ------------------------------------------------------------

    if x_bounds is not None:
        lo, hi = sorted(map(float, x_bounds))
        mask_x = np.isfinite(x) & (x >= lo) & (x <= hi)

        if not np.any(mask_x):
            raise ValueError("x_bounds produced an empty plotting window.")

        x = x[mask_x]
        Z = Z[:, mask_x]

    # ------------------------------------------------------------
    # Z transform
    # ------------------------------------------------------------

    if zscale == "log":
        Z_plot = np.where(np.isfinite(Z) & (Z > 0), np.log(np.maximum(Z, z_eps)), np.nan)
        zlabel = rf"$\log$ {zlabel_base}"
    else:
        Z_plot = Z
        zlabel = zlabel_base

    # ------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------

    y_num = mdates.date2num(kept_dates.to_pydatetime())
    X, Y = np.meshgrid(x, y_num)

    mean_T_used = float(np.nanmean(T_used_list)) if len(T_used_list) else float(T_target)

    if title is None:
        title = default_title

    full_title = f"{title} — T ≈ {365 * mean_T_used:.0f}d"

    if truncate_by_ptails:
        full_title += f" — inner {100 * (1 - aL - aR):.0f}% by {measure_label} CDF"

    # ------------------------------------------------------------
    # Interactive Plotly
    # ------------------------------------------------------------

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
            title=full_title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title="Date",
                zaxis_title=zlabel,
            ),
            margin=dict(l=0, r=0, t=55, b=0),
        )

        if save is not None:
            save_path = Path(save)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_path.suffix.lower() == ".html":
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)

            print(f"[saved] {save_path}")

        if show:
            fig.show()

        return fig

    # ------------------------------------------------------------
    # Static Matplotlib
    # ------------------------------------------------------------

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

    ax.set_title(full_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)

    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    try:
        ax.set_yticks(y_num[::max(1, len(y_num) // 6)])
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


# Backward-compatible alias for your old function name.
def plot_pricing_kernel_3d_surface_by_T(*args, **kwargs):
    kwargs.setdefault("kind", "pricing_kernel")
    return plot_surface_3d_by_T(*args, **kwargs)
def plot_physical_density_panels(result: dict, **kwargs):
    """Maturity panels for the physical density."""
    return plot_surface_panels(result, kind="physical", **kwargs)


def plot_rnd_panels(result: dict, **kwargs):
    """Maturity panels for the risk-neutral density."""
    return plot_surface_panels(result, kind="rnd", **kwargs)


def plot_pricing_kernel_panels(result: dict, **kwargs):
    """Maturity panels for the pricing kernel."""
    return plot_surface_panels(result, kind="pricing_kernel", **kwargs)


def plot_rra_panels(result: dict, **kwargs):
    """Maturity panels for relative risk aversion."""
    return plot_surface_panels(result, kind="rra", **kwargs)

def plot_pqk_multipanel(
    out_dict: Dict[str, dict],
    *,
    title: Optional[str] = None,
    n_panels: Optional[int] = None,
    panel_shape: Tuple[int, int] = (2, 4),
    x_axis: str = "R",
    x_bounds: Optional[Tuple[float, float]] = None,
    truncate_kernel: bool = True,
    ptail_alpha: Tuple[float, float] = (0.10, 0.00),
    truncation_measure: str = "physical",
    kernel_yscale: Literal["linear", "log"] = "linear",
    kernel_log_eps: float = 1e-300,
    kernel_linestyle: str = "--",
    lw_density: float = 1.8,
    lw_kernel: float = 1.5,
    alpha_density: float = 0.95,
    alpha_kernel: float = 0.90,
    show_percentiles: bool = False,
    percentiles: Tuple[float, ...] = (0.05, 0.50, 0.95),
    percentile_measures: Tuple[str, ...] = ("rnd", "physical"),
    percentile_linestyle_rnd: str = "-",
    percentile_linestyle_physical: str = "--",
    percentile_alpha_rnd: float = 0.90,
    percentile_alpha_physical: float = 0.45,
    percentile_linewidth_rnd: float = 1.8,
    percentile_linewidth_physical: float = 1.5,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    legend_loc: str = "upper center",
    show: bool = True,
):
    """
    Multi-panel plot of:

        1. Risk-neutral density q
        2. Physical density p
        3. Pricing kernel M (secondary axis)

    using the NEW interface.

    IMPORTANT:
    Percentile markers now share a SINGLE legend entry:
        - "RND percentiles"
        - "Physical percentiles"

    instead of one legend entry per percentile line.
    """
    import matplotlib.pyplot as plt

    if not isinstance(out_dict, dict) or len(out_dict) == 0:
        raise ValueError("out_dict must be a non-empty dictionary.")

    x_axis = _standardize_x_axis(x_axis)

    # =========================================================
    # Reference grid from first model
    # =========================================================

    first_label = next(iter(out_dict))
    first = out_dict[first_label]

    T_ref = _as_1d(first["T_grid"])
    X_ref, xlabel = _get_plot_x_grid(first, x_axis=x_axis)

    q_key, q_label, _ = _surface_key("rnd", x_axis)
    p_key, p_label, _ = _surface_key("physical", x_axis)

    M_key = "pricing_kernel_surface"

    cdf_q_key = _cdf_key("rnd", x_axis)
    cdf_p_key = _cdf_key("physical", x_axis)

    # =========================================================
    # Panel selection
    # =========================================================

    nrows, ncols = panel_shape
    max_panels = nrows * ncols

    if n_panels is None:
        n_panels = max_panels

    n_panels = min(int(n_panels), max_panels, T_ref.size)

    idxs = _pick_panel_indices(T_ref, n_panels)

    # =========================================================
    # X bounds
    # =========================================================

    if x_bounds is not None:
        lo, hi = sorted(map(float, x_bounds))
        xmask_ref = (
            np.isfinite(X_ref)
            & (X_ref >= lo)
            & (X_ref <= hi)
        )
    else:
        xmask_ref = np.isfinite(X_ref)

    if not np.any(xmask_ref):
        raise ValueError("x_bounds produced an empty x-grid.")

    X_plot = X_ref[xmask_ref]

    # =========================================================
    # Kernel truncation settings
    # =========================================================

    if truncate_kernel:
        aL, aR = map(float, ptail_alpha)

        if not (
            0 <= aL < 1
            and 0 <= aR < 1
            and aL + aR < 1
        ):
            raise ValueError(
                "ptail_alpha must satisfy "
                "0 <= left,right < 1 and left+right < 1."
            )

        truncation_measure = str(
            truncation_measure
        ).lower().strip()

        if truncation_measure in {"physical", "p"}:
            trunc_kind = "physical"

        elif truncation_measure in {
            "risk_neutral",
            "risk-neutral",
            "rnd",
            "q",
        }:
            trunc_kind = "rnd"

        else:
            raise ValueError(
                "truncation_measure must be "
                "'physical' or 'risk_neutral'."
            )

    # =========================================================
    # Kernel y-scale
    # =========================================================

    kernel_yscale = str(kernel_yscale).lower().strip()

    if kernel_yscale not in {"linear", "log"}:
        raise ValueError(
            "kernel_yscale must be 'linear' or 'log'."
        )

    # =========================================================
    # Percentiles
    # =========================================================

    if show_percentiles:
        probs = np.asarray(percentiles, dtype=float)

        if np.any((probs <= 0) | (probs >= 1)):
            raise ValueError(
                "percentiles must lie strictly between 0 and 1."
            )

        percentile_measures = tuple(
            str(x).lower().strip()
            for x in percentile_measures
        )

    else:
        probs = np.asarray([], dtype=float)
        percentile_measures = tuple()

    # =========================================================
    # Interpolation helper
    # =========================================================

    def _interp_row_to_ref(X, row):
        X = np.asarray(X, float).ravel()
        row = np.asarray(row, float).ravel()

        good = np.isfinite(X) & np.isfinite(row)

        if good.sum() < 2:
            return np.full_like(
                X_ref,
                np.nan,
                dtype=float,
            )

        return np.interp(
            X_ref,
            X[good],
            row[good],
            left=np.nan,
            right=np.nan,
        )

    # =========================================================
    # Figure
    # =========================================================

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.8 * ncols, 4.1 * nrows),
        sharex=True,
    )

    axes = np.asarray(axes).ravel()

    ax2_list = []

    # Tracks whether a legend item has already been used.
    legend_seen = set()

    # =========================================================
    # Main loop
    # =========================================================

    for k, j_ref in enumerate(idxs):

        ax = axes[k]
        ax2 = ax.twinx()

        ax2_list.append(ax2)

        # -----------------------------------------------------
        # Plot each model
        # -----------------------------------------------------

        for model_label, result in out_dict.items():

            T = _as_1d(result["T_grid"])
            X, _ = _get_plot_x_grid(
                result,
                x_axis=x_axis,
            )

            q_surface = np.asarray(
                result[q_key],
                dtype=float,
            )

            p_surface = np.asarray(
                result[p_key],
                dtype=float,
            )

            M_surface = np.asarray(
                result[M_key],
                dtype=float,
            )

            j = int(
                np.argmin(np.abs(T - T_ref[j_ref]))
            )

            same_x_grid = (
                X.shape == X_ref.shape
                and np.allclose(
                    X,
                    X_ref,
                    equal_nan=True,
                )
            )

            # -------------------------------------------------
            # Align grids
            # -------------------------------------------------

            if same_x_grid:

                q_ref = q_surface[j, :]
                p_ref = p_surface[j, :]
                M_ref = M_surface[j, :]

            else:

                q_ref = _interp_row_to_ref(
                    X,
                    q_surface[j, :],
                )

                p_ref = _interp_row_to_ref(
                    X,
                    p_surface[j, :],
                )

                M_ref = _interp_row_to_ref(
                    X,
                    M_surface[j, :],
                )

            q_plot = q_ref[xmask_ref]
            p_plot = p_ref[xmask_ref]
            M_plot = M_ref[xmask_ref]

            # -------------------------------------------------
            # q density
            # -------------------------------------------------

            q_line, = ax.plot(
                X_plot,
                q_plot,
                linewidth=lw_density + 0.2,
                alpha=alpha_density,
                label=(
                    f"{q_label} ({model_label})"
                    if f"{q_label} ({model_label})"
                    not in legend_seen
                    else "_nolegend_"
                ),
            )

            legend_seen.add(
                f"{q_label} ({model_label})"
            )

            color = q_line.get_color()

            # -------------------------------------------------
            # p density
            # -------------------------------------------------

            ax.plot(
                X_plot,
                p_plot,
                linewidth=lw_density,
                alpha=0.72,
                color=color,
                linestyle="-",
                label=(
                    f"{p_label} ({model_label})"
                    if f"{p_label} ({model_label})"
                    not in legend_seen
                    else "_nolegend_"
                ),
            )

            legend_seen.add(
                f"{p_label} ({model_label})"
            )

            # -------------------------------------------------
            # Percentile markers
            # -------------------------------------------------

            if show_percentiles:

                # =============================================
                # RND percentiles
                # =============================================

                if any(
                    m in percentile_measures
                    for m in {
                        "rnd",
                        "q",
                        "risk_neutral",
                    }
                ):

                    if cdf_q_key in result:
                        cdf_q_surface = np.asarray(
                            result[cdf_q_key],
                            dtype=float,
                        )
                    else:
                        cdf_q_surface = _get_cdf_surface(
                            result,
                            kind="rnd",
                            x_axis=x_axis,
                        )

                    cdf_q_ref = (
                        cdf_q_surface[j, :]
                        if same_x_grid
                        else _interp_row_to_ref(
                            X,
                            cdf_q_surface[j, :],
                        )
                    )

                    q_marks = _quantiles_from_cdf(
                        X_ref,
                        cdf_q_ref,
                        probs,
                    )

                    for xmark in q_marks:

                        if np.isfinite(xmark):

                            legend_label = (
                                "RND percentiles"
                            )

                            ax.axvline(
                                xmark,
                                color=color,
                                linestyle=(
                                    percentile_linestyle_rnd
                                ),
                                alpha=(
                                    percentile_alpha_rnd
                                ),
                                linewidth=(
                                    percentile_linewidth_rnd
                                ),
                                label=(
                                    legend_label
                                    if legend_label
                                    not in legend_seen
                                    else "_nolegend_"
                                ),
                            )

                            legend_seen.add(
                                legend_label
                            )

                # =============================================
                # Physical percentiles
                # =============================================

                if any(
                    m in percentile_measures
                    for m in {
                        "physical",
                        "p",
                    }
                ):

                    if cdf_p_key in result:
                        cdf_p_surface = np.asarray(
                            result[cdf_p_key],
                            dtype=float,
                        )
                    else:
                        cdf_p_surface = _get_cdf_surface(
                            result,
                            kind="physical",
                            x_axis=x_axis,
                        )

                    cdf_p_ref = (
                        cdf_p_surface[j, :]
                        if same_x_grid
                        else _interp_row_to_ref(
                            X,
                            cdf_p_surface[j, :],
                        )
                    )

                    p_marks = _quantiles_from_cdf(
                        X_ref,
                        cdf_p_ref,
                        probs,
                    )

                    for xmark in p_marks:

                        if np.isfinite(xmark):

                            legend_label = (
                                "Physical percentiles"
                            )

                            ax.axvline(
                                xmark,
                                color=color,
                                linestyle=(
                                    percentile_linestyle_physical
                                ),
                                alpha=(
                                    percentile_alpha_physical
                                ),
                                linewidth=(
                                    percentile_linewidth_physical
                                ),
                                label=(
                                    legend_label
                                    if legend_label
                                    not in legend_seen
                                    else "_nolegend_"
                                ),
                            )

                            legend_seen.add(
                                legend_label
                            )

            # -------------------------------------------------
            # Kernel truncation
            # -------------------------------------------------

            if truncate_kernel:

                if trunc_kind == "physical":

                    F_surface = (
                        np.asarray(
                            result[cdf_p_key],
                            dtype=float,
                        )
                        if cdf_p_key in result
                        else _get_cdf_surface(
                            result,
                            kind="physical",
                            x_axis=x_axis,
                        )
                    )

                else:

                    F_surface = (
                        np.asarray(
                            result[cdf_q_key],
                            dtype=float,
                        )
                        if cdf_q_key in result
                        else _get_cdf_surface(
                            result,
                            kind="rnd",
                            x_axis=x_axis,
                        )
                    )

                F_ref = (
                    F_surface[j, :]
                    if same_x_grid
                    else _interp_row_to_ref(
                        X,
                        F_surface[j, :],
                    )
                )

                F_plot = F_ref[xmask_ref]

                kmask = (
                    np.isfinite(M_plot)
                    & np.isfinite(F_plot)
                    & (F_plot >= aL)
                    & (F_plot <= 1.0 - aR)
                )

            else:

                kmask = np.isfinite(M_plot)

            Xk = X_plot[kmask]
            Mk = M_plot[kmask]

            if kernel_yscale == "log":

                good = (
                    np.isfinite(Mk)
                    & (Mk > 0)
                )

                Xk = Xk[good]

                Mk = np.maximum(
                    Mk[good],
                    kernel_log_eps,
                )

            # -------------------------------------------------
            # Pricing kernel
            # -------------------------------------------------

            if Xk.size > 1:

                ax2.plot(
                    Xk,
                    Mk,
                    linestyle=kernel_linestyle,
                    linewidth=lw_kernel,
                    alpha=alpha_kernel,
                    color=color,
                    label=(
                        f"$M$ ({model_label})"
                        if f"$M$ ({model_label})"
                        not in legend_seen
                        else "_nolegend_"
                    ),
                )

                legend_seen.add(
                    f"$M$ ({model_label})"
                )

        # -----------------------------------------------------
        # Axis formatting
        # -----------------------------------------------------

        ax.set_title(
            f"T ≈ {365 * T_ref[j_ref]:.1f}d"
        )

        ax.grid(True, alpha=0.25)

        if k % ncols == 0:
            ax.set_ylabel("Density")

        if k >= len(idxs) - ncols:
            ax.set_xlabel(xlabel)

        if k % ncols == ncols - 1:
            ax2.set_ylabel("Pricing kernel")

        ax2.set_yscale(kernel_yscale)

        if x_bounds is not None:
            ax.set_xlim(
                X_plot[0],
                X_plot[-1],
            )

    # =========================================================
    # Hide unused axes
    # =========================================================

    for k in range(len(idxs), axes.size):
        axes[k].axis("off")

    # =========================================================
    # Global legend
    # =========================================================

    handles, labels = [], []

    h1, l1 = axes[0].get_legend_handles_labels()

    handles.extend(h1)
    labels.extend(l1)

    if ax2_list:
        h2, l2 = ax2_list[0].get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)

    # Remove duplicate legend entries.
    unique = {}

    for h, l in zip(handles, labels):
        if l not in unique and l != "_nolegend_":
            unique[l] = h

    handles = list(unique.values())
    labels = list(unique.keys())

    # =========================================================
    # Title
    # =========================================================

    if title is None:
        title = (
            "Risk-Neutral Density, "
            "Physical Density, and Pricing Kernel"
        )

    fig.suptitle(
        title,
        y=0.995,
        fontsize=14,
    )

    # =========================================================
    # Global legend
    # =========================================================

    fig.legend(
        handles,
        labels,
        loc=legend_loc,
        bbox_to_anchor=(0.5, 0.965),
        ncol=min(5, max(1, len(labels))),
        frameon=False,
        handlelength=2.8,
        columnspacing=1.4,
    )

    fig.subplots_adjust(
        top=0.88,
        wspace=0.28,
        hspace=0.34,
    )

    _maybe_save(fig, save, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# Backward-compatible alias for your old function name.
def M_Q_K_multipanel_multi(*args, **kwargs):
    return plot_pqk_multipanel(*args, **kwargs)