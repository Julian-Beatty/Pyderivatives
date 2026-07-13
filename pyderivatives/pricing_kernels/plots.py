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
    interactive: bool = False,
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
    
    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Surface(
                    x=x_plot,
                    y=T_plot,
                    z=Z_plot,
                    colorscale=cmap,
                    opacity=alpha,
                    colorbar=dict(title=zlabel),
                    hovertemplate=(
                        f"{xlabel}: %{{x:.6g}}<br>"
                        "T: %{y:.6g} years<br>"
                        f"{zlabel}: %{{z:.6g}}"
                        "<extra></extra>"
                    ),
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title="Maturity T (years)",
                zaxis_title=zlabel,
                camera=dict(
                    eye=dict(
                        x=1.75 * np.cos(np.deg2rad(azim)),
                        y=1.75 * np.sin(np.deg2rad(azim)),
                        z=1.25,
                    )
                ),
            ),
            width=950,
            height=750,
        )

        if save is not None:
            save = Path(save)
            if save.suffix.lower() == ".html":
                fig.write_html(str(save))
            else:
                fig.write_image(str(save), scale=2)

        if show:
            fig.show()

        return fig

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
    rnd_dict: Optional[dict] = None,
    title: Optional[str] = None,
    n_panels: Optional[int] = None,
    panel_shape: Tuple[int, int] = (2, 4),
    snap_percentiles_to_traded_strikes: bool = True,
    target_maturities: Optional[Tuple[float, ...]] = None,
    maturity_tol: Optional[float] = None,
    maturity_units: Literal["years", "days"] = "years",

    show_expiry_dates: bool = False,
    valuation_date: Optional[Union[str, "pd.Timestamp"]] = None,

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
    snap_maturity_tol: float = 1.0 / 365.0,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    legend_loc: str = "upper right",
    show: bool = True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    if show_expiry_dates:
        import pandas as pd

    if not isinstance(out_dict, dict) or len(out_dict) == 0:
        raise ValueError("out_dict must be a non-empty dictionary.")

    def _cdf_prob_at_x(xgrid, cdf_vals, x):
        xgrid = np.asarray(xgrid, float)
        cdf_vals = np.asarray(cdf_vals, float)
    
        good = np.isfinite(xgrid) & np.isfinite(cdf_vals)
    
        if good.sum() < 2:
            return np.nan
    
        return float(
            np.interp(
                x,
                xgrid[good],
                cdf_vals[good],
                left=np.nan,
                right=np.nan,
            )
        )

    x_axis = _standardize_x_axis(x_axis)

    first_label = next(iter(out_dict))
    first = out_dict[first_label]

    T_ref = _as_1d(first["T_grid"])
    X_ref, xlabel = _get_plot_x_grid(first, x_axis=x_axis)

    q_key, q_label, _ = _surface_key("rnd", x_axis)
    p_key, p_label, _ = _surface_key("physical", x_axis)

    M_key = "pricing_kernel_surface"

    cdf_q_key = _cdf_key("rnd", x_axis)
    cdf_p_key = _cdf_key("physical", x_axis)

    nrows, ncols = panel_shape
    max_panels = nrows * ncols

    if target_maturities is not None:
        targets = np.asarray(target_maturities, dtype=float).ravel()

        if maturity_units == "days":
            targets = targets / 365.0
            tol_eff = None if maturity_tol is None else float(maturity_tol) / 365.0
        elif maturity_units == "years":
            tol_eff = None if maturity_tol is None else float(maturity_tol)
        else:
            raise ValueError("maturity_units must be either 'years' or 'days'.")

        idxs = []

        for target in targets:
            if not np.isfinite(target):
                continue

            j = int(np.argmin(np.abs(T_ref - target)))
            dist = abs(float(T_ref[j]) - float(target))

            if tol_eff is not None and dist > tol_eff:
                continue

            if j not in idxs:
                idxs.append(j)

        if len(idxs) == 0:
            raise ValueError(
                "No requested target_maturities were found within maturity_tol."
            )

        idxs = np.asarray(idxs, dtype=int)

        if idxs.size > max_panels:
            idxs = idxs[:max_panels]

    else:
        if n_panels is None:
            n_panels = max_panels

        n_panels = min(int(n_panels), max_panels, T_ref.size)
        idxs = _pick_panel_indices(T_ref, n_panels)

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

    if truncate_kernel:
        aL, aR = map(float, ptail_alpha)

        if not (
            0 <= aL < 1
            and 0 <= aR < 1
            and aL + aR < 1
        ):
            raise ValueError(
                "ptail_alpha must satisfy 0 <= left,right < 1 and left+right < 1."
            )

        truncation_measure = str(truncation_measure).lower().strip()

        if truncation_measure in {"physical", "p"}:
            trunc_kind = "physical"
        elif truncation_measure in {"risk_neutral", "risk-neutral", "rnd", "q"}:
            trunc_kind = "rnd"
        else:
            raise ValueError(
                "truncation_measure must be 'physical' or 'risk_neutral'."
            )

    kernel_yscale = str(kernel_yscale).lower().strip()

    if kernel_yscale not in {"linear", "log"}:
        raise ValueError("kernel_yscale must be 'linear' or 'log'.")

    if show_percentiles:
        probs = np.asarray(percentiles, dtype=float)

        if np.any((probs <= 0) | (probs >= 1)):
            raise ValueError("percentiles must lie strictly between 0 and 1.")

        percentile_measures = tuple(
            str(x).lower().strip()
            for x in percentile_measures
        )
    else:
        probs = np.asarray([], dtype=float)
        percentile_measures = tuple()

    def _interp_row_to_ref(X, row):
        X = np.asarray(X, float).ravel()
        row = np.asarray(row, float).ravel()

        good = np.isfinite(X) & np.isfinite(row)

        if good.sum() < 2:
            return np.full_like(X_ref, np.nan, dtype=float)

        return np.interp(
            X_ref,
            X[good],
            row[good],
            left=np.nan,
            right=np.nan,
        )

    def _format_xmark(xmark):
        if x_axis.lower() in {"k", "strike"}:
            return f"K={xmark:.2f}"
        elif x_axis.lower() in {"r"}:
            return f"R={xmark:.4f}"
        elif x_axis.lower() in {"lr", "logreturn", "log_return"}:
            return f"lr={xmark:.4f}"
        return f"x={xmark:.4f}"

    def _format_panel_title(T_years):
        days = float(T_years) * 365.0

        if days < 1:
            t_label = "Same day"
        elif abs(days - round(days)) < 0.05:
            d_int = int(round(days))
            t_label = f"T={d_int} day" if d_int == 1 else f"T={d_int} days"
        else:
            t_label = f"T={days:.1f} days"

        if show_expiry_dates:
            if valuation_date is None:
                raise ValueError(
                    "valuation_date must be supplied when show_expiry_dates=True."
                )

            val_date = pd.Timestamp(valuation_date)
            expiry_date = val_date + pd.to_timedelta(days, unit="D")
            expiry_str = expiry_date.strftime("%Y-%m-%d")

            return f"{expiry_str} ({t_label})"

        return t_label

    def _get_single_rnd_day():
        if rnd_dict is None:
            return None
    
        if not isinstance(rnd_dict, dict) or len(rnd_dict) == 0:
            return None
    
        # Case 1: user passed one RND result directly:
        # rnd_dict = RND_dict[pd.Timestamp("2026-05-22")]
        if "day" in rnd_dict:
            return rnd_dict["day"]
    
        # Case 2: user passed full date-keyed RND_dict with one date
        if len(rnd_dict) == 1:
            date_key = next(iter(rnd_dict))
            entry = rnd_dict[date_key]
    
            if isinstance(entry, dict) and "day" in entry:
                return entry["day"]
    
            raise ValueError("rnd_dict entry must contain key 'day'.")
    
        # Case 3: user passed full date-keyed RND_dict with many dates
        if valuation_date is None:
            raise ValueError(
                "rnd_dict has multiple date keys, so valuation_date must be supplied."
            )
    
        import pandas as pd
    
        val_date = pd.Timestamp(valuation_date).normalize()
    
        normalized_lookup = {}
    
        for k in rnd_dict.keys():
            try:
                k_date = pd.Timestamp(k).normalize()
            except Exception:
                continue
    
            normalized_lookup[k_date] = k
    
        if val_date not in normalized_lookup:
            raise ValueError(
                f"valuation_date {val_date.date()} was not found in rnd_dict."
            )
    
        date_key = normalized_lookup[val_date]
        entry = rnd_dict[date_key]
    
        if not isinstance(entry, dict) or "day" not in entry:
            raise ValueError("rnd_dict[valuation_date] must contain key 'day'.")
    
        return entry["day"]

    rnd_day = _get_single_rnd_day()

    def _snap_to_traded_x(xmark, T_target_ref):
        if rnd_day is None or not np.isfinite(xmark):
            return xmark

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

        K_obs = K_obs[valid]
        T_obs = T_obs[valid]

        if K_obs.size == 0:
            return xmark

        tmask = np.abs(T_obs - float(T_target_ref)) <= float(snap_maturity_tol)

        if np.any(tmask):
            K_use = K_obs[tmask]
        else:
            K_use = K_obs

        K_snap = float(K_use[np.argmin(np.abs(K_use - xmark))])

        if x_axis.lower() in {"k", "strike"}:
            return K_snap

        S0 = first.get("S0", None)

        if S0 is None:
            S0 = getattr(rnd_day, "S0", None)
        if S0 is None:
            S0 = getattr(rnd_day, "S0", None)

        if S0 is None:
            return K_snap

        S0 = float(S0)

        if x_axis.lower() in {"r"}:
            return K_snap / S0

        if x_axis.lower() in {"lr", "logreturn", "log_return"}:
            return float(np.log(K_snap / S0))

        return K_snap

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.8 * ncols, 4.1 * nrows),
        sharex=True,
    )

    axes = np.asarray(axes).ravel()

    for k, j_ref in enumerate(idxs):
        ax = axes[k]
        ax2 = ax.twinx()

        local_legend_seen = set()
        T_target_ref = float(T_ref[j_ref])

        for model_label, result in out_dict.items():
            T = _as_1d(result["T_grid"])
            X, _ = _get_plot_x_grid(result, x_axis=x_axis)

            q_surface = np.asarray(result[q_key], dtype=float)
            p_surface = np.asarray(result[p_key], dtype=float)
            M_surface = np.asarray(result[M_key], dtype=float)

            j = int(np.argmin(np.abs(T - T_target_ref)))

            same_x_grid = (
                X.shape == X_ref.shape
                and np.allclose(X, X_ref, equal_nan=True)
            )

            if same_x_grid:
                q_ref = q_surface[j, :]
                p_ref = p_surface[j, :]
                M_ref = M_surface[j, :]
            else:
                q_ref = _interp_row_to_ref(X, q_surface[j, :])
                p_ref = _interp_row_to_ref(X, p_surface[j, :])
                M_ref = _interp_row_to_ref(X, M_surface[j, :])

            q_plot = q_ref[xmask_ref]
            p_plot = p_ref[xmask_ref]
            M_plot = M_ref[xmask_ref]

            q_legend = f"{q_label} ({model_label})"
            p_legend = f"{p_label} ({model_label})"
            m_legend = f"$M$ ({model_label})"

            q_line, = ax.plot(
                X_plot,
                q_plot,
                linewidth=lw_density + 0.2,
                alpha=alpha_density,
                label=q_legend if q_legend not in local_legend_seen else "_nolegend_",
            )

            local_legend_seen.add(q_legend)
            color = q_line.get_color()

            ax.plot(
                X_plot,
                p_plot,
                linewidth=lw_density,
                alpha=0.72,
                color=color,
                linestyle="-",
                label=p_legend if p_legend not in local_legend_seen else "_nolegend_",
            )

            local_legend_seen.add(p_legend)

            if show_percentiles:
                if any(m in percentile_measures for m in {"rnd", "q", "risk_neutral"}):
                    if cdf_q_key in result:
                        cdf_q_surface = np.asarray(result[cdf_q_key], dtype=float)
                    else:
                        cdf_q_surface = _get_cdf_surface(
                            result,
                            kind="rnd",
                            x_axis=x_axis,
                        )

                    cdf_q_ref = (
                        cdf_q_surface[j, :]
                        if same_x_grid
                        else _interp_row_to_ref(X, cdf_q_surface[j, :])
                    )

                    q_marks = _quantiles_from_cdf(X_ref, cdf_q_ref, probs)
                    
                    for prob, xmark_raw in zip(probs, q_marks):
                        if np.isfinite(xmark_raw):
                    
                            if snap_percentiles_to_traded_strikes:
                                xmark = _snap_to_traded_x(xmark_raw, T_target_ref)
                                label_prob = _cdf_prob_at_x(X_ref, cdf_q_ref, xmark)
                            else:
                                xmark = xmark_raw
                                label_prob = prob
                    
                            legend_label = (
                                f"RND p{100 * label_prob:.1f}: "
                                f"{_format_xmark(xmark)}"
                            )
                    
                            ax.axvline(
                                xmark,
                                color=color,
                                linestyle=percentile_linestyle_rnd,
                                alpha=percentile_alpha_rnd,
                                linewidth=percentile_linewidth_rnd,
                                label=legend_label,
                            )

                if any(m in percentile_measures for m in {"physical", "p"}):
                    if cdf_p_key in result:
                        cdf_p_surface = np.asarray(result[cdf_p_key], dtype=float)
                    else:
                        cdf_p_surface = _get_cdf_surface(
                            result,
                            kind="physical",
                            x_axis=x_axis,
                        )

                    cdf_p_ref = (
                        cdf_p_surface[j, :]
                        if same_x_grid
                        else _interp_row_to_ref(X, cdf_p_surface[j, :])
                    )

                    p_marks = _quantiles_from_cdf(X_ref, cdf_p_ref, probs)

                    for prob, xmark_raw in zip(probs, p_marks):
                        if np.isfinite(xmark_raw):
                    
                            if snap_percentiles_to_traded_strikes:
                                xmark = _snap_to_traded_x(xmark_raw, T_target_ref)
                                label_prob = _cdf_prob_at_x(X_ref, cdf_p_ref, xmark)
                            else:
                                xmark = xmark_raw
                                label_prob = prob
                    
                            legend_label = (
                                f"Physical p{100 * label_prob:.1f}: "
                                f"{_format_xmark(xmark)}"
                            )
                    
                            ax.axvline(
                                xmark,
                                color=color,
                                linestyle=percentile_linestyle_physical,
                                alpha=percentile_alpha_physical,
                                linewidth=percentile_linewidth_physical,
                                label=legend_label,
                            )

            if truncate_kernel:
                if trunc_kind == "physical":
                    F_surface = (
                        np.asarray(result[cdf_p_key], dtype=float)
                        if cdf_p_key in result
                        else _get_cdf_surface(
                            result,
                            kind="physical",
                            x_axis=x_axis,
                        )
                    )
                else:
                    F_surface = (
                        np.asarray(result[cdf_q_key], dtype=float)
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
                    else _interp_row_to_ref(X, F_surface[j, :])
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
                good = np.isfinite(Mk) & (Mk > 0)
                Xk = Xk[good]
                Mk = np.maximum(Mk[good], kernel_log_eps)

            if Xk.size > 1:
                ax2.plot(
                    Xk,
                    Mk,
                    linestyle=kernel_linestyle,
                    linewidth=lw_kernel,
                    alpha=alpha_kernel,
                    color=color,
                    label=m_legend if m_legend not in local_legend_seen else "_nolegend_",
                )

                local_legend_seen.add(m_legend)

        ax.set_title(_format_panel_title(T_target_ref))
        ax.grid(True, alpha=0.25)

        if k % ncols == 0:
            ax.set_ylabel("Density")

        if k >= len(idxs) - ncols:
            ax.set_xlabel(xlabel)

        if k % ncols == ncols - 1:
            ax2.set_ylabel("Pricing kernel")

        ax2.set_yscale(kernel_yscale)

        if x_bounds is not None:
            ax.set_xlim(X_plot[0], X_plot[-1])

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()

        handles = h1 + h2
        labels = l1 + l2

        unique = {}

        for h, lab in zip(handles, labels):
            if lab != "_nolegend_" and lab not in unique:
                unique[lab] = h

        ax.legend(
            unique.values(),
            unique.keys(),
            loc=legend_loc,
            fontsize=8,
            frameon=True,
        )

    for k in range(len(idxs), axes.size):
        axes[k].axis("off")

    if title is None:
        title = "Risk-Neutral Density, Physical Density, and Pricing Kernel"

    fig.suptitle(
        title,
        y=0.995,
        fontsize=14,
    )

    fig.subplots_adjust(
        top=0.90,
        wspace=0.28,
        hspace=0.34,
    )

    _maybe_save(fig, save, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_pit_calibration_panels(
    model,
    *,
    n_panels: int = 6,
    bins: int = 20,
    panel_shape: tuple[int, int] | None = None,
    title: str | None = None,
    save=None,
    dpi: int = 200,
    show: bool = True,
):
    """
    Plot PIT histogram overlaid with the fitted calibration density.

    Works for:
        - BetaCalibration
        - NonparametricCalibration

    Requires a fitted model with:
        model.fit_history_by_T_
        model.models_by_T_
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import beta as beta_dist
    from scipy.stats import norm

    if not getattr(model, "is_fitted_", False):
        raise RuntimeError("Model must be fitted before plotting PIT calibration panels.")

    Ts = sorted(model.models_by_T_.keys())
    if len(Ts) == 0:
        raise RuntimeError("No fitted maturity models found.")

    n_panels = min(int(n_panels), len(Ts))

    idxs = np.unique(
        np.linspace(0, len(Ts) - 1, n_panels).round().astype(int)
    )
    Ts_plot = [Ts[i] for i in idxs]

    if panel_shape is None:
        ncols = min(3, len(Ts_plot))
        nrows = int(np.ceil(len(Ts_plot) / ncols))
    else:
        nrows, ncols = panel_shape

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        sharex=True,
        sharey=False,
    )

    axes = np.asarray(axes).ravel()

    eps = float(getattr(model, "eps", 1e-10))
    u_grid = np.linspace(eps, 1.0 - eps, 1000)

    for ax, T in zip(axes, Ts_plot):
        hist = model.fit_history_by_T_.get(T)

        if hist is None or hist.empty:
            ax.set_title(f"T ≈ {365*T:.0f}d: no PITs")
            ax.axis("off")
            continue

        u = np.asarray(hist["pit"], dtype=float)
        u = u[np.isfinite(u)]
        u = np.clip(u, eps, 1.0 - eps)

        fitted = model.models_by_T_[T]

        ax.hist(
            u,
            bins=bins,
            range=(0.0, 1.0),
            density=True,
            alpha=0.35,
            edgecolor="black",
            label="PIT histogram",
        )

        method = str(getattr(model, "method_name", "")).lower()

        if hasattr(fitted, "a") and hasattr(fitted, "b"):
            g = beta_dist.pdf(u_grid, fitted.a, fitted.b)
            fitted_label = rf"Beta fit: $a={fitted.a:.3g}$, $b={fitted.b:.3g}$"

        elif hasattr(fitted, "z_grid") and hasattr(fitted, "h_grid"):
            z = norm.ppf(u_grid)
            h_z = np.interp(
                z,
                fitted.z_grid,
                fitted.h_grid,
                left=np.nan,
                right=np.nan,
            )
            phi_z = norm.pdf(z)
            g = h_z / np.maximum(phi_z, eps)
            g = np.where(np.isfinite(g) & (g >= 0), g, np.nan)

            fitted_label = (
                f"KDE fit: h={fitted.bandwidth:.3g}, "
                f"{fitted.bandwidth_method}"
            )

        else:
            raise TypeError(
                "Unsupported fitted model. Expected beta parameters "
                "(a,b) or nonparametric KDE fields (z_grid,h_grid)."
            )

        ax.plot(u_grid, g, linewidth=2.2, label=fitted_label)
        ax.axhline(1.0, linestyle="--", linewidth=1.2, label="Uniform benchmark")

        ax.set_title(f"T ≈ {365*T:.0f}d | n={len(u)}")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    for ax in axes[len(Ts_plot):]:
        ax.axis("off")

    if title is None:
        title = f"PIT Calibration Density Panels — {getattr(model, 'method_name', 'model')}"

    fig.suptitle(title)
    fig.tight_layout()

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

# Backward-compatible alias for your old function name.
def M_Q_K_multipanel_multi(*args, **kwargs):
    return plot_pqk_multipanel(*args, **kwargs)