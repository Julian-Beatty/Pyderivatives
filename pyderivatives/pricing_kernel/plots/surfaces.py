from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


from .utils import (
    _as_1d,
    _maybe_save,
    _title_suffix,
    _get_plot_x_grid,
    _surface_key,
    _get_cdf_surface,
    _slice_surface_bounds,
    _standardize_x_axis,
)


# ============================================================
# Generic 3D surface plot for one transformed result
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
    Generic 3D surface plot.

    Works for:
        physical density
        risk-neutral density
        pricing kernel
        relative risk aversion
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
            f"{surface_key} must have shape {(T_grid.size, x_grid.size)}, got {Z.shape}."
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

        if title is None:
            title = default_title

        suffix = _title_suffix(result)
        if suffix:
            title = f"{title} — {suffix}"

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
# 3D surface through calendar time at fixed maturity
# ============================================================

def _standardize_kind(kind: str) -> str:
    """
    Canonicalize the surface kind.
    """
    key = str(kind).lower().strip()

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

    if key not in aliases:
        raise ValueError("kind must be one of {'pricing_kernel', 'physical', 'rnd', 'rra'}.")

    return aliases[key]


def _surface_key_for_time_plot(kind: str, x_axis: str) -> tuple[str, str, str]:
    """
    Map kind/x-axis to result key, z-label, and title for time-surface plots.
    """
    kind = _standardize_kind(kind)
    x_axis = _standardize_x_axis(x_axis)

    suffix = {
        "r": "lr",
        "R": "r",
        "return": "r",
        "K": "k",
    }[x_axis]

    if kind == "pricing_kernel":
        return "pricing_kernel_surface", r"$M$", "Pricing Kernel Through Time"

    if kind == "rra":
        return "relative_risk_aversion_surface", "Relative risk aversion", "Relative Risk Aversion Through Time"

    if kind == "physical":
        return f"physical_{suffix}_surface", {
            "r": r"$f_P(r)$",
            "R": r"$p_P(R)$",
            "return": r"$p_P(R)$",
            "K": r"$p_P(K)$",
        }[x_axis], "Physical Density Through Time"

    if kind == "rnd":
        return f"rnd_{suffix}_surface", {
            "r": r"$f_Q(r)$",
            "R": r"$p_Q(R)$",
            "return": r"$p_Q(R)$",
            "K": r"$p_Q(K)$",
        }[x_axis], "Risk-Neutral Density Through Time"

    raise RuntimeError("Unreachable kind.")


def _cdf_key_for_time_plot(kind: str, x_axis: str) -> Optional[str]:
    """
    Return CDF key for tail truncation in time-surface plots.
    """
    kind = _standardize_kind(kind)
    x_axis = _standardize_x_axis(x_axis)

    suffix = {
        "r": "lr",
        "R": "r",
        "return": "r",
        "K": "k",
    }[x_axis]

    if kind == "physical":
        return f"physical_cdf_{suffix}_surface"

    if kind == "rnd":
        return f"cdf_{suffix}_surface"

    return None


def _extract_x_for_time_plot(day: dict, x_axis: str):
    """
    Extract x-grid and label for a single transformed result.
    """
    x_axis = _standardize_x_axis(x_axis)

    if x_axis == "r":
        return np.asarray(day["grid_lr"], float).ravel(), r"Log return $r=\log(S_T/S_0)$"

    if x_axis == "R":
        return np.asarray(day["grid_r"], float).ravel(), r"Gross return $R=S_T/S_0$"

    if x_axis == "return":
        return np.asarray(day["grid_r"], float).ravel() - 1.0, r"Simple return $R-1$"

    if x_axis == "K":
        return np.asarray(day["grid_k"], float).ravel(), r"Terminal price / strike $K=S_T$"

    raise RuntimeError("Unreachable x_axis.")


def plot_surface_3d_by_T(
    result_dict: dict,
    T_target: float,
    *,
    kind: str = "pricing_kernel",
    x_axis: str = "r",
    x_bounds: Optional[Tuple[float, float]] = None,
    max_dates: int = 250,
    stride: Optional[int] = None,
    zscale: str = "log",
    z_eps: float = 1e-300,
    title: Optional[str] = None,
    cmap: str = "viridis",
    dpi: int = 200,
    interactive: bool = False,
    show: bool = True,
    alpha: float = 0.9,
    elev: float = 28,
    azim: float = -60,
    truncate_by_ptails: bool = False,
    ptail_alphas: Tuple[float, float] = (0.01, 0.01),
    truncation_measure: str = "physical",
    save=None,
    ax=None,
    add_colorbar: bool = True,
    clear_ax: bool = True,
):
    """
    Plot a 3D surface through calendar time at a fixed target maturity.

    Parameters
    ----------
    result_dict:
        Dictionary keyed by date. Each value should be a transformed result
        returned by transform_info().

    T_target:
        Target maturity in years. The nearest available maturity is used for
        each date.

    kind:
        pricing_kernel, physical, rnd, or rra.

    x_axis:
        r, R, return, or K.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    kind = _standardize_kind(kind)
    x_axis = _standardize_x_axis(x_axis)
    zscale = str(zscale).lower().strip()

    if zscale not in {"linear", "log"}:
        raise ValueError("zscale must be 'linear' or 'log'.")

    if not result_dict:
        raise ValueError("result_dict is empty.")

    items = sorted(result_dict.items(), key=lambda kv: pd.Timestamp(kv[0]))

    if stride is None:
        stride = max(1, int(np.ceil(len(items) / max_dates)))

    items = items[::int(stride)]

    dates = []
    rows = []
    x_ref = None
    xlabel = None

    surface_key, zlabel, default_title = _surface_key_for_time_plot(kind, x_axis)

    for raw_date, day in items:
        if day is None or not isinstance(day, dict):
            continue

        if surface_key not in day or "T_grid" not in day:
            continue

        T_grid = np.asarray(day["T_grid"], float).ravel()
        Z = np.asarray(day[surface_key], float)

        if T_grid.size == 0 or Z.ndim != 2:
            continue

        idx_T = int(np.argmin(np.abs(T_grid - float(T_target))))

        x_grid, xlabel_i = _extract_x_for_time_plot(day, x_axis=x_axis)

        if Z.shape[1] != x_grid.size:
            continue

        z_row = Z[idx_T, :].astype(float)

        # Optional tail truncation using physical or RND CDFs.
        if truncate_by_ptails:
            a_left, a_right = map(float, ptail_alphas)

            if not (
                0.0 <= a_left < 1.0
                and 0.0 <= a_right < 1.0
                and a_left + a_right < 1.0
            ):
                raise ValueError("ptail_alphas must satisfy 0 <= left,right < 1 and left+right < 1.")

            trunc_kind = "physical" if str(truncation_measure).lower() in {"physical", "p"} else "rnd"
            cdf_key = _cdf_key_for_time_plot(trunc_kind, x_axis)

            if cdf_key is not None and cdf_key in day:
                F = np.asarray(day[cdf_key], float)
                if F.shape == Z.shape:
                    F_row = F[idx_T, :]
                    keep = (F_row >= a_left) & (F_row <= 1.0 - a_right)
                    z_row = np.where(keep, z_row, np.nan)

        if x_bounds is not None:
            lo, hi = sorted(map(float, x_bounds))
            mask_x = np.isfinite(x_grid) & (x_grid >= lo) & (x_grid <= hi)
        else:
            mask_x = np.isfinite(x_grid)

        if not np.any(mask_x):
            continue

        x_use = x_grid[mask_x]
        z_use = z_row[mask_x]

        if x_ref is None:
            x_ref = x_use
            xlabel = xlabel_i
        else:
            # Interpolate each date onto the first valid x-grid.
            good = np.isfinite(x_use) & np.isfinite(z_use)
            if good.sum() < 2:
                continue
            z_use = np.interp(x_ref, x_use[good], z_use[good], left=np.nan, right=np.nan)

        if x_ref is not None and z_use.size == x_ref.size:
            dates.append(pd.Timestamp(raw_date))
            rows.append(z_use)

    if not rows:
        raise ValueError("No valid surfaces found for plotting.")

    X = np.asarray(x_ref, float)
    Y_dates = pd.to_datetime(dates)
    Y_num = mdates.date2num(Y_dates.to_pydatetime())
    Z = np.vstack(rows)

    if zscale == "log":
        Z_plot = np.where(np.isfinite(Z) & (Z > z_eps), np.log(Z), np.nan)
        zlabel_plot = rf"$\log$ {zlabel}"
    else:
        Z_plot = Z
        zlabel_plot = zlabel

    if title is None:
        title = default_title

    target_days = 365.0 * float(T_target)
    title = f"{title} — T≈{target_days:.0f}d"

    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=[d.strftime("%Y-%m-%d") for d in Y_dates],
                    z=Z_plot,
                    colorscale=cmap,
                    opacity=alpha,
                    colorbar=dict(title=zlabel_plot),
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title="Date",
                zaxis_title=zlabel_plot,
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

    X_mesh, Y_mesh = np.meshgrid(X, Y_num)

    created_fig = False

    if ax is None:
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure
        if clear_ax:
            ax.clear()

    surf = ax.plot_surface(
        X_mesh,
        Y_mesh,
        Z_plot,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=alpha,
    )

    if add_colorbar:
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label=zlabel_plot)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Date")
    ax.set_zlabel(zlabel_plot)

    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    _maybe_save(fig, save, dpi=dpi)

    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)

    return fig, ax, surf


def plot_pricing_kernel_3d_surface_by_T(*args, **kwargs):
    """
    Convenience wrapper for a pricing-kernel time surface.
    """
    kwargs.setdefault("kind", "pricing_kernel")
    return plot_surface_3d_by_T(*args, **kwargs)