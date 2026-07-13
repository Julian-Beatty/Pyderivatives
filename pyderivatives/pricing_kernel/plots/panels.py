from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np

from .utils import (
    _as_1d,
    _maybe_save,
    _get_meta_value,
    _title_suffix,
    _standardize_x_axis,
    _get_plot_x_grid,
    _surface_key,
    _get_cdf_surface,
    _slice_surface_bounds,
    _pick_panel_indices,
)


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
    Plot stacked maturity panels for one transformed result.

    Works for:
        - physical density
        - risk-neutral density
        - pricing kernel
        - relative risk aversion

    Parameters
    ----------
    result:
        Output dictionary from transform_info().

    kind:
        One of {"physical", "rnd", "pricing_kernel", "rra"}.

    x_axis:
        One of {"r", "R", "return", "K"}.

    truncate_by_alpha:
        If True, masks each panel outside the inner probability mass
        according to either the physical or risk-neutral CDF.
    """
    import matplotlib.pyplot as plt

    yscale = str(yscale).lower().strip()
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")

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
    a_left = a_right = None

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
            F = _get_cdf_surface(result, kind="physical", x_axis=x_axis)

        elif truncation_measure in {"risk_neutral", "risk-neutral", "q", "rnd"}:
            F = _get_cdf_surface(result, kind="rnd", x_axis=x_axis)

        else:
            raise ValueError("truncation_measure must be 'physical' or 'risk_neutral'.")

        F2 = F[np.ix_(tmask, xmask)]

        if F2.shape != Z2.shape:
            raise ValueError("Truncation CDF surface must have the same shape as the plotted surface.")

    idxs = _pick_panel_indices(T2, n_panels)
    nrows = len(idxs)

    fig_h = max(4.0, float(figsize_per_panel) * nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(8.5, fig_h),
        sharex=True,
    )

    if nrows == 1:
        axes = np.array([axes])

    # ------------------------------------------------------------
    # Reference line: r=0, R=1, return=0, or K=S0.
    # ------------------------------------------------------------
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
            ref_x = _get_meta_value(
                result,
                "S0",
                "s0",
                "spot",
                "spot_price",
                default=None,
            )
            ref_label = ref_label or r"$K=S_0$"

    show_ref_eff = (
        bool(show_ref)
        and ref_x is not None
        and np.isfinite(float(ref_x))
        and x2[0] <= float(ref_x) <= x2[-1]
    )

    # ------------------------------------------------------------
    # Draw panels.
    # ------------------------------------------------------------
    for ax, j in zip(axes, idxs):
        x = x2.copy()
        y = Z2[j, :].copy()

        line_label = y_label

        if truncate_by_alpha and F2 is not None:
            keep = (
                np.isfinite(F2[j, :])
                & (F2[j, :] >= float(a_left))
                & (F2[j, :] <= 1.0 - float(a_right))
            )

            y[~keep] = np.nan
            line_label += f" inner {100.0 * (1.0 - float(a_left) - float(a_right)):.0f}%"

        if yscale == "log":
            keep = np.isfinite(x) & np.isfinite(y) & (y > 0)
            x = x[keep]
            y = np.maximum(y[keep], float(ylog_eps))

        else:
            keep = np.isfinite(x) & np.isfinite(y)
            x = x[keep]
            y = y[keep]

        if x.size == 0:
            ax.text(
                0.5,
                0.5,
                "No finite points",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        else:
            ax.plot(x, y, linewidth=2.1, label=line_label)

        if show_ref_eff:
            ax.axvline(
                float(ref_x),
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
                label=ref_label,
            )

        ax.set_title(f"T ≈ {365.0 * float(T2[j]):.1f}d")
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


# ============================================================
# Convenience wrappers
# ============================================================

def plot_physical_density_panels(result: dict, **kwargs):
    """
    Maturity panels for the physical density.
    """
    return plot_surface_panels(result, kind="physical", **kwargs)


def plot_rnd_panels(result: dict, **kwargs):
    """
    Maturity panels for the risk-neutral density.
    """
    return plot_surface_panels(result, kind="rnd", **kwargs)


def plot_pricing_kernel_panels(result: dict, **kwargs):
    """
    Maturity panels for the pricing kernel.
    """
    return plot_surface_panels(result, kind="pricing_kernel", **kwargs)


def plot_rra_panels(result: dict, **kwargs):
    """
    Maturity panels for relative risk aversion.
    """
    return plot_surface_panels(result, kind="rra", **kwargs)