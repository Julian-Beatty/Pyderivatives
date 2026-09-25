from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
from functools import wraps
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SavePath = Optional[Union[str, Path]]


def _option_sides(func):
    """Dispatch joint payloads; legacy/single-side calls still return one figure.

    option_right='both' (default) returns {'c': figure, 'p': figure} for mixed
    data. Saved mixed plots append _call/_put before the filename extension.
    """
    @wraps(func)
    def wrapped(plot_data, *, option_right="both", **kwargs):
        requested = {"call": "c", "put": "p", "calls": "c", "puts": "p"}.get(option_right, option_right)
        if requested not in {"both", "c", "p"}:
            raise ValueError("option_right must be 'both', 'c', or 'p'.")
        sides = plot_data.get("by_right", {plot_data.get("meta", {}).get("option_right", "c"): plot_data})
        chosen = list(sides) if requested == "both" else [requested]
        results = {}
        for right in chosen:
            if right not in sides:
                raise ValueError(f"No observed quotes for option right {right!r}.")
            args = dict(kwargs)
            label = "Call" if right == "c" else "Put"
            args["title"] = f"{label}: {args.get('title', func.__name__.replace('plot_', '').title())}"
            if len(chosen) > 1 and args.get("save") is not None:
                path = Path(args["save"])
                default_suffix = ".html" if args.get("interactive", False) else ".png"
                args["save"] = path.with_name(f"{path.stem}_{label.lower()}{path.suffix or default_suffix}")
            results[right] = func(sides[right], **args)
        return results if len(chosen) > 1 else results[chosen[0]]
    return wrapped


def _save_or_return(fig: plt.Figure, *, save: SavePath, dpi: int):
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")
        plt.close(fig)
        return None
    return fig


@_option_sides
def plot_repair_surface(plot_data: Dict[str, Any], *, title: str = "Observed vs Repaired", save: SavePath = None, dpi: int = 160, interactive: bool = False):
    """Compare observed/repaired surfaces, with maturity increasing left to right.

    interactive=True returns Plotly figures (call .show()) and saves standalone
    HTML when save is supplied. Mixed data returns a c/p dictionary; select
    option_right='c' or 'p' to return one figure. Default is Matplotlib.
    """
    T = np.asarray(plot_data["raw"]["T"], float)
    K = np.asarray(plot_data["raw"]["K"], float)
    C_obs = np.asarray(plot_data["raw"]["C_obs"], float)
    C_rep = np.asarray(plot_data["raw"]["C_rep"], float)

    order = np.lexsort((K, T))
    T, K, C_obs, C_rep = T[order], K[order], C_obs[order], C_rep[order]
    price_label = plot_data["meta"].get("option_right", "c").upper()
    t_min, t_max = float(T.min()), float(T.max())
    if t_min == t_max:
        pad = max(abs(t_min) * 0.05, 1e-6)
        t_min, t_max = t_min - pad, t_max + pad

    # Duplicate quote locations and single-expiry chains cannot be triangulated.
    points, inverse = np.unique(np.column_stack([T, K]), axis=0, return_inverse=True)
    tri = None
    if len(points) >= 3 and np.linalg.matrix_rank(points - points[0]) == 2:
        tri = Triangulation(points[:, 0], points[:, 1])

    if interactive:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                            subplot_titles=("Observed", "Repaired"))
        for col, Z in enumerate((C_obs, C_rep), start=1):
            if tri is not None:
                means = np.bincount(inverse, weights=Z) / np.bincount(inverse)
                triangles = tri.triangles
                fig.add_trace(go.Mesh3d(
                    x=points[:, 0], y=points[:, 1], z=means,
                    i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                    intensity=means, colorscale="Viridis", opacity=0.75,
                    showscale=False, hoverinfo="skip", showlegend=False,
                ), row=1, col=col)
            fig.add_trace(go.Scatter3d(
                x=T, y=K, z=Z, mode="markers", marker=dict(size=3, color="black"),
                hovertemplate="T=%{x:.6f} years<br>K=%{y}<br>Price=%{z:.6f}<extra></extra>",
                showlegend=False,
            ), row=1, col=col)
        scene = dict(xaxis=dict(title="T (years)", range=[t_min, t_max], autorange=False),
                     yaxis=dict(title="K"), zaxis=dict(title=price_label),
                     camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0)))
        fig.update_layout(title=title, scene=scene, scene2=scene, height=600)
        if save is not None:
            path = Path(save)
            if not path.suffix:
                path = path.with_suffix(".html")
            if path.suffix.lower() not in {".html", ".htm"}:
                raise ValueError("Interactive surface plots must be saved as .html or .htm.")
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(path), include_plotlyjs=True, auto_open=False)
            return None
        return fig

    fig = plt.figure(figsize=(14, 6))
    for i, Z in enumerate([C_obs, C_rep]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        if tri is None:
            ax.scatter(T, K, Z)
        else:
            means = np.bincount(inverse, weights=Z) / np.bincount(inverse)
            ax.plot_trisurf(tri, means, alpha=0.75)
            ax.scatter(T, K, Z, s=8, color="black")
        ax.set_title("Observed" if i == 0 else "Repaired")
        ax.view_init(elev=20, azim=-60)
        ax.set_xlim(t_min, t_max)
        ax.set_xlabel("T (years)")
        ax.set_ylabel("K")
        ax.set_zlabel(price_label)

    fig.suptitle(title)
    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


# Compatibility for explicit imports from pyderivatives.arbitrage_repair.
# The top-level package exports plot_repair_surface without shadowing kernel plots.
plot_surface = plot_repair_surface


@_option_sides
def plot_panels(plot_data: Dict[str, Any], *, title: str = "Panels", save: SavePath = None, dpi: int = 160, n_panels: int = 6, show_spreads: bool = False, spread_every: int = 1):
    """Original/repaired curves with optional original bid-to-ask capped bars.

    spread_every displays every nth valid spread within each sorted maturity.
    Bars stay anchored to the original bid/ask, even if repairs lie outside them.
    """
    T = np.asarray(plot_data["raw"]["T"], float)
    K = np.asarray(plot_data["raw"]["K"], float)
    C_obs = np.asarray(plot_data["raw"]["C_obs"], float)
    C_rep = np.asarray(plot_data["raw"]["C_rep"], float)

    if not isinstance(spread_every, (int, np.integer)) or spread_every < 1:
        raise ValueError("spread_every must be a positive integer.")
    if show_spreads:
        if not {"bid", "ask"}.issubset(plot_data["raw"]):
            raise ValueError("Bid/ask data missing from plot_data. Rerun repair with bid/ask columns and the updated package.")
        bid = np.asarray(plot_data["raw"]["bid"], float)
        ask = np.asarray(plot_data["raw"]["ask"], float)
        if bid.shape != T.shape or ask.shape != T.shape:
            raise ValueError("Bid/ask arrays must match the quote arrays.")

    Ts = np.sort(np.unique(T))[: max(1, int(n_panels))]
    fig, ax = plt.subplots(len(Ts), 2, figsize=(12, 2.6 * len(Ts)))
    ax = np.atleast_2d(ax)

    s0 = float(plot_data["meta"]["s0"])
    has_s0 = bool(plot_data["meta"]["has_s0"])

    for i, Ti in enumerate(Ts):
        m = (T == Ti)
        if not np.any(m):
            continue
        o = np.argsort(K[m])

        ax[i, 0].plot(K[m][o], C_obs[m][o], label="Observed")
        ax[i, 0].set_title(f"T={Ti:.3f} Obs")
        ax[i, 0].set_xlabel("K")
        ax[i, 0].set_ylabel("C")

        ax[i, 1].plot(K[m][o], C_rep[m][o])
        ax[i, 1].plot(K[m][o], C_obs[m][o], "--", alpha=0.7, label="Observed")
        ax[i, 1].lines[0].set_label("Repaired")
        ax[i, 1].legend()
        ax[i, 1].set_title(f"T={Ti:.3f} Rep")
        ax[i, 1].set_xlabel("K")
        ax[i, 1].set_ylabel("C")
        for j in (0, 1):
            ax[i, j].set_ylabel(plot_data["meta"].get("option_right", "c").upper())

        if show_spreads:
            indices = np.flatnonzero(m)[o]
            valid = np.isfinite(bid[indices]) & np.isfinite(ask[indices]) & (bid[indices] >= 0) & (bid[indices] <= ask[indices])
            indices = indices[valid][::spread_every]
            center = 0.5 * (bid[indices] + ask[indices])
            half_width = 0.5 * (ask[indices] - bid[indices])
            for j in (0, 1):
                ax[i, j].errorbar(K[indices], center, yerr=half_width,
                                 fmt="none", ecolor="0.35", elinewidth=0.9,
                                 capsize=2, alpha=0.65, zorder=1,
                                 label="Original bid–ask")
                ax[i, j].legend()

        if has_s0 and np.isfinite(s0):
            for j in (0, 1):
                ax[i, j].axvline(s0, ls="--", lw=1)

    fig.suptitle(title)
    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


# Distinct public name; retain plot_panels for existing callers.
plot_repair_panels = plot_panels


@_option_sides
def plot_perturb(plot_data: Dict[str, Any], *, title: str = "Perturbation", save: SavePath = None, dpi: int = 160):
    T = np.asarray(plot_data["raw"]["T"], float)
    K = np.asarray(plot_data["raw"]["K"], float)
    y = np.asarray(plot_data["perturb"]["y"], float)
    ylabel = str(plot_data["perturb"]["ylabel"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(K, y, c=T, cmap="viridis")
    ax.axhline(0, lw=1)
    fig.colorbar(sc, ax=ax, label="T (years)")

    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


@_option_sides
def plot_term(plot_data: Dict[str, Any], *, title: str = "Exact-K Term Structures", save: SavePath = None, dpi: int = 160, ncols: int = 3):
    T = np.asarray(plot_data["raw"]["T"], float)
    K = np.asarray(plot_data["raw"]["K"], float)
    C_obs = np.asarray(plot_data["raw"]["C_obs"], float)
    C_rep = np.asarray(plot_data["raw"]["C_rep"], float)

    strikes = plot_data.get("term", {}).get("strikes", [])
    strikes = [float(x) for x in strikes if np.isfinite(float(x))]

    if len(strikes) == 0:
        fig = plt.figure(figsize=(6, 4))
        plt.title("No strikes appear across multiple maturities.")
        plt.axis("off")
        return _save_or_return(fig, save=save, dpi=dpi)

    n = len(strikes)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, K0 in enumerate(strikes):
        ax = axes[i]
        m = (K == K0)
        if not np.any(m):
            ax.axis("off")
            continue

        Ti = T[m]
        obs_i = C_obs[m]
        rep_i = C_rep[m]

        Tu = np.unique(Ti)
        Tu.sort()
        obs_mean = np.array([np.mean(obs_i[Ti == t]) for t in Tu], float)
        rep_mean = np.array([np.mean(rep_i[Ti == t]) for t in Tu], float)

        ax.plot(Tu, obs_mean, "o-", label="Obs")
        ax.plot(Tu, rep_mean, "o-", label="Rep")
        ax.set_title(f"K={K0:.2f}")
        ax.set_xlabel("T")
        ax.set_ylabel(plot_data["meta"].get("option_right", "c").upper())
        ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


@_option_sides
def plot_heatmap(plot_data: Dict[str, Any], *, title: str = "Repaired − Observed", save: SavePath = None, dpi: int = 160, eps: float = 1e-10, interpolation: str = "nearest", color_scale: str = "symlog", linthresh: float = 0.01, change_threshold: float = 0.01, changed_only: bool = False, dot_size: float = 16):
    """Colored dots at observed (maturity, strike) locations, in price units.

    Color is repaired minus observed, with a symmetric scale centered at zero.
    By default the color scale is linear within +/-0.01 and logarithmic outside
    that band; colorbar ticks remain in price units. Grey dots are changes at or
    below change_threshold. Summary statistics cover all finite (T,K) locations,
    even when changed_only=True. These thresholds only affect visualization.
    eps and interpolation are retained for call compatibility; no ratio or
    grid interpolation is used. Duplicate locations use the payload's means.
    """
    from matplotlib.colors import Normalize, SymLogNorm
    if color_scale not in {"linear", "symlog"}:
        raise ValueError("color_scale must be 'linear' or 'symlog'.")
    if not np.isfinite(linthresh) or linthresh <= 0:
        raise ValueError("linthresh must be positive and finite.")
    if not np.isfinite(change_threshold) or change_threshold < 0:
        raise ValueError("change_threshold must be nonnegative and finite.")
    if not np.isfinite(dot_size) or dot_size <= 0:
        raise ValueError("dot_size must be positive and finite.")
    h = plot_data.get("heatmap", {})
    tmp = h.get("table", None)

    if tmp is None or len(tmp) == 0:
        fig = plt.figure(figsize=(6, 4))
        plt.title("Heatmap: no valid rows.")
        plt.axis("off")
        return _save_or_return(fig, save=save, dpi=dpi)

    tmp = tmp.copy()
    C_col = h.get("C_col", None)
    if C_col is None or C_col not in tmp.columns:
        C_col = tmp.columns[2]

    tmp["adjustment"] = tmp["C_rep"].to_numpy(float) - tmp[C_col].to_numpy(float)

    K_col = h.get("K_col", tmp.columns[0])
    T_col = h.get("T_col", tmp.columns[1])

    tmp = tmp.loc[np.isfinite(tmp[[T_col, K_col, "adjustment"]].to_numpy(float)).all(axis=1)]
    if tmp.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Heatmap: no valid rows.")
        ax.axis("off")
        return _save_or_return(fig, save=save, dpi=dpi)
    T_vals = tmp[T_col].to_numpy(float)
    K_vals = tmp[K_col].to_numpy(float)
    adjustments = tmp["adjustment"].to_numpy(float)
    limit = float(np.max(np.abs(adjustments)))
    limit = max(limit, linthresh)
    changed = np.abs(adjustments) > change_threshold
    normalization = (SymLogNorm(linthresh=linthresh, vmin=-limit, vmax=limit)
                     if color_scale == "symlog" else Normalize(vmin=-limit, vmax=limit))

    fig, ax = plt.subplots(figsize=(12, 7))
    if not changed_only:
        ax.scatter(T_vals[~changed], K_vals[~changed], color="0.83", s=dot_size/2,
                   edgecolors="none", label=f"|change| ≤ {change_threshold:g}")
    # Larger adjustments drawn last so dense clusters cannot hide them.
    selected = np.flatnonzero(changed)
    selected = selected[np.argsort(np.abs(adjustments[selected]))]
    dots = ax.scatter(T_vals[selected], K_vals[selected], c=adjustments[selected], cmap="RdBu_r",
                      norm=normalization, s=dot_size, edgecolors="none",
                      label=f"|change| > {change_threshold:g}")
    scale_label = "symlog" if color_scale == "symlog" else "linear"
    fig.colorbar(dots, ax=ax, label=f"Repaired − Observed (price units; {scale_label} scale)")
    if changed_only:
        # Keep the same domain even when no locations exceed the threshold.
        ax.update_datalim(np.column_stack([T_vals, K_vals]))
        ax.autoscale_view()
    ax.legend(loc="upper left", fontsize=9)
    count = int(changed.sum())
    summary = (f"{count:,}/{len(adjustments):,} locations ({count/len(adjustments):.1%}) with |change| > {change_threshold:g}\n"
               f"All locations: mean |change| = {np.mean(np.abs(adjustments)):.4g}  ·  "
               f"median = {np.median(np.abs(adjustments)):.4g}  ·  "
               f"max = {np.max(np.abs(adjustments)):.4g} price units")
    fig.text(0.08, 0.025, summary, fontsize=10, va="bottom")

    ax.set_title(title)
    ax.set_xlabel("T (years)")
    ax.set_ylabel("K")

    has_s0 = bool(plot_data["meta"]["has_s0"])
    s0 = float(plot_data["meta"]["s0"])
    if has_s0 and np.isfinite(s0):
        ax.axhline(s0, ls="--", lw=1)
        ax.text(np.min(T_vals), s0, f" S0={s0:.2f}", va="bottom", ha="left", fontsize=10)

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return _save_or_return(fig, save=save, dpi=dpi)
