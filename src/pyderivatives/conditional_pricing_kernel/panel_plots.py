from __future__ import annotations
from .eval import evaluate_anchor_surfaces_with_theta_master

import os
import numpy as np

def plot_kernel_bootstrap_quantiles_panel(
    *,
    anchor_day: dict,
    pk_fit: dict,
    theta_spec,
    eval_spec,
    j_indices=None,
    n_panels: int = 6,
    panel_shape: tuple[int, int] = (2, 3),
    safety_clip=None,
    q_hi: float = 0.95,
    q_lo: float | None = 0.05,
    show_q_lines: bool = True,
    show_band: bool = True,
    q_band: tuple[float, float] = (0.025, 0.975),
    R_bounds=None,
    r_bounds=None,
    logy: bool = True,
    title: str | None = None,
    save=None,
    dpi: int = 200,
    # ---- NEW ----
    log_floor: float = 1e-12,
    auto_ylim: bool = True,
    ylim_pad: float = 1.15,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    T_grid = np.asarray(pk_fit["T_grid"], float).ravel()
    nT = T_grid.size
    if nT == 0:
        raise ValueError("pk_fit['T_grid'] is empty.")

    if j_indices is None:
        j_indices = np.linspace(0, nT - 1, int(n_panels), dtype=int)
    else:
        j_indices = np.asarray(j_indices, int).ravel()

    base = evaluate_anchor_surfaces_with_theta_master(
        anchor_day,
        theta_master=pk_fit["theta_master"],
        theta_spec=theta_spec,
        eval_spec=eval_spec,
        safety_clip=safety_clip,
    )

    x = np.asarray(base["R_common"], float).ravel()
    r = np.asarray(base["r_common"], float).ravel()
    if x.size == 0:
        raise ValueError("base['R_common'] is empty.")

    mask = np.isfinite(x)
    if R_bounds is not None:
        Rmin, Rmax = map(float, R_bounds)
        mask &= (x >= Rmin) & (x <= Rmax)
    if r_bounds is not None:
        rmin, rmax = map(float, r_bounds)
        mask &= (r >= rmin) & (r <= rmax)

    xm = x[mask]

    nrows, ncols = panel_shape
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,   # <-- fixes label/title clipping
    )
    axes = np.asarray(axes).ravel()

    draws_by_T = pk_fit.get("theta_boot_draws_by_T", None)

    # ---- NEW: track global y-lims (after log floor) ----
    ymins, ymaxs = [], []

    for ax, j in zip(axes, j_indices):
        j = int(j)
        if not (0 <= j < nT):
            ax.set_visible(False)
            continue

        M_hat = np.asarray(base["anchor_surfaces"]["M_surface"][j], float).ravel()

        if draws_by_T is None or draws_by_T[j] is None:
            ax.set_visible(False)
            continue

        theta_draws = np.asarray(draws_by_T[j], float)
        if theta_draws.ndim != 2 or theta_draws.shape[0] == 0:
            ax.set_visible(False)
            continue

        B = theta_draws.shape[0]

        M_boot = np.empty((B, x.size), float)
        for b in range(B):
            theta_tmp = np.array(pk_fit["theta_master"], float, copy=True)
            theta_tmp[j, :] = theta_draws[b, :]

            out_b = evaluate_anchor_surfaces_with_theta_master(
                anchor_day,
                theta_master=theta_tmp,
                theta_spec=theta_spec,
                eval_spec=eval_spec,
                safety_clip=safety_clip,
            )
            M_boot[b, :] = np.asarray(out_b["anchor_surfaces"]["M_surface"][j], float).ravel()

        lo_band_q, hi_band_q = float(q_band[0]), float(q_band[1])
        if not (0.0 <= lo_band_q < hi_band_q <= 1.0):
            raise ValueError("q_band must satisfy 0 <= q_band[0] < q_band[1] <= 1.")

        qlo_line = None if q_lo is None else float(q_lo)
        qhi_line = None if q_hi is None else float(q_hi)

        if show_band:
            lo_band = np.quantile(M_boot, lo_band_q, axis=0)
            hi_band = np.quantile(M_boot, hi_band_q, axis=0)

        if show_q_lines:
            q_hi_curve = np.quantile(M_boot, qhi_line, axis=0) if qhi_line is not None else None
            q_lo_curve = np.quantile(M_boot, qlo_line, axis=0) if qlo_line is not None else None

        # apply truncation + log-floor
        Mh = M_hat[mask]
        if logy:
            Mh = np.maximum(Mh, float(log_floor))

        ax.plot(xm, Mh, linewidth=2.0, label="M (point)")

        if show_band:
            lo_m = lo_band[mask]
            hi_m = hi_band[mask]
            if logy:
                lo_m = np.maximum(lo_m, float(log_floor))
                hi_m = np.maximum(hi_m, float(log_floor))
            ax.fill_between(xm, lo_m, hi_m, alpha=0.25, label="band")

        if show_q_lines:
            if q_hi_curve is not None:
                qh = q_hi_curve[mask]
                if logy:
                    qh = np.maximum(qh, float(log_floor))
                ax.plot(xm, qh, linewidth=1.8, linestyle="--", label=f"{int(qhi_line*100)}%")
            if q_lo_curve is not None:
                ql = q_lo_curve[mask]
                if logy:
                    ql = np.maximum(ql, float(log_floor))
                ax.plot(xm, ql, linewidth=1.8, linestyle="--", label=f"{int(qlo_line*100)}%")

        ax.set_title(f"T ≈ {T_grid[j]*365:.0f}d")
        if logy:
            ax.set_yscale("log")

        # track global y-range from what's actually plotted
        if auto_ylim:
            ys = []
            ys.append(Mh)
            if show_band:
                ys.append(lo_m); ys.append(hi_m)
            if show_q_lines:
                if q_hi_curve is not None: ys.append(qh)
                if q_lo_curve is not None: ys.append(ql)
            ycat = np.concatenate([np.asarray(v, float).ravel() for v in ys if v is not None])
            ycat = ycat[np.isfinite(ycat) & (ycat > 0)]
            if ycat.size:
                ymins.append(float(np.min(ycat)))
                ymaxs.append(float(np.max(ycat)))

    used = min(len(j_indices), axes.size)
    for ax in axes[used:]:
        ax.set_visible(False)

    # set a stable y-limit across panels (prevents “cut off bottom” look)
    if logy and auto_ylim and len(ymins) and len(ymaxs):
        ylo = min(ymins)
        yhi = max(ymaxs)
        # pad a bit
        ylo = max(float(log_floor), ylo / float(ylim_pad))
        yhi = yhi * float(ylim_pad)
        for ax in axes:
            if ax.get_visible():
                ax.set_ylim(ylo, yhi)

    legend_ax = next((ax for ax in axes if ax.get_visible()), None)
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc="best")

    fig.suptitle(title or "Bootstrap pricing kernel quantiles (by maturity)")
    fig.supxlabel("Gross return R")
    fig.supylabel("Pricing kernel M(R)")

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {save}")

    plt.show()
    return fig



def _slice_bounds_2d(
    T_grid: np.ndarray,
    R_grid: np.ndarray,
    Z: np.ndarray,
    *,
    T_bounds: tuple[float, float] | None = None,
    R_bounds: tuple[float, float] | None = None,
):
    T_grid = np.asarray(T_grid, float).ravel()
    R_grid = np.asarray(R_grid, float).ravel()
    Z = np.asarray(Z, float)

    if Z.shape != (T_grid.size, R_grid.size):
        raise ValueError(f"Surface shape mismatch: Z {Z.shape} vs (len(T),len(R))={(T_grid.size, R_grid.size)}")

    tmask = np.isfinite(T_grid)
    rmask = np.isfinite(R_grid)

    if T_bounds is not None:
        lo, hi = sorted(map(float, T_bounds))
        tmask &= (T_grid >= lo) & (T_grid <= hi)

    if R_bounds is not None:
        lo, hi = sorted(map(float, R_bounds))
        rmask &= (R_grid >= lo) & (R_grid <= hi)

    T_sel = T_grid[tmask]
    R_sel = R_grid[rmask]
    Z_sel = Z[np.ix_(tmask, rmask)]
    return T_sel, R_sel, Z_sel


def _maybe_save_matplotlib(fig, save: str | None, dpi: int = 200):
    if save is None:
        return
    save = str(save)
    folder = os.path.dirname(save) or "."
    os.makedirs(folder, exist_ok=True)
    fig.savefig(save, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {save}")


# def physical_density_surface_plot(
#     out: dict,
#     *,
#     title: str = "Physical Density Surface",
#     cmap: str = "viridis",          # plotly colorscale name OR matplotlib cmap name
#     save: str | None = None,        # .html (plotly) OR .png/.pdf/etc (mpl or plotly)
#     interactive: bool = True,
#     show: bool = True,
#     R_bounds: tuple[float, float] | None = None,
#     T_bounds: tuple[float, float] | None = None,
#     dpi: int = 200,
# ):
#     """
#     Plot physical density surface p(R|T).

#     Requires (in `out`):
#       out["anchor_surfaces"]["pR_surface"]
#       out["R_common"]
#       out["T_anchor"]

#     interactive=True  -> Plotly 3D surface
#     interactive=False -> Matplotlib 3D surface
#     """

#     anchor = out.get("anchor_surfaces", None)
#     if anchor is None or "pR_surface" not in anchor:
#         raise KeyError("Expected out['anchor_surfaces']['pR_surface'].")

#     p = np.asarray(anchor["pR_surface"], float)
#     R_grid = np.asarray(out.get("R_common", []), float).ravel()
#     T_grid = np.asarray(out.get("T_anchor", []), float).ravel()

#     if R_grid.size == 0 or T_grid.size == 0:
#         raise KeyError("Expected out['R_common'] and out['T_anchor'].")

#     if p.shape != (T_grid.size, R_grid.size):
#         raise ValueError("pR_surface must have shape (len(T_anchor), len(R_common)).")

#     T_plot, R_plot, p_plot = _slice_bounds_2d(T_grid, R_grid, p, T_bounds=T_bounds, R_bounds=R_bounds)
#     if T_plot.size < 2 or R_plot.size < 2:
#         raise ValueError("Not enough points to plot after applying bounds.")

#     date_label = out.get("anchor_key_used", out.get("anchor_date_used", ""))
#     plot_title = f"{title} — {date_label}" if date_label else title

#     # -------------------- Matplotlib branch --------------------
#     if not interactive:
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

#         R_mesh, T_mesh = np.meshgrid(R_plot, T_plot)

#         fig = plt.figure(figsize=(9.5, 6.5))
#         ax = fig.add_subplot(111, projection="3d")
#         surf = ax.plot_surface(R_mesh, T_mesh, p_plot, cmap=cmap, linewidth=0, antialiased=True)

#         ax.set_title(plot_title)
#         ax.set_xlabel("Gross return R")
#         ax.set_ylabel("Maturity T (years)")
#         ax.set_zlabel("p(R|T)")

#         cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
#         cbar.set_label("p(R)")

#         fig.tight_layout()
#         _maybe_save_matplotlib(fig, save, dpi=dpi)

#         if show:
#             plt.show()
#         return fig

#     # -------------------- Plotly branch --------------------
#     import plotly.graph_objects as go

#     R_mesh, T_mesh = np.meshgrid(R_plot, T_plot)

#     fig = go.Figure(
#         data=[
#             go.Surface(
#                 x=R_mesh,
#                 y=T_mesh,
#                 z=p_plot,
#                 colorscale=cmap,
#                 colorbar=dict(title="p(R)"),
#             )
#         ]
#     )

#     fig.update_layout(
#         title=plot_title,
#         scene=dict(
#             xaxis_title="Gross return R",
#             yaxis_title="Maturity T (years)",
#             zaxis_title="p(R|T)",
#         ),
#         margin=dict(l=0, r=0, t=50, b=0),
#     )

#     if save is not None:
#         save = str(save)
#         folder = os.path.dirname(save) or "."
#         os.makedirs(folder, exist_ok=True)

#         if save.lower().endswith(".html"):
#             fig.write_html(save)
#             print(f"[saved] {save}")
#         else:
#             # Image export: requires `pip install -U kaleido`
#             fig.write_image(save)
#             print(f"[saved] {save}")

#     if show:
#         fig.show()

#     return fig


def rra_surface_plot(
    out: dict,
    *,
    title: str = "Relative Risk Aversion Surface",
    cmap: str = "viridis",
    save: str | None = None,
    interactive: bool = True,
    show: bool = True,
    R_bounds: tuple[float, float] | None = None,
    T_bounds: tuple[float, float] | None = None,
    dpi: int = 200,
):
    """
    Plot RRA surface RRA(R|T).

    Requires (in `out`):
      out["anchor_surfaces"]["RRA_surface"]
      out["R_common"]
      out["T_anchor"]

    interactive=True  -> Plotly 3D surface
    interactive=False -> Matplotlib 3D surface
    """

    anchor = out.get("anchor_surfaces", None)
    if anchor is None or "RRA_surface" not in anchor:
        raise KeyError("Expected out['anchor_surfaces']['RRA_surface'].")

    RRA = np.asarray(anchor["RRA_surface"], float)
    R_grid = np.asarray(out.get("R_common", []), float).ravel()
    T_grid = np.asarray(out.get("T_anchor", []), float).ravel()

    if R_grid.size == 0 or T_grid.size == 0:
        raise KeyError("Expected out['R_common'] and out['T_anchor'].")

    if RRA.shape != (T_grid.size, R_grid.size):
        raise ValueError("RRA_surface must have shape (len(T_anchor), len(R_common)).")

    T_plot, R_plot, RRA_plot = _slice_bounds_2d(T_grid, R_grid, RRA, T_bounds=T_bounds, R_bounds=R_bounds)
    if T_plot.size < 2 or R_plot.size < 2:
        raise ValueError("Not enough points to plot after applying bounds.")

    date_label = out.get("anchor_key_used", out.get("anchor_date_used", ""))
    plot_title = f"{title} — {date_label}" if date_label else title

    # -------------------- Matplotlib branch --------------------
    if not interactive:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        R_mesh, T_mesh = np.meshgrid(R_plot, T_plot)

        fig = plt.figure(figsize=(9.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(R_mesh, T_mesh, RRA_plot, cmap=cmap, linewidth=0, antialiased=True)

        ax.set_title(plot_title)
        ax.set_xlabel("Gross return R")
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel("RRA(R|T)")

        cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label("RRA")

        fig.tight_layout()
        _maybe_save_matplotlib(fig, save, dpi=dpi)

        if show:
            plt.show()
        return fig

    # -------------------- Plotly branch --------------------
    import plotly.graph_objects as go

    R_mesh, T_mesh = np.meshgrid(R_plot, T_plot)

    fig = go.Figure(
        data=[
            go.Surface(
                x=R_mesh,
                y=T_mesh,
                z=RRA_plot,
                colorscale=cmap,
                colorbar=dict(title="RRA"),
            )
        ]
    )

    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis_title="Gross return R",
            yaxis_title="Maturity T (years)",
            zaxis_title="RRA(R|T)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    if save is not None:
        save = str(save)
        folder = os.path.dirname(save) or "."
        os.makedirs(folder, exist_ok=True)

        if save.lower().endswith(".html"):
            fig.write_html(save)
            print(f"[saved] {save}")
        else:
            # Image export: requires `pip install -U kaleido`
            fig.write_image(save)
            print(f"[saved] {save}")

    if show:
        fig.show()

    return fig

import os
from pathlib import Path
from typing import Optional, Tuple, Union, Sequence

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------

def _get_pk_grids(out: dict) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(out.get("R_common", []), float).ravel()
    T = np.asarray(out.get("T_anchor", []), float).ravel()
    if R.size == 0 or T.size == 0:
        raise KeyError("Expected out['R_common'] and out['T_anchor'].")
    if np.any(np.diff(R) <= 0):
        raise ValueError("out['R_common'] must be strictly increasing.")
    return R, T


def _slice_RT(
    R: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    *,
    R_bounds: Optional[Tuple[float, float]] = None,
    T_bounds: Optional[Tuple[float, float]] = None,
):
    Z = np.asarray(Z, float)
    if Z.shape != (T.size, R.size):
        raise ValueError(f"Surface shape mismatch: Z {Z.shape} vs (len(T),len(R))={(T.size, R.size)}")

    rmask = np.isfinite(R)
    tmask = np.isfinite(T)

    if R_bounds is not None:
        lo, hi = sorted(map(float, R_bounds))
        rmask &= (R >= lo) & (R <= hi)

    if T_bounds is not None:
        lo, hi = sorted(map(float, T_bounds))
        tmask &= (T >= lo) & (T <= hi)

    R2 = R[rmask]
    T2 = T[tmask]
    Z2 = Z[np.ix_(tmask, rmask)]
    return R2, T2, Z2, rmask, tmask


def _rowwise_cdf_trapz(R: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """CDF via cumulative trapezoid; returns normalized CDF in [0,1] if mass>0."""
    R = np.asarray(R, float)
    f = np.asarray(pdf, float)
    n = R.size
    if n < 2:
        return np.zeros(n, float)
    dR = np.diff(R)
    area = 0.5 * (f[:-1] + f[1:]) * dR
    cdf = np.empty(n, float)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(area)
    total = float(cdf[-1])
    if np.isfinite(total) and total > 0:
        cdf /= total
    return cdf


def _truncate_row_by_ptails(
    R: np.ndarray,
    z_row: np.ndarray,
    p_row: np.ndarray,
    *,
    alpha_left: float,
    alpha_right: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trim a curve z_row(R) by PHYSICAL tail mass p_row:
      keep CDF_p in [alpha_left, 1-alpha_right]
    Returns (R_kept, z_kept).
    """
    if not (0.0 <= alpha_left < 1.0 and 0.0 <= alpha_right < 1.0):
        raise ValueError("alphas must be in [0,1).")
    if alpha_left + alpha_right >= 1.0:
        raise ValueError("Need alpha_left + alpha_right < 1.")

    pr = np.where(np.isfinite(p_row) & (p_row >= 0), p_row, 0.0)
    if float(np.trapz(pr, R)) <= 0:
        return np.array([], float), np.array([], float)

    cdf = _rowwise_cdf_trapz(R, pr)
    lo_t = float(alpha_left)
    hi_t = float(1.0 - alpha_right)

    i_lo = int(np.searchsorted(cdf, lo_t, side="left"))
    i_hi = int(np.searchsorted(cdf, hi_t, side="left"))

    i_lo = max(0, min(i_lo, R.size - 1))
    i_hi = max(0, min(i_hi, R.size - 1))
    if i_hi <= i_lo:
        return np.array([], float), np.array([], float)

    keep = slice(i_lo, i_hi + 1)
    rr = R[keep]
    zz = z_row[keep]
    m = np.isfinite(rr) & np.isfinite(zz)
    return rr[m], zz[m]


def _pick_panel_indices(T: np.ndarray, n_panels: int) -> np.ndarray:
    n_pan = int(min(max(n_panels, 1), T.size))
    idx = np.unique(np.round(np.linspace(0, T.size - 1, n_pan)).astype(int))
    return idx


def _maybe_save(fig, save: Optional[Union[str, Path]], dpi: int):
    if save is None:
        return
    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {save}")


# ----------------------------
# Panels: Physical density
# ----------------------------

def physical_density_surface_plot(
    out: dict,
    *,
    title: str = "Physical Density Surface",
    cmap: str = "viridis",
    save: str | None = None,
    interactive: bool = True,
    show: bool = True,
    space: str = "gross",                 # "gross" | "log" | "strike"
    x_bounds: tuple[float, float] | None = None,
    R_bounds: tuple[float, float] | None = None,   # backward compat: treated as x_bounds if provided
    T_bounds: tuple[float, float] | None = None,
    dpi: int = 200,
):
    """
    Plot physical density surface in one of three spaces:

      space="gross"  -> p(R|T)  using out["anchor_surfaces"]["pR_surface"] and out["R_common"]
      space="log"    -> f_P(r|T) using out["anchor_surfaces"]["fP_r_surface"] and out["r_common"]
      space="strike" -> p(K|T)  using out["anchor_surfaces"]["pK_surface"] and out["K_common"]

    Requires:
      out["T_anchor"]
      and the corresponding surface/grid depending on `space`.

    interactive=True  -> Plotly 3D surface (if you have that branch elsewhere)
    interactive=False -> Matplotlib 3D surface
    """
    import numpy as np

    anchor = out.get("anchor_surfaces", None)
    if anchor is None:
        raise KeyError("Expected out['anchor_surfaces'].")

    # --- backward compatibility ---
    if x_bounds is None and R_bounds is not None:
        x_bounds = R_bounds

    space = str(space).lower().strip()

    if space in ("gross", "r", "return", "returns"):
        surf_key = "pR_surface"
        x_key = "R_common"
        x_label = "Gross return R"
        z_label = "p(R|T)"
        default_title = "Physical Density Surface (Gross Return)"
    elif space in ("log", "lr", "logreturn", "log_return", "rlog"):
        surf_key = "fP_r_surface"
        x_key = "r_common"
        x_label = "Log return r = log(S_T/S0)"
        z_label = "f_P(r|T)"
        default_title = "Physical Density Surface (Log Return)"
    elif space in ("strike", "k", "price", "spot"):
        surf_key = "pK_surface"
        x_key = "K_common"
        x_label = "Strike / terminal price K (S_T)"
        z_label = "p(K|T)"
        default_title = "Physical Density Surface (Strike)"
    else:
        raise ValueError("space must be one of: 'gross', 'log', 'strike'")

    if surf_key not in anchor:
        raise KeyError(f"Expected out['anchor_surfaces']['{surf_key}'].")

    p = np.asarray(anchor[surf_key], float)
    X_grid = np.asarray(out.get(x_key, []), float).ravel()
    T_grid = np.asarray(out.get("T_anchor", []), float).ravel()

    if X_grid.size == 0 or T_grid.size == 0:
        raise KeyError(f"Expected out['{x_key}'] and out['T_anchor'].")

    if p.shape != (T_grid.size, X_grid.size):
        raise ValueError(f"{surf_key} must have shape (len(T_anchor), len({x_key})).")

    # Slice bounds using your helper; it expects names T_bounds and R_bounds,
    # so we pass x-bounds via R_bounds param.
    T_plot, X_plot, p_plot = _slice_bounds_2d(
        T_grid, X_grid, p,
        T_bounds=T_bounds,
        R_bounds=x_bounds,
    )
    if T_plot.size < 2 or X_plot.size < 2:
        raise ValueError("Not enough points to plot after applying bounds.")

    date_label = out.get("anchor_key_used", out.get("anchor_date_used", ""))
    base_title = title if title else default_title
    plot_title = f"{base_title} — {date_label}" if date_label else base_title

    # -------------------- Matplotlib branch --------------------
    if not interactive:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        X_mesh, T_mesh = np.meshgrid(X_plot, T_plot)

        fig = plt.figure(figsize=(9.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X_mesh, T_mesh, p_plot, cmap=cmap, linewidth=0, antialiased=True)

        ax.set_title(plot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel(z_label)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label(z_label)

        fig.tight_layout()
        _maybe_save_matplotlib(fig, save, dpi=dpi)

        if show:
            plt.show()
        return fig

    # -------------------- Plotly branch --------------------
    # If you already have a plotly surface helper, plug it in here.
    # Keeping minimal: implement locally if needed.
    import plotly.graph_objects as go

    X_mesh, T_mesh = np.meshgrid(X_plot, T_plot)
    fig = go.Figure(
        data=[
            go.Surface(
                x=X_mesh,
                y=T_mesh,
                z=p_plot,
                colorscale=cmap,
                showscale=True,
                colorbar=dict(title=z_label),
            )
        ]
    )
    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title="Maturity T (years)",
            zaxis_title=z_label,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Save (html recommended for plotly)
    if save:
        if save.lower().endswith(".html"):
            fig.write_html(save)
        else:
            # requires kaleido installed for static images
            fig.write_image(save, scale=2)

    if show:
        fig.show()

    return fig

def physical_density_panels(
    out: dict,
    *,
    n_panels: int = 6,
    title: str = "Physical Density Panels",
    date_str: Optional[str] = None,
    space: str = "gross",                          # "gross" | "log" | "strike"
    x_bounds: Optional[Tuple[float, float]] = None,
    R_bounds: Optional[Tuple[float, float]] = None,  # backward compat alias for x_bounds
    # optional vertical reference line (space-dependent)
    show_ref: bool = True,
    ref_label: Optional[str] = None,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    legend_loc: str = "upper right",
    figsize_per_panel: float = 2.2,
):
    """
    Stacked panels for physical density in different spaces.

    Requires (depending on `space`):
      space="gross":  out["anchor_surfaces"]["pR_surface"], out["R_common"], out["T_anchor"]
      space="log":    out["anchor_surfaces"]["fP_r_surface"], out["r_common"], out["T_anchor"]
      space="strike": out["anchor_surfaces"]["pK_surface"], out["K_common"], out["T_anchor"]
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- backward compatibility ---
    if x_bounds is None and R_bounds is not None:
        x_bounds = R_bounds

    anchor = out.get("anchor_surfaces", {})
    T = np.asarray(out.get("T_anchor", []), float).ravel()
    if T.size == 0:
        raise KeyError("Expected out['T_anchor'].")

    space = str(space).lower().strip()

    # --- select surface + grid + labels ---
    if space in ("gross", "r", "return", "returns"):
        surf_key = "pR_surface"
        grid_key = "R_common"
        curve_label = "p(R|T)"
        x_label = "Gross return R"
        y_label = "p(R|T)"
        ref_x = 1.0
        default_ref_label = "R=1"
    elif space in ("log", "lr", "logreturn", "log_return"):
        surf_key = "fP_r_surface"
        grid_key = "r_common"
        curve_label = "f_P(r|T)"
        x_label = "Log return r = log(S_T/S0)"
        y_label = "f_P(r|T)"
        ref_x = 0.0
        default_ref_label = "r=0"
    elif space in ("strike", "k", "price"):
        surf_key = "pK_surface"
        grid_key = "K_common"
        curve_label = "p(K|T)"
        x_label = "Strike / terminal price K (S_T)"
        y_label = "p(K|T)"
        ref_x = float(out.get("meta", {}).get("S0", out.get("S0", np.nan)))
        default_ref_label = "K=S0"
        if not np.isfinite(ref_x):
            # if S0 not present, disable ref line automatically
            ref_x = None
    else:
        raise ValueError("space must be one of: 'gross', 'log', 'strike'")

    if surf_key not in anchor:
        raise KeyError(f"Expected out['anchor_surfaces']['{surf_key}'].")

    X = np.asarray(out.get(grid_key, []), float).ravel()
    if X.size == 0:
        raise KeyError(f"Expected out['{grid_key}'].")

    p = np.asarray(anchor[surf_key], float)
    if p.shape != (T.size, X.size):
        raise ValueError(f"{surf_key} must have shape (len(T_anchor), len({grid_key})).")

    # --- apply x-bounds (keep all T) ---
    if x_bounds is not None:
        lo, hi = float(x_bounds[0]), float(x_bounds[1])
        if lo >= hi:
            raise ValueError("x_bounds must satisfy (lo < hi).")
        mX = (X >= lo) & (X <= hi)
        X2 = X[mX]
        p2 = p[:, mX]
        T2 = T
    else:
        X2, T2, p2 = X, T, p

    if X2.size < 2 or T2.size < 1:
        raise ValueError("Not enough points to plot after applying bounds.")

    # choose maturity indices
    idxT = _pick_panel_indices(T2, n_panels)
    nrows = idxT.size

    # figure
    fig_h = max(4.0, figsize_per_panel * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8.5, fig_h), sharex=True)
    if nrows == 1:
        axes = np.array([axes])

    # reference line visibility
    ref_label_eff = ref_label if ref_label is not None else default_ref_label
    show_ref_eff = bool(show_ref) and (ref_x is not None) and np.isfinite(ref_x) and (X2[0] <= ref_x <= X2[-1])

    for ax, j in zip(axes, idxT):
        y = np.asarray(p2[j, :], float)
        m = np.isfinite(X2) & np.isfinite(y)

        ax.plot(X2[m], y[m], linewidth=2.2, label=curve_label)

        if show_ref_eff:
            ax.axvline(float(ref_x), linestyle="--", linewidth=1.5, color="black", label=ref_label_eff)

        d = f"{date_str} " if date_str else ""
        ax.set_title(f"{d}(T = {float(T2[j]):.5g} yr)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)

    axes[-1].set_xlabel(x_label)
    fig.suptitle(title)
    fig.tight_layout()
    _maybe_save(fig, save, dpi=dpi)
    plt.show()
    return fig


# ----------------------------
# Panels: Pricing kernel M(R)
#   supports:
#     - rectangular truncation by R_bounds / T_bounds
#     - row-wise ptail_alphas trimming using p(R|T)
# ----------------------------

def pricing_kernel_panels(
    out: dict,
    *,
    n_panels: int = 6,
    title: str = "Pricing Kernel Panels",
    date_str: Optional[str] = None,
    # rectangular window (applied BEFORE ptail trim)
    R_bounds: Optional[Tuple[float, float]] = None,
    T_bounds: Optional[Tuple[float, float]] = None,
    # p-tail trim per-row using PHYSICAL density (best for readability)
    ptail_alphas: Optional[Tuple[float, float]] = None,  # (alpha_left, alpha_right)
    # y scale
    yscale: str = "linear",  # {"linear","log"}
    ylog_eps: float = 1e-300,
    # optional vertical line at R=1
    show_R1: bool = True,
    R1_label: str = "R=1",
    # save
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    legend_loc: str = "upper right",
    figsize_per_panel: float = 2.2,
    # overlay untrimmed curve too?
    show_untrimmed: bool = True,
    untrim_color: str = "0.6",
    trim_color: str = "black",
    trim_ls: str = "--",
):
    """
    Stacked panels for pricing kernel M(R|T).

    Requires:
      out["anchor_surfaces"]["M_surface"]
      out["anchor_surfaces"]["pR_surface"]  (only if ptail_alphas is used)
      out["R_common"], out["T_anchor"]
    """
    anchor = out.get("anchor_surfaces", {})
    if "M_surface" not in anchor:
        raise KeyError("Expected out['anchor_surfaces']['M_surface'].")

    R, T = _get_pk_grids(out)
    M = np.asarray(anchor["M_surface"], float)

    # rectangular slice first (both T and R)
    R2, T2, M2, rmask, tmask = _slice_RT(R, T, M, R_bounds=R_bounds, T_bounds=T_bounds)

    # p-surface for ptail trimming (must be sliced identically)
    p2 = None
    if ptail_alphas is not None:
        if "pR_surface" not in anchor:
            raise KeyError("ptail_alphas requires out['anchor_surfaces']['pR_surface'].")
        p_full = np.asarray(anchor["pR_surface"], float)
        p2 = p_full[np.ix_(tmask, rmask)]

    idxT = _pick_panel_indices(T2, n_panels)

    yscale = str(yscale).lower().strip()
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")
    ylog_eps = float(ylog_eps)
    if yscale == "log" and not (ylog_eps > 0):
        raise ValueError("ylog_eps must be > 0 for log scale.")

    show_R1_eff = bool(show_R1) and (R2.size > 1) and (R2[0] <= 1.0 <= R2[-1])

    nrows = idxT.size
    fig_h = max(4.0, figsize_per_panel * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8.5, fig_h), sharex=True)
    if nrows == 1:
        axes = np.array([axes])

    for ax, j in zip(axes, idxT):
        # untrimmed
        if show_untrimmed:
            y = np.asarray(M2[j, :], float)
            m = np.isfinite(R2) & np.isfinite(y)
            if yscale == "log":
                m &= (y > 0)
                y = np.maximum(y, ylog_eps)
            ax.plot(R2[m], y[m], linewidth=2.2, color=untrim_color, label="M(R)")

        # trimmed (p-tail)
        if ptail_alphas is not None and p2 is not None:
            aL, aR = float(ptail_alphas[0]), float(ptail_alphas[1])
            rr, zz = _truncate_row_by_ptails(
                R2, np.asarray(M2[j, :], float), np.asarray(p2[j, :], float),
                alpha_left=aL, alpha_right=aR
            )
            if rr.size > 1:
                if yscale == "log":
                    mm = np.isfinite(rr) & np.isfinite(zz) & (zz > 0)
                    rr = rr[mm]
                    zz = np.maximum(zz[mm], ylog_eps)
                ax.plot(rr, zz, linewidth=2.4, color=trim_color, ls=trim_ls,
                        label=f"p-tail trim ({aL:.2f},{aR:.2f})")

        if show_R1_eff:
            ax.axvline(1.0, linestyle="--", linewidth=1.2, color="black", alpha=0.8, label=R1_label)

        d = f"{date_str} " if date_str else ""
        ax.set_title(f"{d}(T = {float(T2[j]):.5g} yr)")
        ax.set_ylabel("M(R|T)")
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)

    axes[-1].set_xlabel("Gross return R")
    fig.suptitle(title)
    fig.tight_layout()
    _maybe_save(fig, save, dpi=dpi)
    plt.show()
    return fig


# ----------------------------
# Panels: RRA(R)
#   supports same truncation options as M(R)
# ----------------------------

def rra_panels(
    out: dict,
    *,
    n_panels: int = 6,
    title: str = "RRA Panels",
    date_str: Optional[str] = None,
    # rectangular window (applied BEFORE ptail trim)
    R_bounds: Optional[Tuple[float, float]] = None,
    T_bounds: Optional[Tuple[float, float]] = None,
    # p-tail trim per-row using PHYSICAL density (recommended)
    ptail_alphas: Optional[Tuple[float, float]] = None,
    # optional y clipping
    clip_y: Optional[Tuple[float, float]] = None,
    # optional vertical line at R=1
    show_R1: bool = True,
    R1_label: str = "R=1",
    # save
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    legend_loc: str = "upper right",
    figsize_per_panel: float = 2.2,
    # overlay untrimmed curve too?
    show_untrimmed: bool = True,
    untrim_color: str = "0.6",
    trim_color: str = "black",
    trim_ls: str = "--",
):
    """
    Stacked panels for RRA(R|T).

    Requires:
      out["anchor_surfaces"]["RRA_surface"]
      out["anchor_surfaces"]["pR_surface"]  (only if ptail_alphas is used)
      out["R_common"], out["T_anchor"]
    """
    anchor = out.get("anchor_surfaces", {})
    if "RRA_surface" not in anchor:
        raise KeyError("Expected out['anchor_surfaces']['RRA_surface'].")

    R, T = _get_pk_grids(out)
    RRA = np.asarray(anchor["RRA_surface"], float)

    # rectangular slice first (both T and R)
    R2, T2, RRA2, rmask, tmask = _slice_RT(R, T, RRA, R_bounds=R_bounds, T_bounds=T_bounds)

    # p-surface for ptail trimming (must be sliced identically)
    p2 = None
    if ptail_alphas is not None:
        if "pR_surface" not in anchor:
            raise KeyError("ptail_alphas requires out['anchor_surfaces']['pR_surface'].")
        p_full = np.asarray(anchor["pR_surface"], float)
        p2 = p_full[np.ix_(tmask, rmask)]

    idxT = _pick_panel_indices(T2, n_panels)
    show_R1_eff = bool(show_R1) and (R2.size > 1) and (R2[0] <= 1.0 <= R2[-1])

    nrows = idxT.size
    fig_h = max(4.0, figsize_per_panel * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8.5, fig_h), sharex=True)
    if nrows == 1:
        axes = np.array([axes])

    for ax, j in zip(axes, idxT):
        # untrimmed
        if show_untrimmed:
            y = np.asarray(RRA2[j, :], float)
            m = np.isfinite(R2) & np.isfinite(y)
            ax.plot(R2[m], y[m], linewidth=2.2, color=untrim_color, label="RRA(R)")

        # trimmed (p-tail)
        if ptail_alphas is not None and p2 is not None:
            aL, aR = float(ptail_alphas[0]), float(ptail_alphas[1])
            rr, zz = _truncate_row_by_ptails(
                R2, np.asarray(RRA2[j, :], float), np.asarray(p2[j, :], float),
                alpha_left=aL, alpha_right=aR
            )
            if rr.size > 1:
                ax.plot(rr, zz, linewidth=2.4, color=trim_color, ls=trim_ls,
                        label=f"p-tail trim ({aL:.2f},{aR:.2f})")

        if show_R1_eff:
            ax.axvline(1.0, linestyle="--", linewidth=1.2, color="black", alpha=0.8, label=R1_label)

        if clip_y is not None:
            ax.set_ylim(float(clip_y[0]), float(clip_y[1]))

        d = f"{date_str} " if date_str else ""
        ax.set_title(f"{d}(T = {float(T2[j]):.5g} yr)")
        ax.set_ylabel("RRA(R|T)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc=legend_loc)

    axes[-1].set_xlabel("Gross return R")
    fig.suptitle(title)
    fig.tight_layout()
    _maybe_save(fig, save, dpi=dpi)
    plt.show()
    return fig

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


def _maybe_save(fig, save: Optional[Union[str, Path]], dpi: int = 200) -> None:
    if save is None:
        return
    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {save}")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Union
def P_Q_K_multipanel(
    out: dict,
    *,
    title: Optional[str] = None,
    n_panels: Optional[int] = None,                 # if None uses panel_shape product
    panel_shape: Tuple[int, int] = (2, 4),
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    # ----- truncation controls -----
    truncate: bool = True,
    ptail_alpha: Tuple[float, float] = (0.10, 0.0), # (alpha_left, alpha_right) for q-CDF tails
    trunc_mode: str = "cdf",                        # {"cdf","rbounds","none","cdf+rbounds"}
    r_bounds: Optional[Tuple[float, float]] = None,
    clip_trunc_to_support: bool = True,
    # ----- kernel axis controls -----
    kernel_color: str = "black",
    kernel_linestyle: str = "--",
    kernel_yscale: str = "linear",                  # {"linear","log"}
    kernel_log_eps: float = 1e-300,
    # ----- display controls -----
    legend_loc: str = "upper center",
):
    """
    Multi-panel plot: q_R(R), p_R(R) and pricing kernel M(R) with dual y-axis.

    Key behavior:
      - r_bounds truncates *everything* (q, p, and M) whenever enabled.
      - ptail_alpha/CDF truncation affects *only* the kernel M(R):
          keep kernel where q-CDF in [alpha_left, 1-alpha_right].
        (computed on full support, then intersected with rbounds if active)
      - You can use both by setting trunc_mode in {"cdf+rbounds","cdf"} with r_bounds provided.
    """
    if out is None or "anchor_surfaces" not in out:
        raise KeyError("out must contain out['anchor_surfaces'].")

    anchor = out["anchor_surfaces"]
    T = np.asarray(out.get("T_anchor", []), float).ravel()
    R = np.asarray(out.get("R_common", []), float).ravel()

    qR = np.asarray(anchor.get("qR_surface", []), float)
    pR = np.asarray(anchor.get("pR_surface", []), float)
    M = np.asarray(anchor.get("M_surface", []), float)

    if T.size == 0 or R.size == 0:
        raise ValueError("Missing T_anchor or R_common.")
    if qR.shape != (T.size, R.size) or pR.shape != (T.size, R.size) or M.shape != (T.size, R.size):
        raise ValueError("qR_surface, pR_surface, M_surface must all have shape (len(T_anchor), len(R_common)).")
    if R.size >= 2 and np.any(np.diff(R) <= 0):
        raise ValueError("R_common must be strictly increasing.")

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

    # ptail_alpha validation
    aL, aR = float(ptail_alpha[0]), float(ptail_alpha[1])
    if use_cdf:
        if not (0.0 <= aL < 1.0 and 0.0 <= aR < 1.0):
            raise ValueError("ptail_alpha must be in [0,1) for both tails.")
        if not (aL + aR < 1.0):
            raise ValueError("Require ptail_alpha[0] + ptail_alpha[1] < 1.")

    # Determine rbounds range (if enabled)
    R_min = R_max = None
    if use_rbounds:
        if r_bounds is None or len(r_bounds) != 2:
            raise ValueError("For rbounds truncation, provide r_bounds=(R_min, R_max).")
        R_min, R_max = float(r_bounds[0]), float(r_bounds[1])
        if clip_trunc_to_support:
            R_min = max(R_min, float(R[0]))
            R_max = min(R_max, float(R[-1]))
        if not (np.isfinite(R_min) and np.isfinite(R_max) and R_max > R_min):
            raise ValueError("Invalid r_bounds after clipping.")

    kernel_yscale = str(kernel_yscale).lower().strip()
    if kernel_yscale not in {"linear", "log"}:
        raise ValueError("kernel_yscale must be one of {'linear','log'}.")
    kernel_log_eps = float(kernel_log_eps)
    if kernel_yscale == "log" and not (kernel_log_eps > 0):
        raise ValueError("kernel_log_eps must be > 0 for log scale.")

    # -------------------------
    # choose maturities for panels
    # -------------------------
    nrows, ncols = panel_shape
    nT = T.size
    idx_pool = np.arange(nT)

    n_pan = (nrows * ncols) if n_panels is None else int(n_panels)
    n_pan = max(1, min(n_pan, idx_pool.size))
    idxs = idx_pool[np.linspace(0, idx_pool.size - 1, n_pan, dtype=int)]

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.9 * ncols, 3.6 * nrows),
        sharex=True,
        constrained_layout=False
    )
    axes = np.array(axes).reshape(-1)

    ax2_list = []

    for k, j in enumerate(idxs):
        ax = axes[k]
        ax2 = ax.twinx()
        ax2_list.append(ax2)

        # ---- base mask: rbounds applies to everything ----
        mask_all = np.isfinite(R)
        if use_rbounds:
            mask_all &= (R >= R_min) & (R <= R_max)

        R_all = R[mask_all]
        q_plot = qR[j, :][mask_all]
        p_plot = pR[j, :][mask_all]
        M_plot = M[j, :][mask_all].copy()

        ax.plot(R_all, q_plot, label="q_R(R)", linewidth=1.8)
        ax.plot(R_all, p_plot, label="p_R(R)", linewidth=1.8)

        # ---- optional two-sided CDF trunc for kernel only ----
        kernel_label = "M(R)"
        R_k = R_all
        M_k = M_plot
        if use_cdf:
            # Compute p-CDF on full R (not rbounds-trimmed) so ptail_alpha means “global tails”
            pj = np.maximum(np.asarray(pR[j, :], float), 0.0)
    
            dR_full = np.diff(R)
            inc = 0.5 * (pj[1:] + pj[:-1]) * dR_full
    
            cdf = np.empty_like(R)
            cdf[0] = 0.0
            cdf[1:] = np.cumsum(inc)
            total = float(cdf[-1])
    
            if total > 0 and np.isfinite(total):
                cdf /= total
    
                aL, aR = float(ptail_alpha[0]), float(ptail_alpha[1])
    
                # left cutoff: first index with CDF >= aL
                if aL > 0:
                    idxL = np.where(cdf >= aL)[0]
                    iL = int(idxL[0]) if idxL.size else 0
                else:
                    iL = 0
    
                # right cutoff: last index with CDF <= 1-aR
                if aR > 0:
                    idxR = np.where(cdf <= (1.0 - aR))[0]
                    iR = int(idxR[-1]) if idxR.size else (R.size - 1)
                else:
                    iR = R.size - 1
    
                if iR <= iL:
                    R_k = np.array([], float)
                    M_k = np.array([], float)
                else:
                    R_left = float(R[iL])
                    R_right = float(R[iR])
    
                    # intersect with rbounds-trimmed arrays:
                    keep = (R_k >= R_left) & (R_k <= R_right)
                    R_k = R_k[keep]
                    M_k = M_k[keep]
    
                    kernel_label = f"M(R) (p-tails ≥ {aL:.0%}/{aR:.0%})"
            else:
                # if CDF invalid, suppress kernel for this slice
                R_k = np.array([], float)
                M_k = np.array([], float)
        

        # ---- kernel y-scale adjustments ----
        if kernel_yscale == "log" and M_k.size > 0:
            pos = np.isfinite(M_k) & (M_k > 0) & np.isfinite(R_k)
            R_k = np.asarray(R_k, float)[pos]
            M_k = np.asarray(M_k, float)[pos]
            M_k = np.maximum(M_k, kernel_log_eps)

        if R_k.size > 0:
            ax2.plot(
                R_k, M_k,
                label=kernel_label + ("" if kernel_yscale == "linear" else " (log y)"),
                color=kernel_color,
                linestyle=kernel_linestyle,
                linewidth=1.6,
                alpha=0.95
            )

        # ---- titles/labels ----
        T_days = float(T[j] * 365.0)
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

    for k in range(n_pan, axes.size):
        axes[k].axis("off")

    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2_list[0].get_legend_handles_labels()
    handles = h1 + h2
    labels = l1 + l2

    if title is None:
        title = f"Date={out.get('anchor_key_used', 'N/A')}: q_R vs p_R with Pricing Kernel"

    fig.suptitle(title, y=0.995, fontsize=14)
    fig.legend(
        handles, labels,
        loc=legend_loc,
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.6
    )
    fig.subplots_adjust(top=0.88, wspace=0.28, hspace=0.30)

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")

    plt.show()
    return fig


import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal

def _slice_bounds_2d(
    T: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    *,
    T_bounds: Optional[Tuple[float, float]] = None,
    R_bounds: Optional[Tuple[float, float]] = None,
):
    T = np.asarray(T, float).ravel()
    R = np.asarray(R, float).ravel()
    Z = np.asarray(Z, float)

    if Z.shape != (T.size, R.size):
        raise ValueError(f"Z must have shape (len(T),len(R)) = {(T.size,R.size)}, got {Z.shape}.")

    tmask = np.isfinite(T)
    rmask = np.isfinite(R)

    if T_bounds is not None:
        t0, t1 = float(T_bounds[0]), float(T_bounds[1])
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        tmask &= (T >= lo) & (T <= hi)

    if R_bounds is not None:
        r0, r1 = float(R_bounds[0]), float(R_bounds[1])
        lo, hi = (r0, r1) if r0 <= r1 else (r1, r0)
        rmask &= (R >= lo) & (R <= hi)

    T2 = T[tmask]
    R2 = R[rmask]
    Z2 = Z[np.ix_(tmask, rmask)]
    return T2, R2, Z2


def _normalize_pk_rows(
    M: np.ndarray,
    R: np.ndarray,
    *,
    mode: Literal["none", "anchor", "eq_mean"] = "anchor",
    R0: float = 1.0,
    qR: Optional[np.ndarray] = None,
    eps: float = 1e-300,
) -> np.ndarray:
    """
    Normalize each maturity row by a scalar so shapes are comparable across dates.
      - none: no scaling
      - anchor: divide by M(R0) per row (R0 default 1.0; uses interpolation)
      - eq_mean: scale so ∫ M(R)*qR(R) dR = 1 per row (needs qR)
    """
    M = np.asarray(M, float)
    R = np.asarray(R, float).ravel()
    out = M.copy()

    mode = (mode or "none").lower().strip()
    if mode == "none":
        return out

    if mode == "anchor":
        # normalize each row by M(R0) (interpolated)
        for j in range(out.shape[0]):
            row = out[j, :]
            ok = np.isfinite(R) & np.isfinite(row)
            if ok.sum() < 2:
                out[j, :] = np.nan
                continue
            val = float(np.interp(float(R0), R[ok], row[ok]))
            if (not np.isfinite(val)) or abs(val) < eps:
                out[j, :] = np.nan
            else:
                out[j, :] = row / val
        return out

    if mode == "eq_mean":
        if qR is None:
            raise ValueError("mode='eq_mean' requires qR_surface.")
        qR = np.asarray(qR, float)
        if qR.shape != out.shape:
            raise ValueError("qR_surface shape must match M_surface shape.")
        for j in range(out.shape[0]):
            mj = out[j, :]
            qj = qR[j, :]
            ok = np.isfinite(R) & np.isfinite(mj) & np.isfinite(qj)
            if ok.sum() < 2:
                out[j, :] = np.nan
                continue
            integrand = mj[ok] * np.maximum(qj[ok], 0.0)
            denom = float(np.trapz(integrand, R[ok]))
            if (not np.isfinite(denom)) or denom <= eps:
                out[j, :] = np.nan
            else:
                out[j, :] = mj / denom
        return out

    raise ValueError("normalize must be one of {'none','anchor','eq_mean'}.")


def pricing_kernel_surface_plot(
    out: dict,
    *,
    title: str = "Pricing Kernel Surface",
    cmap: str = "viridis",
    save: str | None = None,          # .html for interactive; image ext for static
    interactive: bool = True,
    show: bool = True,
    R_bounds: tuple[float, float] | None = None,
    T_bounds: tuple[float, float] | None = None,
    zscale: Literal["linear", "log"] = "linear",
    normalize: Literal["none", "anchor", "eq_mean"] = "anchor",
    R0: float = 1.0,
):
    """
    Plot pricing kernel surface M(T,R) from an evaluation output dict `out`.

    Requires:
      out["R_common"], out["T_anchor"], out["anchor_surfaces"]["M_surface"].
    If normalize="eq_mean", also requires out["anchor_surfaces"]["qR_surface"].

    Options:
      - normalize:
          * "none"   : no scaling
          * "anchor" : divide each maturity row by M(R0) (default R0=1.0)
          * "eq_mean": scale so ∫ M(R,T) q(R,T) dR = 1 for each row (per maturity)
      - zscale="log": plots log(M) (only where M>0).
    """
    if "anchor_surfaces" not in out:
        raise KeyError("Expected out['anchor_surfaces'].")

    anchor = out["anchor_surfaces"]
    if "M_surface" not in anchor:
        raise KeyError("Missing pricing kernel. Expected out['anchor_surfaces']['M_surface'].")

    R = np.asarray(out.get("R_common"), float).ravel()
    T = np.asarray(out.get("T_anchor"), float).ravel()
    M = np.asarray(anchor["M_surface"], float)

    if M.shape != (T.size, R.size):
        raise ValueError("M_surface must have shape (len(T_anchor), len(R_common)).")

    # Optional row normalization (shape-preserving)
    qR = None
    if normalize == "eq_mean":
        qR = np.asarray(anchor.get("qR_surface"), float)
    M_norm = _normalize_pk_rows(M, R, mode=normalize, R0=R0, qR=qR)

    # Bounds slice (applies to everything)
    T_plot, R_plot, M_plot = _slice_bounds_2d(T, R, M_norm, T_bounds=T_bounds, R_bounds=R_bounds)
    if T_plot.size < 2 or R_plot.size < 2:
        raise ValueError("Not enough points to plot after bounds.")

    # zscale
    zscale = (zscale or "linear").lower().strip()
    if zscale not in {"linear", "log"}:
        raise ValueError("zscale must be 'linear' or 'log'.")
    if zscale == "log":
        Z = np.where(np.isfinite(M_plot) & (M_plot > 0), np.log(M_plot), np.nan)
        zlab = "log M(T,R)"
    else:
        Z = M_plot
        zlab = "M(T,R)"

    # ---------- Plotly interactive ----------
    if interactive:
        import plotly.graph_objects as go

        R_mesh, T_mesh = np.meshgrid(R_plot, T_plot)
        fig = go.Figure(
            data=[go.Surface(
                x=R_mesh, y=T_mesh, z=Z,
                colorscale=cmap,
                colorbar=dict(title=zlab),
            )]
        )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Gross return R",
                yaxis_title="Maturity T (years)",
                zaxis_title=zlab,
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        if save is not None:
            save = str(save)
            os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
            if save.lower().endswith(".html"):
                fig.write_html(save)
                print(f"[saved] {save}")
            else:
                # Plotly image export needs kaleido
                fig.write_image(save)
                print(f"[saved] {save}")

        if show:
            fig.show()
        return fig

    # ---------- Matplotlib static ----------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    RR, TT = np.meshgrid(R_plot, T_plot)
    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(RR, TT, Z, cmap=cmap, linewidth=0, antialiased=True)
    ax.set_xlabel("Gross return R")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel(zlab)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label=zlab)

    plt.tight_layout()

    if save is not None:
        save = str(save)
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
        print(f"[saved] {save}")

    if show:
        plt.show()

    return fig
