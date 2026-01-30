# pyderivatives/global_pricer/plotting/surfaces.py
from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.tri import Triangulation
from .diagnostics import add_safety_clip_note

def _add_safety_clip_note(fig: plt.Figure, status: str, where: str = "br"):
    txt = f"Safety clip: {status}"
    if where == "tr":
        x, y, va = 0.99, 0.99, "top"
    else:
        x, y, va = 0.99, 0.01, "bottom"
    fig.text(x, y, txt, ha="right", va=va, fontsize=9, alpha=0.85)

# def plot_call_surface_fit_with_obs(
#     res: dict,
#     *,
#     day=None,
#     title: str = "Fitted call surface with observed quotes",
#     figsize=(10, 7),
#     elev: float = 25,
#     azim: float = -60,
#     add_clip_note: bool = True,
# ):
#     """
#     Single 3D axis:
#       - fitted call surface (K_grid x T_grid)
#       - observed quotes as scatter (if day provided)

#     res must have: K_grid, T_grid, C_fit
#     """
#     K = np.asarray(res["K_grid"], float).ravel()
#     T = np.asarray(res["T_grid"], float).ravel()
#     C = np.asarray(res["C_fit"], float)

#     KK, TT = np.meshgrid(K, T)

#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111, projection="3d")

#     ax.plot_surface(KK, TT, C, linewidth=0, antialiased=True, alpha=0.85)

#     if day is not None:
#         ax.scatter(
#             np.asarray(day.K_obs, float),
#             np.asarray(day.T_obs, float),
#             np.asarray(day.C_obs, float),
#             s=20,
#             depthshade=True,
#         )

#     ax.set_title(title)
#     ax.set_xlabel("Strike K")
#     ax.set_ylabel("Maturity T")
#     ax.set_zlabel("Call price")

#     ax.view_init(elev=elev, azim=azim)
#     plt.tight_layout()

#     if add_clip_note:
#         add_safety_clip_note(fig, res.get("safety_clip", None), where="br")

#     plt.show()
#     return fig


# def plot_two_panel_obs_vs_fit(
#     res: dict,
#     *,
#     day=None,
#     title: str = "Observed vs fitted call surface",
#     figsize=(14, 6),
#     elev: float = 25,
#     azim: float = -60,
#     add_clip_note: bool = True,
# ):
#     """
#     Two 3D subplots:
#       left: observed scatter only
#       right: fitted surface
#     """
#     K = np.asarray(res["K_grid"], float).ravel()
#     T = np.asarray(res["T_grid"], float).ravel()
#     C = np.asarray(res["C_fit"], float)
#     KK, TT = np.meshgrid(K, T)

#     fig = plt.figure(figsize=figsize)
#     ax1 = fig.add_subplot(121, projection="3d")
#     ax2 = fig.add_subplot(122, projection="3d")

#     ax1.set_title("Observed quotes")
#     if day is not None:
#         ax1.scatter(
#             np.asarray(day.K_obs, float),
#             np.asarray(day.T_obs, float),
#             np.asarray(day.C_obs, float),
#             s=20,
#             depthshade=True,
#         )
#     ax1.set_xlabel("Strike K")
#     ax1.set_ylabel("Maturity T")
#     ax1.set_zlabel("Call price")
#     ax1.view_init(elev=elev, azim=azim)

#     ax2.set_title(f"Fitted surface ({res.get('model','model')})")
#     ax2.plot_surface(KK, TT, C, linewidth=0, antialiased=True, alpha=0.9)
#     ax2.set_xlabel("Strike K")
#     ax2.set_ylabel("Maturity T")
#     ax2.set_zlabel("Call price")
#     ax2.view_init(elev=elev, azim=azim)

#     fig.suptitle(title, fontsize=18)
#     plt.tight_layout()

#     if add_clip_note:
#         add_safety_clip_note(fig, res.get("safety_clip", None), where="br")

#     plt.show()
#     return fig

# def _plot_surface_3d(K_grid: np.ndarray, T_grid: np.ndarray, Z: np.ndarray, *,
#                      title: str, zlabel: str):
#     from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
#     K = np.asarray(K_grid, float).ravel()
#     T = np.asarray(T_grid, float).ravel()
#     Z = np.asarray(Z, float)

#     KK, TT = np.meshgrid(K, T)

#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_surface(KK, TT, Z, linewidth=0, antialiased=True, alpha=0.9)
#     ax.set_title(title)
#     ax.set_xlabel("Strike K")
#     ax.set_ylabel("Maturity T")
#     ax.set_zlabel(zlabel)
#     fig.tight_layout()
#     return fig


# def plot_iv_surface_3d(res: dict, *, title: str = "IV surface (fitted)", show_clip_note: bool = False):
#     if "iv_surface" not in res:
#         raise KeyError("res missing 'iv_surface'.")
#     fig = _plot_surface_3d(res["K_grid"], res["T_grid"], res["iv_surface"], title=title, zlabel="IV")
#     if show_clip_note:
#         sc = res.get("safety_clip", None)
#         status = "Unused"
#         if isinstance(sc, dict) and sc.get("enabled", False):
#             status = "Used" if sc.get("any_used", False) else "Enabled (not triggered)"
#         _add_safety_clip_note(fig, status=status)
#     return fig


# def plot_rnd_surface_3d(res: dict, *, title: str = "RND surface (Breeden–Litzenberger)", show_clip_note: bool = True):
#     if "rnd_surface" not in res:
#         raise KeyError("res missing 'rnd_surface'. Run compute_rnd=True.")
#     fig = _plot_surface_3d(res["K_grid"], res["T_grid"], res["rnd_surface"], title=title, zlabel="q_T(K)")
#     if show_clip_note:
#         sc = res.get("safety_clip", None)
#         status = "Unused"
#         if isinstance(sc, dict) and sc.get("enabled", False):
#             status = "Used" if sc.get("any_used", False) else "Enabled (not triggered)"
#         _add_safety_clip_note(fig, status=status)
#     return fig


# def plot_cdf_surface_3d(res: dict, *, title: str = "CDF surface", show_clip_note: bool = False):
#     if "cdf_surface" not in res:
#         raise KeyError("res missing 'cdf_surface'. Run compute_cdf=True.")
#     fig = _plot_surface_3d(res["K_grid"], res["T_grid"], res["cdf_surface"], title=title, zlabel="F_T(K)")
#     if show_clip_note:
#         sc = res.get("safety_clip", None)
#         status = "Unused"
#         if isinstance(sc, dict) and sc.get("enabled", False):
#             status = "Used" if sc.get("any_used", False) else "Enabled (not triggered)"
#         _add_safety_clip_note(fig, status=status)
#     return fig
# def plot_fitted_surface_with_observed(
#     *,
#     K_obs: np.ndarray,
#     T_obs: np.ndarray,
#     C_obs: np.ndarray,
#     K_grid: np.ndarray,
#     T_grid: np.ndarray,
#     C_fit: np.ndarray,
#     title: str = "Fitted Call Surface with Observed Quotes",
#     fitted_alpha: float = 0.55,
#     obs_size: float = 18.0,
#     downsample_obs: int | None = None,   # e.g. 1500 if you have tons of points
# ):
#     """
#     Single 3D plot:
#       - fitted surface (rectangular grid) as a surface
#       - observed quotes as scatter on top

#     Assumes:
#       C_fit shape == (len(T_grid), len(K_grid))
#       K_obs, T_obs, C_obs are 1D arrays of equal length (sparse quotes)
#     """
#     K_obs = np.asarray(K_obs, float).ravel()
#     T_obs = np.asarray(T_obs, float).ravel()
#     C_obs = np.asarray(C_obs, float).ravel()

#     K_grid = np.asarray(K_grid, float).ravel()
#     T_grid = np.asarray(T_grid, float).ravel()
#     C_fit = np.asarray(C_fit, float)

#     if C_fit.shape != (T_grid.size, K_grid.size):
#         raise ValueError("C_fit must have shape (len(T_grid), len(K_grid)).")

#     m_obs = np.isfinite(K_obs) & np.isfinite(T_obs) & np.isfinite(C_obs)
#     K_obs, T_obs, C_obs = K_obs[m_obs], T_obs[m_obs], C_obs[m_obs]

#     # optional downsample (keeps plotting fast)
#     if downsample_obs is not None and K_obs.size > downsample_obs:
#         rng = np.random.default_rng(0)
#         idx = rng.choice(K_obs.size, size=int(downsample_obs), replace=False)
#         K_obs, T_obs, C_obs = K_obs[idx], T_obs[idx], C_obs[idx]

#     # fitted grid mesh
#     KK, TT = np.meshgrid(K_grid, T_grid)

#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection="3d")

#     # fitted surface
#     ax.plot_surface(
#         KK, TT, C_fit,
#         linewidth=0,
#         antialiased=True,
#         alpha=float(fitted_alpha),
#     )

#     # observed quotes
#     ax.scatter(
#         K_obs, T_obs, C_obs,
#         s=float(obs_size),
#         depthshade=True,
#     )

#     ax.set_title(title)
#     ax.set_xlabel("Strike K")
#     ax.set_ylabel("Maturity T")
#     ax.set_zlabel("Call Price")

#     plt.tight_layout()
#     plt.show()
#     return fig, ax
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union
def cdf_surface_plot(
    res: dict,
    *,
    title: str = "Risk Neutral CDF Surface",
    cmap: str = "viridis",
    save: Optional[Union[str, Path]] = None,
    interactive: bool = True,
    show: bool = True,
    clip01: bool = True,
):
    """
    Plot CDF surface F(K|T).

    interactive=True  -> Plotly 3D surface
    interactive=False -> Matplotlib 3D surface

    Save behavior (like your delta plot):
      - creates parent folders
      - if save is a directory (or ends with / or \\), auto-names a file
      - if save has no suffix, appends .html (interactive) or .png (static)
    """
    if "cdf_surface" not in res:
        raise KeyError("Missing CDF. Expected res['cdf_surface'].")

    F = np.asarray(res["cdf_surface"], float)
    K_grid = np.asarray(res["K_grid"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()

    if F.shape != (T_grid.size, K_grid.size):
        raise ValueError("CDF surface must have shape (len(T_grid), len(K_grid)).")

    if clip01:
        F = np.clip(F, 0.0, 1.0)

    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

    # -------------------------
    # Helper: normalize save path and create folder
    # -------------------------
    def _normalize_save_path(save_path: Union[str, Path], *, ext: str) -> Path:
        p = Path(save_path)

        # Handle "path/to/dir/" even if it doesn't exist yet
        s = str(save_path)
        looks_like_dir = s.endswith(("/", "\\"))

        # If it's an existing directory OR looks like a directory path, auto-name the file
        if (p.exists() and p.is_dir()) or looks_like_dir:
            p = p / f"cdf_surface{ext}"

        # If user passed no suffix, add the correct one
        if p.suffix == "":
            p = p.with_suffix(ext)

        # Make parent folder(s)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ============================================================
    # Plotly (interactive)
    # ============================================================
    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Surface(
                    x=K_mesh,
                    y=T_mesh,
                    z=F,
                    colorscale=cmap,
                    colorbar=dict(title="F(K)"),
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Strike K",
                yaxis_title="Maturity T (years)",
                zaxis_title="F(K|T)",
                zaxis=dict(range=[0, 1]),
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        if save is not None:
            sp = _normalize_save_path(save, ext=".html")
            # If user explicitly provided .png/.pdf/etc and you want that instead of html,
            # you can switch on suffix here. For now: interactive saves html by default.
            if sp.suffix.lower() == ".html":
                fig.write_html(str(sp))
            else:
                # requires kaleido for image formats
                fig.write_image(str(sp))
            print(f"[saved] {sp}")

        if show:
            fig.show()

        return fig

    # ============================================================
    # Matplotlib (static)
    # ============================================================
    else:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            K_mesh,
            T_mesh,
            F,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
        )

        ax.set_title(title)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel("F(K|T)")
        ax.set_zlim(0.0, 1.0)

        fig.colorbar(surf, shrink=0.6, aspect=12, label="F(K)")

        if save is not None:
            sp = _normalize_save_path(save, ext=".png")
            fig.savefig(sp, dpi=200, bbox_inches="tight")
            print(f"[saved] {sp}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


def rnd_surface_plot(
    res: dict,
    *,
    title: str = "Risk Neutral Density Surface",
    cmap: str = "viridis",
    save: str | None = None,
    interactive: bool = True,
    show: bool = True,
    dpi: int = 300,
    # x-axis options
    x_axis: str = "K",                            # {"K","R","r"}
    x_bounds: tuple[float, float] | None = None,  # bounds in chosen x units
    spot: float | None = None,                    # S0; required for {"R","r"} unless in res
    # ---- NEW (for subplot support) ----
    ax=None,                                      # matplotlib 3d axis to draw into
    add_colorbar: bool = True,                    # whether to add a colorbar
):
    """
    Plot RND surface q(K|T).

    x_axis:
      "K": x = K
      "R": x = K/S0
      "r": x = log(K/S0)

    x_bounds applied in chosen x units.

    Requires:
      res["rnd_surface"], res["K_grid"], res["T_grid"]
    """
    import numpy as np
    from pathlib import Path

    if "rnd_surface" not in res:
        raise KeyError("Missing RND. Expected res['rnd_surface'].")

    q = np.asarray(res["rnd_surface"], float)
    K_grid = np.asarray(res["K_grid"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()

    if q.shape != (T_grid.size, K_grid.size):
        raise ValueError("RND surface must have shape (len(T_grid), len(K_grid)).")

    # ---- spot ----
    if spot is None:
        spot = res.get("S0", None) or res.get("s0", None)
    spot = float(spot) if spot is not None else None

    x_axis = str(x_axis).strip()
    if x_axis not in {"K", "R", "r"}:
        raise ValueError("x_axis must be one of {'K','R','r'}.")
    if x_axis in {"R", "r"} and spot is None:
        raise ValueError("spot/S0 is required when x_axis is 'R' or 'r'.")

    # ---- bounds -> K-mask ----
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
        else:  # r
            k_mask = (K_grid >= spot * np.exp(lo)) & (K_grid <= spot * np.exp(hi))

    if not np.any(k_mask):
        raise ValueError("x_bounds produced an empty plotting window on K_grid.")

    Kp = K_grid[k_mask]
    qp = q[:, k_mask]

    # ---- x transform ----
    if x_axis == "K":
        X = Kp
        xlabel = "Strike K"
    elif x_axis == "R":
        X = Kp / spot
        xlabel = "Gross return R = K/S0"
    else:
        X = np.log(Kp / spot)
        xlabel = "Log return r = log(K/S0)"

    X_mesh, T_mesh = np.meshgrid(X, T_grid)

    # =========================
    # INTERACTIVE (Plotly)
    # =========================
    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X_mesh,
                    y=T_mesh,
                    z=qp,
                    colorscale=cmap,
                    colorbar=dict(title="q(K|T)"),
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title="Maturity T (years)",
                zaxis_title="q(K|T)",
            ),
            margin=dict(l=0, r=0, t=55, b=0),
        )

        if save is not None:
            save = Path(save)
            save.parent.mkdir(parents=True, exist_ok=True)
            if save.suffix.lower() == ".html":
                fig.write_html(save)
            else:
                fig.write_image(save)  # needs kaleido
            print(f"[saved] {save}")

        if show:
            fig.show()

        return fig

    # =========================
    # STATIC (Matplotlib)
    # =========================
    import matplotlib.pyplot as plt

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure

    surf = ax.plot_surface(
        X_mesh, T_mesh, qp,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )

    if add_colorbar:
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="q(K|T)")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("q(K|T)")
    # (optional) add a view_init here if you want parity with IV plots:
    # ax.view_init(elev=28, azim=-60)

    if created_fig:
        plt.tight_layout()

        if save is not None:
            save = Path(save)
            save.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save, dpi=dpi, bbox_inches="tight")
            print(f"[saved] {save}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig, ax, surf



def plot_call_surface_fit_with_obs(
    res: dict,
    *,
    day,
    title: str = "Fitted Call Surface with Observed Quotes",
    fitted_alpha: float = 0.55,
    obs_size: float = 18.0,
    downsample_obs: int | None = None,
):
    """
    Convenience wrapper around plot_fitted_surface_with_observed().
    Expects `res` to have: K_grid, T_grid, C_fit
    Expects `day` to have: K_obs, T_obs, C_obs
    """
    return plot_fitted_surface_with_observed(
        K_obs=day.K_obs,
        T_obs=day.T_obs,
        C_obs=day.C_obs,
        K_grid=res["K_grid"],
        T_grid=res["T_grid"],
        C_fit=res["C_fit"],
        title=title,
        fitted_alpha=fitted_alpha,
        obs_size=obs_size,
        downsample_obs=downsample_obs,
    )

def iv_surface_plot(
    res: dict,
    *,
    title: str = "Implied Volatility Surface",
    cmap: str = "viridis",
    save: str | None = None,
    dpi: int = 200,
    interactive: bool = False,
    show: bool = True,
    # view / style
    alpha: float = 0.9,
    shade: bool = False,
    elev: float = 28,
    azim: float = -60,
    # x-axis options
    x_axis: str = "K",                            # {"K","R","r"}
    x_bounds: tuple[float, float] | None = None,  # bounds in chosen x units
    spot: float | None = None,                    # S0; required for {"R","r"} unless in res
    # ---- NEW (for subplot support) ----
    ax=None,                                      # matplotlib 3d axis to draw into
    add_colorbar: bool = True,                    # whether to add a colorbar
):
    """
    IV surface plot (static Matplotlib or interactive Plotly).

    x_axis:
      "K": x = K
      "R": x = K/S0
      "r": x = log(K/S0)

    x_bounds applied in chosen x units.

    Expects:
      - res["iv_surface"] : array (len(T_grid), len(K_grid))
      - res["K_grid"]     : array (nK,)
      - res["T_grid"]     : array (nT,)
    """
    import numpy as np
    from pathlib import Path

    if "iv_surface" not in res:
        raise KeyError("Missing IV surface. Expected res['iv_surface'].")

    iv = np.asarray(res["iv_surface"], float)
    K = np.asarray(res["K_grid"], float).ravel()
    T = np.asarray(res["T_grid"], float).ravel()

    if iv.shape != (T.size, K.size):
        raise ValueError("IV surface must have shape (len(T_grid), len(K_grid)).")

    # ---- spot ----
    if spot is None:
        spot = res.get("S0", None) or res.get("s0", None)
    spot = float(spot) if spot is not None else None

    x_axis = str(x_axis).strip()
    if x_axis not in {"K", "R", "r"}:
        raise ValueError("x_axis must be one of {'K','R','r'}.")
    if x_axis in {"R", "r"} and spot is None:
        raise ValueError("spot/S0 is required when x_axis is 'R' or 'r'.")

    # ---- bounds -> K-mask ----
    if x_bounds is None:
        k_mask = np.isfinite(K)
    else:
        lo, hi = float(x_bounds[0]), float(x_bounds[1])
        if lo >= hi:
            raise ValueError("x_bounds must be (lo, hi) with lo < hi.")
        if x_axis == "K":
            k_mask = (K >= lo) & (K <= hi)
        elif x_axis == "R":
            k_mask = (K >= lo * spot) & (K <= hi * spot)
        else:  # r
            k_mask = (K >= spot * np.exp(lo)) & (K <= spot * np.exp(hi))

    if not np.any(k_mask):
        raise ValueError("x_bounds produced an empty plotting window on K_grid.")

    Kp = K[k_mask]
    ivp = iv[:, k_mask]

    # ---- x transform ----
    if x_axis == "K":
        X = Kp
        xlabel = "Strike K"
    elif x_axis == "R":
        X = Kp / spot
        xlabel = "Gross return R = K/S0"
    else:
        X = np.log(Kp / spot)
        xlabel = "Log return r = log(K/S0)"

    KK, TT = np.meshgrid(X, T)

    # =========================
    # INTERACTIVE (Plotly)
    # =========================
    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Surface(
                    x=KK,
                    y=TT,
                    z=ivp,
                    colorscale=cmap,
                    colorbar=dict(title="IV"),
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=xlabel,
                yaxis_title="Maturity T (years)",
                zaxis_title="Implied Volatility",
            ),
            margin=dict(l=0, r=0, t=55, b=0),
        )

        if save is not None:
            save = Path(save)
            save.parent.mkdir(parents=True, exist_ok=True)
            if save.suffix.lower() == ".html":
                fig.write_html(save)
            else:
                fig.write_image(save)  # needs kaleido
            print(f"[saved] {save}")

        if show:
            fig.show()

        return fig

    # =========================
    # STATIC (Matplotlib)
    # =========================
    import matplotlib.pyplot as plt

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure

    surf = ax.plot_surface(
        KK, TT, ivp,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=alpha,
        shade=shade,
    )

    if add_colorbar:
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="IV")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Implied Volatility")
    ax.view_init(elev=elev, azim=azim)

    if created_fig:
        plt.tight_layout()

        if save is not None:
            save = Path(save)
            save.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save, dpi=dpi, bbox_inches="tight")
            print(f"[saved] {save}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig, ax, surf

def iv_surface_delta_plot(
    res: dict,
    *,
    day: Optional[object] = None,
    # market inputs (if None, inferred from res/day)
    S0: Optional[float] = None,
    r: Optional[float] = None,
    q: float = 0.0,
    # delta grid / convention
    delta_grid: Optional[np.ndarray] = None,  # e.g. np.linspace(0.05, 0.95, 61)
    delta_type: str = "call",                 # "call" or "put"
    delta_eps: float = 1e-4,                  # clip away from 0/1 for stability
    # plot controls
    title: str = "IV Surface (Delta Space)",
    cmap: str = "viridis",
    interactive: bool = False,
    show: bool = True,
    # static view
    alpha: float = 0.9,
    shade: bool = False,
    elev: float = 28,
    azim: float = -60,
    # save
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
):
    """
    Convert IV surface from strike space to delta space and plot it.

    Expects:
      - res["iv_surface"] : (nT, nK)
      - res["K_grid"]     : (nK,)
      - res["T_grid"]     : (nT,)

    Needs S0, r, q to compute BS delta:
      - If S0/r not provided, we try res["S0"] / res["s0"] and day.S0, day.r.
      - q defaults to 0.0 unless provided or day.q exists.

    Output:
      - Returns fig (plotly Figure if interactive else matplotlib Figure)
      - Also returns (delta_grid, T_grid, iv_delta) so you can reuse.
    """
    # -------------------------
    # pull inputs
    # -------------------------
    if "iv_surface" not in res:
        raise KeyError("Missing IV surface. Expected res['iv_surface'].")
    K_grid = np.asarray(res["K_grid"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()
    iv = np.asarray(res["iv_surface"], float)

    if iv.shape != (T_grid.size, K_grid.size):
        raise ValueError("IV surface must have shape (len(T_grid), len(K_grid)).")
    if np.any(K_grid <= 0) or np.any(T_grid <= 0):
        raise ValueError("K_grid and T_grid must be positive.")

    # infer S0, r, q if needed
    if S0 is None:
        S0 = res.get("S0", None) or res.get("s0", None) or (getattr(day, "S0", None) if day is not None else None)
    if r is None:
        r = res.get("r", None) or (getattr(day, "r", None) if day is not None else None)
    if day is not None and q == 0.0 and hasattr(day, "q") and getattr(day, "q") is not None:
        q = float(getattr(day, "q"))

    if S0 is None or r is None:
        raise ValueError("Need S0 and r to compute deltas. Provide S0=..., r=... or pass day with day.S0/day.r (or store in res).")

    S0 = float(S0)
    r = float(r)
    q = float(q)

    # delta grid default
    if delta_grid is None:
        delta_grid = np.linspace(0.05, 0.95, 61)
    delta_grid = np.asarray(delta_grid, float).ravel()

    # -------------------------
    # delta conversion per maturity
    # -------------------------
    # use scipy if available; otherwise implement standard normal cdf via erf
    try:
        from scipy.stats import norm
        cdf = norm.cdf
    except Exception:
        from math import erf, sqrt
        def cdf(x):
            x = np.asarray(x, float)
            return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))

    iv_delta = np.full((T_grid.size, delta_grid.size), np.nan, float)

    for i, Ti in enumerate(T_grid):
        sig = iv[i, :]
        m = np.isfinite(sig) & (sig > 0) & np.isfinite(K_grid) & (K_grid > 0) & np.isfinite(Ti) & (Ti > 0)
        if np.sum(m) < 4:
            continue

        Km = K_grid[m]
        sigm = sig[m]

        vol_sqrtT = sigm * np.sqrt(Ti)
        # guard against div-by-zero
        ok = np.isfinite(vol_sqrtT) & (vol_sqrtT > 0)
        if np.sum(ok) < 4:
            continue

        Km = Km[ok]
        sigm = sigm[ok]
        vol_sqrtT = vol_sqrtT[ok]

        d1 = (np.log(S0 / Km) + (r - q + 0.5 * sigm * sigm) * Ti) / vol_sqrtT
        call_delta = np.exp(-q * Ti) * cdf(d1)

        if delta_type.lower() == "put":
            delt = call_delta - np.exp(-q * Ti)  # put delta in [-e^{-qT}, 0]
            # map to (0,1) “put absolute delta” if you prefer:
            # delt = np.abs(delt)
        else:
            delt = call_delta  # in (0, e^{-qT})

        keep = np.isfinite(delt) & (delt > delta_eps) & (delt < (1.0 - delta_eps))
        if np.sum(keep) < 4:
            continue

        ds = np.asarray(delt[keep], float)
        vs = np.asarray(sigm[keep], float)

        order = np.argsort(ds)
        ds = ds[order]
        vs = vs[order]

        ds_u, idx_u = np.unique(ds, return_index=True)
        vs_u = vs[idx_u]
        if ds_u.size < 4:
            continue

        iv_delta[i, :] = np.interp(delta_grid, ds_u, vs_u, left=np.nan, right=np.nan)

    # -------------------------
    # Plot
    # -------------------------
    DD, TT = np.meshgrid(delta_grid, T_grid)

    if interactive:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Surface(
                    x=DD,
                    y=TT,
                    z=iv_delta,
                    colorscale=cmap,
                    colorbar=dict(title="IV"),
                )
            ]
        )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Delta (call)" if delta_type.lower() == "call" else "Delta (put)",
                yaxis_title="Maturity T (years)",
                zaxis_title="Implied Volatility",
            ),
            margin=dict(l=0, r=0, t=55, b=0),
        )

        if save is not None:
            save = Path(save)
            save.parent.mkdir(parents=True, exist_ok=True)
            if save.suffix.lower() == ".html":
                fig.write_html(save)
            else:
                # requires: pip install -U kaleido
                fig.write_image(save)
            print(f"[saved] {save}")

        if show:
            fig.show()

        return fig, delta_grid, T_grid, iv_delta

    # static matplotlib
    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        DD, TT, iv_delta,
        cmap=cmap,
        linewidth=0.0,
        antialiased=True,
        alpha=float(alpha),
        shade=bool(shade),
    )
    ax.set_title(title, pad=18)
    ax.set_xlabel("Delta (call)" if delta_type.lower() == "call" else "Delta (put)")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Implied Volatility")
    ax.view_init(elev=float(elev), azim=float(azim))
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("IV")
    fig.tight_layout()

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {save}")

    if show:
        plt.show()

    return fig, delta_grid, T_grid, iv_delta
def call_surface_vs_observed(
    res: dict,
    *,
    day,
    title: str = "Call Surface vs Observed Quotes",
    date_str: Optional[str] = None,
    spot: Optional[float] = None,
    # surface rendering controls
    surface_stride: int = 2,       # larger -> faster / less dense surface
    surface_alpha: float = 0.55,
    # observed scatter controls
    obs_size: float = 18.0,
    obs_alpha: float = 0.9,
    # save controls
    save: Optional[Union[str, Path]] = None,
    dpi: int = 220,
    show: bool = True,
    # ---- NEW ----
    x_axis: str = "K",                            # {"K","R","r"}
    x_bounds: tuple[float, float] | None = None,  # bounds in chosen x units
):
    """
    3D plot: fitted call surface (x,T -> C) with observed quotes overlay.

    x_axis:
      "K": x = K
      "R": x = K/S0
      "r": x = log(K/S0)

    Notes:
      - For x_axis in {"R","r"}, spot/S0 must be available (inferred or passed).
      - x_bounds are applied in chosen x units.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # ---- fitted surface
    if "C_fit" not in res:
        raise KeyError("Missing fitted surface. Expected res['C_fit'].")
    K_grid = np.asarray(res["K_grid"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()
    C_fit  = np.asarray(res["C_fit"], float)

    if C_fit.shape != (T_grid.size, K_grid.size):
        raise ValueError("res['C_fit'] must have shape (len(T_grid), len(K_grid)).")

    # ---- observed quotes
    K_obs = np.asarray(getattr(day, "K_obs"), float).ravel()
    T_obs = np.asarray(getattr(day, "T_obs"), float).ravel()
    C_obs = np.asarray(getattr(day, "C_obs"), float).ravel()

    m = (
        np.isfinite(K_obs) & np.isfinite(T_obs) & np.isfinite(C_obs)
        & (K_obs > 0) & (T_obs > 0) & (C_obs >= 0)
    )
    K_obs, T_obs, C_obs = K_obs[m], T_obs[m], C_obs[m]
    if K_obs.size == 0:
        raise ValueError("No valid observed quotes to plot.")

    # ---- spot inference (needed for R/r)
    if spot is None:
        spot = (
            res.get("S0", None)
            or res.get("s0", None)
            or getattr(day, "S0", None)
            or getattr(day, "s0", None)
            or getattr(day, "spot", None)
        )
    spot = float(spot) if spot is not None else None

    x_axis = str(x_axis).strip()
    if x_axis not in {"K", "R", "r"}:
        raise ValueError("x_axis must be one of {'K','R','r'}.")
    if x_axis in {"R", "r"} and spot is None:
        raise ValueError("spot/S0 is required when x_axis is 'R' or 'r'.")

    # ---- apply x_bounds by masking on K-grid (then transform x)
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
        else:  # r
            k_mask = (K_grid >= spot * np.exp(lo)) & (K_grid <= spot * np.exp(hi))

    if not np.any(k_mask):
        raise ValueError("x_bounds produced an empty plotting window on K_grid.")

    Kp = K_grid[k_mask]
    Cp = C_fit[:, k_mask]

    # ---- x transform for surface and obs
    if x_axis == "K":
        X_grid = Kp
        X_obs = K_obs
        xlabel = "Strike K"
        spot_line_x = spot  # vertical line at K=spot
        show_spot_line = (spot is not None) and np.isfinite(spot)
    elif x_axis == "R":
        X_grid = Kp / spot
        X_obs = K_obs / spot
        xlabel = "Gross return R = K/S0"
        spot_line_x = 1.0
        show_spot_line = True
    else:  # x_axis == "r"
        X_grid = np.log(Kp / spot)
        X_obs = np.log(K_obs / spot)
        xlabel = "Log return r = log(K/S0)"
        spot_line_x = 0.0
        show_spot_line = True

    # ---- mesh for surface
    XX, TT = np.meshgrid(X_grid, T_grid)

    # ---- downsample surface for speed
    s = max(int(surface_stride), 1)
    XXp = XX[::s, ::s]
    TTp = TT[::s, ::s]
    Cp2 = Cp[::s, ::s]

    # ---- plot
    fig = plt.figure(figsize=(11.5, 9.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(XXp, TTp, Cp2, linewidth=0, antialiased=True, alpha=float(surface_alpha))
    ax.scatter(X_obs, T_obs, C_obs, s=float(obs_size), alpha=float(obs_alpha), label="Observed")

    # optional spot "curtain" on base plane
    if show_spot_line and (spot_line_x is not None) and np.isfinite(spot_line_x):
        tmin, tmax = float(np.min(T_grid)), float(np.max(T_grid))
        ax.plot([spot_line_x, spot_line_x], [tmin, tmax], [0.0, 0.0],
                linestyle="--", linewidth=2.0, label="Spot")

    ttl = title
    if date_str:
        ttl = f"{title} — {date_str}"
    ax.set_title(ttl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Call price C")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left")

    plt.tight_layout()

    if save is not None:
        save = Path(save)
        if save.suffix == "":
            save = save.with_suffix(".png")
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {save}")

    if show:
        plt.show()

    return fig

# def plot_delta_surface(
#     res: dict,
#     *,
#     which: Literal["call", "put", "skew"] = "call",
#     title: str = "IV in Delta Space",
#     cmap: str = "viridis",
#     interactive: bool = False,
#     save: Optional[Union[str, Path]] = None,
#     dpi: int = 200,
#     elev: float = 25.0,
#     azim: float = -60.0,
# ):
#     """
#     Plot delta-based surfaces stored in res["delta_dict"].

#     Expects res["delta_dict"] with keys:
#       - "delta_axis": (nD,)
#       - "T_axis": (nT,)
#       - "iv_delta_call": (nT, nD)
#       - "iv_delta_put_abs": (nT, nD)
#       - "delta_skew_surface": (nT, nD)

#     Parameters
#     ----------
#     which:
#       "call" -> iv_delta_call
#       "put"  -> iv_delta_put_abs
#       "skew" -> delta_skew_surface
#     interactive:
#       If True, uses Plotly surface (HTML output if saved).
#       If False, uses Matplotlib 3D surface (PNG output if saved).
#     save:
#       If provided, saves the figure. Creates parent folders if needed.
#       - static: saves via fig.savefig(...)
#       - interactive: saves via plotly.write_html(...)
#     """

#     if "delta_dict" not in res or res["delta_dict"] is None:
#         raise KeyError("Missing delta_dict. Expected res['delta_dict'].")

#     d = res["delta_dict"]

#     delta = np.asarray(d["delta_axis"], float).ravel()
#     T = np.asarray(d["T_axis"], float).ravel()

#     if which == "call":
#         Z = np.asarray(d["iv_delta_call"], float)
#         zlabel = "IV(call delta)"
#     elif which == "put":
#         Z = np.asarray(d["iv_delta_put_abs"], float)
#         zlabel = "IV(|put delta|)"
#     elif which == "skew":
#         Z = np.asarray(d["delta_skew_surface"], float)
#         zlabel = "Delta skew = (IV_call - IV_put)/IV_atm"
#     else:
#         raise ValueError("which must be one of {'call','put','skew'}.")

#     if Z.shape != (T.size, delta.size):
#         raise ValueError(
#             f"Surface has shape {Z.shape}, expected {(T.size, delta.size)} from T_axis/delta_axis."
#         )

#     # mesh: X=delta, Y=T
#     X, Y = np.meshgrid(delta, T)

#     # ---------------------------
#     # Interactive (Plotly)
#     # ---------------------------
#     if interactive:
#         try:
#             import plotly.graph_objects as go
#         except ImportError as e:
#             raise ImportError(
#                 "Plotly is required for interactive=True. Install with: pip install plotly"
#             ) from e

#         fig = go.Figure(
#             data=[
#                 go.Surface(
#                     x=X,
#                     y=Y,
#                     z=Z,
#                     colorscale=cmap,  # plotly accepts many named scales; if not found, it errors
#                     showscale=True,
#                 )
#             ]
#         )
#         fig.update_layout(
#             title=title,
#             scene=dict(
#                 xaxis_title="Delta",
#                 yaxis_title="Maturity T (years)",
#                 zaxis_title=zlabel,
#             ),
#             margin=dict(l=0, r=0, t=50, b=0),
#         )

#         if save is not None:
#             save = Path(save)
#             save.parent.mkdir(parents=True, exist_ok=True)
#             # default extension
#             if save.suffix.lower() not in {".html"}:
#                 save = save.with_suffix(".html")
#             fig.write_html(str(save))
#             print(f"[saved] {save}")

#         fig.show()
#         return fig

#     # ---------------------------
#     # Static (Matplotlib)
#     # ---------------------------
#     fig = plt.figure(figsize=(10.0, 8.0))
#     ax = fig.add_subplot(111, projection="3d")

#     surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)
#     fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.08)

#     ax.set_title(title)
#     ax.set_xlabel("Delta")
#     ax.set_ylabel("Maturity T (years)")
#     ax.set_zlabel(zlabel)
#     ax.view_init(elev=elev, azim=azim)

#     fig.tight_layout()

#     if save is not None:
#         save = Path(save)
#         save.parent.mkdir(parents=True, exist_ok=True)
#         # default extension
#         if save.suffix == "":
#             save = save.with_suffix(".png")
#         fig.savefig(save, dpi=dpi, bbox_inches="tight")
#         print(f"[saved] {save}")

#     plt.show()
#     return fig
def plot_delta_surface(
    res: dict,
    *,
    which: Literal["call", "put", "skew"] = "call",
    title: str = "Surface in Delta Space",
    cmap: str = "viridis",
    interactive: bool = False,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    elev: float = 25.0,
    azim: float = -60.0,
):
    """
    Plot delta-based IV surfaces stored in res["delta_dict"].

    Parameters
    ----------
    res : dict
        Results dictionary containing "delta_dict"
    which : {"call", "put", "skew"}
        Which surface to plot:
        - "call": call IV surface
        - "put": put IV surface (absolute delta)
        - "skew": delta skew surface
    title : str
        Plot title
    cmap : str
        Colormap name
    interactive : bool
        If True, use Plotly for interactive 3D plot; otherwise use Matplotlib
    save : str or Path, optional
        Save path (.png for static, .html for interactive)
    dpi : int
        DPI for saved static plots
    elev : float
        Elevation angle for static 3D view
    azim : float
        Azimuthal angle for static 3D view

    Expected keys in res["delta_dict"]
    ----------------------------------
    delta_axis : (nD,)
    T_axis : (nT,)
    iv_delta_call : (nT, nD)
    iv_delta_put_abs : (nT, nD)
    delta_skew_surface : (nT, nD)
    """
    if "delta_dict" not in res or res["delta_dict"] is None:
        raise KeyError("Missing delta_dict. Expected res['delta_dict'].")

    d = res["delta_dict"]
    delta = np.asarray(d["delta_axis"], float).ravel()
    T = np.asarray(d["T_axis"], float).ravel()

    # Pick surface and label based on which
    if which == "call":
        Z_key = "iv_delta_call"
        zlabel = "IV(call delta)"
    elif which == "put":
        Z_key = "iv_delta_put_abs"
        zlabel = "IV(|put delta|)"
    elif which == "skew":
        Z_key = "delta_skew_surface"
        zlabel = "Delta skew = (IV_call - IV_put)/IV_atm"
    else:
        raise ValueError("which must be one of {'call', 'put', 'skew'}.")

    if Z_key not in d:
        raise KeyError(f"delta_dict is missing '{Z_key}'.")

    Z = np.asarray(d[Z_key], float)
    if Z.shape != (T.size, delta.size):
        raise ValueError(f"Surface has shape {Z.shape}, expected {(T.size, delta.size)} from T_axis/delta_axis.")

    X, Y = np.meshgrid(delta, T)

    # Interactive (Plotly)
    if interactive:
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("Plotly is required for interactive=True. Install with: pip install plotly") from e

        fig = go.Figure(
            data=[go.Surface(x=X, y=Y, z=Z, colorscale=cmap, showscale=True)]
        )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Delta",
                yaxis_title="Maturity T (years)",
                zaxis_title=zlabel,
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        if save is not None:
            save = Path(save)
            save.parent.mkdir(parents=True, exist_ok=True)
            if save.suffix.lower() != ".html":
                save = save.with_suffix(".html")
            fig.write_html(str(save))
            print(f"[saved] {save}")

        fig.show()
        return fig

    # Static (Matplotlib)
    fig = plt.figure(figsize=(10.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.08)

    ax.set_title(title)
    ax.set_xlabel("Delta")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=float(elev), azim=float(azim))

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