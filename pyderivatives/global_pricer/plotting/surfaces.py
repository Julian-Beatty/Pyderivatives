# pyderivatives/global_pricer/plotting/surfaces.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================

XAxis = Literal["k", "lr", "r", "K", "LR", "R"]


def _canonical_x_axis(x_axis: str) -> Literal["k", "lr", "r"]:
    x = str(x_axis).strip().lower()

    if x in {"k", "strike"}:
        return "k"
    if x in {"lr", "logreturn", "log_return", "log-return", "log return"}:
        return "lr"
    if x in {"r", "gross", "grossreturn", "gross_return", "gross-return", "gross return"}:
        return "r"

    raise ValueError("x_axis must be one of {'k', 'lr', 'r'}.")


def _normalize_save_path(
    save: Union[str, Path],
    *,
    default_name: str,
    ext: str,
) -> Path:
    p = Path(save)
    s = str(save)

    looks_like_dir = s.endswith(("/", "\\"))

    if (p.exists() and p.is_dir()) or looks_like_dir:
        p = p / f"{default_name}{ext}"

    if p.suffix == "":
        p = p.with_suffix(ext)

    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _get_grid(res: dict, x_axis: XAxis) -> np.ndarray:
    x_axis = _canonical_x_axis(x_axis)

    key = {
        "k": "grid_k",
        "lr": "grid_lr",
        "r": "grid_r",
    }[x_axis]

    if key not in res:
        raise KeyError(f"Missing required grid: res['{key}'].")

    return np.asarray(res[key], float).ravel()


def _get_rnd_surface(res: dict, x_axis: XAxis) -> np.ndarray:
    x_axis = _canonical_x_axis(x_axis)

    key = {
        "k": "rnd_k_surface",
        "lr": "rnd_lr_surface",
        "r": "rnd_r_surface",
    }[x_axis]

    if key not in res:
        raise KeyError(f"Missing required RND surface: res['{key}'].")

    return np.asarray(res[key], float)


def _x_label(x_axis: XAxis) -> str:
    x_axis = _canonical_x_axis(x_axis)

    return {
        "k": "Strike K",
        "lr": "Log return lr = log(K/S0)",
        "r": "Gross return R = K/S0",
    }[x_axis]


def _rnd_z_label(x_axis: XAxis) -> str:
    x_axis = _canonical_x_axis(x_axis)

    return {
        "k": "q_K(K|T)",
        "lr": "q_lr(lr|T)",
        "r": "q_R(R|T)",
    }[x_axis]


def _mask_axis(x: np.ndarray, x_bounds: Optional[tuple[float, float]]) -> np.ndarray:
    x = np.asarray(x, float).ravel()

    if x_bounds is None:
        mask = np.isfinite(x)
    else:
        lo, hi = map(float, x_bounds)

        if lo >= hi:
            raise ValueError("x_bounds must be (lo, hi) with lo < hi.")

        mask = np.isfinite(x) & (x >= lo) & (x <= hi)

    if not np.any(mask):
        raise ValueError("x_bounds produced an empty plotting window.")

    return mask


def _surface_plotly(
    *,
    x: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    title: str,
    xlabel: str,
    zlabel: str,
    colorbar_title: str,
    cmap: str,
    save: Optional[Union[str, Path]],
    save_default_name: str,
    show: bool,
    z_range: Optional[tuple[float, float]] = None,
):
    import plotly.graph_objects as go

    X_mesh, T_mesh = np.meshgrid(x, T)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X_mesh,
                y=T_mesh,
                z=Z,
                colorscale=cmap,
                colorbar=dict(title=colorbar_title),
            )
        ]
    )

    scene = dict(
        xaxis_title=xlabel,
        yaxis_title="Maturity T (years)",
        zaxis_title=zlabel,
    )

    if z_range is not None:
        scene["zaxis"] = dict(range=list(z_range))

    fig.update_layout(
        title=title,
        scene=scene,
        margin=dict(l=0, r=0, t=55, b=0),
    )

    if save is not None:
        sp = _normalize_save_path(save, default_name=save_default_name, ext=".html")

        if sp.suffix.lower() == ".html":
            fig.write_html(str(sp))
        else:
            fig.write_image(str(sp))

        print(f"[saved] {sp}")

    if show:
        fig.show()

    return fig


def _surface_matplotlib(
    *,
    x: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    title: str,
    xlabel: str,
    zlabel: str,
    colorbar_label: str,
    cmap: str,
    save: Optional[Union[str, Path]],
    save_default_name: str,
    show: bool,
    dpi: int,
    elev: float,
    azim: float,
    alpha: float,
    shade: bool,
    zlim: Optional[tuple[float, float]] = None,
    ax=None,
    add_colorbar: bool = True,
):
    X_mesh, T_mesh = np.meshgrid(x, T)

    created_fig = False

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure

    surf = ax.plot_surface(
        X_mesh,
        T_mesh,
        Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=float(alpha),
        shade=bool(shade),
    )

    if add_colorbar and created_fig:
        fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label=colorbar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel(zlabel)

    if zlim is not None:
        ax.set_zlim(*zlim)

    ax.view_init(elev=float(elev), azim=float(azim))

    if created_fig:
        fig.tight_layout()

    if save is not None:
        sp = _normalize_save_path(save, default_name=save_default_name, ext=".png")
        fig.savefig(sp, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {sp}")

    if show and created_fig:
        plt.show()
    elif created_fig and not show:
        plt.close(fig)

    return fig, ax, surf


# ============================================================
# CDF Surface
# ============================================================

def cdf_surface_plot(
    res: dict,
    *,
    title: str = "Risk-Neutral CDF Surface",
    cmap: str = "viridis",
    save: Optional[Union[str, Path]] = None,
    interactive: bool = True,
    show: bool = True,
    clip01: bool = True,
    dpi: int = 300,
    elev: float = 28,
    azim: float = -60,
    alpha: float = 0.9,
    shade: bool = False,
    x_bounds: Optional[tuple[float, float]] = None,
    ax=None,
    add_colorbar: bool = True,
):
    """
    Plot the risk-neutral CDF surface F(K|T).

    New-interface keys required:
      - res["rnd_cdf_surface"]
      - res["grid_k"]
      - res["T_grid"]
    """
    if "rnd_cdf_surface" not in res:
        raise KeyError("Missing CDF. Expected res['rnd_cdf_surface'].")

    F = np.asarray(res["rnd_cdf_surface"], float)
    K = np.asarray(res["grid_k"], float).ravel()
    T = np.asarray(res["T_grid"], float).ravel()

    if F.shape != (T.size, K.size):
        raise ValueError(f"rnd_cdf_surface has shape {F.shape}, expected {(T.size, K.size)}.")

    mask = _mask_axis(K, x_bounds)

    Kp = K[mask]
    Fp = F[:, mask]

    if clip01:
        Fp = np.clip(Fp, 0.0, 1.0)

    if interactive:
        return _surface_plotly(
            x=Kp,
            T=T,
            Z=Fp,
            title=title,
            xlabel="Strike K",
            zlabel="F(K|T)",
            colorbar_title="F(K)",
            cmap=cmap,
            save=save,
            save_default_name="cdf_surface",
            show=show,
            z_range=(0.0, 1.0),
        )

    return _surface_matplotlib(
        x=Kp,
        T=T,
        Z=Fp,
        title=title,
        xlabel="Strike K",
        zlabel="F(K|T)",
        colorbar_label="F(K)",
        cmap=cmap,
        save=save,
        save_default_name="cdf_surface",
        show=show,
        dpi=dpi,
        elev=elev,
        azim=azim,
        alpha=alpha,
        shade=shade,
        zlim=(0.0, 1.0),
        ax=ax,
        add_colorbar=add_colorbar,
    )


# ============================================================
# RND Surface
# ============================================================

def rnd_surface_plot(
    res: dict,
    *,
    title: str = "Risk-Neutral Density Surface",
    cmap: str = "viridis",
    save: Optional[Union[str, Path]] = None,
    interactive: bool = True,
    show: bool = True,
    dpi: int = 300,
    elev: float = 28,
    azim: float = -60,
    alpha: float = 0.9,
    shade: bool = False,
    x_axis: XAxis = "k",
    x_bounds: Optional[tuple[float, float]] = None,
    ax=None,
    add_colorbar: bool = True,
):
    """
    Plot the risk-neutral density surface.

    New-interface keys used by x_axis:
      x_axis="k"  -> res["grid_k"],  res["rnd_k_surface"]
      x_axis="lr" -> res["grid_lr"], res["rnd_lr_surface"]
      x_axis="r"  -> res["grid_r"],  res["rnd_r_surface"]
    """
    x_axis = _canonical_x_axis(x_axis)

    x = _get_grid(res, x_axis)
    q = _get_rnd_surface(res, x_axis)
    T = np.asarray(res["T_grid"], float).ravel()

    if q.shape != (T.size, x.size):
        raise ValueError(f"RND surface has shape {q.shape}, expected {(T.size, x.size)}.")

    mask = _mask_axis(x, x_bounds)

    xp = x[mask]
    qp = q[:, mask]

    xlabel = _x_label(x_axis)
    zlabel = _rnd_z_label(x_axis)

    if interactive:
        return _surface_plotly(
            x=xp,
            T=T,
            Z=qp,
            title=title,
            xlabel=xlabel,
            zlabel=zlabel,
            colorbar_title=zlabel,
            cmap=cmap,
            save=save,
            save_default_name="rnd_surface",
            show=show,
        )

    return _surface_matplotlib(
        x=xp,
        T=T,
        Z=qp,
        title=title,
        xlabel=xlabel,
        zlabel=zlabel,
        colorbar_label=zlabel,
        cmap=cmap,
        save=save,
        save_default_name="rnd_surface",
        show=show,
        dpi=dpi,
        elev=elev,
        azim=azim,
        alpha=alpha,
        shade=shade,
        ax=ax,
        add_colorbar=add_colorbar,
    )


# ============================================================
# IV Surface
# ============================================================

def iv_surface_plot(
    res: dict,
    *,
    title: str = "Implied Volatility Surface",
    cmap: str = "viridis",
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    interactive: bool = False,
    show: bool = True,
    alpha: float = 0.9,
    shade: bool = False,
    elev: float = 28,
    azim: float = -60,
    x_axis: XAxis = "k",
    x_bounds: Optional[tuple[float, float]] = None,
    ax=None,
    add_colorbar: bool = True,
):
    """
    Plot IV surface.

    New-interface keys:
      - res["iv_surface"]
      - res["T_grid"]
      - grid chosen by x_axis:
          "k"  -> res["grid_k"]
          "lr" -> res["grid_lr"]
          "r"  -> res["grid_r"]
    """
    if "iv_surface" not in res:
        raise KeyError("Missing IV surface. Expected res['iv_surface'].")

    x_axis = _canonical_x_axis(x_axis)

    iv = np.asarray(res["iv_surface"], float)
    x = _get_grid(res, x_axis)
    T = np.asarray(res["T_grid"], float).ravel()

    if iv.shape != (T.size, x.size):
        raise ValueError(f"iv_surface has shape {iv.shape}, expected {(T.size, x.size)}.")

    mask = _mask_axis(x, x_bounds)

    xp = x[mask]
    ivp = iv[:, mask]

    xlabel = _x_label(x_axis)

    if interactive:
        return _surface_plotly(
            x=xp,
            T=T,
            Z=ivp,
            title=title,
            xlabel=xlabel,
            zlabel="Implied Volatility",
            colorbar_title="IV",
            cmap=cmap,
            save=save,
            save_default_name="iv_surface",
            show=show,
        )

    return _surface_matplotlib(
        x=xp,
        T=T,
        Z=ivp,
        title=title,
        xlabel=xlabel,
        zlabel="Implied Volatility",
        colorbar_label="IV",
        cmap=cmap,
        save=save,
        save_default_name="iv_surface",
        show=show,
        dpi=dpi,
        elev=elev,
        azim=azim,
        alpha=alpha,
        shade=shade,
        ax=ax,
        add_colorbar=add_colorbar,
    )


# ============================================================
# IV Surface Converted to Delta Space
# ============================================================

def iv_surface_delta_plot(
    res: dict,
    *,
    day: Optional[object] = None,
    S0: Optional[float] = None,
    r: Optional[float] = None,
    q: float = 0.0,
    delta_grid: Optional[np.ndarray] = None,
    delta_type: Literal["call", "put"] = "call",
    delta_eps: float = 1e-4,
    title: str = "IV Surface in Delta Space",
    cmap: str = "viridis",
    interactive: bool = False,
    show: bool = True,
    alpha: float = 0.9,
    shade: bool = False,
    elev: float = 28,
    azim: float = -60,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
):
    """
    Convert IV surface from strike space to delta space and plot it.

    New-interface keys:
      - res["iv_surface"]
      - res["grid_k"]
      - res["T_grid"]

    Returns:
      - interactive=True:  fig, delta_grid, T_grid, iv_delta
      - interactive=False: fig, delta_grid, T_grid, iv_delta
    """
    if "iv_surface" not in res:
        raise KeyError("Missing IV surface. Expected res['iv_surface'].")

    K = np.asarray(res["grid_k"], float).ravel()
    T = np.asarray(res["T_grid"], float).ravel()
    iv = np.asarray(res["iv_surface"], float)

    if iv.shape != (T.size, K.size):
        raise ValueError(f"iv_surface has shape {iv.shape}, expected {(T.size, K.size)}.")

    if np.any(K <= 0) or np.any(T <= 0):
        raise ValueError("grid_k and T_grid must be positive.")

    if S0 is None:
        S0 = res.get("S0", None) or getattr(day, "S0", None) if day is not None else res.get("S0", None)

    if r is None:
        r = res.get("r", None) or getattr(day, "r", None) if day is not None else res.get("r", None)

    if day is not None and q == 0.0 and hasattr(day, "q") and getattr(day, "q") is not None:
        q = float(getattr(day, "q"))

    if S0 is None or r is None:
        raise ValueError("Need S0 and r to compute deltas.")

    S0 = float(S0)
    r = float(r)
    q = float(q)

    if delta_grid is None:
        delta_grid = np.linspace(0.05, 0.95, 61)

    delta_grid = np.asarray(delta_grid, float).ravel()

    try:
        from scipy.stats import norm

        cdf = norm.cdf
    except Exception:
        from math import erf, sqrt

        def cdf(z):
            z = np.asarray(z, float)
            return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))

    iv_delta = np.full((T.size, delta_grid.size), np.nan, float)

    for i, Ti in enumerate(T):
        sig = iv[i, :]

        valid = (
            np.isfinite(sig)
            & (sig > 0)
            & np.isfinite(K)
            & (K > 0)
            & np.isfinite(Ti)
            & (Ti > 0)
        )

        if np.sum(valid) < 4:
            continue

        Km = K[valid]
        sigm = sig[valid]

        vol_sqrtT = sigm * np.sqrt(Ti)

        ok = np.isfinite(vol_sqrtT) & (vol_sqrtT > 0)

        if np.sum(ok) < 4:
            continue

        Km = Km[ok]
        sigm = sigm[ok]
        vol_sqrtT = vol_sqrtT[ok]

        d1 = (np.log(S0 / Km) + (r - q + 0.5 * sigm**2) * Ti) / vol_sqrtT
        call_delta = np.exp(-q * Ti) * cdf(d1)

        if delta_type == "put":
            delt = np.abs(call_delta - np.exp(-q * Ti))
        else:
            delt = call_delta

        keep = (
            np.isfinite(delt)
            & np.isfinite(sigm)
            & (delt > delta_eps)
            & (delt < 1.0 - delta_eps)
        )

        if np.sum(keep) < 4:
            continue

        ds = delt[keep]
        vs = sigm[keep]

        order = np.argsort(ds)
        ds = ds[order]
        vs = vs[order]

        ds_u, idx_u = np.unique(ds, return_index=True)
        vs_u = vs[idx_u]

        if ds_u.size < 4:
            continue

        iv_delta[i, :] = np.interp(delta_grid, ds_u, vs_u, left=np.nan, right=np.nan)

    xlabel = "Call delta" if delta_type == "call" else "|Put delta|"

    if interactive:
        return (
            _surface_plotly(
                x=delta_grid,
                T=T,
                Z=iv_delta,
                title=title,
                xlabel=xlabel,
                zlabel="Implied Volatility",
                colorbar_title="IV",
                cmap=cmap,
                save=save,
                save_default_name="iv_delta_surface",
                show=show,
            ),
            delta_grid,
            T,
            iv_delta,
        )

    fig, ax, surf = _surface_matplotlib(
        x=delta_grid,
        T=T,
        Z=iv_delta,
        title=title,
        xlabel=xlabel,
        zlabel="Implied Volatility",
        colorbar_label="IV",
        cmap=cmap,
        save=save,
        save_default_name="iv_delta_surface",
        show=show,
        dpi=dpi,
        elev=elev,
        azim=azim,
        alpha=alpha,
        shade=shade,
    )

    return fig, delta_grid, T, iv_delta


# ============================================================
# Call Surface vs Observed Quotes
# ============================================================

def call_surface_vs_observed(
    res: dict,
    *,
    day,
    title: str = "Call Surface vs Observed Quotes",
    date_str: Optional[str] = None,
    spot: Optional[float] = None,
    surface_stride: int = 2,
    surface_alpha: float = 0.55,
    obs_size: float = 18.0,
    obs_alpha: float = 0.9,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    show: bool = True,
    x_axis: XAxis = "k",
    x_bounds: Optional[tuple[float, float]] = None,
    ax=None,
):
    """
    3D plot of fitted call surface with observed quotes overlay.

    New-interface keys:
      - res["C_fit"]
      - res["grid_k"]
      - res["grid_lr"] or res["grid_r"] if x_axis is not "k"
      - res["T_grid"]
    """
    if "C_fit" not in res:
        raise KeyError("Missing fitted call surface. Expected res['C_fit'].")

    x_axis = _canonical_x_axis(x_axis)

    K = np.asarray(res["grid_k"], float).ravel()
    x = _get_grid(res, x_axis)
    T = np.asarray(res["T_grid"], float).ravel()
    C = np.asarray(res["C_fit"], float)

    if C.shape != (T.size, K.size):
        raise ValueError(f"C_fit has shape {C.shape}, expected {(T.size, K.size)}.")

    if x.size != K.size:
        raise ValueError(f"{x_axis=} grid has length {x.size}, expected {K.size}.")

    K_obs = np.asarray(getattr(day, "K_obs"), float).ravel()
    T_obs = np.asarray(getattr(day, "T_obs"), float).ravel()
    C_obs = np.asarray(getattr(day, "C_obs"), float).ravel()

    valid_obs = (
        np.isfinite(K_obs)
        & np.isfinite(T_obs)
        & np.isfinite(C_obs)
        & (K_obs > 0)
        & (T_obs > 0)
        & (C_obs >= 0)
    )

    K_obs = K_obs[valid_obs]
    T_obs = T_obs[valid_obs]
    C_obs = C_obs[valid_obs]

    if K_obs.size == 0:
        raise ValueError("No valid observed quotes to plot.")

    if spot is None:
        spot = (
            res.get("S0", None)
            or getattr(day, "S0", None)
            or getattr(day, "spot", None)
        )

    spot = float(spot) if spot is not None else None

    if x_axis in {"lr", "r"} and spot is None:
        raise ValueError("spot/S0 is required to map observed strikes to lr/r space.")

    mask = _mask_axis(x, x_bounds)

    xp = x[mask]
    Cp = C[:, mask]

    if x_axis == "k":
        X_obs = K_obs
        spot_line_x = spot
    elif x_axis == "lr":
        X_obs = np.log(K_obs / spot)
        spot_line_x = 0.0
    else:
        X_obs = K_obs / spot
        spot_line_x = 1.0

    s = max(int(surface_stride), 1)

    X_mesh, T_mesh = np.meshgrid(xp, T)

    X_mesh = X_mesh[::s, ::s]
    T_mesh = T_mesh[::s, ::s]
    Cp = Cp[::s, ::s]

    created_fig = False

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure

    surf = ax.plot_surface(
        X_mesh,
        T_mesh,
        Cp,
        linewidth=0,
        antialiased=True,
        alpha=float(surface_alpha),
    )

    ax.scatter(
        X_obs,
        T_obs,
        C_obs,
        s=float(obs_size),
        alpha=float(obs_alpha),
        label="Observed",
    )

    if spot_line_x is not None and np.isfinite(spot_line_x):
        if xp.min() <= spot_line_x <= xp.max():
            ax.plot(
                [spot_line_x, spot_line_x],
                [float(np.min(T)), float(np.max(T))],
                [0.0, 0.0],
                linestyle="--",
                linewidth=2.0,
                label="Spot",
            )

    ttl = title if date_str is None else f"{title} — {date_str}"

    ax.set_title(ttl)
    ax.set_xlabel(_x_label(x_axis))
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Call price C")
    ax.view_init(elev=28, azim=-60)

    handles, labels = ax.get_legend_handles_labels()

    if handles and created_fig:
        ax.legend(loc="upper left")

    if created_fig:
        fig.tight_layout()

    if save is not None:
        sp = _normalize_save_path(save, default_name="call_surface_vs_observed", ext=".png")
        fig.savefig(sp, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {sp}")

    if show and created_fig:
        plt.show()
    elif created_fig and not show:
        plt.close(fig)

    return fig, ax, surf


# ============================================================
# Stored Delta Surfaces
# ============================================================

def plot_delta_surface(
    res: dict,
    *,
    which: Literal["call", "put", "skew"] = "call",
    title: str = "Surface in Delta Space",
    cmap: str = "viridis",
    interactive: bool = False,
    show: bool = True,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    elev: float = 25.0,
    azim: float = -60.0,
    alpha: float = 0.95,
    shade: bool = False,
):
    """
    Plot delta-based IV surfaces stored in res["delta_dict"].

    Expected keys in res["delta_dict"]:
      - delta_axis
      - T_axis
      - iv_delta_call
      - iv_delta_put_abs
      - delta_skew_surface
    """
    if "delta_dict" not in res or res["delta_dict"] is None:
        raise KeyError("Missing delta_dict. Expected res['delta_dict'].")

    d = res["delta_dict"]

    delta = np.asarray(d["delta_axis"], float).ravel()
    T = np.asarray(d["T_axis"], float).ravel()

    if which == "call":
        Z_key = "iv_delta_call"
        zlabel = "IV(call delta)"
    elif which == "put":
        Z_key = "iv_delta_put_abs"
        zlabel = "IV(|put delta|)"
    elif which == "skew":
        Z_key = "delta_skew_surface"
        zlabel = "Delta skew = (IV_call - IV_put) / IV_atm"
    else:
        raise ValueError("which must be one of {'call', 'put', 'skew'}.")

    if Z_key not in d:
        raise KeyError(f"delta_dict is missing '{Z_key}'.")

    Z = np.asarray(d[Z_key], float)

    if Z.shape != (T.size, delta.size):
        raise ValueError(f"{Z_key} has shape {Z.shape}, expected {(T.size, delta.size)}.")

    if interactive:
        return _surface_plotly(
            x=delta,
            T=T,
            Z=Z,
            title=title,
            xlabel="Delta",
            zlabel=zlabel,
            colorbar_title=zlabel,
            cmap=cmap,
            save=save,
            save_default_name=f"delta_surface_{which}",
            show=show,
        )

    return _surface_matplotlib(
        x=delta,
        T=T,
        Z=Z,
        title=title,
        xlabel="Delta",
        zlabel=zlabel,
        colorbar_label=zlabel,
        cmap=cmap,
        save=save,
        save_default_name=f"delta_surface_{which}",
        show=show,
        dpi=dpi,
        elev=elev,
        azim=azim,
        alpha=alpha,
        shade=shade,
    )