from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SavePath = Optional[Union[str, Path]]


def _save_or_return(fig: plt.Figure, *, save: SavePath, dpi: int):
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")
        plt.close(fig)
        return None
    return fig


def plot_surface(plot_data: Dict[str, Any], *, title: str = "Observed vs Repaired", save: SavePath = None, dpi: int = 160):
    T = np.asarray(plot_data["raw"]["T"], float)
    K = np.asarray(plot_data["raw"]["K"], float)
    C_obs = np.asarray(plot_data["raw"]["C_obs"], float)
    C_rep = np.asarray(plot_data["raw"]["C_rep"], float)

    tri = Triangulation(T, K)

    fig = plt.figure(figsize=(14, 6))
    for i, Z in enumerate([C_obs, C_rep]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_trisurf(tri, Z)
        ax.set_title("Observed" if i == 0 else "Repaired")
        ax.view_init(elev=20, azim=35)
        ax.set_xlabel("T (years)")
        ax.set_ylabel("K")
        ax.set_zlabel("C")

    fig.suptitle(title)
    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


def plot_panels(plot_data: Dict[str, Any], *, title: str = "Panels", save: SavePath = None, dpi: int = 160, n_panels: int = 6):
    T = np.asarray(plot_data["raw"]["T"], float)
    K = np.asarray(plot_data["raw"]["K"], float)
    C_obs = np.asarray(plot_data["raw"]["C_obs"], float)
    C_rep = np.asarray(plot_data["raw"]["C_rep"], float)

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

        ax[i, 0].plot(K[m][o], C_obs[m][o])
        ax[i, 0].set_title(f"T={Ti:.3f} Obs")
        ax[i, 0].set_xlabel("K")
        ax[i, 0].set_ylabel("C")

        ax[i, 1].plot(K[m][o], C_rep[m][o])
        ax[i, 1].set_title(f"T={Ti:.3f} Rep")
        ax[i, 1].set_xlabel("K")
        ax[i, 1].set_ylabel("C")

        if has_s0 and np.isfinite(s0):
            for j in (0, 1):
                ax[i, j].axvline(s0, ls="--", lw=1)

    fig.suptitle(title)
    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


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
        ax.set_ylabel("C")
        ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


def plot_heatmap(plot_data: Dict[str, Any], *, title: str = "C_rep / C_obs", save: SavePath = None, dpi: int = 160, eps: float = 1e-10, interpolation: str = "nearest"):
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

    tmp["ratio"] = tmp["C_rep"].to_numpy(float) / np.maximum(tmp[C_col].to_numpy(float), eps)

    K_col = h.get("K_col", tmp.columns[0])
    T_col = h.get("T_col", tmp.columns[1])

    pivot = tmp.pivot(index=K_col, columns=T_col, values="ratio").sort_index()
    K_vals = pivot.index.to_numpy(float)
    T_vals = pivot.columns.to_numpy(float)
    Z = np.ma.masked_invalid(pivot.to_numpy(float))

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [np.min(T_vals), np.max(T_vals), np.min(K_vals), np.max(K_vals)]
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent, interpolation=interpolation)
    fig.colorbar(im, ax=ax, label="C_rep / C_obs")

    ax.set_title(title)
    ax.set_xlabel("T")
    ax.set_ylabel("K")

    has_s0 = bool(plot_data["meta"]["has_s0"])
    s0 = float(plot_data["meta"]["s0"])
    if has_s0 and np.isfinite(s0):
        ax.axhline(s0, ls="--", lw=1)
        ax.text(np.min(T_vals), s0, f" S0={s0:.2f}", va="bottom", ha="left", fontsize=10)

    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)
