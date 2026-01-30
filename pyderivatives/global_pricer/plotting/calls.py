from __future__ import annotations

from typing import Iterable, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from ..data import CallSurfaceDay


def _choose_maturities(T_obs: np.ndarray, *, maturities: Optional[Iterable[float]], n: int) -> np.ndarray:
    T_unique = np.unique(np.asarray(T_obs, float))
    T_unique.sort()
    if maturities is not None:
        T_sel = np.array(list(maturities), float).ravel()
        # map requested maturities to nearest available
        out = []
        for t in T_sel:
            j = int(np.argmin(np.abs(T_unique - t)))
            out.append(T_unique[j])
        return np.unique(np.array(out, float))
    # pick up to n evenly spaced unique maturities
    if T_unique.size <= n:
        return T_unique
    idx = np.linspace(0, T_unique.size - 1, n).round().astype(int)
    return T_unique[idx]


def plot_observed_vs_fitted_curves(
    day: CallSurfaceDay,
    res: dict,
    *,
    maturities: Optional[Iterable[float]] = None,
    n: int = 6,
    title: str = "Observed quotes vs fitted curve",
    sort_by_strike: bool = True,
    figsize=(10, 6),
):
    """
    Overlays observed quotes and fitted curve at selected maturities.

    Requires:
      - day.K_obs, day.T_obs, day.C_obs
      - res["K_grid"], res["T_grid"], res["C_fit"] (shape: (len(T_grid), len(K_grid)))
    """
    K_grid = np.asarray(res["K_grid"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()
    C_fit = np.asarray(res["C_fit"], float)

    if C_fit.shape != (T_grid.size, K_grid.size):
        raise ValueError("res['C_fit'] must have shape (len(T_grid), len(K_grid)).")

    T_sel = _choose_maturities(day.T_obs, maturities=maturities, n=int(n))

    plt.figure(figsize=figsize)
    for t in T_sel:
        # observed points at this maturity (exact match)
        m = np.isclose(day.T_obs, t, rtol=0.0, atol=1e-12)
        if not np.any(m):
            continue

        Ko = day.K_obs[m]
        Co = day.C_obs[m]
        if sort_by_strike:
            order = np.argsort(Ko)
            Ko, Co = Ko[order], Co[order]

        # fitted curve: use nearest T in T_grid
        j = int(np.argmin(np.abs(T_grid - t)))
        plt.plot(K_grid, C_fit[j, :], label=f"fit T={T_grid[j]:.4f}")
        plt.scatter(Ko, Co, s=18, alpha=0.9, label=f"obs T={t:.4f}")

    plt.title(title)
    plt.xlabel("Strike K")
    plt.ylabel("Call price C")
    plt.legend(ncols=2, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_observed_trisurf(
    day: CallSurfaceDay,
    *,
    title: str = "Observed call surface (sparse quotes)",
    figsize=(10, 7),
):
    """
    Trisurf of observed quotes in (K, T, C). Handles sparse surfaces.
    """
    x = np.asarray(day.K_obs, float).ravel()
    y = np.asarray(day.T_obs, float).ravel()
    z = np.asarray(day.C_obs, float).ravel()

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]

    # Triangulation expects x,y length == number of points
    tri = Triangulation(x, y)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(tri, z, linewidth=0.2, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Call price C")
    plt.tight_layout()
    plt.show()


def plot_fitted_surface(
    res: dict,
    *,
    title: str = "Fitted call surface (rectangular grid)",
    figsize=(10, 7),
):
    """
    Regular grid surface plot of fitted calls.
    """
    K_grid = np.asarray(res["K_grid"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()
    C_fit = np.asarray(res["C_fit"], float)

    if C_fit.shape != (T_grid.size, K_grid.size):
        raise ValueError("res['C_fit'] must have shape (len(T_grid), len(K_grid)).")

    KK, TT = np.meshgrid(K_grid, T_grid)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(KK, TT, C_fit, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Call price C")
    plt.tight_layout()
    plt.show()
