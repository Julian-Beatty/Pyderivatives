"""
arb_repair_io.py

IO-friendly call-surface arbitrage repair + plot_data-based plotting.

Key features
------------
- RepairConfig includes:
    - enabled_constraints
    - plots_enabled
- CallSurfaceArbRepair.repair_one_date(df_date) returns dict:
    {
      "df_rep": repaired dataframe with column "C_rep",
      "plot_data": pure dict containing arrays + precomputed plot inputs,
      "solve_info": solver status/objective/etc
    }

- Plot functions take ONLY plot_data plus:
    title=...
    save=...   (arbitrary path; creates parent dirs; saves; prints; closes)
    dpi=...

Plot functions (concise names)
------------------------------
- plot_surface(plot_data, ...)
- plot_panels(plot_data, ...)
- plot_perturb(plot_data, ...)
- plot_term(plot_data, ...)
- plot_heatmap(plot_data, ...)

Notes
-----
- plot_data is pickle-friendly (numpy arrays), not JSON-friendly by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, Iterable, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import cvxpy as cp


# ============================================================
# Config
# ============================================================

@dataclass
class RepairConfig:
    # required column names
    col_date: str = "date"
    col_T: str = "rounded_maturity"
    col_S0: str = "stock_price"
    col_r: str = "risk_free_rate"
    col_K: str = "strike"
    col_C: str = "mid_price"

    # finance assumption
    assume_dividend_yield_q: float = 0.0

    # solver
    solver: str = "ECOS"
    verbose: bool = False

    # cross-maturity matching tolerance in normalized strike k=K/F
    k_match_tol: float = 2e-3
    enforce_calendar_adjacent_only: bool = True

    # constraints enabled by default
    enabled_constraints: Tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5")

    # plot compute flags: choose any subset of {"surfaces","panels","perturb","term","heatmap"}
    plots_enabled: Tuple[str, ...] = ("surfaces", "panels", "perturb", "term", "heatmap")

    # plot defaults
    dpi: int = 160

    # heatmap defaults
    heatmap_eps: float = 1e-10
    heatmap_interpolation: str = "nearest"

    # term structure defaults
    n_term_structure_strikes: int = 6
    min_maturities_per_strike: int = 3


# ============================================================
# Utilities
# ============================================================

SavePath = Optional[Union[str, Path]]


def _save_or_return(fig: plt.Figure, *, save: SavePath, dpi: int) -> Optional[plt.Figure]:
    """
    If save is provided, save to that path, mkdir(parents=True), print, close, return None.
    Else return the figure (notebook-friendly).
    """
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")
        plt.close(fig)
        return None
    return fig


def _discount_factor(r: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.exp(-r * T)


def _forward_price(S0: np.ndarray, r: np.ndarray, q: float, T: np.ndarray) -> np.ndarray:
    return S0 * np.exp((r - q) * T)


def _get_s0(df: pd.DataFrame, col_S0: str, col_K: str) -> float:
    if col_S0 in df.columns:
        s0 = np.nanmedian(pd.to_numeric(df[col_S0], errors="coerce").to_numpy(float))
        if np.isfinite(s0):
            return float(s0)
    k0 = np.nanmedian(pd.to_numeric(df[col_K], errors="coerce").to_numpy(float))
    return float(k0) if np.isfinite(k0) else np.nan


# ============================================================
# Constraints
# ============================================================

def _add_within_maturity_constraints(
    *,
    idxs_sorted_by_K: np.ndarray,
    K_sorted: np.ndarray,
    c_var: cp.Variable,
    constraints: list,
    enabled: set,
) -> None:
    """
    Within a fixed maturity:
      C1: nonnegative
      C2: decreasing in strike
      C3: convex in strike (discrete inequality)
    """
    n = len(idxs_sorted_by_K)

    if "C1" in enabled:
        constraints.append(c_var[idxs_sorted_by_K] >= 0)

    if "C2" in enabled and n >= 2:
        for j in range(1, n):
            constraints.append(c_var[idxs_sorted_by_K[j - 1]] >= c_var[idxs_sorted_by_K[j]])

    if "C3" in enabled and n >= 3:
        for j in range(1, n - 1):
            Km1, K0, Kp1 = K_sorted[j - 1], K_sorted[j], K_sorted[j + 1]
            constraints.append(
                (Kp1 - K0) * (c_var[idxs_sorted_by_K[j]] - c_var[idxs_sorted_by_K[j - 1]])
                <=
                (K0 - Km1) * (c_var[idxs_sorted_by_K[j + 1]] - c_var[idxs_sorted_by_K[j]])
            )


def _nearest_match(k1, i1, k2, i2, tol):
    out = []
    a = b = 0
    while a < len(k1) and b < len(k2):
        if abs(k1[a] - k2[b]) <= tol:
            out.append((i1[a], i2[b]))
            a += 1
            b += 1
        elif k1[a] < k2[b]:
            a += 1
        else:
            b += 1
    return out


def _bracket(sorted_k: np.ndarray, x: float):
    if x < sorted_k[0] or x > sorted_k[-1]:
        return None
    j = np.searchsorted(sorted_k, x)
    return j - 1, j


def _add_cross_maturity_constraints(
    *,
    maturities: np.ndarray,
    by_T: Dict[float, Dict[str, np.ndarray]],
    c_var: cp.Variable,
    constraints: list,
    enabled: set,
    k_match_tol: float,
    adjacent_only: bool,
) -> None:
    """
    Cross maturity constraints in normalized strike k=K/F:
      C4: matched k calendar monotonicity
      C5: interpolated k calendar monotonicity
    """
    Ts = sorted(np.unique(maturities).tolist())
    if adjacent_only:
        pairs = list(zip(Ts[:-1], Ts[1:]))
    else:
        pairs = [(Ts[i], Ts[j]) for i in range(len(Ts)) for j in range(i + 1, len(Ts))]

    for T1, T2 in pairs:
        d1, d2 = by_T[T1], by_T[T2]

        if "C4" in enabled:
            for i, j in _nearest_match(d1["k"], d1["idx"], d2["k"], d2["idx"], k_match_tol):
                constraints.append(c_var[j] >= c_var[i])

        if "C5" in enabled and len(d1["k"]) >= 2:
            k1 = d1["k"]
            for t, kt in enumerate(d2["k"]):
                br = _bracket(k1, kt)
                if br is None:
                    continue
                L, R = br
                w = (kt - k1[L]) / (k1[R] - k1[L])
                constraints.append(
                    c_var[d2["idx"][t]]
                    >= (1 - w) * c_var[d1["idx"][L]] + w * c_var[d1["idx"][R]]
                )


# ============================================================
# Plot-data builder
# ============================================================

def build_plot_data(
    df_rep: pd.DataFrame,
    *,
    cfg: RepairConfig,
    perturb_mode: str = "absolute",
    perturb_eps: float = 1e-10,
) -> Dict[str, Any]:
    """
    Build a pure plot-data dict (numpy arrays; pickle-friendly).
    Plotters consume this dict only.
    """
    T = df_rep[cfg.col_T].to_numpy(float)
    K = df_rep[cfg.col_K].to_numpy(float)
    C_obs = df_rep[cfg.col_C].to_numpy(float)
    C_rep = df_rep["C_rep"].to_numpy(float)

    s0 = _get_s0(df_rep, cfg.col_S0, cfg.col_K)

    plot_data: Dict[str, Any] = {
        "meta": {
            "col_T": cfg.col_T,
            "col_K": cfg.col_K,
            "col_C": cfg.col_C,
            "has_s0": bool(np.isfinite(s0)),
            "s0": float(s0) if np.isfinite(s0) else np.nan,
            "perturb_mode": perturb_mode,
        },
        "raw": {
            "T": T,
            "K": K,
            "C_obs": C_obs,
            "C_rep": C_rep,
        },
    }

    # Perturb precompute (so plotter stays simple)
    if perturb_mode == "absolute":
        y = C_rep - C_obs
        ylabel = "C_rep − C_obs"
    elif perturb_mode == "pct_error":
        denom = np.maximum(np.abs(C_obs), perturb_eps)
        y = (C_rep / denom - 1.0) * 100.0
        ylabel = "Percentage Error: (C_rep / C_obs − 1) × 100"
    else:
        raise ValueError("perturb_mode must be 'absolute' or 'pct_error'")
    plot_data["perturb"] = {"y": y, "ylabel": ylabel}

    # Term strikes: choose strikes that appear across multiple maturities
    counts = (
        df_rep.groupby(cfg.col_K)[cfg.col_T]
              .nunique()
              .sort_values(ascending=False)
    )
    counts = counts[counts >= cfg.min_maturities_per_strike]
    strikes = counts.index.tolist()[: cfg.n_term_structure_strikes]
    plot_data["term"] = {"strikes": [float(x) for x in strikes]}

    # Heatmap table: grouped [K,T] with mean obs and mean rep
    tmp = df_rep[[cfg.col_T, cfg.col_K, cfg.col_C, "C_rep"]].copy()
    tmp[cfg.col_T] = pd.to_numeric(tmp[cfg.col_T], errors="coerce")
    tmp[cfg.col_K] = pd.to_numeric(tmp[cfg.col_K], errors="coerce")
    tmp[cfg.col_C] = pd.to_numeric(tmp[cfg.col_C], errors="coerce")
    tmp["C_rep"] = pd.to_numeric(tmp["C_rep"], errors="coerce")
    tmp = tmp.dropna(subset=[cfg.col_T, cfg.col_K, cfg.col_C, "C_rep"])
    tmp = tmp.groupby([cfg.col_K, cfg.col_T], as_index=False).agg({cfg.col_C: "mean", "C_rep": "mean"})
    plot_data["heatmap"] = {"table": tmp, "K_col": cfg.col_K, "T_col": cfg.col_T, "C_col": cfg.col_C}

    return plot_data


# ============================================================
# Plotters (concise names; arbitrary save paths)
# ============================================================

def plot_surface(
    plot_data: Dict[str, Any],
    *,
    title: str = "Observed vs Repaired",
    save: SavePath = None,
    dpi: int = 160,
) -> Optional[plt.Figure]:
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
        ax.set_ylabel("Strike K")
        ax.set_zlabel("Call price")

    fig.suptitle(title)
    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


def plot_panels(
    plot_data: Dict[str, Any],
    *,
    title: str = "Panels",
    save: SavePath = None,
    dpi: int = 160,
    n_panels: int = 6,
) -> Optional[plt.Figure]:
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


def plot_perturb(
    plot_data: Dict[str, Any],
    *,
    title: str = "Perturbation",
    save: SavePath = None,
    dpi: int = 160,
) -> Optional[plt.Figure]:
    T = np.asarray(plot_data["raw"]["T"], float)
    K = np.asarray(plot_data["raw"]["K"], float)
    y = np.asarray(plot_data["perturb"]["y"], float)
    ylabel = str(plot_data["perturb"]["ylabel"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(K, y, c=T, cmap="viridis")
    ax.axhline(0, lw=1)
    fig.colorbar(sc, ax=ax, label="T (years)")

    ax.set_title(title)
    ax.set_xlabel("Strike K")
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


def plot_term(
    plot_data: Dict[str, Any],
    *,
    title: str = "Exact-K Term Structures",
    save: SavePath = None,
    dpi: int = 160,
    ncols: int = 3,
) -> Optional[plt.Figure]:
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

        # aggregate by T (means)
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
        ax.set_xlabel("T (years)")
        ax.set_ylabel("C")
        ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


def plot_heatmap(
    plot_data: Dict[str, Any],
    *,
    title: str = "C_rep / C_obs",
    save: SavePath = None,
    dpi: int = 160,
    eps: float = 1e-10,
    interpolation: str = "nearest",
) -> Optional[plt.Figure]:
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
        # fallback: assume 3rd column is observed
        C_col = tmp.columns[2]

    tmp["ratio"] = tmp["C_rep"].to_numpy(float) / np.maximum(tmp[C_col].to_numpy(float), eps)

    # pivot with inferred K/T cols (first two columns in our builder)
    K_col = h.get("K_col", tmp.columns[0])
    T_col = h.get("T_col", tmp.columns[1])

    pivot = tmp.pivot(index=K_col, columns=T_col, values="ratio").sort_index()
    K_vals = pivot.index.to_numpy(float)
    T_vals = pivot.columns.to_numpy(float)

    Z = np.ma.masked_invalid(pivot.to_numpy(float))

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [np.min(T_vals), np.max(T_vals), np.min(K_vals), np.max(K_vals)]
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation=interpolation,
    )
    fig.colorbar(im, ax=ax, label="C_rep / C_obs")

    ax.set_title(title)
    ax.set_xlabel("T (years)")
    ax.set_ylabel("K")

    has_s0 = bool(plot_data["meta"]["has_s0"])
    s0 = float(plot_data["meta"]["s0"])
    if has_s0 and np.isfinite(s0):
        ax.axhline(s0, ls="--", lw=1)
        ax.text(np.min(T_vals), s0, f" S0={s0:.2f}", va="bottom", ha="left", fontsize=10)

    fig.tight_layout()
    return _save_or_return(fig, save=save, dpi=dpi)


# ============================================================
# Main class
# ============================================================

class CallSurfaceArbRepair:
    """
    IO-friendly workflow:

      rep = CallSurfaceArbRepair(cfg)
      out = rep.repair_one_date(df_date)

      df_rep = out["df_rep"]
      plot_data = out["plot_data"]

      plot_surface(plot_data, save=".../surface.png")
      plot_panels(plot_data, title="...", save=".../panels.png")
    """

    def __init__(self, cfg: RepairConfig):
        self.cfg = cfg

    def repair_one_date(
        self,
        df_date: pd.DataFrame,
        *,
        perturb_mode: str = "absolute",
        perturb_eps: float = 1e-10,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "df_rep": df with added "C_rep",
            "plot_data": plot_data dict (pruned per cfg.plots_enabled),
            "solve_info": dict
          }
        """
        cfg = self.cfg
        enabled = set(cfg.enabled_constraints)

        df = df_date.copy().reset_index(drop=True)

        # numeric arrays
        T = df[cfg.col_T].to_numpy(float)
        K = df[cfg.col_K].to_numpy(float)
        C = df[cfg.col_C].to_numpy(float)
        S0 = df[cfg.col_S0].to_numpy(float)
        r = df[cfg.col_r].to_numpy(float)

        # normalize
        D = _discount_factor(r, T)
        F = _forward_price(S0, r, cfg.assume_dividend_yield_q, T)
        c_norm = C / (D * F)
        k_norm = K / F

        n = len(df)
        c_var = cp.Variable(n)

        constraints = []
        objective = cp.Minimize(cp.norm1(c_var - c_norm))

        # within maturity + collect normalized k grids
        by_T: Dict[float, Dict[str, np.ndarray]] = {}
        for Ti in np.unique(T):
            idx = np.where(T == Ti)[0]

            # within maturity constraints use raw strike ordering
            ordK = np.argsort(K[idx])
            idxK = idx[ordK]
            _add_within_maturity_constraints(
                idxs_sorted_by_K=idxK,
                K_sorted=K[idxK],
                c_var=c_var,
                constraints=constraints,
                enabled=enabled,
            )

            # cross maturity uses normalized strike ordering
            ordk = np.argsort(k_norm[idx])
            by_T[Ti] = {"idx": idx[ordk], "k": k_norm[idx][ordk]}

        _add_cross_maturity_constraints(
            maturities=np.unique(T),
            by_T=by_T,
            c_var=c_var,
            constraints=constraints,
            enabled=enabled,
            k_match_tol=cfg.k_match_tol,
            adjacent_only=cfg.enforce_calendar_adjacent_only,
        )

        # generic bounds (kept from your original)
        constraints += [c_var >= 0, c_var <= 1]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cfg.solver, verbose=cfg.verbose)

        df["C_rep"] = np.asarray(c_var.value).reshape(-1) * D * F

        # Build plot_data, then prune based on cfg.plots_enabled
        plot_data = build_plot_data(df, cfg=cfg, perturb_mode=perturb_mode, perturb_eps=perturb_eps)

        enabled_plots = set(cfg.plots_enabled)
        if "perturb" not in enabled_plots:
            plot_data.pop("perturb", None)
        if "term" not in enabled_plots:
            plot_data.pop("term", None)
        if "heatmap" not in enabled_plots:
            plot_data.pop("heatmap", None)
        # surfaces/panels rely on raw/meta only

        solve_info = {
            "status": str(prob.status),
            "objective_value": float(prob.value) if prob.value is not None else np.nan,
            "solver": cfg.solver,
            "enabled_constraints": tuple(cfg.enabled_constraints),
        }

        return {"df_rep": df, "plot_data": plot_data, "solve_info": solve_info}


# ============================================================
# Minimal example usage (comment out in production)
# ============================================================

if __name__ == "__main__":
    # You would replace this with your real df_date slice.
    # df_date must have columns:
    #   date, rounded_maturity, stock_price, risk_free_rate, strike, mid_price
    #
    # Example (placeholder):
    # df_all = pd.read_parquet("calls.parquet")
    # df_date = df_all[df_all["date"] == "2022-08-30"].copy()
    #
    # out = CallSurfaceArbRepair(RepairConfig()).repair_one_date(df_date)
    # pdict = out["plot_data"]
    # plot_surface(pdict, save="out/surface.png")
    #
    pass
