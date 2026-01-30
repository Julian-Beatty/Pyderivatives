from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

import numpy as np
import pandas as pd
import cvxpy as cp


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

    # plot compute flags: subset of {"surfaces","panels","perturb","term","heatmap"}
    plots_enabled: Tuple[str, ...] = ("surfaces", "panels", "perturb", "term", "heatmap")

    # plot defaults
    dpi: int = 160

    # heatmap defaults
    heatmap_eps: float = 1e-10
    heatmap_interpolation: str = "nearest"

    # term structure defaults
    n_term_structure_strikes: int = 6
    min_maturities_per_strike: int = 3


# -------------------------
# math helpers
# -------------------------

def discount_factor(r: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.exp(-r * T)


def forward_price(S0: np.ndarray, r: np.ndarray, q: float, T: np.ndarray) -> np.ndarray:
    return S0 * np.exp((r - q) * T)


def get_s0(df: pd.DataFrame, col_S0: str, col_K: str) -> float:
    if col_S0 in df.columns:
        s0 = np.nanmedian(pd.to_numeric(df[col_S0], errors="coerce").to_numpy(float))
        if np.isfinite(s0):
            return float(s0)
    k0 = np.nanmedian(pd.to_numeric(df[col_K], errors="coerce").to_numpy(float))
    return float(k0) if np.isfinite(k0) else np.nan


# -------------------------
# constraints
# -------------------------

def add_within_maturity_constraints(
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


def nearest_match(k1, i1, k2, i2, tol):
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


def bracket(sorted_k: np.ndarray, x: float):
    if x < sorted_k[0] or x > sorted_k[-1]:
        return None
    j = np.searchsorted(sorted_k, x)
    return j - 1, j


def add_cross_maturity_constraints(
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
            for i, j in nearest_match(d1["k"], d1["idx"], d2["k"], d2["idx"], k_match_tol):
                constraints.append(c_var[j] >= c_var[i])

        if "C5" in enabled and len(d1["k"]) >= 2:
            k1 = d1["k"]
            for t, kt in enumerate(d2["k"]):
                br = bracket(k1, kt)
                if br is None:
                    continue
                L, R = br
                w = (kt - k1[L]) / (k1[R] - k1[L])
                constraints.append(
                    c_var[d2["idx"][t]]
                    >= (1 - w) * c_var[d1["idx"][L]] + w * c_var[d1["idx"][R]]
                )


# -------------------------
# plot-data builder
# -------------------------

def build_plot_data(
    df_rep: pd.DataFrame,
    *,
    cfg: RepairConfig,
    perturb_mode: str = "absolute",
    perturb_eps: float = 1e-10,
) -> Dict[str, Any]:
    """
    Returns a pure dict holding everything plotters need.
    (Pickle-friendly; numpy arrays.)
    """
    T = df_rep[cfg.col_T].to_numpy(float)
    K = df_rep[cfg.col_K].to_numpy(float)
    C_obs = df_rep[cfg.col_C].to_numpy(float)
    C_rep = df_rep["C_rep"].to_numpy(float)

    s0 = get_s0(df_rep, cfg.col_S0, cfg.col_K)

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

    # perturb
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

    # term strikes (exact K repeated across maturities)
    counts = (
        df_rep.groupby(cfg.col_K)[cfg.col_T]
              .nunique()
              .sort_values(ascending=False)
    )
    counts = counts[counts >= cfg.min_maturities_per_strike]
    strikes = counts.index.tolist()[: cfg.n_term_structure_strikes]
    plot_data["term"] = {"strikes": [float(x) for x in strikes]}

    # heatmap grouped table
    tmp = df_rep[[cfg.col_T, cfg.col_K, cfg.col_C, "C_rep"]].copy()
    tmp[cfg.col_T] = pd.to_numeric(tmp[cfg.col_T], errors="coerce")
    tmp[cfg.col_K] = pd.to_numeric(tmp[cfg.col_K], errors="coerce")
    tmp[cfg.col_C] = pd.to_numeric(tmp[cfg.col_C], errors="coerce")
    tmp["C_rep"] = pd.to_numeric(tmp["C_rep"], errors="coerce")
    tmp = tmp.dropna(subset=[cfg.col_T, cfg.col_K, cfg.col_C, "C_rep"])
    tmp = tmp.groupby([cfg.col_K, cfg.col_T], as_index=False).agg({cfg.col_C: "mean", "C_rep": "mean"})
    plot_data["heatmap"] = {"table": tmp, "K_col": cfg.col_K, "T_col": cfg.col_T, "C_col": cfg.col_C}

    return plot_data


# -------------------------
# core class
# -------------------------

class CallSurfaceArbRepair:
    """
    rep = CallSurfaceArbRepair(cfg)
    out = rep.repair_one_date(df_date)

    out keys:
      - df_rep: df + C_rep
      - plot_data: dict for plotting
      - solve_info: dict
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
        cfg = self.cfg
        enabled = set(cfg.enabled_constraints)

        df = df_date.copy().reset_index(drop=True)

        T = df[cfg.col_T].to_numpy(float)
        K = df[cfg.col_K].to_numpy(float)
        C = df[cfg.col_C].to_numpy(float)
        S0 = df[cfg.col_S0].to_numpy(float)
        r = df[cfg.col_r].to_numpy(float)

        D = discount_factor(r, T)
        F = forward_price(S0, r, cfg.assume_dividend_yield_q, T)

        c_norm = C / (D * F)
        k_norm = K / F

        n = len(df)
        c_var = cp.Variable(n)

        constraints = []
        objective = cp.Minimize(cp.norm1(c_var - c_norm))

        by_T: Dict[float, Dict[str, np.ndarray]] = {}
        for Ti in np.unique(T):
            idx = np.where(T == Ti)[0]

            ordK = np.argsort(K[idx])
            idxK = idx[ordK]
            add_within_maturity_constraints(
                idxs_sorted_by_K=idxK,
                K_sorted=K[idxK],
                c_var=c_var,
                constraints=constraints,
                enabled=enabled,
            )

            ordk = np.argsort(k_norm[idx])
            by_T[Ti] = {"idx": idx[ordk], "k": k_norm[idx][ordk]}

        add_cross_maturity_constraints(
            maturities=np.unique(T),
            by_T=by_T,
            c_var=c_var,
            constraints=constraints,
            enabled=enabled,
            k_match_tol=cfg.k_match_tol,
            adjacent_only=cfg.enforce_calendar_adjacent_only,
        )

        constraints += [c_var >= 0, c_var <= 1]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cfg.solver, verbose=cfg.verbose)

        df["C_rep"] = np.asarray(c_var.value).reshape(-1) * D * F

        plot_data = build_plot_data(df, cfg=cfg, perturb_mode=perturb_mode, perturb_eps=perturb_eps)

        enabled_plots = set(cfg.plots_enabled)
        if "perturb" not in enabled_plots:
            plot_data.pop("perturb", None)
        if "term" not in enabled_plots:
            plot_data.pop("term", None)
        if "heatmap" not in enabled_plots:
            plot_data.pop("heatmap", None)

        solve_info = {
            "status": str(prob.status),
            "objective_value": float(prob.value) if prob.value is not None else np.nan,
            "solver": cfg.solver,
            "enabled_constraints": tuple(cfg.enabled_constraints),
        }

        return {"df_rep": df, "plot_data": plot_data, "solve_info": solve_info}
