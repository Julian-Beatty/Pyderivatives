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

    repair_mode: str = "joint"  # joint, calls, puts (European, fixed carry)
    col_right: str = "option_right"
    objective: str = "normalized_l1"  # normalized_l1, bid_ask, hybrid
    col_bid: str = "best_bid"
    col_ask: str = "best_offer"


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
    j = int(np.clip(np.searchsorted(sorted_k, x), 1, len(sorted_k) - 1))
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

    # Preserve original quote spreads, in the actual option right's price units.
    for key, column in (("bid", cfg.col_bid), ("ask", cfg.col_ask)):
        if column in df_rep:
            plot_data["raw"][key] = pd.to_numeric(df_rep[column], errors="coerce").to_numpy(float)

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
      - df_rep: original observations + C_rep, P_rep, price_rep, repair_adjustment
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
        repair_mode: Optional[str] = None,
        objective: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        enabled = set(cfg.enabled_constraints)
        objective_name = cfg.objective if objective is None else objective
        if objective_name not in {"normalized_l1", "bid_ask", "hybrid"}:
            raise ValueError("objective must be 'normalized_l1', 'bid_ask', or 'hybrid'.")

        df = df_date.copy().reset_index(drop=True)

        mode = cfg.repair_mode if repair_mode is None else repair_mode
        if mode not in {"joint", "calls", "puts"}:
            raise ValueError("repair_mode must be 'joint', 'calls', or 'puts'.")
        if cfg.col_right in df:
            rights = df[cfg.col_right].astype(str).str.strip().str.lower().map(
                {"c": "c", "call": "c", "p": "p", "put": "p"}
            )
            if rights.isna().any():
                raise ValueError("Unknown option right; expected c/call or p/put.")
        else:
            rights = pd.Series("c", index=df.index)
        if mode != "joint":
            keep = rights.eq("c" if mode == "calls" else "p")
            df, rights = df.loc[keep].reset_index(drop=True), rights.loc[keep].reset_index(drop=True)
        if df.empty:
            raise ValueError("No option quotes available for the selected repair mode.")
        for identity in (cfg.col_date, "underlying", "underlying_id"):
            if identity in df and df[identity].nunique(dropna=False) != 1:
                raise ValueError(f"Repair requires one snapshot/underlying: mixed {identity}.")

        T = df[cfg.col_T].to_numpy(float)
        K = df[cfg.col_K].to_numpy(float)
        C = df[cfg.col_C].to_numpy(float)
        S0 = df[cfg.col_S0].to_numpy(float)
        r = df[cfg.col_r].to_numpy(float)

        if not all(np.isfinite(v).all() for v in (T, K, C, S0, r)):
            raise ValueError("Repair inputs must be finite.")
        if np.any(T <= 0) or np.any(K <= 0) or np.any(S0 <= 0) or np.any(C < 0):
            raise ValueError("Require positive maturity, strike and spot, and nonnegative prices.")
        if not np.isfinite(cfg.assume_dividend_yield_q):
            raise ValueError("Dividend yield must be finite.")
        if not np.allclose(S0, S0[0], rtol=1e-10, atol=1e-10):
            raise ValueError("All quotes must use the same snapshot spot price.")
        for ti in np.unique(T):
            if not np.allclose(r[T == ti], r[T == ti][0], rtol=1e-10, atol=1e-10):
                raise ValueError("Quotes at the same maturity must use the same rate.")

        D = discount_factor(r, T)
        F = forward_price(S0, r, cfg.assume_dividend_yield_q, T)
        if not np.isfinite(D * F).all() or not np.isfinite(F).all() or np.any(D * F <= 0) or np.any(F <= 0):
            raise ValueError("Carry inputs produce invalid discount/forward scales.")

        is_put = rights.to_numpy() == "p"
        parity = D * (F - K)
        c_norm = (C + np.where(is_put, parity, 0.0)) / (D * F)
        k_norm = K / F

        # Keep all observations in the loss, sharing one call variable per (T,K).
        nodes, first, inverse = np.unique(
            np.column_stack([T, K]), axis=0, return_index=True, return_inverse=True
        )
        node_T, node_K, node_k = T[first], K[first], k_norm[first]
        n = len(nodes)
        c_var = cp.Variable(n)

        constraints = []
        residual = c_var[inverse] - c_norm
        delta0 = None
        locked = np.zeros(len(df), dtype=bool)
        spread_rows = np.zeros(len(df), dtype=bool)
        if objective_name in {"bid_ask", "hybrid"}:
            missing = {cfg.col_bid, cfg.col_ask} - set(df.columns)
            if missing:
                raise ValueError(f"{objective_name} objective requires columns: {sorted(missing)}")
            bid = df[cfg.col_bid].to_numpy(float)
            ask = df[cfg.col_ask].to_numpy(float)
            locked = (bid == ask) & (C == bid)
            spread_rows = (bid < C) & (C < ask)
            allowed = spread_rows | (locked if objective_name == "hybrid" else False)
            valid = np.isfinite(bid) & np.isfinite(ask) & (bid >= 0) & allowed
            if not valid.all():
                examples = np.flatnonzero(~valid)[:10].tolist()
                raise ValueError(
                    f"{objective_name} requires finite quotes with 0 <= bid < reference price < ask"
                    + (" or bid == reference price == ask for locked fallback" if objective_name == "hybrid" else "")
                    + f"; invalid row positions: {examples}. Crossed, missing, and other boundary-reference quotes are not supported."
                )
            # Parity translation cancels from each reference-to-bound distance.
            # Cohen et al. section 3.2, equations (8)-(10), normalized units.
            total_cost = cp.Constant(0.0)
            if spread_rows.any():
                scale = (D * F)[spread_rows]
                db = (C[spread_rows] - bid[spread_rows]) / scale
                da = (ask[spread_rows] - C[spread_rows]) / scale
                # Only positive-spread rows enter the minima; N counts all quotes.
                delta0 = float(min(1.0 / len(df), np.min(db), np.min(da)))
                spread_residual = residual[np.flatnonzero(spread_rows)]
                costs = cp.Variable(int(spread_rows.sum()), nonneg=True)
                constraints.extend([
                    costs >= -spread_residual - db + delta0,
                    costs >= cp.multiply(-delta0 / db, spread_residual),
                    costs >= cp.multiply(delta0 / da, spread_residual),
                    costs >= spread_residual - da + delta0,
                ])
                total_cost = total_cost + cp.sum(costs)
            if locked.any():
                total_cost = total_cost + cp.norm1(residual[np.flatnonzero(locked)])
            loss = cp.Minimize(total_cost)
        else:
            loss = cp.Minimize(cp.norm1(residual))

        by_T: Dict[float, Dict[str, np.ndarray]] = {}
        for Ti in np.unique(node_T):
            idx = np.where(node_T == Ti)[0]

            ordK = np.argsort(node_K[idx])
            idxK = idx[ordK]
            add_within_maturity_constraints(
                idxs_sorted_by_K=idxK,
                K_sorted=node_K[idxK],
                c_var=c_var,
                constraints=constraints,
                enabled=enabled,
            )

            # Vertical-spread bound makes parity-derived puts increasing as well.
            if len(idxK) > 1:
                constraints.append(c_var[idxK[:-1]] - c_var[idxK[1:]] <= np.diff(node_k[idxK]))
            ordk = np.argsort(node_k[idx])
            by_T[Ti] = {"idx": idx[ordk], "k": node_k[idx][ordk]}

        add_cross_maturity_constraints(
            maturities=np.unique(T),
            by_T=by_T,
            c_var=c_var,
            constraints=constraints,
            enabled=enabled,
            k_match_tol=cfg.k_match_tol,
            adjacent_only=cfg.enforce_calendar_adjacent_only,
        )

        constraints += [c_var >= np.maximum(1.0 - node_k, 0.0), c_var <= 1]

        prob = cp.Problem(loss, constraints)
        prob.solve(solver=cfg.solver, verbose=cfg.verbose)

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or c_var.value is None:
            raise RuntimeError(f"Arbitrage repair failed: {prob.status}")
        df["C_rep"] = np.asarray(c_var.value).reshape(-1)[inverse] * D * F
        df["P_rep"] = df["C_rep"].to_numpy() - parity
        df["price_rep"] = np.where(is_put, df["P_rep"], df["C_rep"])
        df["repair_adjustment"] = df["price_rep"] - C
        df["repair_penalty"] = np.where(spread_rows, "bid_ask", "normalized_l1")
        df["locked_quote_fallback"] = locked if objective_name == "hybrid" else False
        if objective_name in {"bid_ask", "hybrid"}:
            df["spread_exceedance"] = np.maximum.reduce([
                bid - df["price_rep"].to_numpy(),
                df["price_rep"].to_numpy() - ask,
                np.zeros(len(df)),
            ])
        df[cfg.col_right] = rights.to_numpy()

        by_right = {}
        for right in ("c", "p"):
            side = df.loc[df[cfg.col_right].eq(right)].copy()
            if side.empty:
                continue
            side["C_rep"] = side["price_rep"]  # legacy plot payload price key
            payload = build_plot_data(side, cfg=cfg, perturb_mode=perturb_mode, perturb_eps=perturb_eps)
            payload["meta"]["option_right"] = right
            if right == "p":
                payload["perturb"]["ylabel"] = payload["perturb"]["ylabel"].replace("C_", "P_")
            by_right[right] = payload
        plot_data = dict(by_right.get("c", by_right.get("p")))
        plot_data["by_right"] = by_right

        enabled_plots = set(cfg.plots_enabled)
        if "perturb" not in enabled_plots:
            plot_data.pop("perturb", None)
        if "term" not in enabled_plots:
            plot_data.pop("term", None)
        if "heatmap" not in enabled_plots:
            plot_data.pop("heatmap", None)
        for payload in by_right.values():
            for key in ("perturb", "term", "heatmap"):
                if key not in enabled_plots:
                    payload.pop(key, None)

        solve_info = {
            "status": str(prob.status),
            "objective_value": float(prob.value) if prob.value is not None else np.nan,
            "solver": cfg.solver,
            "repair_mode": mode,
            "n_quotes": len(df),
            "n_nodes": n,
            "objective": objective_name,
            "bid_ask_delta0": delta0,
            "n_bid_ask_quotes": int(spread_rows.sum()),
            "n_locked_fallback": int(locked.sum()) if objective_name == "hybrid" else 0,
            "enabled_constraints": tuple(cfg.enabled_constraints),
        }

        return {"df_rep": df, "plot_data": plot_data, "solve_info": solve_info}
