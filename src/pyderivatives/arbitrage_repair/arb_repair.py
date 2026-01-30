import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Iterable

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
    col_date: str = "date"
    col_T: str = "rounded_maturity"
    col_S0: str = "stock_price"
    col_r: str = "risk_free_rate"
    col_K: str = "strike"
    col_C: str = "mid_price"

    assume_dividend_yield_q: float = 0.0

    solver: str = "ECOS"
    verbose: bool = False

    # cross-maturity matching tolerance in normalized strike (k = K/F)
    k_match_tol: float = 2e-3
    enforce_calendar_adjacent_only: bool = True

    out_dir: str = "arb_repair_output"
    dpi: int = 160

    # term structure plots
    n_term_structure_strikes: int = 6
    min_maturities_per_strike: int = 3

    # heatmap
    heatmap_eps: float = 1e-10
    heatmap_interpolation: str = "nearest"


# ============================================================
# Single unified class
# ============================================================

class CallSurfaceArbitrageRepair:
    """
    Repair + plotting in one class.

    Plotting supports two modes:
      - fig_* methods: return matplotlib figure objects (no saving)
      - save_* methods: save to disk and close figures
    """

    def __init__(self, cfg: RepairConfig):
        self.cfg = cfg
        self._init_dirs()

    # -------------------------
    # filesystem helpers
    # -------------------------

    def _ensure_dir(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def _init_dirs(self) -> None:
        root = self._ensure_dir(self.cfg.out_dir)
        self.dir_root = root
        self.dir_surfaces = self._ensure_dir(os.path.join(root, "surfaces"))
        self.dir_panels   = self._ensure_dir(os.path.join(root, "panels"))
        self.dir_perturb  = self._ensure_dir(os.path.join(root, "perturb"))
        self.dir_term     = self._ensure_dir(os.path.join(root, "term_structures"))
        self.dir_heatmaps = self._ensure_dir(os.path.join(root, "heatmaps"))

    # -------------------------
    # core math helpers
    # -------------------------

    @staticmethod
    def _discount_factor(r: np.ndarray, T: np.ndarray) -> np.ndarray:
        return np.exp(-r * T)

    @staticmethod
    def _forward_price(S0: np.ndarray, r: np.ndarray, q: float, T: np.ndarray) -> np.ndarray:
        return S0 * np.exp((r - q) * T)

    @staticmethod
    def _safe_float(x, default=np.nan) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _get_s0(self, df: pd.DataFrame) -> float:
        if self.cfg.col_S0 in df.columns:
            s0 = np.nanmedian(pd.to_numeric(df[self.cfg.col_S0], errors="coerce").to_numpy(float))
            if np.isfinite(s0):
                return float(s0)
        k0 = np.nanmedian(pd.to_numeric(df[self.cfg.col_K], errors="coerce").to_numpy(float))
        return float(k0) if np.isfinite(k0) else np.nan

    # -------------------------
    # constraint builders
    # -------------------------

    def _build_within_maturity_constraints(
        self,
        idxs_K: np.ndarray,
        K_sorted: np.ndarray,
        c_var: cp.Variable,
        constraints: list,
        enabled: set,
    ) -> None:
        n = len(idxs_K)

        if "C1" in enabled:
            constraints.append(c_var[idxs_K] >= 0)

        if "C2" in enabled and n >= 2:
            for j in range(1, n):
                constraints.append(c_var[idxs_K[j - 1]] >= c_var[idxs_K[j]])

        if "C3" in enabled and n >= 3:
            for j in range(1, n - 1):
                Km1, K0, Kp1 = K_sorted[j - 1], K_sorted[j], K_sorted[j + 1]
                constraints.append(
                    (Kp1 - K0) * (c_var[idxs_K[j]] - c_var[idxs_K[j - 1]])
                    <=
                    (K0 - Km1) * (c_var[idxs_K[j + 1]] - c_var[idxs_K[j]])
                )

    @staticmethod
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

    @staticmethod
    def _bracket(k, x):
        if x < k[0] or x > k[-1]:
            return None
        j = np.searchsorted(k, x)
        return j - 1, j

    def _build_cross_maturity_constraints(
        self,
        maturities: np.ndarray,
        by_T_k: Dict[float, Dict[str, np.ndarray]],
        c_var: cp.Variable,
        constraints: list,
        enabled: set,
    ) -> None:
        cfg = self.cfg
        Ts = sorted(maturities)

        if cfg.enforce_calendar_adjacent_only:
            pairs = list(zip(Ts[:-1], Ts[1:]))
        else:
            pairs = [(Ts[i], Ts[j]) for i in range(len(Ts)) for j in range(i + 1, len(Ts))]

        for T1, T2 in pairs:
            d1, d2 = by_T_k[T1], by_T_k[T2]

            if "C4" in enabled:
                for i, j in self._nearest_match(d1["k"], d1["idx"], d2["k"], d2["idx"], cfg.k_match_tol):
                    constraints.append(c_var[j] >= c_var[i])

            if "C5" in enabled and len(d1["k"]) >= 2:
                for t, kt in enumerate(d2["k"]):
                    br = self._bracket(d1["k"], kt)
                    if br is None:
                        continue
                    L, R = br
                    w = (kt - d1["k"][L]) / (d1["k"][R] - d1["k"][L])
                    constraints.append(
                        c_var[d2["idx"][t]]
                        >= (1 - w) * c_var[d1["idx"][L]] + w * c_var[d1["idx"][R]]
                    )

    # -------------------------
    # Repair API
    # -------------------------

    def repair(
        self,
        df_date: pd.DataFrame,
        *,
        enabled_constraints: Iterable[str] = ("C1", "C2", "C3", "C4", "C5"),
    ) -> pd.DataFrame:
        cfg = self.cfg
        enabled = set(enabled_constraints)

        df = df_date.copy().reset_index(drop=True)

        T = df[cfg.col_T].to_numpy(float)
        K = df[cfg.col_K].to_numpy(float)
        C = df[cfg.col_C].to_numpy(float)
        S0 = df[cfg.col_S0].to_numpy(float)
        r = df[cfg.col_r].to_numpy(float)

        D = self._discount_factor(r, T)
        F = self._forward_price(S0, r, cfg.assume_dividend_yield_q, T)

        c_norm = C / (D * F)
        k_norm = K / F

        n = len(df)
        c_var = cp.Variable(n)

        constraints = []
        objective = cp.Minimize(cp.norm1(c_var - c_norm))

        by_T_k: Dict[float, Dict[str, np.ndarray]] = {}
        for Ti in np.unique(T):
            idx = np.where(T == Ti)[0]

            ordK = np.argsort(K[idx])
            idxK = idx[ordK]
            self._build_within_maturity_constraints(
                idxK, K[idxK], c_var, constraints, enabled
            )

            ordk = np.argsort(k_norm[idx])
            by_T_k[Ti] = {"idx": idx[ordk], "k": k_norm[idx][ordk]}

        self._build_cross_maturity_constraints(np.unique(T), by_T_k, c_var, constraints, enabled)

        constraints += [c_var >= 0, c_var <= 1]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cfg.solver, verbose=cfg.verbose)

        df["C_rep"] = np.asarray(c_var.value).reshape(-1) * D * F
        return df

    # ============================================================
    # FIGURE-BUILDING METHODS (no saving, return fig objects)
    # ============================================================

    def fig_surfaces(self, df: pd.DataFrame, *, title: str):
        cfg = self.cfg
        T = df[cfg.col_T].to_numpy(float)
        K = df[cfg.col_K].to_numpy(float)
        tri = Triangulation(T, K)

        fig = plt.figure(figsize=(14, 6))
        for i, col in enumerate([cfg.col_C, "C_rep"]):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            ax.plot_trisurf(tri, df[col].to_numpy(float))
            ax.set_title("Observed" if i == 0 else "Repaired")
            ax.view_init(elev=20, azim=35)
            ax.set_xlabel("Maturity T (years)")
            ax.set_ylabel("Strike K")
            ax.set_zlabel("Call price C")

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def fig_panels_6x2(self, df: pd.DataFrame, *, title: str):
        cfg = self.cfg
        Ts = sorted(df[cfg.col_T].unique())[:6]
        fig, ax = plt.subplots(6, 2, figsize=(12, 16))

        s0 = self._get_s0(df)

        for i, T in enumerate(Ts):
            sub = df[df[cfg.col_T] == T].sort_values(cfg.col_K)

            ax[i, 0].plot(sub[cfg.col_K], sub[cfg.col_C])
            ax[i, 0].set_title(f"T={T:.3f} Observed")
            ax[i, 0].set_xlabel("Strike K")
            ax[i, 0].set_ylabel("Call price C")

            ax[i, 1].plot(sub[cfg.col_K], sub["C_rep"])
            ax[i, 1].set_title(f"T={T:.3f} Repaired")
            ax[i, 1].set_xlabel("Strike K")
            ax[i, 1].set_ylabel("Call price C")

            if np.isfinite(s0):
                for j in (0, 1):
                    ax[i, j].axvline(s0, linestyle="--", linewidth=1)
                    y_top = ax[i, j].get_ylim()[1]
                    ax[i, j].text(s0, y_top, f" S0={s0:.2f}", va="top", ha="left", fontsize=9)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def fig_perturb(
        self,
        df: pd.DataFrame,
        *,
        title: str,
        mode: str = "absolute",   # "absolute" | "pct_error"
        eps: float = 1e-10,
    ):
        cfg = self.cfg
        C_obs = df[cfg.col_C].to_numpy(float)
        C_rep = df["C_rep"].to_numpy(float)

        if mode == "absolute":
            y = C_rep - C_obs
            ylabel = "C_rep − C_obs"
        elif mode == "pct_error":
            denom = np.maximum(np.abs(C_obs), eps)
            y = (C_rep / denom - 1.0) * 100.0
            ylabel = "Percentage Error: (C_rep / C_obs − 1) × 100"
        else:
            raise ValueError("mode must be 'absolute' or 'pct_error'")

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            df[cfg.col_K].to_numpy(float),
            y,
            c=df[cfg.col_T].to_numpy(float),
            cmap="viridis",
        )
        ax.axhline(0, linewidth=1)

        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Maturity T (years)")

        ax.set_title(title)
        ax.set_xlabel("Strike K")
        ax.set_ylabel(ylabel)

        fig.tight_layout()
        return fig

    def fig_term_structures_exact_K(
        self,
        df: pd.DataFrame,
        *,
        title: str,
        strikes: Optional[List[float]] = None,
    ):
        cfg = self.cfg

        if strikes is None:
            counts = (
                df.groupby(cfg.col_K)[cfg.col_T]
                  .nunique()
                  .sort_values(ascending=False)
            )
            counts = counts[counts >= cfg.min_maturities_per_strike]
            strikes = counts.index.tolist()[: cfg.n_term_structure_strikes]

        if len(strikes) == 0:
            # return an empty fig so notebooks don't crash
            fig = plt.figure(figsize=(6, 4))
            plt.title("No strikes appear across multiple maturities.")
            plt.axis("off")
            return fig

        strikes = sorted([self._safe_float(x) for x in strikes if np.isfinite(self._safe_float(x))])

        n = len(strikes)
        ncols = 3
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for i, K0 in enumerate(strikes):
            ax = axes[i]
            sub = df[np.isclose(df[cfg.col_K].to_numpy(float), K0, atol=0.0)]
            if sub.empty:
                ax.axis("off")
                continue

            sub = (
                sub.groupby(cfg.col_T, as_index=False)
                   .agg({cfg.col_C: "mean", "C_rep": "mean"})
                   .sort_values(cfg.col_T)
            )

            ax.plot(sub[cfg.col_T], sub[cfg.col_C], marker="o", label="Observed")
            ax.plot(sub[cfg.col_T], sub["C_rep"], marker="o", label="Repaired")
            ax.set_title(f"K = {K0:.2f}")
            ax.set_xlabel("T (years)")
            ax.set_ylabel("Call price")
            ax.legend()

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(title, y=1.02)
        fig.tight_layout()
        return fig

    def fig_heatmap_ratio_rep_over_obs(self, df: pd.DataFrame, *, title: str):
        cfg = self.cfg

        tmp = df[[cfg.col_T, cfg.col_K, cfg.col_C, "C_rep"]].copy()
        tmp[cfg.col_T] = pd.to_numeric(tmp[cfg.col_T], errors="coerce")
        tmp[cfg.col_K] = pd.to_numeric(tmp[cfg.col_K], errors="coerce")
        tmp[cfg.col_C] = pd.to_numeric(tmp[cfg.col_C], errors="coerce")
        tmp["C_rep"] = pd.to_numeric(tmp["C_rep"], errors="coerce")
        tmp = tmp.dropna(subset=[cfg.col_T, cfg.col_K, cfg.col_C, "C_rep"])

        if tmp.empty:
            fig = plt.figure(figsize=(6, 4))
            plt.title("Heatmap: no valid rows.")
            plt.axis("off")
            return fig

        tmp = tmp.groupby([cfg.col_K, cfg.col_T], as_index=False).agg({cfg.col_C: "mean", "C_rep": "mean"})

        denom = np.maximum(tmp[cfg.col_C].to_numpy(float), cfg.heatmap_eps)
        tmp["ratio"] = tmp["C_rep"].to_numpy(float) / denom

        pivot = tmp.pivot(index=cfg.col_K, columns=cfg.col_T, values="ratio").sort_index()
        K_vals = pivot.index.to_numpy(float)
        T_vals = pivot.columns.to_numpy(float)
        Z = pivot.to_numpy(float)
        Zm = np.ma.masked_invalid(Z)

        fig, ax = plt.subplots(figsize=(10, 6))
        extent = [np.min(T_vals), np.max(T_vals), np.min(K_vals), np.max(K_vals)]
        im = ax.imshow(
            Zm,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation=cfg.heatmap_interpolation,
        )

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("C_rep / C_obs")

        ax.set_title(title)
        ax.set_xlabel("Maturity T (years)")
        ax.set_ylabel("Strike K")

        s0 = self._get_s0(df)
        if np.isfinite(s0):
            ax.axhline(s0, linestyle="--", linewidth=1)
            ax.text(np.min(T_vals), s0, f" S0={s0:.2f}", va="bottom", ha="left", fontsize=10)

        fig.tight_layout()
        return fig

    def figs_all(
        self,
        df_rep: pd.DataFrame,
        *,
        title_prefix: str = "",
        perturb_mode: str = "absolute",
    ) -> Dict[str, plt.Figure]:
        """
        Returns a dict of figures for notebook workflows.
        Keys: surfaces, panels, perturb, term, heatmap
        """
        prefix = (title_prefix + " - ").strip()
        return {
            "surfaces": self.fig_surfaces(df_rep, title=f"{prefix}Observed vs Repaired"),
            "panels":   self.fig_panels_6x2(df_rep, title=f"{prefix}Panels"),
            "perturb":  self.fig_perturb(df_rep, title=f"{prefix}Perturb ({perturb_mode})", mode=perturb_mode),
            "term":     self.fig_term_structures_exact_K(df_rep, title=f"{prefix}Exact-K Term Structures"),
            "heatmap":  self.fig_heatmap_ratio_rep_over_obs(df_rep, title=f"{prefix}Heatmap: C_rep/C_obs"),
        }

    # ============================================================
    # SAVE METHODS (call fig_*, then save+close)
    # ============================================================

    def save_surface_plot(self, df: pd.DataFrame, *, title: str, filename: str) -> None:
        fig = self.fig_surfaces(df, title=title)
        fig.savefig(os.path.join(self.dir_surfaces, filename), dpi=self.cfg.dpi)
        plt.close(fig)

    def save_panel_plot_6x2(self, df: pd.DataFrame, *, title: str, filename: str) -> None:
        fig = self.fig_panels_6x2(df, title=title)
        fig.savefig(os.path.join(self.dir_panels, filename), dpi=self.cfg.dpi)
        plt.close(fig)

    def save_perturb_plot(
        self,
        df: pd.DataFrame,
        *,
        title: str,
        filename: str,
        mode: str = "absolute",
    ) -> None:
        fig = self.fig_perturb(df, title=title, mode=mode)
        fig.savefig(os.path.join(self.dir_perturb, filename), dpi=self.cfg.dpi)
        plt.close(fig)

    def save_term_structures_plot(self, df: pd.DataFrame, *, title: str, filename: str) -> None:
        fig = self.fig_term_structures_exact_K(df, title=title)
        fig.savefig(os.path.join(self.dir_term, filename), dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    def save_heatmap_plot(self, df: pd.DataFrame, *, title: str, filename: str) -> None:
        fig = self.fig_heatmap_ratio_rep_over_obs(df, title=title)
        fig.savefig(os.path.join(self.dir_heatmaps, filename), dpi=self.cfg.dpi, bbox_inches="tight")
        plt.close(fig)

    def save_all_plots(
        self,
        df_rep: pd.DataFrame,
        *,
        tag: str,
        title_prefix: str = "",
        perturb_mode: str = "absolute",
    ) -> None:
        prefix = (title_prefix + " - ").strip()
        self.save_surface_plot(df_rep, title=f"{prefix}Observed vs Repaired", filename=f"surface_{tag}.png")
        self.save_panel_plot_6x2(df_rep, title=f"{prefix}Panels", filename=f"panels_{tag}.png")
        self.save_perturb_plot(df_rep, title=f"{prefix}Perturb ({perturb_mode})", filename=f"perturb_{tag}.png", mode=perturb_mode)
        self.save_term_structures_plot(df_rep, title=f"{prefix}Exact-K Term Structures", filename=f"term_{tag}.png")
        self.save_heatmap_plot(df_rep, title=f"{prefix}Heatmap: C_rep/C_obs", filename=f"heatmap_{tag}.png")
