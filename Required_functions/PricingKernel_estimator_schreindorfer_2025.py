import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, Any, Optional, Tuple, List
import pickle


    

class PricingKernelSurfaceEstimator:
    """
    Option 2: Maturity-specific estimation of the projected pricing kernel,
    following Schreindorfer and Sichert (2025, Journal of Financial Economics).
    
    For a fixed ANCHOR DATE, we estimate a separate pricing-kernel parameter
    vector θ(T_j) for each maturity T_j available in the anchor-date
    risk-neutral density surface.
    
    Methodology:
    -------------
    For each maturity index j on the anchor date with maturity T_anchor[j]:
    
    1. Construct a panel across calendar dates t:
       - On each date t, select the risk-neutral density row whose maturity
         is closest to T_anchor[j].
       - Compute the realized return over the same horizon T_anchor[j]
         (i.e., matching the option-implied return horizon).
       - Transform the selected risk-neutral density f*_t(R) into a physical
         density f_t(R; θ_j) using the pricing-kernel projection
    
             E_t[M_{t+1} | R_{t+1}] = (1 / R_f,t) · f*_t(R_{t+1}) / f_t(R_{t+1})
    
         as in Equation (1) of Schreindorfer and Sichert (2025).
    
    2. Estimate θ_j = (b_j, c_{j,1}, ..., c_{j,N}) by maximizing the
       log-likelihood of realized returns drawn from f_t(R; θ_j),
       as in Equation (3) of Schreindorfer and Sichert (2025).
    
       Estimation is performed separately for each maturity, allowing the
       pricing kernel to vary flexibly across horizons.
    
    Post-estimation (Anchor Date):
    -------------------------------
    Using θ_j for each maturity T_anchor[j], we compute on the anchor date:
    
    - M_surface[j, R]   : projected pricing kernel E[M | R]  
    - pR_surface[j, R]  : physical return density f(R)  
    - qR_surface[j, R]  : risk-neutral return density f*(R)
    
    These objects satisfy the change-of-measure relationship implied by
    no-arbitrage and integrate to one by construction.
    
    Outputs and Visualization:
    --------------------------
    The class provides plotting utilities for:
    
    - The pricing-kernel surface E[M | R, T]
    - The physical density surface f(R | T)
    - Panel plots overlaying:
        * physical density
        * risk-neutral density
        * pricing kernel (secondary axis)
    
    References:
    -----------
    Schreindorfer, D., & Sichert, T. (2025).
    "Conditional Risk and the Pricing Kernel."
    Journal of Financial Economics, 171, 104106.
    https://doi.org/10.1016/j.jfineco.2025.104106
    """

    # =========================
    # init
    # =========================
    def __init__(self, information_dict: Dict[str, Dict[str, Any]], stock_df: pd.DataFrame):
        self.information_dict = information_dict
        self.stock_df = self._ensure_stock_df(stock_df)
        self.fit_out: Optional[Dict[str, Any]] = None  # stores fit + surfaces

    # =========================
    # Save helper
    # =========================
    @staticmethod
    def _maybe_savefig(fig, save: Optional[str], dpi: int = 200):
        if save is None:
            return
        folder = os.path.dirname(save) or "."
        os.makedirs(folder, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    # =========================
    # Stock helpers
    # =========================
    @staticmethod
    def _ensure_stock_df(stock_df: pd.DataFrame) -> pd.DataFrame:
        df = stock_df.copy().iloc[:, :2]
        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _price_on_or_after(stock_df: pd.DataFrame, target_dt: pd.Timestamp) -> Optional[float]:
        idx = stock_df["date"].searchsorted(target_dt, side="left")
        if idx >= len(stock_df):
            return None
        return float(stock_df.loc[idx, "price"])

    @classmethod
    def realized_gross_return(cls, stock_df: pd.DataFrame, date0: pd.Timestamp, S0: float, T_years: float) -> Optional[float]:
        target = date0 + pd.Timedelta(days=float(T_years) * 365.0)
        S_future = cls._price_on_or_after(stock_df, target)
        if S_future is None or not np.isfinite(S_future) or S_future <= 0:
            return None
        if not np.isfinite(S0) or S0 <= 0:
            return None
        return float(S_future / S0)

    # =========================
    # Density helpers
    # =========================
    @staticmethod
    def normalize_pdf(x: np.ndarray, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, float)
        f = np.where(np.isfinite(f), f, 0.0)
        f = np.maximum(f, 0.0)
        mass = np.trapz(f, x)
        if not np.isfinite(mass) or mass <= 0:
            return np.zeros_like(f)
        return f / mass

    @staticmethod
    def select_closest_maturity_row(T_grid: np.ndarray, target_T: float) -> Tuple[int, float]:
        T_grid = np.asarray(T_grid, float)
        j = int(np.argmin(np.abs(T_grid - target_T)))
        return j, float(T_grid[j])

    # =========================
    # Pricing kernel (order N)
    # =========================
    @staticmethod
    def scaled_coeffs_vec(sigma_t: float, b: float, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, float)
        N = c.size
        powers = b * np.arange(1, N + 1, dtype=float)
        return c * (sigma_t ** powers)

    @staticmethod
    def kernel_exponent_no_delta(R: np.ndarray, c_t: np.ndarray) -> np.ndarray:
        z = np.log(np.asarray(R, float))
        c_t = np.asarray(c_t, float)
        N = c_t.size
        Z = np.vstack([z**i for i in range(1, N + 1)])  # (N, n)
        return (c_t[:, None] * Z).sum(axis=0)

    @classmethod
    def compute_delta_t(cls, R_grid: np.ndarray, qR_grid: np.ndarray, Rf_t: float, c_t: np.ndarray) -> float:
        poly = cls.kernel_exponent_no_delta(R_grid, c_t)
        I = np.trapz(qR_grid * np.exp(-poly), R_grid)
        if I <= 0 or (not np.isfinite(I)):
            return np.inf
        return np.log(I) - np.log(Rf_t)

    @classmethod
    def pR_and_M_from_qR(cls, R_grid: np.ndarray, qR_grid: np.ndarray, Rf_t: float,
                         sigma_t: float, b: float, c: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        c_t = cls.scaled_coeffs_vec(sigma_t, b, c)
        delta = cls.compute_delta_t(R_grid, qR_grid, Rf_t, c_t)
        if not np.isfinite(delta):
            z = np.zeros_like(R_grid)
            return z, delta, z
        M = np.exp(delta + cls.kernel_exponent_no_delta(R_grid, c_t))
        pR = qR_grid / (Rf_t * M)
        return pR, delta, M

    @classmethod
    def pR_at_Robs(cls, R_obs: float, R_grid: np.ndarray, qR_grid: np.ndarray, Rf_t: float,
                   sigma_t: float, b: float, c: np.ndarray) -> Tuple[float, float]:
        c_t = cls.scaled_coeffs_vec(sigma_t, b, c)
        delta = cls.compute_delta_t(R_grid, qR_grid, Rf_t, c_t)
        if not np.isfinite(delta):
            return 1e-300, delta
        if (R_obs < R_grid[0]) or (R_obs > R_grid[-1]):
            return 1e-300, delta
        q_obs = float(np.interp(R_obs, R_grid, qR_grid))
        M_obs = float(np.exp(delta + cls.kernel_exponent_no_delta(np.array([R_obs]), c_t)[0]))
        p_obs = q_obs / (Rf_t * M_obs)
        return max(float(p_obs), 1e-300), delta

    # =========================
    # Panel likelihood (var grid)
    # =========================
    @classmethod
    def neg_loglik_panel_vargrid(cls, theta: np.ndarray,
                                 R_obs: np.ndarray,
                                 sigmas: np.ndarray,
                                 Rf: np.ndarray,
                                 R_grids: List[np.ndarray],
                                 qR_grids: List[np.ndarray]) -> float:
        theta = np.asarray(theta, float)
        if theta.size < 2 or (not np.all(np.isfinite(theta))):
            return 1e50
        b = float(theta[0])
        c = theta[1:]
        if b < 0:
            return 1e50

        ll = 0.0
        for t in range(R_obs.size):
            p_obs, delta = cls.pR_at_Robs(float(R_obs[t]), R_grids[t], qR_grids[t],
                                          float(Rf[t]), float(sigmas[t]), b, c)
            if (not np.isfinite(delta)) or (not np.isfinite(p_obs)) or (p_obs <= 0):
                return 1e50
            ll += np.log(p_obs)
        return -ll

    # =========================
    # Anchor picker
    # =========================
    @staticmethod
    def _pick_anchor_date_key(information_dict: Dict[str, Any], anchor_date: str, nearest_if_missing: bool = True) -> str:
        keys = sorted(information_dict.keys())
        if anchor_date in information_dict:
            return anchor_date
        if not nearest_if_missing:
            raise KeyError(f"anchor_date='{anchor_date}' not found in information_dict keys.")
        anchor_dt = pd.to_datetime(anchor_date)
        key_dts = np.array([pd.to_datetime(k) for k in keys])
        idx = int(np.argmin(np.abs((key_dts - anchor_dt).astype("timedelta64[D]").astype(int))))
        return keys[idx]

    # =========================
    # FIT (Option 2)
    # =========================
    def fit_pricing_kernel(
        self,
        N: int = 2,
        anchor_date: str = "2021-06-15",
        fallback_tol_days: float = 0.5,
        r_is_continuous: bool = True,
        min_obs_per_maturity: int = 8,
        bounds_b: Tuple[float, float] = (0.0, 3.0),
        bounds_c: Tuple[float, float] = (-50.0, 50.0),
        nearest_anchor_if_missing: bool = True,
        verbose: bool = True,
        R_common: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:

        info = self.information_dict
        dates = sorted(info.keys())
        if len(dates) == 0:
            raise ValueError("information_dict is empty.")

        anchor_key = self._pick_anchor_date_key(info, anchor_date, nearest_if_missing=nearest_anchor_if_missing)

        # ---- anchor grids ----
        d0 = info[anchor_key]
        S0_0 = float(d0["S0"])
        K0 = np.asarray(d0["K_interp"], float)
        T_anchor = np.asarray(d0["T_interp"], float)
        qS_anchor_surf = np.asarray(d0["rnd_surface"], float)
        if qS_anchor_surf.ndim != 2 or qS_anchor_surf.shape[1] != K0.size or qS_anchor_surf.shape[0] != T_anchor.size:
            raise ValueError(f"Anchor date {anchor_key}: rnd_surface shape mismatch with (T_interp, K_interp).")

        # ---- choose common R grid ----
        if R_common is None:
            R_common = (K0 / S0_0)
            R_common = R_common[np.isfinite(R_common) & (R_common > 0)]
            R_common = np.unique(np.sort(R_common))
        R_common = np.asarray(R_common, float)
        R_common = R_common[np.isfinite(R_common) & (R_common > 0)]
        if R_common.size < 20:
            raise ValueError("R_common grid too small; provide a better grid or check K_interp/S0.")

        tol_years = float(fallback_tol_days) / 365.0
        bounds = [(bounds_b[0], bounds_b[1])] + [(bounds_c[0], bounds_c[1])] * N

        theta_by_j = np.full((T_anchor.size, 1 + N), np.nan, float)  # [b, c1..cN]
        fit_meta = []  # per j status info

        # ---- Fit theta separately for each anchor maturity ----
        for j_anchor, T_target in enumerate(T_anchor):
            # build panel arrays for this maturity
            R_obs_list, sigma_list, Rf_list = [], [], []
            R_grids, qR_grids = [], []
            fallback_events_j = []

            for ds in dates:
                d = info[ds]
                date0 = pd.to_datetime(ds)

                S0 = float(d["S0"])
                r = float(d["r"])
                sigma_t = float(d["atm_vol"])

                K = np.asarray(d["K_interp"], float)
                T_grid = np.asarray(d["T_interp"], float)
                qS_surf = np.asarray(d["rnd_surface"], float)
                if qS_surf.ndim != 2 or qS_surf.shape[1] != K.size:
                    continue

                j, T_chosen = self.select_closest_maturity_row(T_grid, float(T_target))
                diff = abs(T_chosen - float(T_target))
                if diff >= tol_years:
                    fallback_events_j.append((ds, T_chosen * 365.0, diff * 365.0))

                # realized return over EXACT target horizon (anchor maturity)
                R_obs = self.realized_gross_return(self.stock_df, date0, S0, float(T_target))
                if R_obs is None:
                    continue

                qS = self.normalize_pdf(K, qS_surf[j, :])
                if np.trapz(qS, K) <= 0:
                    continue

                R_grid = K / S0
                mpos = np.isfinite(R_grid) & (R_grid > 0)
                if mpos.sum() < 10:
                    continue
                Rg = R_grid[mpos]
                qR = self.normalize_pdf(Rg, qS[mpos] * S0)

                if r_is_continuous:
                    Rf = float(np.exp(r * float(T_target)))
                else:
                    Rf = float((1.0 + r) ** float(T_target))

                R_obs_list.append(float(R_obs))
                sigma_list.append(float(sigma_t))
                Rf_list.append(float(Rf))
                R_grids.append(Rg)
                qR_grids.append(qR)

            n_obs = len(R_obs_list)
            meta = {
                "j_anchor": int(j_anchor),
                "T_years": float(T_target),
                "T_days": float(T_target) * 365.0,
                "n_obs": int(n_obs),
                "fallback_count": int(len(fallback_events_j)),
                "status": "ok"
            }

            if n_obs < min_obs_per_maturity:
                meta["status"] = "skipped_insufficient"
                fit_meta.append(meta)
                continue

            R_obs_arr = np.array(R_obs_list, float)
            sigma_arr = np.array(sigma_list, float)
            Rf_arr = np.array(Rf_list, float)

            x0 = np.zeros(1 + N, float)
            x0[0] = 0.5  # b init

            res = minimize(
                self.neg_loglik_panel_vargrid,
                x0,
                args=(R_obs_arr, sigma_arr, Rf_arr, R_grids, qR_grids),
                method="L-BFGS-B",
                bounds=bounds
            )

            if not (res.success and np.all(np.isfinite(res.x))):
                meta["status"] = "failed_opt"
                meta["message"] = str(getattr(res, "message", ""))
                fit_meta.append(meta)
                continue

            theta_by_j[j_anchor, :] = res.x
            meta["status"] = "fit"
            meta["nll"] = float(res.fun)
            fit_meta.append(meta)

            if verbose and len(fallback_events_j) > 0:
                # print a brief message only (avoid spam)
                print(f"[maturity {meta['T_days']:.1f}d] fallback used on {len(fallback_events_j)} dates (tol={fallback_tol_days}d).")

        # ---- Build anchor-day qR, pR, M surfaces using theta_by_j ----
        nT = T_anchor.size
        qR_surf = np.zeros((nT, R_common.size), float)
        pR_surf = np.zeros((nT, R_common.size), float)
        M_surf = np.zeros((nT, R_common.size), float)
        delta_vec = np.full(nT, np.nan, float)

        r0 = float(d0["r"])
        sigma0 = float(d0["atm_vol"])
        R0 = K0 / S0_0
        mpos0 = np.isfinite(R0) & (R0 > 0)
        R0_pos = R0[mpos0]

        for j in range(nT):
            # risk-neutral returns density on anchor date
            qS_j = self.normalize_pdf(K0, qS_anchor_surf[j, :])
            qR0 = self.normalize_pdf(R0_pos, qS_j[mpos0] * S0_0)

            qR_common = np.zeros_like(R_common)
            m = (R_common >= R0_pos[0]) & (R_common <= R0_pos[-1])
            qR_common[m] = np.interp(R_common[m], R0_pos, qR0)
            qR_common = self.normalize_pdf(R_common, qR_common)
            qR_surf[j, :] = qR_common

            # if theta not fitted for this maturity, leave zeros and continue
            if not np.all(np.isfinite(theta_by_j[j, :])):
                continue

            b_hat = float(theta_by_j[j, 0])
            c_hat = np.asarray(theta_by_j[j, 1:], float)

            Tj = float(T_anchor[j])
            if r_is_continuous:
                Rf_j = float(np.exp(r0 * Tj))
            else:
                Rf_j = float((1.0 + r0) ** Tj)

            pR_j, delta_j, M_j = self.pR_and_M_from_qR(R_common, qR_common, Rf_j, sigma0, b_hat, c_hat)
            pR_j = self.normalize_pdf(R_common, np.maximum(pR_j, 0.0))

            pR_surf[j, :] = pR_j
            M_surf[j, :] = M_j
            delta_vec[j] = delta_j

        self.fit_out = {
            "anchor_requested": anchor_date,
            "anchor_key_used": anchor_key,
            "kernel_order_N": int(N),
            "T_anchor": T_anchor,
            "R_common": R_common,
            "theta_by_maturity": theta_by_j,   # shape (nT, 1+N), NaN where skipped/failed
            "fit_meta": fit_meta,              # list of dicts (status per maturity)
            "anchor_surfaces": {
                "qR_surface": qR_surf,
                "pR_surface": pR_surf,
                "M_surface": M_surf,
                "delta_by_T": delta_vec
            }
        }

        if verbose:
            n_fit = int(np.sum(np.all(np.isfinite(theta_by_j), axis=1)))
            print(f"\nFit complete on anchor={anchor_key}: fitted {n_fit}/{nT} maturities (min_obs_per_maturity={min_obs_per_maturity}).")

        return self.fit_out

    # =========================
    # internal
    # =========================
    def _require_fit(self):
        if self.fit_out is None:
            raise RuntimeError("Call fit_pricing_kernel(...) first.")

    # =========================
    # Plotting functions
    # =========================
    def plot_pricing_kernel_surface(self, title: Optional[str] = None, save: Optional[str] = None, dpi: int = 200):
        self._require_fit()
        out = self.fit_out
        T = out["T_anchor"]
        R = out["R_common"]
        M = out["anchor_surfaces"]["M_surface"]

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        RR, TT = np.meshgrid(R, T)

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(RR, TT, M, linewidth=0, antialiased=True)
        ax.set_xlabel("Gross return R")
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel("M(T,R)")
        if title is None:
            title = f"Pricing Kernel Surface on anchor={out['anchor_key_used']}"
        ax.set_title(title)
        plt.tight_layout()
        self._maybe_savefig(fig, save, dpi=dpi)
        plt.show()

    def plot_physical_density_surface(self, title: Optional[str] = None, save: Optional[str] = None, dpi: int = 200):
        self._require_fit()
        out = self.fit_out
        T = out["T_anchor"]
        R = out["R_common"]
        pR = out["anchor_surfaces"]["pR_surface"]

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        RR, TT = np.meshgrid(R, T)

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(RR, TT, pR, linewidth=0, antialiased=True)
        ax.set_xlabel("Gross return R")
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel("p(T,R)")
        if title is None:
            title = f"Physical Return Density Surface on anchor={out['anchor_key_used']}"
        ax.set_title(title)
        plt.tight_layout()
        self._maybe_savefig(fig, save, dpi=dpi)
        plt.show()

    def plot_panel(
        self,
        title: Optional[str] = None,
        save: Optional[str] = None,
        dpi: int = 200,
        panel_shape: Tuple[int, int] = (2, 4),
        panel_n_curves: Optional[int] = None,
        kernel_color: str = "black",
        kernel_linestyle: str = "--",
        only_fitted: bool = True,
    ):
        self._require_fit()
        out = self.fit_out
        T = out["T_anchor"]
        R = out["R_common"]
        qR = out["anchor_surfaces"]["qR_surface"]
        pR = out["anchor_surfaces"]["pR_surface"]
        M = out["anchor_surfaces"]["M_surface"]
        theta = out["theta_by_maturity"]

        nrows, ncols = panel_shape
        nT = T.size

        # choose indices
        if only_fitted:
            fitted_mask = np.all(np.isfinite(theta), axis=1)
            idx_pool = np.where(fitted_mask)[0]
            if idx_pool.size == 0:
                raise RuntimeError("No maturities were fitted; set only_fitted=False to plot anyway.")
        else:
            idx_pool = np.arange(nT)

        n_panels = nrows * ncols if panel_n_curves is None else int(panel_n_curves)
        n_panels = max(1, min(n_panels, idx_pool.size))

        # evenly spaced across pool
        idxs = idx_pool[np.linspace(0, idx_pool.size - 1, n_panels, dtype=int)]

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

            ax.plot(R, qR[j, :], label="q(R)", linewidth=1.8)
            ax.plot(R, pR[j, :], label="p(R)", linewidth=1.8)
            ax2.plot(
                R, M[j, :],
                label="M(R)",
                color=kernel_color,
                linestyle=kernel_linestyle,
                linewidth=1.6,
                alpha=0.95
            )

            T_days = float(T[j] * 365.0)
            ax.set_title(f"T≈{T_days:.1f}d", fontsize=11)

            if (k % ncols) == 0:
                ax.set_ylabel("Density")
            if k >= (n_panels - ncols):
                ax.set_xlabel("Gross return R")
            if (k % ncols) == (ncols - 1):
                ax2.set_ylabel("Pricing kernel M(R)")
            ax.grid(True, alpha=0.25)

        for k in range(n_panels, axes.size):
            axes[k].axis("off")

        h1, l1 = axes[0].get_legend_handles_labels()
        h2, l2 = ax2_list[0].get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2

        if title is None:
            title = f"Anchor={out['anchor_key_used']}: q vs p with Pricing Kernel (Option 2: θ(T))"

        fig.suptitle(title, y=0.995, fontsize=14)
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=3,
            frameon=False,
            handlelength=2.8,
            columnspacing=1.6
        )
        fig.subplots_adjust(top=0.88, wspace=0.28, hspace=0.30)

        self._maybe_savefig(fig, save, dpi=dpi)
        plt.show()


# ============================================================
# Minimal toy example with bitcoin Options. See 
# ============================================================
if __name__ == "__main__":
    ##Load demo data. The information_dict is just the output of using the fit_surface estimator.
    with open("bitcoin_surfaces_for_price_kernel_demo.pkl", "rb") as f:
        demo_data = pickle.load(f)
        
    
    information_dict=demo_data["information_dict"]
    stock_df=demo_data["stock_df"]

    est = PricingKernelSurfaceEstimator(information_dict, stock_df)
    results = est.fit_pricing_kernel(N=1, anchor_date="2021-08-27", min_obs_per_maturity=8, verbose=True) #N is the polynomial order, for now recommend only 1.
    readable_date = "2021-08-27"
    est.plot_pricing_kernel_surface(title=f"Kernel Surface {readable_date}", save=f"Kernel_surfaces/kernel_{readable_date}.png")
    est.plot_physical_density_surface(title=f"Physical Surface {readable_date}", save=f"Physical_surfaces/p_{readable_date}.png")
    est.plot_panel(title=f"Panels {readable_date}", save=f"Panels/panel_{readable_date}.png", panel_shape=(2,4))


