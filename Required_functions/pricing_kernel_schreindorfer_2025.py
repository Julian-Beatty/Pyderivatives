import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, Any, Optional, Tuple, List, Union
from typing import Sequence


class PricingKernelSurfaceEstimator:
    """
    MATLAB-replica estimator (log-return space likelihood, sigma-polynomial scaling, mass trick),
    but outputs and plots are returned in **R-space** (gross returns).

    Estimation is done in r = log(R) space:
      - input rnd_surface is q_S(K) (strike density) or q_K(K) (density wrt K); we assume it's
        density in strike units integrating to 1 over K for each maturity row.
      - convert q_S(K) -> f_Q(r) via:
            r = log(K/S0),  f_Q(r) = q_S(K) * dK/dr = q_S(K) * K

    MATLAB-style model:
      - theta is (Ksig+1, N) per maturity (packed column-major)
      - coef_i(t) = sum_{k=0..Ksig} theta_{k,i} * sigma_t^k
      - poly_t(r) = sum_{i=1..N} coef_i(t) * r^i
      - likelihood term uses mass trick:
            logLik_t = ln fQ(r_obs) - poly_t(r_obs) - log( ∫ fQ(r) exp(-poly_t(r)) dr )
        delta_t is implicit: delta_t = log(mass_t)

    Evaluation:
      - builds fQ_r, fP_r, M_r on a common r grid
      - converts to R-space using R = exp(r):
            qR(R) = fQ(r) / R
            pR(R) = fP(r) / R
        and renormalizes densities over R

    Plotting:
      - plots are in R-space using qR_surface, pR_surface, M_surface
      - plot_panel includes optional CDF-based truncation of M(R) in the left tail.
    """

    def __init__(
        self,
        information_dict: Dict[str, Dict[str, Any]],
        stock_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
        vix_col: str = "VIXCLS",
    ):
        self.information_dict = information_dict
        self.stock_df = self._ensure_stock_df(stock_df)

        self.vix_col = str(vix_col)
        self.vix_df = self._ensure_vix_df(vix_df, vix_col=self.vix_col) if vix_df is not None else None

        self.fit_out: Optional[Dict[str, Any]] = None

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
    # VIX helpers
    # =========================
    @staticmethod
    def _ensure_vix_df(vix_df: pd.DataFrame, vix_col: str = "VIXCLS") -> pd.DataFrame:
        df = vix_df.copy()
        if "observation_date" not in df.columns:
            raise ValueError("vix_df must contain a column named 'observation_date'.")
        if vix_col not in df.columns:
            raise ValueError(f"vix_df must contain the column '{vix_col}'.")
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df = df.sort_values("observation_date").reset_index(drop=True)
        df[vix_col] = pd.to_numeric(df[vix_col], errors="coerce")
        return df

    @staticmethod
    def _vix_on_or_before(vix_df: pd.DataFrame, target_dt: pd.Timestamp, vix_col: str = "VIXCLS") -> Optional[float]:
        idx = vix_df["observation_date"].searchsorted(target_dt, side="right") - 1
        if idx < 0:
            return None
        val = float(vix_df.loc[idx, vix_col])
        if not np.isfinite(val):
            return None
        return val

    def _sigma_for_date(self, date0: pd.Timestamp, d: Dict[str, Any], use_vix: bool) -> Optional[float]:
        """
        sigma_t in decimals (e.g. 0.18).
        If use_vix: sigma_t = VIX/100 matched by last obs on/before date.
        Else: uses d["atm_vol"] (assumed decimals).
        """
        if not use_vix:
            sigma = float(d.get("atm_vol", np.nan))
            if np.isfinite(sigma) and sigma > 0:
                return sigma
            return None

        if self.vix_df is None:
            return None

        vix = self._vix_on_or_before(self.vix_df, pd.to_datetime(date0), vix_col=self.vix_col)
        if vix is None:
            return None
        sigma = float(vix) / 100.0
        if np.isfinite(sigma) and sigma > 0:
            return sigma
        return None

    # =========================
    # Date key helper
    # =========================
    @staticmethod
    def _pick_date_key(information_dict: Dict[str, Any], date_like, nearest_if_missing: bool = True) -> str:
        keys = sorted(information_dict.keys())
        if len(keys) == 0:
            raise KeyError("information_dict has no keys.")
        dt = pd.to_datetime(date_like)
        key = dt.strftime("%Y-%m-%d")
        if key in information_dict:
            return key
        if not nearest_if_missing:
            raise KeyError(f"date='{key}' not found in information_dict keys.")
        key_dts = pd.to_datetime(keys).to_numpy(dtype="datetime64[ns]")
        dt64 = dt.to_datetime64()
        diffs_days = np.abs((key_dts - dt64).astype("timedelta64[D]").astype(int))
        idx = int(np.argmin(diffs_days))
        return keys[idx]

    # =========================
    # Numeric helpers
    # =========================
    @staticmethod
    def _normalize_trapz(x: np.ndarray, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, float)
        f = np.where(np.isfinite(f), f, 0.0)
        f = np.maximum(f, 0.0)
        mass = np.trapz(f, x)
        if not np.isfinite(mass) or mass <= 0:
            return np.zeros_like(f)
        return f / mass

    @staticmethod
    def _interp_to_uniform_grid(x: np.ndarray, f: np.ndarray, x_vec: np.ndarray) -> np.ndarray:
        """
        Interpolate density f(x) to uniform grid x_vec (outside range -> 0) then renormalize on x_vec.
        """
        x = np.asarray(x, float)
        f = np.asarray(f, float)
        x_vec = np.asarray(x_vec, float)

        out = np.zeros_like(x_vec)
        if x.size < 2:
            return out

        inside = (x_vec >= x[0]) & (x_vec <= x[-1])
        out[inside] = np.interp(x_vec[inside], x, f)
        out = PricingKernelSurfaceEstimator._normalize_trapz(x_vec, out)
        return out

    # =========================
    # Density conversion: q_S(K) -> f_Q(r)
    # =========================
    @staticmethod
    def _qS_to_fQ_r(K: np.ndarray, qS_K: np.ndarray, S0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert strike density q_S(K) to log-return density f_Q(r) with r = log(K/S0).

        Since K = S0 * exp(r), we have dK/dr = K, so:
            f_Q(r) = q_S(K) * K

        Returns:
            r_grid (sorted ascending), fQ_r (normalized over r_grid)
        """
        K = np.asarray(K, float)
        qS_K = np.asarray(qS_K, float)

        m = np.isfinite(K) & (K > 0) & np.isfinite(qS_K)
        if m.sum() < 10:
            return np.array([], float), np.array([], float)

        K2 = K[m]
        q2 = np.maximum(qS_K[m], 0.0)

        r = np.log(K2 / float(S0))
        fQ = q2 * K2  # dK/dr = K

        idx = np.argsort(r)
        r = r[idx]
        fQ = fQ[idx]

        fQ = PricingKernelSurfaceEstimator._normalize_trapz(r, fQ)
        return r, fQ

    # =========================
    # MATLAB LL_poly replica: pack/unpack theta
    # =========================
    @staticmethod
    def _pack_theta(theta_mat: np.ndarray) -> np.ndarray:
        return np.asarray(theta_mat, float).reshape(-1, order="F")

    @staticmethod
    def _unpack_theta(theta_vec: np.ndarray, Ksig: int, N: int) -> np.ndarray:
        return np.asarray(theta_vec, float).reshape((Ksig + 1, N), order="F")

    @staticmethod
    def _r_powers_vec(r: np.ndarray, N: int) -> np.ndarray:
        r = np.asarray(r, float)
        return np.vstack([r ** i for i in range(1, N + 1)])

    @staticmethod
    def _sigma_powers(sig: np.ndarray, Ksig: int) -> np.ndarray:
        sig = np.asarray(sig, float).reshape(-1)
        return np.vstack([sig ** k for k in range(0, Ksig + 1)]).T  # (Tobs, K+1)

    @classmethod
    def LL_poly_matlab(
        cls,
        theta: np.ndarray,
        N: int,
        ln_fQ: np.ndarray,     # (Tobs, n_grid)
        r_obs: np.ndarray,     # (Tobs,)
        sig_t: np.ndarray,     # (Tobs,)
        r_vec: np.ndarray,     # (n_grid,)
        ln_fQ_t: np.ndarray,   # (Tobs,)
        del_r: float,
        Ksig: int,
        return_grad: bool = True,
    ) -> Tuple[float, Optional[np.ndarray], float, np.ndarray]:
        """
        Returns:
            obj (negative mean logLik),
            grad (negative gradient) if return_grad,
            logLik (mean),
            mass (Tobs,)
        """
        theta = np.asarray(theta, float).reshape(-1)
        Tobs, n_grid = ln_fQ.shape
        if theta.size != (Ksig + 1) * N:
            return 1e50, (np.zeros_like(theta) if return_grad else None), -np.inf, np.full(Tobs, np.nan)

        if not np.all(np.isfinite(theta)):
            return 1e50, (np.zeros_like(theta) if return_grad else None), -np.inf, np.full(Tobs, np.nan)

        theta_mat = cls._unpack_theta(theta, Ksig, N)  # (K+1, N)
        sig_pow = cls._sigma_powers(sig_t, Ksig)       # (Tobs, K+1)
        coef = sig_pow @ theta_mat                     # (Tobs, N)

        r_obs_pow = np.vstack([r_obs ** i for i in range(1, N + 1)]).T  # (Tobs, N)
        r_vec_pow = cls._r_powers_vec(r_vec, N)                          # (N, n_grid)

        lnM_realized = np.sum(coef * r_obs_pow, axis=1)                  # (Tobs,)
        poly_grid = coef @ r_vec_pow                                     # (Tobs, n_grid)

        expo = np.clip(ln_fQ - poly_grid, -745, 709)
        fP_unnorm = np.exp(expo)

        mass = fP_unnorm.sum(axis=1) * float(del_r)
        if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
            return 1e50, (np.zeros_like(theta) if return_grad else None), -np.inf, mass

        logLik_terms = ln_fQ_t - lnM_realized - np.log(mass)
        if np.any(~np.isfinite(logLik_terms)):
            return 1e50, (np.zeros_like(theta) if return_grad else None), -np.inf, mass

        logLik = float(np.mean(logLik_terms))
        obj = -logLik

        if not return_grad:
            return obj, None, logLik, mass

        grad_mat = np.zeros((Ksig + 1, N), float)
        for i in range(1, N + 1):
            r_vec_i = r_vec ** i
            num = (fP_unnorm @ r_vec_i) * float(del_r)
            E_i = num / mass
            term = (-(r_obs ** i) + E_i)  # (Tobs,)
            grad_mat[:, i - 1] = np.mean(sig_pow * term[:, None], axis=0)

        grad_vec = -cls._pack_theta(grad_mat)
        return obj, grad_vec, logLik, mass

    # =========================
    # Maturity row selection
    # =========================
    @staticmethod
    def select_closest_maturity_row(T_grid: np.ndarray, target_T: float) -> Tuple[int, float]:
        T_grid = np.asarray(T_grid, float)
        j = int(np.argmin(np.abs(T_grid - target_T)))
        return j, float(T_grid[j])

    # ==========================================================
    # FIT MASTER THETA GRID (MATLAB replica)
    # ==========================================================
    def fit_theta_master_grid(
        self,
        N: int = 1,
        Ksig: int = 3,
        tol_days: int = 5,
        day_step: int = 1,
        min_obs_per_T: int = 12,
        r_grid_size: int = 400,
        verbose: bool = True,
        max_print_per_T: int = 6,
        use_vix: bool = False,
        multistart: bool = True,
        n_random_starts: int = 0,
        theta_bound: Optional[float] = None,
    ) -> Dict[str, Any]:
        info = self.information_dict
        keys = sorted(info.keys())
        if len(keys) == 0:
            raise ValueError("information_dict is empty.")

        min_days = np.inf
        max_days = -np.inf
        for k in keys:
            T = np.asarray(info[k]["T_interp"], float)
            Td = 365.0 * T[np.isfinite(T)]
            if Td.size == 0:
                continue
            min_days = min(min_days, float(np.min(Td)))
            max_days = max(max_days, float(np.max(Td)))

        if not np.isfinite(min_days) or not np.isfinite(max_days):
            raise ValueError("Could not determine min/max maturity from information_dict.")

        min_d = int(np.floor(min_days))
        max_d = int(np.ceil(max_days))
        T_master_days = np.arange(min_d, max_d + 1, int(day_step), dtype=int)
        T_master_years = T_master_days.astype(float) / 365.0
        tol_years = float(tol_days) / 365.0

        P = (Ksig + 1) * N
        theta_master = np.full((T_master_years.size, P), np.nan, float)
        diag_by_T: List[Dict[str, Any]] = []

        bounds = None
        if theta_bound is not None:
            tb = float(theta_bound)
            bounds = [(-tb, tb)] * P

        rng = np.random.default_rng(0)

        for jT, T_target in enumerate(T_master_years):
            T_target_days = int(T_master_days[jT])

            r_obs_list, sigma_list = [], []
            fQ_rows: List[np.ndarray] = []
            r_rows: List[np.ndarray] = []

            exact = 0
            fallback = 0
            skipped_no_neighbor = 0
            thrown_out = 0
            selected_attempts = 0
            printed = 0

            for ds in keys:
                d = info[ds]
                date0 = pd.to_datetime(ds)

                S0 = float(d["S0"])
                sigma_t = self._sigma_for_date(date0, d, use_vix=use_vix)
                if sigma_t is None:
                    thrown_out += 1
                    continue

                K = np.asarray(d["K_interp"], float)
                T_grid = np.asarray(d["T_interp"], float)
                qS_surf = np.asarray(d["rnd_surface"], float)

                if qS_surf.ndim != 2 or qS_surf.shape[1] != K.size:
                    thrown_out += 1
                    continue

                j_row, T_chosen = self.select_closest_maturity_row(T_grid, float(T_target))
                diff_years = abs(T_chosen - float(T_target))

                if diff_years > tol_years:
                    skipped_no_neighbor += 1
                    if verbose and printed < max_print_per_T:
                        print(f"Cannot find q(T={T_target_days}d) or neighbor within tol={tol_days}d on {ds}, skipping")
                        printed += 1
                    continue

                selected_attempts += 1
                if diff_years <= 1e-12:
                    exact += 1
                else:
                    fallback += 1
                    if verbose and printed < max_print_per_T:
                        chosen_days = int(round(T_chosen * 365.0))
                        print(f"Cannot find q(T={T_target_days}d), fallback to q(T={chosen_days}d) on {ds}")
                        printed += 1

                R_obs = self.realized_gross_return(self.stock_df, date0, S0, float(T_target))
                if R_obs is None or (not np.isfinite(R_obs)) or R_obs <= 0:
                    thrown_out += 1
                    continue
                r_obs = float(np.log(R_obs))

                qS_row = qS_surf[j_row, :]
                r_native, fQ_native = self._qS_to_fQ_r(K, qS_row, S0=S0)
                if r_native.size < 20:
                    thrown_out += 1
                    continue

                r_obs_list.append(r_obs)
                sigma_list.append(float(sigma_t))
                r_rows.append(r_native)
                fQ_rows.append(fQ_native)

            n_obs_used = len(r_obs_list)

            meta = {
                "T_days": int(T_target_days),
                "T_years": float(T_target),
                "n_obs_used": int(n_obs_used),
                "n_selected_attempts": int(selected_attempts),
                "n_fallback_attempts": int(fallback),
                "exact": int(exact),
                "skipped_no_neighbor": int(skipped_no_neighbor),
                "thrown_out_densities": int(thrown_out),
                "status": "ok",
                "theta_calculated": False,
                "use_vix": bool(use_vix),
                "N": int(N),
                "Ksig": int(Ksig),
            }

            if n_obs_used < min_obs_per_T:
                meta["status"] = "skipped_insufficient_obs"
                diag_by_T.append(meta)
                continue

            rmins = [float(np.min(rr)) for rr in r_rows if rr.size > 0]
            rmaxs = [float(np.max(rr)) for rr in r_rows if rr.size > 0]
            r_lo = float(np.min(rmins))
            r_hi = float(np.max(rmaxs))
            if not (np.isfinite(r_lo) and np.isfinite(r_hi) and r_hi > r_lo):
                meta["status"] = "failed_bad_r_range"
                diag_by_T.append(meta)
                continue

            r_vec = np.linspace(r_lo, r_hi, int(r_grid_size))
            del_r = float(r_vec[1] - r_vec[0])

            ln_fQ = np.zeros((n_obs_used, r_vec.size), float)
            ln_fQ_t = np.zeros(n_obs_used, float)

            for t in range(n_obs_used):
                fQ_on = self._interp_to_uniform_grid(r_rows[t], fQ_rows[t], r_vec)
                fQ_on = np.maximum(fQ_on, 1e-300)
                ln_fQ[t, :] = np.log(fQ_on)

                ro = float(r_obs_list[t])
                if ro < r_vec[0] or ro > r_vec[-1]:
                    ln_fQ_t[t] = np.log(1e-300)
                else:
                    val = float(np.interp(ro, r_vec, fQ_on))
                    ln_fQ_t[t] = np.log(max(val, 1e-300))

            r_obs_arr = np.asarray(r_obs_list, float)
            sig_arr = np.asarray(sigma_list, float)

            def make_inits() -> List[np.ndarray]:
                inits = []
                tiny = 1e-5
                inits.append(np.ones(P, float) * tiny)
                inits.append(np.ones(P, float) * (0.5 * tiny))

                decay = (1000.0 ** (-np.arange(1, Ksig + 2, dtype=float)))  # length K+1
                theta0_mat = np.tile(decay[:, None], (1, N))  # (K+1, N)
                inits.append(self._pack_theta(theta0_mat))

                inits.append(rng.normal(0.0, 1e-3, size=P))
                for _ in range(int(n_random_starts)):
                    inits.append(rng.normal(0.0, 1e-3, size=P))
                return inits

            inits = make_inits() if multistart else [np.ones(P, float) * 1e-5]
            best = {"obj": np.inf, "x": None, "logLik": -np.inf}

            for x0 in inits:
                res = minimize(
                    fun=lambda th: self.LL_poly_matlab(
                        th, N, ln_fQ, r_obs_arr, sig_arr, r_vec, ln_fQ_t, del_r, Ksig,
                        return_grad=True
                    )[0],
                    x0=np.asarray(x0, float),
                    jac=lambda th: self.LL_poly_matlab(
                        th, N, ln_fQ, r_obs_arr, sig_arr, r_vec, ln_fQ_t, del_r, Ksig,
                        return_grad=True
                    )[1],
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500}
                )
                if not res.success or not np.all(np.isfinite(res.x)):
                    continue

                obj, _, logLik, _ = self.LL_poly_matlab(
                    res.x, N, ln_fQ, r_obs_arr, sig_arr, r_vec, ln_fQ_t, del_r, Ksig,
                    return_grad=False
                )
                if np.isfinite(obj) and obj < best["obj"]:
                    best = {"obj": float(obj), "x": np.array(res.x, float), "logLik": float(logLik)}

            if best["x"] is None:
                meta["status"] = "failed_opt"
                diag_by_T.append(meta)
                continue

            theta_master[jT, :] = best["x"]
            meta["status"] = "fit"
            meta["theta_calculated"] = True
            meta["neg_mean_logLik"] = float(best["obj"])
            meta["mean_logLik"] = float(best["logLik"])
            meta["r_vec_min"] = float(r_vec[0])
            meta["r_vec_max"] = float(r_vec[-1])
            meta["r_grid_size"] = int(r_vec.size)
            diag_by_T.append(meta)

            if verbose:
                print(
                    f"[FIT OK]: theta(T={T_target_days}d) Summary: "
                    f"n_obs_used={n_obs_used}, n_fallback_attempts={fallback}, exact={exact}, "
                    f"Thrown_out_densities={thrown_out}, sigma_source={'VIX' if use_vix else 'ATM'}, "
                    f"N={N}, Ksig={Ksig}"
                )

        theta_not_calculated_count = int(np.sum([not d.get("theta_calculated", False) for d in diag_by_T]))
        theta_calculated_count = int(np.sum([d.get("theta_calculated", False) for d in diag_by_T]))

        if verbose:
            print(
                f"\nMaster theta fit complete: fitted {theta_calculated_count}/{T_master_years.size} maturities "
                f"(tol={tol_days}d, min_obs_per_T={min_obs_per_T})."
            )
            print(f"Theta NOT calculated for {theta_not_calculated_count} maturities.")

        return {
            "T_master_days": T_master_days,
            "T_master_years": T_master_years,
            "theta_master": theta_master,          # (nT, (Ksig+1)*N)
            "diag_by_T": diag_by_T,
            "tol_days_fit": int(tol_days),
            "day_step": int(day_step),
            "theta_not_calculated_count": theta_not_calculated_count,
            "theta_calculated_count": theta_calculated_count,
            "use_vix": bool(use_vix),
            "vix_col": self.vix_col if use_vix else None,
            "N": int(N),
            "Ksig": int(Ksig),
            "r_grid_size": int(r_grid_size),
            "theta_bound": float(theta_bound) if theta_bound is not None else None,
        }

    # ==========================================================
    # EVALUATE SURFACES FOR A DATE USING MASTER THETA
    #   -> outputs in R-space for plotting
    # ==========================================================
    def evaluate_surfaces_for_date_master(
        self,
        date_str: str,
        theta_cache: Dict[str, Any],
        *,
        r_common: Optional[np.ndarray] = None,
        tol_days_eval: int = 5,
        warn: bool = True,
        use_vix: Optional[bool] = None,
    ) -> Dict[str, Any]:
        info = self.information_dict
        date_key = self._pick_date_key(info, date_str, nearest_if_missing=True)
        d = info[date_key]

        T_master_years = np.asarray(theta_cache["T_master_years"], float)
        T_master_days = np.asarray(theta_cache["T_master_days"], int)
        theta_master = np.asarray(theta_cache["theta_master"], float)

        N = int(theta_cache["N"])
        Ksig = int(theta_cache["Ksig"])
        r_grid_size = int(theta_cache.get("r_grid_size", 400))

        if use_vix is None:
            use_vix = bool(theta_cache.get("use_vix", False))

        tol_years_eval = float(tol_days_eval) / 365.0

        S0 = float(d["S0"])
        sigma0 = self._sigma_for_date(pd.to_datetime(date_key), d, use_vix=use_vix)
        if sigma0 is None:
            raise ValueError(f"Could not obtain sigma for date={date_key} using {'VIX' if use_vix else 'ATM vol'}.")

        K = np.asarray(d["K_interp"], float)
        T_day = np.asarray(d["T_interp"], float)
        qS_surf = np.asarray(d["rnd_surface"], float)

        if qS_surf.ndim != 2 or qS_surf.shape[1] != K.size:
            raise ValueError("rnd_surface shape mismatch.")

        nT_day = T_day.size

        # Common r grid (for the anchor date)
        if r_common is None:
            m = np.isfinite(K) & (K > 0)
            if m.sum() < 10:
                raise ValueError("Not enough positive strikes to build r_common.")
            rmin = float(np.min(np.log(K[m] / S0)))
            rmax = float(np.max(np.log(K[m] / S0)))
            if not (np.isfinite(rmin) and np.isfinite(rmax) and rmax > rmin):
                raise ValueError("Bad r_common range.")
            r_common = np.linspace(rmin, rmax, r_grid_size)

        r_common = np.asarray(r_common, float)
        del_r = float(r_common[1] - r_common[0])

        # r-space surfaces
        fQ_surf = np.zeros((nT_day, r_common.size), float)
        fP_surf = np.zeros((nT_day, r_common.size), float)
        M_r_surf = np.zeros((nT_day, r_common.size), float)
        delta_vec = np.full(nT_day, np.nan, float)

        theta_match: List[Dict[str, Any]] = []
        n_rows_no_theta_within_tol = 0
        n_rows_theta_not_fitted = 0
        n_rows_theta_fallback_used = 0
        n_rows_theta_used_ok = 0

        sig_pow = np.array([sigma0 ** k for k in range(0, Ksig + 1)], float)  # (K+1,)
        r_pow_grid = self._r_powers_vec(r_common, N)                           # (N, n_grid)

        for j in range(nT_day):
            Tj = float(T_day[j])
            Tj_days = float(Tj * 365.0)

            idx = int(np.argmin(np.abs(T_master_years - Tj)))
            Tm = float(T_master_years[idx])
            Tm_days = float(Tm * 365.0)
            diff_days = abs(Tm_days - Tj_days)

            row_diag = {
                "row_index": int(j),
                "T_row_years": float(Tj),
                "T_row_days": float(Tj_days),
                "theta_index": int(idx),
                "T_theta_years": float(Tm),
                "T_theta_days": float(Tm_days),
                "diff_days": float(diff_days),
                "status": "ok",
            }

            if abs(Tm - Tj) > tol_years_eval:
                row_diag["status"] = "no_theta_within_tol"
                n_rows_no_theta_within_tol += 1
                if warn:
                    print(f"[WARN {date_key}] No theta within tol for row T={Tj_days:.1f}d (nearest {Tm_days:.1f}d).")
                theta_match.append(row_diag)
                continue

            theta_vec = theta_master[idx, :]
            if not np.all(np.isfinite(theta_vec)):
                row_diag["status"] = "theta_not_fitted"
                n_rows_theta_not_fitted += 1
                if warn:
                    print(f"[WARN {date_key}] Theta for {Tm_days:.1f}d not fitted (NaN). Row T={Tj_days:.1f}d skipped.")
                theta_match.append(row_diag)
                continue

            if diff_days > 1e-12:
                row_diag["status"] = "fallback_theta_used"
                n_rows_theta_fallback_used += 1
                if warn:
                    print(f"[WARN {date_key}] Row T={Tj_days:.1f}d using fallback theta at {Tm_days:.1f}d.")
            else:
                n_rows_theta_used_ok += 1

            r_native, fQ_native = self._qS_to_fQ_r(K, qS_surf[j, :], S0=S0)
            if r_native.size < 20:
                row_diag["status"] = "bad_density_row"
                theta_match.append(row_diag)
                continue

            fQ_on = self._interp_to_uniform_grid(r_native, fQ_native, r_common)
            fQ_on = np.maximum(fQ_on, 1e-300)
            ln_fQ_on = np.log(fQ_on)
            fQ_surf[j, :] = fQ_on

            theta_mat = self._unpack_theta(theta_vec, Ksig, N)   # (K+1, N)
            coef = sig_pow @ theta_mat                           # (N,)
            poly = coef @ r_pow_grid                              # (n_grid,)

            expo = np.clip(ln_fQ_on - poly, -745, 709)
            fP_unnorm = np.exp(expo)

            mass = float(np.sum(fP_unnorm) * del_r)
            if (not np.isfinite(mass)) or mass <= 0:
                row_diag["status"] = "bad_mass"
                theta_match.append(row_diag)
                continue

            delta = float(np.log(mass))
            delta_vec[j] = delta

            fP = fP_unnorm / mass
            fP = self._normalize_trapz(r_common, fP)

            M_r = np.exp(np.clip(delta + poly, -745, 709))

            fP_surf[j, :] = fP
            M_r_surf[j, :] = M_r
            theta_match.append(row_diag)

        # Convert to R-space
        R_common = np.exp(r_common)

        # qR(R) = fQ(r) / R ; pR(R) = fP(r) / R
        qR_surf = fQ_surf / R_common[None, :]
        pR_surf = fP_surf / R_common[None, :]

        # renormalize in R-space row-by-row
        for j in range(nT_day):
            qR_surf[j, :] = self._normalize_trapz(R_common, qR_surf[j, :])
            pR_surf[j, :] = self._normalize_trapz(R_common, pR_surf[j, :])

        # Pricing kernel as function of R is same numeric array, just x-axis relabeled
        M_surf = M_r_surf.copy()

        self.fit_out = {
            "anchor_key_used": date_key,
            "T_anchor": T_day,
            "r_common": r_common,
            "R_common": R_common,
            "sigma_source": "VIX" if use_vix else "ATM",
            "sigma_value": float(sigma0),
            "model_spec": {"N": int(N), "Ksig": int(Ksig), "space": "log-return"},
            "anchor_surfaces": {
                # R-space outputs for your plotting stack:
                "qR_surface": qR_surf,
                "pR_surface": pR_surf,
                "M_surface": M_surf,
                "delta_by_T": delta_vec,
                # keep r-space too (handy for debugging)
                "fQ_r_surface": fQ_surf,
                "fP_r_surface": fP_surf,
                "M_r_surface": M_r_surf,
            },
            "theta_cache_info": {
                "T_master_days": T_master_days,
                "tol_days_eval": int(tol_days_eval),
                "use_vix": bool(use_vix),
                "vix_col": self.vix_col if use_vix else None,
            },
            "theta_match": theta_match,
            "theta_eval_summary": {
                "n_rows_total": int(nT_day),
                "n_rows_no_theta_within_tol": int(n_rows_no_theta_within_tol),
                "n_rows_theta_not_fitted": int(n_rows_theta_not_fitted),
                "n_rows_theta_fallback_used": int(n_rows_theta_fallback_used),
                "n_rows_theta_used_ok": int(n_rows_theta_used_ok),
                "n_rows_failed_total": int(n_rows_no_theta_within_tol + n_rows_theta_not_fitted),
            }
        }
        return self.fit_out

    # =========================
    # internal
    # =========================
    def _require_fit(self):
        if self.fit_out is None:
            raise RuntimeError("Call evaluate_surfaces_for_date_master(...) first.")
    
    def plot_pricing_kernel_surface(
        self,
        title: Optional[str] = None,
        save: Optional[str] = None,
        dpi: int = 200,
        R_bounds: Optional[Tuple[float, float]] = None,
        T_bounds: Optional[Tuple[float, float]] = None,
        cmap: str = "viridis",                 # NEW
        add_colorbar: bool = True,             # NEW
        vmin: Optional[float] = None,          # NEW
        vmax: Optional[float] = None,          # NEW
    ):
        self._require_fit()
        out = self.fit_out
        T = np.asarray(out["T_anchor"], float)
        R = np.asarray(out["R_common"], float)
        M = np.asarray(out["anchor_surfaces"]["M_surface"], float)
    
        # slice by bounds
        T_sel, R_sel, M_sel = _apply_bounds_2d_surface(T, R, M, R_bounds=R_bounds, T_bounds=T_bounds)
    
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        RR, TT = np.meshgrid(R_sel, T_sel)
    
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
    
        surf = ax.plot_surface(
            RR, TT, M_sel,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            antialiased=True,
        )
    
        ax.set_xlabel("Gross return R")
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel("M(T,R)")
    
        if title is None:
            title = f"Pricing Kernel Surface (MATLAB replica) date={out['anchor_key_used']} ({out.get('sigma_source','?')})"
            if R_bounds is not None or T_bounds is not None:
                title += f"\nBounds: R={R_bounds}, T={T_bounds}"
    
        ax.set_title(title)
    
        if add_colorbar:
            cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
            cbar.set_label("M(T,R)")
    
        plt.tight_layout()
        self._maybe_savefig(fig, save, dpi=dpi)
        plt.show()
    
    
    def plot_physical_density_surface(
        self,
        title: Optional[str] = None,
        save: Optional[str] = None,
        dpi: int = 200,
        R_bounds: Optional[Tuple[float, float]] = None,
        T_bounds: Optional[Tuple[float, float]] = None,
        cmap: str = "viridis",                 # NEW
        add_colorbar: bool = True,             # NEW
        vmin: Optional[float] = None,          # NEW
        vmax: Optional[float] = None,          # NEW
    ):
        self._require_fit()
        out = self.fit_out
        T = np.asarray(out["T_anchor"], float)
        R = np.asarray(out["R_common"], float)
        pR = np.asarray(out["anchor_surfaces"]["pR_surface"], float)
    
        # slice by bounds
        T_sel, R_sel, pR_sel = _apply_bounds_2d_surface(T, R, pR, R_bounds=R_bounds, T_bounds=T_bounds)
    
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        RR, TT = np.meshgrid(R_sel, T_sel)
    
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
    
        surf = ax.plot_surface(
            RR, TT, pR_sel,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            antialiased=True,
        )
    
        ax.set_xlabel("Gross return R")
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel("p(T,R)")
    
        if title is None:
            title = f"Physical Density Surface (MATLAB replica) date={out['anchor_key_used']} ({out.get('sigma_source','?')})"
            if R_bounds is not None or T_bounds is not None:
                title += f"\nBounds: R={R_bounds}, T={T_bounds}"
    
        ax.set_title(title)
    
        if add_colorbar:
            cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
            cbar.set_label("p(T,R)")
    
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
        # --- truncation controls ---
        truncate: bool = True,
        alpha: float = 0.10,  # CDF-based left-tail truncation (only used if mode="cdf")
        trunc_mode: str = "cdf",  # {"cdf", "rbounds", "none"}
        r_bounds: Optional[Tuple[float, float]] = None,  # e.g. (0.9, 1.1) for mode="rbounds"
        # --- behavior ---
        clip_trunc_to_support: bool = True,  # if True, intersect r_bounds with available R support
    ):
        """
        Panel plot: q_R(R), p_R(R) on left axis and pricing kernel M(R) on right axis.
    
        Truncation options for the pricing kernel curve (M):
          - trunc_mode="cdf": (default) plot M only after left-tail mass under q_R reaches `alpha`.
          - trunc_mode="rbounds": plot M only for R in [r_bounds[0], r_bounds[1]].
          - trunc_mode="none": plot full M (ignore alpha/r_bounds).
    
        Notes:
          - R_common must be strictly increasing.
          - q_R assumed to be a density in R-space (nonnegative, integrates ~1).
          - If trunc_mode="rbounds" and the bounds do not overlap the grid,
            we fall back to plotting nothing for M on that panel (safe behavior).
        """
        self._require_fit()
        out = self.fit_out
        T = np.asarray(out["T_anchor"], float)
        R = np.asarray(out["R_common"], float)
        qR = np.asarray(out["anchor_surfaces"]["qR_surface"], float)
        pR = np.asarray(out["anchor_surfaces"]["pR_surface"], float)
        M = np.asarray(out["anchor_surfaces"]["M_surface"], float)
    
        if R.size >= 2 and np.any(np.diff(R) <= 0):
            raise ValueError("R_common must be strictly increasing for truncation logic.")
    
        mode = str(trunc_mode).lower().strip()
        if mode not in {"cdf", "rbounds", "none"}:
            raise ValueError("trunc_mode must be one of {'cdf','rbounds','none'}.")
    
        if truncate is False:
            mode = "none"
    
        if mode == "cdf":
            if not (0.0 < alpha < 1.0):
                raise ValueError("alpha must be in (0, 1). Example: alpha=0.10 for 10% left-tail mass.")
    
        if mode == "rbounds":
            if r_bounds is None or len(r_bounds) != 2:
                raise ValueError("For trunc_mode='rbounds', provide r_bounds=(R_min, R_max), e.g. (0.9,1.1).")
            R_min, R_max = float(r_bounds[0]), float(r_bounds[1])
            if not (np.isfinite(R_min) and np.isfinite(R_max) and R_max > R_min):
                raise ValueError("r_bounds must be finite and satisfy R_max > R_min.")
            if clip_trunc_to_support:
                R_min = max(R_min, float(R[0]))
                R_max = min(R_max, float(R[-1]))
            # after clipping, bounds might collapse
            if not (R_max > R_min):
                # no overlap with support: we'll plot no kernel segments
                R_min, R_max = np.nan, np.nan
    
        nrows, ncols = panel_shape
        nT = T.size
        idx_pool = np.arange(nT)
    
        n_panels = nrows * ncols if panel_n_curves is None else int(panel_n_curves)
        n_panels = max(1, min(n_panels, idx_pool.size))
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
    
            ax.plot(R, qR[j, :], label="q_R(R)", linewidth=1.8)
            ax.plot(R, pR[j, :], label="p_R(R)", linewidth=1.8)
    
            # -------- choose kernel segment to plot --------
            R_plot = R
            M_plot = M[j, :]
    
            kernel_label = "M(R)"
            if mode == "cdf":
                qj = np.asarray(qR[j, :], float)
                qj_pos = np.maximum(qj, 0.0)
    
                dR = np.diff(R)
                inc = 0.5 * (qj_pos[1:] + qj_pos[:-1]) * dR
                cdf = np.empty_like(R)
                cdf[0] = 0.0
                cdf[1:] = np.cumsum(inc)
    
                total_mass = float(cdf[-1])
                if total_mass > 0 and np.isfinite(total_mass):
                    cdf_norm = cdf / total_mass
                    idx0 = np.where(cdf_norm >= alpha)[0]
                    start_idx = int(idx0[0]) if idx0.size > 0 else 0
                    R_plot = R[start_idx:]
                    M_plot = M_plot[start_idx:]
                    kernel_label = f"M(R) (start @ q-CDF {alpha:.0%})"
                # else: degenerate q -> fall back to full curve
    
            elif mode == "rbounds":
                if np.isfinite(R_min) and np.isfinite(R_max):
                    mm = (R >= R_min) & (R <= R_max)
                    if np.any(mm):
                        R_plot = R[mm]
                        M_plot = M_plot[mm]
                        kernel_label = f"M(R) on [{R_min:.3g},{R_max:.3g}]"
                    else:
                        # no overlap => plot nothing (safe)
                        R_plot = np.array([], float)
                        M_plot = np.array([], float)
                        kernel_label = f"M(R) on [{R_min:.3g},{R_max:.3g}]"
                else:
                    R_plot = np.array([], float)
                    M_plot = np.array([], float)
                    kernel_label = "M(R) (no overlap)"
    
            # mode == "none": leave full curve
    
            if R_plot.size > 0:
                ax2.plot(
                    R_plot, M_plot,
                    label=kernel_label,
                    color=kernel_color,
                    linestyle=kernel_linestyle,
                    linewidth=1.6,
                    alpha=0.95
                )
    
            T_days = float(T[j] * 365.0)
            ax.set_title(f"T≈{T_days:.1f}d", fontsize=11)
    
            if (k % ncols) == 0:
                ax.set_ylabel("Density (R-space)")
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
            title = f"Date={out['anchor_key_used']}: q_R vs p_R with Pricing Kernel (MATLAB replica)"
    
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
def _apply_bounds_2d_surface(
    T: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    R_bounds: Optional[Tuple[float, float]] = None,
    T_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice a (nT, nR) surface Z(T,R) using optional bounds.

    Returns (T_sel, R_sel, Z_sel) where:
      - T_sel shape (nT_sel,)
      - R_sel shape (nR_sel,)
      - Z_sel shape (nT_sel, nR_sel)
    """
    T = np.asarray(T, float)
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)

    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D (nT, nR). Got shape {Z.shape}.")
    if Z.shape != (T.size, R.size):
        raise ValueError(
            f"Shape mismatch: Z.shape={Z.shape}, expected ({T.size}, {R.size}) from T and R."
        )

    tmask = np.isfinite(T)
    rmask = np.isfinite(R)

    if T_bounds is not None:
        tlo, thi = float(T_bounds[0]), float(T_bounds[1])
        if not (tlo < thi):
            raise ValueError(f"T_bounds must satisfy lo < hi. Got {T_bounds}.")
        tmask &= (T >= tlo) & (T <= thi)

    if R_bounds is not None:
        rlo, rhi = float(R_bounds[0]), float(R_bounds[1])
        if not (rlo < rhi):
            raise ValueError(f"R_bounds must satisfy lo < hi. Got {R_bounds}.")
        rmask &= (R >= rlo) & (R <= rhi)

    T_sel = T[tmask]
    R_sel = R[rmask]
    Z_sel = Z[np.ix_(tmask, rmask)]

    if T_sel.size < 2 or R_sel.size < 2:
        raise ValueError(
            f"Too few points after bounds: nT={T_sel.size}, nR={R_sel.size}. "
            f"Try relaxing R_bounds/T_bounds."
        )
    return T_sel, R_sel, Z_sel


def _maybe_savefig(fig, save: Optional[str], dpi: int = 200):
    if save is None:
        return
    # Accept either directory or full file path
    if os.path.isdir(save):
        raise ValueError(
            "save must be a file path for these methods (or pass a file path via your class helper)."
        )
    fig.savefig(save, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {save}")

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
##Helper Functions, experimental
##Calculates RRA(T,R) Surface, and estimates of risk aversion
##SKip to bottom after loading

# ============================================================
# Add-on: gamma(T) from parametric utility fits (Power / Exponential)
#   Power:       log M = alpha(T) - gamma(T) * log R
#   Exponential: log M = alpha(T) - a(T) * R
# ============================================================
def compute_gamma_by_maturity(
    RRA_surface: np.ndarray,
    R_grid: np.ndarray,
    *,
    pR_surface: Optional[np.ndarray] = None,
    window: Tuple[float, float] = (0.90, 1.10),
    use_p_weights_if_available: bool = True,
    winsor_quantiles: Optional[Tuple[float, float]] = (0.01, 0.99),
) -> np.ndarray:
    """
    Collapse RRA(R,T) -> gamma(T), one number per maturity.

    Definitions
    -----------
    If pR_surface is provided and use_p_weights_if_available=True:

        gamma(T) = ∫ RRA(R,T) p(R,T) dR

    Else:

        gamma(T) = mean{ RRA(R,T) : R in [window[0], window[1]] }

    Parameters
    ----------
    RRA_surface : np.ndarray
        Array of shape (nT, nR).
    R_grid : np.ndarray
        Gross return grid of shape (nR,).
    pR_surface : np.ndarray, optional
        Physical density surface of shape (nT, nR).
    window : tuple(float, float)
        Return window [R_lo, R_hi] for averaging.
    use_p_weights_if_available : bool
        Whether to weight by p(R,T) when provided.
    winsor_quantiles : tuple(float, float) or None
        Quantiles for winsorizing RRA per maturity to stabilize estimates.

    Returns
    -------
    gamma_T : np.ndarray
        Array of shape (nT,) with one gamma per maturity.
    """

    RRA = np.asarray(RRA_surface, float)
    R = np.asarray(R_grid, float)

    if RRA.ndim != 2:
        raise ValueError("RRA_surface must be 2D (nT, nR).")
    if RRA.shape[1] != R.size:
        raise ValueError("RRA_surface second dimension must match len(R_grid).")

    nT, nR = RRA.shape
    gamma_T = np.full(nT, np.nan, float)

    # Near-ATM window mask
    R_lo, R_hi = float(window[0]), float(window[1])
    mask_window = (R >= R_lo) & (R <= R_hi)

    # Integration weights
    wR = _trapz_weights(R)

    for t in range(nT):
        rra_row = RRA[t, :].copy()

        # Winsorize to limit tail explosions
        if winsor_quantiles is not None:
            qlo, qhi = winsor_quantiles
            finite = np.isfinite(rra_row)
            if np.any(finite):
                lo = np.quantile(rra_row[finite], qlo)
                hi = np.quantile(rra_row[finite], qhi)
                rra_row = np.clip(rra_row, lo, hi)

        if use_p_weights_if_available and (pR_surface is not None):
            pR = np.asarray(pR_surface, float)
            if pR.shape != RRA.shape:
                raise ValueError("pR_surface must have the same shape as RRA_surface.")

            p_row = _safe_normalize_pdf(R, pR[t, :])

            num = np.sum(wR * p_row * rra_row)
            den = np.sum(wR * p_row)

            if np.isfinite(num) and np.isfinite(den) and den > 0:
                gamma_T[t] = num / den
            else:
                gamma_T[t] = np.nan

        else:
            m = mask_window & np.isfinite(rra_row)
            if np.sum(m) >= 3:
                gamma_T[t] = float(np.mean(rra_row[m]))
            else:
                gamma_T[t] = np.nan

    return gamma_T
def _weighted_linreg(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """
    Weighted least squares for y = a + b x.
    Returns (a, b).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if np.sum(m) < 3:
        return (np.nan, np.nan)

    x = x[m]; y = y[m]; w = w[m]
    w = w / np.sum(w)

    xbar = np.sum(w * x)
    ybar = np.sum(w * y)
    var = np.sum(w * (x - xbar) ** 2)
    if not np.isfinite(var) or var <= 0:
        return (np.nan, np.nan)

    cov = np.sum(w * (x - xbar) * (y - ybar))
    b = cov / var
    a = ybar - b * xbar
    return float(a), float(b)


def compute_gamma_power_exponential_by_maturity(
    M_surface: np.ndarray,
    R_grid: np.ndarray,
    *,
    pR_surface: Optional[np.ndarray] = None,
    window: Tuple[float, float] = (0.90, 1.10),
    use_p_weights_if_available: bool = True,
    eps: float = 1e-300,
    clip_lnM: Tuple[float, float] = (-700.0, 700.0),
    winsor_quantiles_lnM: Optional[Tuple[float, float]] = (0.01, 0.99),
) -> Dict[str, np.ndarray]:
    """
    For each maturity row t, estimate:
      (A) Power/CRRA:       ln M = alpha - gamma ln R
          -> gamma_power[t] = -slope on lnR
      (B) Exponential/CARA: ln M = alpha - a R
          -> a_exp[t] = -slope on R

    Uses WLS, optionally with p(R,T) * dR weights within the window.

    Returns dict:
      {
        "alpha_power": (nT,),
        "gamma_power": (nT,),
        "alpha_exponential": (nT,),
        "a_exponential": (nT,)
      }
    """
    M = np.asarray(M_surface, float)
    R = np.asarray(R_grid, float)

    if M.ndim != 2:
        raise ValueError("M_surface must be 2D (nT, nR).")
    if R.ndim != 1 or np.any(R <= 0) or np.any(~np.isfinite(R)):
        raise ValueError("R_grid must be 1D, finite, strictly > 0.")
    if M.shape[1] != R.size:
        raise ValueError("M_surface second dimension must match len(R_grid).")

    nT, nR = M.shape

    # Window mask
    wlo, whi = float(window[0]), float(window[1])
    mask = (R >= wlo) & (R <= whi)
    if np.sum(mask) < 3:
        raise ValueError("Window contains fewer than 3 R points; widen it.")

    # Coordinates
    lnR = np.log(R)
    lnM = np.log(np.maximum(M, eps))
    lnM = np.clip(lnM, clip_lnM[0], clip_lnM[1])

    # Integration weights
    wR = _trapz_weights(R)

    # outputs
    alpha_power = np.full(nT, np.nan, float)
    gamma_power = np.full(nT, np.nan, float)
    alpha_exp = np.full(nT, np.nan, float)
    a_exp = np.full(nT, np.nan, float)

    for t in range(nT):
        y = lnM[t, :].copy()

        # winsorize lnM within row for stability
        if winsor_quantiles_lnM is not None:
            qlo, qhi = winsor_quantiles_lnM
            finite = np.isfinite(y)
            if np.any(finite):
                lo = np.quantile(y[finite], qlo)
                hi = np.quantile(y[finite], qhi)
                y = np.clip(y, lo, hi)

        # choose weights
        if use_p_weights_if_available and (pR_surface is not None):
            pR = np.asarray(pR_surface, float)
            if pR.shape != M.shape:
                raise ValueError("pR_surface must have the same shape as M_surface.")
            p_row = _safe_normalize_pdf(R, pR[t, :])
            w = wR * p_row
        else:
            w = np.ones_like(R)

        # apply window + finite masks
        m = mask & np.isfinite(y) & np.isfinite(w) & (w > 0)

        # ----- Power: y = a + b*lnR, gamma = -b
        a1, b1 = _weighted_linreg(lnR[m], y[m], w[m])
        alpha_power[t] = a1
        gamma_power[t] = -b1 if np.isfinite(b1) else np.nan

        # ----- Exponential: y = a + b*R, a_exp = -b
        a2, b2 = _weighted_linreg(R[m], y[m], w[m])
        alpha_exp[t] = a2
        a_exp[t] = -b2 if np.isfinite(b2) else np.nan

    return {
        "alpha_power": alpha_power,
        "gamma_power": gamma_power,
        "alpha_exponential": alpha_exp,
        "a_exponential": a_exp,
        "window": np.array(window, float),
    }


# ============================================================
# Patch: extend compute_rra_for_all_dates(...) to store these too
# ============================================================


def compute_rra_surface_from_M(
    M_surface: np.ndarray,
    R_grid: np.ndarray,
    *,
    eps: float = 1e-300,
    clip_lnM: Tuple[float, float] = (-700.0, 700.0),
) -> np.ndarray:
    """
    Compute the RRA(R,T) surface from the pricing kernel M(R,T) using:

        RRA(R,T) = - d ln M(R,T) / d ln R

    Parameters
    ----------
    M_surface : np.ndarray
        Pricing kernel surface of shape (nT, nR).
    R_grid : np.ndarray
        Gross return grid of shape (nR,), must be strictly positive.
    eps : float, optional
        Small constant to avoid log(0).
    clip_lnM : tuple(float, float), optional
        Clipping bounds for ln M to stabilize numerical derivatives.

    Returns
    -------
    RRA_surface : np.ndarray
        Relative risk aversion surface of shape (nT, nR).
    """

    M = np.asarray(M_surface, float)
    R = np.asarray(R_grid, float)

    if M.ndim != 2:
        raise ValueError("M_surface must be 2D (nT, nR).")
    if R.ndim != 1:
        raise ValueError("R_grid must be 1D.")
    if M.shape[1] != R.size:
        raise ValueError("M_surface second dimension must match len(R_grid).")
    if np.any(~np.isfinite(R)) or np.any(R <= 0):
        raise ValueError("R_grid must be finite and strictly > 0.")

    # Log grids
    lnR = np.log(R)

    # Stable log(M): avoid log(0) and extreme values
    lnM = np.log(np.maximum(M, eps))
    lnM = np.clip(lnM, clip_lnM[0], clip_lnM[1])

    # Derivative of lnM with respect to lnR along the R dimension
    # np.gradient supports non-uniform grids when coordinates are provided
    dlnM_dlnR = np.gradient(lnM, lnR, axis=1, edge_order=2)

    # Relative risk aversion
    RRA = -dlnM_dlnR

    # Clean up numerical junk
    RRA = np.where(np.isfinite(RRA), RRA, np.nan)

    return RRA
def compute_rra_for_all_dates(
    pricing_kernel_information: Dict[str, Dict[str, Any]],
    *,
    date: Optional[str] = None,                 # NEW: run only this date if provided
    strict: bool = True,                        # NEW: raise if date not found
    out_dir: Optional[str] = None,
    make_plots: bool = True,
    window: Tuple[float, float] = (0.90, 1.10),
    use_p_weights_if_available: bool = True,
    winsor_quantiles: Optional[Tuple[float, float]] = (0.01, 0.99),
    surface_zlim: Optional[Tuple[float, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes RRA + gamma(T) objects for each date in `pricing_kernel_information`.

    If `date` is provided, computes results ONLY for that date key.
    """
    result: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------
    # NEW: choose which date keys to process
    # ------------------------------------------------------------
    if date is None:
        items = pricing_kernel_information.items()
    else:
        if date not in pricing_kernel_information:
            msg = f"Requested date='{date}' not found. Available keys (sample): {list(pricing_kernel_information)[:10]}"
            if strict:
                raise KeyError(msg)
            else:
                print("Warning:", msg)
                return {}
        items = [(date, pricing_kernel_information[date])]

    # ------------------------------------------------------------
    # main loop (unchanged logic)
    # ------------------------------------------------------------
    for date_key, pack in items:
        if "R_common" not in pack or "T_anchor" not in pack or "anchor_surfaces" not in pack:
            continue

        R = np.asarray(pack["R_common"], float)
        T = np.asarray(pack["T_anchor"], float)
        anchor = pack["anchor_surfaces"]

        if "M_surface" not in anchor:
            continue

        M = np.asarray(anchor["M_surface"], float)
        pR = np.asarray(anchor["pR_surface"], float) if ("pR_surface" in anchor) else None

        # 1) RRA surface
        RRA = compute_rra_surface_from_M(M, R)

        # 2) gamma(T) by p-weighted average of RRA in window
        gamma_T = compute_gamma_by_maturity(
            RRA,
            R,
            pR_surface=pR,
            window=window,
            use_p_weights_if_available=use_p_weights_if_available,
            winsor_quantiles=winsor_quantiles,
        )

        # 3) gamma(T) by parametric regressions (Power / Exponential)
        param = compute_gamma_power_exponential_by_maturity(
            M,
            R,
            pR_surface=pR,
            window=window,
            use_p_weights_if_available=use_p_weights_if_available,
        )

        out_pack: Dict[str, Any] = {
            "date": date_key,
            "R_common": R,
            "T_anchor": T,
            "RRA_surface": RRA,
            # Option (1)
            "gamma_by_T": gamma_T,
            # Option (2) Power regression
            "gamma_power_by_T": param["gamma_power"],
            "alpha_power_by_T": param["alpha_power"],
            # Option (3) Exponential regression
            "a_exponential_by_T": param["a_exponential"],
            "alpha_exponential_by_T": param["alpha_exponential"],
            # meta
            "gamma_window": window,
            "gamma_weighting": ("pR" if (use_p_weights_if_available and pR is not None) else "window_mean"),
            "winsor_quantiles": winsor_quantiles,
        }

        # Plot/save (same as before)
        if make_plots:
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                save_surf = os.path.join(out_dir, f"RRA_surface_{date_key}.png")
                save_gam = os.path.join(out_dir, f"Gamma_by_T_{date_key}.png")
            else:
                save_surf = None
                save_gam = None

            plot_rra_surface(
                RRA, R, T,
                title=f"RRA Surface {date_key}",
                save_path=save_surf,
                zlim=surface_zlim,
            )
            plot_gamma_curve(
                gamma_T, T,
                title=f"Gamma(T) (p-weighted RRA avg) {date_key}",
                save_path=save_gam,
            )

            out_pack["plot_paths"] = {"rra_surface": save_surf, "gamma_curve": save_gam}

        result[date_key] = out_pack

    return result


# ============================================================
# NEW: Timeseries plotter for power gamma and exponential a too
# ============================================================
def plot_param_timeseries(
    rra_dict: Dict[str, Dict[str, Any]],
    *,
    param_key: str,
    T_targets_days: Optional[Sequence[float]] = None,
    T_targets_years: Optional[Sequence[float]] = None,
    tol_days: float = 3.0,
    T_key: str = "T_anchor",
    out_dir: Optional[str] = None,
    fname_prefix: str = "param_timeseries",
    show: bool = True,
    figsize: Tuple[float, float] = (11, 5),
    verbose: bool = True,
    ylabel: Optional[str] = None,
):
    """
    Generic time-series plotter for any per-maturity parameter stored in rra_dict[date][param_key].

    Examples:
      param_key="gamma_power_by_T"
      param_key="a_exponential_by_T"
      param_key="gamma_by_T"
    """
    if (T_targets_days is None) == (T_targets_years is None):
        raise ValueError("Provide exactly one of T_targets_days or T_targets_years.")

    date_keys = sorted(rra_dict.keys())
    if len(date_keys) == 0:
        raise ValueError("rra_dict is empty.")

    dates = np.array([np.datetime64(k) for k in date_keys])

    if T_targets_days is not None:
        T_targets_days = np.asarray(list(T_targets_days), float)
    else:
        T_targets_days = np.asarray(list(T_targets_years), float) * 365.0

    n_targets = T_targets_days.size
    n_dates = len(date_keys)

    ts = np.full((n_targets, n_dates), np.nan, float)

    n_missing = 0
    n_badshape = 0

    for j, k in enumerate(date_keys):
        pack = rra_dict.get(k, {})
        if (T_key not in pack) or (param_key not in pack):
            n_missing += 1
            continue

        T = np.asarray(pack[T_key], float)
        v = np.asarray(pack[param_key], float)

        if T.ndim != 1 or v.ndim != 1 or T.size != v.size or T.size < 2:
            n_badshape += 1
            continue

        T_days = T * 365.0

        for i, Td in enumerate(T_targets_days):
            idx = int(np.argmin(np.abs(T_days - Td)))
            if float(np.abs(T_days[idx] - Td)) <= tol_days:
                ts[i, j] = float(v[idx])

    if verbose:
        ok = np.isfinite(ts).sum()
        print(f"[plot_param_timeseries:{param_key}] filled points={ok} / {n_targets*n_dates}")
        if n_missing:
            print(f"[plot_param_timeseries:{param_key}] dates missing keys = {n_missing}")
        if n_badshape:
            print(f"[plot_param_timeseries:{param_key}] dates with bad shapes = {n_badshape}")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for i, Td in enumerate(T_targets_days):
        ax.plot(dates, ts[i, :], label=f"T≈{Td:.0f}d")

    ax.set_title(f"Time Series: {param_key}")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel if ylabel is not None else param_key)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=min(4, n_targets), frameon=False)
    fig.autofmt_xdate()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{fname_prefix}_{param_key}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"dates": dates, "T_targets_days": T_targets_days, "ts": ts, "date_keys": date_keys}
def _trapz_weights(x: np.ndarray) -> np.ndarray:
    """
    Trapezoid integration weights so that:
        sum_i w[i] * f[i] ≈ ∫ f(x) dx
    Works for non-uniform grids.
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return np.zeros_like(x)

    dx = np.diff(x)
    w = np.zeros(n, float)
    w[0] = dx[0] / 2.0
    w[-1] = dx[-1] / 2.0
    if n > 2:
        w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    return w


def _safe_normalize_pdf(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Ensure f is a valid PDF on grid x:
      - drop non-finite values
      - enforce non-negativity
      - normalize to integrate to 1
    """
    x = np.asarray(x, float)
    f = np.asarray(f, float)

    f = np.where(np.isfinite(f), f, 0.0)
    f = np.maximum(f, 0.0)

    mass = np.trapz(f, x)
    if (not np.isfinite(mass)) or mass <= 0:
        return np.zeros_like(f)

    return f / mass

def plot_rra_surface(
    RRA_surface: np.ndarray,
    R_grid: np.ndarray,
    T_grid: np.ndarray,
    *,
    title: str = "RRA Surface",
    save_path: Optional[str] = None,
    dpi: int = 200,
    zlim: Optional[Tuple[float, float]] = None,
):
    """
    Plot the RRA(R,T) surface as a 3D plot.

    Parameters
    ----------
    RRA_surface : np.ndarray
        Relative risk aversion surface, shape (nT, nR).
    R_grid : np.ndarray
        Gross return grid, shape (nR,).
    T_grid : np.ndarray
        Maturity grid in years, shape (nT,).
    title : str
        Plot title.
    save_path : str or None
        If provided, save figure to this path.
    dpi : int
        Resolution for saved figure.
    zlim : tuple(float, float) or None
        Optional z-axis limits (min, max).
    """

    RRA = np.asarray(RRA_surface, float)
    R = np.asarray(R_grid, float)
    T = np.asarray(T_grid, float)

    if RRA.ndim != 2:
        raise ValueError("RRA_surface must be 2D (nT, nR).")
    if RRA.shape != (T.size, R.size):
        raise ValueError("RRA_surface shape must match (len(T_grid), len(R_grid)).")

    # Create mesh
    RR, TT = np.meshgrid(R, T)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        RR, TT, RRA,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel("Gross return R")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("RRA(R,T)")
    ax.set_title(title)

    if zlim is not None:
        ax.set_zlim(zlim[0], zlim[1])

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)

    plt.tight_layout()

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()

def plot_gamma_curve(
    gamma_T: np.ndarray,
    T_grid: np.ndarray,
    *,
    title: str = "Coefficient of Risk Aversion vs Maturity",
    save_path: Optional[str] = None,
    dpi: int = 200,
):
    """
    Plot gamma(T) as a function of maturity.

    Parameters
    ----------
    gamma_T : np.ndarray
        Risk aversion by maturity, shape (nT,).
    T_grid : np.ndarray
        Maturity grid in years, shape (nT,).
    title : str
        Plot title.
    save_path : str or None
        If provided, save figure to this path.
    dpi : int
        Resolution for saved figure.
    """

    g = np.asarray(gamma_T, float)
    T = np.asarray(T_grid, float)

    if g.ndim != 1:
        raise ValueError("gamma_T must be 1D.")
    if T.ndim != 1 or T.size != g.size:
        raise ValueError("T_grid must be 1D and match gamma_T length.")

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111)

    ax.plot(T, g, linewidth=2.0)
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Gamma(T)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, Sequence, Tuple


import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, Tuple


def plot_rra_panel_from_dict(
    rra_dict: Dict[str, Dict[str, Any]],
    date: Optional[str] = None,
    n_curves: int = 12,
    which_T: str = "even",      # "even" or "random"
    random_state: Optional[int] = None,
    R_bounds: Optional[Tuple[float, float]] = None,
    clip_y: Optional[Tuple[float, float]] = None,
    xscale: str = "linear",     # NEW: "linear" or "log"
    yscale: str = "linear",     # NEW: "linear" or "log"
    ncols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    lw: float = 1.6,
    alpha: float = 0.95,
    save: Optional[Union[str, os.PathLike]] = None,
    dpi: int = 200,
):
    """
    Panel plot of RRA(R) curves from rra_dict[date]["RRA_surface"].

    Supports log scaling:
      - xscale="log": log-moneyness / log-return view
      - yscale="log": emphasizes tail explosions (positive RRA only)
    """

    keys = sorted(rra_dict.keys())
    if date is None:
        date = keys[0]
    d = rra_dict[date]

    RRA = np.asarray(d["RRA_surface"], float)
    R = np.asarray(d["R_common"], float)

    nT, nR = RRA.shape

    # Optional maturity grid
    T = None
    for k in ("T_anchor", "T_common", "T_grid", "T"):
        if k in d and d[k] is not None:
            T = np.asarray(d[k], float)
            if T.ndim == 1 and T.size == nT:
                break
            T = None

    # Choose maturities
    n_curves = min(n_curves, nT)
    if which_T == "random":
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(nT, n_curves, replace=False))
    else:
        idx = np.unique(np.round(np.linspace(0, nT - 1, n_curves)).astype(int))

    # R filtering
    mask = np.isfinite(R)
    if R_bounds is not None:
        mask &= (R >= R_bounds[0]) & (R <= R_bounds[1])

    R_plot = R[mask]

    if figsize is None:
        nrows = int(math.ceil(len(idx) / ncols))
        figsize = (4.2 * ncols, 3.2 * nrows)

    fig, axes = plt.subplots(
        nrows=int(math.ceil(len(idx) / ncols)),
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for j, iT in enumerate(idx):
        ax = axes_flat[j]

        y = RRA[iT, mask]

        # Log-scale handling for y
        if yscale == "log":
            pos = y > 0
            ax.plot(R_plot[pos], y[pos], lw=lw, alpha=alpha)
        else:
            ax.plot(R_plot, y, lw=lw, alpha=alpha)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if clip_y is not None:
            ax.set_ylim(*clip_y)

        if T is not None:
            ax.set_title(f"T ≈ {365*T[iT]:.0f}d", fontsize=10)
        else:
            ax.set_title(f"row {iT}", fontsize=10)

        ax.set_xlabel("R")
        ax.set_ylabel("RRA(R)")
        ax.grid(True, alpha=0.3)

    # Turn off unused panels
    for k in range(len(idx), len(axes_flat)):
        axes_flat[k].axis("off")

    if title is None:
        title = f"RRA(R) panel — {date}"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save is not None:
        save = os.fspath(save)
        if os.path.isdir(save):
            save = os.path.join(save, f"rra_panel_{date.replace('-', '_')}.png")
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")

    return fig, axes
# ============================================================
# QUICK SIMULATOR: fake stock path + fake RND surfaces
# Produces: information_dict, stock_df
# Then runs: fit theta -> evaluate -> plot
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1) Simulate stock_df (GBM-ish)
# ----------------------------
def simulate_stock_df(
    start="2021-01-01",
    n_days=260,
    S0=100.0,
    mu=0.06,
    sigma=0.20,
    seed=0,
):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    dt = 1.0 / 252.0
    eps = rng.normal(size=n_days)
    logS = np.log(S0) + np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps)
    S = np.exp(logS)
    return pd.DataFrame({"date": dates, "price": S})


# ----------------------------
# 2) Simulate RND surface per date:
#    - K_interp: strike grid
#    - T_interp: maturities (years)
#    - rnd_surface: shape (nT, nK), density in K-space integrating to 1
#
# We'll use a lognormal density for K with parameters tied to S0 and atm_vol.
# This is NOT "option-implied", just a coherent density surface for testing.
# ----------------------------
def lognormal_pdf(x, m, s):
    # pdf of LogNormal with log-mean m and log-std s
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    ok = x > 0
    z = (np.log(x[ok]) - m) / s
    out[ok] = np.exp(-0.5 * z**2) / (x[ok] * s * np.sqrt(2*np.pi))
    return out


def normalize_trapz(x, f):
    f = np.asarray(f, float)
    f = np.where(np.isfinite(f), f, 0.0)
    f = np.maximum(f, 0.0)
    mass = np.trapz(f, x)
    if not np.isfinite(mass) or mass <= 0:
        return np.zeros_like(f)
    return f / mass


def simulate_information_dict(
    stock_df,
    n_anchor_dates=80,
    nK=220,
    T_days=(30, 45, 60, 90, 120, 180),
    k_moneyness=(0.5, 1.6),
    base_atm_vol=0.22,
    vol_jitter=0.04,
    skew_strength=0.10,
    seed=1,
):
    rng = np.random.default_rng(seed)

    # choose a subset of dates to act as option-anchor dates
    all_dates = pd.to_datetime(stock_df["date"])
    pick = np.linspace(0, len(all_dates) - 40, n_anchor_dates, dtype=int)  # leave room for realized returns
    anchor_dates = all_dates.iloc[pick].dt.strftime("%Y-%m-%d").tolist()

    info = {}
    T_years = np.array(T_days, float) / 365.0

    for ds in anchor_dates:
        dt = pd.to_datetime(ds)
        S0 = float(stock_df.loc[stock_df["date"] == dt, "price"].iloc[0])

        # atm vol per date (positive)
        atm_vol = float(np.clip(base_atm_vol + vol_jitter * rng.normal(), 0.05, 0.80))

        # strike grid around spot (fixed moneyness range)
        K = np.linspace(k_moneyness[0]*S0, k_moneyness[1]*S0, nK)

        rnd_surface = np.zeros((T_years.size, K.size), float)

        # For each maturity, create a lognormal density in K-space:
        # K = S0 * exp(r), r ~ N(muT, sigT^2)
        # We'll add a mild maturity-dependent mean shift to mimic skew.
        for j, T in enumerate(T_years):
            sigT = max(1e-6, atm_vol * np.sqrt(T))
            # mild "skew" by shifting mean left as T grows
            muT = np.log(S0) - 0.5 * sigT**2 - skew_strength * (T / T_years.max())

            fK = lognormal_pdf(K, m=muT, s=sigT)
            fK = normalize_trapz(K, fK)
            rnd_surface[j, :] = fK

        info[ds] = {
            "S0": S0,
            "atm_vol": atm_vol,     # decimals
            "K_interp": K,
            "T_interp": T_years,
            "rnd_surface": rnd_surface,  # (nT, nK)
        }

    return info


# ----------------------------
# 3) Run your estimator on simulated data
# ----------------------------

if __name__ == "__main__":
    stock_df = simulate_stock_df(start="2021-01-01", n_days=320, S0=100, mu=0.07, sigma=0.22, seed=42)
    information_dict = simulate_information_dict(stock_df, n_anchor_dates=90, seed=7)
    
    print("Simulated stock_df:", stock_df.shape)
    print("Simulated information_dict dates:", len(information_dict))
    
    # Instantiate your estimator (class must already be defined above)
    est = PricingKernelSurfaceEstimator(information_dict=information_dict, stock_df=stock_df)
    
    # Fit theta cache
    theta_cache = est.fit_theta_master_grid(
        N=2,
        Ksig=1,
        tol_days=5,
        day_step=1,
        min_obs_per_T=10,
        r_grid_size=350,
        verbose=True,
        use_vix=False,
        multistart=True,
        n_random_starts=1,
    )
    
    # Pick an anchor date (middle of sample) and evaluate
    anchor_date = sorted(information_dict.keys())[len(information_dict)//2]
    out = est.evaluate_surfaces_for_date_master(anchor_date, theta_cache, tol_days_eval=5, warn=True)
    
    print("\nAnchor used:", out["anchor_key_used"])
    print("Theta eval summary:", out["theta_eval_summary"])
    
    # Plot kernel surface + physical density surface (zoom near ATM)
    est.plot_pricing_kernel_surface(R_bounds=(0.85, 1.15), T_bounds=(30/365, 180/365))
    est.plot_physical_density_surface(R_bounds=(0.85, 1.15), T_bounds=(30/365, 180/365))
    
    # Panel (show kernel only inside R window)
    est.plot_panel(
        panel_shape=(2, 4),
        trunc_mode="rbounds",
        r_bounds=(0.90, 1.10),
        truncate=True,
    )
    
    # Compute + plot RRA + gamma
    R = np.asarray(out["R_common"], float)
    T = np.asarray(out["T_anchor"], float)
    M = np.asarray(out["anchor_surfaces"]["M_surface"], float)
    pR = np.asarray(out["anchor_surfaces"]["pR_surface"], float)
    
    RRA = compute_rra_surface_from_M(M, R)
    gamma_T = compute_gamma_by_maturity(RRA, R, pR_surface=pR, window=(0.90, 1.10))
    
    plot_rra_surface(RRA, R, T, title=f"RRA Surface (SIM) {out['anchor_key_used']}")
    plot_gamma_curve(gamma_T, T, title=f"Gamma(T) (SIM) {out['anchor_key_used']}")
