import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from Mixture_LWD import *   # assumes MixtureSpec, FittedMixture, fit_mixture_to_calls,
                            # evolutionary_lwm_fit, mixture_pdf, mixture_cdf, etc.
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import pandas as pd

# -------------------------------------------------------------------
# You already defined (in Mixture_LWD.py or above this block):
#   - MixtureSpec
#   - FittedMixture
#   - fit_mixture_to_calls
#   - evolutionary_lwm_fit
#   - mixture_pdf, mixture_cdf
# -------------------------------------------------------------------


# ============================================================
# Black–Scholes helpers (for IV surface)
# ============================================================

def bs_call_price(S0, K, r, sigma, T):
    """Plain Black–Scholes European call (present value)."""
    from scipy.stats import norm

    K = np.asarray(K, float)
    if T <= 0:
        return np.maximum(S0 - K, 0.0)

    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S0, K, r, sigma, T):
    from scipy.stats import norm
    K = np.asarray(K, float)
    if T <= 0:
        return np.zeros_like(K)
    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / vol_sqrtT
    return S0 * np.sqrt(T) * norm.pdf(d1)


def bs_implied_vol(C, S0, K, r, T, tol=1e-6, max_iter=50):
    """
    Scalar implied vol via Newton–Raphson with vega safeguard.
    Returns np.nan if fails or if price is out of arbitrage bounds.
    """
    K = float(K)
    C = float(C)
    # Arbitrage bounds
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    upper = S0
    if not (intrinsic <= C <= upper):
        return np.nan

    # Initial guess
    sigma = 0.2
    for _ in range(max_iter):
        C_model = bs_call_price(S0, K, r, sigma, T)
        diff = C_model - C
        if abs(diff) < tol:
            return sigma
        v = bs_vega(S0, K, r, sigma, T)
        if v < 1e-8:
            break
        sigma -= diff / v
        if sigma <= 0:
            sigma = 0.01
    return np.nan


# ============================================================
# (Optional) strike-direction no-arbitrage cleaner
# (kept here for manual use; NOT called in the main pipeline)
# ============================================================

def clean_strike_slice_with_nans(strikes, C_row, S0, r, T):
    """
    Clean a single-maturity call curve that may contain NaN entries.
    NaN entries are ignored and left unchanged.

    Enforces (on non-NaN entries):
      - Intrinsic <= C(K) <= S0
      - Monotone non-increasing in K
      - Convex in K

    Implemented as constrained LS (SLSQP), hard inequality constraints.
    """
    K = np.asarray(strikes, float)
    C0 = np.asarray(C_row, float)

    # Valid data mask
    valid = ~np.isnan(C0)
    idx = np.where(valid)[0]
    m = len(idx)

    if m <= 2:
        # Not enough points to enforce shape
        return C0.copy()

    K_valid = K[idx]
    C_valid = C0[idx]

    # Bounds for known points: intrinsic value <= C <= S0
    bounds = []
    for j in range(m):
        intrinsic = max(S0 - K_valid[j], 0.0)
        bounds.append((intrinsic, S0))

    # Objective: LS fit to known prices
    def obj(x):
        return 0.5 * np.sum((x - C_valid) ** 2)

    def grad(x):
        return x - C_valid

    constraints = []

    # Monotonicity: C(K_a) >= C(K_{a+1})  =>  x[a] - x[a+1] >= 0
    for a in range(m - 1):
        constraints.append({
            "type": "ineq",
            "fun": lambda x, a=a: x[a] - x[a + 1]
        })

    # Convexity: x[a+1] - 2*x[a] + x[a-1] >= 0
    for a in range(1, m - 1):
        constraints.append({
            "type": "ineq",
            "fun": lambda x, a=a: x[a + 1] - 2 * x[a] + x[a - 1]
        })

    res = minimize(
        fun=obj,
        x0=C_valid,
        jac=grad,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-9}
    )

    if not res.success:
        print("Warning in slice cleaner:", res.message)

    C_clean = C0.copy()
    C_clean[idx] = res.x
    return C_clean


# ============================================================
# Mixture-based call surface + IV + RND pipeline
# ============================================================

@dataclass
class MixtureSurfaceConfig:
    # Mixture composition (used when use_evolutionary=False)
    n_lognormal: int = 3
    n_weibull: int = 0

    # Variance constraint params for mixture fit
    var_c: float = 0.1          # variance constraint c
    var_penalty: float = 1e4    # variance penalty strength

    # Optimization
    random_starts: int = 3
    seed: int = 123

    # Stage 1: full estimation vs NaN-only fill
    full_estimation: bool = False  # if True, replace *all* values by mixture prices

    # Strike refinement when full_estimation=True
    fine_strike_factor: int = 3    # evaluate mixture on factor * original #strikes

    # Strike extension percentage (e.g. 0.4 = ±40%) when full_estimation=True
    strike_extension: float = 0.4

    # ------- NEW: evolutionary vs fixed mixture selection -------
    use_evolutionary: bool = False   # False = fixed (fit_mixture_to_calls), True = evolutionary_lwm_fit
    M_max: int = 5                   # max total mixture components for evolutionary search
    use_wald: bool = True
    wald_alpha: float = 0.05
    wald_p: int = 3
    wald_q: int = 3

    # Stage 2: maturity interpolation
    use_maturity_interp: bool = True
    day_step: int = 1           # maturity interpolation step (days)


class MixtureCallSurfaceCleaner:
    """
    Pipeline:

    1) For each maturity T_i (original grid):
        - Fit a mixture model to C(T_i, K_obs), either:
          * Fixed spec via fit_mixture_to_calls (n_lognormal, n_weibull), or
          * Evolutionary spec selection via evolutionary_lwm_fit (if
            config.use_evolutionary = True).
        - Evaluate mixture on a strike grid:
            * If full_estimation=False:
                - Strike grid = original strikes.
                - Only NaNs are replaced by mixture prices; original quotes
                  are left untouched.
            * If full_estimation=True:
                - Strike grid = refined "thin" grid, extended in both directions
                  by `strike_extension` (e.g. ±40%).
                - Entire row is replaced by mixture-implied prices.

    2) If use_maturity_interp=True:
        - For each strike K_j (on the current strike grid), fit a cubic spline
          in T through the *stage-1* surface.
        - Evaluate on a 1-day grid between min(T) and max(T).
        - At original maturities, we **restore** the stage-1 values (so spline
          never overwrites mixture-generated or original data).

    3) Re-fit a mixture on each interpolated maturity slice (stage-3), again
       using your mixture code, to get a smooth multi-curve mixture surface.

    4) NO final arbitrage check is applied in this version (per your request).

    5) Compute:
        - IV surface via Black–Scholes implied vol from the final call surface.
        - RND surface from the final calls via Breeden–Litzenberger.
    """

    def __init__(self,
                 strikes: np.ndarray,
                 maturities: np.ndarray,
                 S0: float,
                 r: float,
                 config: Optional[MixtureSurfaceConfig] = None):
        # Original observation strike grid
        self.strikes_obs = np.asarray(strikes, float)
        self.strikes = self.strikes_obs.copy()   # current evaluation grid
        self.maturities = np.asarray(maturities, float)
        self.S0 = float(S0)
        self.r = float(r)
        self.config = config or MixtureSurfaceConfig()

        # Mixture spec (used only when use_evolutionary=False)
        self.spec = MixtureSpec(
            n_lognormal=self.config.n_lognormal,
            n_weibull=self.config.n_weibull
        )

        # Placeholders filled by fit_surface()
        self.C_stage1 = None         # mixture-filled on original T grid
        self.T_interp = None         # final maturity grid
        self.C_interp = None         # after maturity spline
        self.C_mixture_final = None  # mixture re-fit on T_interp
        self.C_clean = None          # alias of final surface used for IV/RND
        self.iv_surface = None       # implied vol surface on (T_interp, strikes)
        self.rnd_surface = None      # RND from C_clean
        self.rnd_maturities = None   # maturities for RND (T_interp)

    # ---------- Stage 1: mixture per original maturity ----------

    def _fit_mixture_row(self, K_row, C_row, T, seed_shift=0) -> Optional[np.ndarray]:
        """
        Fit mixture to one maturity slice and return model-implied calls
        on the *current* strike grid self.strikes.

        Uses either:
          - Fixed MixtureSpec (fit_mixture_to_calls) if config.use_evolutionary=False
          - evolutionary_lwm_fit (metric='loss') if config.use_evolutionary=True

        Returns None if too few valid points.
        """
        K_row = np.asarray(K_row, float)
        C_row = np.asarray(C_row, float)
        mask = ~np.isnan(C_row)
        if mask.sum() < 4:
            # Too few quotes to fit a multi-component mixture sensibly
            return None

        K_valid = K_row[mask]
        C_valid = C_row[mask]

        # ---------- Fixed mixture mode ----------
        if not self.config.use_evolutionary:
            fit, _, _ = fit_mixture_to_calls(
                K=K_valid,
                C_mkt=C_valid,
                S0=self.S0,
                r=self.r,
                T=T,
                spec=self.spec,
                theta0=None,
                penalty_lambda=0.0,
                random_starts=self.config.random_starts,
                seed=self.config.seed + seed_shift,
                rnd_true=None,
                k_true=None,
                var_c=self.config.var_c,
                var_penalty=self.config.var_penalty,
                return_theta=False,
            )

        # ---------- Evolutionary mode ----------
        else:
            fit, chosen_spec = evolutionary_lwm_fit(
                K=K_valid,
                C_mkt=C_valid,
                S0=self.S0,
                r=self.r,
                T=T,
                M_max=self.config.M_max,
                penalty_lambda=0.0,
                random_starts=self.config.random_starts,
                seed=self.config.seed + seed_shift,
                var_c=self.config.var_c,
                var_penalty=self.config.var_penalty,
                improvement_tol=1e-4,
                metric="loss",        # use pricing loss; no rnd_true/k_true in real data
                rnd_true=None,
                k_true=None,
                use_wald=self.config.use_wald,
                wald_alpha=self.config.wald_alpha,
                wald_p=self.config.wald_p,
                wald_q=self.config.wald_q,
                weights=None,
                fixed_M=None,
                fixed_M1=None,
            )
            # (If you ever want to inspect the chosen_spec per slice, you can
            #  store it somewhere here.)

        # Use chat() to get discounted calls on *current* strike grid
        C_hat_full = fit.chat(self.strikes, r=self.r, T=T)
        return C_hat_full

    def stage1_mixture_fill(self, C_surface: np.ndarray) -> np.ndarray:
        """
        For each maturity row, fit mixture and interpolate/extrapolate across strikes.

        If full_estimation=False:
            - Strike grid = original grid self.strikes_obs.
            - Keep original quotes where not NaN, only fill NaNs.
        If full_estimation=True:
            - Strike grid = refined thin grid, extended in both directions by
              `strike_extension`.
            - Replace entire row by mixture-implied prices.
        """
        C_surface = np.asarray(C_surface, float)
        nT, nK_obs = C_surface.shape

        if self.config.full_estimation:
            # Define a refined, *extended* strike grid
            nK_fine = max(self.config.fine_strike_factor * nK_obs, nK_obs)

            low = float(self.strikes_obs.min())
            high = float(self.strikes_obs.max())
            ext = max(self.config.strike_extension, 0.0)

            # Extend endpoints multiplicatively by ±ext
            K_min_ext = low * (1.0 - ext)
            K_max_ext = high * (1.0 + ext)

            # Guard against non-positive lower strikes
            if K_min_ext <= 0:
                K_min_ext = max(low * 0.5, 1e-6)

            self.strikes = np.linspace(K_min_ext, K_max_ext, nK_fine)
            C_stage1 = np.empty((nT, nK_fine), dtype=float)
        else:
            # Stay on original strike grid
            self.strikes = self.strikes_obs.copy()
            C_stage1 = np.array(C_surface, copy=True, dtype=float)

        for i, T in enumerate(self.maturities):
            row = C_surface[i, :]
            C_hat = self._fit_mixture_row(self.strikes_obs, row, T, seed_shift=i)
            if C_hat is None:
                # Not enough data; keep what we have
                if self.config.full_estimation:
                    # Fallback: simple interpolation from whatever quotes exist
                    mask = ~np.isnan(row)
                    if mask.sum() >= 2:
                        C_stage1[i, :] = np.interp(
                            self.strikes,
                            self.strikes_obs[mask],
                            row[mask]
                        )
                    elif mask.sum() == 1:
                        C_stage1[i, :] = row[mask][0]
                    else:
                        C_stage1[i, :] = np.nan
                else:
                    # Just keep the original coarse row (with NaNs)
                    continue
            else:
                if self.config.full_estimation:
                    # Entire row is mixture prices on extended/thin grid
                    C_stage1[i, :] = C_hat
                else:
                    # Only fill NaNs on the original grid
                    mask_nan = np.isnan(row)
                    C_stage1[i, mask_nan] = C_hat[mask_nan]

        self.C_stage1 = C_stage1
        return C_stage1

    # ---------- Stage 2: spline across maturities ----------

    def stage2_maturity_spline(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cubic spline across maturities for each strike.

        - We fit a spline in T through whatever Stage 1 produced.
        - On new maturity grid T_interp, we evaluate the spline.
        - At original maturities, we *restore* the Stage-1 values
          (so mixture-generated/original quotes are never overwritten).
        """
        C_stage1 = np.asarray(self.C_stage1, float)
        if not self.config.use_maturity_interp:
            self.T_interp = self.maturities.copy()
            self.C_interp = C_stage1
            return self.T_interp, self.C_interp

        T_orig = self.maturities
        T_min = float(T_orig.min())
        T_max = float(T_orig.max())
        dt_years = self.config.day_step / 365.0

        n_steps = int(np.floor((T_max - T_min) / dt_years)) + 1
        T_interp = T_min + dt_years * np.arange(n_steps)
        if T_interp[-1] < T_max:
            T_interp = np.append(T_interp, T_max)

        nK = self.strikes.size
        C_interp = np.empty((T_interp.size, nK), dtype=float)

        # For each strike, spline in T, then overwrite original maturities
        for j in range(nK):
            y = C_stage1[:, j]
            mask_valid = ~np.isnan(y)

            if mask_valid.sum() == 0:
                # All NaNs
                C_interp[:, j] = np.nan
                continue
            elif mask_valid.sum() == 1:
                # Only one point: flat in time
                C_interp[:, j] = y[mask_valid][0]
            else:
                # Fit spline on valid points only
                T_valid = T_orig[mask_valid]
                y_valid = y[mask_valid]
                cs = CubicSpline(T_valid, y_valid, bc_type='natural')
                col = cs(T_interp)

                # Restore Stage-1 values at original maturities where y is not NaN
                for i_orig, T0 in enumerate(T_orig):
                    if np.isnan(y[i_orig]):
                        continue
                    k = np.argmin(np.abs(T_interp - T0))
                    col[k] = y[i_orig]

                C_interp[:, j] = col

        self.T_interp = T_interp
        self.C_interp = C_interp
        return T_interp, C_interp

    # ---------- Stage 3: re-fit mixture on interpolated grid ----------

    def stage3_refit_mixture_on_interp(self) -> np.ndarray:
        """
        For each maturity in T_interp, re-fit mixture to the interpolated
        slice and get model-implied calls. This yields a fully parametric
        multi-curve mixture surface on (T_interp, strikes).
        """
        assert self.T_interp is not None and self.C_interp is not None
        C_interp = self.C_interp
        T_interp = self.T_interp
        nT_new, _ = C_interp.shape

        C_mix_final = np.empty_like(C_interp)
        for i, T in enumerate(T_interp):
            row = C_interp[i, :]
            C_hat = self._fit_mixture_row(self.strikes, row, T, seed_shift=1000 + i)
            if C_hat is None:
                C_mix_final[i, :] = row
            else:
                C_mix_final[i, :] = C_hat

        self.C_mixture_final = C_mix_final
        return C_mix_final

    # ---------- RND from final calls via Breeden–Litzenberger ----------

    def compute_rnd_surface(self) -> np.ndarray:
        """
        Compute RND q_T(K) = exp(r T) * d^2 C / dK^2 for each maturity
        from the FINAL call surface (C_clean alias).
        """
        assert self.C_clean is not None
        C = self.C_clean
        T_grid = self.T_interp
        K = self.strikes

        nT, nK = C.shape
        rnd = np.empty_like(C)

        for i, T in enumerate(T_grid):
            dC_dK = np.gradient(C[i, :], K, edge_order=2)
            d2C_dK2 = np.gradient(dC_dK, K, edge_order=2)
            rnd[i, :] = np.exp(self.r * T) * d2C_dK2

        self.rnd_surface = rnd
        self.rnd_maturities = T_grid
        return rnd

    # ---------- IV surface from final calls ----------

    def compute_iv_surface(self) -> np.ndarray:
        """
        Compute Black–Scholes implied vol for each (T, K)
        from the FINAL call surface (C_clean alias).
        """
        assert self.C_clean is not None
        C = self.C_clean
        T_grid = self.T_interp
        K_grid = self.strikes

        nT, nK = C.shape
        iv = np.empty_like(C)

        for i, T in enumerate(T_grid):
            for j, K in enumerate(K_grid):
                iv[i, j] = bs_implied_vol(C[i, j], self.S0, K, self.r, T)

        self.iv_surface = iv
        return iv

    # ---------- Master method ----------

    def fit_surface(self, C_surface: np.ndarray):
        """
        Run the full pipeline on an initial C_surface(T,K_obs).

        Returns a dict with:
            - 'T_interp'
            - 'C_stage1'
            - 'C_interp'
            - 'C_mixture_final'
            - 'C_clean'      (alias of final mixture surface)
            - 'iv_surface'
            - 'rnd_surface'
        """
        # Stage 1: mixture across strikes per original maturity
        self.stage1_mixture_fill(C_surface)

        # Stage 2: spline across maturities
        self.stage2_maturity_spline()

        # Stage 3: re-fit mixture on interpolated grid
        self.stage3_refit_mixture_on_interp()

        # NO final arbitrage cleaner (per your last instruction).
        # Just define C_clean as the final mixture surface:
        self.C_clean = self.C_mixture_final

        # IV + RND based on C_clean
        self.compute_iv_surface()
        self.compute_rnd_surface()

        return {
            "T_interp": self.T_interp,
            "C_stage1": self.C_stage1,
            "C_interp": self.C_interp,
            "C_mixture_final": self.C_mixture_final,
            "C_clean": self.C_clean,
            "iv_surface": self.iv_surface,
            "rnd_surface": self.rnd_surface,
        }

    # ---------- Plotting: call surface, IV surface, RND ----------

    def plot_call_and_iv_surfaces(self):
        """
        3D plots of final call surface (C_clean) and IV surface.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        assert self.C_clean is not None and self.iv_surface is not None

        K = self.strikes
        T = self.T_interp
        KK, TT = np.meshgrid(K, T)

        fig = plt.figure(figsize=(14, 5))

        # Call surface
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title("Final call surface C_clean(T, K)")
        ax1.plot_surface(KK, TT, self.C_clean, rstride=1, cstride=1, linewidth=0.2)
        ax1.set_xlabel("Strike K")
        ax1.set_ylabel("Maturity T")
        ax1.set_zlabel("Call price")

        # IV surface
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title("Implied volatility surface σ(T, K)")
        ax2.plot_surface(KK, TT, self.iv_surface, rstride=1, cstride=1, linewidth=0.2)
        ax2.set_xlabel("Strike K")
        ax2.set_ylabel("Maturity T")
        ax2.set_zlabel("Implied vol")

        plt.tight_layout()
        plt.show()

    def plot_some_rnds_panel(self, n_curves: int = 3):
        """
        Plot selected RND curves q_T(K) in separate panels rather than overlaid.

        Parameters
        ----------
        n_curves : int
            Number of RND slices (maturities) to plot.
            They will be chosen as evenly spaced indices across maturities.
        """
        assert self.rnd_surface is not None, "Compute RND surface first."

        rnd = self.rnd_surface
        T_grid = self.rnd_maturities
        K = self.strikes

        nT = T_grid.size
        if nT == 0:
            return

        # choose evenly spaced maturities
        idxs = np.linspace(0, nT - 1, min(n_curves, nT), dtype=int)

        # Create figure with one subplot per RND
        fig, axes = plt.subplots(
            nrows=len(idxs),
            ncols=1,
            figsize=(7, 3 * len(idxs)),
            sharex=True
        )

        if len(idxs) == 1:
            axes = [axes]

        for ax, idx in zip(axes, idxs):
            ax.plot(K, rnd[idx, :], lw=2)
            ax.set_ylabel("q_T(s)")
            ax.set_title(f"RND at T = {T_grid[idx]:.4f} yr")

        axes[-1].set_xlabel("Terminal price s = K")

        plt.tight_layout()
        plt.show()

    def plot_some_rnds(self, n_curves: int = 3):
        """
        Plot RNDs q_T(K) for a subset of maturities from the final surface.
        """
        assert self.rnd_surface is not None

        rnd = self.rnd_surface
        T_grid = self.rnd_maturities
        K = self.strikes

        nT = T_grid.size
        if nT == 0:
            return

        idxs = np.linspace(0, nT - 1, min(n_curves, nT), dtype=int)

        plt.figure(figsize=(7, 5))
        for idx in idxs:
            plt.plot(K, rnd[idx, :], label=f"T={T_grid[idx]:.3f} yr")

        plt.title("Risk-neutral densities from final calls")
        plt.xlabel("Terminal price s = K")
        plt.ylabel("q_T(s)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_rnd_surface(self):
        """
        3D surface plot of the full RND surface q_T(K).
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        assert self.rnd_surface is not None

        K = self.strikes
        T = self.rnd_maturities
        KK, TT = np.meshgrid(K, T)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Risk-neutral density surface q_T(K)")
        ax.plot_surface(KK, TT, self.rnd_surface, rstride=1, cstride=1, linewidth=0.2)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T")
        ax.set_zlabel("q_T(K)")
        plt.tight_layout()
        plt.show()


# ============================================================
# option_df → surface helper
# ============================================================

def extract_call_surface_from_df(
    option_df: pd.DataFrame,
    price_col: str = "mid_price",
    maturity_col: str = "rounded_maturity",
    right_col: str = "option_right",
    call_code: str = "c",
):
    """
    Helper function easily extracts information from option_df
    Build (strikes, maturities, C_arb, S0, r) from an option dataframe.

    Parameters
    ----------
    option_df : DataFrame
        Must contain at least:
            - 'strike'
            - `price_col`  (default 'mid_price')
            - `maturity_col` (default 'rounded_maturity', in years)
        Optionally:
            - 'option_right' (to filter calls)
            - 'underlying_price' or 'stock_price'
            - 'risk_free_rate' or 'risk_f'
    price_col : str
        Column with call prices (no-arb surface will be built from this).
    maturity_col : str
        Column with time-to-maturity in years (or anything numeric).
    right_col : str
        Column indicating call/put (if present).
    call_code : str
        Value in `right_col` that denotes calls.

    Returns
    -------
    strikes : 1D np.ndarray, shape (nK,)
    maturities : 1D np.ndarray, shape (nT,)
    C_arb : 2D np.ndarray, shape (nT, nK)
        Call price matrix; NaNs where (T, K) is missing.
    S0 : float
        Underlying level (median across rows).
    r : float
        Risk-free rate (median across rows).
    """
    df = option_df.copy()

    # Keep calls only if we have a right column
    if right_col in df.columns:
        df = df[df[right_col] == call_code]

    # Drop rows missing the key fields for the surface
    df = df.dropna(subset=["strike", maturity_col, price_col])

    # Sorted unique grids
    strikes = np.sort(df["strike"].unique())
    maturities = np.sort(df[maturity_col].unique())

    # Pivot into a maturity × strike matrix
    surface = df.pivot_table(
        index=maturity_col,
        columns="strike",
        values=price_col,
        aggfunc="mean"     # avg if multiple quotes per (T,K)
    )

    # Ensure full rectangular grid; missing combos become NaN
    surface = surface.reindex(index=maturities, columns=strikes)

    C_arb = surface.values.astype(float)

    # Underlying S0: use underlying_price if present, else stock_price
    if "underlying_price" in df.columns:
        S0 = float(df["underlying_price"].median())
    elif "stock_price" in df.columns:
        S0 = float(df["stock_price"].median())
    else:
        raise ValueError("No underlying price column found (need 'underlying_price' or 'stock_price').")

    # Risk-free rate: median across the dataframe
    if "risk_free_rate" in df.columns:
        r = float(df["risk_free_rate"].median())
    elif "risk_f" in df.columns:
        r = float(df["risk_f"].median())
    else:
        # If you prefer, change this to raise instead of defaulting to 0.
        r = 0.0

    return strikes, maturities, C_arb, S0, r


# ============================================================
# Comparison & curve-plot helpers
# ============================================================

def plot_original_vs_final_surface(cleaner,
                                   strikes_orig,
                                   maturities_orig,
                                   C_orig,
                                   title="Call Surface Comparison"):
    """
    Plot original call surface vs final fitted/cleaned call surface.

    Parameters
    ----------
    cleaner : MixtureCallSurfaceCleaner
        Must have run cleaner.fit_surface() already.

    strikes_orig : 1D array
        Original strike grid.

    maturities_orig : 1D array
        Original maturity grid.

    C_orig : 2D array
        Original call surface before smoothing / mixture / cleaning.
        Can be either shape (len(maturities_orig), len(strikes_orig))
        or the transpose; this function auto-detects.

    title : str
        Title printed above the figure (e.g. 'Chevron Call Surface on 2023-03-16')
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    strikes_orig = np.asarray(strikes_orig, float)
    maturities_orig = np.asarray(maturities_orig, float)
    C_orig = np.asarray(C_orig, float)

    nT = len(maturities_orig)
    nK = len(strikes_orig)

    # --- Fix orientation if needed ---
    if C_orig.shape == (nT, nK):
        C_plot = C_orig
    elif C_orig.shape == (nK, nT):
        # many libraries store as (K, T); transpose to (T, K)
        C_plot = C_orig.T
    else:
        raise ValueError(
            f"C_orig has shape {C_orig.shape}, "
            f"but expected ({nT}, {nK}) or ({nK}, {nT})."
        )

    # Extract final surface
    K_final = cleaner.strikes
    T_final = cleaner.T_interp
    C_final = cleaner.C_clean

    # Meshes for final surface
    KK_final, TT_final = np.meshgrid(K_final, T_final)

    # Mesh for original (used only for axis limits)
    KK_orig, TT_orig = np.meshgrid(strikes_orig, maturities_orig)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=18, y=0.97)

    # ---------------------------------------------------
    # 1) Original Call Surface (scatter of actual quotes)
    # ---------------------------------------------------
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Original Call Surface (observed quotes)")

    mask = ~np.isnan(C_plot)
    ax1.scatter(
        KK_orig[mask],
        TT_orig[mask],
        C_plot[mask],
        s=15,
        depthshade=True,
    )

    ax1.set_xlabel("Strike K")
    ax1.set_ylabel("Maturity T")
    ax1.set_zlabel("Call Price")

    # ---------------------------------------------------
    # 2) Final Estimated Surface
    # ---------------------------------------------------
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Final Estimated Call Surface\n(Mixture + Spline)")
    ax2.plot_surface(
        KK_final, TT_final, C_final,
        rstride=1, cstride=1, linewidth=0.2, alpha=0.9
    )
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Maturity T")
    ax2.set_zlabel("Call Price")

    plt.tight_layout()
    plt.show()

def plot_random_observed_vs_mixture_curve(
    cleaner,
    strikes_orig,
    maturities_orig,
    C_orig,
    n_curves: int = 3,
    random_state: int | None = None,
    title_prefix: str = "Observed vs Mixture call curve",
    plot_all_original: bool = False,
    plot_interpolated: bool = False,
):
    """
    Plot call curves from the MixtureCallSurfaceCleaner.

    Part 1: Original vs Stage-1 Mixture (on original maturity grid)
    ---------------------------------------------------------------
    For a subset (or all) maturities on the original grid, plot:
        - observed call curve C_orig(T_i, K_orig)
        - stage-1 mixture curve C_stage1(T_i, K_mixture)

    Part 2: Interpolated + Mixture (on T_interp grid, optional)
    -----------------------------------------------------------
    If plot_interpolated=True, in a SECOND figure we plot pure mixture curves
    from stage 3 (C_mixture_final) on the interpolated maturity grid T_interp.
    These do *not* necessarily have corresponding observed calls.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------- Part 1: original vs stage-1 mixture ----------

    # Ensure stage 1 exists
    assert cleaner.C_stage1 is not None, "Run cleaner.fit_surface() before calling this."
    C_stage1 = np.asarray(cleaner.C_stage1, float)

    strikes_orig = np.asarray(strikes_orig, float)
    maturities_orig = np.asarray(maturities_orig, float)
    C_orig = np.asarray(C_orig, float)

    # We assume the first rows correspond to the original maturities used
    nT_orig = min(C_orig.shape[0], C_stage1.shape[0])
    if nT_orig == 0:
        raise ValueError("No maturities available to plot on the original grid.")

    rng = np.random.default_rng(random_state)

    if plot_all_original:
        idxs_orig = np.arange(nT_orig)
    else:
        n_sel = max(1, min(n_curves, nT_orig))
        idxs_orig = rng.choice(nT_orig, size=n_sel, replace=False)
        idxs_orig = np.sort(idxs_orig)

    K_mix = cleaner.strikes  # strike grid used for mixture fits

    fig, axes = plt.subplots(
        nrows=len(idxs_orig), ncols=1,
        figsize=(6, 3 * len(idxs_orig)),
        sharex=False
    )
    if len(idxs_orig) == 1:
        axes = [axes]

    for ax, i in zip(axes, idxs_orig):
        T_obs = maturities_orig[i]

        # Observed curve at this maturity
        C_obs = C_orig[i, :]
        mask = ~np.isnan(C_obs)

        # Stage-1 mixture curve at this maturity (same row index, mixture strikes)
        C_mix_row = C_stage1[i, :]

        ax.plot(strikes_orig[mask], C_obs[mask], "o", label="Observed", ms=4)
        ax.plot(K_mix, C_mix_row, "-", label="Mixture (stage 1)", lw=2)

        ax.set_xlabel("Strike K")
        ax.set_ylabel("Call price")
        ax.set_title(f"{title_prefix} (T ≈ {T_obs:.3f} yr)")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # ---------- Part 2: interpolated + mixture-only curves (optional) ----------

    if plot_interpolated:
        assert cleaner.C_mixture_final is not None, (
            "cleaner.C_mixture_final is None. "
            "Make sure stage3_refit_mixture_on_interp() ran inside fit_surface()."
        )

        C_mix_interp = np.asarray(cleaner.C_mixture_final, float)
        T_interp = np.asarray(cleaner.T_interp, float)
        K_mix = cleaner.strikes

        nT_interp = C_mix_interp.shape[0]
        if nT_interp == 0:
            return  # nothing to plot

        if plot_all_original:
            # If we plotted all originals, also show *all* interpolated curves
            idxs_interp = np.arange(nT_interp)
        else:
            n_sel2 = max(1, min(n_curves, nT_interp))
            idxs_interp = rng.choice(nT_interp, size=n_sel2, replace=False)
            idxs_interp = np.sort(idxs_interp)

        fig2, axes2 = plt.subplots(
            nrows=len(idxs_interp), ncols=1,
            figsize=(6, 3 * len(idxs_interp)),
            sharex=False
        )
        if len(idxs_interp) == 1:
            axes2 = [axes2]

        for ax, i in zip(axes2, idxs_interp):
            T_val = T_interp[i]
            C_mix_row = C_mix_interp[i, :]

            ax.plot(K_mix, C_mix_row, "-", lw=2)
            ax.set_xlabel("Strike K")
            ax.set_ylabel("Call price")
            ax.set_title(f"Interpolated mixture call curve (T ≈ {T_val:.3f} yr)")

        plt.tight_layout()
        plt.show()



# ============================================================
# Example usage on a toy surface (you can replace this with
# your option_df -> surface extraction)
# ============================================================

if __name__ == "__main__":
    # Toy setup: Black–Scholes surface with noise + NaNs + arbitrage-ish tweaks
    S0 = 100.0
    r = 0.02
    strikes = np.linspace(60, 140, 20)
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 2.1, 2.2])
    sigmas = np.array([0.20, 0.22, 0.24, 0.26, 0.28, 0.29])

    C_true = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        C_true[i, :] = bs_call_price(S0, strikes, r, sigmas[i], T)

    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.002 * C_true)
    C_noisy = C_true + noise

    C_arb = C_noisy.copy()
    # Induce some obvious bumps + NaNs for the demo
    C_arb[1, 4] += 5.0
    C_arb[2, 7] -= 3.0
    C_arb[3, 2] -= 5.0
    C_arb[0:2, 4] = np.nan
    C_arb[2, 9] = np.nan
    C_arb[3:6, 10] = np.nan

    # Mixture config: e.g. 2 lognormals + 1 Weibull
    config = MixtureSurfaceConfig(
        n_lognormal=2,
        n_weibull=1,
        var_c=0.1,
        var_penalty=1e4,
        random_starts=1,
        seed=123,
        full_estimation=True,      # evaluate mixtures on a thinner strike grid
        fine_strike_factor=3,
        use_maturity_interp=True,
        day_step=7,
    )
    
    config = MixtureSurfaceConfig(
        use_evolutionary=True, #Uses the evolutionary algorithmn. Will take longer, but closer in spirit to yifan li 2024.
        M_max=2,
        use_wald=True,
        wald_alpha=0.05,
        use_maturity_interp=True,
        full_estimation=True,
        day_step=14# evaluate mixtures on a thinner strike grid


    )
# evaluate mixtures on a thinner strike grid



    cleaner = MixtureCallSurfaceCleaner(
        strikes=strikes,
        maturities=maturities,
        S0=S0,
        r=r,
        config=config,
    )

    result = cleaner.fit_surface(C_arb)

    # Plot cleaned call + IV surfaces
    cleaner.plot_call_and_iv_surfaces()

    # Plot some RND curves from final calls
    cleaner.plot_some_rnds(n_curves=6)

    # Plot full RND surface
    cleaner.plot_rnd_surface()
