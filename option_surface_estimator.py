"""
Call Surface Estimation Pipeline (Mixtures / Generalized Gamma / GLD)

This file generalizes your original mixture-based surface smoother so the user can choose
ONE per-maturity ("row") model:

  - "mixture_fixed"       : fit a fixed (#lognormals, #weibulls) mixture per maturity
  - "mixture_evolutionary": use your evolutionary selection per maturity
  - "gen_gamma"           : fit Generalized Gamma per maturity (closed-form calls)
  - "gld"                 : fit Corrado (2001) RS-GLD per maturity (quantile-based)

Pipeline (same structure as your original):
  Stage 1: per-maturity fit on the original maturity grid, fill NaNs or fully replace
  Stage 2: (optional) spline across maturities for each strike to a day grid
  Stage 3: per-maturity re-fit on the interpolated grid for a fully parametric surface
  Then: IV surface (BS implied vols) and RND surface (Breeden–Litzenberger)

Citations (as requested in your earlier message for GenGamma):
  - "Estimating risk-neutral density with parametric models in interest rate markets" (2009)
  - Bouzai, B. (2022). "The Generalized Gamma Distribution as a Useful Risk-Neutral Distribution
    under Heston’s Stochastic Volatility Model." Journal of Risk and Financial Management.

Notes:
  - This file assumes you already have:
        Mixture_LWD.py providing:
          MixtureSpec, fit_mixture_to_calls, evolutionary_lwm_fit
        Generalized_Gamma.py providing:
          GenGammaRNDModel
        generalized_lambda_pricer.py (or your own) providing:
          GLDRNDModel  (Corrado 2001 implementation you wrote)
  - If your GLD class is in another file, adjust the import below.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence, Literal, Any, Dict

from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.stats import norm

# ------------------ Your project imports ------------------
from Mixture_LWD import (  # noqa: F401
    MixtureSpec,
    fit_mixture_to_calls,
    evolutionary_lwm_fit,
)

from Generalized_Gamma import GenGammaRNDModel

# If your GLD class is in generalized_lambda_pricer.py, keep this:
from generalized_lambda_pricer import GLDRNDModel





def plot_original_vs_final_surface(
    est,
    strikes_orig,
    maturities_orig,
    C_orig,
    title="Call Surface Comparison",
):
    """
    Plot original call surface (scatter) vs final fitted surface (surface).

    Works for any row model as long as:
      - est.strikes exists (final strike grid)
      - est.T_interp exists (final maturity grid)
      - est.C_clean exists (final call surface)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    strikes_orig = np.asarray(strikes_orig, float)
    maturities_orig = np.asarray(maturities_orig, float)
    C_orig = np.asarray(C_orig, float)

    nT = len(maturities_orig)
    nK = len(strikes_orig)

    # --- Fix orientation if needed ---
    if C_orig.shape == (nT, nK):
        C_plot = C_orig
    elif C_orig.shape == (nK, nT):
        C_plot = C_orig.T
    else:
        raise ValueError(
            f"C_orig has shape {C_orig.shape}, "
            f"but expected ({nT}, {nK}) or ({nK}, {nT})."
        )

    # Extract final surface from estimator
    if est.C_clean is None or est.T_interp is None or est.strikes is None:
        raise ValueError("Run est.fit_surface(...) before plotting.")

    K_final = np.asarray(est.strikes, float)
    T_final = np.asarray(est.T_interp, float)
    C_final = np.asarray(est.C_clean, float)

    KK_final, TT_final = np.meshgrid(K_final, T_final)
    KK_orig, TT_orig = np.meshgrid(strikes_orig, maturities_orig)

    # Model name for labeling (optional)
    model_name = "Model"
    if hasattr(est, "config") and hasattr(est.config, "row_model") and hasattr(est.config.row_model, "model"):
        model_name = str(est.config.row_model.model)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=18, y=0.97)

    # 1) Original Call Surface (scatter)
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Original Call Surface (observed quotes)")

    mask = np.isfinite(C_plot)
    ax1.scatter(KK_orig[mask], TT_orig[mask], C_plot[mask], s=15, depthshade=True)
    ax1.set_xlabel("Strike K")
    ax1.set_ylabel("Maturity T")
    ax1.set_zlabel("Call Price")

    # 2) Final Estimated Surface (surface)
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title(f"Final Estimated Call Surface\n({model_name} + maturity spline)")

    ax2.plot_surface(KK_final, TT_final, C_final, rstride=1, cstride=1, linewidth=0.2, alpha=0.9)
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Maturity T")
    ax2.set_zlabel("Call Price")

    plt.tight_layout()
    plt.show()

def plot_random_observed_vs_model_curve(
    est,
    strikes_orig: np.ndarray,
    maturities_orig: np.ndarray,
    C_orig: np.ndarray,
    n_curves: int = 3,
    random_state: Optional[int] = None,
    title_prefix: str = "Observed vs Stage-1 Model Call Curve",
    plot_all_original: bool = False,
    plot_interpolated: bool = False,
):
    """
    Plot call curves implied by the estimator versus observed curves.

    Part 1 (always): Observed vs Stage-1 model curves on the ORIGINAL maturity grid
    ---------------------------------------------------------------------------
    For selected maturities T_i in `maturities_orig`, plot:
        - Observed call curve: C_orig[i, :] on strikes_orig (dots, ignoring NaNs)
        - Stage-1 model curve: est.C_stage1[i, :] on est.strikes (line)

    Part 2 (optional): Stage-3 model-only curves on the INTERPOLATED maturity grid
    -----------------------------------------------------------------------------
    If plot_interpolated=True, we create a second figure showing:
        - est.C_mixture_final (or est.C_model_final) on est.T_interp × est.strikes

    Requirements on `est`:
        - est.C_stage1 exists after est.fit_surface(...)
        - est.strikes is the strike grid used for Stage-1 evaluation
        - If plot_interpolated=True: est.C_mixture_final (or est.C_clean) exists
    """
    # ---------- sanity + cast ----------
    strikes_orig = np.asarray(strikes_orig, float)
    maturities_orig = np.asarray(maturities_orig, float)
    C_orig = np.asarray(C_orig, float)

    if not hasattr(est, "C_stage1") or est.C_stage1 is None:
        raise ValueError("est.C_stage1 is missing. Run est.fit_surface(...) first.")

    C_stage1 = np.asarray(est.C_stage1, float)
    K_model = np.asarray(est.strikes, float)

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

    # ---------- Figure 1: observed vs stage-1 ----------
    fig, axes = plt.subplots(
        nrows=len(idxs_orig),
        ncols=1,
        figsize=(7.5, 3.0 * len(idxs_orig)),
        sharex=False,
    )
    if len(idxs_orig) == 1:
        axes = [axes]

    for ax, i in zip(axes, idxs_orig):
        T_obs = float(maturities_orig[i])

        C_obs_row = C_orig[i, :]
        mask = np.isfinite(C_obs_row)

        # stage-1 model curve at the same row index
        C_stage1_row = C_stage1[i, :]

        if np.any(mask):
            ax.plot(strikes_orig[mask], C_obs_row[mask], "o", ms=4, label="Observed")
        else:
            ax.text(0.02, 0.85, "No observed points (all NaN)", transform=ax.transAxes)

        ax.plot(K_model, C_stage1_row, "-", lw=2, label="Stage-1 model")

        ax.set_title(f"{title_prefix} (T ≈ {T_obs:.4f} yr)")
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Call price")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()

    # ---------- Figure 2: interpolated/model-only curves (optional) ----------
    if plot_interpolated:
        # Prefer est.C_mixture_final if present; else fall back to est.C_clean
        if hasattr(est, "C_mixture_final") and est.C_mixture_final is not None:
            C_final = np.asarray(est.C_mixture_final, float)
        elif hasattr(est, "C_clean") and est.C_clean is not None:
            C_final = np.asarray(est.C_clean, float)
        else:
            raise ValueError(
                "plot_interpolated=True requires est.C_mixture_final or est.C_clean. "
                "Make sure your pipeline produced a final surface."
            )

        if not hasattr(est, "T_interp") or est.T_interp is None:
            raise ValueError("plot_interpolated=True requires est.T_interp.")

        T_interp = np.asarray(est.T_interp, float)
        nT_interp = C_final.shape[0]
        if nT_interp == 0:
            return

        if plot_all_original:
            idxs_interp = np.arange(nT_interp)
        else:
            n_sel2 = max(1, min(n_curves, nT_interp))
            idxs_interp = rng.choice(nT_interp, size=n_sel2, replace=False)
            idxs_interp = np.sort(idxs_interp)

        fig2, axes2 = plt.subplots(
            nrows=len(idxs_interp),
            ncols=1,
            figsize=(7.5, 3.0 * len(idxs_interp)),
            sharex=False,
        )
        if len(idxs_interp) == 1:
            axes2 = [axes2]

        for ax, i in zip(axes2, idxs_interp):
            ax.plot(K_model, C_final[i, :], "-", lw=2)
            ax.set_title(f"Model call curve on interpolated grid (T ≈ {T_interp[i]:.4f} yr)")
            ax.set_xlabel("Strike K")
            ax.set_ylabel("Call price")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

# ============================================================
# Black–Scholes helpers (for IV surface)
# ============================================================

def bs_call_price(S0: float, K: np.ndarray, r: float, sigma: float, T: float) -> np.ndarray:
    """
    Black–Scholes European call (present value).

    Parameters
    ----------
    S0 : float
        Spot price.
    K : array
        Strikes.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity (years).

    Returns
    -------
    calls : array
        Call prices at strikes K.
    """
    K = np.asarray(K, float)
    if T <= 0:
        return np.maximum(S0 - K, 0.0)

    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S0: float, K: np.ndarray, r: float, sigma: float, T: float) -> np.ndarray:
    """
    Black–Scholes vega (dC/dsigma).

    Returns
    -------
    vega : array
        Vega at each strike.
    """
    K = np.asarray(K, float)
    if T <= 0:
        return np.zeros_like(K)
    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / vol_sqrtT
    return S0 * np.sqrt(T) * norm.pdf(d1)


def bs_implied_vol(C: float, S0: float, K: float, r: float, T: float,
                   tol: float = 1e-6, max_iter: int = 50) -> float:
    """
    Scalar implied vol via Newton–Raphson with vega safeguard.

    Returns np.nan if:
      - price violates simple arbitrage bounds
      - Newton fails to converge / vega too small
    """
    K = float(K)
    C = float(C)

    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    upper = S0
    if not (intrinsic <= C <= upper):
        return np.nan

    sigma = 0.2
    for _ in range(max_iter):
        C_model = bs_call_price(S0, np.array([K]), r, sigma, T)[0]
        diff = C_model - C
        if abs(diff) < tol:
            return sigma

        v = bs_vega(S0, np.array([K]), r, sigma, T)[0]
        if v < 1e-8:
            break

        sigma -= diff / v
        if sigma <= 0:
            sigma = 0.01

    return np.nan


# ============================================================
# Row model selection config
# ============================================================

RowModelType = Literal["mixture_fixed", "mixture_evolutionary", "gen_gamma", "gld"]


@dataclass
class RowModelConfig:
    """
    Controls WHICH model is fit per maturity slice (row) and its hyperparameters.

    model:
      - "mixture_fixed"
      - "mixture_evolutionary"
      - "gen_gamma"
      - "gld"
    """
    model: RowModelType = "mixture_evolutionary"

    # ---- Mixture settings ----
    n_lognormal: int = 2
    n_weibull: int = 1
    M_max: int = 5

    use_wald: bool = True
    wald_alpha: float = 0.05
    wald_p: int = 3
    wald_q: int = 3

    var_c: float = 0.1
    var_penalty: float = 1e4

    random_starts: int = 3
    seed: int = 123

    # ---- Generalized Gamma settings ----
    gengamma_a0_mult: float = 1.0
    gengamma_d0: float = 1.5
    gengamma_p0: float = 2.0
    gengamma_forward_penalty: float = 1e-4

    gengamma_a_bounds_mult: Tuple[float, float] = (0.2, 5.0)
    gengamma_d_bounds: Tuple[float, float] = (0.2, 25.0)
    gengamma_p_bounds: Tuple[float, float] = (0.2, 50.0)

    # ---- GLD settings ----
    gld_sigma0: float = 0.45
    gld_k30: float = 0.05
    gld_k40: float = 0.02
    gld_forward_penalty: float = 1e-4

    gld_sigma_bounds: Tuple[float, float] = (1e-4, 2.5)
    gld_k3_bounds: Tuple[float, float] = (-2.4, 5.5)
    gld_k4_bounds: Tuple[float, float] = (-2.4, 5.5)

    gld_eps_p: float = 1e-12
    gld_bisect_max_iter: int = 180
    gld_bisect_tol: float = 1e-12
    gld_pdf_pgrid_size: int = 20000


@dataclass
class SurfaceConfig:
    row_model: RowModelConfig = field(default_factory=RowModelConfig)

    # Stage 1 behavior
    full_estimation: bool = True
    fine_strike_factor: int = 3
    strike_extension: float = 0.4

    # Stage 2 maturity interpolation
    use_maturity_interp: bool = True
    day_step: int = 7                      # maturity step in days


# ============================================================
# Adapter to unify row-fit API across model families
# ============================================================

class RowFitAdapter:
    """
    Unify per-row fit object interface.

    Required:
      chat(K_eval, r, T) -> call prices on K_eval

    Optional:
      qhat(S_eval) -> density on S_eval (only if model provides)
    """
    def __init__(self, fit_obj: Any, kind: RowModelType):
        self.fit_obj = fit_obj
        self.kind = kind

    def chat(self, K: np.ndarray, r: float, T: float) -> np.ndarray:
        # Mixture fits: your mixture objects take (K, r, T)
        if self.kind in ("mixture_fixed", "mixture_evolutionary"):
            return self.fit_obj.chat(K, r=r, T=T)
        # GenGamma/GLD: store r,T in the fitted object
        return self.fit_obj.chat(K)

    def qhat(self, S: np.ndarray) -> np.ndarray:
        if hasattr(self.fit_obj, "qhat"):
            return self.fit_obj.qhat(S)
        raise AttributeError("Row model does not expose qhat().")


# ============================================================
# General surface estimator (was MixtureCallSurfaceCleaner)
# ============================================================

class CallSurfaceEstimator:
    """
    Generalized call surface estimator with pluggable per-maturity model.

    Stages:
      1) per maturity (original maturities): fit row model; fill NaNs or replace row
      2) optional spline in maturity for each strike to daily grid
      3) per interpolated maturity: re-fit row model for fully parametric surface

    Outputs:
      - C_clean (final call surface)
      - iv_surface (BS IV)
      - rnd_surface (Breeden–Litzenberger)
    """

    def __init__(self,
                 strikes: np.ndarray,
                 maturities: np.ndarray,
                 S0: float,
                 r: float,
                 config: Optional[SurfaceConfig] = None):
        self.strikes_obs = np.asarray(strikes, float)
        self.strikes = self.strikes_obs.copy()

        self.maturities = np.asarray(maturities, float)
        self.S0 = float(S0)
        self.r = float(r)

        self.config = config or SurfaceConfig()

        # Filled by fit_surface()
        self.C_stage1: Optional[np.ndarray] = None
        self.T_interp: Optional[np.ndarray] = None
        self.C_interp: Optional[np.ndarray] = None
        self.C_model_final: Optional[np.ndarray] = None
        self.C_clean: Optional[np.ndarray] = None
        self.iv_surface: Optional[np.ndarray] = None
        self.rnd_surface: Optional[np.ndarray] = None
        self.rnd_maturities: Optional[np.ndarray] = None
        self.K_interp: Optional[np.ndarray] = None   # final strike grid used after stage 1
        
    def compute_logreturn_moments_table(self) -> "pd.DataFrame":
        """
        Compute log-return moments from the RND surface and return a table (DataFrame).
    
        Uses:
            x = log(S_T / F_T),  F_T = S0 * exp(r T)
    
        NaNs in the RND are safely ignored:
          - integration is performed only on valid support
          - density is renormalized on that support
    
        Columns:
            T, mean, var, vol, vol_ann, skew, kurt, area_q (diagnostic)
        """
        import pandas as pd
        import numpy as np
    
        assert self.rnd_surface is not None
        assert self.rnd_maturities is not None
    
        K_full = np.asarray(self.strikes, float)
        rnd = np.asarray(self.rnd_surface, float)
        T_grid = np.asarray(self.rnd_maturities, float)
    
        rows = []
    
        for i, T in enumerate(T_grid):
            q_full = rnd[i, :]
    
            # -------------------------------
            # 1) Valid support mask
            # -------------------------------
            mask = (
                np.isfinite(K_full)
                & np.isfinite(q_full)
                & (q_full > 0.0)
            )
    
            if mask.sum() < 5:
                rows.append({
                    "T": float(T),
                    "mean": np.nan,
                    "var": np.nan,
                    "vol": np.nan,
                    "vol_ann": np.nan,
                    "skew": np.nan,
                    "kurt": np.nan,
                    "area_q": np.nan,
                })
                continue
    
            K = K_full[mask]
            q = q_full[mask]
    
            # -------------------------------
            # 2) Normalize density on support
            # -------------------------------
            area = float(np.trapz(q, K))
            if (not np.isfinite(area)) or area <= 0.0:
                rows.append({
                    "T": float(T),
                    "mean": np.nan,
                    "var": np.nan,
                    "vol": np.nan,
                    "vol_ann": np.nan,
                    "skew": np.nan,
                    "kurt": np.nan,
                    "area_q": area,
                })
                continue
    
            q /= area  # proper density on truncated support
    
            # -------------------------------
            # 3) Log-return variable
            # -------------------------------
            F = self.S0 * np.exp(self.r * T)
            x = np.log(K / F)
    
            # -------------------------------
            # 4) Moments
            # -------------------------------
            mean_x = float(np.trapz(x * q, K))
            var_x = float(np.trapz((x - mean_x) ** 2 * q, K))
    
            if var_x <= 0 or not np.isfinite(var_x):
                rows.append({
                    "T": float(T),
                    "mean": mean_x,
                    "var": var_x,
                    "vol": np.nan,
                    "vol_ann": np.nan,
                    "skew": np.nan,
                    "kurt": np.nan,
                    "area_q": area,
                })
                continue
    
            vol_x = float(np.sqrt(var_x))
            vol_ann = float(vol_x / np.sqrt(T)) if T > 0 else np.nan
    
            skew_x = float(
                np.trapz((x - mean_x) ** 3 * q, K) / (var_x ** 1.5)
            )
            kurt_x = float(
                np.trapz((x - mean_x) ** 4 * q, K) / (var_x ** 2)
            )
    
            rows.append({
                "T": float(T),
                "mean": mean_x,
                "var": var_x,
                "vol": vol_x,
                "vol_ann": vol_ann,
                "skew": skew_x,
                "kurt": kurt_x,
                "area_q": area,   # diagnostic: mass captured before renormalization
            })
    
        df = pd.DataFrame(rows)
        df = df[["T", "mean", "var", "vol", "vol_ann", "skew", "kurt", "area_q"]]
    
        self.rnd_moments_table = df
        return df


    # --------------------------------------------------------
    # Row-model fitter (key generalization)
    # --------------------------------------------------------

    def _fit_row_model(self, K_row: np.ndarray, C_row: np.ndarray, T: float, seed_shift: int = 0
                       ) -> Optional[RowFitAdapter]:
        """
        Fit one maturity slice using the chosen row model.

        Returns
        -------
        adapter : RowFitAdapter or None
            None if insufficient valid quotes.
        """
        cfg = self.config.row_model

        K_row = np.asarray(K_row, float)
        C_row = np.asarray(C_row, float)
        mask = ~np.isnan(C_row)

        if mask.sum() < 4:
            return None

        K_valid = K_row[mask]
        C_valid = C_row[mask]

        # ---------------- Mixture: fixed ----------------
        if cfg.model == "mixture_fixed":
            print(f"Using fixed mixtures: Lognormal={cfg.n_lognormal}, Weibull={cfg.n_weibull}")
            spec = MixtureSpec(n_lognormal=cfg.n_lognormal, n_weibull=cfg.n_weibull)

            fit, _, _ = fit_mixture_to_calls(
                K=K_valid,
                C_mkt=C_valid,
                S0=self.S0,
                r=self.r,
                T=T,
                spec=spec,
                theta0=None,
                penalty_lambda=1e7,
                random_starts=cfg.random_starts,
                seed=cfg.seed + seed_shift,
                rnd_true=None,
                k_true=None,
                var_c=cfg.var_c,
                var_penalty=cfg.var_penalty,
                return_theta=False,
            )
            return RowFitAdapter(fit, "mixture_fixed")

        # ---------------- Mixture: evolutionary ----------------
        if cfg.model == "mixture_evolutionary":
            print(f"Using evolutionary mixtures: Lognormal={cfg.M_max}")

            fit, _chosen_spec = evolutionary_lwm_fit(
                K=K_valid,
                C_mkt=C_valid,
                S0=self.S0,
                r=self.r,
                T=T,
                M_max=cfg.M_max,
                penalty_lambda=0.0,
                random_starts=cfg.random_starts,
                seed=cfg.seed + seed_shift,
                var_c=cfg.var_c,
                var_penalty=cfg.var_penalty,
                improvement_tol=1e-4,
                metric="loss",
                rnd_true=None,
                k_true=None,
                use_wald=cfg.use_wald,
                wald_alpha=cfg.wald_alpha,
                wald_p=cfg.wald_p,
                wald_q=cfg.wald_q,
                weights=None,
                fixed_M=None,
                fixed_M1=None,
            )
            return RowFitAdapter(fit, "mixture_evolutionary")

        # ---------------- GenGamma ----------------
        if cfg.model == "gen_gamma":
            print("Using Generalized Gamma")

            a0 = cfg.gengamma_a0_mult * self.S0
            a_bounds = (cfg.gengamma_a_bounds_mult[0] * self.S0,
                        cfg.gengamma_a_bounds_mult[1] * self.S0)

            gg = GenGammaRNDModel(
                S0=self.S0,
                r=self.r,
                T=T,
                K=K_valid,
                C_mkt=C_valid,
                a0=a0,
                d0=cfg.gengamma_d0,
                p0=cfg.gengamma_p0,
                forward_penalty_lambda=cfg.gengamma_forward_penalty,
                a_bounds=a_bounds,
                d_bounds=cfg.gengamma_d_bounds,
                p_bounds=cfg.gengamma_p_bounds,
            )
            return RowFitAdapter(gg, "gen_gamma")

        # ---------------- GLD ----------------
        if cfg.model == "gld":
            print("Using Generalized Lambda")

            gld = GLDRNDModel(
                S0=self.S0,
                r=self.r,
                T=T,
                K=K_valid,
                C_mkt=C_valid,
                sigma0=cfg.gld_sigma0,
                k30=cfg.gld_k30,
                k40=cfg.gld_k40,
                forward_penalty_lambda=cfg.gld_forward_penalty,
                sigma_bounds=cfg.gld_sigma_bounds,
                k3_bounds=cfg.gld_k3_bounds,
                k4_bounds=cfg.gld_k4_bounds,
                eps_p=cfg.gld_eps_p,
                bisect_max_iter=cfg.gld_bisect_max_iter,
                bisect_tol=cfg.gld_bisect_tol,
                pdf_pgrid_size=cfg.gld_pdf_pgrid_size,
            )
            return RowFitAdapter(gld, "gld")

        raise ValueError(f"Unknown row model: {cfg.model}")

    # --------------------------------------------------------
    # Stage 1: fit per original maturity, fill NaNs or replace
    # --------------------------------------------------------

    def stage1_rowmodel_fill(self, C_surface: np.ndarray) -> np.ndarray:
        """
        Stage 1: For each maturity row, fit the chosen row model.

        If full_estimation=False:
          - strike grid = original strikes
          - only NaNs are replaced with model prices

        If full_estimation=True:
          - strike grid becomes refined + extended
          - entire row replaced with model prices
        """
        C_surface = np.asarray(C_surface, float)
        nT, nK_obs = C_surface.shape

        if self.config.full_estimation:
            nK_fine = max(self.config.fine_strike_factor * nK_obs, nK_obs)

            low = float(self.strikes_obs.min())
            high = float(self.strikes_obs.max())
            ext = max(self.config.strike_extension, 0.0)

            K_min_ext = low * (1.0 - ext)
            K_max_ext = high * (1.0 + ext)
            if K_min_ext <= 0:
                K_min_ext = max(low * 0.5, 1e-6)

            self.strikes = np.linspace(K_min_ext, K_max_ext, nK_fine)
            C_stage1 = np.empty((nT, nK_fine), dtype=float)
        else:
            self.strikes = self.strikes_obs.copy()
            C_stage1 = np.array(C_surface, copy=True, dtype=float)
        self.K_interp = self.strikes.copy()

        for i, T in enumerate(self.maturities):
            row_obs = C_surface[i, :]
            rowfit = self._fit_row_model(self.strikes_obs, row_obs, T, seed_shift=i)

            if rowfit is None:
                # Not enough points -> fallback behavior
                if self.config.full_estimation:
                    mask = ~np.isnan(row_obs)
                    if mask.sum() >= 2:
                        C_stage1[i, :] = np.interp(self.strikes,
                                                   self.strikes_obs[mask],
                                                   row_obs[mask])
                    elif mask.sum() == 1:
                        C_stage1[i, :] = row_obs[mask][0]
                    else:
                        C_stage1[i, :] = np.nan
                else:
                    continue
            else:
                C_hat = rowfit.chat(self.strikes, r=self.r, T=T)
                if self.config.full_estimation:
                    C_stage1[i, :] = C_hat
                else:
                    mask_nan = np.isnan(row_obs)
                    C_stage1[i, mask_nan] = C_hat[mask_nan]

        self.C_stage1 = C_stage1
        return C_stage1

    # --------------------------------------------------------
    # Stage 2: maturity spline per strike
    # --------------------------------------------------------

    def stage2_maturity_spline(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optional: spline in maturity for each strike.

        - Fits cubic spline in T to each strike column (ignoring NaNs).
        - Evaluates on a day grid.
        - Restores Stage-1 values at original maturities (so spline never overwrites).
        """
        assert self.C_stage1 is not None
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

        for j in range(nK):
            y = C_stage1[:, j]
            mask_valid = ~np.isnan(y)

            if mask_valid.sum() == 0:
                C_interp[:, j] = np.nan
                continue

            if mask_valid.sum() == 1:
                C_interp[:, j] = y[mask_valid][0]
                continue

            T_valid = T_orig[mask_valid]
            y_valid = y[mask_valid]
            cs = CubicSpline(T_valid, y_valid, bc_type="natural")
            col = cs(T_interp)

            # restore original maturities where stage1 had data
            for i0, T0 in enumerate(T_orig):
                if np.isnan(y[i0]):
                    continue
                k = int(np.argmin(np.abs(T_interp - T0)))
                col[k] = y[i0]

            C_interp[:, j] = col

        self.T_interp = T_interp
        self.C_interp = C_interp
        return T_interp, C_interp

    # --------------------------------------------------------
    # Stage 3: refit row model on interpolated grid
    # --------------------------------------------------------

    def stage3_refit_rowmodel_on_interp(self) -> np.ndarray:
        """
        For each maturity on T_interp, re-fit the chosen row model to that slice.
        """
        assert self.T_interp is not None and self.C_interp is not None

        C_interp = np.asarray(self.C_interp, float)
        T_interp = np.asarray(self.T_interp, float)

        C_final = np.empty_like(C_interp)
        for i, T in enumerate(T_interp):
            row = C_interp[i, :]
            rowfit = self._fit_row_model(self.strikes, row, T, seed_shift=1000 + i)

            if rowfit is None:
                C_final[i, :] = row
            else:
                C_final[i, :] = rowfit.chat(self.strikes, r=self.r, T=T)

        self.C_model_final = C_final
        return C_final

    # --------------------------------------------------------
    # RND and IV surfaces
    # --------------------------------------------------------
    def compute_rnd_surface(self, eps: float = 1e-12) -> np.ndarray:
        """
        Breeden–Litzenberger RND:
          q_T(K) = exp(r T) * d^2 C(T,K) / dK^2
    
        Post-processing:
          - Replace NaN / inf / <= 0 values with a tiny epsilon
        """
        assert self.C_clean is not None and self.T_interp is not None
    
        C = np.asarray(self.C_clean, float)
        T_grid = np.asarray(self.T_interp, float)
        K = np.asarray(self.strikes, float)
    
        rnd = np.empty_like(C)
    
        for i, T in enumerate(T_grid):
            dC_dK = np.gradient(C[i, :], K, edge_order=2)
            d2C_dK2 = np.gradient(dC_dK, K, edge_order=2)
    
            qi = np.exp(self.r * T) * d2C_dK2
    
            # 🔒 stabilize
            qi = np.where(np.isfinite(qi) & (qi > eps), qi, eps)
    
            rnd[i, :] = qi
    
        self.rnd_surface = rnd
        self.rnd_maturities = T_grid
        return rnd

    def compute_iv_surface(self) -> np.ndarray:
        """
        BS implied volatility surface from C_clean.
        """
        assert self.C_clean is not None and self.T_interp is not None
        C = np.asarray(self.C_clean, float)
        T_grid = np.asarray(self.T_interp, float)
        K_grid = np.asarray(self.strikes, float)

        iv = np.empty_like(C)
        for i, T in enumerate(T_grid):
            for j, K in enumerate(K_grid):
                iv[i, j] = bs_implied_vol(C[i, j], self.S0, K, self.r, T)

        self.iv_surface = iv
        return iv

    # --------------------------------------------------------
    # Master method
    # --------------------------------------------------------

    def fit_surface(self, C_surface: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run the full pipeline.

        Returns a dict with:
          - T_interp
          - C_stage1
          - C_interp
          - C_model_final
          - C_clean
          - iv_surface
          - rnd_surface
        """
        self.stage1_rowmodel_fill(C_surface)
        self.stage2_maturity_spline()
        self.stage3_refit_rowmodel_on_interp()

        self.C_clean = self.C_model_final
        self.compute_iv_surface()
        self.compute_rnd_surface()
        self.rnd_moments_table = self.compute_logreturn_moments_table()



        return {
            "K_interp": self.K_interp,
            "T_interp": self.T_interp,
            "C_stage1": self.C_stage1,
            "C_interp": self.C_interp,
            "C_model_final": self.C_model_final,
            "C_clean": self.C_clean,
            "iv_surface": self.iv_surface,
            "rnd_surface": self.rnd_surface,
            "rnd_moments_table": self.rnd_moments_table,
            
        }

    # --------------------------------------------------------
    # Plotting helpers
    # --------------------------------------------------------

    def plot_call_and_iv_surfaces(self):
        """3D plots of final call surface and IV surface."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        assert self.C_clean is not None and self.iv_surface is not None
        K = self.strikes
        T = self.T_interp
        KK, TT = np.meshgrid(K, T)

        fig = plt.figure(figsize=(14, 5))

        ax1 = fig.add_subplot(121, projection="3d")
        ax1.set_title("Final call surface C_clean(T, K)")
        ax1.plot_surface(KK, TT, self.C_clean, rstride=1, cstride=1, linewidth=0.2)
        ax1.set_xlabel("Strike K")
        ax1.set_ylabel("Maturity T")
        ax1.set_zlabel("Call price")

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.set_title("Implied volatility surface σ(T, K)")
        ax2.plot_surface(KK, TT, self.iv_surface, rstride=1, cstride=1, linewidth=0.2)
        ax2.set_xlabel("Strike K")
        ax2.set_ylabel("Maturity T")
        ax2.set_zlabel("Implied vol")

        plt.tight_layout()
        plt.show()

    def plot_some_rnds(
        self,
        n_curves: int = 3,
        layout: str = "overlay",   # "overlay" or "panels"
        ncols: int = 2
    ):
        """
        Plot RND curves for a subset of maturities.
    
        Parameters
        ----------
        n_curves : int
            Number of maturities to plot
        layout : {"overlay", "panels"}
            Overlay curves in one axis or plot in separate panels
        ncols : int
            Number of columns when layout="panels"
        """
        assert self.rnd_surface is not None and self.rnd_maturities is not None
    
        rnd = self.rnd_surface
        T_grid = self.rnd_maturities
        K = self.strikes
    
        nT = T_grid.size
        idxs = np.linspace(0, nT - 1, min(n_curves, nT), dtype=int)
    
        # --------------------------------------------------
        # Overlay (your original behavior)
        # --------------------------------------------------
        if layout == "overlay":
            plt.figure(figsize=(7, 5))
            for idx in idxs:
                plt.plot(K, rnd[idx, :], label=f"T={T_grid[idx]:.3f} yr")
    
            plt.title("Risk-neutral densities")
            plt.xlabel("Terminal price s = K")
            plt.ylabel("q_T(s)")
            plt.legend()
            plt.tight_layout()
            plt.show()
            return
    
        # --------------------------------------------------
        # Panel layout
        # --------------------------------------------------
        if layout == "panels":
            n_panels = len(idxs)
            nrows = int(np.ceil(n_panels / ncols))
    
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(3.8 * ncols, 3.0 * nrows),
                sharex=True,
                sharey=True
            )
    
            axes = np.atleast_1d(axes).ravel()
    
            for ax, idx in zip(axes, idxs):
                ax.plot(K, rnd[idx, :], lw=2)
                ax.set_title(f"T = {T_grid[idx]:.3f} yr")
                ax.set_xlabel("s = K")
                ax.set_ylabel("q_T(s)")
                ax.grid(alpha=0.3)
    
            # Hide unused axes
            for ax in axes[len(idxs):]:
                ax.axis("off")
    
            fig.suptitle("Risk-neutral densities by maturity", y=0.98)
            plt.tight_layout()
            plt.show()
            return
    
        raise ValueError("layout must be 'overlay' or 'panels'")


    def plot_rnd_surface(self, title: str = "Risk-neutral density surface"):
        """3D hillshade plot of RND surface."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib.colors import LightSource
    
        assert self.rnd_surface is not None and self.rnd_maturities is not None
    
        K = self.strikes
        T = self.rnd_maturities
        KK, TT = np.meshgrid(K, T)
        Z = self.rnd_surface
    
        # ---- Hillshade lighting ----
        ls = LightSource(
            azdeg=315,   # light direction (NW-ish)
            altdeg=45    # elevation angle
        )
    
        rgb = ls.shade(
            Z,
            cmap=plt.cm.gray,   # grayscale works best for hillshade
            vert_exag=1.0,
            blend_mode='overlay'
        )
    
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    
        ax.set_title(title)
    
        ax.plot_surface(
            KK, TT, Z,
            facecolors=rgb,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False  # IMPORTANT: disable Matplotlib's default shading
        )
    
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T")
        ax.set_zlabel("f(k)")
    
        plt.tight_layout()
        plt.show()
    

# ============================================================
# option_df → surface helper (same idea as yours)
# ============================================================

def extract_call_surface_from_df(
    option_df: pd.DataFrame,
    price_col: str = "mid_price",
    maturity_col: str = "rounded_maturity",
    right_col: str = "option_right",
    call_code: str = "c",
):
    """
    Extract (strikes, maturities, C_surface, S0, r) from an option dataframe.

    Returns
    -------
    strikes : 1D np.ndarray
    maturities : 1D np.ndarray
    C_surface : 2D np.ndarray (nT, nK) with NaNs for missing quotes
    S0 : float
    r : float
    """
    df = option_df.copy()

    if right_col in df.columns:
        df = df[df[right_col] == call_code]

    df = df.dropna(subset=["strike", maturity_col, price_col])

    strikes = np.sort(df["strike"].unique())
    maturities = np.sort(df[maturity_col].unique())

    surface = df.pivot_table(
        index=maturity_col,
        columns="strike",
        values=price_col,
        aggfunc="mean",
    )
    surface = surface.reindex(index=maturities, columns=strikes)
    C = surface.values.astype(float)

    if "underlying_price" in df.columns:
        S0 = float(df["underlying_price"].median())
    elif "stock_price" in df.columns:
        S0 = float(df["stock_price"].median())
    else:
        raise ValueError("Need 'underlying_price' or 'stock_price' column for S0.")

    if "risk_free_rate" in df.columns:
        r = float(df["risk_free_rate"].median())
    elif "risk_f" in df.columns:
        r = float(df["risk_f"].median())
    else:
        r = 0.0

    return strikes, maturities, C, S0, r


# ============================================================
# Example usage (toy surface)
# ============================================================
if __name__ == "__main__":
    # --- toy BS surface with noise + NaNs ---
    S0 = 140.0
    r = 0.02
    strikes = np.linspace(60, 140, 20)
    maturities = np.array([0.25, 0.5, 1.0, 1.1, 1.2, 1.3])
    sigmas = np.array([0.20, 0.22, 0.24, 0.26, 0.28, 0.29])

    C_true = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        C_true[i, :] = bs_call_price(S0, strikes, r, sigmas[i], T)

    rng = np.random.default_rng(0)
    C_noisy = C_true + rng.normal(scale=0.02 * C_true)

    C = C_noisy.copy()
    C[1, 4] += 5.0
    C[2, 7] -= 3.0
    C[3, 2] -= 5.0
    C[0:2, 4] = np.nan
    C[2, 9] = np.nan
    C[3:6, 10] = np.nan

    # --- choose row model here ---
    cfg = SurfaceConfig(
        row_model=RowModelConfig(
            model="gen_gamma",  # <-- swap: "mixture_fixed","mixture_evolutionary or "gen_gamma" or "gld"
            M_max=2,
            use_wald=True,
            wald_alpha=0.05,
            random_starts=1,
            seed=123,
        ),
        full_estimation=True,
        fine_strike_factor=3,
        strike_extension=0.7,
        use_maturity_interp=True,
        day_step=14,
    )

    est = CallSurfaceEstimator(
        strikes=strikes,
        maturities=maturities,
        S0=S0,
        r=r,
        config=cfg,
    )

    _ = est.fit_surface(C)

    # --- surfaces ---
    est.plot_call_and_iv_surfaces()
    est.plot_some_rnds(n_curves=15)
    est.plot_rnd_surface()

    plot_random_observed_vs_model_curve(
    est=est,
    strikes_orig=strikes,
    maturities_orig=maturities,
    C_orig=C,
    plot_all_original=True,
    title_prefix="Observed vs Stage-1 Model Call Curve",
    plot_interpolated=False,)   # True if you also want interpolated curves
    
    plot_random_observed_vs_model_curve(
    est=est,
    strikes_orig=strikes,
    maturities_orig=maturities,
    C_orig=C,
    plot_all_original=True,
    title_prefix="Observed vs Stage-1 Model Call Curve",
    plot_interpolated=False,   # True if you also want interpolated curves
    )

    plot_original_vs_final_surface(
        est=est,
        strikes_orig=strikes,
        maturities_orig=maturities,
        C_orig=C,
        title="Toy Surface: Observed vs Final Estimated Call Surface",
    )