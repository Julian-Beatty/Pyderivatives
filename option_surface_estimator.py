"""
Call Surface Estimation Pipeline (Mixtures / Hermite / Global Models)

This file generalizes your original mixture-based surface smoother so the user can choose
ONE per-maturity ("row") model (mixtures/hermite) OR a global parametric model fit to the
entire surface (Bates / Kou / Heston / Kou+Heston HKDE).

Row models (per maturity):
  - "mixture_fixed"
  - "mixture_evolutionary"
  - "hermite"

Global models (fit entire surface at once):
  - "bates"
  - "kou"
  - "heston"
  - "kou_heston"  (HKDE)

Pipeline (row models):
  Stage 1: per-maturity fit on original maturities
           - ALWAYS full estimation (dense strike grid + extension)
  Stage 2: (optional) spline across maturities for each strike to a day grid
  Stage 3: per-maturity re-fit on the interpolated grid

Global models:
  - No stage1/2/3; we fit once to (K_obs, T_obs, C_obs) vectors
  - Then evaluate C_clean on dense strike grid x interpolated maturities (if enabled)

Outputs:
  - C_clean(T, K)
  - iv_surface via BS inversion (on C_clean)
  - rnd_surface via Breeden–Litzenberger (on C_clean)
  - cdf_surface from rnd_surface
  - moments table from RND surface

Requirements (your project files):
  - Mixture_LWD.py providing: MixtureSpec, fit_mixture_to_calls, evolutionary_lwm_fit
  - Hermite_Expansion_Jarrow_Rudd.py providing: HermiteRNDRowModel with .chat() and .qhat()
  - bates.py providing: BatesSurfaceFitter
  - Kou_model.py providing: KouJDModel
  - Heston_surface.py providing: HestonModel
  - kou_heston.py providing: HKDEModel, HKDEParams
"""

import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence, Literal, Any, Dict, Union

from scipy.interpolate import CubicSpline
from scipy.stats import norm

# ------------------ Your project imports ------------------
from Mixture_LWD import (  # noqa: F401
    MixtureSpec,
    fit_mixture_to_calls,
    evolutionary_lwm_fit,
)

from Hermite_Expansion_Jarrow_Rudd import *  # expects HermiteRNDRowModel

# Global model imports (your uploaded files)
from bates import BatesSurfaceFitter
from Kou_model import KouJDModel  # <-- the class name in your file
from Heston_surface import HestonModel
from kou_heston import HKDEModel, HKDEParams


# ============================================================
# Save helper
# ============================================================
def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "figure"


def _resolve_save_path(
    save: Optional[Union[str, os.PathLike]],
    default_name: str,
    ext: str = ".png",
) -> Optional[Path]:
    """
    If save is:
      - None: do nothing
      - a directory: save into that directory using default_name
      - a file path: save exactly there
    """
    if save is None:
        return None

    p = Path(save).expanduser().resolve()

    if p.exists() and p.is_dir():
        return p / (default_name + ext)

    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    p.mkdir(parents=True, exist_ok=True)
    return p / (default_name + ext)


def _maybe_savefig(
    fig: plt.Figure,
    save: Optional[Union[str, os.PathLike]],
    default_name: str,
    dpi: int = 200,
):
    path = _resolve_save_path(save, default_name=default_name, ext=".png")
    if path is not None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ============================================================
# Plot annotation helper (Safety clip note)
# ============================================================
def _add_safety_clip_note(fig: plt.Figure, status: str, where: str = "br"):
    """
    Adds a small note to the figure: "Safety clip: Used/Unused".

    where:
      - "br": bottom-right (default)
      - "tr": top-right
    """
    txt = f"Safety clip: {status}"
    if where == "tr":
        x, y, va = 0.99, 0.99, "top"
    else:
        x, y, va = 0.99, 0.01, "bottom"

    fig.text(
        x,
        y,
        txt,
        ha="right",
        va=va,
        fontsize=9,
        alpha=0.85,
    )


# ============================================================
# Plot helpers (surface + observed-vs-model call curves)
# ============================================================
def plot_original_vs_final_surface(
    est,
    strikes_orig,
    maturities_orig,
    C_orig,
    title="Call Surface Comparison",
    save: Optional[Union[str, os.PathLike]] = None,
    dpi: int = 200,
):
    """
    Plot original call surface (scatter) vs final fitted surface (surface).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    strikes_orig = np.asarray(strikes_orig, float)
    maturities_orig = np.asarray(maturities_orig, float)
    C_orig = np.asarray(C_orig, float)

    nT = len(maturities_orig)
    nK = len(strikes_orig)

    if C_orig.shape == (nT, nK):
        C_plot = C_orig
    elif C_orig.shape == (nK, nT):
        C_plot = C_orig.T
    else:
        raise ValueError(
            f"C_orig has shape {C_orig.shape}, expected ({nT}, {nK}) or ({nK}, {nT})."
        )

    if est.C_clean is None or est.T_interp is None or est.strikes is None:
        raise ValueError("Run est.fit_surface(...) before plotting.")

    K_final = np.asarray(est.strikes, float)
    T_final = np.asarray(est.T_interp, float)
    C_final = np.asarray(est.C_clean, float)

    KK_final, TT_final = np.meshgrid(K_final, T_final)
    KK_orig, TT_orig = np.meshgrid(strikes_orig, maturities_orig)

    model_name = "Model"
    if hasattr(est, "config") and hasattr(est.config, "row_model") and hasattr(est.config.row_model, "model"):
        model_name = str(est.config.row_model.model)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=18, y=0.97)

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Original Call Surface (observed quotes)")
    mask = np.isfinite(C_plot)
    ax1.scatter(KK_orig[mask], TT_orig[mask], C_plot[mask], s=15, depthshade=True)
    ax1.set_xlabel("Strike K")
    ax1.set_ylabel("Maturity T")
    ax1.set_zlabel("Call Price")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title(f"Final Estimated Call Surface\n({model_name})")
    ax2.plot_surface(KK_final, TT_final, C_final, rstride=1, cstride=1, linewidth=0.2, alpha=0.9)
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Maturity T")
    ax2.set_zlabel("Call Price")

    # Safety-clip note
    if hasattr(est, "safety_clip_status"):
        _add_safety_clip_note(fig, est.safety_clip_status, where="br")

    plt.tight_layout()
    _maybe_savefig(fig, save, default_name=_slugify(title) + "_surface_comparison", dpi=dpi)
    plt.show()


def plot_random_observed_vs_model_curve(
    est,
    strikes_orig: np.ndarray,
    maturities_orig: np.ndarray,
    C_orig: np.ndarray,
    n_curves: int = 6,
    random_state: Optional[int] = None,
    title_prefix: str = "Observed vs Model Call Curve",
    plot_all_original: bool = False,
    save: Optional[Union[str, os.PathLike]] = None,
    dpi: int = 200,
):
    """
    Panel plot: observed calls vs model calls at a subset of maturities.

    Uses:
      - If row pipeline: est.C_stage1 (on original maturities)
      - If global model: est.C_fit_obs_surface (on observed grid)
    """
    strikes_orig = np.asarray(strikes_orig, float)
    maturities_orig = np.asarray(maturities_orig, float)
    C_orig = np.asarray(C_orig, float)

    rng = np.random.default_rng(random_state)

    if hasattr(est, "C_stage1") and (est.C_stage1 is not None):
        C_model_rows = np.asarray(est.C_stage1, float)
        K_model = np.asarray(est.strikes, float)
        model_label = "Stage-1 model"
    elif hasattr(est, "C_fit_obs_surface") and (est.C_fit_obs_surface is not None):
        C_model_rows = np.asarray(est.C_fit_obs_surface, float)
        K_model = np.asarray(est.strikes_obs, float)
        model_label = "Global model (fit on observed grid)"
    else:
        raise ValueError("No model surface found. Run est.fit_surface(...) first.")

    nT = min(C_orig.shape[0], C_model_rows.shape[0])
    if nT == 0:
        raise ValueError("No maturities available to plot.")

    if plot_all_original:
        idxs = np.arange(nT)
    else:
        n_sel = max(1, min(n_curves, nT))
        idxs = np.sort(rng.choice(nT, size=n_sel, replace=False))

    fig, axes = plt.subplots(
        nrows=len(idxs),
        ncols=1,
        figsize=(7.5, 3.0 * len(idxs)),
        sharex=False,
    )
    if len(idxs) == 1:
        axes = [axes]

    for ax, i in zip(axes, idxs):
        T_obs = float(maturities_orig[i])

        C_obs_row = C_orig[i, :]
        mask = np.isfinite(C_obs_row)

        C_model_row = C_model_rows[i, :]

        if np.any(mask):
            ax.plot(strikes_orig[mask], C_obs_row[mask], "o", ms=4, label="Observed")
        else:
            ax.text(0.02, 0.85, "No observed points (all NaN)", transform=ax.transAxes)

        ax.plot(K_model, C_model_row, "-", lw=2, label=model_label)

        ax.set_title(f"{title_prefix} (T ≈ {T_obs:.4f} yr)")
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Call price")
        ax.grid(alpha=0.3)
        ax.legend()

    if hasattr(est, "safety_clip_status"):
        _add_safety_clip_note(fig, est.safety_clip_status, where="br")

    plt.tight_layout()
    _maybe_savefig(fig, save, default_name=_slugify(title_prefix) + "_call_curves", dpi=dpi)
    plt.show()


# ============================================================
# Black–Scholes helpers (for IV surface)
# ============================================================
def bs_call_price(S0: float, K: np.ndarray, r: float, sigma: float, T: float) -> np.ndarray:
    K = np.asarray(K, float)
    if T <= 0:
        return np.maximum(S0 - K, 0.0)
    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S0: float, K: np.ndarray, r: float, sigma: float, T: float) -> np.ndarray:
    K = np.asarray(K, float)
    if T <= 0:
        return np.zeros_like(K)
    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / vol_sqrtT
    return S0 * np.sqrt(T) * norm.pdf(d1)


def bs_implied_vol(
    C: float,
    S0: float,
    K: float,
    r: float,
    T: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
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
RowModelType = Literal[
    "mixture_fixed",
    "mixture_evolutionary",
    "hermite",
    "bates",
    "kou",
    "heston",
    "kou_heston",
]


@dataclass
class RowModelConfig:
    """
    Controls WHICH model is used:
      - row models (mixture/hermite) OR
      - global models (bates/kou/heston/kou_heston).
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

    # ---- Hermite settings ----
    hermite_order: int = 5
    hermite_sigma0: float = 0.25
    hermite_c0: Optional[Tuple[float, ...]] = None  # length = order-2

    hermite_sigma_bounds: Tuple[float, float] = (1e-4, 3.0)
    hermite_c_bounds: Tuple[float, float] = (-2.0, 12.0)

    hermite_s_grid_mult: Tuple[float, float] = (0.15, 4.0)
    hermite_s_grid_size: int = 9000

    hermite_w_neg: float = 3e4
    hermite_w_mass: float = 8e3
    hermite_w_forward: float = 8e3

    # ---- Global Bates settings ----
    bates_u_max: float = 240.0
    bates_n_quad: int = 96
    bates_max_nfev: int = 200
    bates_verbose: int = 1
    bates_x0: Optional[Dict[str, float]] = None
    bates_bounds: Optional[Tuple[Dict[str, float], Dict[str, float]]] = None
    bates_weights: Optional[np.ndarray] = None
    bates_fixed: Optional[Dict[str, float]] = None

    # ---- Global Kou settings ----
    kou_quad_N: int = 120
    kou_quad_u_max: float = 180.0
    kou_q: float = 0.0
    kou_x0: Optional[Tuple[float, float, float, float, float]] = None  # (sigma, lam, p_up, eta1, eta2)
    kou_bounds: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    kou_weights: Optional[np.ndarray] = None
    kou_max_nfev: int = 400
    kou_verbose: int = 1

    # ---- Global Heston settings ----
    heston_quad_N: int = 120
    heston_quad_u_max: float = 180.0
    heston_q: float = 0.0
    heston_x0: Optional[Tuple[float, float, float, float, float]] = None  # (kappa, theta, sigma, v0, rho)
    heston_bounds: Optional[Any] = None
    heston_weights: Optional[np.ndarray] = None
    heston_max_nfev: int = 450
    heston_verbose: int = 1

    # ---- Global Kou+Heston (HKDE) settings ----
    hkde_quad_N: int = 96
    hkde_quad_u_max: float = 200.0
    hkde_q: float = 0.0

    # ---- Global Kou+Heston (HKDE) vega-weight settings ----
    hkde_vega_floor: float = 1e-3
    hkde_w_cap: float = 2e3

    # x0 is HKDEParams OR None
    hkde_x0: Optional["HKDEParams"] = None

    # bounds must be (lb, ub) vectors of length 9 in the HKDEParams order:
    # [v0, theta, kappa, sigma_v, rho, lam, p_up, eta1, eta2]
    hkde_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # if you already have IVs and want vega-weighted residuals, pass an iv array here
    hkde_iv_obs: Optional[np.ndarray] = None
    hkde_use_vega_weights: bool = False

    hkde_max_nfev: int = 200
    hkde_verbose: int = 1


@dataclass
class SurfaceConfig:
    row_model: RowModelConfig = field(default_factory=RowModelConfig)

    # Dense strike grid + extension (used in row stage1 and global models)
    fine_strike_factor: int = 3
    strike_extension: float = 0.4

    # Stage 2 maturity interpolation (row models only; global models also evaluate on this grid)
    use_maturity_interp: bool = True
    day_step: int = 7

    # ----------------------------
    # RND safety-clip controls
    # ----------------------------
    apply_safety_clip: bool = False           # DEFAULT OFF
    safety_clip_center: str = "spot"          # "spot", "meanS", "mode"
    safety_clip_jump_factor: float = np.e     # e^1 threshold


# ============================================================
# Adapter to unify row-fit API across mixture/hermite
# ============================================================
class RowFitAdapter:
    """
    Unify per-row fit object interface.

    Required:
      chat(K_eval, r, T) -> call prices on K_eval

    Optional:
      qhat(S_eval) -> density on S_eval (only if model provides)
    """
    def __init__(self, fit_obj: Any, kind: str):
        self.fit_obj = fit_obj
        self.kind = kind

    def chat(self, K: np.ndarray, r: float, T: float) -> np.ndarray:
        if self.kind in ("mixture_fixed", "mixture_evolutionary"):
            return self.fit_obj.chat(K, r=r, T=T)
        return self.fit_obj.chat(K)  # hermite keeps params internally

    def qhat(self, S: np.ndarray) -> np.ndarray:
        if hasattr(self.fit_obj, "qhat"):
            return self.fit_obj.qhat(S)
        raise AttributeError("Row model does not expose qhat().")


# ============================================================
# Call surface estimator
# ============================================================
class CallSurfaceEstimator:
    """
    Generalized call surface estimator with pluggable row model OR global model.

    Outputs:
      - C_clean (final call surface)
      - iv_surface (BS IV)
      - rnd_surface (Breeden–Litzenberger)
      - cdf_surface (integral of rnd row-wise)
      - rnd_moments_table (log-return moments under Q)
    """

    def __init__(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        S0: float,
        r: float,
        config: Optional[SurfaceConfig] = None,
    ):
        self.strikes_obs = np.asarray(strikes, float)
        self.maturities = np.asarray(maturities, float)
        self.S0 = float(S0)
        self.r = float(r)

        self.config = config or SurfaceConfig()

        # dense strike grid is set during stage1 (row models) or during global fit
        self.strikes: Optional[np.ndarray] = None
        self.K_interp: Optional[np.ndarray] = None

        # row pipeline intermediates
        self.C_stage1: Optional[np.ndarray] = None
        self.T_interp: Optional[np.ndarray] = None
        self.C_interp: Optional[np.ndarray] = None
        self.C_model_final: Optional[np.ndarray] = None

        # final outputs
        self.C_clean: Optional[np.ndarray] = None
        self.iv_surface: Optional[np.ndarray] = None
        self.rnd_surface: Optional[np.ndarray] = None
        self.rnd_maturities: Optional[np.ndarray] = None
        self.rnd_moments_table: Optional[pd.DataFrame] = None

        # CDF surface outputs
        self.cdf_surface: Optional[np.ndarray] = None
        self.cdf_maturities: Optional[np.ndarray] = None

        # diagnostics storage for global models
        self.C_obs_surface: Optional[np.ndarray] = None
        self.C_fit_obs_surface: Optional[np.ndarray] = None  # fitted on observed grid

        # global model objects/params (optional)
        self.bates_model = None
        self.bates_params = None

        self.kou_model = None
        self.kou_params = None

        self.heston_model = None
        self.heston_params = None

        self.hkde_model = None
        self.hkde_params = None

        # Safety-clip diagnostics
        self.left_jump_clip_info: list = []
        self.safety_clip_status: str = "Unused"   # "Unused" or "Used"

    # ----------------------------
    # Utility: maturity grid
    # ----------------------------
    def _build_T_interp_grid(self) -> np.ndarray:
        T_orig = np.asarray(self.maturities, float)

        if not self.config.use_maturity_interp:
            self.T_interp = T_orig.copy()
            return self.T_interp

        T_min = float(T_orig.min())
        T_max = float(T_orig.max())
        dt_years = float(self.config.day_step) / 365.0

        n_steps = int(np.floor((T_max - T_min) / dt_years)) + 1
        T_interp = T_min + dt_years * np.arange(n_steps)

        if T_interp.size == 0 or T_interp[-1] < T_max:
            T_interp = np.append(T_interp, T_max)

        self.T_interp = np.asarray(T_interp, float)
        return self.T_interp

    # ----------------------------
    # Utility: dense strike grid
    # ----------------------------
    def _build_dense_strike_grid(self) -> np.ndarray:
        nK_obs = self.strikes_obs.size
        nK_fine = max(self.config.fine_strike_factor * nK_obs, nK_obs)

        low = float(self.strikes_obs.min())
        high = float(self.strikes_obs.max())
        ext = max(self.config.strike_extension, 0.0)

        K_min_ext = low * (1.0 - ext)
        K_max_ext = high * (1.0 + ext)
        if K_min_ext <= 0:
            K_min_ext = min(low * 0.2, 5)

        K_dense = np.linspace(K_min_ext, K_max_ext, nK_fine)
        self.strikes = K_dense
        self.K_interp = K_dense.copy()
        return K_dense

    # ----------------------------
    # ATM volatility helper
    # ----------------------------
    def _compute_atm_vol(self) -> Dict[str, float]:
        if self.iv_surface is None or self.strikes is None or self.T_interp is None:
            return {"atm_vol": np.nan, "atm_K": np.nan, "atm_T": np.nan}

        K = np.asarray(self.strikes, float)
        T = np.asarray(self.T_interp, float)
        iv = np.asarray(self.iv_surface, float)

        j = int(np.argmin(np.abs(K - self.S0)))
        i = int(np.argmin(np.abs(T - float(T.min()))))

        atm_vol = float(iv[i, j]) if np.isfinite(iv[i, j]) else np.nan
        return {"atm_vol": atm_vol, "atm_K": float(K[j]), "atm_T": float(T[i])}

    # ----------------------------
    # Moments table from RND surface
    # ----------------------------
    def compute_logreturn_moments_table(self) -> "pd.DataFrame":
        assert self.rnd_surface is not None
        assert self.rnd_maturities is not None
        assert self.strikes is not None

        K_full = np.asarray(self.strikes, float)
        rnd = np.asarray(self.rnd_surface, float)
        T_grid = np.asarray(self.rnd_maturities, float)

        rows = []
        for i, T in enumerate(T_grid):
            q_full = rnd[i, :]

            mask = (np.isfinite(K_full) & np.isfinite(q_full) & (q_full > 0.0))
            if mask.sum() < 5:
                rows.append({"T": float(T), "mean": np.nan, "var": np.nan, "vol": np.nan,
                             "vol_ann": np.nan, "skew": np.nan, "kurt": np.nan, "area_q": np.nan})
                continue

            K = K_full[mask]
            q = q_full[mask]

            area = float(np.trapz(q, K))
            if (not np.isfinite(area)) or area <= 0.0:
                rows.append({"T": float(T), "mean": np.nan, "var": np.nan, "vol": np.nan,
                             "vol_ann": np.nan, "skew": np.nan, "kurt": np.nan, "area_q": area})
                continue

            q = q / area

            F = self.S0 * np.exp(self.r * T)
            x = np.log(K / F)

            mean_x = float(np.trapz(x * q, K))
            var_x = float(np.trapz((x - mean_x) ** 2 * q, K))

            if var_x <= 0 or not np.isfinite(var_x):
                rows.append({"T": float(T), "mean": mean_x, "var": var_x, "vol": np.nan,
                             "vol_ann": np.nan, "skew": np.nan, "kurt": np.nan, "area_q": area})
                continue

            vol_x = float(np.sqrt(var_x))
            vol_ann = float(vol_x / np.sqrt(T)) if T > 0 else np.nan

            skew_x = float(np.trapz((x - mean_x) ** 3 * q, K) / (var_x ** 1.5))
            kurt_x = float(np.trapz((x - mean_x) ** 4 * q, K) / (var_x ** 2))

            rows.append({
                "T": float(T),
                "mean": mean_x,
                "var": var_x,
                "vol": vol_x,
                "vol_ann": vol_ann,
                "skew": skew_x,
                "kurt": kurt_x,
                "area_q": area,
            })

        df = pd.DataFrame(rows)
        df = df[["T", "mean", "var", "vol", "vol_ann", "skew", "kurt", "area_q"]]
        self.rnd_moments_table = df
        return df

    # ============================================================
    # Global model fits (Bates / Kou / Heston / HKDE)
    # ============================================================
    def fit_global_bates(self, C_surface: np.ndarray):
        C_surface = np.asarray(C_surface, float)
        nT, nK = C_surface.shape
        if nT != self.maturities.size or nK != self.strikes_obs.size:
            raise ValueError("C_surface shape must be (len(maturities), len(strikes)).")

        cfg = self.config.row_model

        KK, TT = np.meshgrid(self.strikes_obs, self.maturities)
        K_obs = KK.ravel()
        T_obs = TT.ravel()
        C_obs = C_surface.ravel()

        mask = np.isfinite(C_obs)
        K_obs, T_obs, C_obs = K_obs[mask], T_obs[mask], C_obs[mask]

        if C_obs.size < 10:
            raise ValueError("Not enough finite option prices to fit Bates globally.")

        self._build_dense_strike_grid()

        names = ["kappa", "theta", "sigma", "v0", "rho", "lam", "muJ", "sigJ"]

        x0 = None
        if cfg.bates_x0 is not None:
            if isinstance(cfg.bates_x0, dict):
                x0 = np.array([cfg.bates_x0[n] for n in names], float)
            else:
                x0 = np.asarray(cfg.bates_x0, float).ravel()

        bounds = None
        if cfg.bates_bounds is not None:
            lb_d, ub_d = cfg.bates_bounds
            lb = np.array([lb_d[n] for n in names], float)
            ub = np.array([ub_d[n] for n in names], float)
            bounds = (lb, ub)

        if cfg.bates_fixed is not None:
            if bounds is None:
                lb = np.array([1e-4, 1e-8, 1e-6, 1e-8, -0.999, 0.0, -2.0, 1e-6], float)
                ub = np.array([50.0, 2.0, 10.0, 2.0, 0.999, 10.0, 2.0, 2.0], float)
                bounds = (lb, ub)
            lb, ub = bounds
            for k, v in cfg.bates_fixed.items():
                j = names.index(k)
                lb[j] = float(v)
                ub[j] = float(v)
            bounds = (lb, ub)

        mdl = BatesSurfaceFitter(u_max=float(cfg.bates_u_max), n_quad=int(cfg.bates_n_quad)).fit(
            S0=self.S0,
            K_obs=K_obs,
            T_obs=T_obs,
            C_obs=C_obs,
            r=self.r,
            q=0.0,
            x0=x0,
            bounds=bounds,
            weights=cfg.bates_weights,
            verbose=int(cfg.bates_verbose),
            max_nfev=int(cfg.bates_max_nfev),
        )

        self.bates_model = mdl
        self.bates_params = mdl.params_

        C_fit_obs = np.full((self.maturities.size, self.strikes_obs.size), np.nan, float)
        for i, T in enumerate(self.maturities):
            C_fit_obs[i, :] = mdl.chat(self.strikes_obs, np.array([float(T)]))[0]
        self.C_fit_obs_surface = C_fit_obs

        T_eval = self._build_T_interp_grid()
        self.C_clean = mdl.chat(self.strikes, T_eval)

        self.C_stage1 = None
        self.C_interp = None
        self.C_model_final = self.C_clean
        self.C_obs_surface = np.asarray(C_surface, float)

    def fit_global_kou(self, C_surface: np.ndarray):
        C_surface = np.asarray(C_surface, float)
        nT, nK = C_surface.shape
        if nT != self.maturities.size or nK != self.strikes_obs.size:
            raise ValueError("C_surface shape must be (len(maturities), len(strikes)).")

        cfg = self.config.row_model

        KK, TT = np.meshgrid(self.strikes_obs, self.maturities)
        K_obs = KK.ravel()
        T_obs = TT.ravel()
        C_obs = C_surface.ravel()

        mask = np.isfinite(C_obs)
        K_obs, T_obs, C_obs = K_obs[mask], T_obs[mask], C_obs[mask]

        if C_obs.size < 10:
            raise ValueError("Not enough finite option prices to fit Kou globally.")

        self._build_dense_strike_grid()

        mdl = KouJDModel(
            S0=self.S0,
            r=self.r,
            q=float(cfg.kou_q),
            quad_N=int(cfg.kou_quad_N),
            quad_u_max=float(cfg.kou_quad_u_max),
        )

        fit_p = mdl.fit(
            strikes_obs=K_obs,
            maturities_obs=T_obs,
            C_obs=C_obs,
            x0=cfg.kou_x0,
            bounds=cfg.kou_bounds,
            weights=cfg.kou_weights,
            verbose=int(cfg.kou_verbose),
            max_nfev=int(cfg.kou_max_nfev),
        )

        self.kou_model = mdl
        self.kou_params = fit_p

        C_fit_obs = np.full((self.maturities.size, self.strikes_obs.size), np.nan, float)
        for i, T in enumerate(self.maturities):
            C_fit_obs[i, :] = mdl.callhat(self.strikes_obs, np.full_like(self.strikes_obs, float(T)))
        self.C_fit_obs_surface = C_fit_obs

        T_eval = self._build_T_interp_grid()
        C_clean = np.empty((T_eval.size, self.strikes.size), float)
        for i, T in enumerate(T_eval):
            C_clean[i, :] = mdl.callhat(self.strikes, np.full_like(self.strikes, float(T)))
        self.C_clean = C_clean

        self.C_stage1 = None
        self.C_interp = None
        self.C_model_final = self.C_clean
        self.C_obs_surface = np.asarray(C_surface, float)

    def fit_global_heston(self, C_surface: np.ndarray):
        C_surface = np.asarray(C_surface, float)
        nT, nK = C_surface.shape
        if nT != self.maturities.size or nK != self.strikes_obs.size:
            raise ValueError("C_surface shape must be (len(maturities), len(strikes)).")

        cfg = self.config.row_model

        KK, TT = np.meshgrid(self.strikes_obs, self.maturities)
        K_obs = KK.ravel()
        T_obs = TT.ravel()
        C_obs = C_surface.ravel()

        mask = np.isfinite(C_obs)
        K_obs, T_obs, C_obs = K_obs[mask], T_obs[mask], C_obs[mask]

        if C_obs.size < 10:
            raise ValueError("Not enough finite option prices to fit Heston globally.")

        self._build_dense_strike_grid()

        mdl = HestonModel(
            S0=self.S0,
            r=self.r,
            q=float(cfg.heston_q),
            quad_N=int(cfg.heston_quad_N),
            quad_u_max=float(cfg.heston_quad_u_max),
        )

        fit_p = mdl.fit(
            K_obs, T_obs, C_obs,
            x0=cfg.heston_x0,
            bounds=cfg.heston_bounds,
            weights=cfg.heston_weights,
            verbose=int(cfg.heston_verbose),
            max_nfev=int(cfg.heston_max_nfev),
        )

        self.heston_model = mdl
        self.heston_params = fit_p

        C_fit_obs = np.full((self.maturities.size, self.strikes_obs.size), np.nan, float)
        for i, T in enumerate(self.maturities):
            C_fit_obs[i, :] = mdl.callhat(self.strikes_obs, np.full_like(self.strikes_obs, float(T)))
        self.C_fit_obs_surface = C_fit_obs

        T_eval = self._build_T_interp_grid()
        C_clean = np.empty((T_eval.size, self.strikes.size), float)
        for i, T in enumerate(T_eval):
            C_clean[i, :] = mdl.callhat(self.strikes, np.full_like(self.strikes, float(T)))
        self.C_clean = C_clean

        self.C_stage1 = None
        self.C_interp = None
        self.C_model_final = self.C_clean
        self.C_obs_surface = np.asarray(C_surface, float)

    def fit_global_kou_heston(self, C_surface: np.ndarray):
        C_surface = np.asarray(C_surface, float)
        nT, nK = C_surface.shape
        if nT != self.maturities.size or nK != self.strikes_obs.size:
            raise ValueError("C_surface shape must be (len(maturities), len(strikes)).")

        cfg = self.config.row_model

        KK, TT = np.meshgrid(self.strikes_obs, self.maturities)
        K_obs = KK.ravel()
        T_obs = TT.ravel()
        C_obs = C_surface.ravel()

        mask = np.isfinite(C_obs)
        K_obs, T_obs, C_obs = K_obs[mask], T_obs[mask], C_obs[mask]

        if C_obs.size < 10:
            raise ValueError("Not enough finite option prices to fit HKDE globally.")

        self._build_dense_strike_grid()

        mdl = HKDEModel(S0=self.S0, r=self.r, q=float(cfg.hkde_q))

        if cfg.hkde_x0 is None:
            x0 = HKDEParams(
                v0=0.04, theta=0.04, kappa=2.0, sigma_v=0.5, rho=-0.5,
                lam=0.5, p_up=0.5, eta1=8.0, eta2=12.0
            )
        else:
            x0 = cfg.hkde_x0

        bounds = cfg.hkde_bounds
        if bounds is ...:
            bounds = None
        elif bounds is not None:
            lb, ub = bounds
            lb = np.asarray(lb, float).ravel()
            ub = np.asarray(ub, float).ravel()
            if lb.size != 9 or ub.size != 9:
                raise ValueError("hkde_bounds must be (lb, ub) with length-9 vectors.")
            bounds = (lb, ub)

        fit_p = mdl.fit_to_calls(
            K_obs=K_obs, T_obs=T_obs, C_obs=C_obs,
            x0=x0,
            bounds=bounds,

            iv_obs=cfg.hkde_iv_obs,
            use_vega_weights=bool(cfg.hkde_use_vega_weights),
            vega_floor=float(cfg.hkde_vega_floor),
            w_cap=float(cfg.hkde_w_cap),

            Umax=float(cfg.hkde_quad_u_max),
            n_quad=int(cfg.hkde_quad_N),
            verbose=int(cfg.hkde_verbose),
            max_nfev=int(cfg.hkde_max_nfev),
        )

        self.hkde_model = mdl
        self.hkde_params = fit_p

        C_fit_obs = np.full((self.maturities.size, self.strikes_obs.size), np.nan, float)
        for i, T in enumerate(self.maturities):
            C_fit_obs[i, :] = mdl.call_prices(
                self.strikes_obs, float(T), fit_p,
                Umax=float(cfg.hkde_quad_u_max), n_quad=int(cfg.hkde_quad_N)
            )
        self.C_fit_obs_surface = C_fit_obs

        T_eval = self._build_T_interp_grid()
        C_clean = np.empty((T_eval.size, self.strikes.size), float)
        for i, T in enumerate(T_eval):
            C_clean[i, :] = mdl.call_prices(
                self.strikes, float(T), fit_p,
                Umax=float(cfg.hkde_quad_u_max), n_quad=int(cfg.hkde_quad_N)
            )
        self.C_clean = C_clean

        self.C_stage1 = None
        self.C_interp = None
        self.C_model_final = self.C_clean
        self.C_obs_surface = np.asarray(C_surface, float)

    # ============================================================
    # Row-model fitter (mixtures/hermite)
    # ============================================================
    def _fit_row_model(
        self,
        K_row: np.ndarray,
        C_row: np.ndarray,
        T: float,
        seed_shift: int = 0,
    ) -> Optional[RowFitAdapter]:
        cfg = self.config.row_model

        K_row = np.asarray(K_row, float)
        C_row = np.asarray(C_row, float)
        mask = ~np.isnan(C_row)

        if mask.sum() < 4:
            return None

        K_valid = K_row[mask]
        C_valid = C_row[mask]

        if cfg.model == "mixture_fixed":
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

        if cfg.model == "mixture_evolutionary":
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

        if cfg.model == "hermite":
            n_c = int(cfg.hermite_order) - 2
            if cfg.hermite_c0 is None:
                c0 = None
            else:
                c0 = np.asarray(cfg.hermite_c0, float).reshape(-1)
                if c0.size != n_c:
                    raise ValueError(f"hermite_c0 must have length {n_c} for order={cfg.hermite_order}")

            herm = HermiteRNDRowModel(
                S0=self.S0,
                r=self.r,
                q=0.0,
                T=T,
                K=K_valid,
                C_mkt=C_valid,
                order=int(cfg.hermite_order),
                sigma0=float(cfg.hermite_sigma0),
                c0=c0,
                sigma_bounds=cfg.hermite_sigma_bounds,
                c_bounds=cfg.hermite_c_bounds,
                s_grid_mult=cfg.hermite_s_grid_mult,
                s_grid_size=int(cfg.hermite_s_grid_size),
                w_neg=float(cfg.hermite_w_neg),
                w_mass=float(cfg.hermite_w_mass),
                w_forward=float(cfg.hermite_w_forward),
                weights=None,
            )
            return RowFitAdapter(herm, "hermite")

        raise ValueError(f"Unknown row model for row fitting: {cfg.model}")

    # ============================================================
    # Stage 1 (row models): ALWAYS full estimation
    # ============================================================
    def stage1_rowmodel_fill(self, C_surface: np.ndarray) -> np.ndarray:
        C_surface = np.asarray(C_surface, float)
        nT, _nK_obs = C_surface.shape

        self._build_dense_strike_grid()
        assert self.strikes is not None

        C_stage1 = np.empty((nT, self.strikes.size), dtype=float)

        for i, T in enumerate(self.maturities):
            row_obs = C_surface[i, :]
            rowfit = self._fit_row_model(self.strikes_obs, row_obs, float(T), seed_shift=i)

            if rowfit is None:
                mask = ~np.isnan(row_obs)
                if mask.sum() >= 2:
                    C_stage1[i, :] = np.interp(self.strikes, self.strikes_obs[mask], row_obs[mask])
                elif mask.sum() == 1:
                    C_stage1[i, :] = row_obs[mask][0]
                else:
                    C_stage1[i, :] = np.nan
            else:
                C_stage1[i, :] = rowfit.chat(self.strikes, r=self.r, T=float(T))

        self.C_stage1 = C_stage1
        return C_stage1

    # ============================================================
    # Stage 2 (row models): maturity spline
    # ============================================================
    def stage2_maturity_spline(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.C_stage1 is not None
        assert self.strikes is not None

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

            for i0, T0 in enumerate(T_orig):
                if np.isnan(y[i0]):
                    continue
                k = int(np.argmin(np.abs(T_interp - T0)))
                col[k] = y[i0]

            C_interp[:, j] = col

        self.T_interp = T_interp
        self.C_interp = C_interp
        return T_interp, C_interp

    # ============================================================
    # Stage 3 (row models): re-fit on interpolated maturities
    # ============================================================
    def stage3_refit_rowmodel_on_interp(self) -> np.ndarray:
        assert self.T_interp is not None and self.C_interp is not None
        assert self.strikes is not None

        C_interp = np.asarray(self.C_interp, float)
        T_interp = np.asarray(self.T_interp, float)

        C_final = np.empty_like(C_interp)
        for i, T in enumerate(T_interp):
            row = C_interp[i, :]
            rowfit = self._fit_row_model(self.strikes, row, float(T), seed_shift=1000 + i)

            if rowfit is None:
                C_final[i, :] = row
            else:
                C_final[i, :] = rowfit.chat(self.strikes, r=self.r, T=float(T))

        self.C_model_final = C_final
        return C_final

    # ============================================================
    # RND, CDF, IV (from C_clean)
    # ============================================================
    def _clip_left_if_exp_jump(
        self,
        qi: np.ndarray,
        K: np.ndarray,
        eps: float,
        *,
        center: str = "spot",     # "spot" or "meanS" or "mode"
        jump: float = np.e,       # e^1
        min_floor: float = 1e-30  # avoid division by ~0
    ):
        """
        Walk left from center. If qi[left] / qi[right] > jump at any step,
        clip that index and everything left to eps.

        Returns (qi_clipped, info)
        """
        qi = np.asarray(qi, float).copy()
        K = np.asarray(K, float)

        n = qi.size
        info = {"clipped": False, "cut_idx": None, "K_cut": None, "jump": float(jump), "center": center}

        if n < 5:
            return qi, info

        if center == "spot":
            ic = int(np.argmin(np.abs(K - float(self.S0))))
        elif center == "mode":
            ic = int(np.argmax(qi))
        elif center == "meanS":
            area = float(np.trapz(qi, K))
            if not (np.isfinite(area) and area > 0):
                return qi, info
            qn = qi / area
            meanS = float(np.trapz(K * qn, K))
            ic = int(np.argmin(np.abs(K - meanS)))
            info["meanS"] = meanS
        else:
            raise ValueError("center must be 'spot', 'mode', or 'meanS'")

        for j in range(ic - 1, -1, -1):
            right = max(qi[j + 1], min_floor)
            left = max(qi[j], min_floor)
            ratio = left / right

            if ratio > jump:
                qi[: j + 1] = eps
                info.update({"clipped": True, "cut_idx": j, "K_cut": float(K[j]), "ratio": float(ratio)})
                return qi, info

        return qi, info

    def compute_rnd_surface(
        self,
        eps: float = 1e-12,
        *,
        apply_safety_clip: Optional[bool] = None,
        jump_factor: Optional[float] = None,
        center: Optional[str] = None,
    ) -> np.ndarray:
        """
        Breeden–Litzenberger:
          q_T(K) = exp(r T) * d^2 C(T,K) / dK^2

        Post-processing:
          - Replace NaN/inf/<=0 values with eps
          - Optional left-tail safety clip (config-controlled)
        """
        assert self.C_clean is not None and self.T_interp is not None
        assert self.strikes is not None

        C = np.asarray(self.C_clean, float)
        T_grid = np.asarray(self.T_interp, float)
        K = np.asarray(self.strikes, float)

        if apply_safety_clip is None:
            apply_safety_clip = bool(getattr(self.config, "apply_safety_clip", False))
        if jump_factor is None:
            jump_factor = float(getattr(self.config, "safety_clip_jump_factor", np.e))
        if center is None:
            center = str(getattr(self.config, "safety_clip_center", "spot"))

        rnd = np.empty_like(C)

        self.left_jump_clip_info = []
        self.safety_clip_status = "Unused"

        for i, T in enumerate(T_grid):
            dC_dK = np.gradient(C[i, :], K, edge_order=2)
            d2C_dK2 = np.gradient(dC_dK, K, edge_order=2)
            qi = np.exp(self.r * float(T)) * d2C_dK2
            qi = np.where(np.isfinite(qi) & (qi > eps), qi, eps)

            if apply_safety_clip:
                qi, info = self._clip_left_if_exp_jump(
                    qi, K, eps,
                    center=center,
                    jump=float(jump_factor),
                )
                self.left_jump_clip_info.append({"T": float(T), **info})
                if info.get("clipped", False):
                    self.safety_clip_status = "Used"
            else:
                self.left_jump_clip_info.append({"T": float(T), "clipped": False, "cut_idx": None, "K_cut": None})

            rnd[i, :] = qi

        self.rnd_surface = rnd
        self.rnd_maturities = T_grid
        return rnd

    def compute_cdf_surface(self, *, renormalize: bool = True) -> np.ndarray:
        """
        Compute and store CDF surface from self.rnd_surface.

        CDF_i(K_j) = ∫_{K_min}^{K_j} q_i(u) du

        If renormalize=True:
          each row q_i is renormalized on the strike support so final CDF ends near 1.
        """
        assert self.rnd_surface is not None and self.rnd_maturities is not None
        assert self.strikes is not None

        K = np.asarray(self.strikes, float)
        rnd = np.asarray(self.rnd_surface, float)

        cdf = np.empty_like(rnd)

        for i in range(rnd.shape[0]):
            q = np.asarray(rnd[i, :], float)

            if renormalize:
                area = float(np.trapz(q, K))
                if np.isfinite(area) and area > 0:
                    q = q / area

            out = np.zeros_like(q)
            for j in range(1, q.size):
                out[j] = out[j - 1] + 0.5 * (q[j] + q[j - 1]) * (K[j] - K[j - 1])

            out = np.clip(out, 0.0, 1.0)
            cdf[i, :] = out

        self.cdf_surface = cdf
        self.cdf_maturities = np.asarray(self.rnd_maturities, float)
        return cdf

    def compute_iv_surface(self) -> np.ndarray:
        assert self.C_clean is not None and self.T_interp is not None
        assert self.strikes is not None

        C = np.asarray(self.C_clean, float)
        T_grid = np.asarray(self.T_interp, float)
        K_grid = np.asarray(self.strikes, float)

        iv = np.empty_like(C)
        for i, T in enumerate(T_grid):
            for j, K in enumerate(K_grid):
                iv[i, j] = bs_implied_vol(C[i, j], self.S0, float(K), self.r, float(T))

        self.iv_surface = iv
        return iv

    # ============================================================
    # Master method
    # ============================================================
    def fit_surface(self, C_surface: np.ndarray) -> Dict[str, Any]:
        """
        Fit according to config.row_model.model.
        Returns a dict of outputs for convenience.

        Added outputs:
          - r
          - S0
          - atm_vol (and atm_K, atm_T)
          - safety_clip_status
          - cdf_surface
        """
        C_surface = np.asarray(C_surface, float)
        self.C_obs_surface = C_surface.copy()

        # ---- Global branches ----
        if self.config.row_model.model == "bates":
            self.fit_global_bates(C_surface)
            self.compute_iv_surface()
            self.compute_rnd_surface()
            self.compute_cdf_surface()
            self.compute_logreturn_moments_table()
            atm = self._compute_atm_vol()
            return {
                "S0": self.S0,
                "r": self.r,
                **atm,
                "safety_clip_status": self.safety_clip_status,
                "K_interp": self.K_interp,
                "T_interp": self.T_interp,
                "C_clean": self.C_clean,
                "iv_surface": self.iv_surface,
                "rnd_surface": self.rnd_surface,
                "cdf_surface": self.cdf_surface,
                "rnd_moments_table": self.rnd_moments_table,
                "bates_params": self.bates_params,
                "C_fit_obs_surface": self.C_fit_obs_surface,
            }

        if self.config.row_model.model == "kou_heston":
            self.fit_global_kou_heston(C_surface)
            self.compute_iv_surface()
            self.compute_rnd_surface()
            self.compute_cdf_surface()
            self.compute_logreturn_moments_table()
            atm = self._compute_atm_vol()
            return {
                "S0": self.S0,
                "r": self.r,
                **atm,
                "safety_clip_status": self.safety_clip_status,
                "K_interp": self.K_interp,
                "T_interp": self.T_interp,
                "C_clean": self.C_clean,
                "iv_surface": self.iv_surface,
                "rnd_surface": self.rnd_surface,
                "cdf_surface": self.cdf_surface,
                "rnd_moments_table": self.rnd_moments_table,
                "hkde_params": self.hkde_params,
                "C_fit_obs_surface": self.C_fit_obs_surface,
            }

        if self.config.row_model.model == "kou":
            self.fit_global_kou(C_surface)
            self.compute_iv_surface()
            self.compute_rnd_surface()
            self.compute_cdf_surface()
            self.compute_logreturn_moments_table()
            atm = self._compute_atm_vol()
            return {
                "S0": self.S0,
                "r": self.r,
                **atm,
                "safety_clip_status": self.safety_clip_status,
                "K_interp": self.K_interp,
                "T_interp": self.T_interp,
                "C_clean": self.C_clean,
                "iv_surface": self.iv_surface,
                "rnd_surface": self.rnd_surface,
                "cdf_surface": self.cdf_surface,
                "rnd_moments_table": self.rnd_moments_table,
                "kou_params": self.kou_params,
                "C_fit_obs_surface": self.C_fit_obs_surface,
            }

        if self.config.row_model.model == "heston":
            self.fit_global_heston(C_surface)
            self.compute_iv_surface()
            self.compute_rnd_surface()
            self.compute_cdf_surface()
            self.compute_logreturn_moments_table()
            atm = self._compute_atm_vol()
            return {
                "S0": self.S0,
                "r": self.r,
                **atm,
                "safety_clip_status": self.safety_clip_status,
                "K_interp": self.K_interp,
                "T_interp": self.T_interp,
                "C_clean": self.C_clean,
                "iv_surface": self.iv_surface,
                "rnd_surface": self.rnd_surface,
                "cdf_surface": self.cdf_surface,
                "rnd_moments_table": self.rnd_moments_table,
                "heston_params": self.heston_params,
                "C_fit_obs_surface": self.C_fit_obs_surface,
            }

        # ---- Row-model pipeline (mixtures / hermite) ----
        self.stage1_rowmodel_fill(C_surface)
        self.stage2_maturity_spline()
        self.stage3_refit_rowmodel_on_interp()

        self.C_clean = self.C_model_final
        self.compute_iv_surface()
        self.compute_rnd_surface()
        self.compute_cdf_surface()
        self.compute_logreturn_moments_table()
        atm = self._compute_atm_vol()

        return {
            "S0": self.S0,
            "r": self.r,
            **atm,
            "safety_clip_status": self.safety_clip_status,
            "K_interp": self.K_interp,
            "T_interp": self.T_interp,
            "C_stage1": self.C_stage1,
            "C_interp": self.C_interp,
            "C_model_final": self.C_model_final,
            "C_clean": self.C_clean,
            "iv_surface": self.iv_surface,
            "rnd_surface": self.rnd_surface,
            "cdf_surface": self.cdf_surface,
            "rnd_moments_table": self.rnd_moments_table,
        }

    # ============================================================
    # Plotting helpers (final surfaces + RND/CDF curves)
    # ============================================================
    def plot_call_and_iv_surfaces(
        self,
        save: Optional[Union[str, os.PathLike]] = None,
        dpi: int = 200,
    ):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        assert self.C_clean is not None and self.iv_surface is not None
        assert self.strikes is not None and self.T_interp is not None

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

        _add_safety_clip_note(fig, self.safety_clip_status, where="br")

        plt.tight_layout()
        _maybe_savefig(fig, save, default_name="call_and_iv_surfaces", dpi=dpi)
        plt.show()

    def plot_some_rnds(
        self,
        n_curves: int = 3,
        layout: str = "overlay",   # "overlay" or "panels"
        ncols: int = 2,
        save: Optional[Union[str, os.PathLike]] = None,
        dpi: int = 200,
    ):
        assert self.rnd_surface is not None and self.rnd_maturities is not None
        assert self.strikes is not None

        rnd = self.rnd_surface
        T_grid = self.rnd_maturities
        K = self.strikes

        nT = T_grid.size
        idxs = np.linspace(0, nT - 1, min(n_curves, nT), dtype=int)

        if layout == "overlay":
            fig = plt.figure(figsize=(7, 5))
            for idx in idxs:
                plt.plot(K, rnd[idx, :], label=f"T={T_grid[idx]:.3f} yr")
            plt.title("Risk-neutral densities")
            plt.xlabel("Terminal price s = K")
            plt.ylabel("q_T(s)")
            plt.legend()
            _add_safety_clip_note(fig, self.safety_clip_status, where="br")
            plt.tight_layout()
            _maybe_savefig(fig, save, default_name="rnd_curves_overlay", dpi=dpi)
            plt.show()
            return

        if layout == "panels":
            n_panels = len(idxs)
            nrows = int(np.ceil(n_panels / ncols))

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(3.8 * ncols, 3.0 * nrows),
                sharex=True,
                sharey=True,
            )

            axes = np.atleast_1d(axes).ravel()

            for ax, idx in zip(axes, idxs):
                ax.plot(K, rnd[idx, :], lw=2)
                ax.set_title(f"T = {T_grid[idx]:.3f} yr")
                ax.set_xlabel("s = K")
                ax.set_ylabel("q_T(s)")
                ax.grid(alpha=0.3)

            for ax in axes[len(idxs):]:
                ax.axis("off")

            fig.suptitle("Risk-neutral densities by maturity", y=0.98)
            _add_safety_clip_note(fig, self.safety_clip_status, where="br")
            plt.tight_layout()
            _maybe_savefig(fig, save, default_name="rnd_curves_panels", dpi=dpi)
            plt.show()
            return

        raise ValueError("layout must be 'overlay' or 'panels'")

    def plot_rnd_surface(
        self,
        title: str = "Risk-neutral density surface",
        save: Optional[Union[str, os.PathLike]] = None,
        dpi: int = 200,
    ):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib.colors import LightSource

        assert self.rnd_surface is not None and self.rnd_maturities is not None
        assert self.strikes is not None

        K = self.strikes
        T = self.rnd_maturities
        KK, TT = np.meshgrid(K, T)
        Z = self.rnd_surface

        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(Z, cmap=plt.cm.viridis, vert_exag=1.0, blend_mode="overlay")

        fig = plt.figure(figsize=(8.5, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title, pad=14)

        ax.plot_surface(
            KK, TT, Z,
            facecolors=rgb,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        ax.set_xlabel("Strike K", labelpad=10)
        ax.set_ylabel("Maturity T", labelpad=12)

        ax.set_zlabel("")
        ax.text2D(0.98, 0.52, "q(K)", transform=ax.transAxes,
                  rotation=90, va="center", ha="right")

        ax.zaxis.set_tick_params(pad=6)
        ax.view_init(elev=30, azim=-60)
        plt.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.92)

        _add_safety_clip_note(fig, self.safety_clip_status, where="br")
        _maybe_savefig(fig, save, default_name=_slugify(title) + "_rnd_surface", dpi=dpi)
        plt.show()

    def plot_cdf_surface(
        self,
        title: str = "CDF surface",
        save: Optional[Union[str, os.PathLike]] = None,
        dpi: int = 200,
    ):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        assert self.cdf_surface is not None and self.cdf_maturities is not None
        assert self.strikes is not None

        K = self.strikes
        T = self.cdf_maturities
        KK, TT = np.meshgrid(K, T)
        Z = self.cdf_surface

        fig = plt.figure(figsize=(8.5, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title, pad=14)

        ax.plot_surface(KK, TT, Z, rstride=1, cstride=1, linewidth=0.2)

        ax.set_xlabel("Strike K", labelpad=10)
        ax.set_ylabel("Maturity T", labelpad=12)
        ax.set_zlabel("CDF", labelpad=10)

        ax.view_init(elev=30, azim=-60)
        _add_safety_clip_note(fig, self.safety_clip_status, where="br")
        plt.tight_layout()
        _maybe_savefig(fig, save, default_name=_slugify(title) + "_cdf_surface", dpi=dpi)
        plt.show()

    def plot_some_cdfs(
        self,
        n_curves: int = 3,
        layout: str = "overlay",   # "overlay" or "panels"
        ncols: int = 2,
        save: Optional[Union[str, os.PathLike]] = None,
        dpi: int = 200,
    ):
        assert self.cdf_surface is not None and self.cdf_maturities is not None
        assert self.strikes is not None

        cdf = self.cdf_surface
        T_grid = self.cdf_maturities
        K = self.strikes

        nT = T_grid.size
        idxs = np.linspace(0, nT - 1, min(n_curves, nT), dtype=int)

        if layout == "overlay":
            fig = plt.figure(figsize=(7, 5))
            for idx in idxs:
                plt.plot(K, cdf[idx, :], label=f"T={T_grid[idx]:.3f} yr")
            plt.title("CDFs by maturity")
            plt.xlabel("Terminal price s = K")
            plt.ylabel("F_T(s)")
            plt.ylim(0, 1.02)
            plt.legend()
            _add_safety_clip_note(fig, self.safety_clip_status, where="br")
            plt.tight_layout()
            _maybe_savefig(fig, save, default_name="cdf_curves_overlay", dpi=dpi)
            plt.show()
            return

        if layout == "panels":
            n_panels = len(idxs)
            nrows = int(np.ceil(n_panels / ncols))

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(3.8 * ncols, 3.0 * nrows),
                sharex=True,
                sharey=True,
            )

            axes = np.atleast_1d(axes).ravel()

            for ax, idx in zip(axes, idxs):
                ax.plot(K, cdf[idx, :], lw=2)
                ax.set_title(f"T = {T_grid[idx]:.3f} yr")
                ax.set_xlabel("s = K")
                ax.set_ylabel("F_T(s)")
                ax.set_ylim(0, 1.02)
                ax.grid(alpha=0.3)

            for ax in axes[len(idxs):]:
                ax.axis("off")

            fig.suptitle("CDFs by maturity", y=0.98)
            _add_safety_clip_note(fig, self.safety_clip_status, where="br")
            plt.tight_layout()
            _maybe_savefig(fig, save, default_name="cdf_curves_panels", dpi=dpi)
            plt.show()
            return

        raise ValueError("layout must be 'overlay' or 'panels'")


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
# Minimal toy example
# ============================================================
if __name__ == "__main__":
    S0 = 140.0
    r = 0.02
    strikes = np.linspace(100, 140, 20)
    maturities = np.array([0.25, 0.5, 1.0, 1.1, 1.2, 1.3])
    sigmas = np.array([0.20, 0.22, 0.24, 0.26, 0.28, 0.29])

    # synthetic "market" calls from BS
    C_true = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        C_true[i, :] = bs_call_price(S0, strikes, r, sigmas[i], T)

    rng = np.random.default_rng(0)
    C_noisy = C_true + rng.normal(scale=0.02 * np.maximum(C_true, 1e-6))

    C = C_noisy.copy()
    C[1, 4] += 5.0
    C[2, 7] -= 3.0
    C[3, 2] -= 5.0
    C[0:2, 4] = np.nan
    C[2, 9] = np.nan
    C[3:6, 10] = np.nan

    # Pick ONE model (uncomment cfg and keep est line):
    cfg = SurfaceConfig(
        row_model=RowModelConfig(model="kou_heston", hkde_verbose=1),
        use_maturity_interp=True,
        day_step=14,
        strike_extension=0.5,
        fine_strike_factor=3,
        apply_safety_clip=True, ###Clips densities. If issue set to false
        safety_clip_center="spot",
        safety_clip_jump_factor=np.e, ###Clips density on left tail if it rises by more than 0.1
    )

    # For demo purposes, just use row mixture by default with safety clip OFF:
    # cfg = SurfaceConfig(
    #     row_model=RowModelConfig(model="mixture_evolutionary"),
    #     use_maturity_interp=True,
    #     day_step=7,
    #     strike_extension=0.8,
    #     fine_strike_factor=3,
    #     apply_safety_clip=False,  # set True if you want
    # )

    est = CallSurfaceEstimator(strikes=strikes, maturities=maturities, S0=S0, r=r, config=cfg)
    out = est.fit_surface(C)

    save_dir = None  # e.g. "./figs" or "C:/Users/you/Desktop/figs"

    est.plot_call_and_iv_surfaces(save=save_dir)
    est.plot_some_rnds(n_curves=6, layout="panels", save=save_dir)
    est.plot_rnd_surface(save=save_dir)

    est.plot_some_cdfs(n_curves=6, layout="panels", save=save_dir)
    est.plot_cdf_surface(save=save_dir)

    plot_random_observed_vs_model_curve(
        est=est,
        strikes_orig=strikes,
        maturities_orig=maturities,
        C_orig=C,
        plot_all_original=True,
        title_prefix="Observed vs Model Call Curve",
        save=save_dir,
    )

    plot_original_vs_final_surface(
        est=est,
        strikes_orig=strikes,
        maturities_orig=maturities,
        C_orig=C,
        title="Toy Surface: Observed vs Final Estimated Call Surface",
        save=save_dir,
    )
