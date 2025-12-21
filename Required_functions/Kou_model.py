"""
New version (Kou 2002): Double-Exponential Jump-Diffusion (no stochastic volatility)

USER INPUT STYLE (as requested):
  strikes_obs:    (N,)  array of strikes per quote
  maturities_obs: (N,)  array of maturities per quote (in years)
  C_obs:          (N,) or (N,1) column vector of call prices

Calibration:
  fit(strikes_obs, maturities_obs, C_obs)

Evaluation:
  callhat(strikes, maturities)  -> vector (same length as strikes/maturities)
  ivhat(strikes, maturities)    -> vector
  rndhat(K_grid, T_unique)      -> (nT, nK) surface using Breeden–Litzenberger

Bounds (NEW):
  - fit(..., bounds=None) uses DEFAULT DICT BOUNDS (yours)
  - fit(..., bounds=(lb,ub)) where lb/ub can be dicts OR vectors/tuples
  - Prints whether default bounds are used (when verbose)

Plots:
  plot_call_fit_panels()  -> panels per maturity (observed vs fitted calls)
  plot_rnd_panels()       -> fitted RND curves (no observed RND)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any, Union
from scipy.optimize import least_squares, brentq
from scipy.stats import norm
import matplotlib.pyplot as plt


# ============================================================
# Black–Scholes utilities
# ============================================================

def bs_call_price(S0, K, r, q, T, sigma):
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    sigma = np.asarray(sigma, float)
    tiny = 1e-16

    vol = np.maximum(sigma, 1e-12)
    sqrtT = np.sqrt(np.maximum(T, tiny))
    d1 = (np.log(np.maximum(S0, tiny) / np.maximum(K, tiny)) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT
    return np.exp(-q * T) * S0 * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)


def implied_vol_call(C, S0, K, r, q, T, vol_bounds=(1e-10, 5.0)):
    if T <= 0:
        return np.nan
    C = float(C)
    discS = np.exp(-q * T) * S0
    discK = np.exp(-r * T) * K
    intrinsic = max(discS - discK, 0.0)
    upper = discS
    if not (intrinsic - 1e-10 <= C <= upper + 1e-10):
        return np.nan

    f = lambda sig: bs_call_price(S0, K, r, q, T, sig) - C
    lo, hi = vol_bounds
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        for _ in range(8):
            hi *= 1.6
            fhi = f(hi)
            if flo * fhi <= 0:
                break
        else:
            return np.nan
    return brentq(f, lo, hi, maxiter=200)


# ============================================================
# Kou parameters
# ============================================================

@dataclass
class KouParams:
    sigma: float   # diffusion vol > 0
    lam: float     # jump intensity >= 0
    p_up: float    # prob of upward jump in (0,1)
    eta1: float    # rate up, must be > 1
    eta2: float    # rate down, > 0


# ============================================================
# Kou model class
# ============================================================

BoundsLike = Optional[
    Tuple[
        Union[Dict[str, float], Sequence[float], np.ndarray],
        Union[Dict[str, float], Sequence[float], np.ndarray],
    ]
]

class KouJDModel:
    """
    Kou (2002) Double-Exponential Jump-Diffusion.

    Call pricing uses CF-based P1/P2 representation:
      C = S0 e^{-qT} P1 - K e^{-rT} P2

    Requires eta1 > 1 so φ_X(u-i) exists (M_Z(1) finite).
    """

    # PARAMETER ORDER (for vectors)
    _PARAM_NAMES = ("sigma", "lam", "p_up", "eta1", "eta2")

    def __init__(
        self,
        S0: float,
        r: float,
        q: float = 0.0,
        quad_N: int = 96,
        quad_u_max: float = 150.0,
        tiny: float = 1e-14,
    ):
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self.quad_N = int(quad_N)
        self.quad_u_max = float(quad_u_max)
        self.tiny = float(tiny)

        self.params_: Optional[KouParams] = None

        # stored quote data
        self.K_obs: Optional[np.ndarray] = None
        self.T_obs: Optional[np.ndarray] = None
        self.C_obs: Optional[np.ndarray] = None
        self.C_fit: Optional[np.ndarray] = None

        # RND cache
        self.rnd_surface: Optional[np.ndarray] = None
        self.rnd_strikes: Optional[np.ndarray] = None
        self.rnd_maturities: Optional[np.ndarray] = None

        # Gauss–Legendre nodes/weights on [0, u_max]
        x, w = np.polynomial.legendre.leggauss(self.quad_N)
        u = 0.5 * (x + 1.0) * self.quad_u_max
        wu = 0.5 * self.quad_u_max * w
        self._u = u.astype(np.complex128)
        self._wu = wu.astype(np.float64)

    # --------------------------
    # DEFAULT BOUNDS (yours)
    # --------------------------
    @staticmethod
    def default_bounds_dict() -> Tuple[Dict[str, float], Dict[str, float]]:
        lb = {
            "sigma": 0.005,
            "lam":   0.02,
            "p_up":  0.15,
            "eta1":  4.0,
            "eta2":  4.0,
        }
        ub = {
            "sigma": 0.75,
            "lam":   1.50,
            "p_up":  0.85,
            "eta1":  30.0,
            "eta2":  30.0,
        }
        return lb, ub

    def _normalize_bounds(
        self,
        bounds: BoundsLike,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Returns:
          (lb_vec, ub_vec, using_default_bounds)
        Accepts:
          - bounds=None -> uses default dict bounds
          - bounds=(lb,ub) where lb/ub are dicts keyed by PARAM_NAMES
          - bounds=(lb,ub) where lb/ub are sequences/arrays of length 5 in PARAM_NAMES order
        """
        using_default = bounds is None
        if bounds is None:
            lb_d, ub_d = self.default_bounds_dict()
        else:
            lb_d, ub_d = bounds

        names = self._PARAM_NAMES

        # dict form
        if isinstance(lb_d, dict) or isinstance(ub_d, dict):
            if not (isinstance(lb_d, dict) and isinstance(ub_d, dict)):
                raise ValueError("If bounds are dict-form, BOTH lb and ub must be dicts.")
            missing = [k for k in names if (k not in lb_d or k not in ub_d)]
            if missing:
                raise KeyError(f"Bounds dict missing keys: {missing}. Required: {list(names)}")
            lb = np.array([float(lb_d[k]) for k in names], float)
            ub = np.array([float(ub_d[k]) for k in names], float)
            return lb, ub, using_default

        # vector/tuple form
        lb = np.asarray(lb_d, float).ravel()
        ub = np.asarray(ub_d, float).ravel()
        if lb.size != len(names) or ub.size != len(names):
            raise ValueError(f"Vector bounds must have length {len(names)} in order {names}.")
        return lb, ub, using_default

    @staticmethod
    def _pack_params(x: Sequence[float]) -> KouParams:
        return KouParams(
            sigma=float(x[0]),
            lam=float(x[1]),
            p_up=float(x[2]),
            eta1=float(x[3]),
            eta2=float(x[4]),
        )

    # --------------------------
    # Jump MGF / CF
    # --------------------------
    @staticmethod
    def M_Z(u: complex, p_up: float, eta1: float, eta2: float) -> complex:
        p = p_up
        q = 1.0 - p
        return p * (eta1 / (eta1 - u)) + q * (eta2 / (eta2 + u))

    @staticmethod
    def phi_Z(u: complex, p_up: float, eta1: float, eta2: float) -> complex:
        return KouJDModel.M_Z(1j * u, p_up, eta1, eta2)

    @staticmethod
    def kappa(p_up: float, eta1: float, eta2: float) -> float:
        return float(np.real(KouJDModel.M_Z(1.0, p_up, eta1, eta2) - 1.0))

    # --------------------------
    # CF of X_T = ln S_T
    # --------------------------
    def phi_X(self, u: np.ndarray, T: float, p: KouParams) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        sigma, lam, p_up, eta1, eta2 = p.sigma, p.lam, p.p_up, p.eta1, p.eta2
        kappa = self.kappa(p_up, eta1, eta2)
        m = np.log(self.S0) + (self.r - self.q - lam * kappa - 0.5 * sigma * sigma) * T
        jump_cf = self.phi_Z(u, p_up, eta1, eta2)
        return np.exp(1j * u * m - 0.5 * sigma * sigma * (u * u) * T + lam * T * (jump_cf - 1.0))

    # --------------------------
    # P1/P2 (vectorized over strikes for a fixed T)
    # --------------------------
    def _P2(self, K: np.ndarray, T: float, p: KouParams) -> np.ndarray:
        K = np.asarray(K, float)
        lnK = np.log(np.maximum(K, self.tiny))

        u = self._u
        phi = self.phi_X(u, T, p)

        expo = np.exp(-1j * u[:, None] * lnK[None, :])
        denom = 1j * u[:, None]
        val = np.real(expo * phi[:, None] / denom)
        integral = (self._wu[:, None] * val).sum(axis=0)
        return 0.5 + integral / np.pi

    def _P1(self, K: np.ndarray, T: float, p: KouParams) -> np.ndarray:
        K = np.asarray(K, float)
        lnK = np.log(np.maximum(K, self.tiny))

        u = self._u
        phi_shift = self.phi_X(u - 1j, T, p)
        phi_mi = self.phi_X(np.array([-1j], dtype=np.complex128), T, p)[0]  # φ(-i)

        expo = np.exp(-1j * u[:, None] * lnK[None, :])
        denom = (1j * u[:, None]) * phi_mi
        val = np.real(expo * phi_shift[:, None] / denom)
        integral = (self._wu[:, None] * val).sum(axis=0)
        return 0.5 + integral / np.pi

    def call_price_T(self, K: np.ndarray, T: float, p: KouParams) -> np.ndarray:
        K = np.asarray(K, float)
        P1 = self._P1(K, T, p)
        P2 = self._P2(K, T, p)
        return np.exp(-self.q * T) * self.S0 * P1 - np.exp(-self.r * T) * K * P2

    # --------------------------
    # Public evaluation on quote vectors
    # --------------------------
    def callhat(
        self,
        strikes: Sequence[float],
        maturities: Sequence[float],
        params: Optional[KouParams] = None,
    ) -> np.ndarray:
        p = params if params is not None else self.params_
        if p is None:
            raise ValueError("No params provided and model has not been fitted yet.")

        K = np.asarray(strikes, float).reshape(-1)
        T = np.asarray(maturities, float).reshape(-1)
        if K.shape != T.shape:
            raise ValueError("strikes and maturities must have the same shape for quote-vector input.")

        C = np.empty_like(K, dtype=float)
        Tu = np.unique(T)
        for tval in Tu:
            mask = (T == tval)
            C[mask] = self.call_price_T(K[mask], float(tval), p)
        return C

    def ivhat(
        self,
        strikes: Sequence[float],
        maturities: Sequence[float],
        params: Optional[KouParams] = None,
    ) -> np.ndarray:
        C = self.callhat(strikes, maturities, params=params)
        K = np.asarray(strikes, float).reshape(-1)
        T = np.asarray(maturities, float).reshape(-1)

        IV = np.full_like(C, np.nan, dtype=float)
        for i in range(C.size):
            IV[i] = implied_vol_call(float(C[i]), self.S0, float(K[i]), self.r, self.q, float(T[i]))
        return IV

    # --------------------------
    # Calibration on quote vectors (NOW: dict bounds + print default usage)
    # --------------------------
    def fit(
        self,
        strikes_obs: Sequence[float],
        maturities_obs: Sequence[float],
        C_obs: np.ndarray,
        x0: Optional[Tuple[float, float, float, float, float]] = None,
        bounds: BoundsLike = None,  # <-- NOW accepts dict bounds (or vectors), or None for default
        weights: Optional[np.ndarray] = None,
        verbose: int = 1,
        max_nfev: int = 400,
    ) -> KouParams:

        K = np.asarray(strikes_obs, float).reshape(-1)
        T = np.asarray(maturities_obs, float).reshape(-1)
        C_obs = np.asarray(C_obs, float).reshape(-1)
        if not (K.shape == T.shape == C_obs.shape):
            raise ValueError("strikes_obs, maturities_obs, and C_obs must be the same length.")

        self.K_obs, self.T_obs, self.C_obs = K.copy(), T.copy(), C_obs.copy()

        if x0 is None:
            x0 = (0.25, 0.4, 0.35, 8.0, 12.0)

        lb, ub, using_default_bounds = self._normalize_bounds(bounds)

        if verbose:
            print(f"[KouJDModel.fit] using_default_bounds = {using_default_bounds}")
            print(f"[KouJDModel.fit] bounds order = {self._PARAM_NAMES}")
            print(f"[KouJDModel.fit] lb = {lb}")
            print(f"[KouJDModel.fit] ub = {ub}")

        if weights is None:
            w = np.ones_like(C_obs, float)
        else:
            w = np.asarray(weights, float).reshape(-1)
            if w.shape != C_obs.shape:
                raise ValueError("weights must be same length as C_obs.")

        y = C_obs

        def resid(x):
            p = self._pack_params(x)
            # quick sanity region (avoid nonsense & integrability)
            if (p.sigma <= 0) or (p.lam < 0) or (p.p_up <= 0) or (p.p_up >= 1) or (p.eta1 <= 1.0) or (p.eta2 <= 0):
                return np.full_like(y, 1e6, float)
            C_fit = self.callhat(K, T, params=p)
            return np.sqrt(w) * (C_fit - y)

        res = least_squares(
            resid,
            x0=np.array(x0, float),
            bounds=(lb, ub),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=int(max_nfev),
            verbose=verbose,
        )

        self.params_ = self._pack_params(res.x)
        self.C_fit = self.callhat(self.K_obs, self.T_obs, params=self.params_)
        return self.params_

    # --------------------------
    # RND via Breeden–Litzenberger (needs uniform K_grid)
    # --------------------------
    def rndhat(
        self,
        K_grid: Sequence[float],
        T_grid: Sequence[float],
        params: Optional[KouParams] = None,
        enforce_nonneg: bool = True,
    ) -> np.ndarray:
        p = params if params is not None else self.params_
        if p is None:
            raise ValueError("No params provided and model has not been fitted yet.")

        K = np.asarray(K_grid, float).reshape(-1)
        Tgrid = np.asarray(T_grid, float).reshape(-1)

        dK = np.diff(K)
        if np.max(np.abs(dK - dK.mean())) > 1e-8 * max(1.0, np.abs(dK.mean())):
            raise ValueError("rndhat expects an (approximately) uniform K_grid.")

        C = np.zeros((Tgrid.size, K.size), float)
        for i, Tv in enumerate(Tgrid):
            C[i, :] = self.call_price_T(K, float(Tv), p)

        h = float(dK.mean())
        CKK = (C[:, 2:] - 2.0 * C[:, 1:-1] + C[:, :-2]) / (h * h)
        q_surf = np.full_like(C, np.nan, dtype=float)
        for i, Tv in enumerate(Tgrid):
            q_mid = np.exp(self.r * Tv) * CKK[i, :]
            if enforce_nonneg:
                q_mid = np.maximum(q_mid, 0.0)
            q_surf[i, 1:-1] = q_mid

        self.rnd_surface = q_surf
        self.rnd_strikes = K.copy()
        self.rnd_maturities = Tgrid.copy()
        return q_surf

    # --------------------------
    # Plot panels (group by maturity)
    # --------------------------
    def plot_call_fit_panels(self, max_panels: int = 6, title: str = "Observed vs fitted call curves"):
        if self.C_obs is None or self.C_fit is None:
            raise ValueError("Run fit() first.")

        K = self.K_obs
        T = self.T_obs
        Cobs = self.C_obs
        Cfit = self.C_fit

        Tu = np.unique(T)
        if Tu.size > max_panels:
            idxs = np.linspace(0, Tu.size - 1, max_panels, dtype=int)
            Tu_plot = Tu[idxs]
        else:
            Tu_plot = Tu

        ncols = 3
        nrows = int(np.ceil(Tu_plot.size / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), sharex=False)
        axes = np.atleast_1d(axes).ravel()

        for i, Tv in enumerate(Tu_plot):
            ax = axes[i]
            m = (T == Tv)
            order = np.argsort(K[m])
            ax.plot(K[m][order], Cobs[m][order], marker="o", linestyle="None", label="Observed")
            ax.plot(K[m][order], Cfit[m][order], linewidth=2.0, label="Fitted")
            ax.set_title(f"T={Tv:.4f} yr  (n={m.sum()})")
            ax.set_xlabel("Strike K")
            ax.set_ylabel("Call price")
            ax.grid(True, alpha=0.25)
            ax.legend()

        for ax in axes[Tu_plot.size:]:
            ax.axis("off")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_rnd_panels(self, max_panels: int = 6, title: str = "Fitted risk-neutral densities q_T(K)"):
        if self.rnd_surface is None or self.rnd_maturities is None or self.rnd_strikes is None:
            raise ValueError("Run rndhat() first.")

        q = self.rnd_surface
        T = self.rnd_maturities
        K = self.rnd_strikes

        if T.size > max_panels:
            idxs = np.linspace(0, T.size - 1, max_panels, dtype=int)
        else:
            idxs = np.arange(T.size)

        ncols = 3
        nrows = int(np.ceil(idxs.size / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), sharex=True)
        axes = np.atleast_1d(axes).ravel()

        for j, ti in enumerate(idxs):
            ax = axes[j]
            ax.plot(K, q[ti, :], linewidth=2.0)
            ax.set_title(f"T={T[ti]:.4f} yr")
            ax.set_xlabel("Terminal price s = K")
            ax.set_ylabel("q_T(s)")
            ax.grid(True, alpha=0.25)

        for ax in axes[idxs.size:]:
            ax.axis("off")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_rnd_surface_matplotlib(self, title: str = "Kou RND surface q_T(K)"):
        if self.rnd_surface is None or self.rnd_maturities is None or self.rnd_strikes is None:
            raise ValueError("Run rndhat(K_grid, T_grid) first.")

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        K = self.rnd_strikes
        T = self.rnd_maturities
        Z = self.rnd_surface

        KK, TT = np.meshgrid(K, T)

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")

        Zm = np.ma.masked_invalid(Z)
        surf = ax.plot_surface(KK, TT, Zm, rstride=1, cstride=1, linewidth=0, antialiased=True)

        ax.set_title(title)
        ax.set_xlabel("Strike / terminal price K")
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel("q_T(K)")
        ax.view_init(elev=25, azim=-135)

        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="q_T(K)")
        plt.tight_layout()
        plt.show()

    def plot_rnd_surface_interactive(self, title: str = "Kou RND surface q_T(K)"):
        if self.rnd_surface is None or self.rnd_maturities is None or self.rnd_strikes is None:
            raise ValueError("Run rndhat(K_grid, T_grid) first.")

        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("Plotly is not installed. Run: pip install plotly") from e

        K = self.rnd_strikes
        T = self.rnd_maturities
        Z = self.rnd_surface

        fig = go.Figure(data=[go.Surface(x=K, y=T, z=np.nan_to_num(Z, nan=0.0))])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Strike / terminal price K",
                yaxis_title="Maturity T (years)",
                zaxis_title="q_T(K)",
            ),
            width=950,
            height=650,
        )
        fig.show()


# ============================================================
# Demo: synthetic calibration test with quote-vector input
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    S0, r, q = 100.0, 0.02, 0.0
    model = KouJDModel(S0=S0, r=r, q=q, quad_N=120, quad_u_max=180.0)

    strikes_grid = np.linspace(60, 140, 33)
    maturities_grid = np.array([0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00])

    KK, TT = np.meshgrid(strikes_grid, maturities_grid)
    K_obs = KK.ravel()
    T_obs = TT.ravel()

    true_p = KouParams(sigma=0.20, lam=0.60, p_up=0.30, eta1=10.0, eta2=15.0)

    C_true = model.callhat(K_obs, T_obs, params=true_p)

    rng = np.random.default_rng(1)
    C_obs = np.maximum(C_true + rng.normal(scale=0.01, size=C_true.shape), 0.0)

    # ---- Example 1: DEFAULT bounds (prints using_default_bounds=True)
    fit_p = model.fit(K_obs, T_obs, C_obs, verbose=1)

    # ---- Example 2: custom DICT bounds (prints using_default_bounds=False)
    # lb_d, ub_d = KouJDModel.default_bounds_dict()
    # lb_d["sigma"] = 0.01
    # ub_d["lam"] = 2.0
    # fit_p = model.fit(K_obs, T_obs, C_obs, bounds=(lb_d, ub_d), verbose=1)

    print("\nTrue params:", true_p)
    print("Fitted params:", fit_p)

    model.plot_call_fit_panels(max_panels=6)

    K_rnd = np.linspace(50, 160, 301)
    T_unique = np.unique(T_obs)
    model.rndhat(K_rnd, T_unique, enforce_nonneg=True)

    model.plot_rnd_panels(max_panels=6)
    model.plot_rnd_surface_matplotlib()
    model.plot_rnd_surface_interactive()
