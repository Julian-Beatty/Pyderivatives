"""
New version (Heston 1993): stochastic volatility (no jumps)

Bounds (NEW):
  - fit(..., bounds=None) uses DEFAULT DICT BOUNDS (your requested defaults)
  - fit(..., bounds=(lb,ub)) where lb/ub can be dicts OR vectors/tuples
  - Prints whether default bounds are used (when verbose)
  - NEW: auto-clip x0 into bounds (and prints what it clipped)
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
# Parameters
# ============================================================

@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma: float   # vol-of-vol
    v0: float
    rho: float


BoundsLike = Optional[
    Tuple[
        Union[Dict[str, float], Sequence[float], np.ndarray],
        Union[Dict[str, float], Sequence[float], np.ndarray],
    ]
]


# ============================================================
# Heston Model
# ============================================================

class HestonModel:
    _PARAM_NAMES = ("kappa", "theta", "sigma", "v0", "rho")

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

        self.params_: Optional[HestonParams] = None

        self.K_obs: Optional[np.ndarray] = None
        self.T_obs: Optional[np.ndarray] = None
        self.C_obs: Optional[np.ndarray] = None
        self.C_fit: Optional[np.ndarray] = None

        self.rnd_surface: Optional[np.ndarray] = None
        self.rnd_strikes: Optional[np.ndarray] = None
        self.rnd_maturities: Optional[np.ndarray] = None

        x, w = np.polynomial.legendre.leggauss(self.quad_N)
        u = 0.5 * (x + 1.0) * self.quad_u_max
        wu = 0.5 * self.quad_u_max * w
        self._u = u.astype(np.complex128)
        self._wu = wu.astype(np.float64)

    # --------------------------
    # DEFAULT BOUNDS (your requested defaults)
    # --------------------------
    @staticmethod
    def default_bounds_dict() -> Tuple[Dict[str, float], Dict[str, float]]:
        lb = {"kappa": 0.30, "theta": 0.005, "sigma": 0.10, "v0": 0.005, "rho": -0.85}
        ub = {"kappa": 12.0, "theta": 0.90,  "sigma": 0.75, "v0": 0.90,  "rho":  0.80}
        return lb, ub

    def _normalize_bounds(self, bounds: BoundsLike) -> Tuple[np.ndarray, np.ndarray, bool]:
        using_default = bounds is None
        if bounds is None:
            lb_d, ub_d = self.default_bounds_dict()
        else:
            lb_d, ub_d = bounds

        names = self._PARAM_NAMES

        if isinstance(lb_d, dict) or isinstance(ub_d, dict):
            if not (isinstance(lb_d, dict) and isinstance(ub_d, dict)):
                raise ValueError("If bounds are dict-form, BOTH lb and ub must be dicts.")
            missing = [k for k in names if (k not in lb_d or k not in ub_d)]
            if missing:
                raise KeyError(f"Bounds dict missing keys: {missing}. Required: {list(names)}")
            lb = np.array([float(lb_d[k]) for k in names], float)
            ub = np.array([float(ub_d[k]) for k in names], float)
        else:
            lb = np.asarray(lb_d, float).ravel()
            ub = np.asarray(ub_d, float).ravel()
            if lb.size != len(names) or ub.size != len(names):
                raise ValueError(f"Vector bounds must have length {len(names)} in order {names}.")

        if np.any(lb >= ub):
            bad = [names[i] for i in range(len(names)) if not (lb[i] < ub[i])]
            raise ValueError(f"Each lb must be strictly < ub. Bad: {bad}")

        return lb, ub, using_default

    @staticmethod
    def _clip_x0_into_bounds(x0: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x0 = np.asarray(x0, float).copy()
        eps = 1e-12
        x0_min = lb + eps
        x0_max = ub - eps
        x0_clipped = np.minimum(np.maximum(x0, x0_min), x0_max)
        changed = (x0_clipped != x0)
        return x0_clipped, changed

    # --------------------------
    # Stable Heston CF terms
    # --------------------------
    @staticmethod
    def _CD_terms(u: np.ndarray, T: float, p: HestonParams) -> Tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=np.complex128)
        iu = 1j * u

        kappa, theta, sigma, v0, rho = p.kappa, p.theta, p.sigma, p.v0, p.rho
        a = kappa * theta
        d = np.sqrt((rho * sigma * iu - kappa) ** 2 + (sigma ** 2) * (iu + u * u))

        denom = (kappa - rho * sigma * iu + d)
        denom = np.where(np.abs(denom) < 1e-30, 1e-30 + 0j, denom)
        g = (kappa - rho * sigma * iu - d) / denom

        expmdT = np.exp(-d * T)
        one_minus_gexp = 1.0 - g * expmdT
        one_minus_g = 1.0 - g
        one_minus_gexp = np.where(np.abs(one_minus_gexp) < 1e-30, 1e-30 + 0j, one_minus_gexp)
        one_minus_g = np.where(np.abs(one_minus_g) < 1e-30, 1e-30 + 0j, one_minus_g)

        D = ((kappa - rho * sigma * iu - d) / (sigma ** 2)) * ((1.0 - expmdT) / one_minus_gexp)
        C = (a / (sigma ** 2)) * ((kappa - rho * sigma * iu - d) * T - 2.0 * np.log(one_minus_gexp / one_minus_g))
        return C, D

    def phi_X(self, u: np.ndarray, T: float, p: HestonParams) -> np.ndarray:
        u = np.asarray(u, dtype=np.complex128)
        C, D = self._CD_terms(u, T, p)
        mu = np.log(self.S0) + (self.r - self.q) * T
        return np.exp(1j * u * mu + C + D * p.v0)

    # --------------------------
    # P1/P2
    # --------------------------
    def _P2(self, K: np.ndarray, T: float, p: HestonParams) -> np.ndarray:
        K = np.asarray(K, float)
        lnK = np.log(np.maximum(K, self.tiny))
        u = self._u
        phi = self.phi_X(u, T, p)
        expo = np.exp(-1j * u[:, None] * lnK[None, :])
        denom = 1j * u[:, None]
        val = np.real(expo * phi[:, None] / denom)
        integral = (self._wu[:, None] * val).sum(axis=0)
        return 0.5 + integral / np.pi

    def _P1(self, K: np.ndarray, T: float, p: HestonParams) -> np.ndarray:
        K = np.asarray(K, float)
        lnK = np.log(np.maximum(K, self.tiny))
        u = self._u
        phi_shift = self.phi_X(u - 1j, T, p)
        phi_mi = self.phi_X(np.array([-1j], dtype=np.complex128), T, p)[0]
        expo = np.exp(-1j * u[:, None] * lnK[None, :])
        denom = (1j * u[:, None]) * phi_mi
        val = np.real(expo * phi_shift[:, None] / denom)
        integral = (self._wu[:, None] * val).sum(axis=0)
        return 0.5 + integral / np.pi

    def call_price_T(self, K: np.ndarray, T: float, p: HestonParams) -> np.ndarray:
        K = np.asarray(K, float)
        P1 = self._P1(K, T, p)
        P2 = self._P2(K, T, p)
        return np.exp(-self.q * T) * self.S0 * P1 - np.exp(-self.r * T) * K * P2

    # --------------------------
    # Quote-vector API
    # --------------------------
    def callhat(self, K: Sequence[float], T: Sequence[float], params: Optional[HestonParams] = None) -> np.ndarray:
        p = params if params is not None else self.params_
        if p is None:
            raise ValueError("No params provided and model has not been fitted yet.")
        K = np.asarray(K, float).reshape(-1)
        T = np.asarray(T, float).reshape(-1)
        if K.shape != T.shape:
            raise ValueError("K and T must have the same shape (quote vectors).")

        C = np.empty_like(K, dtype=float)
        for Tv in np.unique(T):
            m = (T == Tv)
            C[m] = self.call_price_T(K[m], float(Tv), p)
        return C

    def ivhat(self, K: Sequence[float], T: Sequence[float], params: Optional[HestonParams] = None) -> np.ndarray:
        C = self.callhat(K, T, params=params)
        K = np.asarray(K, float).reshape(-1)
        T = np.asarray(T, float).reshape(-1)
        IV = np.full_like(C, np.nan, dtype=float)
        for i in range(C.size):
            IV[i] = implied_vol_call(float(C[i]), self.S0, float(K[i]), self.r, self.q, float(T[i]))
        return IV

    # --------------------------
    # Calibration (dict bounds + prints + auto clip x0)
    # --------------------------
    def fit(
        self,
        K_obs: Sequence[float],
        T_obs: Sequence[float],
        C_obs: np.ndarray,
        x0: Optional[Tuple[float, float, float, float, float]] = None,
        bounds: BoundsLike = None,
        weights: Optional[np.ndarray] = None,
        verbose: int = 1,
        max_nfev: int = 400,
    ) -> HestonParams:
        K = np.asarray(K_obs, float).reshape(-1)
        T = np.asarray(T_obs, float).reshape(-1)
        C = np.asarray(C_obs, float).reshape(-1)
        if not (K.shape == T.shape == C.shape):
            raise ValueError("K_obs, T_obs, C_obs must be same length.")

        self.K_obs, self.T_obs, self.C_obs = K.copy(), T.copy(), C.copy()

        if x0 is None:
            x0 = (3.0, 0.04, 0.50, 0.04, -0.70)

        lb, ub, using_default_bounds = self._normalize_bounds(bounds)

        x0v = np.array(x0, float).ravel()
        if x0v.size != len(self._PARAM_NAMES):
            raise ValueError(f"x0 must have length {len(self._PARAM_NAMES)} in order {self._PARAM_NAMES}")

        x0c, changed = self._clip_x0_into_bounds(x0v, lb, ub)

        if verbose:
            print(f"[HestonModel.fit] using_default_bounds = {using_default_bounds}")
            print(f"[HestonModel.fit] bounds order = {self._PARAM_NAMES}")
            print("[HestonModel.fit] (bounds source) " + ("DEFAULT" if using_default_bounds else "USER-SUPPLIED"))
            if np.any(changed):
                changed_names = [self._PARAM_NAMES[i] for i in range(len(self._PARAM_NAMES)) if changed[i]]
                print(f"[HestonModel.fit] x0 was outside bounds; clipped params = {changed_names}")

        if weights is None:
            w = np.ones_like(C, float)
        else:
            w = np.asarray(weights, float).reshape(-1)
            if w.shape != C.shape:
                raise ValueError("weights must be same length as C_obs.")

        def pack(x):
            return HestonParams(
                kappa=float(x[0]),
                theta=float(x[1]),
                sigma=float(x[2]),
                v0=float(x[3]),
                rho=float(x[4]),
            )

        def resid(x):
            p = pack(x)
            if (p.kappa <= 0) or (p.theta <= 0) or (p.sigma <= 0) or (p.v0 <= 0) or (abs(p.rho) >= 0.999):
                return np.full_like(C, 1e6, float)
            Cfit = self.callhat(K, T, params=p)
            return np.sqrt(w) * (Cfit - C)

        res = least_squares(
            resid,
            x0=x0c,
            bounds=(lb, ub),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=int(max_nfev),
            verbose=verbose,
        )

        self.params_ = pack(res.x)
        self.C_fit = self.callhat(self.K_obs, self.T_obs, params=self.params_)
        return self.params_

    # --------------------------
    # RND via Breeden–Litzenberger (uniform K_grid)
    # --------------------------
    def rndhat(
        self,
        K_grid: Sequence[float],
        T_grid: Sequence[float],
        params: Optional[HestonParams] = None,
        enforce_nonneg: bool = True,
    ) -> np.ndarray:
        p = params if params is not None else self.params_
        if p is None:
            raise ValueError("Fit the model first (or pass params).")

        K = np.asarray(K_grid, float).reshape(-1)
        Tgrid = np.asarray(T_grid, float).reshape(-1)

        dK = np.diff(K)
        if np.max(np.abs(dK - dK.mean())) > 1e-8 * max(1.0, np.abs(dK.mean())):
            raise ValueError("rndhat expects an (approximately) uniform K_grid for finite differences.")

        Csurf = np.zeros((Tgrid.size, K.size), float)
        for i, Tv in enumerate(Tgrid):
            Csurf[i, :] = self.call_price_T(K, float(Tv), p)

        h = float(dK.mean())
        dC_dK = np.gradient(Csurf, h, axis=1)
        d2C_dK2 = np.gradient(dC_dK, h, axis=1)

        qsurf = np.exp(self.r * Tgrid[:, None]) * d2C_dK2
        if enforce_nonneg:
            qsurf = np.maximum(qsurf, 0.0)

        self.rnd_surface = qsurf
        self.rnd_strikes = K.copy()
        self.rnd_maturities = Tgrid.copy()
        return qsurf

    # --------------------------
    # Plots
    # --------------------------
    def plot_call_fit_panels(self, max_panels: int = 6, title: str = "Observed vs fitted call curves"):
        if self.C_fit is None:
            raise ValueError("Run fit() first.")
        K, T = self.K_obs, self.T_obs
        Cobs, Cfit = self.C_obs, self.C_fit

        Tu = np.unique(T)
        Tu_plot = Tu[np.linspace(0, Tu.size - 1, min(max_panels, Tu.size), dtype=int)]

        ncols = 3
        nrows = int(np.ceil(Tu_plot.size / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
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
        if self.rnd_surface is None:
            raise ValueError("Run rndhat() first.")
        q = self.rnd_surface
        T = self.rnd_maturities
        K = self.rnd_strikes

        idxs = np.linspace(0, T.size - 1, min(max_panels, T.size), dtype=int)

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


# ============================================================
# Synthetic calibration test (DEMO shows default + override)
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    S0, r, q = 100.0, 0.02, 0.00
    mdl = HestonModel(S0=S0, r=r, q=q, quad_N=120, quad_u_max=180.0)

    strikes_grid = np.linspace(70, 130, 25)
    maturities_grid = np.array([0.10, 0.20, 0.35, 0.50, 0.75, 1.00])
    KK, TT = np.meshgrid(strikes_grid, maturities_grid)
    K_obs = KK.ravel()
    T_obs = TT.ravel()

    true_p = HestonParams(kappa=3.0, theta=0.04, sigma=0.60, v0=0.04, rho=-0.70)
    C_true = mdl.callhat(K_obs, T_obs, params=true_p)

    rng = np.random.default_rng(1)
    C_obs = np.maximum(C_true + rng.normal(scale=0.02, size=C_true.shape), 0.0)

    # ---- Example 1: DEFAULT dict bounds (prints using_default_bounds=True)
    print("\n=== EXAMPLE 1: DEFAULT BOUNDS ===")
    fit_p = mdl.fit(
        K_obs, T_obs, C_obs,
        x0=(2.0, 0.05, 0.80, 0.05, -0.50),  # sigma=0.80 will be clipped to 0.75
        verbose=1,
        max_nfev=250
    )
    print("FITTED (default bounds):", fit_p)

    # ---- Example 2: OVERRIDE default bounds with DICT (prints using_default_bounds=False)
    print("\n=== EXAMPLE 2: OVERRIDE DEFAULT BOUNDS (DICT) ===")
    lb_d, ub_d = HestonModel.default_bounds_dict()

    # override: allow larger sigma and slightly wider rho range
    ub_d["sigma"] = 1.25
    lb_d["rho"] = -0.95
    ub_d["rho"] = 0.95

    mdl2 = HestonModel(S0=S0, r=r, q=q, quad_N=120, quad_u_max=180.0)
    fit_p2 = mdl2.fit(
        K_obs, T_obs, C_obs,
        x0=(2.0, 0.05, 0.95, 0.05, -0.50),   # now sigma=0.95 is allowed
        bounds=(lb_d, ub_d),
        verbose=1,
        max_nfev=250
    )
    print("FITTED (overridden bounds):", fit_p2)

    # ---- Plot panels + RND for the overridden-fit model
    mdl2.plot_call_fit_panels(max_panels=6)
    K_rnd = np.linspace(50, 160, 301)
    T_unique = np.unique(T_obs)
    mdl2.rndhat(K_rnd, T_unique, enforce_nonneg=True)
    mdl2.plot_rnd_panels(max_panels=6)
