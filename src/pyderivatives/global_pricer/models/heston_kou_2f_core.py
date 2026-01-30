# heston_kou_2f_core.py
# ============================================================
# Two-Factor Heston–Kou (2F SV + double-exponential Kou jumps)
# - "0th order": exact CF inversion (no 1st/2nd order decomposition terms)
# - Dict x0 + dict bounds defaults
# - Cached Gauss–Legendre quadrature on [0, Umax]
#Source: Approximate option pricing under a two‑factor Heston–Kou stochastic volatility model (2025 )
# Param dict keys:
#   v01, theta1, kappa1, sigma1, rho1,
#   v02, theta2, kappa2, sigma2, rho2,
#   lam, p_up, eta1, eta2
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares

_PARAM_NAMES = (
    "v01", "theta1", "kappa1", "sigma1", "rho1",
    "v02", "theta2", "kappa2", "sigma2", "rho2",
    "lam", "p_up", "eta1", "eta2",
)


@dataclass(frozen=True)
class HK2FParams:
    v01: float
    theta1: float
    kappa1: float
    sigma1: float
    rho1: float

    v02: float
    theta2: float
    kappa2: float
    sigma2: float
    rho2: float

    lam: float
    p_up: float
    eta1: float
    eta2: float

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "HK2FParams":
        missing = [k for k in _PARAM_NAMES if k not in d]
        if missing:
            raise KeyError(f"Missing keys in params dict: {missing}")
        return HK2FParams(**{k: float(d[k]) for k in _PARAM_NAMES})

    def to_vec(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in _PARAM_NAMES], dtype=float)

    @staticmethod
    def from_vec(x: np.ndarray) -> "HK2FParams":
        x = np.asarray(x, float).ravel()
        if x.size != len(_PARAM_NAMES):
            raise ValueError(f"Expected length {len(_PARAM_NAMES)}.")
        return HK2FParams(**{k: float(x[i]) for i, k in enumerate(_PARAM_NAMES)})


# ----------------------------
# Defaults (you can tune later)
# ----------------------------
def default_heston_kou_2f_x0() -> Dict[str, float]:
    x0= dict(
            # fast factor (short-term skew)
            v01=0.04,
            theta1=0.35,
            kappa1=17.0,
            sigma1=0.2,
            rho1=-0.6,
    
            # slow factor (term structure)
            v02=0.06,
            theta2=0.35,
            kappa2=15.5,
            sigma2=0.2,
            rho2=-0.20,
    
            # jumps (same as 1-factor)
            lam=0.6,
            p_up=0.50,
            eta1=20.0,
            eta2=20.0,
        )
    return x0


def default_heston_kou_2f_bounds() -> Tuple[Dict[str, float], Dict[str, float]]:
    lb = dict(
        v01=0.005, theta1=0.1, kappa1=0.8,  sigma1=0.1, rho1=-0.75,
        v02=0.005, theta2=0.1, kappa2=0.8,  sigma2=0.1, rho2=-0.75,
        lam=0.02, p_up=0.05, eta1=10.0, eta2=10.0,
    )

    ub = dict(
        v01=1.0, theta1=1.0, kappa1=5000.0, sigma1=1.0, rho1=0.75,
        v02=1.0, theta2=1.0, kappa2=5000.0,  sigma2=1.0, rho2=0.75,
        lam=30.5, p_up=0.95, eta1=30.0, eta2=30.0,
    )
    return lb, ub


def _normalize_bounds(
    bounds: Optional[Tuple[Dict[str, float], Dict[str, float]]],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    using_default = False
    if bounds is None:
        using_default = True
        bounds = default_heston_kou_2f_bounds()

    lb_d, ub_d = bounds

    # dict bounds
    if isinstance(lb_d, dict) and isinstance(ub_d, dict):
        missing = [k for k in _PARAM_NAMES if (k not in lb_d) or (k not in ub_d)]
        if missing:
            raise KeyError(f"Missing bounds for keys: {missing}")
        lb = np.array([float(lb_d[k]) for k in _PARAM_NAMES], dtype=float)
        ub = np.array([float(ub_d[k]) for k in _PARAM_NAMES], dtype=float)
        return lb, ub, using_default

    # array bounds
    lb = np.asarray(lb_d, float).ravel()
    ub = np.asarray(ub_d, float).ravel()
    if lb.size != len(_PARAM_NAMES) or ub.size != len(_PARAM_NAMES):
        raise ValueError(f"Bounds arrays must have length {len(_PARAM_NAMES)}.")
    return lb, ub, using_default


# ============================================================
# Core
# ============================================================
class HestonKou2FCore:
    def __init__(self, S0: float, r: float, q: float = 0.0):
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self._quad_cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]] = {}

    # ---------- Quadrature ----------
    def gauss_legendre_0U(self, n: int, U: float) -> Tuple[np.ndarray, np.ndarray]:
        key = (int(n), float(U))
        if key in self._quad_cache:
            return self._quad_cache[key]
        x, w = np.polynomial.legendre.leggauss(int(n))
        u = 0.5 * (x + 1.0) * float(U)
        wu = 0.5 * float(U) * w
        u = np.asarray(u, float)
        u = np.where(np.abs(u) < 1e-12, 1e-12, u)
        wu = np.asarray(wu, float)
        self._quad_cache[key] = (u, wu)
        return u, wu

    # ---------- Kou pieces ----------
    @staticmethod
    def _phi_J(u: np.ndarray, p_up: float, eta1: float, eta2: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        iu = 1j * u
        return (p_up * eta1 / (eta1 - iu)) + ((1.0 - p_up) * eta2 / (eta2 + iu))

    @staticmethod
    def _kappa_J(p_up: float, eta1: float, eta2: float) -> float:
        Ej = (p_up * eta1 / (eta1 - 1.0)) + ((1.0 - p_up) * eta2 / (eta2 + 1.0))
        return float(Ej - 1.0)

    # ---------- One-factor Heston little-trap contribution ----------
    @staticmethod
    def _heston_CD(
        u: np.ndarray,
        T: float,
        *,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=complex)
        iu = 1j * u

        a = kappa * theta
        b = kappa

        d = np.sqrt((rho * sigma * iu - b) ** 2 + sigma ** 2 * (iu + u * u))
        gp = (b - rho * sigma * iu + d) / (b - rho * sigma * iu - d)
        g = 1.0 / gp
        exp_minus_dT = np.exp(-d * T)

        eps = 1e-16
        denom = (1.0 - g * exp_minus_dT) + eps
        denom0 = (1.0 - g) + eps

        C = (a / (sigma ** 2)) * ((b - rho * sigma * iu - d) * T - 2.0 * np.log(denom / denom0))
        D = ((b - rho * sigma * iu - d) / (sigma ** 2)) * ((1.0 - exp_minus_dT) / denom)
        return C, D

    # ---------- Full CF ----------
    def cf(self, u: np.ndarray, T: float, p: HK2FParams) -> np.ndarray:
        u = np.asarray(u, dtype=complex)

        if p.eta1 <= 1.0:
            raise ValueError("Requires eta1 > 1 for finite E[e^J].")
        if not (0.0 < p.p_up < 1.0):
            raise ValueError("Requires 0 < p_up < 1.")
        if p.lam < 0:
            raise ValueError("Requires lam >= 0.")
        if p.sigma1 <= 0 or p.sigma2 <= 0:
            raise ValueError("Requires sigma1,sigma2 > 0.")
        if p.kappa1 <= 0 or p.kappa2 <= 0:
            raise ValueError("Requires kappa1,kappa2 > 0.")

        x0 = np.log(self.S0)

        # jump compensator
        kJ = self._kappa_J(p.p_up, p.eta1, p.eta2)
        drift = (self.r - self.q - p.lam * kJ)

        out = np.exp(1j * u * (x0 + drift * T))

        # factor 1 contribution
        C1, D1 = self._heston_CD(u, T, kappa=p.kappa1, theta=p.theta1, sigma=p.sigma1, rho=p.rho1)
        out *= np.exp(C1 + D1 * p.v01)

        # factor 2 contribution
        C2, D2 = self._heston_CD(u, T, kappa=p.kappa2, theta=p.theta2, sigma=p.sigma2, rho=p.rho2)
        out *= np.exp(C2 + D2 * p.v02)

        # Kou jump CF
        phiJ = self._phi_J(u, p.p_up, p.eta1, p.eta2)
        out *= np.exp(p.lam * T * (phiJ - 1.0))

        return out

    # ---------- Calls ----------
    def call_prices(
        self,
        K: np.ndarray,
        T: float,
        p: HK2FParams,
        *,
        Umax: float = 200.0,
        n_quad: int = 96,
        approx_order: int = 0,
    ) -> np.ndarray:
        if approx_order != 0:
            raise NotImplementedError("Only approx_order=0 is implemented (exact CF inversion).")

        K = np.asarray(K, float).ravel()
        if T <= 0:
            return np.maximum(self.S0 - K, 0.0)

        u, w = self.gauss_legendre_0U(n_quad, Umax)
        lnK = np.log(K)

        phi_mi = self.S0 * np.exp((self.r - self.q) * T)  # phi(-i)

        phi_u = self.cf(u, T, p)
        phi_u_shift = self.cf(u - 1j, T, p)

        E = np.exp(-1j * np.outer(u, lnK))

        P2 = 0.5 + (1.0 / np.pi) * (w @ np.real(E * (phi_u[:, None] / (1j * u[:, None]))))
        P1 = 0.5 + (1.0 / np.pi) * (w @ np.real(E * (phi_u_shift[:, None] / (1j * u[:, None] * phi_mi))))

        P1 = np.clip(P1, 0.0, 1.0)
        P2 = np.clip(P2, 0.0, 1.0)

        C = self.S0 * np.exp(-self.q * T) * P1 - K * np.exp(-self.r * T) * P2
        return np.maximum(C, 0.0)

    # ---------- Fit ----------
    def fit_to_calls(
        self,
        *,
        K_obs: np.ndarray,
        T_obs: np.ndarray,
        C_obs: np.ndarray,
        x0: Optional[Dict[str, float]] = None,
        bounds: Optional[Tuple[Dict[str, float], Dict[str, float]]] = None,
        Umax: float = 200.0,
        n_quad: int = 96,
        max_nfev: int = 200,
        verbose: int = 1,
        penalty: float = 1e6,
        approx_order: int = 0,
    ) -> HK2FParams:
        if approx_order != 0:
            raise NotImplementedError("Only approx_order=0 is implemented (exact CF inversion).")

        if x0 is None:
            x0 = default_heston_kou_2f_x0()

        K_obs = np.asarray(K_obs, float).ravel()
        T_obs = np.asarray(T_obs, float).ravel()
        C_obs = np.asarray(C_obs, float).ravel()
        if not (K_obs.size == T_obs.size == C_obs.size):
            raise ValueError("K_obs, T_obs, C_obs must have same length.")

        m = (
            np.isfinite(K_obs) & np.isfinite(T_obs) & np.isfinite(C_obs)
            & (K_obs > 0) & (T_obs > 0) & (C_obs >= 0)
        )
        K_obs, T_obs, C_obs = K_obs[m], T_obs[m], C_obs[m]

        p0 = HK2FParams.from_dict(x0)
        x0v = p0.to_vec()

        lb, ub, using_default = _normalize_bounds(bounds)
        if verbose:
            print(f"using_default_bounds={using_default}")

        # group indices by maturity
        T_unique = np.unique(T_obs)
        idx_by_T = [np.where(T_obs == t)[0] for t in T_unique]

        def safe_region(pp: HK2FParams) -> bool:
            if pp.v01 <= 0 or pp.theta1 <= 0 or pp.kappa1 <= 0:
                return False
            if pp.v02 <= 0 or pp.theta2 <= 0 or pp.kappa2 <= 0:
                return False
            if pp.sigma1 <= 1e-10 or pp.sigma2 <= 1e-10:
                return False
            if abs(pp.rho1) >= 0.999 or abs(pp.rho2) >= 0.999:
                return False
            if pp.lam < 0:
                return False
            if not (0.0 < pp.p_up < 1.0):
                return False
            if pp.eta1 <= 1.0 or pp.eta2 <= 0:
                return False
            return True

        def residuals(x: np.ndarray) -> np.ndarray:
            pp = HK2FParams.from_vec(x)
            if not safe_region(pp):
                return np.full(C_obs.shape, penalty, dtype=float)

            model = np.empty_like(C_obs)
            for t, idx in zip(T_unique, idx_by_T):
                model[idx] = self.call_prices(
                    K_obs[idx], float(t), pp, Umax=Umax, n_quad=n_quad, approx_order=approx_order
                )

            if not np.all(np.isfinite(model)):
                return np.full(C_obs.shape, penalty, dtype=float)

            return model - C_obs

        res = least_squares(
            residuals,
            x0v,
            bounds=(lb, ub),
            method="trf",
            verbose=2 if verbose else 0,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=int(max_nfev),
        )

        return HK2FParams.from_vec(res.x)

    def call_surface(
        self,
        K_grid: np.ndarray,
        T_grid: np.ndarray,
        p: HK2FParams,
        *,
        Umax: float = 200.0,
        n_quad: int = 96,
        approx_order: int = 0,
    ) -> np.ndarray:
        K_grid = np.asarray(K_grid, float).ravel()
        T_grid = np.asarray(T_grid, float).ravel()
        out = np.empty((T_grid.size, K_grid.size), dtype=float)
        for i, T in enumerate(T_grid):
            out[i, :] = self.call_prices(K_grid, float(T), p, Umax=Umax, n_quad=n_quad, approx_order=approx_order)
        return out
