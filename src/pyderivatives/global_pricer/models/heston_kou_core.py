# pyderivatives/global_pricer/models/hkde_core.py
# ============================================================
# HKDE (Heston + Kou) -- CALL PRICING + CALIBRATION ONLY
# - Gauss–Legendre quadrature cached
# - Vectorized call pricing across strikes for fixed T
# - Calibration groups quotes by maturity
#
# Dict-only interface:
#   - x0:     dict with keys: v0, theta, kappa, sigma_v, rho, lam, p_up, eta1, eta2
#   - bounds: (lb_dict, ub_dict) with same keys
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
from scipy.optimize import least_squares


_PARAM_NAMES = ("v0", "theta", "kappa", "sigma_v", "rho", "lam", "p_up", "eta1", "eta2")

def default_hkde_x0() -> Dict[str, float]:
    # Your preferred starting point
    return dict(
        v0=0.03, theta=0.35, kappa=15.0, sigma_v=0.5, rho=-0.30,
        lam=0.8, p_up=0.50, eta1=15.0, eta2=15.0,
    )


def default_hkde_bounds() -> Tuple[Dict[str, float], Dict[str, float]]:
    # Your preferred bounds
    lb = dict(
        v0=0.005, theta=0.2, kappa=15.0, sigma_v=0.10, rho=-0.8,
        lam=0.02, p_up=0.01, eta1=10.0, eta2=10.0
    )
    ub = dict(
        v0=1.3, theta=1.0, kappa=3000.0, sigma_v=1.3, rho=0.8,
        lam=25.0, p_up=0.95, eta1=30.0, eta2=30.0
    )
    return lb, ub

@dataclass(frozen=True)
class HKDEParams:
    v0: float
    theta: float
    kappa: float
    sigma_v: float
    rho: float
    lam: float
    p_up: float
    eta1: float
    eta2: float

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "HKDEParams":
        missing = [k for k in _PARAM_NAMES if k not in d]
        if missing:
            raise KeyError(f"Missing keys in params dict: {missing}")
        return HKDEParams(**{k: float(d[k]) for k in _PARAM_NAMES})

    def to_vec(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in _PARAM_NAMES], dtype=float)

    @staticmethod
    def from_vec(x: np.ndarray) -> "HKDEParams":
        x = np.asarray(x, float).ravel()
        if x.size != len(_PARAM_NAMES):
            raise ValueError(f"Expected length {len(_PARAM_NAMES)}.")
        return HKDEParams(**{k: float(x[i]) for i, k in enumerate(_PARAM_NAMES)})





def _normalize_bounds_dict(
    bounds: Optional[Tuple[Dict[str, float], Dict[str, float]]]
) -> Tuple[np.ndarray, np.ndarray, bool]:
    using_default = False
    if bounds is None:
        using_default = True
        bounds = default_hkde_bounds()

    lb_d, ub_d = bounds
    if not (isinstance(lb_d, dict) and isinstance(ub_d, dict)):
        raise TypeError("bounds must be (lb_dict, ub_dict).")

    missing = [k for k in _PARAM_NAMES if (k not in lb_d) or (k not in ub_d)]
    if missing:
        raise KeyError(f"Missing bounds for keys: {missing}")

    lb = np.array([float(lb_d[k]) for k in _PARAM_NAMES], dtype=float)
    ub = np.array([float(ub_d[k]) for k in _PARAM_NAMES], dtype=float)
    return lb, ub, using_default


class HKDEModel:
    """
    Heston stochastic volatility + Kou double-exponential jumps.
    Call pricing via CF inversion (Heston little-trap + Kou jump CF).
    """

    def __init__(self, S0: float, r: float, q: float = 0.0):
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self._quad_cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]] = {}

    # ---------- Quadrature cache on [0, U] ----------
    def gauss_legendre_0U(self, n: int, U: float) -> Tuple[np.ndarray, np.ndarray]:
        key = (int(n), float(U))
        if key in self._quad_cache:
            return self._quad_cache[key]
        x, w = np.polynomial.legendre.leggauss(n)
        u = 0.5 * (x + 1.0) * U
        wu = 0.5 * U * w
        u = np.asarray(u, float)
        u = np.where(np.abs(u) < 1e-12, 1e-12, u)
        self._quad_cache[key] = (u, wu)
        return u, wu

    # ---------- Kou jump CF pieces ----------
    @staticmethod
    def _phi_J(u: np.ndarray, p_up: float, eta1: float, eta2: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        iu = 1j * u
        return (p_up * eta1 / (eta1 - iu)) + ((1.0 - p_up) * eta2 / (eta2 + iu))

    @staticmethod
    def _kappa_J(p_up: float, eta1: float, eta2: float) -> float:
        # E[e^J] - 1 (requires eta1 > 1)
        Ej = (p_up * eta1 / (eta1 - 1.0)) + ((1.0 - p_up) * eta2 / (eta2 + 1.0))
        return float(Ej - 1.0)

    # ---------- Heston CF (little trap, complex-safe) ----------
    @staticmethod
    def _cf_heston(
        u: np.ndarray, T: float, S0: float, r: float, q: float,
        v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
        drift_adj: float
    ) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        iu = 1j * u

        x0 = np.log(S0)
        a = kappa * theta
        b = kappa

        d = np.sqrt((rho * sigma_v * iu - b) ** 2 + sigma_v ** 2 * (iu + u * u))
        gp = (b - rho * sigma_v * iu + d) / (b - rho * sigma_v * iu - d)
        g = 1.0 / gp
        exp_minus_dT = np.exp(-d * T)

        eps = 1e-16
        denom = (1.0 - g * exp_minus_dT) + eps
        denom0 = (1.0 - g) + eps

        C = (
            iu * (x0 + (r - q - drift_adj) * T)
            + (a / (sigma_v ** 2)) * (
                (b - rho * sigma_v * iu - d) * T
                - 2.0 * np.log(denom / denom0)
            )
        )
        D = ((b - rho * sigma_v * iu - d) / (sigma_v ** 2)) * ((1.0 - exp_minus_dT) / denom)

        return np.exp(C + D * v0)

    # ---------- Full HKDE CF for ln S_T ----------
    def cf(self, u: np.ndarray, T: float, p: HKDEParams) -> np.ndarray:
        u = np.asarray(u, dtype=complex)

        if p.eta1 <= 1.0:
            raise ValueError("HKDE requires eta1 > 1 (so E[e^J] exists).")

        kJ = self._kappa_J(p.p_up, p.eta1, p.eta2)
        drift_adj = p.lam * kJ

        phi_h = self._cf_heston(
            u=u, T=T, S0=self.S0, r=self.r, q=self.q,
            v0=p.v0, kappa=p.kappa, theta=p.theta,
            sigma_v=p.sigma_v, rho=p.rho,
            drift_adj=drift_adj,
        )
        phiJ = self._phi_J(u, p.p_up, p.eta1, p.eta2)
        phi_kou = np.exp(p.lam * T * (phiJ - 1.0))
        return phi_h * phi_kou

    # ============================================================
    # CALL PRICING (vectorized over strikes for fixed maturity)
    # ============================================================
    def call_prices(
        self,
        K: np.ndarray,
        T: float,
        p: HKDEParams,
        *,
        Umax: float = 200.0,
        n_quad: int = 96,
    ) -> np.ndarray:
        K = np.asarray(K, float).ravel()
        if T <= 0:
            return np.maximum(self.S0 - K, 0.0)

        u, w = self.gauss_legendre_0U(n_quad, Umax)
        lnK = np.log(K)

        # phi(-i) = E[S_T] = S0 * exp((r-q)T) under Q
        phi_mi = self.S0 * np.exp((self.r - self.q) * T)

        phi_u = self.cf(u, T, p)
        phi_u_shift = self.cf(u - 1j, T, p)

        E = np.exp(-1j * np.outer(u, lnK))

        integrand_P2 = np.real(E * (phi_u[:, None] / (1j * u[:, None])))
        P2 = 0.5 + (1.0 / np.pi) * (w @ integrand_P2)

        integrand_P1 = np.real(E * (phi_u_shift[:, None] / (1j * u[:, None] * phi_mi)))
        P1 = 0.5 + (1.0 / np.pi) * (w @ integrand_P1)

        C = self.S0 * np.exp(-self.q * T) * P1 - K * np.exp(-self.r * T) * P2
        return np.maximum(C, 0.0)

    # ============================================================
    # CALIBRATION (no vega weights, dict x0 + dict bounds)
    # ============================================================
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
    ) -> HKDEParams:
        if x0 is None:
            x0 = default_hkde_x0()
        K_obs = np.asarray(K_obs, float).ravel()
        T_obs = np.asarray(T_obs, float).ravel()
        C_obs = np.asarray(C_obs, float).ravel()
        if not (K_obs.size == T_obs.size == C_obs.size):
            raise ValueError("K_obs, T_obs, C_obs must have same length.")

        m = np.isfinite(K_obs) & np.isfinite(T_obs) & np.isfinite(C_obs) & (K_obs > 0) & (T_obs > 0) & (C_obs >= 0)
        K_obs, T_obs, C_obs = K_obs[m], T_obs[m], C_obs[m]

        p0 = HKDEParams.from_dict(x0)
        x0v = p0.to_vec()

        lb, ub, using_default = _normalize_bounds_dict(bounds)
        if verbose:
            print(f"using_default_bounds={using_default}")

        # group indices by maturity (speeds up CF reuse)
        T_unique = np.unique(T_obs)
        idx_by_T = [np.where(T_obs == t)[0] for t in T_unique]

        def safe_region(pp: HKDEParams) -> bool:
            # quick stability checks beyond bounds
            if pp.v0 <= 0 or pp.theta <= 0 or pp.kappa <= 0:
                return False
            if pp.sigma_v <= 1e-8:
                return False
            if abs(pp.rho) >= 0.999:
                return False
            if pp.eta1 <= 1.0:
                return False
            if not (0.0 < pp.p_up < 1.0):
                return False
            return True

        def residuals(x: np.ndarray) -> np.ndarray:
            pp = HKDEParams.from_vec(x)
            if not safe_region(pp):
                return np.full(C_obs.shape, penalty, dtype=float)

            model = np.empty_like(C_obs)
            for t, idx in zip(T_unique, idx_by_T):
                model[idx] = self.call_prices(K_obs[idx], float(t), pp, Umax=Umax, n_quad=n_quad)

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

        return HKDEParams.from_vec(res.x)
    def call_surface(
        self,
        K_grid: np.ndarray,
        T_grid: np.ndarray,
        p: HKDEParams,
        *,
        Umax: float = 200.0,
        n_quad: int = 96,
    ) -> np.ndarray:
        K_grid = np.asarray(K_grid, float).ravel()
        T_grid = np.asarray(T_grid, float).ravel()
    
        out = np.empty((T_grid.size, K_grid.size), dtype=float)
        for i, T in enumerate(T_grid):
            out[i, :] = self.call_prices(K_grid, float(T), p, Umax=Umax, n_quad=n_quad)
    
        return out

