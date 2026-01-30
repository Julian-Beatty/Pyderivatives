# kou_core.py
# ============================================================
# Kou (Double-Exponential Jump Diffusion) -- CALL PRICING + CALIBRATION
#
# Under Q:
#   dS_t / S_{t-} = (r - q - lam * kappaJ) dt + sigma dW_t + (e^J - 1) dN_t
#   J ~ double-exponential:
#       with prob p_up:   density eta1 * exp(-eta1 j) for j>=0
#       with prob 1-p_up: density eta2 * exp(+eta2 j) for j<0
#
# CF for ln S_T:
#   phi(u) = exp( iu * (x0 + (r-q-0.5*sigma^2 - lam*kappaJ)T) - 0.5*sigma^2*u^2*T
#                 + lam*T*(phi_J(u)-1) )
# where:
#   phi_J(u) = p_up*eta1/(eta1 - iu) + (1-p_up)*eta2/(eta2 + iu)
#   kappaJ = E[e^J - 1] = p_up*eta1/(eta1-1) + (1-p_up)*eta2/(eta2+1) - 1   (requires eta1>1)
#
# Dict interface:
#   x0: dict keys  sigma, lam, p_up, eta1, eta2
#   bounds: (lb_dict, ub_dict) same keys
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares

_PARAM_NAMES = ("sigma", "lam", "p_up", "eta1", "eta2")


@dataclass(frozen=True)
class KouParams:
    sigma: float
    lam: float
    p_up: float
    eta1: float
    eta2: float

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "KouParams":
        missing = [k for k in _PARAM_NAMES if k not in d]
        if missing:
            raise KeyError(f"Missing keys in params dict: {missing}")
        return KouParams(**{k: float(d[k]) for k in _PARAM_NAMES})

    def to_vec(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in _PARAM_NAMES], dtype=float)

    @staticmethod
    def from_vec(x: np.ndarray) -> "KouParams":
        x = np.asarray(x, float).ravel()
        if x.size != len(_PARAM_NAMES):
            raise ValueError(f"Expected length {len(_PARAM_NAMES)}.")
        return KouParams(**{k: float(x[i]) for i, k in enumerate(_PARAM_NAMES)})


# ----------------------------
# Defaults (tune these as you like)
# ----------------------------
def default_kou_x0() -> Dict[str, float]:
    return dict(
        sigma=0.70,    # diffusion vol
        lam=2.0,       # jump intensity
        p_up=0.50,     # up-jump probability
        eta1=10.0,     # rate for + jumps  (mean + jump ~ 1/eta1)
        eta2=10.0,     # rate for - jumps  (mean - jump ~ -1/eta2)
    )


def default_kou_bounds() -> Tuple[Dict[str, float], Dict[str, float]]:
    lb = dict(
        sigma=0.05,
        lam=0.0,
        p_up=0.01,
        eta1=2.0,      # must exceed 1 for kappaJ finite; keep >1
        eta2=2.0,
    )
    ub = dict(
        sigma=3.0,
        lam=30.0,
        p_up=0.99,
        eta1=50.0,
        eta2=50.0,
    )
    return lb, ub


def _normalize_bounds(bounds: Optional[Tuple[Dict[str, float], Dict[str, float]]]) -> Tuple[np.ndarray, np.ndarray, bool]:
    using_default = False
    if bounds is None:
        using_default = True
        bounds = default_kou_bounds()
    lb_d, ub_d = bounds

    if isinstance(lb_d, dict) and isinstance(ub_d, dict):
        missing = [k for k in _PARAM_NAMES if (k not in lb_d) or (k not in ub_d)]
        if missing:
            raise KeyError(f"Missing bounds for keys: {missing}")
        lb = np.array([float(lb_d[k]) for k in _PARAM_NAMES], dtype=float)
        ub = np.array([float(ub_d[k]) for k in _PARAM_NAMES], dtype=float)
        return lb, ub, using_default

    lb = np.asarray(lb_d, float).ravel()
    ub = np.asarray(ub_d, float).ravel()
    if lb.size != len(_PARAM_NAMES) or ub.size != len(_PARAM_NAMES):
        raise ValueError(f"Bounds arrays must have length {len(_PARAM_NAMES)}.")
    return lb, ub, using_default


class KouCore:
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
        x, w = np.polynomial.legendre.leggauss(int(n))
        u = 0.5 * (x + 1.0) * float(U)
        wu = 0.5 * float(U) * w
        u = np.asarray(u, float)
        u = np.where(np.abs(u) < 1e-12, 1e-12, u)  # avoid divide by 0
        wu = np.asarray(wu, float)
        self._quad_cache[key] = (u, wu)
        return u, wu

    # ---------- Kou jump CF components ----------
    @staticmethod
    def _phi_J(u: np.ndarray, p_up: float, eta1: float, eta2: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        iu = 1j * u
        return (p_up * eta1 / (eta1 - iu)) + ((1.0 - p_up) * eta2 / (eta2 + iu))

    @staticmethod
    def _kappa_J(p_up: float, eta1: float, eta2: float) -> float:
        # E[e^J] - 1; requires eta1 > 1
        Ej = (p_up * eta1 / (eta1 - 1.0)) + ((1.0 - p_up) * eta2 / (eta2 + 1.0))
        return float(Ej - 1.0)

    # ---------- CF for ln S_T ----------
    def cf(self, u: np.ndarray, T: float, p: KouParams) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        iu = 1j * u

        if p.eta1 <= 1.0:
            raise ValueError("Kou requires eta1 > 1 so E[e^J] exists (finite kappaJ).")
        if not (0.0 < p.p_up < 1.0):
            raise ValueError("Kou requires 0 < p_up < 1.")
        if p.sigma <= 0.0:
            raise ValueError("Kou requires sigma > 0.")
        if p.lam < 0.0:
            raise ValueError("Kou requires lam >= 0.")

        x0 = np.log(self.S0)
        kJ = self._kappa_J(p.p_up, p.eta1, p.eta2)
        mu = (self.r - self.q) - 0.5 * p.sigma**2 - p.lam * kJ

        phiJ = self._phi_J(u, p.p_up, p.eta1, p.eta2)
        jump_term = p.lam * T * (phiJ - 1.0)
        diff_term = iu * (x0 + mu * T) - 0.5 * (p.sigma**2) * (u**2) * T

        return np.exp(diff_term + jump_term)

    # ============================================================
    # CALL PRICING (vectorized over strikes for fixed T)
    # ============================================================
    def call_prices(
        self,
        K: np.ndarray,
        T: float,
        p: KouParams,
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

        P1 = np.clip(P1, 0.0, 1.0)
        P2 = np.clip(P2, 0.0, 1.0)

        C = self.S0 * np.exp(-self.q * T) * P1 - K * np.exp(-self.r * T) * P2
        return np.maximum(C, 0.0)

    # ============================================================
    # CALIBRATION (dict x0 + dict/array bounds)
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
    ) -> KouParams:
        if x0 is None:
            x0 = default_kou_x0()

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

        p0 = KouParams.from_dict(x0)
        x0v = p0.to_vec()

        lb, ub, using_default = _normalize_bounds(bounds)
        if verbose:
            print(f"using_default_bounds={using_default}")

        # group by maturity (speeds CF reuse)
        T_unique = np.unique(T_obs)
        idx_by_T = [np.where(T_obs == t)[0] for t in T_unique]

        def safe_region(pp: KouParams) -> bool:
            if pp.sigma <= 1e-10:
                return False
            if pp.lam < 0:
                return False
            if not (0.0 < pp.p_up < 1.0):
                return False
            if pp.eta1 <= 1.0:
                return False
            if pp.eta2 <= 0.0:
                return False
            return True

        def residuals(x: np.ndarray) -> np.ndarray:
            pp = KouParams.from_vec(x)
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

        return KouParams.from_vec(res.x)
