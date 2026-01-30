# bates_core.py
# ============================================================
# Bates (Heston + Merton lognormal jumps) -- CALL PRICING + CALIBRATION
# - Gauss–Legendre quadrature cached on [0, U]
# - Vectorized call pricing across strikes for fixed T
# - Calibration groups quotes by maturity (fast)
#
# Dict-only interface (consistent with your HKDE core):
#   - x0:     dict with keys: v0, theta, kappa, sigma_v, rho, lamJ, muJ, sigJ
#   - bounds: (lb_dict, ub_dict) with same keys (or arrays of length 8)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares

# Parameter ordering (MUST stay consistent everywhere)
_PARAM_NAMES = ("v0", "theta", "kappa", "sigma_v", "rho", "lamJ", "muJ", "sigJ")


# ----------------------------
# Params container
# ----------------------------
@dataclass(frozen=True)
class BatesParams:
    v0: float
    theta: float
    kappa: float
    sigma_v: float
    rho: float
    lamJ: float  # jump intensity
    muJ: float   # mean log jump size
    sigJ: float  # std  log jump size

    @staticmethod
    def from_dict(d: Dict[str, float]) -> "BatesParams":
        missing = [k for k in _PARAM_NAMES if k not in d]
        if missing:
            raise KeyError(f"Missing keys in params dict: {missing}")
        return BatesParams(**{k: float(d[k]) for k in _PARAM_NAMES})

    def to_vec(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in _PARAM_NAMES], dtype=float)

    @staticmethod
    def from_vec(x: np.ndarray) -> "BatesParams":
        x = np.asarray(x, float).ravel()
        if x.size != len(_PARAM_NAMES):
            raise ValueError(f"Expected length {len(_PARAM_NAMES)}.")
        return BatesParams(**{k: float(x[i]) for i, k in enumerate(_PARAM_NAMES)})


# ----------------------------
# Defaults
# ----------------------------
def default_bates_bounds() -> Tuple[Dict[str, float], Dict[str, float]]:
    lb = dict(
        v0=0.005, theta=0.005, kappa=0.2, sigma_v=0.10, rho=-0.95,
        lamJ=0.0, muJ=-1.5, sigJ=0.05,
    )
    ub = dict(
        v0=1.5, theta=1.5, kappa=30.0, sigma_v=2.0, rho=0.95,
        lamJ=20.0, muJ=1.5, sigJ=1.5,
    )
    return lb, ub


def default_bates_x0() -> Dict[str, float]:
    return dict(
        v0=0.04,
        theta=0.35,
        kappa=5.0,
        sigma_v=0.6,
        rho=-0.4,
        lamJ=1.0,
        muJ=0.0,
        sigJ=0.25,
    )


# ----------------------------
# Bounds normalizer (dict or arrays)
# ----------------------------
def _normalize_bounds_dict(
    bounds: Optional[Tuple[Dict[str, float], Dict[str, float]]],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    using_default = False
    if bounds is None:
        using_default = True
        bounds = default_bates_bounds()

    lb_d, ub_d = bounds

    # Dict bounds
    if isinstance(lb_d, dict) and isinstance(ub_d, dict):
        missing = [k for k in _PARAM_NAMES if (k not in lb_d) or (k not in ub_d)]
        if missing:
            raise KeyError(f"Missing bounds for keys: {missing}")
        lb = np.array([float(lb_d[k]) for k in _PARAM_NAMES], dtype=float)
        ub = np.array([float(ub_d[k]) for k in _PARAM_NAMES], dtype=float)
        return lb, ub, using_default

    # Array bounds
    lb = np.asarray(lb_d, float).ravel()
    ub = np.asarray(ub_d, float).ravel()
    if lb.size != len(_PARAM_NAMES) or ub.size != len(_PARAM_NAMES):
        raise ValueError(f"Bounds arrays must have length {len(_PARAM_NAMES)}.")
    return lb, ub, using_default


# ============================================================
# Core model
# ============================================================
class BatesCore:
    """
    Bates = Heston stochastic volatility + Merton lognormal jumps.

    Call pricing via Heston-style CF inversion:
      C = S0 e^{-qT} P1 - K e^{-rT} P2

    Where P1, P2 computed from integrals of characteristic function of ln S_T.
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

        x, w = np.polynomial.legendre.leggauss(int(n))
        u = 0.5 * (x + 1.0) * float(U)
        wu = 0.5 * float(U) * w

        u = np.asarray(u, float)
        wu = np.asarray(wu, float)

        # Avoid division by 0 in integrands (1/(iu))
        u = np.where(np.abs(u) < 1e-12, 1e-12, u)

        self._quad_cache[key] = (u, wu)
        return u, wu

    # ---------- Bates CF for ln S_T ----------
    def cf(self, u: np.ndarray, T: float, p: BatesParams) -> np.ndarray:
        i = 1j
        u = np.asarray(u, dtype=complex)

        # Jump compensator: kappaJ = E[e^Y - 1]
        kappaJ = np.exp(p.muJ + 0.5 * p.sigJ**2) - 1.0
        # Drift adjustment: r -> r - lamJ*kappaJ (so S is martingale under Q)
        r_adj = self.r - p.lamJ * kappaJ

        # ---- Heston (little trap style, eps-stabilized) ----
        x0 = np.log(self.S0)
        a = p.kappa * p.theta
        b = p.kappa

        d = np.sqrt((p.rho * p.sigma_v * i * u - b) ** 2 + (p.sigma_v ** 2) * (i * u + u * u))
        g = (b - p.rho * p.sigma_v * i * u - d) / (b - p.rho * p.sigma_v * i * u + d)

        exp_minus_dT = np.exp(-d * T)

        eps = 1e-16
        denom = (1.0 - g * exp_minus_dT) + eps
        denom0 = (1.0 - g) + eps

        C = (
            i * u * (x0 + (r_adj - self.q) * T)
            + (a / (p.sigma_v ** 2)) * ((b - p.rho * p.sigma_v * i * u - d) * T - 2.0 * np.log(denom / denom0))
        )
        D = ((b - p.rho * p.sigma_v * i * u - d) / (p.sigma_v ** 2)) * ((1.0 - exp_minus_dT) / denom)

        phi_h = np.exp(C + D * p.v0)

        # ---- Merton jump CF on log jump Y ~ N(muJ, sigJ^2) ----
        phi_j = np.exp(p.lamJ * T * (np.exp(i * u * p.muJ - 0.5 * (p.sigJ ** 2) * (u ** 2)) - 1.0))

        return phi_h * phi_j

    # ============================================================
    # CALL PRICING (vectorized over strikes for fixed maturity)
    # ============================================================
    def call_prices(
        self,
        K: np.ndarray,
        T: float,
        p: BatesParams,
        *,
        Umax: float = 200.0,
        n_quad: int = 96,
    ) -> np.ndarray:
        K = np.asarray(K, float).ravel()
        if T <= 0:
            return np.maximum(self.S0 - K, 0.0)

        u, w = self.gauss_legendre_0U(n_quad, Umax)
        lnK = np.log(K)

        phi_u = self.cf(u, T, p)
        phi_u_shift = self.cf(u - 1j, T, p)

        # exp(-iu lnK) with outer(u, lnK) -> (nU, nK)
        E = np.exp(-1j * np.outer(u, lnK))

        # P2
        integrand_P2 = np.real(E * (phi_u[:, None] / (1j * u[:, None])))
        P2 = 0.5 + (1.0 / np.pi) * (w @ integrand_P2)

        # P1 uses phi(-i) normalization; phi(-i) = E[S_T] = S0 * exp((r-q)T)
        phi_mi = self.S0 * np.exp((self.r - self.q) * T)
        integrand_P1 = np.real(E * (phi_u_shift[:, None] / (1j * u[:, None] * phi_mi)))
        P1 = 0.5 + (1.0 / np.pi) * (w @ integrand_P1)

        # Numerical safety
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
    ) -> BatesParams:
        """
        Calibrate Bates params to observed calls.

        Inputs:
          - x0: dict with keys in _PARAM_NAMES (or None -> defaults)
          - bounds: (lb_dict, ub_dict) OR (lb_array, ub_array) (or None -> defaults)
        Returns:
          - BatesParams
        """
        if x0 is None:
            x0 = default_bates_x0()

        K_obs = np.asarray(K_obs, float).ravel()
        T_obs = np.asarray(T_obs, float).ravel()
        C_obs = np.asarray(C_obs, float).ravel()
        if not (K_obs.size == T_obs.size == C_obs.size):
            raise ValueError("K_obs, T_obs, C_obs must have same length.")

        # filter
        m = (
            np.isfinite(K_obs) & np.isfinite(T_obs) & np.isfinite(C_obs)
            & (K_obs > 0) & (T_obs > 0) & (C_obs >= 0)
        )
        K_obs, T_obs, C_obs = K_obs[m], T_obs[m], C_obs[m]

        p0 = BatesParams.from_dict(x0)
        x0v = p0.to_vec()

        lb, ub, using_default = _normalize_bounds_dict(bounds)
        if verbose:
            print(f"using_default_bounds={using_default}")

        # group indices by maturity (speeds up CF reuse)
        T_unique = np.unique(T_obs)
        idx_by_T = [np.where(T_obs == t)[0] for t in T_unique]

        def safe_region(pp: BatesParams) -> bool:
            if pp.v0 <= 0 or pp.theta <= 0 or pp.kappa <= 0:
                return False
            if pp.sigma_v <= 1e-10:
                return False
            if abs(pp.rho) >= 0.999:
                return False
            if pp.lamJ < 0:
                return False
            if pp.sigJ <= 1e-10:
                return False
            return True

        def residuals(x: np.ndarray) -> np.ndarray:
            pp = BatesParams.from_vec(x)
            if not safe_region(pp):
                return np.full(C_obs.shape, penalty, dtype=float)

            model = np.empty_like(C_obs)
            for t, idx in zip(T_unique, idx_by_T):
                model[idx] = self.call_prices(
                    K_obs[idx],
                    float(t),
                    pp,
                    Umax=Umax,
                    n_quad=n_quad,
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

        return BatesParams.from_vec(res.x)
