# bates_surface_fitter.py
# Bates (1996): Heston stochastic volatility + Merton lognormal jumps
#
# NEW (bounds quality-of-life, like your pipeline wants):
#   - fit(..., bounds=...) now accepts:
#       * array form: (lb_vec, ub_vec) length 8, ordered:
#           [kappa, theta, sigma, v0, rho, lam, muJ, sigJ]
#       * dict form:  (lb_dict, ub_dict) with keys:
#           "kappa","theta","sigma","v0","rho","lam","muJ","sigJ"
#   - DEFAULT bounds (if bounds=None):
#       conservative-ish Heston+jump box (you can edit below)
#
# After fitting:
#   fit.chat(strikes, maturities)    -> call surface C(T,K)
#   fit.ivhat(strikes, maturities)   -> BS implied vol surface (pointwise inversion)
#   fit.rndhat(strikes, maturities)  -> RND surface via Breeden–Litzenberger
#
# Plotting (all methods UNDER THE CLASS):
#   fit.plot_call_surface(...)
#   fit.plot_rnd_surface(...)
#   fit.plot_call_and_rnd_surfaces(...)
#   fit.plot_call_fit_panels(...)    -> observed calls vs fitted calls (panel by maturity)
#   fit.plot_rnd_panels(...)         -> FITTED RND panels only (with optional extrapolation)

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares, brentq

from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# 1) Parameter container
# ============================================================

@dataclass
class BatesParams:
    # Heston
    kappa: float
    theta: float
    sigma: float
    v0: float
    rho: float
    # Jumps (Merton lognormal jumps in log-price)
    lam: float
    muJ: float
    sigJ: float


# ============================================================
# 2) Bates characteristic function + pricer
# ============================================================

def _jump_kappa(muJ: float, sigJ: float) -> float:
    # k = E[e^Y - 1],  Y ~ N(muJ, sigJ^2)
    return float(np.exp(muJ + 0.5 * sigJ * sigJ) - 1.0)


def _bates_char_logS(u: np.ndarray, T: float, S0: float, r: float, q: float, p: BatesParams) -> np.ndarray:
    """
    Characteristic function of log(S_T) for Bates:
      φ(u) = E[exp(i u log S_T)]
    """
    u = np.asarray(u, dtype=np.complex128)
    iu = 1j * u

    # drift adjustment for jumps
    k = _jump_kappa(p.muJ, p.sigJ)
    r_adj = r - p.lam * k

    # Heston components
    kappa, theta, sigma, v0, rho = p.kappa, p.theta, p.sigma, p.v0, p.rho
    a = kappa * theta
    b = kappa

    d = np.sqrt((rho * sigma * iu - b) ** 2 + sigma * sigma * (iu + u * u))
    g = (b - rho * sigma * iu - d) / (b - rho * sigma * iu + d)

    # stabilized form
    exp_m_dT = np.exp(-d * T)
    G = g * exp_m_dT

    C = (r_adj - q) * iu * T + (a / (sigma * sigma)) * (
        (b - rho * sigma * iu - d) * T - 2.0 * np.log((1.0 - G) / (1.0 - g))
    )
    D = ((b - rho * sigma * iu - d) / (sigma * sigma)) * ((1.0 - exp_m_dT) / (1.0 - G))

    heston_cf = np.exp(iu * np.log(S0) + C + D * v0)

    # Merton jump CF factor
    jump_cf = np.exp(p.lam * T * (np.exp(iu * p.muJ - 0.5 * p.sigJ * p.sigJ * u * u) - 1.0))

    return heston_cf * jump_cf


def bates_call_price(
    S0: float,
    K: np.ndarray,
    r: float,
    q: float,
    T: float,
    p: BatesParams,
    *,
    u_max: float = 150.0,
    n_quad: int = 96,
) -> np.ndarray:
    """
    European call under Bates using Heston-style P1/P2 integrals:
      C = S0 e^{-qT} P1 - K e^{-rT} P2
    """
    K = np.asarray(K, float)
    if T <= 0:
        return np.maximum(S0 - K, 0.0)

    # Gauss-Legendre on [0, u_max]
    x, w = leggauss(n_quad)
    u = 0.5 * (x + 1.0) * u_max
    wu = 0.5 * u_max * w
    u_c = u.astype(np.complex128)

    logK = np.log(K)

    # φ(-i) normalization
    phi_mi = _bates_char_logS(np.array([-1j], dtype=np.complex128), T, S0, r, q, p)[0]
    phi_u = _bates_char_logS(u_c, T, S0, r, q, p)
    phi_u_mi = _bates_char_logS(u_c - 1j, T, S0, r, q, p)

    E = np.exp(-1j * np.outer(logK, u))

    denom = 1j * u
    denom = np.where(np.abs(denom) < 1e-14, 1j * 1e-14, denom)

    integrand_P2 = np.real(E * (phi_u / denom))
    integrand_P1 = np.real(E * (phi_u_mi / (denom * phi_mi)))

    P2 = 0.5 + (1.0 / np.pi) * (integrand_P2 @ wu)
    P1 = 0.5 + (1.0 / np.pi) * (integrand_P1 @ wu)

    return S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2


# ============================================================
# 3) Black–Scholes utilities (for IV inversion)
# ============================================================

def _bs_call_price(S0: float, K: np.ndarray, r: float, q: float, T: float, vol: float) -> np.ndarray:
    K = np.asarray(K, float)
    if T <= 0:
        return np.maximum(S0 - K, 0.0)
    if vol <= 0:
        fwd = S0 * np.exp((r - q) * T)
        disc = np.exp(-r * T)
        return disc * np.maximum(fwd - K, 0.0)

    from math import erf
    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    Phi = lambda x: 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    Nd1 = np.vectorize(Phi)(d1)
    Nd2 = np.vectorize(Phi)(d2)

    return S0 * np.exp(-q * T) * Nd1 - K * np.exp(-r * T) * Nd2


def implied_vol_call(
    price: float, S0: float, K: float, r: float, q: float, T: float,
    vol_lo: float = 1e-6, vol_hi: float = 5.0
) -> float:
    if T <= 0:
        return np.nan

    # static arbitrage bounds for calls
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    lower = max(0.0, S0 * disc_q - K * disc_r)
    upper = S0 * disc_q
    if not (lower - 1e-10 <= price <= upper + 1e-10):
        return np.nan

    def f(v):
        return _bs_call_price(S0, np.array([K]), r, q, T, v)[0] - price

    try:
        return brentq(f, vol_lo, vol_hi, maxiter=200)
    except Exception:
        return np.nan


# ============================================================
# 4) Class
# ============================================================

class BatesSurfaceFitter:
    """
    Bates (Heston + Merton jumps) surface fitter.

    After fit:
      - chat(strikes, maturities): call surface C(T,K)
      - ivhat(strikes, maturities): BS implied vols of model calls
      - rndhat(strikes, maturities): BL RND q_T(K) = exp(rT) d^2C/dK^2
    """

    _PARAM_NAMES = ("kappa", "theta", "sigma", "v0", "rho", "lam", "muJ", "sigJ")

    def __init__(self, *, u_max: float = 150.0, n_quad: int = 96):
        self.u_max = float(u_max)
        self.n_quad = int(n_quad)

        self.S0: Optional[float] = None
        self.r: Optional[float] = None
        self.q: Optional[float] = None

        self.params_: Optional[BatesParams] = None
        self.fit_result_: Optional[Dict[str, Any]] = None

        # store training data for panel plots
        self.K_obs_: Optional[np.ndarray] = None
        self.T_obs_: Optional[np.ndarray] = None
        self.C_obs_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None

        # --- quadrature objects for "default evaluation" (chat/rndhat/etc.) ---
        self._build_quadrature(self.u_max, self.n_quad)

        # --- caches used during fast fit ---
        self._fit_cache: Optional[Dict[float, Dict[str, Any]]] = None
        self._fit_quad: Optional[Dict[str, Any]] = None

    # ----------------------------
    # bounds normalization (array or dict) + default
    # ----------------------------
    @classmethod
    def _normalize_bounds(
        cls,
        bounds: Optional[
            Tuple[
                Union[np.ndarray, Dict[str, float]],
                Union[np.ndarray, Dict[str, float]],
            ]
        ],
    ) -> Tuple[np.ndarray, np.ndarray]:
        # DEFAULT (edit if you want)
        if bounds is None:
            lb_d = dict(
                kappa=1e-3,
                theta=1e-6,
                sigma=1e-3,
                v0=1e-6,
                rho=-0.95,
                lam=0.0,
                muJ=-0.5,
                sigJ=1e-3,
            )
            ub_d = dict(
                kappa=20.0,
                theta=1.0,
                sigma=3.0,
                v0=1.0,
                rho=0.95,
                lam=5.0,
                muJ=0.5,
                sigJ=1.5,
            )
            bounds = (lb_d, ub_d)

        lb, ub = bounds

        # dict form
        if isinstance(lb, dict) or isinstance(ub, dict):
            if not (isinstance(lb, dict) and isinstance(ub, dict)):
                raise TypeError("If using dict bounds, both lb and ub must be dicts.")
            missing = [n for n in cls._PARAM_NAMES if (n not in lb) or (n not in ub)]
            if missing:
                raise KeyError(f"Missing bounds for: {missing}")
            lbv = np.array([float(lb[n]) for n in cls._PARAM_NAMES], float)
            ubv = np.array([float(ub[n]) for n in cls._PARAM_NAMES], float)
            return lbv, ubv

        # array form
        lb = np.asarray(lb, float).ravel()
        ub = np.asarray(ub, float).ravel()
        if lb.size != len(cls._PARAM_NAMES) or ub.size != len(cls._PARAM_NAMES):
            raise ValueError(f"Array bounds must have length {len(cls._PARAM_NAMES)}.")
        return lb, ub

    # ----------------------------
    # quadrature builder
    # ----------------------------
    @staticmethod
    def _make_quadrature(u_max: float, n_quad: int) -> Dict[str, Any]:
        x, w = leggauss(int(n_quad))
        u = 0.5 * (x + 1.0) * float(u_max)
        wu = 0.5 * float(u_max) * w
        u_c = u.astype(np.complex128)
        denom = 1j * u
        denom = np.where(np.abs(denom) < 1e-14, 1j * 1e-14, denom)
        return {"u_max": float(u_max), "n_quad": int(n_quad), "u": u, "wu": wu, "u_c": u_c, "denom": denom}

    def _build_quadrature(self, u_max: float, n_quad: int) -> None:
        self._quad = self._make_quadrature(u_max, n_quad)

    # ----------------------------
    # internal call pricing
    # ----------------------------
    def _call(self, K: np.ndarray, T: float) -> np.ndarray:
        assert self.params_ is not None
        assert self.S0 is not None and self.r is not None and self.q is not None
        return bates_call_price(
            self.S0, K, self.r, self.q, float(T), self.params_,
            u_max=self.u_max, n_quad=self.n_quad
        )

    def _calls_cached_E(
        self,
        K: np.ndarray,
        T: float,
        p: BatesParams,
        E: np.ndarray,
        quad: Dict[str, Any],
    ) -> np.ndarray:
        assert self.S0 is not None and self.r is not None and self.q is not None

        u_c = quad["u_c"]
        wu = quad["wu"]
        denom = quad["denom"]

        phi_mi = _bates_char_logS(np.array([-1j], dtype=np.complex128), T, self.S0, self.r, self.q, p)[0]
        phi_u = _bates_char_logS(u_c, T, self.S0, self.r, self.q, p)
        phi_u_mi = _bates_char_logS(u_c - 1j, T, self.S0, self.r, self.q, p)

        integrand_P2 = np.real(E * (phi_u / denom))
        integrand_P1 = np.real(E * (phi_u_mi / (denom * phi_mi)))

        P2 = 0.5 + (1.0 / np.pi) * (integrand_P2 @ wu)
        P1 = 0.5 + (1.0 / np.pi) * (integrand_P1 @ wu)

        return self.S0 * np.exp(-self.q * T) * P1 - K * np.exp(-self.r * T) * P2

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def _extended_range_from_observed(
        K_obs: np.ndarray,
        *,
        strike_extension: float = 0.0,
        K_min: Optional[float] = None,
        K_max: Optional[float] = None,
    ) -> Tuple[float, float]:
        K_obs = np.asarray(K_obs, float)
        kmin_obs, kmax_obs = float(np.min(K_obs)), float(np.max(K_obs))

        if K_min is None:
            kmin = kmin_obs * (1.0 - float(strike_extension))
        else:
            kmin = float(K_min)

        if K_max is None:
            kmax = kmax_obs * (1.0 + float(strike_extension))
        else:
            kmax = float(K_max)

        kmin = max(kmin, 1e-10)
        if kmax <= kmin:
            kmax = kmin * 1.01

        return kmin, kmax

    # ----------------------------
    # fit
    # ----------------------------
    def fit(
        self,
        *,
        S0: float,
        K_obs: np.ndarray,
        T_obs: np.ndarray,
        C_obs: np.ndarray,
        r: float,
        q: float,
        x0: Optional[Union[np.ndarray, Dict[str, float]]] = None,
        bounds: Optional[
            Tuple[
                Union[np.ndarray, Dict[str, float]],
                Union[np.ndarray, Dict[str, float]],
            ]
        ] = None,
        weights: Optional[np.ndarray] = None,
        verbose: int = 1,
        max_nfev: int = 200,
        # --- speed knobs ---
        use_fast: bool = True,
        fit_u_max: Optional[float] = None,
        fit_n_quad: Optional[int] = None,
    ) -> "BatesSurfaceFitter":
        
        
        using_default_bounds = (bounds is None)   # <-- ADD
        lb, ub = self._normalize_bounds(bounds)
        bounds_vec = (lb, ub)
        
        
        K_obs = np.asarray(K_obs, float).ravel()
        T_obs = np.asarray(T_obs, float).ravel()
        C_obs = np.asarray(C_obs, float).ravel()
        assert K_obs.shape == T_obs.shape == C_obs.shape

        n = K_obs.size
        if weights is None:
            weights = np.ones(n, float)
        else:
            weights = np.asarray(weights, float).ravel()
            assert weights.shape == (n,)

        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)

        # x0 can be dict or vector
        if x0 is None:
            x0_vec = np.array([1.5, 0.04, 0.5, 0.04, -0.6, 0.1, -0.05, 0.2], float)
        elif isinstance(x0, dict):
            missing = [nm for nm in self._PARAM_NAMES if nm not in x0]
            if missing:
                raise KeyError(f"x0 dict missing keys: {missing}")
            x0_vec = np.array([float(x0[nm]) for nm in self._PARAM_NAMES], float)
        else:
            x0_vec = np.asarray(x0, float).ravel()
            if x0_vec.size != len(self._PARAM_NAMES):
                raise ValueError(f"x0 vector must have length {len(self._PARAM_NAMES)}.")

        # bounds can be dict or vector; default if None
        lb, ub = self._normalize_bounds(bounds)
        bounds_vec = (lb, ub)

        def unpack(x: np.ndarray) -> BatesParams:
            return BatesParams(
                kappa=float(x[0]), theta=float(x[1]), sigma=float(x[2]),
                v0=float(x[3]), rho=float(x[4]),
                lam=float(x[5]), muJ=float(x[6]), sigJ=float(x[7]),
            )

        # choose quadrature for fit (often cheaper), but keep self._quad for final evaluation
        if fit_u_max is None:
            fit_u_max = self.u_max
        if fit_n_quad is None:
            fit_n_quad = self.n_quad

        fit_quad = self._make_quadrature(float(fit_u_max), int(fit_n_quad))
        self._fit_quad = fit_quad

        # precompute per-maturity caches for fast fit (E matrices)
        self._fit_cache = None
        if use_fast:
            cache: Dict[float, Dict[str, Any]] = {}
            uniqT = np.unique(T_obs)
            u_fit = fit_quad["u"]
            for Ti in uniqT:
                idx = (T_obs == Ti)
                Ki = K_obs[idx]
                logK = np.log(Ki)
                E = np.exp(-1j * np.outer(logK, u_fit))  # (nKi, nU)
                cache[float(Ti)] = {"idx": idx, "K": Ki, "E": E}
            self._fit_cache = cache

        def resid(x: np.ndarray) -> np.ndarray:
            p = unpack(x)
            C_fit = np.empty_like(C_obs)

            if use_fast and self._fit_cache is not None:
                for Ti, d in self._fit_cache.items():
                    idx = d["idx"]
                    Ki = d["K"]
                    E = d["E"]
                    C_fit[idx] = self._calls_cached_E(Ki, float(Ti), p, E, fit_quad)
            else:
                u_fit = fit_quad["u"]
                for Ti in np.unique(T_obs):
                    idx = (T_obs == Ti)
                    Ki = K_obs[idx]
                    logK = np.log(Ki)
                    E = np.exp(-1j * np.outer(logK, u_fit))
                    C_fit[idx] = self._calls_cached_E(Ki, float(Ti), p, E, fit_quad)

            return (C_fit - C_obs) * weights

        res = least_squares(
            resid,
            x0=x0_vec,
            bounds=bounds_vec,
            method="trf",
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=int(max_nfev),
            verbose=2 if verbose else 0,
        )

        self.params_ = unpack(res.x)

        # compute fitted calls on training data using the DEFAULT evaluation grid (self.u_max/self.n_quad)
        C_fit = np.empty_like(C_obs)
        for Ti in np.unique(T_obs):
            idx = (T_obs == Ti)
            C_fit[idx] = self._call(K_obs[idx], float(Ti))
            
            
        self.fit_result_ = {
            "success": bool(res.success),
            "message": res.message,
            "x": res.x.copy(),
            "cost": float(res.cost),
            "nfev": int(res.nfev),
            "residuals": (C_fit - C_obs),
            "C_fit": C_fit,
            "optimizer": res,
            "use_fast": bool(use_fast),
            "fit_u_max": float(fit_u_max),
            "fit_n_quad": int(fit_n_quad),
            "using_default_bounds": bool(using_default_bounds),  # <-- ADD
        }

        
        # store training data for panel plots
        self.K_obs_ = K_obs.copy()
        self.T_obs_ = T_obs.copy()
        self.C_obs_ = C_obs.copy()
        self.weights_ = weights.copy()
        if verbose:
            print("Bates fit: using_default_bounds =", using_default_bounds)

        return self

    # ----------------------------
    # evaluate
    # ----------------------------
    def chat(self, strikes: np.ndarray, maturities: np.ndarray) -> np.ndarray:
        assert self.params_ is not None, "Call .fit(...) first."

        K = np.asarray(strikes, float)
        T = np.asarray(maturities, float)

        if K.ndim == 1 and T.ndim == 1:
            out = np.empty((T.size, K.size), float)
            for i, Ti in enumerate(T):
                out[i, :] = self._call(K, float(Ti))
            return out

        if K.shape == T.shape:
            out = np.empty_like(K, float)
            it = np.nditer(K, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                out[idx] = self._call(np.array([float(K[idx])]), float(T[idx]))[0]
                it.iternext()
            return out

        raise ValueError("Provide strikes & maturities as 1D arrays, or same-shaped grids.")

    def ivhat(self, strikes: np.ndarray, maturities: np.ndarray, *, vol_hi: float = 5.0) -> np.ndarray:
        C = self.chat(strikes, maturities)
        K = np.asarray(strikes, float)
        T = np.asarray(maturities, float)

        assert self.S0 is not None and self.r is not None and self.q is not None

        if K.ndim == 1 and T.ndim == 1:
            iv = np.empty_like(C, float)
            for i, Ti in enumerate(T):
                for j, Kj in enumerate(K):
                    iv[i, j] = implied_vol_call(
                        float(C[i, j]), self.S0, float(Kj), self.r, self.q, float(Ti),
                        vol_hi=vol_hi
                    )
            return iv

        if np.asarray(C).shape == np.asarray(K).shape == np.asarray(T).shape:
            iv = np.empty_like(C, float)
            it = np.nditer(C, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                iv[idx] = implied_vol_call(
                    float(C[idx]), self.S0, float(K[idx]), self.r, self.q, float(T[idx]),
                    vol_hi=vol_hi
                )
                it.iternext()
            return iv

        raise ValueError("Provide strikes & maturities as 1D arrays, or same-shaped grids.")

    def rndhat(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        *,
        clip_negative: bool = True,
    ) -> np.ndarray:
        assert self.params_ is not None, "Call .fit(...) first."
        assert self.r is not None

        K = np.asarray(strikes, float).ravel()
        T = np.asarray(maturities, float).ravel()

        if np.any(np.diff(K) <= 0):
            raise ValueError("strikes must be strictly increasing for rndhat.")

        C = self.chat(K, T)  # (nT, nK)

        q = np.empty_like(C, float)
        for i, Ti in enumerate(T):
            dC_dK = np.gradient(C[i, :], K, edge_order=2)
            d2C_dK2 = np.gradient(dC_dK, K, edge_order=2)
            q[i, :] = np.exp(self.r * float(Ti)) * d2C_dK2

        if clip_negative:
            q = np.maximum(q, 0.0)

        return q

    # ----------------------------
    # plotting (UNDER THE CLASS)
    # ----------------------------
    @staticmethod
    def _plot_surface_3d(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        zlabel: str,
        hillshade: bool = True,
    ):
        fig = plt.figure(figsize=(8.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")

        if hillshade:
            ls = LightSource(azdeg=315, altdeg=45)
            rgb = ls.shade(Z, cmap=plt.cm.gray, vert_exag=1.0, blend_mode="overlay")
            ax.plot_surface(X, Y, Z, facecolors=rgb, linewidth=0, antialiased=True, shade=False)
        else:
            ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        fig.tight_layout()
        return fig, ax

    def plot_call_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        *,
        title: str = "Bates fitted call surface",
        hillshade: bool = True,
        show: bool = True,
    ):
        K = np.asarray(strikes, float)
        T = np.asarray(maturities, float)
        C = self.chat(K, T)
        KK, TT = np.meshgrid(K, T)

        fig, ax = self._plot_surface_3d(
            KK, TT, C,
            title=title,
            xlabel="Strike K",
            ylabel="Maturity T (years)",
            zlabel="Call price C",
            hillshade=hillshade,
        )
        if show:
            plt.show()
        return fig, ax

    def plot_rnd_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        *,
        title: str = "Bates implied risk-neutral density surface",
        hillshade: bool = True,
        clip_negative: bool = True,
        show: bool = True,
    ):
        K = np.asarray(strikes, float)
        T = np.asarray(maturities, float)
        Q = self.rndhat(K, T, clip_negative=clip_negative)
        KK, TT = np.meshgrid(K, T)

        fig, ax = self._plot_surface_3d(
            KK, TT, Q,
            title=title,
            xlabel="Strike K",
            ylabel="Maturity T (years)",
            zlabel="q_T(K)",
            hillshade=hillshade,
        )
        if show:
            plt.show()
        return fig, ax

    def plot_call_and_rnd_surfaces(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        *,
        hillshade: bool = True,
        clip_negative: bool = True,
        show: bool = True,
    ):
        fig1, ax1 = self.plot_call_surface(strikes, maturities, hillshade=hillshade, show=False)
        fig2, ax2 = self.plot_rnd_surface(strikes, maturities, hillshade=hillshade, clip_negative=clip_negative, show=False)
        if show:
            plt.show()
        return (fig1, ax1), (fig2, ax2)

    def plot_call_fit_panels(
        self,
        *,
        maturities: Optional[np.ndarray] = None,
        n_panels: int = 6,
        strikes_fine: int = 200,
        sharey: bool = False,
        show: bool = True,
    ):
        assert self.params_ is not None, "Call .fit(...) first."
        assert self.K_obs_ is not None and self.T_obs_ is not None and self.C_obs_ is not None

        K_obs = self.K_obs_
        T_obs = self.T_obs_
        C_obs = self.C_obs_

        uniqT = np.unique(T_obs) if maturities is None else np.asarray(maturities, float).ravel()
        uniqT = np.array(sorted(uniqT))

        if uniqT.size == 0:
            raise ValueError("No maturities to plot.")

        m = min(int(n_panels), uniqT.size)
        idxs = np.linspace(0, uniqT.size - 1, m, dtype=int)
        Ts = uniqT[idxs]

        ncols = 3 if m >= 3 else m
        nrows = int(np.ceil(m / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), sharey=sharey)
        axes = np.atleast_1d(axes).ravel()

        for ax_i, Ti in enumerate(Ts):
            ax = axes[ax_i]
            mask = (T_obs == Ti)
            Ki = K_obs[mask]
            Ci = C_obs[mask]

            if Ki.size == 0:
                ax.set_axis_off()
                continue

            order = np.argsort(Ki)
            Ki = Ki[order]
            Ci = Ci[order]

            Kmin, Kmax = float(Ki.min()), float(Ki.max())
            Kfine = np.linspace(Kmin, Kmax, int(strikes_fine))
            Cfit = self._call(Kfine, float(Ti))

            ax.scatter(Ki, Ci, s=20, alpha=0.85, label="Observed")
            ax.plot(Kfine, Cfit, linewidth=2.0, label="Fitted")

            ax.set_title(f"T = {Ti:.4f} yr")
            ax.set_xlabel("Strike K")
            ax.set_ylabel("Call price C")
            ax.grid(True, alpha=0.25)

            y_all = np.r_[Ci, Cfit]
            ypad = 0.05 * (y_all.max() - y_all.min() + 1e-12)
            ax.set_ylim(y_all.min() - ypad, y_all.max() + ypad)

            if ax_i == 0:
                ax.legend(frameon=False)

        for j in range(m, axes.size):
            axes[j].set_axis_off()

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def plot_rnd_panels(
        self,
        *,
        maturities: Optional[np.ndarray] = None,
        n_panels: int = 6,
        strikes_fine: int = 500,
        strike_extension: float = 0.0,
        K_min: Optional[float] = None,
        K_max: Optional[float] = None,
        clip_negative: bool = True,
        show: bool = True,
    ):
        assert self.params_ is not None, "Call .fit(...) first."
        assert self.r is not None
        assert self.K_obs_ is not None and self.T_obs_ is not None

        K_obs_all = self.K_obs_
        T_obs_all = self.T_obs_

        uniqT = np.unique(T_obs_all) if maturities is None else np.asarray(maturities, float).ravel()
        uniqT = np.array(sorted(uniqT))
        if uniqT.size == 0:
            raise ValueError("No maturities to plot.")

        m = min(int(n_panels), uniqT.size)
        idxs = np.linspace(0, uniqT.size - 1, m, dtype=int)
        Ts = uniqT[idxs]

        ncols = 3 if m >= 3 else m
        nrows = int(np.ceil(m / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax_i, Ti in enumerate(Ts):
            ax = axes[ax_i]

            mask = (T_obs_all == Ti)
            Ki_obs = K_obs_all[mask]

            if Ki_obs.size == 0:
                ax.set_axis_off()
                continue

            kmin, kmax = self._extended_range_from_observed(
                Ki_obs,
                strike_extension=strike_extension,
                K_min=K_min,
                K_max=K_max,
            )

            Kfine = np.linspace(kmin, kmax, int(strikes_fine))
            Cfit = self._call(Kfine, float(Ti))

            dC_dK_fit = np.gradient(Cfit, Kfine, edge_order=2)
            d2C_dK2_fit = np.gradient(dC_dK_fit, Kfine, edge_order=2)
            q_fit = np.exp(self.r * float(Ti)) * d2C_dK2_fit

            if clip_negative:
                q_fit = np.maximum(q_fit, 0.0)

            ax.plot(Kfine, q_fit, linewidth=2.0, label="Fitted BL")

            ax.set_title(f"T = {Ti:.4f} yr")
            ax.set_xlabel("Strike K")
            ax.set_ylabel("q_T(K)")
            ax.grid(True, alpha=0.25)

            if ax_i == 0:
                ax.legend(frameon=False)

        for j in range(m, axes.size):
            axes[j].set_axis_off()

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes


# ============================================================
# 5) Minimal end-to-end example (synthetic data)
# ============================================================

if __name__ == "__main__":
    true_p = BatesParams(
        kappa=2.0, theta=0.04, sigma=0.6, v0=0.04, rho=-0.7,
        lam=0.4, muJ=-0.08, sigJ=0.25
    )
    S0, r, q = 100.0, 0.03, 0.00

    # synthetic "market" surface
    T_grid = np.array([0.10, 0.25, 0.50, 1.00])
    K_grid = np.linspace(60, 140, 17)
    C_surf = np.vstack([bates_call_price(S0, K_grid, r, q, float(Ti), true_p, u_max=240.0, n_quad=96) for Ti in T_grid])

    # long-form observations
    TT, KK = np.meshgrid(T_grid, K_grid, indexing="ij")
    K_obs, T_obs, C_obs = KK.ravel(), TT.ravel(), C_surf.ravel()

    # add noise
    rng = np.random.default_rng(0)
    C_obs = np.maximum(C_obs + rng.normal(scale=0.05, size=C_obs.shape), 0.0)

    # weights
    w = 1.0 / np.maximum(C_obs, 0.25)

    # Example dict bounds:
    lb_d = dict(kappa=0.30, theta=0.005, sigma=0.10, v0=0.005, rho=-0.85, lam=0.02, muJ=-0.20, sigJ=0.05)
    ub_d = dict(kappa=12.0, theta=0.90,  sigma=0.75, v0=0.90,  rho=0.80,  lam=1.50, muJ=0.10,  sigJ=0.60)

    fit = BatesSurfaceFitter(u_max=240.0, n_quad=96).fit(
        S0=S0, K_obs=K_obs, T_obs=T_obs, C_obs=C_obs, r=r, q=q,
        weights=w, verbose=1, max_nfev=120,
        use_fast=True,
        fit_u_max=140.0,
        fit_n_quad=64,
        bounds=(lb_d, ub_d),   # <-- dict form
    )

    print("Fitted params:", fit.params_)
    print("Success:", fit.fit_result_["success"], "| Cost:", fit.fit_result_["cost"])

    fit.plot_call_fit_panels(n_panels=6)

    K_plot = np.linspace(0.5 * S0, 1.5 * S0, 121)
    T_plot = np.array(np.linspace(0.1, 1, 100))
    fit.plot_call_and_rnd_surfaces(K_plot, T_plot)

    fit.plot_rnd_panels(n_panels=6, strikes_fine=600, strike_extension=0.30)
