from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.interpolate import SmoothBivariateSpline, UnivariateSpline
from scipy.optimize import brentq, curve_fit
from scipy.stats import norm

from .base import GlobalModel, FitResult
from ..registry import register_model


# ============================================================
# Global IV(delta,T) smoothing spline with stable maturity term
# ------------------------------------------------------------
# Workflow:
#   1) Convert observed call prices C(K,T) -> Black-Scholes IV.
#   2) For each option, convert K -> call delta using that maturity's
#      ATM IV. This follows the Bank of England / Malz logic:
#      BS is used only as a coordinate transform.
#   3) Add pseudo delta-tail points near delta=0 and delta=1 with
#      endpoint IVs. This forces horizontal IV extrapolation in delta.
#   4) Estimate a stable maturity term structure g(T), default:
#
#          g(T) = a + b * (1 - exp(-cT))
#
#   5) Fit a 2D smoothing spline to residuals:
#
#          IV(delta,T) - g(T)
#
#   6) Pricing:
#
#          IV(delta,T) = g(T) + spline(delta,T)
#          C(K,T) = BS_call(K,T,IV)
#
#   7) Maturity interpolation/extrapolation:
#      The term structure g(T) governs the global level; the residual
#      spline is evaluated with linear-tail safeguards in T.
#
# Registered as model_name="splines".
# ============================================================


@dataclass(frozen=True)
class GlobalIVDeltaSplineParams:
    smoothing: float
    pseudo_intervals: float
    min_iv: float
    max_iv: float
    min_delta: float
    max_delta: float
    term_code: float
    residual_clip: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "smoothing": float(self.smoothing),
            "pseudo_intervals": float(self.pseudo_intervals),
            "min_iv": float(self.min_iv),
            "max_iv": float(self.max_iv),
            "min_delta": float(self.min_delta),
            "max_delta": float(self.max_delta),
            "term_code": float(self.term_code),
            "residual_clip": float(self.residual_clip),
        }


def _bs_call_price_vec(S0: float, K, T, r: float, q: float, sigma) -> np.ndarray:
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    sigma = np.asarray(sigma, float)
    K, T, sigma = np.broadcast_arrays(K, T, sigma)

    out = np.maximum(S0 * np.exp(-q * np.maximum(T, 0.0)) - K * np.exp(-r * np.maximum(T, 0.0)), 0.0)

    m = (T > 0) & (K > 0) & (S0 > 0) & (sigma > 0) & np.isfinite(sigma)
    if not np.any(m):
        return out.astype(float)

    vol_sqrtT = sigma[m] * np.sqrt(T[m])
    d1 = (np.log(S0 / K[m]) + (r - q + 0.5 * sigma[m] ** 2) * T[m]) / vol_sqrtT
    d2 = d1 - vol_sqrtT

    out[m] = S0 * np.exp(-q * T[m]) * norm.cdf(d1) - K[m] * np.exp(-r * T[m]) * norm.cdf(d2)
    return np.asarray(out, float)


def _bs_delta_vec(S0: float, K, T, r: float, q: float, sigma) -> np.ndarray:
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    sigma = np.asarray(sigma, float)
    K, T, sigma = np.broadcast_arrays(K, T, sigma)

    out = np.zeros_like(K, dtype=float)
    m = (T > 0) & (K > 0) & (S0 > 0) & (sigma > 0) & np.isfinite(sigma)
    if not np.any(m):
        return out

    vol_sqrtT = sigma[m] * np.sqrt(T[m])
    d1 = (np.log(S0 / K[m]) + (r - q + 0.5 * sigma[m] ** 2) * T[m]) / vol_sqrtT
    out[m] = np.exp(-q * T[m]) * norm.cdf(d1)
    return np.asarray(out, float)


def _bs_vega_vec(S0: float, K, T, r: float, q: float, sigma) -> np.ndarray:
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    sigma = np.asarray(sigma, float)
    K, T, sigma = np.broadcast_arrays(K, T, sigma)

    out = np.zeros_like(K, dtype=float)
    m = (T > 0) & (K > 0) & (S0 > 0) & (sigma > 0) & np.isfinite(sigma)
    if not np.any(m):
        return out

    vol_sqrtT = sigma[m] * np.sqrt(T[m])
    d1 = (np.log(S0 / K[m]) + (r - q + 0.5 * sigma[m] ** 2) * T[m]) / vol_sqrtT
    out[m] = S0 * np.exp(-q * T[m]) * norm.pdf(d1) * np.sqrt(T[m])
    return np.asarray(out, float)


def _call_bounds(S0: float, K, T, r: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    K, T = np.broadcast_arrays(K, T)
    lower = np.maximum(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper = S0 * np.exp(-q * T)
    return lower, upper


def _implied_vol_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    C: float,
    *,
    min_iv: float,
    max_iv: float,
) -> float:
    if not (np.isfinite(K) and np.isfinite(T) and np.isfinite(C) and K > 0 and T > 0 and C >= 0):
        return np.nan

    lb, ub = _call_bounds(S0, np.array([K]), np.array([T]), r, q)
    lb = float(lb[0])
    ub = float(ub[0])

    C = float(np.clip(C, lb + 1e-12, ub - 1e-12))

    def f(sig: float) -> float:
        return float(_bs_call_price_vec(S0, np.array([K]), np.array([T]), r, q, np.array([sig]))[0] - C)

    try:
        flo = f(min_iv)
        fhi = f(max_iv)
        if flo * fhi > 0:
            return float(min_iv if abs(flo) < abs(fhi) else max_iv)
        return float(brentq(f, float(min_iv), float(max_iv), xtol=1e-11, rtol=1e-11, maxiter=150))
    except Exception:
        return np.nan


def _clean_inputs(K, T, C, S0: float, r: float, q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(K, float).ravel()
    T = np.asarray(T, float).ravel()
    C = np.asarray(C, float).ravel()

    m = np.isfinite(K) & np.isfinite(T) & np.isfinite(C) & (K > 0) & (T > 0) & (C >= 0)
    K, T, C = K[m], T[m], C[m]

    if K.size == 0:
        return K, T, C

    # Clip to no-arb bounds.
    lb, ub = _call_bounds(S0, K, T, r, q)
    C = np.clip(C, lb, ub)

    return K, T, C


def _exponential_term(T: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    T = np.asarray(T, float)
    c = max(float(c), 1e-8)
    return a + b * (1.0 - np.exp(-c * T))


def _nelson_siegel_term(T: np.ndarray, beta0: float, beta1: float, beta2: float, lam: float) -> np.ndarray:
    T = np.asarray(T, float)
    lam = max(float(lam), 1e-8)
    x = lam * np.maximum(T, 1e-10)
    level = (1.0 - np.exp(-x)) / x
    slope = level - np.exp(-x)
    return beta0 + beta1 * level + beta2 * slope


class _LinearTail1D:
    """
    Linear-tail wrapper around UnivariateSpline for stable maturity extrapolation.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, *, s: float = 0.0, k: int = 3):
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        ux, inv = np.unique(x, return_inverse=True)
        if ux.size != x.size:
            yy = np.zeros_like(ux)
            nn = np.zeros_like(ux)
            for i, j in enumerate(inv):
                yy[j] += y[i]
                nn[j] += 1.0
            x = ux
            y = yy / np.maximum(nn, 1.0)

        self.x_min = float(x[0])
        self.x_max = float(x[-1])
        self.y_min = float(y[0])
        self.y_max = float(y[-1])
        self.constant = x.size == 1

        kk = int(min(k, max(1, x.size - 1)))
        self.spline = UnivariateSpline(x, y, s=float(s), k=kk, ext=0)
        self.left_slope = float(self.spline.derivative()(self.x_min)) if x.size > 1 else 0.0
        self.right_slope = float(self.spline.derivative()(self.x_max)) if x.size > 1 else 0.0

    def __call__(self, xnew) -> np.ndarray:
        z = np.asarray(xnew, float)
        if self.constant:
            return np.full_like(z, self.y_min, dtype=float)
        out = np.asarray(self.spline(z), float)
        left = z < self.x_min
        right = z > self.x_max
        out[left] = self.y_min + self.left_slope * (z[left] - self.x_min)
        out[right] = self.y_max + self.right_slope * (z[right] - self.x_max)
        return out


@register_model("splines")
class SplinesModel(GlobalModel):
    """
    Global IV(delta,T) smoothing spline.

    Example:

        pr = GlobalSurfacePricer("splines")

        pr.fit(day, x0={
            "smoothing": 0.99,
            "term_structure": "exponential",
            "pseudo_intervals": 3.0,
            "min_iv": 1e-4,
            "max_iv": 6.0,
            "min_strikes_per_maturity": 4,
            "verbose": True,
        })

    Notes
    -----
    smoothing:
        Paper-style number in [0,1). Larger means smoother.
        This maps pragmatically into scipy's spline smoothing target.

    term_structure:
        "exponential"      : a + b(1-exp(-cT)), recommended default.
        "nelson_siegel"    : four-parameter Nelson-Siegel.
        "global_spline"    : 1D maturity spline with linear tails.
        "none"             : no term structure.
    """

    name = "splines"

    def __init__(self, *, S0: float, r: float, q: float = 0.0, Umax: float = 200.0, n_quad: int = 96):
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self.Umax = float(Umax)
        self.n_quad = int(n_quad)

        self.params_: Optional[GlobalIVDeltaSplineParams] = None
        self.term_structure_: str = "exponential"
        self.term_params_: Optional[np.ndarray] = None
        self.term_spline_: Optional[_LinearTail1D] = None
        self.resid_spline_: Optional[SmoothBivariateSpline] = None

        self.T_min_: float = np.nan
        self.T_max_: float = np.nan
        self.delta_min_: float = np.nan
        self.delta_max_: float = np.nan
        self.fit_info_: Dict[str, float] = {}

    def _term_eval(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, float)
        if self.term_structure_ == "none":
            return np.zeros_like(T, dtype=float)

        if self.term_structure_ == "exponential":
            if self.term_params_ is None:
                raise RuntimeError("Missing exponential term parameters.")
            return _exponential_term(T, *self.term_params_)

        if self.term_structure_ == "nelson_siegel":
            if self.term_params_ is None:
                raise RuntimeError("Missing Nelson-Siegel term parameters.")
            return _nelson_siegel_term(T, *self.term_params_)

        if self.term_structure_ == "global_spline":
            if self.term_spline_ is None:
                raise RuntimeError("Missing maturity spline.")
            return self.term_spline_(T)

        raise ValueError(f"Unknown term_structure {self.term_structure_!r}.")

    @staticmethod
    def _fit_term_structure(T: np.ndarray, iv: np.ndarray, weights: np.ndarray, term_structure: str):
        T = np.asarray(T, float).ravel()
        iv = np.asarray(iv, float).ravel()
        weights = np.asarray(weights, float).ravel()
        weights = np.maximum(weights, 1e-8)

        term_structure = term_structure.lower().strip()

        if term_structure == "none":
            return None, None

        # Collapse duplicated maturities using weighted average IV level.
        unique_T = np.unique(T)
        T_level = []
        iv_level = []
        w_level = []
        for tt in unique_T:
            idx = np.where(np.isclose(T, tt))[0]
            if idx.size == 0:
                continue
            ww = weights[idx]
            T_level.append(float(tt))
            iv_level.append(float(np.average(iv[idx], weights=ww)))
            w_level.append(float(np.sum(ww)))

        T_level = np.asarray(T_level, float)
        iv_level = np.asarray(iv_level, float)
        w_level = np.asarray(w_level, float)

        if T_level.size == 0:
            return None, None

        if term_structure == "global_spline":
            s = max(0.01 * T_level.size * float(np.var(iv_level)), 0.0)
            return None, _LinearTail1D(T_level, iv_level, s=s, k=min(3, T_level.size - 1))

        if term_structure == "exponential":
            if T_level.size < 3:
                # fallback constant level with tiny b,c
                level = float(np.average(iv_level, weights=w_level))
                return np.array([level, 0.0, 1.0], float), None

            a0 = float(iv_level[-1])
            b0 = float(iv_level[0] - iv_level[-1])
            c0 = 2.0

            def f(TT, a, b, c):
                return _exponential_term(TT, a, b, c)

            try:
                popt, _ = curve_fit(
                    f,
                    T_level,
                    iv_level,
                    p0=np.array([a0, b0, c0], float),
                    sigma=1.0 / np.sqrt(np.maximum(w_level, 1e-8)),
                    bounds=([1e-5, -5.0, 1e-5], [10.0, 5.0, 100.0]),
                    maxfev=20000,
                )
                return np.asarray(popt, float), None
            except Exception:
                level = float(np.average(iv_level, weights=w_level))
                return np.array([level, 0.0, 1.0], float), None

        if term_structure == "nelson_siegel":
            if T_level.size < 4:
                level = float(np.average(iv_level, weights=w_level))
                return np.array([level, 0.0, 0.0, 1.0], float), None

            beta0 = float(iv_level[-1])
            beta1 = float(iv_level[0] - iv_level[-1])
            beta2 = 0.0
            lam = 2.0

            def f(TT, b0, b1, b2, l):
                return _nelson_siegel_term(TT, b0, b1, b2, l)

            try:
                popt, _ = curve_fit(
                    f,
                    T_level,
                    iv_level,
                    p0=np.array([beta0, beta1, beta2, lam], float),
                    sigma=1.0 / np.sqrt(np.maximum(w_level, 1e-8)),
                    bounds=([1e-5, -5.0, -5.0, 1e-5], [10.0, 5.0, 5.0, 100.0]),
                    maxfev=30000,
                )
                return np.asarray(popt, float), None
            except Exception:
                level = float(np.average(iv_level, weights=w_level))
                return np.array([level, 0.0, 0.0, 1.0], float), None

        raise ValueError("term_structure must be 'exponential', 'nelson_siegel', 'global_spline', or 'none'.")

    def fit(
        self,
        K_obs,
        T_obs,
        C_obs,
        x0: Optional[Dict[str, float]] = None,
        bounds=None,
        max_nfev: int = 200,
        **kwargs,
    ) -> FitResult:
        cfg = dict(x0 or {})

        smoothing = float(cfg.get("smoothing", cfg.get("smoothing_lambda", cfg.get("lambda", 0.99))))
        term_structure = str(cfg.get("term_structure", "exponential")).lower().strip()
        pseudo_intervals = float(cfg.get("pseudo_intervals", 3.0))
        min_iv = float(cfg.get("min_iv", 1e-4))
        max_iv = float(cfg.get("max_iv", 6.0))
        min_delta = float(cfg.get("min_delta", 1e-5))
        max_delta = float(cfg.get("max_delta", 1.0 - 1e-5))
        min_strikes_per_maturity = int(cfg.get("min_strikes_per_maturity", cfg.get("min_strikes", 4)))
        residual_clip = float(cfg.get("residual_clip", 2.0))
        verbose = bool(cfg.get("verbose", False))

        K, T, C = _clean_inputs(K_obs, T_obs, C_obs, self.S0, self.r, self.q)
        if K.size == 0:
            raise ValueError("No valid call quotes for SplinesModel.")

        # Convert prices to IVs.
        iv = np.array(
            [
                _implied_vol_call(self.S0, float(k), float(t), self.r, self.q, float(c), min_iv=min_iv, max_iv=max_iv)
                for k, t, c in zip(K, T, C)
            ],
            dtype=float,
        )
        m = np.isfinite(iv) & (iv > 0)
        K, T, C, iv = K[m], T[m], C[m], iv[m]

        if K.size < 8:
            raise ValueError("Too few valid implied volatilities for global IV/delta spline.")

        # Maturity-level ATM IV used for K->delta coordinate transform.
        atm_iv_by_T: Dict[float, float] = {}
        T_unique = np.unique(T)
        for tt in T_unique:
            idx = np.where(np.isclose(T, tt))[0]
            if idx.size < min_strikes_per_maturity:
                continue
            F = self.S0 * np.exp((self.r - self.q) * float(tt))
            atm_idx_local = idx[np.argmin(np.abs(K[idx] - F))]
            atm_iv_by_T[float(tt)] = float(np.clip(iv[atm_idx_local], min_iv, max_iv))

        keep = np.array([float(tt) in atm_iv_by_T for tt in T], dtype=bool)
        K, T, C, iv = K[keep], T[keep], C[keep], iv[keep]
        if K.size < 8:
            raise ValueError("Too few maturities/strikes after ATM-IV maturity filtering.")

        atm_iv_vec = np.array([atm_iv_by_T[float(tt)] for tt in T], dtype=float)

        delta = _bs_delta_vec(self.S0, K, T, self.r, self.q, atm_iv_vec)
        delta = np.clip(delta, min_delta, max_delta)

        # Vega weights for IV errors.
        vega = _bs_vega_vec(self.S0, K, T, self.r, self.q, iv)
        vega = np.maximum(vega, 1e-8)

        # Add horizontal pseudo tails in delta/maturity space.
        delta_aug = [delta]
        T_aug = [T]
        iv_aug = [iv]
        w_aug = [vega]

        pseudo_count = 0
        for tt in np.unique(T):
            idx = np.where(np.isclose(T, tt))[0]
            if idx.size < min_strikes_per_maturity:
                continue

            order = idx[np.argsort(K[idx])]
            K_t = K[order]
            iv_t = iv[order]
            T_t = float(tt)

            diffs = np.diff(K_t)
            avg_dK = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else max(0.05 * self.S0, 1e-6)
            K_left = max(1e-8, float(K_t[0] - pseudo_intervals * avg_dK))
            K_right = float(K_t[-1] + pseudo_intervals * avg_dK)
            atm_iv_t = atm_iv_by_T[T_t]

            d_left = _bs_delta_vec(self.S0, np.array([K_left]), np.array([T_t]), self.r, self.q, np.array([atm_iv_t]))[0]
            d_right = _bs_delta_vec(self.S0, np.array([K_right]), np.array([T_t]), self.r, self.q, np.array([atm_iv_t]))[0]
            d_left = float(np.clip(d_left, min_delta, max_delta))
            d_right = float(np.clip(d_right, min_delta, max_delta))

            med_w = float(np.median(vega[idx]))
            pseudo_w = max(0.25 * med_w, 1e-8)

            delta_aug.append(np.array([d_left, d_right], float))
            T_aug.append(np.array([T_t, T_t], float))
            iv_aug.append(np.array([iv_t[0], iv_t[-1]], float))
            w_aug.append(np.array([pseudo_w, pseudo_w], float))
            pseudo_count += 2

        delta_all = np.concatenate(delta_aug)
        T_all = np.concatenate(T_aug)
        iv_all = np.concatenate(iv_aug)
        weights_all = np.concatenate(w_aug)

        # Estimate stable maturity term g(T).
        self.term_structure_ = term_structure
        term_params, term_spline = self._fit_term_structure(T_all, iv_all, weights_all, term_structure)
        self.term_params_ = term_params
        self.term_spline_ = term_spline

        g_all = self._term_eval(T_all)
        resid = iv_all - g_all

        # Clip residuals to prevent pathological far extrapolation.
        resid = np.clip(resid, -abs(residual_clip), abs(residual_clip))

        # Convert paper-style smoothing in [0,1) to scipy s.
        smoothing = float(np.clip(smoothing, 0.0, 0.999999))
        resid_var = float(np.average((resid - np.average(resid, weights=np.maximum(weights_all, 1e-8))) ** 2,
                                     weights=np.maximum(weights_all, 1e-8)))
        s_target = max(smoothing * len(resid) * resid_var, 0.0)

        # SmoothBivariateSpline weights use positive weights. Scale weights for numerical stability.
        w_spline = np.sqrt(weights_all / np.median(weights_all))
        w_spline = np.clip(w_spline, 1e-4, 1e4)

        # Need enough points for cubic in both dimensions; fallback degrees if sparse.
        nT = np.unique(T_all).size
        nD = np.unique(np.round(delta_all, 10)).size
        kx = 3 if nD >= 4 else max(1, nD - 1)
        ky = 3 if nT >= 4 else max(1, nT - 1)

        last_err = None
        self.resid_spline_ = None
        for mult in (1.0, 10.0, 100.0, 1000.0):
            try:
                self.resid_spline_ = SmoothBivariateSpline(
                    delta_all,
                    T_all,
                    resid,
                    w=w_spline,
                    s=s_target * mult,
                    kx=kx,
                    ky=ky,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e

        if self.resid_spline_ is None:
            raise RuntimeError(f"Could not fit global IV(delta,T) spline: {last_err}")

        self.T_min_ = float(np.min(T_all))
        self.T_max_ = float(np.max(T_all))
        self.delta_min_ = float(np.min(delta_all))
        self.delta_max_ = float(np.max(delta_all))

        term_code = {
            "none": 0.0,
            "exponential": 1.0,
            "nelson_siegel": 2.0,
            "global_spline": 3.0,
        }[term_structure]

        self.params_ = GlobalIVDeltaSplineParams(
            smoothing=smoothing,
            pseudo_intervals=pseudo_intervals,
            min_iv=min_iv,
            max_iv=max_iv,
            min_delta=min_delta,
            max_delta=max_delta,
            term_code=term_code,
            residual_clip=residual_clip,
        )

        self.fit_info_ = {
            "n_iv": float(K.size),
            "n_augmented": float(len(iv_all)),
            "n_pseudo": float(pseudo_count),
            "n_maturities": float(np.unique(T).size),
            "T_min": self.T_min_,
            "T_max": self.T_max_,
            "delta_min": self.delta_min_,
            "delta_max": self.delta_max_,
            "s_target": float(s_target),
            "kx": float(kx),
            "ky": float(ky),
        }

        if verbose:
            print("[global_iv_delta_spline] fit_info:", self.fit_info_)
            if self.term_params_ is not None:
                print("[global_iv_delta_spline] term_params:", self.term_params_)

        return FitResult(params=self.params_.to_dict(), success=True, info=dict(self.fit_info_))

    def _iv_surface(self, K_grid: np.ndarray, T_grid: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        if self.resid_spline_ is None:
            raise RuntimeError("Call fit(...) before price_surface(...).")

        K_grid = np.asarray(K_grid, float).ravel()
        T_grid = np.asarray(T_grid, float).ravel()

        min_iv = float(params.get("min_iv", self.params_.min_iv if self.params_ else 1e-4))
        max_iv = float(params.get("max_iv", self.params_.max_iv if self.params_ else 6.0))
        min_delta = float(params.get("min_delta", self.params_.min_delta if self.params_ else 1e-5))
        max_delta = float(params.get("max_delta", self.params_.max_delta if self.params_ else 1.0 - 1e-5))
        residual_clip = float(params.get("residual_clip", self.params_.residual_clip if self.params_ else 2.0))

        out = np.zeros((T_grid.size, K_grid.size), float)

        for i, tt in enumerate(T_grid):
            tt_safe = max(float(tt), 1e-10)

            # Use term structure IV as the coordinate-transform volatility.
            term_iv = float(np.clip(self._term_eval(np.array([tt_safe]))[0], min_iv, max_iv))
            delta = _bs_delta_vec(
                self.S0,
                K_grid,
                np.full_like(K_grid, tt_safe),
                self.r,
                self.q,
                np.full_like(K_grid, term_iv),
            )
            delta = np.clip(delta, min_delta, max_delta)

            # Stable maturity extrapolation:
            # SmoothBivariateSpline itself can extrapolate poorly outside T.
            # So clamp residual spline evaluation in T to fit range and let g(T)
            # govern maturity extrapolation.
            t_eval = float(np.clip(tt_safe, self.T_min_, self.T_max_))
            resid = np.asarray(self.resid_spline_.ev(delta, np.full_like(delta, t_eval)), float)
            resid = np.clip(resid, -abs(residual_clip), abs(residual_clip))

            iv = self._term_eval(np.full_like(delta, tt_safe)) + resid
            out[i, :] = np.clip(iv, min_iv, max_iv)

        return out

    def call_prices(self, K: np.ndarray, T: float, params: Dict[str, float], **kwargs) -> np.ndarray:
        K = np.asarray(K, float).ravel()
        C = self.price_surface(K, np.array([float(T)]), params, **kwargs)
        return np.asarray(C[0, :], float)

    def price_surface(self, K_grid: np.ndarray, T_grid: np.ndarray, params: Dict[str, float], **kwargs) -> np.ndarray:
        if self.resid_spline_ is None:
            raise RuntimeError("Call fit(...) before price_surface(...).")

        K_grid = np.asarray(K_grid, float).ravel()
        T_grid = np.asarray(T_grid, float).ravel()
        T_grid = np.maximum(T_grid, 1e-10)

        params = dict(params or {})
        iv = self._iv_surface(K_grid, T_grid, params)

        C = np.zeros((T_grid.size, K_grid.size), float)
        for i, tt in enumerate(T_grid):
            C[i, :] = _bs_call_price_vec(
                self.S0,
                K_grid,
                np.full_like(K_grid, float(tt)),
                self.r,
                self.q,
                iv[i, :],
            )
            lb, ub = _call_bounds(self.S0, K_grid, float(tt), self.r, self.q)
            C[i, :] = np.clip(C[i, :], lb, ub)

            # light monotonicity cleanup
            if K_grid.size > 1 and np.all(np.diff(K_grid) >= 0):
                C[i, :] = np.minimum.accumulate(C[i, :])
                C[i, :] = np.clip(C[i, :], lb, ub)

        return C
