from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from .base import GlobalModel, FitResult
from ..registry import register_model


# ============================================================
# Direct mixture of N lognormals + call-price maturity interpolation
# ------------------------------------------------------------
# This model does exactly this:
#
#   1) For each observed maturity T_i independently, fit a mixture of
#      N lognormal risk-neutral densities directly to the observed call
#      curve C(K,T_i).
#
#   2) For any requested maturity T*, evaluate the fitted mixture call
#      curves at the requested strike grid K_grid for all observed T_i.
#
#   3) Interpolate/extrapolate CALL PRICES across maturity at each strike:
#
#          C_synth(K,T*) = linear_T_interp[ C_fit(K,T_i) ]
#
#   4) Fit a fresh mixture of N lognormals to that synthetic call curve
#      C_synth(K,T*).
#
#   5) Return the call prices from the fitted mixture at T*.
#
# Important:
#   - No implied-volatility inversion.
#   - No spline pre-smoothing.
#   - No interpolation of mixture parameters across maturity.
#   - Maturity interpolation/extrapolation happens in CALL PRICE space.
#
# Registered names:
#   "lognormal_mixture"
#   "mixture_lognormal"
# ============================================================


@dataclass(frozen=True)
class LognormalMixtureParams:
    n_components: float
    min_sigma: float
    max_sigma: float
    weight_floor: float
    mean_ratio_min: float
    mean_ratio_max: float
    variance_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_components": float(self.n_components),
            "min_sigma": float(self.min_sigma),
            "max_sigma": float(self.max_sigma),
            "weight_floor": float(self.weight_floor),
            "mean_ratio_min": float(self.mean_ratio_min),
            "mean_ratio_max": float(self.mean_ratio_max),
            "variance_ratio": float(self.variance_ratio),
        }


def _call_bounds(S0: float, K, T, r: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    K, T = np.broadcast_arrays(K, T)
    lower = np.maximum(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper = S0 * np.exp(-q * T)
    return lower, upper


def _clean_inputs(K, T, C, S0: float, r: float, q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(K, float).ravel()
    T = np.asarray(T, float).ravel()
    C = np.asarray(C, float).ravel()

    m = np.isfinite(K) & np.isfinite(T) & np.isfinite(C) & (K > 0) & (T > 0) & (C >= 0)
    K, T, C = K[m], T[m], C[m]

    if K.size == 0:
        return K, T, C

    lb, ub = _call_bounds(S0, K, T, r, q)
    C = np.clip(C, lb, ub)

    return K, T, C


def _unique_strike_curve(K: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(K, float).ravel()
    C = np.asarray(C, float).ravel()

    order = np.argsort(K)
    K, C = K[order], C[order]

    uK = np.unique(K)
    if uK.size != K.size:
        CC = np.zeros_like(uK)
        for i, kk in enumerate(uK):
            CC[i] = float(np.median(C[np.isclose(K, kk)]))
        K, C = uK, CC

    return K, C


def _softmax(z: np.ndarray, floor: float = 0.0) -> np.ndarray:
    z = np.asarray(z, float).ravel()
    z = z - np.max(z)
    e = np.exp(z)
    w = e / np.sum(e)

    if floor > 0:
        n = w.size
        floor = min(float(floor), 0.49 / n)
        w = floor + (1.0 - n * floor) * w
        w = w / np.sum(w)

    return w


def _weights_from_logits(logits: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    logits = np.asarray(logits, float).ravel()
    z = np.concatenate([logits, [0.0]])
    return _softmax(z, floor=floor)


def _normalize_mean_ratios(raw_ratios: np.ndarray, weights: np.ndarray, lo: float, hi: float) -> np.ndarray:
    raw = np.clip(np.asarray(raw_ratios, float).ravel(), lo, hi)
    weights = np.asarray(weights, float).ravel()

    denom = float(np.sum(weights * raw))
    if denom <= 1e-12 or not np.isfinite(denom):
        return np.ones_like(raw)

    ratios = raw / denom
    ratios = np.clip(ratios, lo, hi)

    # Re-normalize once more after clipping.
    denom2 = float(np.sum(weights * ratios))
    if denom2 > 1e-12 and np.isfinite(denom2):
        ratios = ratios / denom2

    return ratios


def _call_lognormal_forward(K: np.ndarray, T: float, r: float, F: float, sigma: float) -> np.ndarray:
    """
    Discounted call price when S_T is lognormal with mean F.

    If E[S_T]=F and log-vol is sigma:
        C = exp(-rT) [ F N(d1) - K N(d2) ]
        d1 = [ln(F/K) + 0.5 sigma^2 T] / (sigma sqrt(T))
        d2 = d1 - sigma sqrt(T)
    """
    K = np.asarray(K, float).ravel()
    T = float(T)
    F = float(F)
    sigma = float(sigma)

    if T <= 0 or sigma <= 0:
        return np.maximum(F - K, 0.0)

    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT

    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def _mixture_call_prices(
    K: np.ndarray,
    T: float,
    r: float,
    F0: float,
    weights: np.ndarray,
    mean_ratios: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    K = np.asarray(K, float).ravel()
    out = np.zeros_like(K, dtype=float)

    for pi, mr, sig in zip(weights, mean_ratios, sigmas):
        Fj = F0 * float(mr)
        out += float(pi) * _call_lognormal_forward(K, T, r, Fj, float(sig))

    return out


def _linear_interp_extrap_T(T_fit: np.ndarray, Y_fit: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
    """
    Linear interpolation/extrapolation in maturity.

    T_fit : shape (nT_fit,)
    Y_fit : shape (nT_fit, nK)
    T_grid: shape (nT_out,)

    Returns shape (nT_out, nK).
    """
    T_fit = np.asarray(T_fit, float).ravel()
    T_grid = np.asarray(T_grid, float).ravel()
    Y_fit = np.asarray(Y_fit, float)

    if T_fit.size == 1:
        return np.repeat(Y_fit[:1, :], T_grid.size, axis=0)

    out = np.zeros((T_grid.size, Y_fit.shape[1]), dtype=float)

    for j in range(Y_fit.shape[1]):
        y = Y_fit[:, j]
        vals = np.interp(T_grid, T_fit, y)

        left = T_grid < T_fit[0]
        if np.any(left):
            slope = (y[1] - y[0]) / (T_fit[1] - T_fit[0])
            vals[left] = y[0] + slope * (T_grid[left] - T_fit[0])

        right = T_grid > T_fit[-1]
        if np.any(right):
            slope = (y[-1] - y[-2]) / (T_fit[-1] - T_fit[-2])
            vals[right] = y[-1] + slope * (T_grid[right] - T_fit[-1])

        out[:, j] = vals

    return out


@register_model("lognormal_mixture")
@register_model("mixture_lognormal")
class LognormalMixtureModel(GlobalModel):
    """
    Direct mixture-of-lognormals call-curve model.

    Fit:
        Fit each observed maturity independently.

    Price:
        For requested T_grid:
          1. Evaluate observed-maturity fitted mixture call curves on K_grid.
          2. Interpolate/extrapolate call prices across maturity.
          3. Fit a fresh mixture to each synthetic requested-maturity curve.
          4. Return the refitted mixture prices.

    Example
    -------
    pr = GlobalSurfacePricer("lognormal_mixture")

    pr.fit(day, x0={
        "n_components": 3,
        "variance_ratio": 0.05,
        "min_sigma": 1e-4,
        "max_sigma": 6.0,
        "weight_floor": 1e-6,
        "mean_ratio_min": 0.25,
        "mean_ratio_max": 4.0,
        "min_strikes_per_maturity": 8,
        "verbose": True,
    })
    """

    name = "lognormal_mixture"

    def __init__(self, *, S0: float, r: float, q: float = 0.0, Umax: float = 200.0, n_quad: int = 96):
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self.Umax = float(Umax)
        self.n_quad = int(n_quad)

        self.params_: Optional[LognormalMixtureParams] = None
        self.N_: int = 0
        self.T_fit_: Optional[np.ndarray] = None

        self.weights_fit_: Optional[np.ndarray] = None
        self.mean_ratios_fit_: Optional[np.ndarray] = None
        self.sigmas_fit_: Optional[np.ndarray] = None

        self.fit_info_: Dict[str, float] = {}

    def _fit_one_curve(
        self,
        K: np.ndarray,
        C: np.ndarray,
        T: float,
        *,
        N: int,
        min_sigma: float,
        max_sigma: float,
        weight_floor: float,
        mean_ratio_min: float,
        mean_ratio_max: float,
        variance_ratio: float,
        max_nfev: int,
        x0_override: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        K = np.asarray(K, float).ravel()
        C = np.asarray(C, float).ravel()
        T = float(T)

        K, C = _unique_strike_curve(K, C)

        lb, ub = _call_bounds(self.S0, K, T, self.r, self.q)
        C = np.clip(C, lb, ub)

        # light monotone repair on synthetic/observed input
        if K.size > 1 and np.all(np.diff(K) >= 0):
            C = np.minimum.accumulate(C)
            C = np.clip(C, lb, ub)

        F0 = self.S0 * np.exp((self.r - self.q) * T)

        # Parameters:
        #   logits: N-1
        #   log raw mean ratios: N
        #   log sigmas: N
        if x0_override is None:
            logits0 = np.zeros(N - 1, dtype=float)

            if N == 1:
                ratios0 = np.array([1.0], dtype=float)
                sigmas0 = np.array([0.35], dtype=float)
            else:
                ratios0 = np.exp(np.linspace(-0.10, 0.10, N))
                sigmas0 = np.linspace(0.20, 0.90, N)

            log_ratios0 = np.log(np.clip(ratios0, mean_ratio_min, mean_ratio_max))
            log_sigmas0 = np.log(np.clip(sigmas0, min_sigma, max_sigma))
            x0 = np.concatenate([logits0, log_ratios0, log_sigmas0])
        else:
            x0 = np.asarray(x0_override, float).ravel()

        lb_x = np.concatenate([
            np.full(N - 1, -20.0),
            np.full(N, np.log(mean_ratio_min)),
            np.full(N, np.log(min_sigma)),
        ])
        ub_x = np.concatenate([
            np.full(N - 1, 20.0),
            np.full(N, np.log(mean_ratio_max)),
            np.full(N, np.log(max_sigma)),
        ])
        x0 = np.clip(x0, lb_x + 1e-10, ub_x - 1e-10)

        scale = np.maximum(0.01 * self.S0, np.maximum(C, 1e-4 * self.S0))

        def unpack(x: np.ndarray):
            logits = x[: N - 1]
            log_ratios = x[N - 1 : N - 1 + N]
            log_sigmas = x[N - 1 + N :]

            weights = _weights_from_logits(logits, floor=weight_floor)
            raw_ratios = np.exp(log_ratios)
            mean_ratios = _normalize_mean_ratios(raw_ratios, weights, mean_ratio_min, mean_ratio_max)
            sigmas = np.exp(log_sigmas)

            # Stable label ordering.
            idx = np.lexsort((sigmas, mean_ratios))
            weights = weights[idx]
            mean_ratios = mean_ratios[idx]
            sigmas = sigmas[idx]

            return weights, mean_ratios, sigmas

        def residuals(x: np.ndarray) -> np.ndarray:
            weights, mean_ratios, sigmas = unpack(x)
            pred = _mixture_call_prices(K, T, self.r, F0, weights, mean_ratios, sigmas)

            res = (pred - C) / scale

            # Soft variance constraint:
            # min_j sigma_j^2 T >= variance_ratio * max_j sigma_j^2 T
            # This prevents tiny spike components.
            if N > 1 and variance_ratio > 0:
                tv = sigmas * sigmas * T
                mx = max(float(np.max(tv)), 1e-12)
                violation = max(0.0, float(variance_ratio) * mx - float(np.min(tv)))
                var_penalty = np.array([100.0 * violation / mx], dtype=float)
            else:
                var_penalty = np.array([], dtype=float)

            # Mild regularization.
            reg = 1e-4 * x

            return np.concatenate([res, var_penalty, reg])

        res = least_squares(
            residuals,
            x0,
            bounds=(lb_x, ub_x),
            max_nfev=int(max_nfev),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        weights, mean_ratios, sigmas = unpack(res.x)
        sse = float(np.sum(residuals(res.x) ** 2))

        return weights, mean_ratios, sigmas, sse

    def fit(
        self,
        K_obs,
        T_obs,
        C_obs,
        x0: Optional[Dict[str, float]] = None,
        bounds=None,
        max_nfev: int = 1000,
        **kwargs,
    ) -> FitResult:
        cfg = dict(x0 or {})

        N = int(cfg.get("n_components", cfg.get("N", 3)))
        if N < 1:
            raise ValueError("n_components must be >= 1.")

        min_sigma = float(cfg.get("min_sigma", 1e-4))
        max_sigma = float(cfg.get("max_sigma", 6.0))
        weight_floor = float(cfg.get("weight_floor", 1e-6))
        mean_ratio_min = float(cfg.get("mean_ratio_min", 0.25))
        mean_ratio_max = float(cfg.get("mean_ratio_max", 4.0))
        variance_ratio = float(cfg.get("variance_ratio", cfg.get("c", 0.05)))
        min_strikes = int(cfg.get("min_strikes_per_maturity", cfg.get("min_strikes", max(6, 3 * N))))
        verbose = bool(cfg.get("verbose", False))

        K, T, C = _clean_inputs(K_obs, T_obs, C_obs, self.S0, self.r, self.q)
        if K.size == 0:
            raise ValueError("No valid call quotes for LognormalMixtureModel.")

        kept_T = []
        weights_list = []
        ratios_list = []
        sigmas_list = []
        sse_list = []
        skipped = 0
        failures = []

        for tt in np.unique(T):
            idx = np.where(np.isclose(T, tt))[0]
            if np.unique(K[idx]).size < min_strikes:
                skipped += 1
                failures.append(f"T={float(tt):.8f}: too few unique strikes")
                continue

            try:
                w, mr, sig, sse = self._fit_one_curve(
                    K[idx],
                    C[idx],
                    float(tt),
                    N=N,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    weight_floor=weight_floor,
                    mean_ratio_min=mean_ratio_min,
                    mean_ratio_max=mean_ratio_max,
                    variance_ratio=variance_ratio,
                    max_nfev=max_nfev,
                )
                kept_T.append(float(tt))
                weights_list.append(w)
                ratios_list.append(mr)
                sigmas_list.append(sig)
                sse_list.append(sse)
            except Exception as e:
                skipped += 1
                failures.append(f"T={float(tt):.8f}: {type(e).__name__}: {e}")
                if verbose:
                    print(f"[lognormal_mixture] skipped T={float(tt):.8f}: {type(e).__name__}: {e}")

        if len(kept_T) == 0:
            msg = "No maturities could be fit with direct lognormal mixture."
            if failures:
                msg += "\nFirst failures:\n  - " + "\n  - ".join(failures[:10])
            raise ValueError(msg)

        order = np.argsort(kept_T)
        self.T_fit_ = np.asarray(kept_T, float)[order]
        self.weights_fit_ = np.asarray(weights_list, float)[order, :]
        self.mean_ratios_fit_ = np.asarray(ratios_list, float)[order, :]
        self.sigmas_fit_ = np.asarray(sigmas_list, float)[order, :]
        self.N_ = N

        self.params_ = LognormalMixtureParams(
            n_components=float(N),
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            weight_floor=weight_floor,
            mean_ratio_min=mean_ratio_min,
            mean_ratio_max=mean_ratio_max,
            variance_ratio=variance_ratio,
        )

        tv = self.sigmas_fit_ ** 2 * self.T_fit_[:, None]
        variance_ratio_min = float(np.min(np.min(tv, axis=1) / np.maximum(np.max(tv, axis=1), 1e-12)))

        fwd_err = np.abs(np.sum(self.weights_fit_ * self.mean_ratios_fit_, axis=1) - 1.0)

        self.fit_info_ = {
            "n_obs": float(K.size),
            "n_components": float(N),
            "n_maturities_kept": float(self.T_fit_.size),
            "n_maturities_skipped": float(skipped),
            "T_min": float(np.min(self.T_fit_)),
            "T_max": float(np.max(self.T_fit_)),
            "sse_median": float(np.median(sse_list)) if sse_list else np.nan,
            "variance_ratio_min": variance_ratio_min,
            "forward_ratio_error_max": float(np.max(fwd_err)) if fwd_err.size else np.nan,
        }

        if verbose:
            print("[lognormal_mixture_call_interp] fit_info:", self.fit_info_)
            print("[lognormal_mixture_call_interp] weights median:", np.median(self.weights_fit_, axis=0))
            print("[lognormal_mixture_call_interp] mean_ratios median:", np.median(self.mean_ratios_fit_, axis=0))
            print("[lognormal_mixture_call_interp] sigmas median:", np.median(self.sigmas_fit_, axis=0))

        return FitResult(params=self.params_.to_dict(), success=True, info=dict(self.fit_info_))

    def _observed_maturity_call_surface(self, K_grid: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        if self.T_fit_ is None:
            raise RuntimeError("Call fit(...) before pricing.")

        K_grid = np.asarray(K_grid, float).ravel()
        C_fit = np.zeros((self.T_fit_.size, K_grid.size), dtype=float)

        for i, tt in enumerate(self.T_fit_):
            F0 = self.S0 * np.exp((self.r - self.q) * float(tt))
            C_fit[i, :] = _mixture_call_prices(
                K_grid,
                float(tt),
                self.r,
                F0,
                self.weights_fit_[i],
                self.mean_ratios_fit_[i],
                self.sigmas_fit_[i],
            )

            lb, ub = _call_bounds(self.S0, K_grid, float(tt), self.r, self.q)
            C_fit[i, :] = np.clip(C_fit[i, :], lb, ub)

            if K_grid.size > 1 and np.all(np.diff(K_grid) >= 0):
                C_fit[i, :] = np.minimum.accumulate(C_fit[i, :])
                C_fit[i, :] = np.clip(C_fit[i, :], lb, ub)

        return C_fit

    def call_prices(self, K: np.ndarray, T: float, params: Dict[str, float], **kwargs) -> np.ndarray:
        K = np.asarray(K, float).ravel()
        C = self.price_surface(K, np.array([float(T)]), params, **kwargs)
        return np.asarray(C[0, :], float)

    def price_surface(self, K_grid: np.ndarray, T_grid: np.ndarray, params: Dict[str, float], **kwargs) -> np.ndarray:
        if self.T_fit_ is None:
            raise RuntimeError("Call fit(...) before price_surface(...).")

        params = dict(params or {})
        K_grid = np.asarray(K_grid, float).ravel()
        T_grid = np.asarray(T_grid, float).ravel()
        T_grid = np.maximum(T_grid, 1e-10)

        N = int(params.get("n_components", self.N_))
        min_sigma = float(params.get("min_sigma", self.params_.min_sigma if self.params_ else 1e-4))
        max_sigma = float(params.get("max_sigma", self.params_.max_sigma if self.params_ else 6.0))
        weight_floor = float(params.get("weight_floor", self.params_.weight_floor if self.params_ else 1e-6))
        mean_ratio_min = float(params.get("mean_ratio_min", self.params_.mean_ratio_min if self.params_ else 0.25))
        mean_ratio_max = float(params.get("mean_ratio_max", self.params_.mean_ratio_max if self.params_ else 4.0))
        variance_ratio = float(params.get("variance_ratio", self.params_.variance_ratio if self.params_ else 0.05))
        max_nfev = int(kwargs.get("max_nfev", 1000))

        if N != self.N_:
            raise ValueError("Changing n_components at price time is not supported. Refit the model.")

        # 1. Evaluate fitted mixture call curves at observed maturities.
        C_obsT = self._observed_maturity_call_surface(K_grid, params)

        # 2. Interpolate/extrapolate CALL PRICES across maturity.
        C_synth = _linear_interp_extrap_T(self.T_fit_, C_obsT, T_grid)

        # 3. Fit a fresh mixture to each synthetic requested-maturity call curve.
        out = np.zeros_like(C_synth)

        for i, tt in enumerate(T_grid):
            lb, ub = _call_bounds(self.S0, K_grid, float(tt), self.r, self.q)
            target = np.clip(C_synth[i, :], lb, ub)

            if K_grid.size > 1 and np.all(np.diff(K_grid) >= 0):
                target = np.minimum.accumulate(target)
                target = np.clip(target, lb, ub)

            try:
                w, mr, sig, _ = self._fit_one_curve(
                    K_grid,
                    target,
                    float(tt),
                    N=N,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    weight_floor=weight_floor,
                    mean_ratio_min=mean_ratio_min,
                    mean_ratio_max=mean_ratio_max,
                    variance_ratio=variance_ratio,
                    max_nfev=max_nfev,
                )

                F0 = self.S0 * np.exp((self.r - self.q) * float(tt))
                out[i, :] = _mixture_call_prices(K_grid, float(tt), self.r, F0, w, mr, sig)
            except Exception:
                # If the refit fails, return the interpolated call curve rather than crashing.
                out[i, :] = target

            out[i, :] = np.clip(out[i, :], lb, ub)
            if K_grid.size > 1 and np.all(np.diff(K_grid) >= 0):
                out[i, :] = np.minimum.accumulate(out[i, :])
                out[i, :] = np.clip(out[i, :], lb, ub)

        return out
