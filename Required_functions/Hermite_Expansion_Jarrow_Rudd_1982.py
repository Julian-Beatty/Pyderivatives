# ============================================================
# Hermite_Expansion_Jarrow_Rudd.py  (REVISED)
#
# Generalized Jarrow–Rudd / Gram–Charlier (probabilists' Hermite) expansion
# allowing user-chosen order in {1,2,3,4,5}.
#
# - Fits a single-maturity call cross-section by least squares on call prices
#   + penalties (mass=1, forward/martingale, negative-density area).
#
# - Provides:
#     jr.callhat(K)                  : fitted calls on arbitrary strike grid
#     jr.qhat(s_like, method=...)    : fitted density on arbitrary grid
#     jr.rnd_grid()                  : internal (s_grid, q_grid) used in pricing
#
# - Includes:
#     HermiteRNDRowModel adapter for your surface pipeline
#     (chat, qhat) like GenGammaRNDModel / GLDRNDModel.
#
# Notes:
# - "order" here means the highest Hermite term H_n used in the correction:
#     f_Z(z) = phi(z) * [1 + sum_{n=3..order} c_n/n! * H_n(z)]
#   For order <= 2, this reduces to the lognormal (Normal Z) with no correction.
#
# - As with any Gram–Charlier / Hermite expansion, q(s) can go negative.
#   Use w_neg + tight coefficient bounds for stability, especially at order=5.
# ============================================================

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Sequence

import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm

# Only needed for the standalone demo that generates synthetic Heston calls:
from Heston_model import *  # noqa: F401,F403 (assumes HestonParams, heston_call_prices_fast, etc.)


# -----------------------------
# Helpers: Black–Scholes calls
# -----------------------------

def bs_call(S0: float, K: np.ndarray, r: float, q: float, T: float, sigma: float) -> np.ndarray:
    K = np.asarray(K, float)
    if T <= 0:
        return np.maximum(S0 - K, 0.0)
    vol = max(float(sigma), 1e-12)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT
    return np.exp(-q * T) * S0 * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)


# ------------------------------------------
# Probabilists' Hermite polynomials H_n(z)
# ------------------------------------------

def hermite_prob(n: int, z: np.ndarray) -> np.ndarray:
    """
    Probabilists' Hermite polynomial H_n(z) via recurrence:
      H_0 = 1
      H_1 = z
      H_n = z H_{n-1} - (n-1) H_{n-2}
    """
    z = np.asarray(z, float)
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return np.ones_like(z)
    if n == 1:
        return z

    Hnm2 = np.ones_like(z)  # H0
    Hnm1 = z                # H1
    for k in range(2, n + 1):
        Hn = z * Hnm1 - (k - 1) * Hnm2
        Hnm2, Hnm1 = Hnm1, Hn
    return Hnm1


# ------------------------------------------
# Generalized JR / Gram–Charlier density q_T(s)
# ------------------------------------------

def jr_rnd_ST_N(
    s: np.ndarray,
    S0: float, r: float, q: float, T: float,
    sigma: float,
    coeffs: np.ndarray,   # [c3, c4, ..., c_order] length = max(order-2, 0)
    order: int
) -> np.ndarray:
    """
    Risk-neutral density of S_T via Gram–Charlier/Hermite expansion on log-returns.

    Let X = ln(S_T/S0). Under Q we set:
      X = mu + sigma*sqrt(T)*Z
      mu = (r - q - 0.5*sigma^2)*T

    with standardized Z having density:
      f_Z(z) = phi(z) * [ 1 + sum_{n=3..order} c_n/n! * H_n(z) ]

    Transform s = S0*exp(X):
      f_S(s) = f_X(x) * (1/s),  f_X(x) = (1/(sigma*sqrt(T))) * f_Z(z)

    For order <= 2, the correction sum is empty -> pure lognormal (Normal Z).
    """
    s = np.asarray(s, float)
    s_safe = np.maximum(s, 1e-300)

    order = int(order)
    if order < 1 or order > 5:
        raise ValueError("order must be in {1,2,3,4,5}")

    vol = max(float(sigma), 1e-12)
    sqrtT = np.sqrt(T)
    mu = (r - q - 0.5 * vol * vol) * T

    x = np.log(s_safe / S0)
    z = (x - mu) / (vol * sqrtT)

    phi = np.exp(-0.5 * z*z) / np.sqrt(2*np.pi)

    adj = np.ones_like(z)
    if order >= 3:
        coeffs = np.asarray(coeffs, float)
        need = order - 2
        if coeffs.size != need:
            raise ValueError(f"coeffs must have length {need} for order={order} (c3..c{order})")
        for j, n in enumerate(range(3, order + 1)):
            cn = float(coeffs[j])
            adj = adj + (cn / float(math.factorial(n))) * hermite_prob(n, z)

    fX = (1.0 / (vol * sqrtT)) * phi * adj
    fS = fX / s_safe
    return fS


def jr_call_from_rnd(
    K: np.ndarray,
    s_grid: np.ndarray,
    q_grid: np.ndarray,
    r: float, T: float
) -> np.ndarray:
    """
    Price calls by numerical integration:
      C(K) = e^{-rT} ∫ (s-K)^+ q(s) ds
    """
    K = np.asarray(K, float)
    s = np.asarray(s_grid, float)
    q = np.asarray(q_grid, float)
    disc = np.exp(-r * T)

    C = np.empty_like(K, dtype=float)
    for i, k in enumerate(K):
        payoff = np.maximum(s - k, 0.0)
        C[i] = disc * np.trapz(payoff * q, s)
    return C


# -----------------------------
# Model container + calibration
# -----------------------------

@dataclass
class JRFit:
    order: int
    sigma: float
    coeffs: np.ndarray  # c3..c_order (length max(order-2,0))
    success: bool
    cost: float
    message: str


class JarrowRuddHermiteN:
    """
    Calibrate (sigma, c3..c_order) to a single-maturity call cross-section.

    Methods:
      - fit(K_obs, C_obs, ...)
      - callhat(K)
      - qhat(s_like, method="direct"|"bl")
      - rnd_grid()
      - plot_diagnostics()

    Notes:
      - Gram–Charlier / Hermite expansions can yield negative densities.
      - We stabilize with penalties on:
          * total mass: ∫ q ds ≈ 1
          * forward:   ∫ s q ds ≈ S0*exp((r-q)T)
          * neg area:  ∫ (-min(q,0)) ds ≈ 0
    """

    def __init__(
        self,
        S0: float, r: float, q: float, T: float,
        order: int = 4,
        s_grid_mult: Tuple[float, float] = (0.25, 2.75),
        s_grid_size: int = 6000,
        w_neg: float = 2e4,
        w_mass: float = 5e3,
        w_forward: float = 5e3,
        tiny: float = 1e-14
    ):
        order = int(order)
        if order < 1 or order > 5:
            raise ValueError("order must be in {1,2,3,4,5}")

        self.S0, self.r, self.q, self.T = float(S0), float(r), float(q), float(T)
        self.order = order

        self.s_grid_mult = s_grid_mult
        self.s_grid_size = int(s_grid_size)
        self.w_neg = float(w_neg)
        self.w_mass = float(w_mass)
        self.w_forward = float(w_forward)
        self.tiny = float(tiny)

        self.fit_: Optional[JRFit] = None
        self.s_grid_: Optional[np.ndarray] = None
        self.q_grid_: Optional[np.ndarray] = None

    def _make_s_grid(self, K_obs: np.ndarray) -> np.ndarray:
        lowK, highK = float(np.min(K_obs)), float(np.max(K_obs))
        s_min = max(self.tiny, self.s_grid_mult[0] * lowK)
        s_max = self.s_grid_mult[1] * highK
        # log-spaced grid helps tail integration stability
        return np.exp(np.linspace(np.log(s_min), np.log(s_max), self.s_grid_size))

    def _penalties(self, s: np.ndarray, q: np.ndarray) -> np.ndarray:
        mass = np.trapz(q, s)
        fwd_target = self.S0 * np.exp((self.r - self.q) * self.T)
        fwd = np.trapz(s * q, s)

        neg = np.minimum(q, 0.0)
        neg_area = np.trapz(-neg, s)

        return np.array([
            np.sqrt(self.w_mass) * (mass - 1.0),
            np.sqrt(self.w_forward) * (fwd - fwd_target) / max(fwd_target, 1.0),
            np.sqrt(self.w_neg) * neg_area
        ], dtype=float)

    def fit(
        self,
        K_obs: np.ndarray,
        C_obs: np.ndarray,
        sigma0: float = 0.25,
        c0: Optional[Sequence[float]] = None,  # initial c3..c_order
        sigma_bounds: Tuple[float, float] = (1e-4, 3.0),
        c_bounds: Tuple[float, float] = (-1.5, 1.5),
        weights: Optional[np.ndarray] = None,
        max_nfev: int = 4000
    ) -> JRFit:
        K_obs = np.asarray(K_obs, float)
        C_obs = np.asarray(C_obs, float)
        assert K_obs.shape == C_obs.shape

        s_grid = self._make_s_grid(K_obs)

        # weights
        if weights is None:
            w = 1.0 / np.maximum(C_obs, 1e-3)
            w = w / np.mean(w)
        else:
            w = np.asarray(weights, float)

        # parameter vector x = [sigma, c3..c_order]
        n_c = max(self.order - 2, 0)
        if c0 is None:
            c0v = np.zeros(n_c, dtype=float)
        else:
            c0v = np.asarray(list(c0), float).ravel()
            if c0v.size < n_c:
                c0v = np.pad(c0v, (0, n_c - c0v.size))
            else:
                c0v = c0v[:n_c]

        x0 = np.concatenate([[float(sigma0)], c0v])

        lb = np.concatenate([[float(sigma_bounds[0])], np.full(n_c, float(c_bounds[0]))])
        ub = np.concatenate([[float(sigma_bounds[1])], np.full(n_c, float(c_bounds[1]))])

        def residuals(x: np.ndarray) -> np.ndarray:
            sigma = float(x[0])
            coeffs = np.asarray(x[1:], float)

            qg = jr_rnd_ST_N(s_grid, self.S0, self.r, self.q, self.T, sigma, coeffs, self.order)
            C_hat = jr_call_from_rnd(K_obs, s_grid, qg, self.r, self.T)

            res_price = np.sqrt(w) * (C_hat - C_obs)
            res_pen = self._penalties(s_grid, qg)
            return np.concatenate([res_price, res_pen])

        sol = least_squares(
            residuals,
            x0=x0,
            bounds=(lb, ub),
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=int(max_nfev),
        )

        sigma_hat = float(sol.x[0])
        coeffs_hat = np.asarray(sol.x[1:], float).copy()

        qg_hat = jr_rnd_ST_N(s_grid, self.S0, self.r, self.q, self.T, sigma_hat, coeffs_hat, self.order)

        self.s_grid_ = s_grid
        self.q_grid_ = qg_hat
        self.fit_ = JRFit(
            order=self.order,
            sigma=sigma_hat,
            coeffs=coeffs_hat,
            success=bool(sol.success),
            cost=float(sol.cost),
            message=str(sol.message),
        )
        return self.fit_

    def callhat(self, K: np.ndarray) -> np.ndarray:
        assert self.fit_ is not None and self.s_grid_ is not None and self.q_grid_ is not None, "fit() first"
        return jr_call_from_rnd(np.asarray(K, float), self.s_grid_, self.q_grid_, self.r, self.T)

    def rnd_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.fit_ is not None and self.s_grid_ is not None and self.q_grid_ is not None, "fit() first"
        return self.s_grid_, self.q_grid_

    def qhat(self, s_like: np.ndarray, method: Literal["direct", "bl"] = "direct") -> np.ndarray:
        """
        Evaluate density on a user grid.

        method="direct": model density q_T(s) at s_like.
        method="bl"    : q(K)=exp(rT) d^2C/dK^2 computed from fitted calls on s_like grid.
        """
        assert self.fit_ is not None, "fit() first"
        s_like = np.asarray(s_like, float)

        if method == "direct":
            return jr_rnd_ST_N(
                s_like, self.S0, self.r, self.q, self.T,
                self.fit_.sigma, self.fit_.coeffs, self.fit_.order
            )

        if method == "bl":
            C = self.callhat(s_like)
            dC = np.gradient(C, s_like, edge_order=2)
            d2C = np.gradient(dC, s_like, edge_order=2)
            return np.exp(self.r * self.T) * d2C

        raise ValueError("method must be 'direct' or 'bl'")

    def plot_diagnostics(self):
        s, q = self.rnd_grid()
        mass = np.trapz(q, s)
        fwd = np.trapz(s * q, s)
        fwd_target = self.S0 * np.exp((self.r - self.q) * self.T)
        neg_area = np.trapz(-np.minimum(q, 0.0), s)

        print("---- RND diagnostics ----")
        print(f"order                 : {self.order}")
        print(f"Mass ∫q ds             : {mass:.6f}  (target 1)")
        print(f"Forward ∫s q ds        : {fwd:.6f}  (target {fwd_target:.6f})")
        print(f"Neg area ∫(-min(q,0))ds: {neg_area:.6e}")


# ============================================================
# ✅ Row-model wrapper for your surface pipeline
# ============================================================

class HermiteRNDRowModel:
    """
    Adapter class used by CallSurfaceEstimator when cfg.model == "hermite".

    Expected methods:
      - chat(K_eval): returns fitted call prices on K_eval
      - qhat(s_eval): returns fitted density on s_eval (defaults to direct)

    It fits internally at construction time (so it behaves like GenGammaRNDModel / GLDRNDModel).
    """

    def __init__(
        self,
        S0: float,
        r: float,
        q: float,
        T: float,
        K: np.ndarray,
        C_mkt: np.ndarray,
        order: int = 4,
        sigma0: float = 0.25,
        c0: Optional[Sequence[float]] = None,
        sigma_bounds: Tuple[float, float] = (1e-4, 3.0),
        c_bounds: Tuple[float, float] = (-1.5, 1.5),
        s_grid_mult: Tuple[float, float] = (0.15, 4.0),
        s_grid_size: int = 9000,
        w_neg: float = 3e4,
        w_mass: float = 8e3,
        w_forward: float = 8e3,
        weights: Optional[np.ndarray] = None,
        max_nfev: int = 4000,
    ):
        self._jr = JarrowRuddHermiteN(
            S0=S0, r=r, q=q, T=T,
            order=int(order),
            s_grid_mult=s_grid_mult,
            s_grid_size=s_grid_size,
            w_neg=w_neg,
            w_mass=w_mass,
            w_forward=w_forward,
        )
        self._jr.fit(
            K_obs=np.asarray(K, float),
            C_obs=np.asarray(C_mkt, float),
            sigma0=float(sigma0),
            c0=c0,
            sigma_bounds=sigma_bounds,
            c_bounds=c_bounds,
            weights=weights,
            max_nfev=max_nfev,
        )

    def chat(self, K_eval: np.ndarray) -> np.ndarray:
        return self._jr.callhat(K_eval)

    def qhat(self, s_eval: np.ndarray, method: Literal["direct", "bl"] = "direct") -> np.ndarray:
        return self._jr.qhat(s_eval, method=method)


# ============================================================
# Demo: Extrapolation beyond observed strike range
# ============================================================

def make_extrapolation_strikes(
    strikes_obs: np.ndarray,
    low_mult: float = 0.6,
    high_mult: float = 1.6,
    n: int = 120
) -> np.ndarray:
    strikes_obs = np.asarray(strikes_obs, float)
    K_min = float(np.min(strikes_obs))
    K_max = float(np.max(strikes_obs))
    return np.linspace(low_mult * K_min, high_mult * K_max, int(n))


def demo_jarrow_rudd_calibration(seed: int = 0, order: int = 5):
    rng = np.random.default_rng(seed)

    # synthetic "market" from Heston
    true_p = HestonParams(kappa=0.5, theta=0.05, sigma=0.25, v0=0.02, rho=0.9)
    S0, r, q_div, T = 60.0, 0.02, 0.00, 0.15

    strikes_obs = np.linspace(30, 140, 150)
    C_mkt = heston_call_prices_fast(S0, strikes_obs, r, q_div, T, true_p)

    # "true" BL density on observed strike grid (for reference only)
    dC_dK = np.gradient(C_mkt, strikes_obs, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, strikes_obs, edge_order=2)
    q_true = np.exp(r * T) * d2C_dK2

    # Fit generalized Hermite order
    jr = JarrowRuddHermiteN(
        S0=S0, r=r, q=q_div, T=T,
        order=int(order),
        s_grid_mult=(0.10, 4.00),
        s_grid_size=9000,
        w_neg=3e4,
        w_mass=8e3,
        w_forward=8e3
    )

    # Initial coeff guesses for c3..c_order
    n_c = max(int(order) - 2, 0)
    c0 = np.zeros(n_c)
    if n_c >= 1:
        c0[0] = 0.5  # mild c3 start (optional)
    if n_c >= 2:
        c0[1] = 0.2  # mild c4 start
    if n_c >= 3:
        c0[2] = 0.0  # c5 start

    fit = jr.fit(
        K_obs=strikes_obs,
        C_obs=C_mkt,
        sigma0=0.25,
        c0=c0,
        sigma_bounds=(1e-4, 3.0),
        # IMPORTANT: for order=5 keep bounds tighter to reduce explosions
        c_bounds=(-1.5, 1.5),
        weights=None,
        max_nfev=4000
    )

    strikes_ext = make_extrapolation_strikes(strikes_obs, low_mult=0.6, high_mult=1.6, n=400)

    C_hat_obs = jr.callhat(strikes_obs)
    C_hat_ext = jr.callhat(strikes_ext)

    q_hat_direct = jr.qhat(strikes_obs, method="direct")
    q_hat_bl = jr.qhat(strikes_obs, method="bl")

    s_grid, q_grid = jr.rnd_grid()

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)

    ax = axes[0]
    ax.plot(strikes_obs, q_true, "k--", lw=2, label="True (BL on observed calls)")
    ax.plot(strikes_obs, q_hat_direct, "-", lw=2, label=f"Fitted qhat direct (order={order})")
    ax.plot(strikes_obs, q_hat_bl, ":", lw=2, label="Fitted qhat (BL from fitted calls)")
    ax.set_title("Risk-Neutral Density (on observed grid)")
    ax.set_xlabel("Terminal price s (using K grid)")
    ax.set_ylabel("q(s)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(strikes_obs, C_mkt, "o", ms=4, label="Observed calls")
    ax2.plot(strikes_obs, C_hat_obs, "-", lw=2, label="Fitted calls (Hermite)")
    ax2.set_title("Call Curve Fit (Observed Strike Range)")
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Call price C(K)")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.plot(strikes_ext, C_hat_ext, "-", lw=2, label="Model-implied calls (extrapolated)")
    ax3.axvline(float(strikes_obs.min()), color="k", ls="--", alpha=0.6)
    ax3.axvline(float(strikes_obs.max()), color="k", ls="--", alpha=0.6)
    ax3.set_title("Call Price Extrapolation Beyond Observed Strikes")
    ax3.set_xlabel("Strike K")
    ax3.set_ylabel("Call price C(K)")
    ax3.legend(loc="best")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(s_grid, q_grid, lw=2)
    plt.title("Model-implied q_T(s) on s-grid (used for pricing integrals)")
    plt.xlabel("Terminal price s")
    plt.ylabel("q_T(s)")
    plt.grid(alpha=0.3)
    plt.show()

    print("---- Fit ----")
    print(fit)
    jr.plot_diagnostics()


if __name__ == "__main__":
    demo_jarrow_rudd_calibration(seed=0, order=5)
