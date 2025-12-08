import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from scipy.special import hyp1f1, gamma
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from Heston_model import HestonParams, heston_call_prices_fast


# ============================================================
# 1. Data structures
# ============================================================

@dataclass
class DFCHParams:
    """
    Parameterization of the DFCH call price functional.

    Free parameters (7):
        a1, a2, a3, b2, b3, b4, m1

    Derived via constraints (Bu & Hadri, 2007, eq. (17)-(21)):
        a5 = -1/2
        a6 =  1/2
        b1 = 1 + a2 * b3
        m2  from martingale constraint E[S_T] = F_T
        a4, c1, c2 from eqs. (18)-(20)
    """
    a1: float
    a2: float
    a3: float
    b2: float
    b3: float
    b4: float
    m1: float

    # derived parameters
    a4: float
    a5: float
    a6: float
    b1: float
    m2: float
    c1: float
    c2: float


@dataclass
class DFCHFitResult:
    params: DFCHParams
    success: bool
    message: str
    cost: float
    nfev: int
    call_price_func: Callable[[np.ndarray], np.ndarray]
    rnd_func: Callable[[np.ndarray], np.ndarray]


# ============================================================
# 2. Helpers: parameter transforms + constraints
# ============================================================

def _unpack_free_theta(theta: np.ndarray,
                       S0: float,
                       r: float,
                       q: float,
                       T: float) -> DFCHParams:
    """
    Map unconstrained theta in R^7 to valid DFCH parameters
    and enforce constraints (17)-(21) in Bu & Hadri (2007).

    theta (free) = (u1,...,u7):
        a1  = u1                  (any real, but scale small)
        a2  = exp(u2)             > 0
        a3  = a2 + exp(u3)        > a2
        b2  = -exp(u4)            < 0
        b3  = exp(u5)             > 0
        b4  = -exp(u6)            < 0
        m1  = F_T + u7            (center around forward, not spot)
    """
    u1, u2, u3, u4, u5, u6, u7 = theta

    a1 = u1                       # you can also do 1e-3*u1 if you want to shrink it
    a2 = np.exp(u2)
    a3 = a2 + np.exp(u3)
    b2 = -np.exp(u4)
    b3 = np.exp(u5)
    b4 = -np.exp(u6)

    # forward under Q
    F_T = S0 * np.exp((r - q) * T)
    m1 = F_T + u7

    # fixed / constrained parameters
    a5 = -0.5
    a6 = 0.5
    b1 = 1.0 + a2 * b3

    # martingale constraint to pin down m2 (eq. 21)
    # E(z) = e^{rT} a1 Γ(a3)/Γ(a3-a2) (-b2)^(-a2) (m1 - m2) + m2
    # => m2 = (F_T - A m1)/(1 - A)
    try:
        A_num = np.exp(r * T) * a1 * gamma(a3)
        A_den = gamma(a3 - a2) * ((-b2) ** a2)
        A = A_num / A_den
    except Exception:
        A = 0.0

    if np.isclose(1.0 - A, 0.0):
        A = A - 1e-6

    m2 = (F_T - A * m1) / (1.0 - A)

    # a4 from eq. (20)
    # a4 = 1/(2 sqrt(-b4 π)) [ e^{-rT} - a1 (-b2)^(-a2) Γ(a3)/Γ(a3 - a2) ]
    try:
        inner_term = np.exp(-r * T) - a1 * ((-b2) ** (-a2)) * gamma(a3) / gamma(a3 - a2)
        a4 = 0.5 * (1.0 / np.sqrt(-b4 * np.pi)) * inner_term
    except Exception:
        a4 = 0.0

    # c2 from eq. (19): c2 = -e^{-rT} + a4 sqrt(-b4 π)
    c2 = -np.exp(-r * T) + a4 * np.sqrt(-b4 * np.pi)

    # c1 from eq. (18): c1 = -c2 m2
    c1 = -c2 * m2

    return DFCHParams(
        a1=a1, a2=a2, a3=a3, b2=b2, b3=b3, b4=b4, m1=m1,
        a4=a4, a5=a5, a6=a6, b1=b1, m2=m2, c1=c1, c2=c2
    )


# ============================================================
# 3. Call price functional C(K)
# ============================================================

def dfch_call_price(K: np.ndarray,
                    params: DFCHParams) -> np.ndarray:
    """
    C(K) = c1 + c2 K
           + 1_{K>m1} a1 (K - m1)^b1 1F1(a2; a3; b2 (K-m1)^b3)
           + a4 1F1(a5; a6; b4 (K - m2)^2)
    """
    K = np.asarray(K, float)

    a1 = params.a1
    a2 = params.a2
    a3 = params.a3
    b2 = params.b2
    b3 = params.b3
    b4 = params.b4
    m1 = params.m1
    m2 = params.m2
    a4 = params.a4
    a5 = params.a5
    a6 = params.a6
    b1 = params.b1
    c1 = params.c1
    c2 = params.c2

    C = c1 + c2 * K

    # asymmetric branch
    mask = K > m1
    Km1 = np.clip(K - m1, 0.0, None)
    z1 = b2 * (Km1 ** b3)
    with np.errstate(over='ignore', invalid='ignore'):
        term1 = np.zeros_like(K)
        term1[mask] = a1 * (Km1[mask] ** b1) * hyp1f1(a2, a3, z1[mask])

    # symmetric branch
    z2 = b4 * (K - m2) ** 2
    with np.errstate(over='ignore', invalid='ignore'):
        term2 = a4 * hyp1f1(a5, a6, z2)

    return C + term1 + term2


# ============================================================
# 4. RND via Breeden–Litzenberger
# ============================================================

def dfch_rnd(K_grid: np.ndarray,
             params: DFCHParams,
             r: float,
             T: float,
             h: Optional[float] = None) -> np.ndarray:
    """
    f(K) = e^{rT} * ∂^2 C(K) / ∂K^2 via centered finite differences.
    """
    K_grid = np.asarray(K_grid, float)
    if h is None:
        h = 1e-3 * np.median(K_grid)

    K_plus = K_grid + h
    K_minus = K_grid - h

    C_plus = dfch_call_price(K_plus, params)
    C_mid = dfch_call_price(K_grid, params)
    C_minus = dfch_call_price(K_minus, params)

    second_deriv = (C_plus - 2.0 * C_mid + C_minus) / (h ** 2)
    f = np.exp(r * T) * second_deriv
    return np.maximum(f, 0.0)


# ============================================================
# 5. Nonlinear LS fitting (single-start and multi-start)
# ============================================================

def _residuals_theta(theta: np.ndarray,
                     K: np.ndarray,
                     C_mkt: np.ndarray,
                     S0: float,
                     r: float,
                     q: float,
                     T: float,
                     weights: Optional[np.ndarray]) -> np.ndarray:
    params = _unpack_free_theta(theta, S0, r, q, T)
    C_model = dfch_call_price(K, params)
    res = C_model - C_mkt
    if weights is not None:
        res = res * np.sqrt(weights)
    return np.where(np.isfinite(res), res, 1e6)


def fit_dfch(K: np.ndarray,
             C_mkt: np.ndarray,
             S0: float,
             r: float,
             q: float,
             T: float,
             weights: Optional[np.ndarray] = None,
             theta0: Optional[np.ndarray] = None,
             verbose: int = 0) -> DFCHFitResult:
    """
    Single-start DFCH calibration (used inside multistart).
    """
    K = np.asarray(K, float)
    C_mkt = np.asarray(C_mkt, float)

    if weights is None:
        weights = np.ones_like(K)
    else:
        weights = np.asarray(weights, float)

    if theta0 is None:
        # heuristic initial guess
        u1 = 1e-4
        u2 = np.log(1.5)
        u3 = np.log(0.8)
        u4 = np.log(5.0)   # b2 ~ -5
        u5 = np.log(1.0)   # b3 ~ 1
        u6 = np.log(5.0)   # b4 ~ -5
        u7 = 0.0           # m1 ~ F_T
        theta0 = np.array([u1, u2, u3, u4, u5, u6, u7], float)

    res_fun = lambda th: _residuals_theta(th, K, C_mkt, S0, r, q, T, weights)

    opt = least_squares(
        res_fun,
        theta0,
        method="trf",
        jac="2-point",
        max_nfev=4000,
        verbose=verbose
    )

    params_hat = _unpack_free_theta(opt.x, S0, r, q, T)

    def call_price_func(K_eval: np.ndarray) -> np.ndarray:
        return dfch_call_price(K_eval, params_hat)

    def rnd_func(K_eval: np.ndarray) -> np.ndarray:
        return dfch_rnd(K_eval, params_hat, r=r, T=T)

    return DFCHFitResult(
        params=params_hat,
        success=opt.success,
        message=opt.message,
        cost=opt.cost,
        nfev=opt.nfev,
        call_price_func=call_price_func,
        rnd_func=rnd_func
    )


def fit_dfch_multistart(K: np.ndarray,
                        C_mkt: np.ndarray,
                        S0: float,
                        r: float,
                        q: float,
                        T: float,
                        weights: Optional[np.ndarray] = None,
                        n_starts: int = 10,
                        seed: int = 0,
                        verbose_each: int = 0) -> DFCHFitResult:
    """
    Simple multi-start wrapper: run fit_dfch n_starts times with random
    perturbations of theta0 and keep the best.
    """
    rng = np.random.default_rng(seed)

    best_res: Optional[DFCHFitResult] = None

    for j in range(n_starts):
        base_theta0 = np.array([1e-4, np.log(1.5), np.log(0.8),
                                np.log(5.0), np.log(1.0), np.log(5.0), 0.0])
        # add Gaussian noise to initial guess
        theta0 = base_theta0 + rng.normal(scale=0.5, size=7)

        res = fit_dfch(K, C_mkt, S0, r, q, T,
                       weights=weights,
                       theta0=theta0,
                       verbose=verbose_each)

        if (best_res is None) or (res.cost < best_res.cost):
            best_res = res

    return best_res


# ============================================================
# 6. Demo vs Heston
# ============================================================

if __name__ == "__main__":
    true_p = HestonParams(kappa=0.5, theta=0.05, sigma=0.25, v0=0.02, rho=-0.9)
    S0, r, q, T = 120.0, 0.02, 0.00, 0.5

    strikes = np.linspace(50, 200, 100)
    C_mkt = heston_call_prices_fast(S0, strikes, r, q, T, true_p)

    # (optional) very simple weights: emphasize near-ATM
    F_T = S0 * np.exp((r - q) * T)
    weights = 1.0 / (1.0 + ((strikes / F_T - 1.0) ** 2))

    fit = fit_dfch_multistart(strikes, C_mkt, S0, r, q, T,
                              weights=weights,
                              n_starts=15,
                              seed=42,
                              verbose_each=0)

    print("Success:", fit.success)
    print("Message:", fit.message)
    print("Cost:", fit.cost)
    print("Estimated m1, m2:", fit.params.m1, fit.params.m2)

    C_fit = fit.call_price_func(strikes)

    # Heston RND via BL for comparison
    dC_dK = np.gradient(C_mkt, strikes, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, strikes, edge_order=2)
    q_true = np.exp(r * T) * d2C_dK2

    q_hyp = fit.rnd_func(strikes)

    # --- Call prices ---
    plt.figure(figsize=(8, 5))
    plt.plot(strikes, C_fit, label="hypergeometric")
    plt.plot(strikes, C_mkt, label="heston")
    plt.xlabel("Strike K")
    plt.ylabel("Call price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Densities ---
    plt.figure(figsize=(8, 5))
    plt.plot(strikes, q_hyp, label="DFCH RND")
    plt.plot(strikes, q_true, "--", label="Heston RND")
    plt.xlabel("Terminal price s = K")
    plt.ylabel("q(s)")
    plt.legend()
    plt.tight_layout()
    plt.show()
