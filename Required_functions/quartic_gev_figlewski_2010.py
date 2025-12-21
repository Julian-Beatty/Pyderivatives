import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Iterable
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm
from Heston_model import*
# ============================================================
# Heston (closed-form via single integrals)
# ============================================================
@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma: float
    v0: float
    rho: float

def _heston_char(u: complex, T: float, S0: float, r: float, q: float, p: HestonParams) -> complex:
    iu = 1j * u
    a = p.kappa * p.theta
    b = p.kappa
    d = np.sqrt((p.rho * p.sigma * iu - b)**2 + p.sigma**2 * (iu + u*u))
    g = (b - p.rho * p.sigma * iu - d) / (b - p.rho * p.sigma * iu + d)
    exp_neg_dT = np.exp(-d * T)
    lnG = np.log1p(-g * exp_neg_dT) - np.log1p(-g)
    C = iu * (r - q) * T + (a / p.sigma**2) * ((b - p.rho * p.sigma * iu - d) * T - 2.0 * lnG)
    D = ((b - p.rho * p.sigma * iu - d) / p.sigma**2) * ((1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT))
    return np.exp(C + D * p.v0 + iu * np.log(S0))

def _Pj(j: int, S0: float, K: float, r: float, q: float, T: float, p: HestonParams) -> float:
    if K <= 0.0:
        return 1.0
    lnK = np.log(K)
    def integrand(u: float) -> float:
        if j == 1:
            phi = _heston_char(u - 1j, T, S0, r, q, p) / (S0 * np.exp((r - q) * T))
        else:
            phi = _heston_char(u, T, S0, r, q, p)
        return np.real(np.exp(-1j * u * lnK) * phi / (1j * u))
    val, _ = quad(integrand, 0.0, np.inf, limit=200, epsabs=1e-9, epsrel=1e-8)
    return 0.5 + (1.0 / np.pi) * val

def heston_call_price(S0: float, K: float, r: float, q: float, T: float, p: HestonParams) -> float:
    return S0 * np.exp(-q * T) * _Pj(1, S0, K, r, q, T, p) - K * np.exp(-r * T) * _Pj(2, S0, K, r, q, T, p)

def heston_call_prices(S0: float, K: Iterable[float], r: float, q: float, T: float, p: HestonParams) -> np.ndarray:
    K = np.asarray(list(K), dtype=float)
    return np.array([heston_call_price(S0, k, r, q, T, p) for k in K], dtype=float)

# ============================================================
# Black–Scholes discounted call and IV inversion
# ============================================================
def bs_call_disc(S0, K, r, q, T, sigma):
    if sigma <= 0:
        return max(S0*np.exp(-q*T) - K*np.exp(-r*T), 0.0)
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_iv_from_price_disc(S0, K, r, q, T, price, lo=1e-6, hi=5.0):
    lower = max(S0*np.exp(-q*T) - K*np.exp(-r*T), 0.0)
    upper = S0*np.exp(-q*T)
    target = float(np.clip(price, lower, upper))
    f = lambda vol: bs_call_disc(S0, K, r, q, T, vol) - target
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        hi_try = hi
        for _ in range(10):
            hi_try *= 2.0
            if f(lo) * f(hi_try) <= 0:
                hi = hi_try
                break
        if f(lo) * f(hi) > 0:
            return np.nan
    try:
        return brentq(f, lo, hi, maxiter=200, xtol=1e-10)
    except ValueError:
        return np.nan

# ============================================================
# Quartic-with-ATM-knot IV model in log-moneyness
# ============================================================
def design_quartic_knot(x):
    x = np.asarray(x)
    return np.vstack([
        np.ones_like(x),
        x, x**2, x**3, x**4,
        np.where(x > 0.0, x**4, 0.0)  # (x)_+^4
    ]).T

def fit_iv_quartic_knot_safe(K, iv, F, min_points=6):
    """
    Robust quartic-with-ATM-knot IV fit that handles NaNs.
    - Uses only finite IVs.
    - If too few, linearly interpolates IV in log-moneyness (edge hold).
    - If still too few, falls back to constant IV (median or 20%).
    Returns (beta, iv_fit, info).
    """
    K = np.asarray(K)
    iv = np.asarray(iv)
    x = np.log(K / F)

    def _fit_plain(x_sub, iv_sub):
        X = design_quartic_knot(x_sub)
        beta, *_ = np.linalg.lstsq(X, iv_sub, rcond=None)
        return beta

    finite = np.isfinite(iv)
    if finite.sum() >= min_points:
        beta = _fit_plain(x[finite], iv[finite])
        iv_fit = design_quartic_knot(x) @ beta
        return beta, iv_fit, {"mode": "finite_only", "n_used": int(finite.sum())}

    # interpolate in log-moneyness if we have at least two finite points
    if finite.sum() >= 2:
        order = np.argsort(x)
        xs = x[order]
        ivs = iv[order]
        msk = np.isfinite(ivs)
        iv_interp = np.interp(xs, xs[msk], ivs[msk])   # edge hold implicitly
        iv_filled = np.empty_like(iv)
        iv_filled[order] = iv_interp

        beta = _fit_plain(x, iv_filled)
        iv_fit = design_quartic_knot(x) @ beta
        return beta, iv_fit, {"mode": "interp_logmoneyness", "n_used": int(finite.sum())}

    # final fallback: constant IV
    iv_const = np.nanmedian(iv) if np.isfinite(iv).any() else 0.20
    beta = np.zeros(6); beta[0] = iv_const
    iv_fit = np.full_like(iv, iv_const, dtype=float)
    return beta, iv_fit, {"mode": "constant_fallback", "iv_const": float(iv_const)}

def eval_iv_quartic_knot(K, beta, F):
    x = np.log(K / F)
    return design_quartic_knot(x) @ beta

# ============================================================
# Breeden–Litzenberger: q(K) = e^{rT} * d^2 C / dK^2
# ============================================================
def rnd_from_calls(K, C_disc, r, T):
    dC_dK   = np.gradient(C_disc, K, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK,  K, edge_order=2)
    return np.exp(r*T) * d2C_dK2

# ============================================================
# Demo / usage
# ============================================================
if __name__ == "__main__":
    # Setup
    S0, r, q, T = 120.0, 0.02, 0.0, 0.5
    strikes = np.linspace(50, 175, 25)
    hparams = HestonParams(kappa=0.5, theta=0.05, sigma=0.25, v0=0.02, rho=-0.7)

    # Heston discounted calls and RND (for comparison)
    C_mkt = heston_call_prices_fast(S0, strikes, r, q, T, hparams)
    q_K   = rnd_from_calls(strikes, C_mkt, r, T)

    # Heston IVs (some may be NaN if inversion fails)
    iv_heston = np.array([bs_iv_from_price_disc(S0, K, r, q, T, C)
                          for K, C in zip(strikes, C_mkt)])

    # Fit quartic-with-knot IV, robust to NaNs
    F = S0 * np.exp((r - q) * T)
    beta, iv_fit, meta = fit_iv_quartic_knot_safe(strikes, iv_heston, F)
    iv_fit = np.clip(iv_fit, 1e-6, 5.0)  # keep vols reasonable
    print("IV fit mode:", meta)

    # Convert fitted IVs back to discounted call prices
    C_fit = np.array([bs_call_disc(S0, K, r, q, T, sig)
                      for K, sig in zip(strikes, iv_fit)])

    # Derive fitted RND from fitted calls
    q_fit = rnd_from_calls(strikes, C_fit, r, T)

    # Diagnostics
    finite_mask = np.isfinite(iv_heston)
    rmse_iv = np.sqrt(np.mean((iv_fit[finite_mask] - iv_heston[finite_mask])**2)) \
              if finite_mask.any() else np.nan
    rmse_calls = np.sqrt(np.mean((C_fit - C_mkt)**2))
    print("RMSE (IV over finite pts):", rmse_iv)
    print("RMSE (Calls):             ", rmse_calls)

    # ----------------- PLOTS -----------------
    # 1) RND comparison
    plt.figure(figsize=(7.2, 5.2))
    plt.plot(strikes, q_K,   label="Heston RND (from BL)", lw=2)
    plt.plot(strikes, q_fit, label="Fitted RND (IV quartic w/ ATM knot)", lw=2, ls="--")
    plt.title("Risk-Neutral Density: Heston vs Fitted")
    plt.xlabel("Strike K"); plt.ylabel("q(K)")
    plt.legend(); plt.tight_layout(); plt.show()

    # 2) IV and Calls (side-by-side)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.8))
    fig.suptitle("Fitted IV & Calls vs Heston")

    # IV: scatter only on finite points
    mask = np.isfinite(iv_heston)
    ax1.scatter(strikes[mask], iv_heston[mask], label="Heston IV", s=30)
    ax1.plot(strikes, iv_fit, label="Fitted IV (quartic + knot)", lw=2, ls="--")
    ax1.set_xlabel("Strike K"); ax1.set_ylabel("Implied Volatility")
    ax1.legend(); ax1.grid(alpha=0.25)

    # Calls
    ax2.plot(strikes, C_mkt, label="Heston calls (discounted)", lw=2)
    ax2.plot(strikes, C_fit, label="Fitted calls (from fitted IV)", lw=2, ls="--")
    ax2.set_xlabel("Strike K"); ax2.set_ylabel("Call price")
    ax2.legend(); ax2.grid(alpha=0.25)
    
    ax3.plot(strikes, q_K,   label="Heston RND (from BL)", lw=2)
    ax3.plot(strikes, q_fit, label="Fitted RND (IV quartic w/ ATM knot)", lw=2, ls="--")
    ax3.set_xlabel("Strike K"); ax2.set_ylabel("Call price")
    ax3.legend(); ax2.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
