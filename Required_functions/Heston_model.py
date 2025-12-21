import numpy as np
from dataclasses import dataclass
from typing import Iterable
from scipy.integrate import quad        # still here if you want the old pricer
from scipy.optimize import least_squares
from numpy.polynomial.legendre import leggauss

# =========================
# Heston pricer pieces (yours)
# =========================
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

# -------------------------------------------------
# FAST pricer: vectorized Gauss–Legendre on [0, U]
# -------------------------------------------------
def _heston_probabilities_GL(
    Ks: np.ndarray, S0: float, r: float, q: float, T: float, p: HestonParams,
    U_MAX: float = 200.0, N: int = 128
):
    """
    Compute P1 and P2 for all strikes at once using Gauss–Legendre quadrature on [0, U_MAX].
    Returns arrays P1, P2 with shape (nK,).
    """
    Ks = np.asarray(Ks, float)
    nK = Ks.size
    lnK = np.log(np.maximum(Ks, 1e-300))
    F = S0 * np.exp((r - q) * T)

    # Legendre nodes/weights on [-1,1], map to [0,U_MAX]
    x, w = leggauss(N)                       # shape (N,)
    u = 0.5 * (x + 1.0) * U_MAX              # map to [0, U_MAX]
    du = 0.5 * U_MAX
    w = w * du                               # scale weights for [0,U_MAX]

    # Characteristic function values (vector over u)
    phi_u   = np.array([_heston_char(ui,      T, S0, r, q, p) for ui in u])
    phi_u1i = np.array([_heston_char(ui - 1j, T, S0, r, q, p) for ui in u])

    # Common denominators (avoid division in the inner loop)
    denom_u   = 1j * u
    denom_u1i = 1j * u

    # For each strike, need exp(-i u lnK)
    e_minus_iulnK = np.exp(-1j * np.outer(u, lnK))   # shape (N, nK)

    # Integrands (vectorized): Re[ e^{-iu lnK} * phi(...) / (i u * ...) ]
    # P1 integrand uses phi(u - i) / (S0 e^{(r-q)T})
    integrand_P1 = np.real( e_minus_iulnK * (phi_u1i[:, None] / (denom_u1i[:, None] * F)) )
    integrand_P2 = np.real( e_minus_iulnK * (phi_u[:,   None] /  denom_u[:,   None]) )

    # Integrals via dot with weights
    I1 = (w[:, None] * integrand_P1).sum(axis=0)   # shape (nK,)
    I2 = (w[:, None] * integrand_P2).sum(axis=0)

    P1 = 0.5 + (1.0 / np.pi) * I1
    P2 = 0.5 + (1.0 / np.pi) * I2
    return P1, P2

def heston_call_prices_fast(
    S0: float, Ks: Iterable[float], r: float, q: float, T: float, p: HestonParams,
    U_MAX: float = 200.0, N: int = 128
) -> np.ndarray:
    Ks = np.asarray(list(Ks), float)
    P1, P2 = _heston_probabilities_GL(Ks, S0, r, q, T, p, U_MAX=U_MAX, N=N)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    return S0 * disc_q * P1 - Ks * disc_r * P2

# =========================
# Calibration (prices only, no weights) — uses FAST pricer
# =========================
def calibrate_heston_prices(
    S0: float,
    strikes: np.ndarray,
    prices_mkt: np.ndarray,   # discounted calls
    r: float, q: float, T: float,
    p0: HestonParams = HestonParams(1.0, 0.04, 0.5, 0.04, -0.4),
    bounds=((1e-6, 10.0), (1e-6, 2.0), (1e-6, 5.0), (0.0, 2.0), (-0.999, 0.999)),
    U_MAX: float = 200.0, N: int = 128,      # quadrature controls
    max_nfev: int = 80,                      # optimizer budget (reduce from 200)
    xtol: float = 1e-6, ftol: float = 1e-6, gtol: float = 1e-6
):
    """
    Plain least-squares fit in price space:
       minimize || HestonPrices_fast(p) - prices_mkt ||_2
    No IVs, no weights.
    """
    K = np.asarray(strikes, float)
    C_mkt = np.asarray(prices_mkt, float)

    def pack(p: HestonParams):
        return np.array([p.kappa, p.theta, p.sigma, p.v0, p.rho], float)

    def unpack(x: np.ndarray) -> HestonParams:
        return HestonParams(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))

    lb = np.array([b[0] for b in bounds], float)
    ub = np.array([b[1] for b in bounds], float)
    x0 = pack(p0)

    # cache to avoid allocating per call
    def residuals(x: np.ndarray):
        p = unpack(x)
        C_fit = heston_call_prices_fast(S0, K, r, q, T, p, U_MAX=U_MAX, N=N)
        return C_fit - C_mkt

    sol = least_squares(
        residuals, x0, bounds=(lb, ub), method="trf",
        loss="linear", max_nfev=max_nfev, xtol=xtol, ftol=ftol, gtol=gtol
    )

    p_hat = unpack(sol.x)
    C_fit = heston_call_prices_fast(S0, K, r, q, T, p_hat, U_MAX=U_MAX, N=N)

    return {
        "success": sol.success,
        "message": sol.message,
        "params": p_hat,
        "cost": sol.cost,
        "nfev": sol.nfev,
        "residuals": sol.fun,
        "C_fit": C_fit,
        "C_mkt": C_mkt,
    }

# =========================
# Example
# =========================
if __name__ == "__main__":
    true_p = HestonParams(kappa=0.5, theta=0.05, sigma=0.25, v0=0.02, rho=-0.6)
    S0, r, q, T = 120.0, 0.02, 0.00, 0.5
    Ks = np.linspace(25, 250, 80)
    # “Market” prices (use fast pricer here too, to avoid mixed discretizations)
    C_mkt = heston_call_prices_fast(S0, Ks, r, q, T, true_p)

    # Calibrate (no IVs, no weights) with faster integrator and smaller eval budget
    p0 = HestonParams(1.0, 0.08, 0.6, 0.03, -0.2)
    out = calibrate_heston_prices(
        S0, Ks, C_mkt, r, q, T, p0=p0,
        U_MAX=200.0, N=128, max_nfev=80
    )

    print("Success:", out["success"], "|", out["message"])
    print("Fitted params:", out["params"])
    C_fitted = heston_call_prices_fast(S0, Ks, r, q, T,out["params"])

    # Quick visual
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,5))
    plt.plot(Ks, out["C_mkt"], label="Market (discounted)", lw=2)
    plt.plot(Ks,C_fitted, label="Calibrated Heston (fast GL)", lw=2, ls="--")
    plt.xlabel("Strike K"); plt.ylabel("Call price")
    plt.title("Heston Calibration (prices only, fast quadrature)")
    plt.legend(); plt.tight_layout(); plt.show()
