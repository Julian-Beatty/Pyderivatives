from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def gauss_legendre_0U(n: int, U: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre nodes and weights on [0, U]."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    u = 0.5 * (x + 1.0) * float(U)
    wu = 0.5 * float(U) * w
    u = np.where(np.abs(u) < 1e-12, 1e-12, u)
    return np.asarray(u, float), np.asarray(wu, float)


def affine_heston_logreturn_cf(
    u: np.ndarray,
    T: float,
    *,
    v0: float,
    theta_q: float,
    kappa_q: float,
    sigma_v: float,
    rho: float,
    gamma_return: float,
    gamma_variance: float,
) -> np.ndarray:
    """Physical CF for log return under the Shackleton affine drift mapping.

    Physical dynamics (without jumps):

        dS/S = gamma_return * V dt + sqrt(V) dW1^P
        dV   = [gamma_variance * V + kappa_q(theta_q - V)]dt
               + sigma_v sqrt(V) dW2^P.

    Thus kappa_P = kappa_q - gamma_variance while the CIR intercept
    a = kappa_q * theta_q is unchanged.  The CF is for X_T-X_0.
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    T = float(T)

    kappa_p = float(kappa_q) - float(gamma_variance)
    if not np.isfinite(kappa_p) or kappa_p <= 1e-10:
        return np.full_like(u, np.nan + 1j * np.nan)

    a = float(kappa_q) * float(theta_q)
    sig2 = float(sigma_v) ** 2
    if sig2 <= 0 or v0 < 0 or theta_q <= 0 or kappa_q <= 0:
        return np.full_like(u, np.nan + 1j * np.nan)

    # d log S = (gamma_return - 1/2)V dt + sqrt(V)dW.
    beta = float(gamma_return) - 0.5
    b_u = kappa_p - float(rho) * float(sigma_v) * iu
    d = np.sqrt(b_u * b_u + sig2 * (u * u - 2.0 * iu * beta))

    # Enforce the numerically stable square-root branch.
    d = np.where(np.real(d) < 0, -d, d)
    g = (b_u - d) / (b_u + d)
    exp_minus_dT = np.exp(-d * T)

    eps = 1e-16
    denom = 1.0 - g * exp_minus_dT + eps
    denom0 = 1.0 - g + eps

    D = ((b_u - d) / sig2) * ((1.0 - exp_minus_dT) / denom)
    C = (a / sig2) * (
        (b_u - d) * T
        - 2.0 * np.log(denom / denom0)
    )

    return np.exp(C + D * float(v0))


def kou_jump_cf(u: np.ndarray, *, p_up: float, eta1: float, eta2: float) -> np.ndarray:
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    return (
        float(p_up) * float(eta1) / (float(eta1) - iu)
        + (1.0 - float(p_up)) * float(eta2) / (float(eta2) + iu)
    )


def kou_exponential_compensator(*, p_up: float, eta1: float, eta2: float) -> float:
    if float(eta1) <= 1.0:
        return np.nan
    ej = (
        float(p_up) * float(eta1) / (float(eta1) - 1.0)
        + (1.0 - float(p_up)) * float(eta2) / (float(eta2) + 1.0)
    )
    return float(ej - 1.0)


def translated_kou_jump_factor(
    u: np.ndarray,
    T: float,
    *,
    lam: float,
    p_up: float,
    eta1: float,
    eta2: float,
    jump_translation: float,
) -> np.ndarray:
    """Compound-Poisson CF factor for J_P = J_Q + jump_translation.

    The intensity and original Kou shape parameters remain fixed.  The stock
    drift uses the physical compensator E_P[e^{J_P}-1].
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u

    phi_q = kou_jump_cf(u, p_up=p_up, eta1=eta1, eta2=eta2)
    phi_p = np.exp(iu * float(jump_translation)) * phi_q

    kappa_q = kou_exponential_compensator(p_up=p_up, eta1=eta1, eta2=eta2)
    if not np.isfinite(kappa_q):
        return np.full_like(u, np.nan + 1j * np.nan)

    # E[e^(J_Q + delta)] - 1 = exp(delta) * E[e^J_Q] - 1.
    kappa_p = np.exp(float(jump_translation)) * (1.0 + kappa_q) - 1.0
    exponent = float(lam) * float(T) * (phi_p - 1.0 - iu * kappa_p)
    return np.exp(exponent)


def invert_cf_to_pdf(
    x_grid: np.ndarray,
    cf_values: np.ndarray,
    u: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Invert a log-return characteristic function on an arbitrary x grid."""
    x = np.asarray(x_grid, float).ravel()
    cf = np.asarray(cf_values, complex).ravel()
    u = np.asarray(u, float).ravel()
    weights = np.asarray(weights, float).ravel()

    if not (cf.size == u.size == weights.size) or np.any(~np.isfinite(cf)):
        return np.full_like(x, np.nan)

    phase = np.exp(-1j * np.outer(x, u))
    pdf = np.real((phase * cf[None, :]) @ weights) / np.pi
    # Small negative inversion noise is removed; normalization is performed by
    # StochasticRiskPremiaTransform afterward.
    return np.where(np.isfinite(pdf), np.maximum(pdf, 0.0), np.nan)
