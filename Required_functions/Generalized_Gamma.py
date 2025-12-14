import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from scipy.special import gamma, gammaincc  # upper incomplete gamma ratio
from Heston_model import *  # assumes HestonParams, heston_call_prices_fast, etc.
"""
Generalized-Gamma Risk-Neutral Density (RND) model for S_T and closed-form call pricing.

References
----------
- Bouzai, B. (2022). "The Generalized Gamma Distribution as a Useful Risk-Neutral Distribution under
  Heston’s Stochastic Volatility Model." Journal of Risk and Financial Management, 15(6), 238.

- "Estimating risk-neutral density with parametric models in interest rate markets" (2009).
  (Include the full bibliographic entry you’re using in your paper’s reference list.)

Notes
-----
This implementation:
  - Calibrates (a, d, p) by least squares on call prices using a closed-form C(K).
  - Optionally adds a forward-matching penalty to keep E[S_T] close to S0 * exp(rT).
  - Provides qhat(s) (density), chat(K) (fitted prices), and BS implied vols (iv).

The generalized gamma is in Stacy form:
  f(s) = d / (a^(d p) Γ(p)) * s^(d p - 1) * exp(-(s/a)^d),   s>0.

Closed-form call:
  C(K) = e^{-rT} [ E[S 1_{S>K}] - K P(S>K) ],
  where z=(K/a)^d,  P(S>K)=gammaincc(p, z),
  and E[S 1_{S>K}] = a * Γ(p+1/d, z) / Γ(p)
                   = a * Γ(p+1/d)*gammaincc(p+1/d, z) / Γ(p).
"""



# ============================================================
# 1) Parameter container
# ============================================================

@dataclass
class GenGammaParams:
    """Generalized gamma parameters (Stacy form)."""
    a: float  # scale > 0
    d: float  # shape > 0
    p: float  # shape > 0


# ============================================================
# 2) Core model class
# ============================================================

class GenGammaRNDModel:
    """
    Generalized Gamma risk-neutral density model for S_T:

        S_T ~ GenGamma(a, d, p) on (0, ∞)

    Defaults are meant to be "reasonable" for equity/crypto-like underlyings:
      - a_bounds defaults to (0.2*S0, 5.0*S0)
      - d_bounds defaults to (0.2, 25.0)
      - p_bounds defaults to (0.2, 50.0)
    """

    def __init__(
        self,
        S0: float,
        r: float,
        T: float,
        K: Sequence[float],
        C_mkt: Sequence[float],
        a0: float = 100.0,
        d0: float = 1.5,
        p0: float = 2.0,
        forward_penalty_lambda: float = 0.0,
        a_bounds: Optional[Tuple[float, float]] = None,
        d_bounds: Tuple[float, float] = (0.2, 25.0),
        p_bounds: Tuple[float, float] = (0.2, 50.0),
        opt_maxiter: int = 2000,
        opt_ftol: float = 1e-10,
    ):
        # --- market inputs
        self.S0 = float(S0)
        self.r = float(r)
        self.T = float(T)
        self.K = np.asarray(K, dtype=float)
        self.C_mkt = np.asarray(C_mkt, dtype=float)

        # --- calibration controls
        self.forward_penalty_lambda = float(forward_penalty_lambda)
        self.opt_maxiter = int(opt_maxiter)
        self.opt_ftol = float(opt_ftol)

        # --- validation
        if self.S0 <= 0.0:
            raise ValueError("S0 must be positive.")
        if self.T <= 0.0:
            raise ValueError("T must be positive.")
        if self.K.shape != self.C_mkt.shape:
            raise ValueError("K and C_mkt must have the same shape.")
        if a0 <= 0.0 or d0 <= 0.0 or p0 <= 0.0:
            raise ValueError("Initial (a0,d0,p0) must be > 0.")

        # --- bounds
        if a_bounds is None:
            a_bounds = (0.2 * self.S0, 5.0 * self.S0)

        self.a_bounds = a_bounds
        self.d_bounds = d_bounds
        self.p_bounds = p_bounds

        for name, (lo, hi) in zip(
            ["a_bounds", "d_bounds", "p_bounds"],
            [self.a_bounds, self.d_bounds, self.p_bounds],
        ):
            if not (lo > 0 and hi > lo):
                raise ValueError(f"{name} must satisfy 0 < lower < upper.")

        # Calibrate on initialization
        self.params, self._opt_result = self._calibrate(a0, d0, p0)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def chat(self, strikes: Sequence[float]) -> np.ndarray:
        """Return fitted call prices C_hat(K) on a strike grid."""
        K_eval = np.asarray(strikes, dtype=float)
        return self._call_prices(K_eval, self.params)

    def qhat(self, S_grid: Sequence[float]) -> np.ndarray:
        """Return fitted risk-neutral density q_hat(s) on a terminal-price grid."""
        S_grid = np.asarray(S_grid, dtype=float)
        return self._gengamma_pdf(S_grid, self.params)

    def iv(self, strikes: Sequence[float]) -> np.ndarray:
        """Return BS implied vols computed from fitted call prices."""
        K_eval = np.asarray(strikes, dtype=float)
        C_target = self.chat(K_eval)
        return self._implied_vols_bs(K_eval, C_target)

    @property
    def optimization_result(self):
        """Expose the SciPy optimizer output."""
        return self._opt_result

    # --------------------------------------------------------
    # Generalized Gamma primitives
    # --------------------------------------------------------

    @staticmethod
    def _gengamma_pdf(s: np.ndarray, params: GenGammaParams) -> np.ndarray:
        """
        Generalized gamma PDF (Stacy form):
          f(s) = d / (a^(d p) Γ(p)) * s^(d p - 1) * exp(-(s/a)^d), s>0
        """
        a, d, p = params.a, params.d, params.p
        s = np.asarray(s, dtype=float)

        pdf = np.zeros_like(s)
        m = s > 0.0
        sm = s[m]

        coeff = d / (a ** (d * p) * gamma(p))
        pdf[m] = coeff * sm ** (d * p - 1.0) * np.exp(-(sm / a) ** d)
        return pdf

    @staticmethod
    def _gengamma_moment(order: float, params: GenGammaParams) -> float:
        """
        Generalized gamma moment:
          E[S^r] = a^r * Γ(p + r/d) / Γ(p)
        """
        a, d, p = params.a, params.d, params.p
        return a ** order * gamma(p + order / d) / gamma(p)

    # --------------------------------------------------------
    # Closed-form call pricing
    # --------------------------------------------------------

    def _call_prices(self, K: np.ndarray, params: GenGammaParams) -> np.ndarray:
        """
        Vectorized closed-form call prices.

        C(K) = e^{-rT} [ E[S 1_{S>K}] - K P(S>K) ].

        For K>0:
          z=(K/a)^d
          P(S>K)=gammaincc(p, z)
          E[S 1_{S>K}] = a * Γ(p+1/d)*gammaincc(p+1/d, z) / Γ(p)
        """
        a, d, p = params.a, params.d, params.p
        K = np.asarray(K, dtype=float)

        disc = np.exp(-self.r * self.T)
        C = np.zeros_like(K)

        # K <= 0 => payoff is always in the money => price = e^{-rT} E[S]
        m_zero = K <= 0.0
        if np.any(m_zero):
            E_ST = self._gengamma_moment(1.0, params)
            C[m_zero] = disc * E_ST

        # K > 0
        m_pos = ~m_zero
        if np.any(m_pos):
            Kp = K[m_pos]
            z = (Kp / a) ** d

            tail_prob = gammaincc(p, z)  # Γ(p,z)/Γ(p)

            s1 = p + 1.0 / d
            E_trunc = a * gamma(s1) * gammaincc(s1, z) / gamma(p)

            C[m_pos] = disc * (E_trunc - Kp * tail_prob)

        return C

    # --------------------------------------------------------
    # Calibration with bounds
    # --------------------------------------------------------

    def _calibrate(self, a0: float, d0: float, p0: float):
        """
        Calibrate (a,d,p) by minimizing squared pricing errors with L-BFGS-B,
        enforcing positivity via log-parameters and enforcing box bounds.
        """
        def pack(a: float, d: float, p: float) -> np.ndarray:
            return np.array([np.log(a), np.log(d), np.log(p)], dtype=float)

        def unpack(theta: np.ndarray) -> GenGammaParams:
            return GenGammaParams(
                a=float(np.exp(theta[0])),
                d=float(np.exp(theta[1])),
                p=float(np.exp(theta[2])),
            )

        # Clip initial guesses to bounds
        a0 = float(np.clip(a0, *self.a_bounds))
        d0 = float(np.clip(d0, *self.d_bounds))
        p0 = float(np.clip(p0, *self.p_bounds))
        theta0 = pack(a0, d0, p0)

        # Bounds in log-space
        theta_bounds = [
            (np.log(self.a_bounds[0]), np.log(self.a_bounds[1])),
            (np.log(self.d_bounds[0]), np.log(self.d_bounds[1])),
            (np.log(self.p_bounds[0]), np.log(self.p_bounds[1])),
        ]

        def objective(theta: np.ndarray) -> float:
            params = unpack(theta)
            C_model = self._call_prices(self.K, params)

            resid = C_model - self.C_mkt
            sse = float(np.sum(resid * resid))

            # Optional forward-matching penalty: keep E[S_T] close to forward
            if self.forward_penalty_lambda > 0.0:
                E_ST = self._gengamma_moment(1.0, params)
                forward_target = self.S0 * np.exp(self.r * self.T)
                sse += self.forward_penalty_lambda * float((E_ST - forward_target) ** 2)

            return sse

        res = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            bounds=theta_bounds,
            options={"maxiter": self.opt_maxiter, "ftol": self.opt_ftol},
        )

        return unpack(res.x), res

    # --------------------------------------------------------
    # Black–Scholes helpers for implied vol
    # --------------------------------------------------------

    def _bs_call_price(self, K: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Vectorized Black–Scholes call price."""
        K = np.asarray(K, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sqrtT = np.sqrt(self.T)

        d1 = (np.log(self.S0 / K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        N = norm.cdf
        return self.S0 * N(d1) - K * np.exp(-self.r * self.T) * N(d2)

    def _implied_vols_bs(self, K: np.ndarray, C_target: np.ndarray) -> np.ndarray:
        """
        Solve for BS implied volatility using brentq for each (K, C) pair.
        Returns NaN when a valid root does not exist.
        """
        K = np.asarray(K, dtype=float)
        C_target = np.asarray(C_target, dtype=float)

        iv = np.full_like(K, np.nan, dtype=float)
        sigma_low, sigma_high = 1e-6, 5.0
        disc = np.exp(-self.r * self.T)

        for i, (Ki, Ci) in enumerate(zip(K, C_target)):
            if Ki <= 0.0 or Ci <= 0.0:
                continue

            intrinsic = max(self.S0 - Ki * disc, 0.0)
            Ci = max(Ci, intrinsic)

            C_low = self._bs_call_price(np.array([Ki]), np.array([sigma_low]))[0]
            C_high = self._bs_call_price(np.array([Ki]), np.array([sigma_high]))[0]
            if not (C_low <= Ci <= C_high):
                continue

            def f(sig):
                return self._bs_call_price(np.array([Ki]), np.array([sig]))[0] - Ci

            try:
                iv[i] = brentq(f, sigma_low, sigma_high, maxiter=100, xtol=1e-8)
            except ValueError:
                iv[i] = np.nan

        return iv


# ============================================================
# 3) Example usage: Heston world + GenGamma fit
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- "true" Heston world
    true_p = HestonParams(kappa=0.5, theta=0.05, sigma=0.4, v0=0.02, rho=0.9)
    S0, r, q_div, T = 120.0, 0.02, 0.00, 0.5

    strikes = np.linspace(25, 200, 20)
    C_mkt = heston_call_prices_fast(S0, strikes, r, q_div, T, true_p)

    # --- optional noise
    rng = np.random.default_rng(0)
    noise = 0.0  # e.g., rng.normal(scale=0.02 * C_mkt)
    C_mkt_noise = C_mkt + noise

    # --- "true" RND proxy via Breeden–Litzenberger (coarse, on strike grid)
    dC_dK   = np.gradient(C_mkt, strikes, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, strikes, edge_order=2)
    q_true  = np.exp(r * T) * d2C_dK2

    # --- fit generalized gamma
    fit = GenGammaRNDModel(
        S0=S0,
        r=r,
        T=T,
        K=strikes,
        C_mkt=C_mkt_noise,
        a0=S0,
        d0=1.5,
        p0=2.0,
        forward_penalty_lambda=1e-4,
        # a_bounds=None -> defaults to (0.2*S0, 5.0*S0)
        # d_bounds and p_bounds already have defaults
    )

    print("Fitted Generalized Gamma parameters:")
    print("a =", fit.params.a)
    print("d =", fit.params.d)
    print("p =", fit.params.p)

    fine_strikes = np.linspace(25, 200, 200)
    q_hat = fit.qhat(fine_strikes)
    c_hat = fit.chat(fine_strikes)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    ax = axes[0]
    ax.plot(strikes, q_true, "k--", label="True Heston RND (coarse BL)", lw=2)
    ax.plot(fine_strikes, q_hat, label="GenGamma RND (fitted)", lw=2)
    ax.set_xlabel("Terminal price s")
    ax.set_ylabel("q(s)")
    ax.set_title("Risk-Neutral Density: Heston vs Generalized Gamma")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(strikes, C_mkt, "o", ms=4, label="True Heston calls")
    ax2.plot(fine_strikes, c_hat, "-", lw=2, label="Fitted GenGamma calls")
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Call price C(K)")
    ax2.set_title("Call Curve Fit: Heston vs Generalized Gamma")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
