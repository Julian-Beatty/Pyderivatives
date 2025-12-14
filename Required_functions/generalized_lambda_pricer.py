# New version

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple
from scipy.optimize import minimize, brentq
from scipy.special import beta as beta_fn
from scipy.stats import norm
from Heston_model import *  # assumes HestonParams, heston_call_prices_fast, etc.
from typing import Sequence, Tuple, Optional


# ============================================================
# 1) Parameter container
# ============================================================

@dataclass
class GLDParams:
    """
    Corrado (2001) GLD parameters (RS-GLD style).

    sigma : "vol/variance" parameter used in Corrado's mapping:
            A = sqrt(exp(sigma^2 * T) - 1)
    k3,k4 : shape parameters of the RS-GLD quantile function.
    """
    sigma: float
    k3: float
    k4: float


# ============================================================
# 2) Core model
# ============================================================

class GLDRNDModel:
    """
    Corrado (2001) generalized lambda distribution (GLD) option pricing model.

    Key idea:
      - Model S_T under Q via its percentile (quantile) function S(p), p in (0,1).
      - For each strike K, solve S(pK) = K for pK, then price calls by integrating payoff over p:
            C(K) = e^{-rT} ∫_{pK}^1 (S(p) - K) dp.
      - Density is q(s) = dp/ds = 1 / (dS/dp) evaluated at p(s).

    Performance:
      - Pricing uses a vectorized bisection routine to invert S(p)=K for all strikes.
      - Density uses interpolation inversion (fast) rather than root-finding per grid point.
    """

    def __init__(
        self,
        S0: float,
        r: float,
        T: float,
        K: Sequence[float],
        C_mkt: Sequence[float],
        sigma0: float = 0.30,
        k30: float = 0.05,
        k40: float = 0.02,
        forward_penalty_lambda: float = 0.0,
        sigma_bounds: Tuple[float, float] = (1e-4, 3.0),
        k3_bounds: Tuple[float, float] = (-0.49, 5.0),
        k4_bounds: Tuple[float, float] = (-0.49, 5.0),
        eps_p: float = 1e-10,
        bisect_max_iter: int = 60,
        bisect_tol: float = 1e-12,
        pdf_pgrid_size: int = 20000,
    ):
        # --- market inputs
        self.S0 = float(S0)
        self.r = float(r)
        self.T = float(T)
        self.K = np.asarray(K, dtype=float)
        self.C_mkt = np.asarray(C_mkt, dtype=float)

        # --- calibration / numerical controls
        self.forward_penalty_lambda = float(forward_penalty_lambda)
        self.sigma_bounds = sigma_bounds
        self.k3_bounds = k3_bounds
        self.k4_bounds = k4_bounds
        self.eps_p = float(eps_p)
        self.bisect_max_iter = int(bisect_max_iter)
        self.bisect_tol = float(bisect_tol)
        self.pdf_pgrid_size = int(pdf_pgrid_size)

        # --- basic validation
        if self.S0 <= 0.0:
            raise ValueError("S0 must be positive.")
        if self.T <= 0.0:
            raise ValueError("T must be positive.")
        if self.K.shape != self.C_mkt.shape:
            raise ValueError("K and C_mkt must have the same shape.")

        # Calibrate immediately
        self.params, self._opt_result = self._calibrate(sigma0, k30, k40)

    # --------------------------------------------------------
    # Public API (same naming as your GenGamma class)
    # --------------------------------------------------------

    def chat(self, strikes: Sequence[float]) -> np.ndarray:
        """Return fitted call prices C_hat(K) for an input strike grid."""
        K_eval = np.asarray(strikes, dtype=float)
        return self._call_prices(K_eval, self.params)

    def qhat(self, S_grid: Sequence[float]) -> np.ndarray:
        """Return fitted risk-neutral density q_hat(s) over a terminal-price grid."""
        S_grid = np.asarray(S_grid, dtype=float)
        return self._pdf_ST(S_grid, self.params)

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
    # GLD building blocks
    # --------------------------------------------------------

    @staticmethod
    def _k2(k3: float, k4: float) -> float:
        """
        Compute the RS-GLD scaling term k2(k3,k4) so the standardized quantile has (approximately)
        zero mean and unit variance.

        Uses the RS-GLD variance identity:
            A = 1/(k3+1) - 1/(k4+1)
            B = 1/(2k3+1) + 1/(2k4+1) - 2*Beta(k3+1, k4+1)
            Var = B - A^2
            k2 = sign(k3+k4) * sqrt(Var)

        Feasibility guards:
          - k3 > -1/2 and k4 > -1/2 (for second moments to exist)
          - k3 + k4 > 0 (a standard RS-GLD admissibility constraint)
          - Var > 0
        """
        if (k3 <= -0.5) or (k4 <= -0.5):
            return np.nan
        if (k3 + k4) <= 0.0:
            return np.nan

        A = 1.0 / (k3 + 1.0) - 1.0 / (k4 + 1.0)
        B = (
            1.0 / (2.0 * k3 + 1.0)
            + 1.0 / (2.0 * k4 + 1.0)
            - 2.0 * beta_fn(k3 + 1.0, k4 + 1.0)
        )
        Var = B - A * A
        if not np.isfinite(Var) or Var <= 0.0:
            return np.nan
        return np.sign(k3 + k4) * np.sqrt(Var)

    def _A_scale(self, sigma: float) -> float:
        """
        Corrado (2001) mapping from sigma to a scale factor:
            A = sqrt(exp(sigma^2 * T) - 1).
        """
        return float(np.sqrt(np.exp((sigma * sigma) * self.T) - 1.0))

    def _x_of_p(self, p: np.ndarray, k3: float, k4: float) -> np.ndarray:
        """
        Standardized GLD quantile function x(p):
            x(p) = (p^k3 - (1-p)^k4) / k2(k3,k4)
        """
        k2 = self._k2(k3, k4)
        if not np.isfinite(k2) or k2 == 0.0:
            return np.full_like(p, np.nan, dtype=float)
        return (np.power(p, k3) - np.power(1.0 - p, k4)) / k2

    def _S_of_p(self, p: np.ndarray, params: GLDParams) -> np.ndarray:
        """
        Security-price percentile (quantile) function S(p):
            S(p) = F * [1 + A * x(p)]
        where F = S0 * exp(rT).
        """
        p = np.asarray(p, dtype=float)
        p = np.clip(p, self.eps_p, 1.0 - self.eps_p)
        A = self._A_scale(params.sigma)
        x = self._x_of_p(p, params.k3, params.k4)
        F = self.S0 * np.exp(self.r * self.T)
        return F * (1.0 + A * x)

    def _dSdp(self, p: np.ndarray, params: GLDParams) -> np.ndarray:
        """
        Derivative dS/dp, used to compute the density:
            q(s) = dp/ds = 1 / (dS/dp) evaluated at p(s).
        """
        p = np.asarray(p, dtype=float)
        p = np.clip(p, self.eps_p, 1.0 - self.eps_p)

        k2 = self._k2(params.k3, params.k4)
        if not np.isfinite(k2) or k2 == 0.0:
            return np.full_like(p, np.nan, dtype=float)

        A = self._A_scale(params.sigma)
        F = self.S0 * np.exp(self.r * self.T)

        # d/dp[p^k3] = k3 * p^(k3-1)
        # d/dp[-(1-p)^k4] = + k4 * (1-p)^(k4-1)
        term = (
            params.k3 * np.power(p, params.k3 - 1.0)
            + params.k4 * np.power(1.0 - p, params.k4 - 1.0)
        )
        return F * A * (term / k2)

    # --------------------------------------------------------
    # Inversion: fast vectorized bisection for p(K)
    # --------------------------------------------------------

    def _p_of_K_bisect(
        self,
        K: np.ndarray,
        params: GLDParams,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> np.ndarray:
        """
        Solve S(p)=K for p via vectorized bisection for all strikes simultaneously.

        Returns:
          pK array with NaN where K is outside numerical support [S(eps), S(1-eps)].

        Notes:
          - This assumes S(p) is strictly increasing in p, which holds for admissible parameters
            (and is checked implicitly by bracketing).
        """
        if max_iter is None:
            max_iter = self.bisect_max_iter
        if tol is None:
            tol = self.bisect_tol

        K = np.asarray(K, dtype=float)
        p_lo0, p_hi0 = self.eps_p, 1.0 - self.eps_p

        # Numerical support endpoints
        S_lo = self._S_of_p(np.array([p_lo0]), params)[0]
        S_hi = self._S_of_p(np.array([p_hi0]), params)[0]

        pK = np.full_like(K, np.nan, dtype=float)

        # Only solve where K is finite, positive, and inside support
        m = (K > 0.0) & np.isfinite(K) & (K >= S_lo) & (K <= S_hi)
        if not np.any(m):
            return pK

        Km = K[m]
        lo = np.full_like(Km, p_lo0, dtype=float)
        hi = np.full_like(Km, p_hi0, dtype=float)

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            Smid = self._S_of_p(mid, params)

            go_right = Smid < Km
            lo = np.where(go_right, mid, lo)
            hi = np.where(go_right, hi, mid)

            if np.max(hi - lo) < tol:
                break

        pK[m] = 0.5 * (lo + hi)
        return pK

    # --------------------------------------------------------
    # Call pricing (vectorized)
    # --------------------------------------------------------

    def _call_prices(self, K: np.ndarray, params: GLDParams) -> np.ndarray:
        """
        Vectorized call pricing under GLD via p(K):

            C(K) = e^{-rT} ∫_{p(K)}^1 (S(p) - K) dp

        Using closed-form integrals:
          ∫ p^k3 dp and ∫ (1-p)^k4 dp.
        """
        K = np.asarray(K, dtype=float)
        disc = np.exp(-self.r * self.T)

        A = self._A_scale(params.sigma)
        k2 = self._k2(params.k3, params.k4)

        if not np.isfinite(k2) or k2 == 0.0 or not np.isfinite(A) or A <= 0.0:
            return np.full_like(K, np.nan, dtype=float)

        C = np.full_like(K, np.nan, dtype=float)

        # K <= 0: call ~ S0 - K e^{-rT}
        m0 = (K <= 0.0) & np.isfinite(K)
        if np.any(m0):
            C[m0] = self.S0 - K[m0] * disc

        # K > 0: compute p(K) and price
        m = (K > 0.0) & np.isfinite(K)
        if not np.any(m):
            return C

        pK = self._p_of_K_bisect(K[m], params)
        ok = np.isfinite(pK)
        if not np.any(ok):
            return C

        Km = K[m][ok]
        p = pK[ok]
        one_minus_p = 1.0 - p

        # Integral terms
        I1 = (1.0 - p ** (params.k3 + 1.0)) / (params.k3 + 1.0)
        I2 = (one_minus_p ** (params.k4 + 1.0)) / (params.k4 + 1.0)

        # Pricing identity:
        # C = (S0 - K e^{-rT})*(1-p) + S0*(A/k2)*(I1 - I2)
        Cm = (self.S0 - Km * disc) * one_minus_p + self.S0 * (A / k2) * (I1 - I2)
        Cm = np.maximum(Cm, 0.0)

        tmp = C[m]
        tmp[ok] = Cm
        C[m] = tmp

        return C

    # --------------------------------------------------------
    # Density q(s) via interpolation inversion (fast)
    # --------------------------------------------------------

    def _pdf_ST(self, S_grid: np.ndarray, params: GLDParams) -> np.ndarray:
        """
        Compute q(s) = dp/ds = 1/(dS/dp) using a single dense p-grid and inversion by interpolation.

        This avoids root-finding for each s in S_grid.
        """
        S_grid = np.asarray(S_grid, dtype=float)

        p_grid = np.linspace(self.eps_p, 1.0 - self.eps_p, self.pdf_pgrid_size)
        S_vals = self._S_of_p(p_grid, params)

        # Require monotone S(p) to invert reliably
        if np.any(~np.isfinite(S_vals)) or not np.all(np.diff(S_vals) > 0):
            return np.full_like(S_grid, np.nan, dtype=float)

        p_of_s = np.interp(S_grid, S_vals, p_grid, left=np.nan, right=np.nan)
        dSdp = self._dSdp(p_of_s, params)

        pdf = np.full_like(S_grid, np.nan, dtype=float)
        m = (S_grid > 0) & np.isfinite(p_of_s) & np.isfinite(dSdp) & (dSdp > 0)
        pdf[m] = 1.0 / dSdp[m]
        pdf[S_grid <= 0] = 0.0
        return pdf

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    def _calibrate(self, sigma0: float, k30: float, k40: float):
        """
        Calibrate (sigma, k3, k4) by minimizing squared pricing errors.

        Feasibility is enforced by:
          - bounds in the optimizer
          - returning a large penalty if k2/A are invalid or pricing yields NaNs
        """

        def pack(sigma: float, k3: float, k4: float) -> np.ndarray:
            return np.array([sigma, k3, k4], dtype=float)

        def unpack(theta: np.ndarray) -> GLDParams:
            return GLDParams(sigma=float(theta[0]), k3=float(theta[1]), k4=float(theta[2]))

        theta0 = pack(sigma0, k30, k40)
        bounds = [self.sigma_bounds, self.k3_bounds, self.k4_bounds]

        def objective(theta: np.ndarray) -> float:
            params = unpack(theta)

            k2 = self._k2(params.k3, params.k4)
            if not np.isfinite(k2) or k2 == 0.0:
                return 1e18

            A = self._A_scale(params.sigma)
            if not np.isfinite(A) or A <= 0.0:
                return 1e18

            C_model = self._call_prices(self.K, params)
            if np.any(~np.isfinite(C_model)):
                return 1e18

            resid = C_model - self.C_mkt
            sse = float(np.sum(resid * resid))

            # Optional numeric check/penalty for mean matching forward
            if self.forward_penalty_lambda > 0.0:
                F_target = self.S0 * np.exp(self.r * self.T)
                p_grid = np.linspace(self.eps_p, 1.0 - self.eps_p, 2001)
                mean_num = float(np.trapz(self._S_of_p(p_grid, params), p_grid))
                sse += self.forward_penalty_lambda * (mean_num - F_target) ** 2

            return sse

        res = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        return unpack(res.x), res

    # --------------------------------------------------------
    # Black–Scholes helpers for IV
    # --------------------------------------------------------

    def _bs_call_price(self, K: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Vectorized Black–Scholes call price."""
        K = np.asarray(K, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        sqrtT = np.sqrt(self.T)
        d1 = (np.log(self.S0 / K) + (self.r + 0.5 * sigma**2) * self.T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT

        N = norm.cdf
        return self.S0 * N(d1) - K * np.exp(-self.r * self.T) * N(d2)

    def _implied_vols_bs(self, K: np.ndarray, C_target: np.ndarray) -> np.ndarray:
        """
        Compute BS implied vols per strike using brentq.
        Returns NaN where no valid root exists.
        """
        K = np.asarray(K, dtype=float)
        C_target = np.asarray(C_target, dtype=float)

        iv = np.full_like(K, np.nan, dtype=float)
        sigma_low, sigma_high = 1e-6, 5.0
        disc = np.exp(-self.r * self.T)

        for i, (Ki, Ci) in enumerate(zip(K, C_target)):
            if Ci <= 0.0 or Ki <= 0.0:
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
# 3) Example usage (Heston world + GLD fit)
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- "true" Heston world
    true_p = HestonParams(kappa=0.5, theta=0.05, sigma=0.25, v0=0.02, rho=0.45)
    S0, r, q_div, T = 120.0, 0.02, 0.00, 0.5

    strikes = np.linspace(25, 200, 20)
    C_mkt = heston_call_prices_fast(S0, strikes, r, q_div, T, true_p)

    # --- add noise (optional)
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.02 * C_mkt)
    C_mkt_noise = C_mkt + noise

    # --- "true" RND proxy via BL on strike grid (coarse)
    dC_dK = np.gradient(C_mkt, strikes, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, strikes, edge_order=2)
    q_true = np.exp(r * T) * d2C_dK2

    # --- fit GLD
    fit = GLDRNDModel(
        S0=S0,
        r=r,
        T=T,
        K=strikes,
        C_mkt=C_mkt_noise,
        sigma0=0.30,
        k30=0.05,
        k40=0.02,
        forward_penalty_lambda=0.0,
        sigma_bounds=(1e-4, 2.5),
        k3_bounds=(-0.49, 3.0),
        k4_bounds=(-0.49, 3.0),
        eps_p=1e-10,
        bisect_max_iter=60,
        bisect_tol=1e-12,
        pdf_pgrid_size=20000,
    )

    print("Fitted GLD parameters:")
    print("sigma =", fit.params.sigma)
    print("k3 =", fit.params.k3)
    print("k4 =", fit.params.k4)

    fine_strikes = np.linspace(25, 200, 200)
    q_hat = fit.qhat(fine_strikes)
    c_hat = fit.chat(fine_strikes)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    ax = axes[0]
    ax.plot(strikes, q_true, "k--", label="True Heston RND (coarse BL)", lw=2)
    ax.plot(fine_strikes, q_hat, label="GLD RND (fitted)", lw=2)
    ax.set_xlabel("Terminal price s")
    ax.set_ylabel("q(s)")
    ax.set_title("Risk-Neutral Density: Heston vs GLD (Corrado 2001)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(strikes, C_mkt, "o", ms=4, label="True Heston calls")
    ax2.plot(fine_strikes, c_hat, "-", lw=2, label="Fitted GLD calls")
    ax2.scatter(strikes, C_mkt_noise, label="Noisy Heston calls")
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Call price C(K)")
    ax2.set_title("Call Curve Fit: Heston vs GLD (Corrado 2001)")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
