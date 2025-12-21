import numpy as np
from dataclasses import dataclass
from typing import Tuple, Sequence, Optional
import matplotlib.pyplot as plt
from Heston_model import *  # assumes HestonParams, heston_call_prices_fast, etc.


# ============================================================
# 1. Spectral basis (Monnier 2013)
# ============================================================

@dataclass
class SpectralBasis:
    """
    Spectral basis for the restricted call/put operators on I = [0, B].

    Implements (phi_k), (psi_k), and singular values lambda_k, k=0,...,N
    following Monnier (2013), Section 4.
    """
    B: float
    N: int
    rho0: float = 1.875104069  # value used in the paper for k=0

    def __post_init__(self):
        self.rho = np.zeros(self.N + 1)
        self.lambdas = np.zeros(self.N + 1)

        for k in range(self.N + 1):
            if k == 0:
                rho_k = self.rho0
            else:
                # beta_k ≈ 2 exp(-π/2 - kπ)  (Monnier, p. 662)
                beta_k = 2.0 * np.exp(-np.pi / 2.0 - k * np.pi)
                rho_k = np.pi / 2.0 + k * np.pi + ((-1) ** k) * beta_k
            self.rho[k] = rho_k
            self.lambdas[k] = (self.B / rho_k) ** 2  # λ_k = (B / ρ_k)^2

    # ----- coefficient helpers -----

    def _coeffs(self, k: int) -> Tuple[float, float, float, float]:
        """
        Return (a_k1, a_k2, a_k3, a_k4) for given k, using numerically stable
        forms from equation (4.3) and subsequent lines.
        """
        B = self.B
        rho_k = self.rho[k]
        em = np.exp(-rho_k)  # e^{-ρ_k}, very small for k >= 1

        ak2 = 1.0 / (np.sqrt(B) * (1.0 + ((-1) ** k) * em))
        ak1 = ((-1) ** k) * em * ak2
        ak3 = -1.0 / np.sqrt(B)
        ak4 = (1.0 / np.sqrt(B)) * (1.0 - ((-1) ** k) * em) / (1.0 + ((-1) ** k) * em)

        return ak1, ak2, ak3, ak4

    # ----- basis evaluation -----

    def _fk(self, k: int, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (f_k1, f_k2, f_k3, f_k4) evaluated at x."""
        rho_k = self.rho[k]
        z = rho_k * x / self.B
        f1 = np.exp(z)
        f2 = np.exp(-z)
        f3 = np.cos(z)
        f4 = np.sin(z)
        return f1, f2, f3, f4

    def phi(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the singular vectors φ_k(x), k=0..N, at points x.

        Returns
        -------
        Phi : array, shape (len(x), N+1)
        """
        x = np.asarray(x, float)
        m = x.size
        Phi = np.empty((m, self.N + 1), float)

        for k in range(self.N + 1):
            ak1, ak2, ak3, ak4 = self._coeffs(k)
            f1, f2, f3, f4 = self._fk(k, x)
            h1 = ak1 * f1 + ak2 * f2
            h2 = ak3 * f3 + ak4 * f4
            Phi[:, k] = h1 + h2  # φ_k = h_{k,1} + h_{k,2}
        return Phi

    def psi(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the singular vectors ψ_k(x), k=0..N, at points x.

        Returns
        -------
        Psi : array, shape (len(x), N+1)
        """
        x = np.asarray(x, float)
        m = x.size
        Psi = np.empty((m, self.N + 1), float)

        for k in range(self.N + 1):
            ak1, ak2, ak3, ak4 = self._coeffs(k)
            f1, f2, f3, f4 = self._fk(k, x)
            h1 = ak1 * f1 + ak2 * f2
            h2 = ak3 * f3 + ak4 * f4
            Psi[:, k] = h1 - h2  # ψ_k = h_{k,1} - h_{k,2}
        return Psi


# ============================================================
# 2. Result container with convenience methods
# ============================================================

@dataclass
class SpectralRNDResult:
    s_grid: np.ndarray           # grid of underlying prices in [0,B]
    q_rnd: np.ndarray            # estimated RND q(s) on s_grid
    call_fit: np.ndarray         # fitted call prices at input strikes
    strikes: np.ndarray          # input strikes (sorted)
    call_mid: np.ndarray         # input calls (sorted, original)
    B: float
    lambdas: np.ndarray          # singular values λ_k
    omega: np.ndarray            # spectral coefficients ω_k
    basis: SpectralBasis
    inv_lambda: np.ndarray
    S0: float
    r: float
    q_div: float
    T: float
    F0: float
    N: int
    select_method: str
    fit_mse: float               # MSE between fitted and observed calls
    norm_const: float            # ∫ max(q_raw,0) ds used for normalization
    fit_weight: float            # regularization weight actually used
    clip_negative: bool = True

    # ---------- convenience evaluators ----------

    def calls(self, strikes_new: np.ndarray) -> np.ndarray:
        """
        Return fitted call prices C(K) at new strikes K (discounted, today).
        """
        K_new = np.asarray(strikes_new, float)
        Phi_new = self.basis.phi(K_new)                    # (m_new, N+1)
        P_tilde_new = Phi_new @ self.omega                # undiscounted puts
        C_tilde_new = P_tilde_new + (self.F0 - K_new)     # put-call parity
        C_new = np.exp(-self.r * self.T) * C_tilde_new
        return C_new

    def rnd(self, s_new: np.ndarray) -> np.ndarray:
        """
        Return estimated RND q(s) evaluated at new terminal-price grid s_new.
        """
        s_new = np.asarray(s_new, float)
        Psi_new = self.basis.psi(s_new)                   # (m_new, N+1)
        q_raw = Psi_new @ (self.inv_lambda * self.omega)
        if self.clip_negative:
            q_raw = np.maximum(q_raw, 0.0)
        return q_raw / self.norm_const


# ============================================================
# 3. Main function: N fixed, CV/oracle over regularization
# ============================================================

def spectral_rnd_from_call_mids(
    strikes: np.ndarray,
    call_mid: np.ndarray,
    S0: float,
    r: float,
    q_div: float,
    T: float,
    N: Optional[int] = None,           # if None, N = len(strikes)//2
    B: Optional[float] = None,
    s_grid_size: int = 400,
    fit_weight: float = 1e-2,          # default regularization if select_reg="fixed"
    select_reg: str = "fixed",         # "fixed", "cv_calls", or "oracle"
    reg_grid: Optional[Sequence[float]] = None,
    oracle_s: Optional[np.ndarray] = None,
    oracle_q: Optional[np.ndarray] = None,
) -> SpectralRNDResult:
    """
    Monnier-style spectral RND recovery using *call mid prices*.

    We:
      1) Convert calls to *undiscounted* puts using put–call parity.
      2) Express the undiscounted put function P̃(K) in the φ_k basis:
            P̃_N(K) = Σ_k ω_k φ_k(K)
      3) For given N, minimize
            J(ω; α) = (1/2) ωᵀ Ω⁴ ω  + (α/2) ||Aω - b||²
         with  A = Φ(K),  b = P̃,  Ω = diag(λ_k^{-1}), α = fit_weight.
         This trades off smoothness (first term) vs. call fit (second term).

    Regularization selection
    ------------------------
    select_reg = "fixed"   : use the provided fit_weight.
    select_reg = "cv_calls": search over reg_grid to minimize MSE between
                             observed and fitted calls.
    select_reg = "oracle"  : search over reg_grid to minimize integrated
                             squared error between q_est(s) and oracle_q(s)
                             given on grid oracle_s.

    N is *not* selected by CV here. By default we use:
        N = len(strikes) // 2

    Parameters
    ----------
    strikes : array_like
        Strike vector K_i (not necessarily sorted).
    call_mid : array_like
        Mid call prices C_i at those strikes.
    S0 : float
        Spot price today.
    r : float
        Continuously compounded risk-free rate.
    q_div : float
        Continuous dividend yield (or convenience yield).
    T : float
        Time to maturity (in years).
    N : int or None
        Spectral cutoff. If None, N = len(strikes)//2.
    B : float, optional
        Upper bound of price interval I=[0,B]. If None, set to 2*F0, or
        big enough to contain oracle_s if provided.
    s_grid_size : int
        Number of grid points for the returned RND.
    fit_weight : float
        Regularization α when select_reg == "fixed".
    select_reg : {"fixed", "cv_calls", "oracle"}
        Strategy for choosing α.
    reg_grid : sequence of float, optional
        Candidate α values for CV/oracle. If None, a default grid is used.
    oracle_s, oracle_q : arrays, optional
        True RND q_true(s) on grid oracle_s, used only when
        select_reg == "oracle".

    Returns
    -------
    SpectralRNDResult
    """

    # ---------- basic preprocessing ----------
    strikes = np.asarray(strikes, float)
    call_mid = np.asarray(call_mid, float)

    # sort by strike
    order = np.argsort(strikes)
    K = strikes[order]
    C_mid = call_mid[order]

    # Forward price
    F0 = S0 * np.exp((r - q_div) * T)

    # Work with *undiscounted* prices
    C_tilde = np.exp(r * T) * C_mid

    # Put–call parity (undiscounted):
    # C̃ - P̃ = F0 - K  =>  P̃ = C̃ - (F0 - K)
    P_tilde = C_tilde - (F0 - K)

    # Choose N if not given
    if N is None:
        N_eff = max(2, len(K) // 2)
    else:
        N_eff = int(N)

    # Interval [0,B]
    if B is None:
        B = 2.0 * F0
        if oracle_s is not None:
            B = max(B, float(np.max(oracle_s)))

    # grid for density output
    s_grid = np.linspace(0.0, B, s_grid_size)

    # Default regularization grid for selection
    if select_reg in ("cv_calls", "oracle") and reg_grid is None:
        # You can tweak this grid; 0.01 is often good, so we include it.
        reg_grid =  np.logspace(-6, 3, 100)

    # ---------- helper: solve for a given regularization α ----------

    def fit_for_reg(alpha: float, method_label: str) -> SpectralRNDResult:
        basis = SpectralBasis(B=B, N=N_eff)
        lambdas = basis.lambdas
        inv_lambda = 1.0 / lambdas

        # Design matrix: P̃_N(K_i) = Σ_k ω_k φ_k(K_i) = (Φ_K ω)_i
        Phi_K = basis.phi(K)  # shape (m, N+1)
        A = Phi_K
        b_vec = P_tilde

        # Quadratic penalty: H = Ω⁴ with Ω = diag(λ_k^{-1})
        Omega4_diag = inv_lambda ** 4
        H = np.diag(Omega4_diag)  # (N+1, N+1)

        # Tikhonov-type system: (H + α AᵀA) ω = α Aᵀ b
        M = H + alpha * (A.T @ A)
        rhs = alpha * (A.T @ b_vec)

        omega = np.linalg.solve(M, rhs)

        # Recover q_N(s) on grid
        Psi_grid = basis.psi(s_grid)  # shape (s_grid_size, N+1)
        q_raw = Psi_grid @ (inv_lambda * omega)
        q_raw = np.maximum(q_raw, 0.0)  # enforce non-negativity softly
        norm_const = np.trapz(q_raw, s_grid)
        if norm_const <= 0:
            norm_const = 1.0
        q_vals = q_raw / norm_const

        # Fitted calls at the input strikes (discounted)
        P_tilde_fit = A @ omega
        C_tilde_fit = P_tilde_fit + (F0 - K)
        C_fit = np.exp(-r * T) * C_tilde_fit

        mse = float(np.mean((C_fit - C_mid) ** 2))

        return SpectralRNDResult(
            s_grid=s_grid,
            q_rnd=q_vals,
            call_fit=C_fit,
            strikes=K,
            call_mid=C_mid,
            B=B,
            lambdas=lambdas,
            omega=omega,
            basis=basis,
            inv_lambda=inv_lambda,
            S0=S0,
            r=r,
            q_div=q_div,
            T=T,
            F0=F0,
            N=N_eff,
            select_method=method_label,
            fit_mse=mse,
            norm_const=norm_const,
            fit_weight=alpha,
            clip_negative=True,
        )

    # ---------- regularization selection logic ----------

    if select_reg == "fixed":
        # just use the provided fit_weight
        return fit_for_reg(float(fit_weight), "fixed")

    elif select_reg == "cv_calls":
        best_res: Optional[SpectralRNDResult] = None
        best_score = np.inf

        for alpha in reg_grid:
            res_cand = fit_for_reg(float(alpha), "cv_calls")
            score = res_cand.fit_mse
            if score < best_score:
                best_score = score
                best_res = res_cand

        return best_res

    elif select_reg == "oracle":
        if oracle_s is None or oracle_q is None:
            raise ValueError("For select_reg='oracle', provide oracle_s and oracle_q.")
        oracle_s_arr = np.asarray(oracle_s, float)
        oracle_q_arr = np.asarray(oracle_q, float)

        best_res: Optional[SpectralRNDResult] = None
        best_score = np.inf

        for alpha in reg_grid:
            res_cand = fit_for_reg(float(alpha), "oracle")
            # evaluate estimated RND on oracle grid
            q_est = res_cand.rnd(oracle_s_arr)
            diff = q_est - oracle_q_arr
            ise = float(np.trapz(diff ** 2, oracle_s_arr))  # integrated squared error
            if ise < best_score:
                best_score = ise
                best_res = res_cand

        return best_res

    else:
        raise ValueError("select_reg must be 'fixed', 'cv_calls', or 'oracle'.")


# ============================================================
# 4. Example usage
# ============================================================

if __name__ == "__main__":
    true_p = HestonParams(kappa=0.5, theta=0.05, sigma=0.25, v0=0.02, rho=-0.6)
    S0, r, q_div, T = 120.0, 0.02, 0.00, 0.5

    # strikes and "true" Heston calls
    strikes = np.linspace(50, 200, 100)
    C_mkt = heston_call_prices_fast(S0, strikes, r, q_div, T, true_p)

    # add a bit of noise
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.0005 * C_mkt)
    C_mkt_noise = C_mkt + noise

    # "true" RND on the strike grid via Breeden–Litzenberger (for oracle demo)
    dC_dK   = np.gradient(C_mkt, strikes, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK,  strikes, edge_order=2)
    q_true  = np.exp(r * T) * d2C_dK2
    
    dC_dK   = np.gradient(C_mkt_noise, strikes, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK,  strikes, edge_order=2)
    q_noisy  = np.exp(r * T) * d2C_dK2

    # ---------- fixed regularization ----------
    res_fixed = spectral_rnd_from_call_mids(
        strikes=strikes,
        call_mid=C_mkt_noise,
        S0=S0,
        r=r,
        q_div=q_div,
        T=T,
        select_reg="fixed",
        fit_weight=0.01,   # the value you found works nicely
    )

    # ---------- CV over regularization (fit to calls) ----------
    res_cv = spectral_rnd_from_call_mids(
        strikes=strikes,
        call_mid=C_mkt_noise,
        S0=S0,
        r=r,
        q_div=q_div,
        T=T,
        select_reg="cv_calls",
        reg_grid=[1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    )

    # ---------- Oracle selection over regularization ----------
    res_oracle = spectral_rnd_from_call_mids(
        strikes=strikes,
        call_mid=C_mkt_noise,
        S0=S0,
        r=r,
        q_div=q_div,
        T=T,
        select_reg="oracle",
        reg_grid=[1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        oracle_s=strikes,
        oracle_q=q_true,
    )

    # Evaluate RNDs on a fine grid for plotting
    s_fine = np.linspace(50, 200, 500)
    q_cv = res_cv.rnd(s_fine)
    q_or = res_oracle.rnd(s_fine)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    
    # -------------------------------------------------------------
    # Panel 1: Risk–Neutral Densities
    # -------------------------------------------------------------
    ax = axes[0]
    ax.plot(s_fine, q_cv, label="RND (CV over reg)", lw=2)
    ax.plot(s_fine, q_or, label="RND (oracle over reg)", lw=2)
    ax.plot(strikes, q_true, "k--", label="True (BL on call grid)", lw=2)
    
    ax.set_xlabel("Terminal price s")
    ax.set_ylabel("q(s)")
    ax.set_title("Risk-Neutral Density Estimates")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    
    # -------------------------------------------------------------
    # Panel 2: Call Prices (Observed vs Fitted)
    # -------------------------------------------------------------
    K_fine = np.linspace(50, 200, 200)
    C_fit_cv = res_cv.calls(K_fine)
    
    ax2 = axes[1]
    ax2.plot(strikes, C_mkt_noise, "o", ms=4, label="Noisy calls (input)")
    ax2.plot(K_fine, C_fit_cv, "-", lw=2, label="Fitted calls (CV reg)")
    
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Call price C(K)")
    ax2.set_title("Call Curve Fit")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
