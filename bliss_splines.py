import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.interpolate import BSpline
from scipy.stats import norm
from scipy.optimize import brentq
#from New_Simulation_density import*
import matplotlib.pyplot as plt
from Heston_model import *  # assumes HestonParams, heston_call_prices_fast, etc.

# -------------------------------------------------------------------
# Utilities: knots, design matrix, curvature penalty matrix
# -------------------------------------------------------------------

def make_uniform_knots(x, degree=3, n_internal=6):
    """
    Open uniform knot vector on [min(x), max(x)] with n_internal interior knots.
    """
    x = np.asarray(x)
    x_min, x_max = float(np.min(x)), float(np.max(x))

    # interior knots (exclude boundaries)
    if n_internal > 0:
        interior = np.linspace(x_min, x_max, n_internal + 2)[1:-1]
    else:
        interior = np.array([])

    # open uniform: repeat boundaries degree+1 times
    t = np.concatenate(
        [
            np.repeat(x_min, degree + 1),
            interior,
            np.repeat(x_max, degree + 1),
        ]
    )
    return t


def bspline_design_matrix(x, knots, degree):
    """
    Build B-spline design matrix B_{ij} = B_j(x_i)
    for a common knot vector and degree.
    """
    x = np.asarray(x)
    n_basis = len(knots) - degree - 1
    B = np.empty((x.shape[0], n_basis))

    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl_j = BSpline(knots, c, degree, extrapolate=True)
        B[:, j] = spl_j(x)

    return B


def curvature_penalty_matrix(knots, degree, n_grid=200):
    """
    Approximate R = ∫ B''(k)^T B''(k) dk with numerical quadrature.
    This yields the matrix for the roughness penalty c^T R c.
    """
    n_basis = len(knots) - degree - 1
    # use the interior support of the spline
    x_min, x_max = knots[degree], knots[-degree - 1]
    xs = np.linspace(x_min, x_max, n_grid)
    dx = (x_max - x_min) / (n_grid - 1)

    B2 = np.empty((n_grid, n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl_j = BSpline(knots, c, degree, extrapolate=False).derivative(2)
        B2[:, j] = spl_j(xs)

    # R ≈ ∫ B''(x)^T B''(x) dx ≈ (B2^T B2) * dx
    R = B2.T @ B2 * dx
    return R


# -------------------------------------------------------------------
# Data class for the fitted spline + result container
# -------------------------------------------------------------------

@dataclass
class SmoothingSplineIV:
    """Callable object representing the fitted IV spline."""
    bspline: BSpline
    alpha: float           # penalty weight in J = SSE + alpha * c'Rc
    lambda_reg: float      # equivalent lambda = 1 / (1 + alpha)

    def __call__(self, K):
        """Evaluate smoothed implied volatility at strikes K."""
        return self.bspline(np.asarray(K))

    def second_derivative(self, K):
        """Evaluate second derivative of the smoothed IV."""
        return self.bspline.derivative(2)(np.asarray(K))


@dataclass
class SplineIVResult:
    """
    High-level result, analogous to LLResultCalls.
    Stores everything you typically want to inspect.
    """
    method: str                     # 'cv' or 'oracle'
    smoother: SmoothingSplineIV     # callable IV spline
    strikes: np.ndarray             # data strikes (where IV was observed)
    iv_obs: np.ndarray              # observed/noisy IV at strikes
    iv_smooth: np.ndarray           # smoothed IV at strikes
    C_smooth: Optional[np.ndarray]  # smoothed calls at strikes (if S0,r,T given)
    rnd_grid: Optional[np.ndarray]  # grid for RND (None if not computed)
    rnd_est: Optional[np.ndarray]   # estimated RND on rnd_grid
    alpha: float                    # chosen alpha
    lambda_reg: float               # 1/(1+alpha)
    alpha_grid: Optional[np.ndarray]  # grid of alpha values
    error_curve: Optional[np.ndarray] # CV or oracle error at each alpha

    # convenience helpers
    def iv(self, K):
        """Evaluate smoothed IV at any strikes K."""
        return self.smoother(K)

    def calls(self, K, S0, r, q, T):
        """Evaluate smoothed call prices at any strikes K."""
        iv_vals = self.smoother(K)
        K = np.asarray(K, float)
        return np.array([bs_call_disc(S0, k, r, q, T, sig)
                         for k, sig in zip(K, iv_vals)])

    def rnd(self, K_grid, S0, r, q, T):
        """Compute RND on any grid from the stored spline."""
        return rnd_from_iv_spline(self.smoother, S0, r, q, T, K_grid)


# -------------------------------------------------------------------
# Black–Scholes (discounted) and IV
# -------------------------------------------------------------------

def bs_call_disc(S0, K, r, q, T, sigma):
    """
    Black–Scholes discounted call price C(0,K) with continuous r, q.
    """
    if sigma <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_iv_from_price_disc(S0, K, r, q, T, price, lo=1e-6, hi=5.0):
    """
    Invert discounted BS call price to get implied vol using brentq.
    """
    lower = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    upper = S0 * np.exp(-q * T)
    target = float(np.clip(price, lower, upper))

    def f(vol):
        return bs_call_disc(S0, K, r, q, T, vol) - target

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


# -------------------------------------------------------------------
# RND from an IV spline (Breeden–Litzenberger)
# -------------------------------------------------------------------

def rnd_from_iv_spline(spline_iv: SmoothingSplineIV,
                       S0: float, r: float, q: float, T: float,
                       K_grid: np.ndarray):
    """
    Compute risk-neutral density q(K) = exp(rT) * d^2 C(0,K) / dK^2,
    where C(0,K) is the discounted call price built from the IV spline.
    """
    K_grid = np.asarray(K_grid, dtype=float)
    iv_grid = spline_iv(K_grid)

    C_grid = np.array([bs_call_disc(S0, k, r, q, T, sig)
                       for k, sig in zip(K_grid, iv_grid)])
    dC_dK = np.gradient(C_grid, K_grid, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, K_grid, edge_order=2)

    qK = np.exp(r * T) * d2C_dK2
    # clamp small numerical negatives
    qK = np.maximum(qK, 0.0)
    return qK


# -------------------------------------------------------------------
# Main smoother + CV + ORACLE OPTION (returns SplineIVResult)
# -------------------------------------------------------------------

def fit_iv_smoothing_spline(
    strikes,
    iv,
    weights=None,
    degree=3,
    n_internal_knots=None,
    alpha_grid=None,
    n_folds=10,
    random_state=42,
    # ----- ORACLE options -----
    oracle=False,
    true_strikes=None,
    true_rnd=None,
    S0=None,
    r=None,
    q=0.0,
    T=None,
    # where to output RND (if None and S0,r,T given, use strikes)
    rnd_strikes=None,
    return_diagnostics=False,   # kept for compatibility; now always stored
):
    """
    Fit a penalized smoothing spline to implied volatilities.

    Returns a SplineIVResult instance containing:
      - smoother (SmoothingSplineIV)
      - iv_obs, iv_smooth
      - C_smooth (if S0,r,T provided)
      - rnd_grid, rnd_est (if S0,r,T provided)
      - alpha, lambda_reg, alpha_grid, error_curve
    """

    strikes = np.asarray(strikes, dtype=float)
    iv = np.asarray(iv, dtype=float)
    N = strikes.shape[0]

    if weights is None:
        weights = np.ones(N, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # Choose default # of interior knots based on sample size
    if n_internal_knots is None:
        n_internal_knots = int(np.clip(N // 4, 3, 10))

    # Build knots, design matrix, and curvature penalty
    knots = make_uniform_knots(strikes, degree=degree, n_internal=n_internal_knots)
    B = bspline_design_matrix(strikes, knots, degree)
    R = curvature_penalty_matrix(knots, degree, n_grid=200)

    # Penalty grid (on alpha = (1 - lambda) / lambda)
    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 9, 200)

    BW_full = B * weights[:, None]
    all_idx = np.arange(N)

    def fit_coeffs(alpha, idx_train):
        """Solve (B' W B + alpha R) c = B' W y on a subset."""
        Bt = B[idx_train]
        wt = weights[idx_train]
        yt = iv[idx_train]

        BW = Bt * wt[:, None]
        A = BW.T @ Bt + alpha * R
        b = BW.T @ yt
        return np.linalg.solve(A, b)

    # ===== ORACLE MODE =====
    if oracle:
        if (true_strikes is None) or (true_rnd is None) or \
           (S0 is None) or (r is None) or (T is None):
            raise ValueError("Oracle mode requires true_strikes, true_rnd, S0, r, T.")

        true_strikes = np.asarray(true_strikes, dtype=float)
        true_rnd = np.asarray(true_rnd, dtype=float)

        oracle_errors = []

        for alpha in alpha_grid:
            # fit on ALL data (no CV)
            c_hat = fit_coeffs(alpha, all_idx)
            bspline = BSpline(knots, c_hat, degree, extrapolate=True)
            spline_tmp = SmoothingSplineIV(bspline=bspline,
                                           alpha=alpha,
                                           lambda_reg=1.0 / (1.0 + alpha))

            # get RND from this spline at true_strikes
            q_hat = rnd_from_iv_spline(spline_tmp, S0, r, q, T, true_strikes)

            # RND oracle loss: mean squared error
            mse_alpha = np.mean((q_hat - true_rnd) ** 2)
            oracle_errors.append(mse_alpha)

        error_curve = np.asarray(oracle_errors)
        best_idx = int(np.argmin(error_curve))
        alpha_best = float(alpha_grid[best_idx])
        lambda_best = 1.0 / (1.0 + alpha_best)
        method = "oracle"

    # ===== STANDARD CV MODE =====
    else:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(N)
        folds = np.array_split(indices, n_folds)

        cv_errors = []

        for alpha in alpha_grid:
            fold_mse = []

            for val_idx in folds:
                train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=True)

                c_hat = fit_coeffs(alpha, train_idx)

                # Predict on validation fold
                B_val = B[val_idx]
                y_val = iv[val_idx]
                y_pred = B_val @ c_hat

                fold_mse.append(np.mean((y_val - y_pred) ** 2))

            cv_errors.append(np.mean(fold_mse))

        error_curve = np.asarray(cv_errors)
        best_idx = int(np.argmin(error_curve))
        alpha_best = float(alpha_grid[best_idx])
        lambda_best = 1.0 / (1.0 + alpha_best)
        method = "cv"

    # Refit on full data using best alpha
    A_full = BW_full.T @ B + alpha_best * R
    b_full = BW_full.T @ iv
    coeffs = np.linalg.solve(A_full, b_full)

    bspline = BSpline(knots, coeffs, degree, extrapolate=True)
    smoother = SmoothingSplineIV(bspline=bspline,
                                 alpha=alpha_best,
                                 lambda_reg=lambda_best)

    # ------------------ Store IV + calls at data strikes ------------------
    iv_smooth = smoother(strikes)
    C_smooth = None
    if (S0 is not None) and (r is not None) and (T is not None):
        C_smooth = np.array([bs_call_disc(S0, k, r, q, T, sig)
                             for k, sig in zip(strikes, iv_smooth)])

    # ------------------ Compute estimated RND ------------------
    rnd_grid = None
    rnd_est = None
    if (S0 is not None) and (r is not None) and (T is not None):
        # default: use strikes if no explicit rnd_strikes given
        if rnd_strikes is None:
            rnd_grid = strikes.copy()
        else:
            rnd_grid = np.asarray(rnd_strikes, dtype=float)

        rnd_est = rnd_from_iv_spline(smoother, S0, r, q, T, rnd_grid)

    # ------------------ Build result object ------------------
    result = SplineIVResult(
        method=method,
        smoother=smoother,
        strikes=strikes,
        iv_obs=iv,
        iv_smooth=iv_smooth,
        C_smooth=C_smooth,
        rnd_grid=rnd_grid,
        rnd_est=rnd_est,
        alpha=alpha_best,
        lambda_reg=lambda_best,
        alpha_grid=alpha_grid,
        error_curve=error_curve,
    )

    return result

#demo
from New_Simulation_density import*  
if __name__ == "__main__":

    true_p = HestonParams(kappa=0.5, theta=0.05, sigma=0.25, v0=0.02, rho=-0.6)
    S0, r, q, T = 120.0, 0.02, 0.00, 0.5
    strikes = np.linspace(75, 180, 80)
    
        # “Market” prices (fast Heston pricer from your Heston_model module)
    C_mkt = heston_call_prices_fast(S0, strikes, r, q, T, true_p)
    #rng = np.random.default_rng(0)
    #noise = rng.normal(scale=0.0005 * C_mkt)
    
    noise_dict={'mode': 'bondarenko', 'scale': 0.005}
    C_mkt_noise=noisy_data_function(strikes,C_mkt,S0,noise_dict)
    
    
    #C_mkt_noise = C_mkt + noise
    
    
    dC_dK   = np.gradient(C_mkt, strikes, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK,  strikes, edge_order=2)
    q_K     = np.exp(r * T) * d2C_dK2
    
    dC_dKn   = np.gradient(C_mkt_noise, strikes, edge_order=2)
    d2C_dK2n = np.gradient(dC_dKn,  strikes, edge_order=2)
    q_Kn     = np.exp(r * T) * d2C_dK2n
        
    
    
    iv_heston_noisy = np.array([bs_iv_from_price_disc(S0, K, r, q, T, C)
                              for K, C in zip(strikes, C_mkt_noise)])
    iv_heston = np.array([bs_iv_from_price_disc(S0, K, r, q, T, C)
                              for K, C in zip(strikes, C_mkt)])
    
    #plt.plot(strikes,iv_heston_noisy)
    
    
    # True RND (analytical for BS lognormal)
    def bs_rnd(K, S0, r, q, T, sigma):
        mu_ln = np.log(S0) + (r - q - 0.5*sigma**2)*T
        return np.exp(-0.5*((np.log(K)-mu_ln)/sigma/np.sqrt(T))**2)/(K*sigma*np.sqrt(2*np.pi*T))
    
    
    
    # Fit 3 methods
    s_cv= fit_iv_smoothing_spline(
        strikes=strikes,
        iv=iv_heston_noisy,
        degree=3,           # or 4
        S0=S0,
        r=r,
        q=q,
        T=T,
        n_folds=10,
        alpha_grid=np.logspace(-3,8,100)
    )
    
    s_cv10= fit_iv_smoothing_spline(
        strikes=strikes,
        iv=iv_heston_noisy,
        degree=3,           # or 4
        S0=S0,
        r=r,
        q=q,
        T=T,
        n_folds=10,
        alpha_grid=np.logspace(-3,8,100)

    )
    
    s_oracle= fit_iv_smoothing_spline(
        strikes=strikes,
        iv=iv_heston_noisy,
        degree=3,           # or 4
        oracle=True,
        true_strikes=strikes,  # where true_rnd is given
        true_rnd=q_K,
        S0=S0,
        r=r,
        q=q,
        T=T,
        alpha_grid=np.logspace(-3,8,100)

    )
    #iv_fit_oracle = s_oracle.iv_smooth
    #iv_fit = s_cv.iv_smooth

    # ---- Plot RND comparison ----
    F_T = S0 * np.exp(r * T)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # ---------------- Left: RNDs ----------------
    ax = axes[0]
    ax.plot(strikes, q_K,           linestyle='-',  linewidth=2,  label="True RND")
    ax.plot(strikes, s_oracle.rnd_est, linestyle='--', linewidth=1.8, label="Spline (oracle h)")
   # ax.plot(strikes, s_cv.rnd_est,     linestyle='-.', linewidth=1.8, label="Spline 5(CV h)")
    ax.plot(strikes, s_cv10.rnd_est,     linestyle='-.', linewidth=1.8, label="Spline 10(CV h)")
    #ax.plot(strikes, q_Kn,     linestyle='-.', linewidth=1.8, label="Noisy Rnd")

    ax.axvline(F_T, linestyle=':', linewidth=1.5, label=r"Forward $F_T$")
    
    ax.set_xlabel(r"Terminal price $s = K$")
    ax.set_ylabel(r"$q(s)$")
    ax.set_title("Risk-Neutral Densities")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # ---------------- Right: Call prices ----------------
    ax2 = axes[1]
    ax2.plot(strikes, C_mkt,              linestyle='-',  linewidth=2,  label="Market calls")
    ax2.plot(strikes, s_oracle.C_smooth,  linestyle='--', linewidth=1.8, label="Spline (oracle h)")
    #ax2.plot(strikes, s_cv.C_smooth,      linestyle='-.', linewidth=1.8, label="Spline (CV h)")
    ax2.plot(strikes, s_cv10.C_smooth,      linestyle='-.', linewidth=1.8, label="Spline (CV h10)")

    ax2.set_xlabel("Strike K")
    ax2.set_ylabel("Call price C(K)")
    ax2.set_title("Call Price Fits")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    ####

    error_curve = s_oracle.error_curve          # CV or oracle MSE(α)
    error_curve_cv = s_cv.error_curve          # CV or oracle MSE(α)
    error_curve_cv10 = s_cv10.error_curve          # CV or oracle MSE(α)

    alpha_grid = s_cv10.alpha_grid     # grid of α values
    alpha_grid_oracle = s_oracle.alpha_grid     # grid of α values


    best_alpha = s_oracle.alpha 
    best_alpha_cv = s_cv.alpha     
    best_alpha_cv10 = s_cv10.alpha     

    plt.figure(figsize=(7,5))
    
    # plot the CV curve
    plt.scatter(alpha_grid, error_curve,  label="Oracle Error")
    plt.scatter(alpha_grid, error_curve_cv10,  label="10-fold CV Error")
    #plt.scatter(alpha_grid_oracle, error_curve,  label="Oracle Error 10")


    # highlight the optimal alpha
    #plt.axvline(best_alpha_cv, color='red', linestyle='--', label=f"Best αcv5 = {best_alpha_cv:.2e}")
    plt.axvline(best_alpha_cv10, color='green', linestyle='--', label=f"Best α CV= {best_alpha_cv10:.2e}")
    plt.axvline(best_alpha, color='blue', linestyle='--', label=f"Best α oracle= {best_alpha:.2e}")

    
    plt.xscale("log")        # α grid is logarithmic → easier to see
    plt.yscale("log")        # α grid is logarithmic → easier to see

    plt.xlabel("α (log scale)")
    plt.ylabel("Error (log scale)")
    plt.title("Cross-Validation Diagnostics for Smoothing Spline IV")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(best_alpha_cv)
    print(best_alpha_cv10)

    print(best_alpha)
    

        


