import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from scipy.stats import norm

# import mixture tools
from Mixture_LWD import MixtureSpec, fit_mixture_to_calls, mixture_pdf

# ---------------------- Local linear components ----------------------
def _gauss(u):
    return np.exp(-0.5*u*u) / np.sqrt(2*np.pi)

def _ll_moments(x0, K, C, h):
    u = K - x0
    w = _gauss(u / h)
    S0 = np.sum(w)
    S1 = np.sum(u * w)
    S2 = np.sum((u**2) * w)
    T0 = np.sum(C * w)
    T1 = np.sum(C * u * w)
    denom = S2 * S0 - S1 * S1
    return S0, S1, S2, T0, T1, denom

def ll_beta0_at(x0, K, C, h):
    S0, S1, S2, T0, T1, denom = _ll_moments(x0, K, C, h)
    if denom <= 1e-18 or S0 <= 1e-18:
        return np.nan
    return float((S2*T0 - S1*T1) / denom)

def ll_beta1_at(x0, K, C, h):
    S0, S1, S2, T0, T1, denom = _ll_moments(x0, K, C, h)
    if denom <= 1e-18 or S0 <= 1e-18:
        return np.nan
    return float((S0*T1 - S1*T0) / denom)

def call_fit_curve(K_eval, K, C, h):
    C_hat = np.array([ll_beta0_at(x0, K, C, h) for x0 in K_eval], float)
    if np.isnan(C_hat).any():
        m = np.isfinite(C_hat)
        C_hat[~m] = np.interp(K_eval[~m], K_eval[m], C_hat[m])
    return C_hat

def beta1_curve(K_eval, K, C, h):
    b1 = np.array([ll_beta1_at(x0, K, C, h) for x0 in K_eval], float)
    if np.isnan(b1).any():
        m = np.isfinite(b1)
        b1[~m] = np.interp(K_eval[~m], K_eval[m], b1[m])
    return b1

def q_from_beta1(K_eval, beta1, r, T):
    x = np.asarray(K_eval, float)
    dx = np.mean(np.diff(x))
    C2 = np.gradient(beta1, dx)
    q = np.exp(r*T) * np.maximum(C2, 0)
    # normalize to integrate to 1
    q /= np.trapz(q, x)
    return x, q

# ---------------------- CV bandwidth ----------------------
def cv_bandwidth_on_calls(K, C, h_grid=None, kfold=5, rng=0, sample_weights=None):
    if h_grid is None:
        diffs = np.diff(np.sort(K))
        avg_spacing = float(np.mean(diffs))
        h_min = max(1.0 * avg_spacing, 1e-6)
        h_max = max(0.30*(np.max(K)-np.min(K)), h_min*1.01)
        h_grid = np.linspace(h_min, h_max, 61)
    rng = np.random.default_rng(rng)
    idx = np.arange(len(K))
    folds = np.array_split(rng.permutation(idx), kfold)

    def fold_loss(h):
        loss = 0.0
        for f in range(kfold):
            val = folds[f]
            trn = np.setdiff1d(idx, val)
            K_tr, C_tr = K[trn], C[trn]
            K_val, C_val = K[val], C[val]
            C_hat = call_fit_curve(K_val, K_tr, C_tr, h)
            w = np.ones_like(C_val) if sample_weights is None else sample_weights[val]
            loss += np.nanmean(w * (C_val - C_hat)**2)
        return loss / kfold

    losses = np.array([fold_loss(h) for h in h_grid])
    h_star = float(h_grid[np.nanargmin(losses)])
    return h_star, h_grid, losses

# ---------------------- Aït-Sahalia global bandwidth (Eq. 3.23) ----------------------
def global_bandwidth_AitSahalia(K, C, p=1, kernel_const=0.776):
    n = len(K)
    omega = np.ones_like(K)
    stdK = np.std(K)
    h_pilot = stdK * n**(-1.0 / (2*p + 3))
    C_hat = call_fit_curve(K, K, C, h_pilot)
    ssr = np.sum((C - C_hat)**2)
    # second derivative wrt strike
    C2 = np.gradient(np.gradient(C_hat, K), K)
    num = ssr * np.trapz(omega, K)
    denom = np.sum((C2**2) * omega)
    h_global = kernel_const * ((num / denom) * (1.0/n))**(1.0/(2*p+3))
    return float(h_global), ssr, h_pilot

# ---------------------- Oracle (CV to true RND) ----------------------
def oracle_bandwidth_on_rnd(K, C, r, T, q_true, K_true, h_grid=None):
    """Select h by minimizing MSE to the true RND."""
    if h_grid is None:
        diffs = np.diff(np.sort(K))
        avg_spacing = float(np.mean(diffs))
        h_min = max(1.0 * avg_spacing, 1e-6)
        h_max = max(0.30*(np.max(K)-np.min(K)), h_min*1.01)
        h_grid = np.linspace(h_min, h_max, 61)

    losses = []
    for h in h_grid:
        b1 = beta1_curve(K_true, K, C, h)
        _, q_est = q_from_beta1(K_true, b1, r, T)
        mse = np.mean((q_est - q_true)**2)
        losses.append(mse)
    h_star = float(h_grid[np.nanargmin(losses)])
    return h_star, h_grid, np.array(losses)

# ---------------------- Mixture-based bandwidth (new) ----------------------
def mixture_bandwidth_on_rnd_from_calls(
    K: np.ndarray,
    C_disc: np.ndarray,
    r: float,
    T: float,
    S0: float,
    h_grid: Optional[np.ndarray] = None,
    n_lognormal: int = 2,
    n_weibull: int = 0,
    eval_grid_size: int = 200,
    penalty_lambda: float = 0.0,
    random_starts: int = 1,
    seed: int = 123,
    var_c: float = 0.1,
    var_penalty: float = 1e4,
):
    """
    1) Fit a mixture model (default: 2 lognormals, 0 Weibulls) to discounted calls C_disc.
    2) Treat the resulting mixture pdf as 'true' RND.
    3) For each h in h_grid, compute local-linear RND q_h and choose h that
       minimizes MSE(q_h, q_mix) on a common grid.

    Returns
    -------
    h_star : float
        Best bandwidth.
    h_grid : np.ndarray
        Grid of candidate bandwidths used.
    losses : np.ndarray
        MSE values for each h in h_grid.
    """
    K = np.asarray(K, float)
    C_disc = np.asarray(C_disc, float)

    # default h_grid similar to cv_bandwidth_on_calls
    if h_grid is None:
        diffs = np.diff(np.sort(K))
        avg_spacing = float(np.mean(diffs))
        h_min = max(1.0 * avg_spacing, 1e-6)
        h_max = max(0.30 * (np.max(K) - np.min(K)), h_min * 1.01)
        h_grid = np.linspace(h_min, h_max, 61)
    else:
        h_grid = np.asarray(h_grid, float)

    # 1) Fit mixture (2 lognormals by default)
    spec = MixtureSpec(n_lognormal=n_lognormal, n_weibull=n_weibull)

    fit_mix, _, _ = fit_mixture_to_calls(
        K=K,
        C_mkt=C_disc,
        S0=S0,
        r=r,
        T=T,
        spec=spec,
        theta0=None,
        penalty_lambda=penalty_lambda,
        random_starts=random_starts,
        seed=seed,
        rnd_true=None,
        k_true=None,
        var_c=var_c,
        var_penalty=var_penalty,
        return_theta=False,
    )

    # 2) Build evaluation grid and mixture 'true' density
    K_true = np.linspace(K.min(), K.max(), eval_grid_size)
    q_mix = mixture_pdf(K_true, fit_mix.weights, fit_mix.types, fit_mix.params)
    # normalize mixture pdf on this grid to be safe
    q_mix /= np.trapz(q_mix, K_true)

    # 3) For each h, compute local-linear RND and MSE vs q_mix
    losses = []
    for h in h_grid:
        b1 = beta1_curve(K_true, K, C_disc, h)
        _, q_est = q_from_beta1(K_true, b1, r, T)
        mse = np.mean((q_est - q_mix)**2)
        losses.append(mse)

    losses = np.asarray(losses, float)
    h_star = float(h_grid[np.nanargmin(losses)])

    return h_star, h_grid, losses

# ---------------------- Weighted KDE on RND (Scott's rule) ----------------------
def weighted_kde_scott(x, weights=None, grid=None):
    """
    Weighted Gaussian KDE with Scott's rule bandwidth.
    x      : support points (e.g. K_eval)
    weights: nonnegative weights (e.g. q_fit at those points), need not sum to 1
    grid   : evaluation grid (defaults to x)
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 points for KDE.")

    if weights is None:
        w = np.ones_like(x, float)
    else:
        w = np.asarray(weights, float)
    w = np.maximum(w, 0.0)
    if not np.any(w > 0):
        raise ValueError("All weights are zero or negative in KDE.")
    # normalize weights
    w = w / np.sum(w)

    # weighted mean and std
    mean = np.sum(w * x)
    var = np.sum(w * (x - mean)**2)
    std = np.sqrt(max(var, 1e-12))

    # Scott's rule bandwidth for 1D
    h = std * n**(-1.0 / 5.0)
    if h <= 0:
        h = (x.max() - x.min()) * 1e-3 or 1.0

    if grid is None:
        grid = x
    grid = np.asarray(grid, float)

    # Evaluate KDE: f(z) = sum_i w_i * phi((z - x_i)/h) / h
    z = grid[:, None]  # (M,1)
    diff = (z - x[None, :]) / h  # (M,n)
    dens = np.dot(_gauss(diff), w) / h  # (M,)

    # Renormalize to integrate to 1 on the grid
    dens /= np.trapz(dens, grid)

    return grid, dens, h

def weighted_kde_with_bandwidth(x, weights, grid, h):
    """
    Gaussian KDE with user-specified bandwidth h.
    """
    x = np.asarray(x, float)
    grid = np.asarray(grid, float)

    if weights is None:
        w = np.ones_like(x, float)
    else:
        w = np.asarray(weights, float)
    w = np.maximum(w, 0.0)
    w = w / np.sum(w)

    z = grid[:, None]
    diff = (z - x[None, :]) / h
    dens = np.dot(_gauss(diff), w) / h
    dens /= np.trapz(dens, grid)
    return dens

# ---------------------- Price calls from a density ----------------------
def calls_from_density(K, x_grid, q_grid, r, T):
    """
    Compute discounted call prices C(K) = e^{-rT} ∫ (S-K)^+ q(S) dS
    from a density q_grid on x_grid.
    """
    K = np.asarray(K, float)
    xg = np.asarray(x_grid, float)
    qg = np.asarray(q_grid, float)

    C_undisc = []
    for K_i in K:
        payoff = np.maximum(xg - K_i, 0.0)
        C_undisc.append(np.trapz(payoff * qg, xg))
    C_undisc = np.array(C_undisc)
    C_disc = np.exp(-r*T) * C_undisc
    return C_disc

# ---------------------- Combined driver ----------------------
@dataclass
class LLResultCalls:
    h_used: float
    method: str
    h_grid: np.ndarray
    cv_mse_vals: np.ndarray
    K_eval: np.ndarray       # grid for local-linear RND
    q_fit: np.ndarray        # local-linear RND on K_eval
    C_hat_at_data: np.ndarray  # smoothed calls at input strikes
    ssr: float = np.nan
    h_pilot: float = np.nan
    # KDE-smoothed RND and corresponding calls
    K_kde: Optional[np.ndarray] = None
    q_kde: Optional[np.ndarray] = None
    C_from_kde: Optional[np.ndarray] = None
    h_kde: float = np.nan

def fit_ll_rnd_on_calls(strikes, C_disc, r, T,
                        bandwidth_method="cv",
                        h_grid=None, eval_points=1,
                        sample_weights=None, kfold=5, rng=0,
                        q_true=None, K_true=None,
                        use_kde=False,
                        kde_scale: float = 1.0,
                        # mixture bandwidth options
                        S0_for_mixture: Optional[float] = None,
                        mixture_n_lognormal: int = 2,
                        mixture_n_weibull: int = 0,
                        mixture_eval_grid_size: int = 200,
                        mixture_penalty_lambda: float = 0.0,
                        mixture_random_starts: int = 5,
                        mixture_var_c: float = 0.1,
                        mixture_var_penalty: float = 1e4):
    """
    Fit local-linear RND from call prices with different bandwidth selection rules.

    bandwidth_method ∈ {"cv", "ait_sahalia", "oracle", "mixture"}.

    If use_kde is True:
        - Compute Scott's rule bandwidth h_scott from the LL RND grid.
        - Then set h_kde = h_scott / kde_scale.
          (So kde_scale=1 -> Scott; kde_scale=6 -> h = h_scott / 6.)
    """
    # For now: evaluate RND on same grid as strikes
    eval_points = len(strikes)

    # ----- choose bandwidth h_used for local linear calls/RND -----
    if bandwidth_method == "cv":
        h_used, h_grid_built, cv_vals = cv_bandwidth_on_calls(
            strikes, C_disc, h_grid=h_grid, kfold=kfold, rng=rng, sample_weights=sample_weights
        )
        ssr = np.nan; h_pilot = np.nan

    elif bandwidth_method == "ait_sahalia":
        h_used, ssr, h_pilot = global_bandwidth_AitSahalia(strikes, C_disc)
        h_grid_built = np.array([h_used]); cv_vals = np.array([])

    elif bandwidth_method == "oracle":
        if q_true is None or K_true is None:
            raise ValueError("q_true and K_true must be provided for oracle bandwidth.")
        h_used, h_grid_built, cv_vals = oracle_bandwidth_on_rnd(
            strikes, C_disc, r, T, q_true, K_true, h_grid
        )
        ssr = np.nan; h_pilot = np.nan

    elif bandwidth_method == "mixture":
        if S0_for_mixture is None:
            raise ValueError("S0_for_mixture must be provided when bandwidth_method='mixture'.")
        h_used, h_grid_built, cv_vals = mixture_bandwidth_on_rnd_from_calls(
            K=strikes,
            C_disc=C_disc,
            r=r,
            T=T,
            S0=S0_for_mixture,
            h_grid=h_grid,
            n_lognormal=mixture_n_lognormal,
            n_weibull=mixture_n_weibull,
            eval_grid_size=mixture_eval_grid_size,
            penalty_lambda=mixture_penalty_lambda,
            random_starts=mixture_random_starts,
            seed=rng,
            var_c=mixture_var_c,
            var_penalty=mixture_var_penalty,
        )
        ssr = np.nan; h_pilot = np.nan

    else:
        raise ValueError("bandwidth_method must be 'cv', 'ait_sahalia', 'oracle', or 'mixture'")

    # ----- local-linear RND from call prices -----
    K_eval = np.linspace(strikes.min(), strikes.max(), eval_points)
    b1 = beta1_curve(K_eval, strikes, C_disc, h_used)
    Kq, q_fit = q_from_beta1(K_eval, b1, r, T)
    C_hat = call_fit_curve(strikes, strikes, C_disc, h_used)

    # ---------------- KDE PART OPTIONAL ----------------
    if use_kde:
        # make sure scale is sensible
        if kde_scale is None or kde_scale <= 0:
            kde_scale = 1.0

        # get Scott's bandwidth first (and ignore its own density)
        _, _, h_scott = weighted_kde_scott(Kq, weights=q_fit, grid=Kq)
        h_kde = h_scott / float(kde_scale)

        # now recompute KDE with the scaled bandwidth
        q_kde = weighted_kde_with_bandwidth(Kq, q_fit, Kq, h_kde)
        q_kde = q_kde / np.trapz(q_kde, Kq)   # renormalize just in case
        C_from_kde = calls_from_density(strikes, Kq, q_kde, r, T)
        K_kde = Kq
    else:
        K_kde = None
        q_kde = None
        C_from_kde = None
        h_kde = np.nan
    # ----------------------------------------------------

    return LLResultCalls(
        h_used=h_used,
        method=bandwidth_method,
        h_grid=h_grid_built,
        cv_mse_vals=cv_vals,
        K_eval=Kq,
        q_fit=q_fit,
        C_hat_at_data=C_hat,
        ssr=ssr,
        h_pilot=h_pilot,
        K_kde=K_kde,
        q_kde=q_kde,
        C_from_kde=C_from_kde,
        h_kde=h_kde,
    )

# ---------------------- Black–Scholes call (for demos/tests) ----------------------
def bs_call_price(S0, K, r, q, T, sigma):
    K = np.asarray(K, float)
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# ============================================================
# Demo / test harness
# ============================================================

if __name__ == "__main__":
    S0, r, q, T = 120.0, 0.02, 0.00, 0.5
    strikes = np.linspace(25, 250, 75)
    
    sigma = 0.25
    C_mkt = bs_call_price(S0, strikes, r, q, T, sigma)
    
    # True RND (analytical for BS lognormal)
    def bs_rnd(K, S0, r, q, T, sigma):
        mu_ln = np.log(S0) + (r - q - 0.5*sigma**2)*T
        return np.exp(-0.5*((np.log(K)-mu_ln)/sigma/np.sqrt(T))**2)/(K*sigma*np.sqrt(2*np.pi*T))
    
    K_true = np.linspace(25, 250, 50)
    q_true = bs_rnd(K_true, S0, r, q, T, sigma)
    q_true /= np.trapz(q_true, K_true)
    
    # Fit 4 methods
    res_cv = fit_ll_rnd_on_calls(strikes, C_mkt, r, T,
                                 bandwidth_method="cv", kfold=3, rng=42,
                                 use_kde=True)
    res_as = fit_ll_rnd_on_calls(strikes, C_mkt, r, T,
                                 bandwidth_method="ait_sahalia")
    res_oracle = fit_ll_rnd_on_calls(strikes, C_mkt, r, T,
                                     bandwidth_method="oracle",
                                     q_true=q_true, K_true=K_true,S0_for_mixture=S0)
    res_mix = fit_ll_rnd_on_calls(strikes, C_mkt, r, T,
                                  bandwidth_method="mixture",
                                  S0_for_mixture=S0,
)
    
    print(f"CV h        = {res_cv.h_used:.4f}")
    print(f"Aït-Sahalia h = {res_as.h_used:.4f}")
    print(f"Oracle h    = {res_oracle.h_used:.4f}")
    print(f"Mixture h   = {res_mix.h_used:.4f}")
    
    # ---- Plot RND comparison ----
    plt.figure(figsize=(7,5))
    plt.plot(K_true, q_true, 'k-', lw=2, label="True RND")
    plt.plot(res_cv.K_eval, res_cv.q_fit, label=f"Local-linear (CV) h={res_cv.h_used:.3f}")
    plt.plot(res_as.K_eval, res_as.q_fit, label=f"Aït-Sahalia h={res_as.h_used:.3f}")
    plt.plot(res_mix.K_eval, res_mix.q_fit, label=f"Mixture h={res_mix.h_used:.3f}")
    if res_cv.q_kde is not None:
        plt.plot(res_cv.K_eval, res_cv.q_kde, label=f"KDE (Scott)")
    plt.axvline(S0*np.exp(r*T), ls='--', label="Forward F_T")
    plt.xlabel("Terminal price s = K")
    plt.ylabel("q(s)")
    plt.title("RND Estimates under Different Bandwidth Selection Rules")
    plt.legend()
    plt.tight_layout()
    plt.show()
