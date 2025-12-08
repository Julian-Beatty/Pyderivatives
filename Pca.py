# import numpy as np
# import matplotlib.pyplot as plt
# from dataclasses import dataclass
# from typing import Optional, Sequence, Tuple
# from scipy.stats import norm
# from scipy.optimize import lsq_linear
# from scipy.interpolate import UnivariateSpline
# from Heston_model import *  # assumes HestonParams, heston_call_prices_fast, etc.

# # ---------- LPCA building blocks (lognormal kernels) ----------

# def ln_pdf(s, mu, sig):
#     out = np.zeros_like(s)
#     m = s > 0
#     sp = s[m]
#     out[m] = (1.0/(sp*sig*np.sqrt(2*np.pi))) * np.exp(-0.5*((np.log(sp)-mu)/sig)**2)
#     return out

# def call_kernel_lognormal(mu, sigma, K):
#     lnK = np.log(K)
#     d1 = (mu + sigma**2 - lnK)/sigma
#     d2 = (mu - lnK)/sigma
#     m1 = np.exp(mu + 0.5*sigma**2) * norm.cdf(d1)
#     m0 = norm.cdf(d2)
#     return m1 - K*m0

# def h0_log_from_coverage(K, m=23, pad=1.2, min_h=1e-3):
#     K = np.asarray(K, float)
#     K = K[K > 0]
#     span_log = pad * max(1e-9, np.log(K.max()) - np.log(K.min()))
#     h0 = 2.0 * span_log / (m - 1)        # Δz_log = 0.5*h0
#     return max(h0, min_h)

# def geometric_centers(F, h0_log, m=23, lo_clip=0.1, hi_clip=4.0, S0=None):
#     # geometric (log-uniform) grid around the forward
#     lo = F * np.exp(-0.5*(m-1)*(0.5*h0_log))
#     hi = F * np.exp(+0.5*(m-1)*(0.5*h0_log))
#     if S0 is not None:
#         lo = max(lo, lo_clip*S0)
#         hi = min(hi, hi_clip*S0)
#     return np.exp(np.linspace(np.log(lo), np.log(hi), m))

# @dataclass
# class LPCAFit:
#     w: np.ndarray
#     centers: np.ndarray         # exp(mu) medians
#     h0_log: float
#     discount: float             # e^{-rT}
#     forward: float              # S0 e^{(r-q)T}
#     fitted_calls: np.ndarray
#     rmse: float
#     m: int                      # <--- number of kernels used

# def lpca_fit_fixed_h0(
#     K, C, *, S0, r, q, T,
#     h0_log: float, m: int = 23,
#     centers: Optional[np.ndarray] = None,
#     lambda_eq: float = 2e4, l2_reg: float = 1e-6
# ) -> LPCAFit:
#     """
#     Fit LPCA with a fixed bandwidth h0_log and m kernels.

#     This is the basic fitter; model selection over m is handled by
#     `lpca_fit_select_m` below.
#     """
#     K = np.asarray(K, float).ravel()
#     C = np.asarray(C, float).ravel()
#     D = np.exp(-r*T)
#     F = S0*np.exp((r-q)*T)
#     if centers is None:
#         centers = geometric_centers(F, h0_log, m=m, S0=S0)

#     mu = np.log(centers)
#     A = np.column_stack([call_kernel_lognormal(mj, h0_log, K) for mj in mu])
#     Atil = D * A
#     y = C

#     # soft equalities: sum w ≈ 1, sum w E[S]_j ≈ F, with E[S]_j = exp(mu_j + 0.5*h0_log**2)
#     means = np.exp(mu + 0.5*h0_log**2)
#     E = np.vstack([np.ones(m), means])
#     b = np.array([1.0, F])

#     A_aug = np.vstack([Atil, np.sqrt(lambda_eq)*E, np.sqrt(l2_reg)*np.eye(m)])
#     y_aug = np.concatenate([y, np.sqrt(lambda_eq)*b, np.zeros(m)])
#     sol = lsq_linear(A_aug, y_aug, bounds=(0.0, np.inf), lsmr_tol='auto')

#     w = np.maximum(sol.x, 0.0)
#     s1 = w.sum()
#     if s1 > 0:
#         w /= s1

#     fitted = Atil @ w
#     rmse = float(np.sqrt(np.mean((y - fitted)**2)))
#     return LPCAFit(
#         w=w,
#         centers=centers,
#         h0_log=h0_log,
#         discount=D,
#         forward=F,
#         fitted_calls=fitted,
#         rmse=rmse,
#         m=m,
#     )

# def lpca_price_calls(fit: LPCAFit, K_new):
#     K_new = np.asarray(K_new, float).ravel()
#     mu = np.log(fit.centers)
#     A = np.column_stack([call_kernel_lognormal(mj, fit.h0_log, K_new) for mj in mu])
#     return fit.discount * (A @ fit.w)

# def lpca_density(fit: LPCAFit, s_grid):
#     s = np.asarray(s_grid, float).ravel()
#     mu = np.log(fit.centers)
#     mats = np.column_stack([ln_pdf(s, mj, fit.h0_log) for mj in mu])
#     return mats @ fit.w

# # ---------- Model selection over m using true RND (rnd_true, k_true) ----------

# def lpca_fit_select_m(
#     K,
#     C,
#     *,
#     S0: float,
#     r: float,
#     q: float,
#     T: float,
#     m_values: Sequence[int] | int,
#     pad: float = 1.2,
#     lambda_eq: float = 2e4,
#     l2_reg: float = 1e-6,
#     rnd_true: Optional[np.ndarray] = None,
#     k_true: Optional[np.ndarray] = None,
# ) -> Tuple[LPCAFit, int]:
#     """
#     Fit LPCA for each m in m_values and select the one that best matches the
#     true density rnd_true on the grid k_true (in MSE).

#     If rnd_true and k_true are not supplied, fall back to selecting the fit
#     with the smallest call-price RMSE.

#     Returns
#     -------
#     best_fit : LPCAFit
#         The selected LPCA fit. (You can read best_fit.m to see chosen m.)
#     best_m : int
#         The corresponding m (number of kernels).
#     """
#     # Normalize m_values to a list
#     if np.isscalar(m_values):
#         m_list = [int(m_values)]
#     else:
#         m_list = [int(m) for m in m_values]

#     if len(m_list) == 0:
#         raise ValueError("m_values must contain at least one candidate integer.")

#     K = np.asarray(K, float).ravel()
#     C = np.asarray(C, float).ravel()

#     use_density_criterion = (rnd_true is not None) and (k_true is not None)

#     if use_density_criterion:
#         k_true_arr = np.asarray(k_true, float).ravel()
#         rnd_true_arr = np.asarray(rnd_true, float).ravel()
#         if k_true_arr.shape != rnd_true_arr.shape:
#             raise ValueError("k_true and rnd_true must have the same shape.")
#         best_metric = np.inf
#     else:
#         k_true_arr = None
#         rnd_true_arr = None
#         best_metric = np.inf

#     best_fit: Optional[LPCAFit] = None
#     best_m: Optional[int] = None

#     F = S0 * np.exp((r - q) * T)

#     for m in m_list:
#         # bandwidth and centers for this m
#         h0_log = h0_log_from_coverage(K, m=m, pad=pad)
#         centers = geometric_centers(F, h0_log, m=m, S0=S0)

#         fit = lpca_fit_fixed_h0(
#             K, C,
#             S0=S0, r=r, q=q, T=T,
#             h0_log=h0_log, m=m, centers=centers,
#             lambda_eq=lambda_eq, l2_reg=l2_reg
#         )

#         if use_density_criterion:
#             q_est = lpca_density(fit, k_true_arr)
#             metric = float(np.mean((q_est - rnd_true_arr)**2))  # density MSE
#         else:
#             metric = fit.rmse  # price RMSE

#         if metric < best_metric:
#             best_metric = metric
#             best_fit = fit
#             best_m = m

#     if best_fit is None or best_m is None:
#         raise RuntimeError("No successful LPCA fit for any m in m_values.")

#     return best_fit, best_m

# # ---------- Reference Heston RND via Breeden–Litzenberger ----------

# def rnd_from_call_curve_BL(K_dense, C_dense, r, T):
#     """
#     q(s) = e^{rT} * d^2C/dK^2 evaluated at s = K.
#     Uses a smoothing spline in (K, C) and its 2nd derivative.
#     """
#     # smoothing: s=0 keeps it as an interpolating spline; small s>0 if needed
#     spline = UnivariateSpline(K_dense, C_dense, s=0, k=3)
#     C2 = spline.derivative(n=2)(K_dense)
#     q = np.exp(r*T) * np.maximum(C2, 0.0)  # clip tiny negatives from numerics
#     # renormalize numerically (optional)
#     area = np.trapz(q, K_dense)
#     if area > 0:
#         q /= area
#     return q
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
from scipy.stats import norm
from scipy.optimize import lsq_linear
from scipy.interpolate import UnivariateSpline
from Heston_model import *  # assumes HestonParams, heston_call_prices_fast, etc.

# =====================================================
# LPCA building blocks (lognormal kernels)
# =====================================================

def ln_pdf(s, mu, sig):
    out = np.zeros_like(s)
    m = s > 0
    sp = s[m]
    out[m] = (1.0/(sp*sig*np.sqrt(2*np.pi))) * np.exp(-0.5*((np.log(sp)-mu)/sig)**2)
    return out

def call_kernel_lognormal(mu, sigma, K):
    lnK = np.log(K)
    d1 = (mu + sigma**2 - lnK)/sigma
    d2 = (mu - lnK)/sigma
    m1 = np.exp(mu + 0.5*sigma**2) * norm.cdf(d1)
    m0 = norm.cdf(d2)
    return m1 - K*m0

def h0_log_from_coverage(K, m=23, pad=1.2, min_h=1e-3):
    """
    Choose a log-normal bandwidth h0_log based on coverage of the strike range.
    Rough Bondarenko-style rule:
        span_log = pad * (log(K_max) - log(K_min))
        h0_log   = 2 * span_log / (m - 1)   so that Δz_log = 0.5 * h0_log
    """
    K = np.asarray(K, float)
    K = K[K > 0]
    span_log = pad * max(1e-9, np.log(K.max()) - np.log(K.min()))
    h0 = 2.0 * span_log / (m - 1)        # Δz_log = 0.5*h0
    return max(h0, min_h)

def geometric_centers(F, h0_log, m=23, lo_clip=0.1, hi_clip=4.0, S0=None):
    """
    Geometric (log-uniform) grid around the forward price F, with spacing
    linked to h0_log and clipped relative to S0.
    """
    lo = F * np.exp(-0.5*(m-1)*(0.5*h0_log))
    hi = F * np.exp(+0.5*(m-1)*(0.5*h0_log))
    if S0 is not None:
        lo = max(lo, lo_clip*S0)
        hi = min(hi, hi_clip*S0)
    return np.exp(np.linspace(np.log(lo), np.log(hi), m))

# =====================================================
# Dataclass to hold LPCA fit
# =====================================================

@dataclass
class LPCAFit:
    w: np.ndarray
    centers: np.ndarray         # exp(mu) medians
    h0_log: float
    discount: float             # e^{-rT}
    forward: float              # S0 e^{(r-q)T}
    fitted_calls: np.ndarray
    rmse: float
    m: int                      # number of kernels used

# =====================================================
# Core LPCA fitter with fixed h0_log and centers
# =====================================================

def lpca_fit_fixed_h0(
    K,
    C,
    *,
    S0,
    r,
    q,
    T,
    h0_log: float,
    m: int = 23,
    centers: Optional[np.ndarray] = None,
    lambda_eq: float = 2e4,
    l2_reg: float = 1e-6
) -> LPCAFit:
    """
    Fit LPCA with a fixed bandwidth h0_log and m kernels.

    This is the basic fitter; model selection over m is handled by
    `lpca_fit_select_m` or `lpca_fit_select_m_cv` below.
    """
    K = np.asarray(K, float).ravel()
    C = np.asarray(C, float).ravel()
    D = np.exp(-r*T)
    F = S0 * np.exp((r - q) * T)

    if centers is None:
        centers = geometric_centers(F, h0_log, m=m, S0=S0)

    mu = np.log(centers)

    # Design matrix: kernel prices for each kernel
    A = np.column_stack([call_kernel_lognormal(mj, h0_log, K) for mj in mu])
    Atil = D * A
    y = C

    # soft equalities: sum w ≈ 1, sum w E[S]_j ≈ F
    means = np.exp(mu + 0.5*h0_log**2)
    E = np.vstack([np.ones(m), means])
    b = np.array([1.0, F])

    # Augmented system with equality penalties and L2 regularization
    A_aug = np.vstack([Atil, np.sqrt(lambda_eq) * E, np.sqrt(l2_reg) * np.eye(m)])
    y_aug = np.concatenate([y, np.sqrt(lambda_eq) * b, np.zeros(m)])

    sol = lsq_linear(A_aug, y_aug, bounds=(0.0, np.inf), lsmr_tol='auto')

    w = np.maximum(sol.x, 0.0)
    s1 = w.sum()
    if s1 > 0:
        w /= s1

    fitted = Atil @ w
    rmse = float(np.sqrt(np.mean((y - fitted)**2)))
    return LPCAFit(
        w=w,
        centers=centers,
        h0_log=h0_log,
        discount=D,
        forward=F,
        fitted_calls=fitted,
        rmse=rmse,
        m=m,
    )

# =====================================================
# Pricing + density from an LPCA fit
# =====================================================

def lpca_price_calls(fit: LPCAFit, K_new):
    K_new = np.asarray(K_new, float).ravel()
    mu = np.log(fit.centers)
    A = np.column_stack([call_kernel_lognormal(mj, fit.h0_log, K_new) for mj in mu])
    return fit.discount * (A @ fit.w)

def lpca_density(fit: LPCAFit, s_grid):
    s = np.asarray(s_grid, float).ravel()
    mu = np.log(fit.centers)
    mats = np.column_stack([ln_pdf(s, mj, fit.h0_log) for mj in mu])
    return mats @ fit.w

# =====================================================
# Model selection over m using TRUE density (if known)
# =====================================================

def lpca_fit_select_m(
    K,
    C,
    *,
    S0: float,
    r: float,
    q: float,
    T: float,
    m_values: Union[Sequence[int], int],
    pad: float = 1.2,
    lambda_eq: float = 2e4,
    l2_reg: float = 1e-6,
    rnd_true: Optional[np.ndarray] = None,
    k_true: Optional[np.ndarray] = None,
) -> Tuple[LPCAFit, int]:
    """
    Fit LPCA for each m in m_values and select the one that best matches the
    true density rnd_true on the grid k_true (in MSE).

    If rnd_true and k_true are not supplied, fall back to selecting the fit
    with the smallest call-price RMSE.

    Returns
    -------
    best_fit : LPCAFit
        The selected LPCA fit. (You can read best_fit.m to see chosen m.)
    best_m : int
        The corresponding m (number of kernels).
    """
    # Normalize m_values to a list
    if isinstance(m_values, (int, np.integer)):
        m_list = [int(m_values)]
    else:
        m_list = [int(m) for m in m_values]

    if len(m_list) == 0:
        raise ValueError("m_values must contain at least one candidate integer.")

    K = np.asarray(K, float).ravel()
    C = np.asarray(C, float).ravel()

    use_density_criterion = (rnd_true is not None) and (k_true is not None)

    if use_density_criterion:
        k_true_arr = np.asarray(k_true, float).ravel()
        rnd_true_arr = np.asarray(rnd_true, float).ravel()
        if k_true_arr.shape != rnd_true_arr.shape:
            raise ValueError("k_true and rnd_true must have the same shape.")
        best_metric = np.inf
    else:
        k_true_arr = None
        rnd_true_arr = None
        best_metric = np.inf

    best_fit = None
    best_m = None

    F = S0 * np.exp((r - q) * T)

    for m in m_list:
        # bandwidth and centers for this m
        h0_log = h0_log_from_coverage(K, m=m, pad=pad)
        centers = geometric_centers(F, h0_log, m=m, S0=S0)

        fit = lpca_fit_fixed_h0(
            K,
            C,
            S0=S0,
            r=r,
            q=q,
            T=T,
            h0_log=h0_log,
            m=m,
            centers=centers,
            lambda_eq=lambda_eq,
            l2_reg=l2_reg,
        )

        if use_density_criterion:
            q_est = lpca_density(fit, k_true_arr)
            metric = float(np.mean((q_est - rnd_true_arr) ** 2))  # density MSE
        else:
            metric = fit.rmse  # price RMSE

        if metric < best_metric:
            best_metric = metric
            best_fit = fit
            best_m = m

    if best_fit is None or best_m is None:
        raise RuntimeError("No successful LPCA fit for any m in m_values.")

    return best_fit, best_m

# =====================================================
# NEW: Model selection over m via K-fold CV on call prices
# =====================================================

def lpca_fit_select_m_cv(
    K,
    C,
    *,
    S0: float,
    r: float,
    q: float,
    T: float,
    m_values: Union[Sequence[int], int],
    pad: float = 1.2,
    lambda_eq: float = 2e4,
    l2_reg: float = 1e-6,
    n_folds: int = 5,
    random_state: Optional[int] = None,
) -> Tuple[LPCAFit, int, dict]:
    """
    Select m (number of mixtures) via K-fold cross-validation on observed call prices.

    For each m in m_values:
      1. Compute h0_log and centers using the full K coverage (Bondarenko-style).
      2. Run K-fold CV on (K, C), refitting LPCA on the training folds only.
      3. Evaluate RMSE of call prices on the validation folds.
    The m with the smallest overall CV RMSE is selected.

    Returns
    -------
    best_fit : LPCAFit
        Final LPCA fit using all quotes with the chosen m.
    best_m : int
        Selected number of kernels.
    cv_errors : dict
        Mapping m -> CV RMSE (on call prices).
    """
    K = np.asarray(K, float).ravel()
    C = np.asarray(C, float).ravel()
    n_obs = K.size
    if n_obs != C.size:
        raise ValueError("K and C must have the same length.")

    # Normalize m_values to a list
    if isinstance(m_values, (int, np.integer)):
        m_list = [int(m_values)]
    else:
        m_list = [int(m) for m in m_values]

    if len(m_list) == 0:
        raise ValueError("m_values must contain at least one candidate integer.")

    # Adjust n_folds if we have fewer quotes
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if n_folds > n_obs:
        n_folds = n_obs  # at most leave-one-out

    # Build folds (shuffle indices)
    rng = np.random.default_rng(random_state)
    idx = np.arange(n_obs)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)

    F = S0 * np.exp((r - q) * T)

    best_fit = None
    best_m = None
    best_cv_rmse = np.inf
    cv_errors = {}

    for m in m_list:
        # sanity: need more quotes than kernels in each training fold
        min_train_size = min(n_obs - len(fold) for fold in folds)
        if min_train_size <= m:
            raise ValueError(
                f"Not enough quotes to fit m={m} with n_folds={n_folds}. "
                f"Minimum training size per fold is {min_train_size}."
            )

        # Bandwidth and centers for this m (based on full coverage)
        h0_log = h0_log_from_coverage(K, m=m, pad=pad)
        centers = geometric_centers(F, h0_log, m=m, S0=S0)

        # Accumulate squared errors across all validation points
        se_total = 0.0
        n_val_total = 0

        for fold_idx in folds:
            if fold_idx.size == 0:
                continue

            mask_val = np.zeros(n_obs, dtype=bool)
            mask_val[fold_idx] = True
            mask_train = ~mask_val

            K_train, C_train = K[mask_train], C[mask_train]
            K_val, C_val = K[mask_val], C[mask_val]

            # Fit on training set only
            fit_fold = lpca_fit_fixed_h0(
                K_train,
                C_train,
                S0=S0,
                r=r,
                q=q,
                T=T,
                h0_log=h0_log,
                m=m,
                centers=centers,
                lambda_eq=lambda_eq,
                l2_reg=l2_reg,
            )

            # Predict on validation set and accumulate squared error
            C_pred = lpca_price_calls(fit_fold, K_val)
            err = C_val - C_pred
            se_total += float(np.sum(err ** 2))
            n_val_total += C_val.size

        if n_val_total == 0:
            continue

        cv_rmse = float(np.sqrt(se_total / n_val_total))
        cv_errors[m] = cv_rmse

        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_m = m

    if best_m is None:
        raise RuntimeError("No successful CV fit for any m in m_values.")

    # Refit using all quotes with selected m
    h0_best = h0_log_from_coverage(K, m=best_m, pad=pad)
    centers_best = geometric_centers(F, h0_best, m=best_m, S0=S0)
    best_fit = lpca_fit_fixed_h0(
        K,
        C,
        S0=S0,
        r=r,
        q=q,
        T=T,
        h0_log=h0_best,
        m=best_m,
        centers=centers_best,
        lambda_eq=lambda_eq,
        l2_reg=l2_reg,
    )

    return best_fit, best_m, cv_errors

# =====================================================
# Reference Heston RND via Breeden–Litzenberger
# =====================================================

def rnd_from_call_curve_BL(K_dense, C_dense, r, T):
    """
    q(s) = e^{rT} * d^2C/dK^2 evaluated at s = K.
    Uses a smoothing spline in (K, C) and its 2nd derivative.
    """
    # smoothing: s=0 keeps it as an interpolating spline; small s>0 if needed
    spline = UnivariateSpline(K_dense, C_dense, s=0, k=3)
    C2 = spline.derivative(n=2)(K_dense)
    q = np.exp(r*T) * np.maximum(C2, 0.0)  # clip tiny negatives from numerics

    # renormalize numerically (optional)
    area = np.trapz(q, K_dense)
    if area > 0:
        q /= area
    return q

# ===================== EXAMPLE / DRIVER CODE ============================

# # # You said you already have:
if __name__ == "__main__":

    S0, r, q, T = 120.0, 0.02, 0.00, 0.5
    strikes = np.linspace(25, 300, 100)
    
    # TRUE market: Heston (closed-form, discounted calls)
    hparams = HestonParams(kappa=0.5, theta=0.25, sigma=0.15, v0=0.02, rho=-0.9)
    C_mkt = heston_call_prices_fast(S0, strikes, r, q, T, hparams)  # discounted prices
    
    # --- Reference Heston RND via BL on a dense Heston call curve ---
    K_dense = np.linspace(max(1e-6, strikes.min()), strikes.max(), 2000)
    C_dense = heston_call_prices_fast(S0, K_dense, r, q, T, hparams)  # discounted calls
    q_heston = rnd_from_call_curve_BL(K_dense, C_dense, r, T)
    q_heston2 = rnd_from_call_curve_BL(strikes, C_mkt, r, T)
    
    # --- Fit LPCA with model selection over m, using RND closeness criterion ---
    m_candidates = [46]  # can be any array-like of integers
    
    fit, m_best = lpca_fit_select_m(
        strikes, C_mkt,
        S0=S0, r=r, q=q, T=T,
        m_values=m_candidates,
        rnd_true=q_heston2,
        k_true=strikes,
        pad=1.2,
        lambda_eq=0.0,    # you can turn these back on if you like
        l2_reg=0.0  # grid for density comparison
    )
    
    # m_grid = range(15, 66)  # e.g., 15–30 kernels
    
    # best_fit_true, best_m_true = lpca_fit_select_m(
    #     strikes,
    #     C_mkt,
    #     S0=S0,
    #     r=r,
    #     q=q,
    #     T=T,
    #     m_values=m_grid,
    #     pad=1.2,
    #     lambda_eq=2e4,
    #     l2_reg=1e-6,
    #     rnd_true=q_heston2,
    #     k_true=strikes,
    #)
    # Now you can get the chosen m in two ways:
    # 1) From the return value: m_best
    # 2) From the fit object:   fit.m
    
    # --- Fitted calls on a fine grid for plotting ---
    K_fine = np.linspace(strikes.min(), strikes.max(), 400)
    C_fit = lpca_price_calls(fit, K_fine)
    
    # --- PCA RND on the same grid as BL for apples-to-apples ---
    s_grid = K_dense.copy()
    q_pca = lpca_density(fit, s_grid)
    
    # ========================== PLOTS ================================
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # ---------- (a) Calls ----------
    axes[0].set_title(
        f"Calls: Heston Market vs PCA Fit\n"
        f"$m={fit.m}$, $h_0^{{log}}={fit.h0_log:.3f}$, RMSE={fit.rmse:.5f}"
    )
    axes[0].scatter(strikes, C_mkt, s=16, label="Heston (Market) Calls")
    axes[0].plot(K_fine, C_fit, label="PCA Fitted Calls", linewidth=2)
    axes[0].set_xlabel("Strike $K$")
    axes[0].set_ylabel("Call Price")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # ---------- (b) Risk-Neutral Densities ----------
    axes[1].set_title("Risk-Neutral Density: Heston (BL) vs PCA")
    axes[1].plot(s_grid, q_pca, label=f"PCA RND (LPCA, m={fit.m})", linewidth=2)
    axes[1].plot(K_dense, q_heston, label="Heston RND (BL from Calls)", alpha=0.8, linewidth=2)
    axes[1].axvline(S0*np.exp((r-q)*T), linestyle="--", color="k", label="Forward $F_T$")
    axes[1].set_xlabel("Terminal Price $s = K$")
    axes[1].set_ylabel("Density $q(s)$")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print(fit.m)