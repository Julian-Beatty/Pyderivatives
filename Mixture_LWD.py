import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import gammaincc, gamma
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import List, Literal, Tuple, Iterable, Optional, Any

from Heston_model import *  # assumes HestonParams, heston_call_prices_fast, etc.

# ---------- Component definitions ----------

ComponentType = Literal["lognormal", "weibull"]

@dataclass
class MixtureSpec:
    """Describe a mixture with specific counts of components by type."""
    n_lognormal: int
    n_weibull: int

@dataclass
class FittedMixture:
    weights: np.ndarray               # shape (n_comp,)
    types: List[ComponentType]        # list of "lognormal"/"weibull"
    params: List[Tuple[float, float]] # per-component params:
                                      #   lognormal -> (mu, sigma)
                                      #   weibull   -> (k, lam)

    # Wald regression extras (filled by evolutionary_lwm_fit when use_wald=True)
    eps_hat: Optional[np.ndarray] = None   # weighted residuals
    eps_fit: Optional[np.ndarray] = None   # fitted values from auxiliary regression
    eps: Optional[np.ndarray] = None       # alias for eps_hat
    K_norm: Optional[np.ndarray] = None    # normalized moneyness in [-1, 1]

    # Free field for any debugging / tags
    test: Optional[Any] = None

    # Store market environment for pricing helpers
    S0: Optional[float] = None
    r: Optional[float] = None
    T: Optional[float] = None

    # ---------- Convenience evaluation methods ----------

    def qhat(self, x: np.ndarray) -> np.ndarray:
        """
        Risk-neutral density (pdf) of S_T at grid x.

        Usage:
            q = fit.qhat(x_grid)
        """
        x = np.asarray(x, dtype=float)
        return mixture_pdf(x, self.weights, self.types, self.params)

    def chat(
        self,
        K: np.ndarray,
        r: Optional[float] = None,
        T: Optional[float] = None
    ) -> np.ndarray:
        """
        Discounted call prices C(0, K) from the fitted mixture.

        Usage:
            C_hat = fit.chat(K_grid)
        or:
            C_hat = fit.chat(K_grid, r=r_override, T=T_override)
        """
        K = np.asarray(K, dtype=float)

        # prefer explicit args, fall back to stored r, T
        r_eff = self.r if r is None else r
        T_eff = self.T if T is None else T

        if r_eff is None or T_eff is None:
            raise ValueError("Need r and T to compute discounted calls; "
                             "either fit via evolutionary_lwm_fit (which stores r,T) "
                             "or pass r and T explicitly to chat().")

        C_undisc = mixture_call_undisc(K, self.weights, self.types, self.params)
        return np.exp(-r_eff * T_eff) * C_undisc


# ---------- Analytic building blocks (mixture components) ----------

def call_lognormal_undisc(K: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Undiscounted call price E[(S-K)^+] when S~LogNormal(mu,sigma^2)."""
    K = np.asarray(K, dtype=float)
    eps = 1e-12
    sigma = max(sigma, 1e-8)
    out = np.empty_like(K, dtype=float)
    pos = K > eps
    if np.any(pos):
        d1 = (mu - np.log(K[pos]) + sigma**2) / sigma
        d2 = d1 - sigma
        ES = np.exp(mu + 0.5*sigma**2)
        out[pos] = ES * norm.cdf(d1) - K[pos] * norm.cdf(d2)
    if np.any(~pos):
        out[~pos] = np.exp(mu + 0.5*sigma**2)
    return out

def upper_incomplete_gamma(a: float, x: np.ndarray) -> np.ndarray:
    """Γ(a, x) = upper incomplete gamma. scipy gives regularized versions; use Γ(a)*Q(a,x)."""
    x = np.asarray(x, dtype=float)
    return gamma(a) * gammaincc(a, x)

def call_weibull_undisc(K: np.ndarray, k: float, lam: float) -> np.ndarray:
    """Undiscounted call price E[(S-K)^+] when S~Weibull(k,lam). Closed form."""
    K = np.asarray(K, dtype=float)
    z = (K/lam)**k
    term1 = lam * upper_incomplete_gamma(1.0 + 1.0/k, z)
    term2 = K * np.exp(-z)
    return term1 - term2

def mean_lognormal(mu: float, sigma: float) -> float:
    return np.exp(mu + 0.5*sigma**2)

def mean_weibull(k: float, lam: float) -> float:
    return lam * gamma(1.0 + 1.0/k)

def pdf_lognormal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x, dtype=float)
    pos = x > 0
    if np.any(pos):
        xs = x[pos]
        pdf[pos] = (1.0/(xs*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(xs)-mu)**2/(2*sigma**2))
    return pdf

def cdf_lognormal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cdf = np.zeros_like(x, dtype=float)
    pos = x > 0
    if np.any(pos):
        xs = x[pos]
        cdf[pos] = norm.cdf((np.log(xs)-mu)/sigma)
    return cdf

def pdf_weibull(x: np.ndarray, k: float, lam: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x, dtype=float)
    pos = x >= 0
    if np.any(pos):
        xs = x[pos]
        pdf[pos] = (k/lam) * (xs/lam)**(k-1) * np.exp(-(xs/lam)**k)
    return pdf

def cdf_weibull(x: np.ndarray, k: float, lam: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cdf = np.zeros_like(x, dtype=float)
    pos = x >= 0
    if np.any(pos):
        xs = x[pos]
        cdf[pos] = 1.0 - np.exp(-(xs/lam)**k)
    return cdf

# ---------- Mixture model pricing & density ----------

def unpack_theta(theta: np.ndarray, spec: MixtureSpec):
    """Map unconstrained theta -> (weights, params) with positivity and simplex via softmax."""
    idx = 0
    n_comp = spec.n_lognormal + spec.n_weibull

    logits = theta[idx:idx+n_comp]; idx += n_comp
    w = np.exp(logits - logits.max())
    w = w / w.sum()

    types: List[ComponentType] = []
    params: List[Tuple[float, float]] = []

    # Lognormal params: (mu, log_sigma)
    for _ in range(spec.n_lognormal):
        mu = theta[idx]; idx += 1
        ls = theta[idx]; idx += 1
        sigma = np.exp(ls) + 1e-8
        types.append("lognormal")
        params.append((mu, sigma))

    # Weibull params: (log_k, log_lam)
    for _ in range(spec.n_weibull):
        lk = theta[idx]; idx += 1
        ll = theta[idx]; idx += 1
        k = np.exp(lk) + 1e-8
        lam = np.exp(ll) + 1e-8
        types.append("weibull")
        params.append((k, lam))

    return w, types, params

def mixture_mean(w, types, params) -> float:
    m = 0.0
    for wi, t, (a, b) in zip(w, types, params):
        if t == "lognormal":
            m += wi * mean_lognormal(a, b)
        else:
            m += wi * mean_weibull(a, b)
    return m

def mixture_call_undisc(K: np.ndarray, w, types, params) -> np.ndarray:
    total = np.zeros_like(K, dtype=float)
    for wi, t, (a, b) in zip(w, types, params):
        if t == "lognormal":
            total += wi * call_lognormal_undisc(K, a, b)
        else:
            total += wi * call_weibull_undisc(K, a, b)
    return total

def mixture_pdf(x: np.ndarray, w, types, params) -> np.ndarray:
    f = np.zeros_like(x, dtype=float)
    for wi, t, (a, b) in zip(w, types, params):
        if t == "lognormal":
            f += wi * pdf_lognormal(x, a, b)
        else:
            f += wi * pdf_weibull(x, a, b)
    return f

def mixture_cdf(x: np.ndarray, w, types, params) -> np.ndarray:
    F = np.zeros_like(x, dtype=float)
    for wi, t, (a, b) in zip(w, types, params):
        if t == "lognormal":
            F += wi * cdf_lognormal(x, a, b)
        else:
            F += wi * cdf_weibull(x, a, b)
    return F

# ---------- Helpers for evolutionary algorithm ----------

def pack_theta_from_mixture(fit: FittedMixture, spec: MixtureSpec) -> np.ndarray:
    """Create a theta vector consistent with unpack_theta that reproduces a given mixture."""
    n_comp = spec.n_lognormal + spec.n_weibull
    assert len(fit.weights) == n_comp
    assert len(fit.types) == n_comp

    logits = np.log(fit.weights + 1e-12)
    parts = [logits]
    for t, (a, b) in zip(fit.types, fit.params):
        if t == "lognormal":
            parts.append(np.array([a, np.log(b)]))
        else:
            parts.append(np.array([np.log(a), np.log(b)]))
    return np.concatenate(parts)

def extend_mixture_initial_guess(
    prev_fit: FittedMixture,
    prev_spec: MixtureSpec,
    new_spec: MixtureSpec,
    F0: float,
    rng: np.random.Generator,
    eps_weight: float = 0.05,
) -> np.ndarray:
    """
    Build an initial theta for (new_spec) by taking the previous optimal
    mixture (prev_fit, prev_spec) and adding one new component (LN or W).
    """
    w_old = prev_fit.weights.copy()
    types_old = prev_fit.types
    params_old = prev_fit.params

    n_old = prev_spec.n_lognormal + prev_spec.n_weibull
    n_new = new_spec.n_lognormal + new_spec.n_weibull
    assert n_new == n_old + 1, "This helper only handles M -> M+1."

    # New weights: scale old down and add a small positive weight for the new component
    w_new = np.empty(n_new)
    w_new[:-1] = (1.0 - eps_weight) * w_old
    w_new[-1] = eps_weight
    w_new /= w_new.sum()

    types: List[ComponentType] = []
    params: List[Tuple[float, float]] = []

    ln_old = [p for t, p in zip(types_old, params_old) if t == "lognormal"]
    wb_old = [p for t, p in zip(types_old, params_old) if t == "weibull"]

    n_ln_new = new_spec.n_lognormal
    n_wb_new = new_spec.n_weibull
    n_ln_old = prev_spec.n_lognormal
    n_wb_old = prev_spec.n_weibull

    # Existing lognormals
    for i in range(min(n_ln_old, n_ln_new)):
        types.append("lognormal")
        params.append(ln_old[i])
    # Extra lognormal if needed
    if n_ln_new > n_ln_old:
        sigma = rng.uniform(0.2, 0.6)
        mu = np.log(F0) - 0.5*sigma**2 + rng.normal(0, 0.1)
        types.append("lognormal")
        params.append((mu, sigma))

    # Existing Weibulls
    for i in range(min(n_wb_old, n_wb_new)):
        types.append("weibull")
        params.append(wb_old[i])
    # Extra Weibull if needed
    if n_wb_new > n_wb_old:
        k = rng.uniform(0.8, 2.0)
        lam = F0 / float(gamma(1.0 + 1.0/k))
        lam *= rng.uniform(0.8, 1.2)
        types.append("weibull")
        params.append((k, lam))

    logits = np.log(w_new + 1e-12)
    parts = [logits]
    for t, (a, b) in zip(types, params):
        if t == "lognormal":
            parts.append(np.array([a, np.log(b)]))
        else:
            parts.append(np.array([np.log(a), np.log(b)]))
    return np.concatenate(parts)

# ---------- Fit mixture to option prices (discounted) ----------

def fit_mixture_to_calls(
    K: np.ndarray,
    C_mkt: np.ndarray,
    S0: float,
    r: float,
    T: float,
    spec: MixtureSpec,
    theta0: Optional[np.ndarray] = None,
    penalty_lambda: float = 1e6,
    random_starts: int = 5,
    seed: int = 123,
    rnd_true: np.ndarray | None = None,
    k_true: np.ndarray | None = None,
    var_c: float = 0.1,           # variance constraint parameter c
    var_penalty: float = 1e4,     # strength of variance penalty
    return_theta: bool = False,
) -> Tuple[FittedMixture, Optional[np.ndarray], float]:
    """
    Fit mixture by minimizing squared price errors + martingale penalty +
    variance-constraint penalty:

    Loss = sum_i (C_model(K_i)-C_mkt_i)^2
           + λ (E[S_T]-F0)^2
           + var_penalty * max(0, c*max_i ˜k_i - min_i ˜k_i)^2

    where ˜k_i = sigma_i * sqrt(6π)/π for lognormal components, and
          ˜k_i = k_i for Weibull components.

    If rnd_true and k_true are provided, we still optimize this loss for each
    random start, but we *select* the final solution as the one whose implied
    RND is closest to rnd_true (in MSE on the grid k_true).
    """
    rng = np.random.default_rng(seed)
    K = np.asarray(K, dtype=float)
    C_mkt = np.asarray(C_mkt, dtype=float)
    F0 = S0 * np.exp(r*T)

    n_comp = spec.n_lognormal + spec.n_weibull

    def variance_penalty(theta: np.ndarray) -> float:
        # If c <= 0 or only one component, no variance constraint
        if var_c <= 0.0 or n_comp <= 1:
            return 0.0

        w, types, params = unpack_theta(theta, spec)

        tilde = []
        for t, (a, b) in zip(types, params):
            if t == "lognormal":
                sigma = b
                tilde_k = sigma * np.sqrt(6.0 * np.pi) / np.pi
            else:
                k = a
                tilde_k = k
            tilde.append(tilde_k)

        tilde = np.asarray(tilde, dtype=float)

        # If anything is non-finite or crazy large, just whack with a big constant penalty
        if (not np.all(np.isfinite(tilde))) or np.max(tilde) > 1e6:
            return 1e20

        max_t = float(tilde.max())
        min_t = float(tilde.min())

        # Guard against degenerate max_t
        if max_t <= 0.0:
            return 0.0

        viol = var_c * max_t - min_t
        if viol <= 0.0:
            return 0.0

        # Clip viol so viol**2 can't overflow
        viol = min(viol, 1e10)
        return viol * viol

    def loss(theta):
        w, types, params = unpack_theta(theta, spec)
        C_model_undisc = mixture_call_undisc(K, w, types, params)
        C_model = np.exp(-r*T) * C_model_undisc
        price_err = C_model - C_mkt
        m = mixture_mean(w, types, params)
        pen_mart = (m - F0)**2
        pen_var = variance_penalty(theta)
        L = np.sum(price_err**2) + penalty_lambda * pen_mart + var_penalty * pen_var
        return float(L) if np.isfinite(L) else 1e20

    def random_theta():
        logits = rng.normal(0, 0.1, size=n_comp)
        thetas = [logits]
        for _ in range(spec.n_lognormal):
            sigma = rng.uniform(0.2, 0.6)
            mu = np.log(F0) - 0.5*sigma**2 + rng.normal(0, 0.1)
            thetas += [np.array([mu, np.log(sigma)])]
        for _ in range(spec.n_weibull):
            k = rng.uniform(0.8, 2.0)
            lam = F0 / float(gamma(1.0 + 1.0/k))
            lam *= rng.uniform(0.8, 1.2)
            thetas += [np.array([np.log(k), np.log(lam)])]
        return np.concatenate(thetas)

    # starting points
    starts: List[np.ndarray] = []

    # Expected dimension given the current spec
    expected_dim = 3 * n_comp  # n_comp logits + 2 params per component

    if theta0 is not None:
        theta0 = np.asarray(theta0, float)
        if theta0.shape[0] == expected_dim:
            starts.append(theta0)
        else:
            print(
                f"[fit_mixture_to_calls] Warning: theta0 length "
                f"{theta0.shape[0]} != expected {expected_dim} for spec={spec}. "
                "Ignoring warm start."
            )

    # Fill up with random starts
    while len(starts) < max(1, random_starts):
        starts.append(random_theta())

    candidates: List[Tuple[float, np.ndarray]] = []  # (loss_value, theta)
    for th0 in starts:
        res = minimize(loss, th0, method="L-BFGS-B")
        if np.isfinite(res.fun):
            candidates.append((float(res.fun), res.x))

    if not candidates:
        raise RuntimeError("No feasible solution: try more random_starts or adjust penalties.")

    # if true density is available, choose theta by density MSE
    if rnd_true is not None and k_true is not None:
        k_true_arr = np.asarray(k_true, dtype=float)
        rnd_true_arr = np.asarray(rnd_true, dtype=float)
        if k_true_arr.shape != rnd_true_arr.shape:
            raise ValueError("k_true and rnd_true must have the same shape.")

        best_theta = None
        best_density_mse = np.inf
        best_loss = np.inf
        for Lval, theta in candidates:
            w, types, params = unpack_theta(theta, spec)
            pdf_fit = mixture_pdf(k_true_arr, w, types, params)
            mse = float(np.mean((pdf_fit - rnd_true_arr)**2))
            if mse < best_density_mse:
                best_density_mse = mse
                best_theta = theta
                best_loss = Lval
        loss_best = best_loss
    else:
        loss_best, best_theta = min(candidates, key=lambda x: x[0])

    w, types, params = unpack_theta(best_theta, spec)
    fit = FittedMixture(weights=w, types=types, params=params)
    if return_theta:
        return fit, best_theta, loss_best
    else:
        return fit, None, loss_best

# ---------- Density MSE helper (for Option C) ----------

def density_mse_for_fit(
    fit: FittedMixture,
    k_true: np.ndarray,
    rnd_true: np.ndarray,
) -> float:
    """Compute MSE between mixture pdf and a benchmark density on the same grid."""
    k_true = np.asarray(k_true, dtype=float)
    rnd_true = np.asarray(rnd_true, dtype=float)
    pdf_fit = mixture_pdf(k_true, fit.weights, fit.types, fit.params)
    return float(np.mean((pdf_fit - rnd_true)**2))

# ---------- Gallant / Li-style Wald specification test ----------

def wald_spec_test_mixture(
    K: np.ndarray,
    C_mkt: np.ndarray,
    S0: float,
    r: float,
    T: float,
    fit: FittedMixture,
    weights: Optional[np.ndarray] = None,
    p: int = 3,
    q: int = 3,
    return_details: bool = False,
) -> Tuple[float, float, Optional[dict]]:
    """
    Gallant-style auxiliary regression + HAC-robust Wald test
    for residual structure, adapted to the mixture model.

    Returns
    -------
    wald_stat : float
        Wald test statistic.
    p_value   : float
        p-value (chi^2 under H0).
    details   : dict or None
        If return_details=True, contains:
          - 'eps_hat': weighted residuals
          - 'eps_fit': fitted regression values
          - 'K_norm': normalized moneyness in [-1,1]
    """
    K = np.asarray(K, dtype=float)
    C_mkt = np.asarray(C_mkt, dtype=float)
    n = K.size

    # 1) Fitted prices from mixture (discounted)
    C_fit_undisc = mixture_call_undisc(K, fit.weights, fit.types, fit.params)
    C_fit = np.exp(-r * T) * C_fit_undisc

    # 2) Weighted residuals: ê_n = sqrt(ω_n) (C_n - Ĉ_n)
    if weights is None:
        w = np.ones_like(C_mkt, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != C_mkt.shape:
            raise ValueError("weights must have same shape as C_mkt")
    eps_hat = np.sqrt(w) * (C_mkt - C_fit)

    # 3) Normalized moneyness K̄_n ∈ [-1,1]
    K_min, K_max = float(K.min()), float(K.max())
    if K_max == K_min:
        K_norm = np.zeros_like(K, dtype=float)
    else:
        K_norm = 2.0 * (K - K_min) / (K_max - K_min) - 1.0

    # 4) Build auxiliary regression design matrix X:
    #    [1, K̄, K̄^2, ..., K̄^p, sin(2π j K̄), cos(2π j K̄), j=1..q]
    cols = []

    # polynomial terms up to degree p
    for d in range(1, p + 1):
        cols.append(K_norm**d)

    # Fourier terms up to frequency q
    for j in range(1, q + 1):
        cols.append(np.sin(2.0 * np.pi * j * K_norm))
        cols.append(np.cos(2.0 * np.pi * j * K_norm))

    X = np.column_stack(cols) if cols else np.empty((n, 0))
    # add intercept
    X = sm.add_constant(X, has_constant="add")

    # 5) OLS with HAC (Newey–West) covariance
    maxlags = int(np.floor(0.75 * n ** (1.0 / 3.0)))
    maxlags = max(maxlags, 1)

    model = sm.OLS(eps_hat, X)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    # 6) Wald test: all non-constant coefficients = 0
    k_params = X.shape[1]
    R = np.zeros((k_params - 1, k_params))
    R[np.arange(k_params - 1), np.arange(1, k_params)] = 1.0
    r_vec = np.zeros(k_params - 1)

    wald_res = res.wald_test((R, r_vec), use_f=False)

    # statsmodels already gives us the chi-square stat & p-value
    wald_stat = float(np.atleast_1d(wald_res.statistic)[0])
    p_value = float(np.atleast_1d(wald_res.pvalue)[0])

    details = None
    if return_details:
        eps_fit = X @ res.params
        details = {
            "eps_hat": eps_hat,
            "eps_fit": eps_fit,
            "K_norm": K_norm,
        }

    return wald_stat, p_value, details

# ---------- Evolutionary algorithm over M and M1 with Wald test ----------

def evolutionary_lwm_fit(
    K: np.ndarray,
    C_mkt: np.ndarray,
    S0: float,
    r: float,
    T: float,
    M_max: int = 5,
    penalty_lambda: float = 0,
    random_starts: int = 5,
    seed: int = 123,
    var_c: float = 0.1,
    var_penalty: float = 1e4,
    improvement_tol: float = 1e-4,
    # Option C controls:
    metric: Literal["loss", "density"] = "loss",
    rnd_true: np.ndarray | None = None,
    k_true: np.ndarray | None = None,
    # Wald-spec-test controls
    use_wald: bool = True,
    wald_alpha: float = 0.05,
    wald_p: int = 3,
    wald_q: int = 3,
    weights: Optional[np.ndarray] = None,
    # NEW: allow fixed mixture instead of evolutionary search
    fixed_M: Optional[int] = None,
    fixed_M1: Optional[int] = None,
) -> Tuple[FittedMixture, MixtureSpec]:
    """
    Evolutionary Li–Lu–Qu style scheme with optional Gallant-type Wald
    specification test for choosing M.

    metric = "loss"    -> within each M, pick spec by pricing+penalty loss
    metric = "density" -> within each M, pick spec by density MSE vs rnd_true

    If use_wald=True (default), we increase M sequentially and run a
    residual-specification Wald test after fitting the best spec at each M.
    We stop at the first M for which the Wald test fails to reject H0.

    NEW:
    If fixed_M and fixed_M1 are provided, the function skips the evolutionary
    loop and simply fits MixtureSpec(n_lognormal=fixed_M1, n_weibull=fixed_M-fixed_M1)
    using the same loss/metric logic as before.

    In all cases where a Wald test is run, the chosen FittedMixture has:
      - fit.eps_hat (and alias fit.eps)
      - fit.eps_fit
      - fit.K_norm
    populated from the auxiliary Fourier/polynomial regression.

    Also, fit.S0, fit.r, fit.T are populated so that fit.chat() works.
    """
    rng = np.random.default_rng(seed)
    K = np.asarray(K, dtype=float)
    C_mkt = np.asarray(C_mkt, dtype=float)
    F0 = S0 * np.exp(r*T)

    # ---------- Fixed-mixture shortcut (no evolutionary loop) ----------

    if fixed_M is not None:
        if fixed_M1 is None:
            raise ValueError("fixed_M is given but fixed_M1 is None.")
        if not (0 <= fixed_M1 <= fixed_M):
            raise ValueError("fixed_M1 must be between 0 and fixed_M.")

        spec = MixtureSpec(n_lognormal=fixed_M1, n_weibull=fixed_M - fixed_M1)

        # For metric='density', need k_true and rnd_true
        if metric == "density":
            if rnd_true is None or k_true is None:
                raise ValueError("For metric='density', rnd_true and k_true must be provided.")
            k_true_arr = np.asarray(k_true, dtype=float)
            rnd_true_arr = np.asarray(rnd_true, dtype=float)
            if k_true_arr.shape != rnd_true_arr.shape:
                raise ValueError("k_true and rnd_true must have the same shape.")
        else:
            k_true_arr = rnd_true_arr = None  # just for type consistency

        fit, theta_est, loss_val = fit_mixture_to_calls(
            K=K,
            C_mkt=C_mkt,
            S0=S0,
            r=r,
            T=T,
            spec=spec,
            theta0=None,
            penalty_lambda=penalty_lambda,
            random_starts=random_starts,
            seed=rng.integers(0, 2**32 - 1),
            rnd_true=None,   # local optimization always on price-based loss
            k_true=None,
            var_c=var_c,
            var_penalty=var_penalty,
            return_theta=True,
        )

        # store market params for chat/qhat
        fit.S0 = S0
        fit.r = r
        fit.T = T
        fit.test = "hello"  # your debug tag

        # Run Wald once on this fixed mixture (if requested) and store details
        if use_wald:
            wald_stat, p_val, details = wald_spec_test_mixture(
                K=K,
                C_mkt=C_mkt,
                S0=S0,
                r=r,
                T=T,
                fit=fit,
                weights=weights,
                p=wald_p,
                q=wald_q,
                return_details=True,
            )
            if details is not None:
                fit.eps_hat = details["eps_hat"]
                fit.eps_fit = details["eps_fit"]
                fit.eps = details["eps_hat"]
                fit.K_norm = details["K_norm"]

        if metric == "loss":
            return fit, spec
        else:
            _ = density_mse_for_fit(fit, k_true_arr, rnd_true_arr)
            return fit, spec

    # ---------- Original evolutionary logic below ----------

    if metric == "density":
        if rnd_true is not None and k_true is not None:
            k_true_arr = np.asarray(k_true, dtype=float)
            rnd_true_arr = np.asarray(rnd_true, dtype=float)
            if k_true_arr.shape != rnd_true_arr.shape:
                raise ValueError("k_true and rnd_true must have the same shape.")
        else:
            raise ValueError("For metric='density', rnd_true and k_true must be provided.")
    else:
        k_true_arr = rnd_true_arr = None

    best_fit: Optional[FittedMixture] = None
    best_spec: Optional[MixtureSpec] = None
    best_theta: Optional[np.ndarray] = None
    best_metric_val: Optional[float] = None
    best_M1_prev: Optional[int] = None

    last_fit: Optional[FittedMixture] = None
    last_spec: Optional[MixtureSpec] = None

    # warm start from previous M only (for M -> M+1 extension)
    warm_fit: Optional[FittedMixture] = None
    warm_spec: Optional[MixtureSpec] = None

    for M in range(1, M_max + 1):
        # candidate M1 values
        if M == 1:
            candidate_M1 = [0, 1]
        else:
            candidate_M1 = [best_M1_prev, best_M1_prev + 1]
        candidate_M1 = sorted({m1 for m1 in candidate_M1 if 0 <= m1 <= M})

        M_results = []

        for M1 in candidate_M1:
            spec = MixtureSpec(n_lognormal=M1, n_weibull=M - M1)

            # warm start only from previous M, and only if total components increase by 1
            if warm_fit is None or warm_spec is None:
                theta0 = None
            else:
                n_old = warm_spec.n_lognormal + warm_spec.n_weibull
                n_new = spec.n_lognormal + spec.n_weibull
                if n_new == n_old + 1:
                    theta0 = extend_mixture_initial_guess(
                        prev_fit=warm_fit,
                        prev_spec=warm_spec,
                        new_spec=spec,
                        F0=F0,
                        rng=rng,
                        eps_weight=0.05,
                    )
                else:
                    theta0 = None

            fit, theta_est, loss_val = fit_mixture_to_calls(
                K=K,
                C_mkt=C_mkt,
                S0=S0,
                r=r,
                T=T,
                spec=spec,
                theta0=theta0,
                penalty_lambda=penalty_lambda,
                random_starts=random_starts,
                seed=rng.integers(0, 2**32 - 1),
                rnd_true=None,   # local opt always price-based
                k_true=None,
                var_c=var_c,
                var_penalty=var_penalty,
                return_theta=True,
            )

            # store market params for chat/qhat
            fit.S0 = S0
            fit.r = r
            fit.T = T

            if metric == "loss":
                metric_val = loss_val
            else:
                metric_val = density_mse_for_fit(fit, k_true_arr, rnd_true_arr)

            M_results.append((metric_val, loss_val, fit, theta_est, spec, M1))

        # pick best spec for this M according to chosen metric
        metric_M, loss_M, fit_M, theta_M, spec_M, M1_M = min(M_results, key=lambda x: x[0])

        # update warm start for next M
        warm_fit, warm_spec = fit_M, spec_M

        last_fit, last_spec = fit_M, spec_M

        # Update "global best" purely for evolutionary warm starts & fallback
        if best_metric_val is None or metric_M < best_metric_val:
            best_metric_val = metric_M
            best_fit = fit_M
            best_theta = theta_M
            best_spec = spec_M
            best_M1_prev = M1_M

        # Optional: early stop if metric stops improving (when not using Wald)
        if (not use_wald) and (best_metric_val is not None):
            rel_improve = (best_metric_val - metric_M) / max(best_metric_val, 1e-12)
            if rel_improve < improvement_tol:
                break

        # Wald specification test to decide whether to stop increasing M
        if use_wald:
            wald_stat, p_val, details = wald_spec_test_mixture(
                K=K,
                C_mkt=C_mkt,
                S0=S0,
                r=r,
                T=T,
                fit=fit_M,
                weights=weights,
                p=wald_p,
                q=wald_q,
                return_details=True,
            )

            # store Wald regression extras on the current best-at-this-M fit
            if details is not None:
                fit_M.eps_hat = details["eps_hat"]
                fit_M.eps_fit = details["eps_fit"]
                fit_M.eps = details["eps_hat"]
                fit_M.K_norm = details["K_norm"]

            if p_val > wald_alpha:
                # First M where we cannot reject H0: residuals look like noise.
                return fit_M, spec_M

    # Fallbacks
    if use_wald:
        if last_fit is not None and last_spec is not None:
            return last_fit, last_spec
        if best_fit is not None and best_spec is not None:
            return best_fit, best_spec
        raise RuntimeError("Evolutionary Wald selection failed to find any feasible model.")
    else:
        if best_fit is None or best_spec is None:
            raise RuntimeError("Evolutionary fit failed to find any feasible model.")
        return best_fit, best_spec

# ---------- Convenience: RND + CDF evaluation ----------

def evaluate_rnd(fit: FittedMixture, x_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the fitted mixture RND (pdf and cdf) on a given grid x_grid.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    pdf = mixture_pdf(x_grid, fit.weights, fit.types, fit.params)
    cdf = mixture_cdf(x_grid, fit.weights, fit.types, fit.params)
    return pdf, cdf

# ---------- Example main ----------

if __name__ == "__main__":
    # Market setup
    true_p = HestonParams(kappa=0.5, theta=0.05, sigma=0.25, v0=0.02, rho=-0.0)
    S0, r, q, T = 120.0, 0.02, 0.00, 0.5
    strikes = np.linspace(100, 250, 80)

    # “Market” prices (fast Heston pricer from your Heston_model module)
    C_mkt = heston_call_prices_fast(S0, strikes, r, q, T, true_p)

    # Breeden–Litzenberger density from Heston calls (true RND on strike grid)
    dC_dK   = np.gradient(C_mkt, strikes, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK,  strikes, edge_order=2)
    q_K     = np.exp(r * T) * d2C_dK2

    # Evolutionary selection of M and M1 with Wald spec test + (optional) density metric
    fit, chosen_spec = evolutionary_lwm_fit(
        K=strikes,
        C_mkt=C_mkt,
        S0=S0,
        r=r,
        T=T,
        M_max=3,
        penalty_lambda=0.0,
        random_starts=1,
        seed=42,
        var_c=0.1,
        var_penalty=1e4,
        improvement_tol=1e-4,
        metric="density",   # or "loss"
        rnd_true=q_K,
        k_true=strikes,
        use_wald=True,
        wald_alpha=0.05,
        wald_p=1,
        wald_q=1,
        weights=None,
        fixed_M = 2,
        fixed_M1=0,
        # or an open-interest vector
    )
    print(fit.test)

    print("Chosen spec: M1 (lognormals) =", chosen_spec.n_lognormal,
          ", total M =", chosen_spec.n_lognormal + chosen_spec.n_weibull)

    print("Estimated weights:", fit.weights)
    for i, (t, (a1, b1)) in enumerate(zip(fit.types, fit.params)):
        if t == "lognormal":
            print(f"  Comp {i+1}: Lognormal  mu={a1:.4f}, sigma={b1:.4f}, mean={mean_lognormal(a1,b1):.4f}")
        else:
            print(f"  Comp {i+1}: Weibull    k={a1:.4f}, lam={b1:.4f}, mean={mean_weibull(a1,b1):.4f}")

    # Fitted calls from mixture (discounted)
    C_fit = np.exp(-r*T) * mixture_call_undisc(strikes, fit.weights, fit.types, fit.params)

    # Mixture RND on a dense grid
    x_grid = np.linspace(100, 300, 200)
    pdf_mix, cdf_mix = evaluate_rnd(fit, x_grid)

    # --- Plots ---
    plt.figure(figsize=(16.0, 4.6))

    # (1) Calls
    ax1 = plt.subplot(1, 4, 1)
    ax1.set_title("Call prices: Heston market vs mixture fit")
    ax1.plot(strikes, C_mkt, "o", ms=4, label="Market (Heston)")
    ax1.plot(strikes, C_fit, "-", lw=2, label="Mixture fit")
    ax1.set_xlabel("Strike"); ax1.set_ylabel("Call Price")
    ax1.legend()

    # (2) RND (pdf)
    ax2 = plt.subplot(1, 4, 2)
    ax2.set_title("RND (pdf)")
    ax2.plot(x_grid, pdf_mix, lw=2, label="Mixture (pdf)")
    ax2.plot(strikes, q_K, ".", ms=3, label="BL from Heston calls")
    ax2.set_xlabel("S_T"); ax2.legend()

    # (3) CDF
    ax3 = plt.subplot(1, 4, 3)
    ax3.set_title("CDF (mixture)")
    ax3.plot(x_grid, cdf_mix, lw=2)
    ax3.set_xlabel("S_T")

    plt.tight_layout()
    plt.show()
    
    
    ##########Wald diagnostics

    print("Chosen spec: M1 (lognormals) =", chosen_spec.n_lognormal,
          ", total M =", chosen_spec.n_lognormal + chosen_spec.n_weibull)

    print("Estimated weights:", fit.weights)
    for i, (t, (a1, b1)) in enumerate(zip(fit.types, fit.params)):
        if t == "lognormal":
            print(f"  Comp {i+1}: Lognormal  mu={a1:.4f}, sigma={b1:.4f}, mean={mean_lognormal(a1,b1):.4f}")
        else:
            print(f"  Comp {i+1}: Weibull    k={a1:.4f}, lam={b1:.4f}, mean={mean_weibull(a1,b1):.4f}")

    # Example of using stored Wald regression stuff:
    if fit.eps is not None and fit.eps_fit is not None and fit.K_norm is not None:
        plt.figure(figsize=(6,4))
        plt.scatter(fit.K_norm, fit.eps, s=10, label="Residuals")
        plt.plot(fit.K_norm, fit.eps_fit, lw=2, label="Aux regression fit")
        plt.xlabel("Normalized moneyness K_norm")
        plt.ylabel("Weighted residuals")
        plt.legend()
        plt.tight_layout()
        plt.show()
###If you want to evaluate on finer grid
    K_new = np.linspace(50, 200, 150)
    C_hat_new = fit.chat(K_new)      # discounted calls
    q_hat_new = fit.qhat(K_new)
    
    plt.plot(K_new,q_hat_new)    