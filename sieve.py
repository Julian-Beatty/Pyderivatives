import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Dict
from scipy.special import eval_hermite, factorial
from scipy.optimize import minimize, brentq
from scipy.stats import norm
import matplotlib.pyplot as plt


# ============================================================
# Hermite basis and trapezoid weights
# ============================================================

def hermite_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Orthonormal Hermite functions h_j(x), j = 0,...,degree.
    Returns H with shape (m, degree+1), where H[:, j] = h_j(x).
    """
    x = np.asarray(x, float)
    m = x.size
    H = np.empty((m, degree + 1), float)

    phi = np.exp(-0.5 * x**2)

    for j in range(degree + 1):
        Hj = eval_hermite(j, x)  # physicists' Hermite polynomial H_j
        norm_const = np.sqrt((2.0**j) * factorial(j) * np.sqrt(np.pi))
        H[:, j] = (Hj * phi) / norm_const

    return H  # shape (m, degree+1)


def trapz_weights(x: np.ndarray) -> np.ndarray:
    """
    Trapezoidal integration weights on increasing grid x.
    This is the analog of 'ci' in the R code.
    """
    x = np.asarray(x, float)
    dx = np.diff(x)
    if not np.all(dx > 0):
        raise ValueError("x must be strictly increasing.")
    w = np.empty_like(x)
    w[0] = dx[0]
    w[-1] = dx[-1]
    if x.size > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


# ============================================================
# Black–Scholes helpers
# ============================================================

def bs_call_price(S0: float, K, rf: float, sigma: float, tau: float):
    """
    Black–Scholes call price with continuous rf and volatility sigma.
    rf and tau must be in consistent time units.
    Vectorized in K.
    """
    K = np.asarray(K, float)
    if tau <= 0.0:
        return np.maximum(S0 - K, 0.0)
    if sigma <= 0.0:
        return np.maximum(S0 - K * np.exp(-rf * tau), 0.0)

    S0 = float(S0)
    sig_sqrt_tau = sigma * np.sqrt(tau)
    d1 = (np.log(S0 / K) + (rf + 0.5 * sigma**2) * tau) / sig_sqrt_tau
    d2 = d1 - sig_sqrt_tau
    disc = np.exp(-rf * tau)
    return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)


def implied_vol_from_call_price(
    S0: float,
    K: float,
    rf: float,
    tau: float,
    C: float,
    vol_low: float = 1e-6,
    vol_high: float = 3.0,
) -> float:
    """
    Implied volatility from a single call price via brentq.
    rf, tau in arbitrary but consistent time units.
    """
    C = float(C)
    lower_bound = max(S0 - K * np.exp(-rf * tau), 0.0)
    upper_bound = S0
    C_clip = float(np.clip(C, lower_bound, upper_bound))

    def f(vol):
        return bs_call_price(S0, K, rf, vol, tau) - C_clip

    lo, hi = vol_low, vol_high
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        # try to enlarge bracket
        for _ in range(10):
            hi *= 2.0
            f_hi = f(hi)
            if f_lo * f_hi <= 0:
                break
        else:
        # fallback: modest vol if cannot bracket
            return 0.2

    try:
        return brentq(f, lo, hi, maxiter=200, xtol=1e-10)
    except ValueError:
        return 0.2


def estimate_sigma_atm_from_calls(
    Kcall: np.ndarray,
    Pcall: np.ndarray,
    st: float,
    rf: float,
    tau: float,
) -> float:
    """
    Estimate Sigma from ATM call(s):
      1) find strike(s) closest to spot
      2) average their prices
      3) compute BS implied vol in same tau units.
    """
    Kcall = np.asarray(Kcall, float)
    Pcall = np.asarray(Pcall, float)

    dK = np.abs(Kcall - st)
    min_dK = np.min(dK)
    atm_idx = np.where(dK == min_dK)[0]
    K_atm = np.mean(Kcall[atm_idx])
    C_atm = np.mean(Pcall[atm_idx])

    sigma_atm = implied_vol_from_call_price(st, K_atm, rf, tau, C_atm)
    return float(sigma_atm)


# ============================================================
# K-fold CV for alphaR (ridge penalty)
# ============================================================

def cv_alphaR(
    R: np.ndarray,
    P: np.ndarray,
    alphaR_grid: np.ndarray,
    n_folds: int = 10,
    random_state: int = 0,
) -> Tuple[float, np.ndarray]:
    """
    K-fold cross-validation to choose alphaR.

    For each alphaR:
        beta = argmin ||P - R beta||^2 + alphaR ||beta||^2  (ridge)
    We return the best alphaR and the CV MSE curve.
    """
    n, d = R.shape
    P = np.asarray(P, float)
    alphaR_grid = np.asarray(alphaR_grid, float)

    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)

    cv_mse = []

    for aR in alphaR_grid:
        fold_err = 0.0
        for f in folds:
            val_idx = f
            tr_idx = np.setdiff1d(idx, val_idx, assume_unique=True)

            R_tr, P_tr = R[tr_idx], P[tr_idx]
            R_val, P_val = R[val_idx], P[val_idx]

            RtR = R_tr.T @ R_tr
            RtP = R_tr.T @ P_tr

            beta = np.linalg.solve(RtR + aR * np.eye(d), RtP)
            P_hat = R_val @ beta
            fold_err += np.mean((P_val - P_hat) ** 2)

        cv_mse.append(fold_err / n_folds)

    cv_mse = np.asarray(cv_mse)
    best_idx = int(np.argmin(cv_mse))
    best_alphaR = float(alphaR_grid[best_idx])
    return best_alphaR, cv_mse


# ============================================================
# Result container
# ============================================================

@dataclass
class HermiteSieveRResult:
    beta: np.ndarray        # Hermite coefficients (length J+1)
    degree: int             # J (order of Hermite polynomial)
    alphaR: float           # selected grid alphaR
    alpha: float            # scaled penalty alpha = alphaR * chat * n^(1/3)
    negtol: float           # lower bound used in inequality constraints
    into1: bool             # whether integral constraint was enforced

    x_grid: np.ndarray      # Hermite x-grid (x)
    R_grid: np.ndarray      # return grid R_t
    RND_R: np.ndarray       # density f(R)

    ST_grid: np.ndarray     # underlying levels S_T on the x_grid
    RND_ST: np.ndarray      # RND over S_T on ST_grid

    K: np.ndarray           # call strikes used in estimation
    P: np.ndarray           # call prices used in estimation
    C_fit: np.ndarray       # fitted call prices on K

    st: float               # spot
    rf: float               # risk-free (same units as tau)
    tau: float              # time to maturity
    Sigma: float            # volatility per sqrt(time unit of tau)
    ncall: int              # number of calls used

    m_grid: int             # number of integration grid points (x)
    degree_mode: str        # 'fixed', 'auto', 'cv_grid', 'oracle'
    degree_candidates: Optional[np.ndarray] = None  # tried J-values
    degree_scores: Optional[np.ndarray] = None      # score(J)

    # =============== New convenience methods ====================

    def call_prices(self, K_new: np.ndarray) -> np.ndarray:
        """
        Evaluate fitted call prices on an arbitrary strike grid K_new.

        Uses the same Hermite representation as in the estimator and
        reconstructs the payoff matrix G(K_new, x), then integrates.
        """
        K_new = np.asarray(K_new, float)
        Sigma = float(self.Sigma)
        st = float(self.st)
        rf = float(self.rf)
        tau = float(self.tau)
        sqrt_tau = np.sqrt(tau)

        x = self.x_grid
        J = self.degree

        # Hermite basis and weights on the original x-grid
        H = hermite_basis(x, J)       # (m_grid, J+1)
        Ts = H.T                      # (J+1, m_grid)
        ci = trapz_weights(x)         # (m_grid,)

        # Build payoff matrix G for K_new
        m_loc = x.size
        n_new = K_new.size
        G_new = np.empty((n_new, m_loc), float)

        coeff = st
        exp_x = np.exp(Sigma * sqrt_tau * x)

        for i, K_i in enumerate(K_new):
            k_i = (np.log(K_i / st) - rf * tau) / (Sigma * sqrt_tau)
            temp = coeff * (exp_x - np.exp(Sigma * sqrt_tau * k_i))
            temp[temp < 0.0] = 0.0
            G_new[i, :] = temp

        Gw_new = G_new * ci[None, :]        # (n_new x m_loc)
        R_mat_new = Gw_new @ Ts.T           # (n_new x (J+1))

        # Fitted call prices on K_new
        return R_mat_new @ self.beta

    def rnd_ST(self, S_grid_new: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitted risk-neutral density q(S_T) on an arbitrary
        S_grid_new (positive underlying prices).
        """
        S_grid_new = np.asarray(S_grid_new, float)
        Sigma = float(self.Sigma)
        st = float(self.st)
        rf = float(self.rf)
        tau = float(self.tau)
        sqrt_tau = np.sqrt(tau)

        if np.any(S_grid_new <= 0):
            raise ValueError("S_grid_new must be strictly positive.")

        # Map S_T -> x via the same transform used in the constructor:
        # S_T = st * exp(rf*tau + Sigma*sqrt(tau) * x)
        # => x = (ln(S_T/st) - rf*tau) / (Sigma * sqrt(tau))
        x_new = (np.log(S_grid_new / st) - rf * tau) / (Sigma * sqrt_tau)

        # Hermite basis at x_new
        H_new = hermite_basis(x_new, self.degree)   # (len(S_grid_new), J+1)

        # Numerator: h(x)^T beta
        num = H_new @ self.beta                    # (len(S_grid_new),)

        # q(S_T) = [h(x)^T beta] / (Sigma * sqrt(tau) * S_T)
        q_new = num / (Sigma * sqrt_tau * S_grid_new)

        return q_new

    def rnd_R(self, R_grid_new: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitted density of the (log) return R on arbitrary R_grid_new.
        Recall the stored R_grid corresponds to R = rf*tau + Sigma*sqrt(tau)*x.
        """
        R_grid_new = np.asarray(R_grid_new, float)
        Sigma = float(self.Sigma)
        rf = float(self.rf)
        tau = float(self.tau)
        sqrt_tau = np.sqrt(tau)

        # Map R -> x: R = rf*tau + Sigma*sqrt(tau) * x
        x_new = (R_grid_new - rf * tau) / (Sigma * sqrt_tau)

        H_new = hermite_basis(x_new, self.degree)   # (len(R_grid_new), J+1)
        # f_R(R) = [h(x)^T beta] / (Sigma * sqrt(tau))
        f_R_new = (H_new @ self.beta) / (Sigma * sqrt_tau)
        return f_R_new


# ============================================================
# True BS RND
# ============================================================

def bs_lognormal_rnd(S0, rf, sigma, tau, S_grid):
    """
    True BS risk-neutral density of S_T (lognormal).
    """
    S_grid = np.asarray(S_grid, float)
    mu_log = np.log(S0) + (rf - 0.5 * sigma**2) * tau
    var_log = sigma**2 * tau
    logS = np.log(S_grid)
    return (1.0 / (S_grid * np.sqrt(2 * np.pi * var_log))
            * np.exp(-0.5 * (logS - mu_log)**2 / var_log))


# ============================================================
# Unified Hermite sieve (calls-only) with oracle option
# ============================================================

def hermite_sieve_calls_only(
    Kcall: np.ndarray,
    Pcall: np.ndarray,
    st: float,
    rf: float,
    tau: float,
    Sigma: Optional[float] = None,
    # control parameters
    method: int = 1,
    degree: Optional[int] = None,
    degree_mode: str = "auto",      # 'auto', 'fixed', 'cv_grid', 'oracle'
    degree_grid: Optional[Sequence[int]] = None,
    negtol: float = -1e-3,
    into1: bool = False,
    m_grid: int = 1000,
    minR: float = -10.0,
    maxR: float = 10.0,
    alphaR_grid: Optional[np.ndarray] = None,
    nFold: int = 10,
    random_state: int = 0,
    # oracle-only arguments:
    S_true: Optional[np.ndarray] = None,
    q_true: Optional[np.ndarray] = None,
) -> HermiteSieveRResult:
    """
    Unified Hermite spectral sieve estimator using *call options only*.

    degree_mode:
      - 'auto'   : J = ceil(2*(n/log n)^0.2) (Lu-style default).
      - 'fixed'  : use the supplied 'degree' exactly.
      - 'cv_grid': grid-search J over 'degree_grid', choose J with smallest
                   CV error on call prices (using the *best* alphaR for each J).
      - 'oracle' : grid-search J using L2 distance to true RND (S_true, q_true).

    Oracle mode requires S_true and q_true.
    """

    # ---------- data ----------
    Kcall = np.asarray(Kcall, float)
    Pcall = np.asarray(Pcall, float)

    ncall = Kcall.size
    n = ncall

    # ---------- estimate Sigma from ATM call if not provided ----------
    if Sigma is None:
        Sigma = estimate_sigma_atm_from_calls(Kcall, Pcall, st, rf, tau)

    sqrt_tau = np.sqrt(tau)
    k = (np.log(Kcall / st) - rf * tau) / (Sigma * sqrt_tau)

    # ---------- alphaR grid ----------
    if alphaR_grid is None:
        temp = np.linspace(0.0, 1e-4, 25)
        alphaR_grid = np.unique(
            np.concatenate([temp, temp * 10, temp * 100, temp * 1000])
        )
    alphaR_grid = np.asarray(alphaR_grid, float)

    # ============================================================
    # Helper to fit a single J (core solver)
    # ============================================================
    def _fit_for_degree(J: int) -> Tuple[HermiteSieveRResult, float]:
        # Hermite x-grid
        x = np.linspace(minR, maxR, m_grid)
        m_loc = x.size

        # Hermite basis and weights
        H = hermite_basis(x, J)   # (m_loc, J+1)
        Ts = H.T                  # (J+1, m_loc)
        ci = trapz_weights(x)     # (m_loc,)

        # payoff matrix G (calls only)
        G = np.empty((n, m_loc), float)
        coeff = st
        exp_x = np.exp(Sigma * sqrt_tau * x)

        for i in range(n):
            temp = coeff * (exp_x - np.exp(Sigma * sqrt_tau * k[i]))
            temp[temp < 0.0] = 0.0
            G[i, :] = temp

        # weighted by ci
        G_weighted = G * ci[None, :]        # (n x m_loc)
        R_mat = G_weighted @ Ts.T           # (n x d)
        d = J + 1
        RtR = R_mat.T @ R_mat

        # CV over alphaR
        best_alphaR, cv_vals = cv_alphaR(
            R=R_mat, P=Pcall, alphaR_grid=alphaR_grid,
            n_folds=nFold, random_state=random_state
        )

        # scale alpha like in Lu/R code: alpha = alphaR * chat * n^(1/3)
        chat = RtR[0, 0] / n
        alpha = float(best_alphaR * chat * (n ** (1.0 / 3.0)))

        if method == 1:
            Q_pen = alpha * np.eye(d)
        else:
            raise NotImplementedError("Only method=1 is implemented here.")

        # objective and gradient
        def objective(beta):
            beta = np.asarray(beta)
            resid = Pcall - R_mat @ beta
            return (resid @ resid) / n + beta @ (Q_pen @ beta)

        def grad(beta):
            beta = np.asarray(beta)
            resid = Pcall - R_mat @ beta
            return (-2.0 / n) * (R_mat.T @ resid) + 2.0 * (Q_pen @ beta)

        # constraints: integral (optional) + pointwise lower bound
        A1 = ci @ Ts.T      # (d,)
        A2 = Ts.T           # (m_loc, d)

        cons = []
        if into1:
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda b, A=A1: float(A @ b - 1.0),
                    "jac": lambda b, A=A1: A,
                }
            )

        for row in A2:
            r_row = row.copy()

            def fun(b, r_row=r_row):
                return float(r_row @ b - negtol)

            def jac(b, r_row=r_row):
                return r_row

            cons.append({"type": "ineq", "fun": fun, "jac": jac})

        # initial β from unconstrained ridge
        beta0 = np.linalg.solve(RtR + Q_pen, R_mat.T @ Pcall)

        res_opt = minimize(
            objective,
            beta0,
            method="SLSQP",
            jac=grad,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
        )

        beta_hat = np.asarray(res_opt.x, float)

        # RND in R- and S_T-space
        RND_R = (Ts.T @ beta_hat) / (Sigma * sqrt_tau)  # (m_loc,)
        R_grid = x * Sigma * sqrt_tau + rf * tau
        ST_grid = np.exp(x * Sigma * sqrt_tau + rf * tau + np.log(st))
        RND_ST = (Ts.T @ beta_hat) / (Sigma * sqrt_tau * ST_grid)

        C_fit = R_mat @ beta_hat

        # Score for this J: use the minimum CV MSE over alphaR
        J_score = float(np.min(cv_vals))

        res_local = HermiteSieveRResult(
            beta=beta_hat,
            degree=J,
            alphaR=float(best_alphaR),
            alpha=alpha,
            negtol=negtol,
            into1=into1,
            x_grid=x,
            R_grid=R_grid,
            RND_R=RND_R,
            ST_grid=ST_grid,
            RND_ST=RND_ST,
            K=Kcall,
            P=Pcall,
            C_fit=C_fit,
            st=st,
            rf=rf,
            tau=tau,
            Sigma=Sigma,
            ncall=n,
            m_grid=m_grid,
            degree_mode=degree_mode,
            degree_candidates=None,
            degree_scores=None,
        )

        return res_local, J_score

    # ============================================================
    # Degree selection branches
    # ============================================================

    # fixed J
    if degree_mode == "fixed":
        if degree is None:
            raise ValueError("degree_mode='fixed' requires 'degree' to be specified.")
        res_single, _ = _fit_for_degree(int(degree))
        return res_single

    # automatic J
    if degree_mode == "auto":
        J_auto = int(np.ceil(2.0 * (n / np.log(n))**0.2))
        res_single, _ = _fit_for_degree(J_auto)
        res_single.degree_mode = "auto"
        return res_single

    # CV grid over J (calls only)
    if degree_mode == "cv_grid":
        if degree_grid is None:
            degree_grid = list(range(1, 31))

        degree_grid = [int(J) for J in degree_grid if J >= 1]
        if len(degree_grid) == 0:
            raise ValueError("degree_grid is empty or invalid.")

        results = []
        scores = []

        for J in degree_grid:
            res_J, score_J = _fit_for_degree(J)
            results.append(res_J)
            scores.append(score_J)

        scores = np.asarray(scores, float)
        best_idx = int(np.argmin(scores))
        best_res = results[best_idx]
        best_res.degree_mode = "cv_grid"
        best_res.degree_candidates = np.asarray(degree_grid, int)
        best_res.degree_scores = scores

        return best_res

    # ORACLE selection over J using true RND
    if degree_mode == "oracle":
        if S_true is None or q_true is None:
            raise ValueError("degree_mode='oracle' requires S_true and q_true.")

        S_true = np.asarray(S_true, float)
        q_true = np.asarray(q_true, float)
        q_true /= np.trapz(q_true, S_true)

        if degree_grid is None:
            degree_grid = list(range(1, 31))

        degree_grid = [int(J) for J in degree_grid if J >= 1]
        if len(degree_grid) == 0:
            raise ValueError("degree_grid is empty or invalid.")

        oracle_scores: Dict[int, float] = {}
        results: Dict[int, HermiteSieveRResult] = {}

        for J in degree_grid:
            res_J, _ = _fit_for_degree(J)

            S_hat = res_J.ST_grid
            q_hat = res_J.RND_ST.copy()
            q_hat[q_hat < 0.0] = 0.0
            q_hat /= np.trapz(q_hat, S_hat)

            q_hat_on_true = np.interp(S_true, S_hat, q_hat, left=0.0, right=0.0)
            l2_err = float(np.sqrt(np.trapz((q_true - q_hat_on_true) ** 2, S_true)))

            oracle_scores[J] = l2_err
            results[J] = res_J

        best_J = min(oracle_scores, key=oracle_scores.get)
        best_res = results[best_J]

        best_res.degree_mode = "oracle"
        best_res.degree_candidates = np.asarray(list(oracle_scores.keys()), int)
        best_res.degree_scores = np.asarray(
            [oracle_scores[J] for J in best_res.degree_candidates], float
        )

        return best_res

    raise ValueError("degree_mode must be one of: 'auto', 'fixed', 'cv_grid', 'oracle'.")


# ============================================================
# Demo / test harness
# ============================================================

if __name__ == "__main__":
    # ---------- True BS parameters ----------
    S0    = 100.0
    r     = 0.02       # continuous risk-free
    sigma = 0.25       # annual vol
    T     = 0.5        # maturity in years

    # ---------- Generate strikes and call prices ----------
    Kcall = np.linspace(50, 150, 100)
    Pcall_true = bs_call_price(S0, Kcall, r, sigma, T)

    # Optional: add noise (e.g. 20% Gaussian relative noise)
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.15 * Pcall_true)
    Pcall_obs = Pcall_true + noise

    # ---------- True RND on a fine S-grid (oracle) ----------
    S_true = np.linspace(50, 150, 50)
    q_true = bs_lognormal_rnd(S0, r, sigma, T, S_true)

    # ---------- 1) Call-based CV over J ----------
    res_cv = hermite_sieve_calls_only(
        Kcall=Kcall,
        Pcall=Pcall_obs,
        st=S0,
        rf=r,
        tau=T,
        Sigma=None,
        degree_mode="auto",
        degree_grid=range(1, 15),
        m_grid=100,
        nFold=10,
        random_state=0,
        alphaR_grid=[0.002],
    )

    print("CALL-CV chosen J:", res_cv.degree)
    if res_cv.degree_candidates is not None:
        print("CALL-CV tried J:", res_cv.degree_candidates)
        print("CALL-CV scores per J:", res_cv.degree_scores)

    # ---------- 2) Oracle CV over J using true RND ----------
    res_oracle = hermite_sieve_calls_only(
        Kcall=Kcall,
        Pcall=Pcall_obs,
        st=S0,
        rf=r,
        tau=T,
        Sigma=None,
        degree_mode="oracle",
        degree_grid=range(1, 15),
        m_grid=100,
        nFold=10,
        random_state=0,
        alphaR_grid=[0.002],
        S_true=S_true,
        q_true=q_true,
    )

    print("\nORACLE chosen J:", res_oracle.degree)
    if res_oracle.degree_candidates is not None:
        print("ORACLE tried J:", res_oracle.degree_candidates)
        print("ORACLE L2 scores per J:", res_oracle.degree_scores)

    # ---------- Compare oracle RND vs true ----------
    S_hat = res_oracle.ST_grid
    q_hat = res_oracle.RND_ST.copy()
    q_hat[q_hat < 0.0] = 0.0
    q_hat /= np.trapz(q_hat, S_hat)
    q_hat_on_true = np.interp(S_true, S_hat, q_hat, left=0.0, right=0.0)
    C_fit=res_oracle.C_fit
    l2_oracle = np.sqrt(np.trapz((q_true - q_hat_on_true)**2, S_true))
    print(f"Oracle L2(best J) = {l2_oracle:.6e}")

    # ---------- Plot RNDs ----------
    plt.figure(figsize=(7, 4))
    plt.plot(S_true, q_true / np.trapz(q_true, S_true), "k--", label="True BS RND")
    plt.plot(S_hat, q_hat, label="Hermite sieve RND (oracle J)")
    plt.plot(res_cv.ST_grid, res_cv.RND_ST.copy(), label="Hermite sieve RND")

    plt.xlabel("S_T")
    plt.ylabel("Density")
    plt.title("RND: Black–Scholes vs Hermite sieve (oracle J)")
    plt.legend()
    plt.grid(True)

    # ---------- Plot call prices ----------
    plt.figure(figsize=(7, 4))
    plt.plot(Kcall, Pcall_true, "k.", label="True BS calls")
    plt.plot(Kcall, Pcall_obs, "x", alpha=0.4, label="Noisy calls")
    plt.plot(res_oracle.K, res_oracle.C_fit, "-", label="Sieve-implied calls (oracle J)")
    plt.plot(res_cv.K, res_cv.C_fit, "-", label="Sieve-implied calls")

    plt.xlabel("Strike")
    plt.ylabel("Call price")
    plt.title("Call price fit: true vs noisy vs sieve (oracle J)")
    plt.legend()
    plt.grid(True)

    plt.show()
