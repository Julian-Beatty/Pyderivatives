# ============================================================
# HKDE (Heston + Kou Double-Exponential Jumps) - Calls + RND
# Faster version + Stable Vega-Weighted Calibration
#   - cache Gauss-Legendre nodes/weights
#   - vectorize call pricing across strikes for each maturity
#   - reuse CF evaluations within a maturity
#   - calibration groups quotes by maturity (fast)
#   - NEW: robust capped inverse-vega weighting (auto IV extraction)
#
# NEW (for your pipeline consistency):
#   - fit_to_calls now accepts bounds in:
#       * array form: (lb_array, ub_array)
#       * dict form:  (lb_dict,  ub_dict)  with keys:
#           "v0","theta","kappa","sigma_v","rho","lam","p_up","eta1","eta2"
#   - DEFAULT bounds (if bounds=None) set to your conservative box:
#       v0,theta >= 0.005
#       kappa >= 0.30
#       sigma_v >= 0.10
#       rho in [-0.85, 0.8]
#       lam in [0.02, 1.50]
#       p_up in [0.15, 0.85]
#       eta1,eta2 in [4, 30]
#
# NEW:
#   - prints using_default_bounds=True/False exactly once per calibration
# ============================================================

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union
from scipy.optimize import least_squares, brentq
import matplotlib.pyplot as plt
from scipy.stats import norm


# -----------------------------
# Black–Scholes helpers (for IV + Vega weights)
# -----------------------------
def bs_call_price(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)
    if sigma <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return float(S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_implied_vol_call(
    C: float, S0: float, K: float, T: float, r: float, q: float,
    sigma_lo: float = 1e-6, sigma_hi: float = 6.0
) -> float:
    """
    Robust implied vol for a CALL via brentq.
    Returns np.nan if price is outside arbitrage bounds.
    """
    if T <= 0 or S0 <= 0 or K <= 0:
        return np.nan

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    lower = max(S0 * disc_q - K * disc_r, 0.0)
    upper = S0 * disc_q
    if not (lower - 1e-10 <= C <= upper + 1e-10):
        return np.nan

    def f(sig: float) -> float:
        return bs_call_price(S0, K, T, r, q, sig) - C

    f_lo = f(sigma_lo)
    f_hi = f(sigma_hi)
    if np.isnan(f_lo) or np.isnan(f_hi):
        return np.nan
    if f_lo * f_hi > 0:
        for s_hi in (10.0, 20.0):
            f_hi2 = f(s_hi)
            if f_lo * f_hi2 <= 0:
                return float(brentq(f, sigma_lo, s_hi, maxiter=200))
        return np.nan

    return float(brentq(f, sigma_lo, sigma_hi, maxiter=200))


def bs_vega(S0: float, K: float, T: float, r: float, q: float, iv: float) -> float:
    if T <= 0 or iv <= 0 or S0 <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S0 / K) + (r - q + 0.5 * iv * iv) * T) / (iv * np.sqrt(T))
    return float(S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def capped_inv_vega_weight(
    S0: float, K: float, T: float, r: float, q: float, iv: float,
    vega_floor: float = 1e-3, w_cap: float = 2e3
) -> float:
    """
    Stable inverse-vega weights.
    - vega_floor prevents division by ~0 for very short maturities
    - w_cap prevents a few quotes from dominating the objective
    """
    v = bs_vega(S0, K, T, r, q, iv)
    w = 1.0 / max(v, vega_floor)
    return float(min(w, w_cap))


# -----------------------------
# Parameters
# -----------------------------
@dataclass
class HKDEParams:
    v0: float
    theta: float
    kappa: float
    sigma_v: float
    rho: float
    lam: float
    p_up: float
    eta1: float
    eta2: float


# -----------------------------
# HKDE Model
# -----------------------------
class HKDEModel:
    """
    HKDE = Heston stochastic volatility + Kou double-exponential jumps.
    Faster implementation:
      - cache quadrature nodes/weights
      - vectorized pricing across strikes per maturity
      - reuse CF evaluations within a maturity
      - calibration groups by maturity

    Bounds:
      - array form: (lb_array, ub_array) length 9
      - dict form:  (lb_dict,  ub_dict) keys in _PARAM_NAMES
      - default: if bounds=None, uses your conservative box

    NEW:
      - fit_to_calls prints using_default_bounds=True/False once
    """

    _PARAM_NAMES = ("v0", "theta", "kappa", "sigma_v", "rho", "lam", "p_up", "eta1", "eta2")

    def __init__(self, S0: float, r: float, q: float = 0.0):
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self.params_: Optional[HKDEParams] = None
        self._quad_cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]] = {}

    # ---- Cached Gauss–Legendre on [0,U]
    def gauss_legendre_0U(self, n: int, U: float) -> Tuple[np.ndarray, np.ndarray]:
        key = (int(n), float(U))
        if key in self._quad_cache:
            return self._quad_cache[key]
        x, w = np.polynomial.legendre.leggauss(n)
        u = 0.5 * (x + 1.0) * U
        wu = 0.5 * U * w
        u = np.asarray(u, float)
        u = np.where(np.abs(u) < 1e-12, 1e-12, u)
        self._quad_cache[key] = (u, wu)
        return u, wu

    # ---- Kou jump CF pieces ----
    @staticmethod
    def _phi_J(u: np.ndarray, p_up: float, eta1: float, eta2: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        iu = 1j * u
        return (p_up * eta1 / (eta1 - iu)) + ((1.0 - p_up) * eta2 / (eta2 + iu))

    @staticmethod
    def _kappa_J(p_up: float, eta1: float, eta2: float) -> float:
        Ej = (p_up * eta1 / (eta1 - 1.0)) + ((1.0 - p_up) * eta2 / (eta2 + 1.0))
        return float(Ej - 1.0)

    # ---- Heston CF (Little Trap, complex-safe) ----
    @staticmethod
    def _cf_heston(
        u: np.ndarray, T: float, S0: float, r: float, q: float,
        v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
        drift_adj: float
    ) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        iu = 1j * u

        x0 = np.log(S0)
        a = kappa * theta
        b = kappa

        d = np.sqrt((rho * sigma_v * iu - b) ** 2 + sigma_v ** 2 * (iu + u * u))
        gp = (b - rho * sigma_v * iu + d) / (b - rho * sigma_v * iu - d)
        g = 1.0 / gp
        exp_minus_dT = np.exp(-d * T)

        eps = 1e-16
        denom = (1.0 - g * exp_minus_dT) + eps
        denom0 = (1.0 - g) + eps

        C = (iu * (x0 + (r - q - drift_adj) * T) +
             (a / (sigma_v ** 2)) * ((b - rho * sigma_v * iu - d) * T
                                     - 2.0 * np.log(denom / denom0)))
        D = ((b - rho * sigma_v * iu - d) / (sigma_v ** 2)) * ((1.0 - exp_minus_dT) / denom)

        return np.exp(C + D * v0)

    # ---- Full HKDE CF for ln S_T ----
    def cf(self, u: np.ndarray, T: float, p: HKDEParams) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        if p.eta1 <= 1.0:
            raise ValueError("Need eta1 > 1 for martingale drift (E[e^J] exists).")

        kJ = self._kappa_J(p.p_up, p.eta1, p.eta2)
        drift_adj = p.lam * kJ

        phi_h = self._cf_heston(
            u=u, T=T, S0=self.S0, r=self.r, q=self.q,
            v0=p.v0, kappa=p.kappa, theta=p.theta, sigma_v=p.sigma_v, rho=p.rho,
            drift_adj=drift_adj
        )
        phiJ = self._phi_J(u, p.p_up, p.eta1, p.eta2)
        phi_kou = np.exp(p.lam * T * (phiJ - 1.0))
        return phi_h * phi_kou

    # ============================================================
    # Vectorized call pricing across a strike vector (fixed T)
    # ============================================================
    def call_prices(
        self, K_vec: np.ndarray, T: float, p: HKDEParams,
        Umax: float = 200.0, n_quad: int = 96
    ) -> np.ndarray:
        K_vec = np.asarray(K_vec, float).ravel()
        if T <= 0:
            return np.maximum(self.S0 - K_vec, 0.0)

        u, w = self.gauss_legendre_0U(n_quad, Umax)
        lnK = np.log(K_vec)

        # φ_X(-i) = E[S_T] = S0 * exp((r-q)T) under Q
        phi_mi = self.S0 * np.exp((self.r - self.q) * T)

        phi_u = self.cf(u, T, p)
        phi_u_shift = self.cf(u - 1j, T, p)

        E = np.exp(-1j * np.outer(u, lnK))

        integrand_P2 = np.real(E * (phi_u[:, None] / (1j * u[:, None])))
        P2 = 0.5 + (1.0 / np.pi) * (w @ integrand_P2)

        integrand_P1 = np.real(E * (phi_u_shift[:, None] / (1j * u[:, None] * phi_mi)))
        P1 = 0.5 + (1.0 / np.pi) * (w @ integrand_P1)

        C = self.S0 * np.exp(-self.q * T) * P1 - K_vec * np.exp(-self.r * T) * P2
        return np.maximum(C, 0.0)

    def call_surface(
        self, K_grid: np.ndarray, T_grid: np.ndarray, p: HKDEParams,
        Umax: float = 200.0, n_quad: int = 96
    ) -> np.ndarray:
        K_grid = np.asarray(K_grid, float)
        T_grid = np.asarray(T_grid, float)
        out = np.zeros((T_grid.size, K_grid.size), float)
        for i, T in enumerate(T_grid):
            out[i, :] = self.call_prices(K_grid, float(T), p, Umax=Umax, n_quad=n_quad)
        return out

    # ---- RND (reuse CF + vectorize over s_grid)
    def rnd_q(
        self, s_grid: np.ndarray, T: float, p: HKDEParams,
        Umax: float = 200.0, n_quad: int = 128
    ) -> np.ndarray:
        s_grid = np.asarray(s_grid, float)
        if np.any(s_grid <= 0):
            raise ValueError("s_grid must be positive.")
        x = np.log(s_grid)

        u, w = self.gauss_legendre_0U(n_quad, Umax)
        phi_u = self.cf(u, T, p)

        E = np.exp(-1j * np.outer(u, x))
        fx = (1.0 / np.pi) * (w @ np.real(E * phi_u[:, None]))
        q = np.maximum(fx / s_grid, 0.0)
        return q

    def rnd_surface(
        self, s_grid: np.ndarray, T_grid: np.ndarray, p: HKDEParams,
        Umax: float = 200.0, n_quad: int = 128
    ) -> np.ndarray:
        s_grid = np.asarray(s_grid, float)
        T_grid = np.asarray(T_grid, float)
        out = np.zeros((T_grid.size, s_grid.size), float)
        for i, T in enumerate(T_grid):
            out[i, :] = self.rnd_q(s_grid, float(T), p, Umax=Umax, n_quad=n_quad)
        return out

    # -----------------------------
    # bounds normalization (array or dict) + default
    # -----------------------------
    @classmethod
    def _normalize_bounds(
        cls,
        bounds: Optional[Tuple[Union[np.ndarray, Dict[str, float]], Union[np.ndarray, Dict[str, float]]]]
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        using_default_bounds = False

        # DEFAULT (your conservative box)
        if bounds is None:
            using_default_bounds = True
            lb_d = dict(
                v0=0.005, theta=0.005, kappa=0.30, sigma_v=0.10, rho=-0.85,
                lam=0.02, p_up=0.15, eta1=4.0, eta2=4.0
            )
            ub_d = dict(
                v0=0.9, theta=0.9, kappa=12.0, sigma_v=0.75, rho=0.8,
                lam=1.50, p_up=0.85, eta1=30.0, eta2=30.0
            )
            bounds = (lb_d, ub_d)

        lb, ub = bounds

        # dict form
        if isinstance(lb, dict) or isinstance(ub, dict):
            if not (isinstance(lb, dict) and isinstance(ub, dict)):
                raise TypeError("If using dict bounds, both lb and ub must be dicts.")
            missing = [n for n in cls._PARAM_NAMES if (n not in lb) or (n not in ub)]
            if missing:
                raise KeyError(f"Missing bounds for: {missing}")
            lbv = np.array([float(lb[n]) for n in cls._PARAM_NAMES], float)
            ubv = np.array([float(ub[n]) for n in cls._PARAM_NAMES], float)
            return lbv, ubv, using_default_bounds

        # array form (legacy)
        lb = np.asarray(lb, float).ravel()
        ub = np.asarray(ub, float).ravel()
        if lb.size != len(cls._PARAM_NAMES) or ub.size != len(cls._PARAM_NAMES):
            raise ValueError(f"Array bounds must have length {len(cls._PARAM_NAMES)}.")
        return lb, ub, using_default_bounds

    # -----------------------------
    # Calibration (stable vega weighting inside)
    # -----------------------------
    def fit_to_calls(
        self,
        K_obs, T_obs, C_obs,
        x0: HKDEParams,
        bounds=None,
        iv_obs=None,
        use_vega_weights: bool = True,
        vega_floor: float = 1e-3,
        w_cap: float = 2e3,
        Umax: float = 200.0,
        n_quad: int = 96,
        verbose: int = 1,
        max_nfev: int = 200
    ):

        K_obs = np.asarray(K_obs, float).ravel()
        T_obs = np.asarray(T_obs, float).ravel()
        C_obs = np.asarray(C_obs, float).ravel()
        assert K_obs.size == T_obs.size == C_obs.size

        m = (
            np.isfinite(K_obs) & np.isfinite(T_obs) & np.isfinite(C_obs) &
            (K_obs > 0) & (T_obs > 0) & (C_obs >= 0)
        )
        K_obs, T_obs, C_obs = K_obs[m], T_obs[m], C_obs[m]

        # ---- AUTO-IV and weights (computed once, not inside residual loop)
        weights = None
        if use_vega_weights:
            if iv_obs is None:
                iv_obs = np.array([
                    bs_implied_vol_call(float(C_obs[i]), self.S0, float(K_obs[i]), float(T_obs[i]), self.r, self.q)
                    for i in range(C_obs.size)
                ], float)
            else:
                iv_obs = np.asarray(iv_obs, float).ravel()[m]

            weights = np.ones_like(C_obs, float)
            for i in range(C_obs.size):
                if np.isfinite(iv_obs[i]) and iv_obs[i] > 0:
                    weights[i] = capped_inv_vega_weight(
                        self.S0, float(K_obs[i]), float(T_obs[i]), self.r, self.q, float(iv_obs[i]),
                        vega_floor=float(vega_floor), w_cap=float(w_cap)
                    )
                else:
                    weights[i] = 1.0

        # ---- Pre-group indices by maturity
        T_unique = np.unique(T_obs)
        idx_by_T = [np.where(T_obs == t)[0] for t in T_unique]

        def pack(pp: HKDEParams) -> np.ndarray:
            return np.array([pp.v0, pp.theta, pp.kappa, pp.sigma_v, pp.rho,
                             pp.lam, pp.p_up, pp.eta1, pp.eta2], float)

        def unpack(x: np.ndarray) -> HKDEParams:
            return HKDEParams(
                v0=float(x[0]), theta=float(x[1]), kappa=float(x[2]), sigma_v=float(x[3]), rho=float(x[4]),
                lam=float(x[5]), p_up=float(x[6]), eta1=float(x[7]), eta2=float(x[8])
            )

        # normalize bounds (dict or array; default if None) + PRINT FLAG
        lb, ub, using_default_bounds = self._normalize_bounds(bounds)
        bounds_vec = (lb, ub)
        print(f"using_default_bounds={using_default_bounds}")

        x0v = pack(x0)

        def residuals(x: np.ndarray) -> np.ndarray:
            pp = unpack(x)

            # safe region
            if (
                (pp.eta1 <= 1.0) or (pp.sigma_v <= 1e-6) or (pp.v0 <= 0) or (pp.theta <= 0) or (pp.kappa <= 0)
                or (abs(pp.rho) >= 0.999)
            ):
                return np.full(C_obs.shape, 1e6, dtype=float)

            model = np.empty_like(C_obs)

            for t, idx in zip(T_unique, idx_by_T):
                K_block = K_obs[idx]
                model[idx] = self.call_prices(K_block, float(t), pp, Umax=Umax, n_quad=n_quad)

            if not np.all(np.isfinite(model)):
                return np.full(C_obs.shape, 1e6, dtype=float)

            err = model - C_obs
            if weights is not None:
                return weights * err
            return err

        res = least_squares(
            residuals, x0v,
            bounds=bounds_vec,
            method="trf",
            verbose=2 if verbose else 0,
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=int(max_nfev)
        )
        self.params_ = unpack(res.x)
        return self.params_

    # -----------------------------
    # Plotting
    # -----------------------------
    @staticmethod
    def _plot_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str, xlabel: str, ylabel: str, zlabel: str):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        plt.tight_layout()
        plt.show()

    def plot_call_surface(self, K_grid: np.ndarray, T_grid: np.ndarray, C_surf: np.ndarray, title: str = "Call surface"):
        KK, TT = np.meshgrid(K_grid, T_grid)
        self._plot_surface(KK, TT, C_surf, title, "Strike K", "Maturity T (years)", "Call price C")

    def plot_rnd_surface(self, s_grid: np.ndarray, T_grid: np.ndarray, q_surf: np.ndarray, title: str = "Risk-neutral density surface"):
        SS, TT = np.meshgrid(s_grid, T_grid)
        self._plot_surface(SS, TT, q_surf, title, "Terminal price s", "Maturity T (years)", "q_T(s)")


# ============================================================
# DEMO
# ============================================================
if __name__ == "__main__":
    S0 = 100.0
    r = 0.03
    q = 0.00

    model = HKDEModel(S0=S0, r=r, q=q)

    true_p = HKDEParams(
        v0=0.04, theta=0.06, kappa=2.5, sigma_v=0.60, rho=-0.55,
        lam=1.2, p_up=0.35, eta1=8.0, eta2=12.0
    )

    # NOTE: include very short + long maturities to stress test
    T_grid = np.array([0.01, 0.02, 0.05, 0.10, 0.35, 0.75, 1.50, 2.00])
    K_grid = np.linspace(60, 140, 100)

    C_true = model.call_surface(K_grid, T_grid, true_p, Umax=220.0, n_quad=128)

    rng = np.random.default_rng(0)
    C_mkt = np.maximum(C_true + rng.normal(scale=0.04 * np.maximum(C_true, 1e-4)), 0.0)

    KK, TT = np.meshgrid(K_grid, T_grid)
    K_obs = KK.ravel()
    T_obs = TT.ravel()
    C_obs = C_mkt.ravel()

    x0 = HKDEParams(
        v0=0.03, theta=0.05, kappa=1.5, sigma_v=0.50, rho=-0.3,
        lam=0.8, p_up=0.5, eta1=6.0, eta2=10.0
    )

    # --- Fit WITHOUT passing bounds -> uses your default conservative bounds
    fit_p = model.fit_to_calls(
        K_obs=K_obs, T_obs=T_obs, C_obs=C_obs,
        x0=x0,
        bounds=None,  # or omit entirely

        # vega-weighting Optional (stable)
        use_vega_weights=True,
        vega_floor=1e-3,
        w_cap=2e3,

        Umax=260.0, n_quad=160,
        verbose=1,
        max_nfev=400
    )
    print("\nTrue params:", true_p)
    print("Fit  params:", fit_p)

    C_fit = model.call_surface(K_grid, T_grid, fit_p, Umax=220.0, n_quad=128)
    model.plot_call_surface(K_grid, T_grid, C_mkt, title="Market (synthetic) call surface")
    model.plot_call_surface(K_grid, T_grid, C_fit, title="Fitted HKDE call surface")

    s_grid = np.linspace(20, 220, 201)
    q_surf = model.rnd_surface(s_grid, T_grid, fit_p, Umax=260.0, n_quad=160)
    model.plot_rnd_surface(s_grid, T_grid, q_surf, title="Fitted HKDE risk-neutral density surface")

    # --- Optional: example dict bounds override (prints using_default_bounds=False)
    # lb_d = dict(v0=0.005, theta=0.005, kappa=0.30, sigma_v=0.10, rho=-0.85, lam=0.02, p_up=0.15, eta1=4.0, eta2=4.0)
    # ub_d = dict(v0=0.9,   theta=0.9,   kappa=12.0, sigma_v=0.75, rho=0.8,   lam=1.50, p_up=0.85, eta1=30.0, eta2=30.0)
    # fit_p2 = model.fit_to_calls(K_obs, T_obs, C_obs, x0=x0, bounds=(lb_d, ub_d))
