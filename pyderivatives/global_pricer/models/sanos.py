from __future__ import annotations

"""
SANOS-style smooth, strictly arbitrage-free non-parametric call surface.

Production-style implementation inspired by:
    Buehler, Horvath, Kratsios, Limmer, Saqur (2026),
    "Smooth strictly Arbitrage-free Non-parametric Option Surfaces".

The model works in normalized forward units. At each fitted expiry T_j,

    c_j(k) = sum_i q[j, i] * BSCall(anchor_i, k, smoothness * V_j),

where q_j is a discrete unit-mean probability distribution over normalized
anchor strikes and V_j is an increasing ATM total-variance backbone.

The q_j are fitted globally by linear programming. The common anchor grid is
constructed from the union of the observed forward-normalized strikes, with
the unit-forward anchor 1.0 added when it is not already present.
Constraints enforce:
  * q_j >= 0,
  * sum_i q[j,i] = 1,
  * sum_i q[j,i] * anchor_i = 1,
  * increasing discrete call prices across fitted expiries (convex order).

The fitted arrays are retained on the model instance. FitResult.params remains
a flat scalar dictionary, preserving the existing GlobalSurfacePricer design.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, linprog
from scipy.stats import norm

from .base import GlobalModel, FitResult
from ..registry import register_model


# -----------------------------------------------------------------------------
# Black-Scholes helpers in normalized/pure units
# -----------------------------------------------------------------------------

def _pure_bs_call(spot: np.ndarray | float, strike: np.ndarray | float, variance: float) -> np.ndarray:
    """Undiscounted Black-Scholes call with E[S_T]=spot and total variance."""
    s = np.asarray(spot, float)
    k = np.asarray(strike, float)
    s, k = np.broadcast_arrays(s, k)
    v = float(max(variance, 0.0))

    if v <= 1e-16:
        return np.maximum(s - k, 0.0)

    sv = np.sqrt(v)
    out = np.maximum(s - k, 0.0)
    m = (s > 0.0) & (k > 0.0)
    if np.any(m):
        d1 = (np.log(s[m] / k[m]) + 0.5 * v) / sv
        d2 = d1 - sv
        out[m] = s[m] * norm.cdf(d1) - k[m] * norm.cdf(d2)
    return out


def _pure_bs_call_scalar(spot: float, strike: float, variance: float) -> float:
    return float(_pure_bs_call(np.array([spot]), np.array([strike]), variance)[0])


def _implied_total_variance_from_pure_call(k: float, c: float) -> float:
    """ATM-backbone helper: invert c = BSCall(1, k, total_variance)."""
    intrinsic = max(1.0 - float(k), 0.0)
    c = float(np.clip(c, intrinsic + 1e-12, 1.0 - 1e-12))

    def f(v: float) -> float:
        return _pure_bs_call_scalar(1.0, float(k), float(v)) - c

    lo, hi = 1e-12, 25.0
    flo, fhi = f(lo), f(hi)
    if flo >= 0.0:
        return lo
    if fhi <= 0.0:
        return hi
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))


# -----------------------------------------------------------------------------
# Internal fitted state retained on the model instance
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class _SANOSState:
    maturities: np.ndarray       # shape (M,)
    anchors: np.ndarray          # shape (N,)
    weights: np.ndarray          # shape (M,N)
    variances: np.ndarray        # shape (M,)
    smoothness: float
    time_interpolation: str
    long_variance_slope: float
    fit_mae_pure: float
    fit_rmse_pure: float
    lp_status: int
    lp_message: str


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

@register_model("sanos")
@register_model("smooth_arbitrage_free")
class SANOSModel(GlobalModel):
    """
    SANOS-style global call-surface model.

    Parameters are supplied through ``x0`` to preserve the package's existing
    model API. Recognized keys are:

      smoothness          default 0.25
      variance_floor      default 1e-8
      variance_step_floor default 1e-8
      time_code           0=linear, 1=smoothstep (default 1)
      quote_weight_power  default 0.0; weight ~= max(C, floor)^(-power)
      anchor_round_decimals
                          default 12; rounding used when deduplicating
                          forward-normalized observed strikes

    ``bounds`` and ``max_nfev`` are accepted for interface compatibility but
    are not used; calibration is a single linear program.

    There is no artificial six-quote minimum. Sparse surfaces are attempted
    with a reduced anchor grid, although extremely sparse data may still be
    weakly identified or infeasible for the linear program.
    """

    name = "sanos"

    def __init__(
        self,
        *,
        S0: float,
        r: float,
        q: float = 0.0,
        Umax: float = 500.0,
        n_quad: int = 500,
        **kwargs,
    ):
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        # Kept only for constructor compatibility with GlobalSurfacePricer.
        self.Umax = float(Umax)
        self.n_quad = int(n_quad)
        self._state: Optional[_SANOSState] = None

    # ---------------------------- normalization ----------------------------
    def _forward(self, T: np.ndarray | float) -> np.ndarray:
        T = np.asarray(T, float)
        return self.S0 * np.exp((self.r - self.q) * T)

    def _scale(self, T: np.ndarray | float) -> np.ndarray:
        """DF(T)*F(T) = S0*exp(-qT), used to normalize cash calls."""
        T = np.asarray(T, float)
        return self.S0 * np.exp(-self.q * T)

    def _to_pure(self, K: np.ndarray, T: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        F = self._forward(T)
        scale = self._scale(T)
        return K / F, C / scale

    # --------------------------- backbone variance -------------------------
    def _estimate_variance_backbone(
        self,
        K: np.ndarray,
        T: np.ndarray,
        C: np.ndarray,
        maturities: np.ndarray,
        *,
        variance_floor: float,
        variance_step_floor: float,
    ) -> np.ndarray:
        k, c = self._to_pure(K, T, C)
        V = np.empty(maturities.size, dtype=float)

        for j, tj in enumerate(maturities):
            idx = np.where(np.isclose(T, tj, rtol=0.0, atol=1e-12))[0]
            if idx.size == 0:
                raise RuntimeError("Internal maturity grouping error.")

            # Closest-to-forward quote is the most stable scalar backbone proxy.
            ii = idx[int(np.argmin(np.abs(k[idx] - 1.0)))]
            try:
                v = _implied_total_variance_from_pure_call(float(k[ii]), float(c[ii]))
            except Exception:
                # Conservative fallback based on annualized 25% volatility.
                v = 0.25**2 * float(tj)
            V[j] = max(float(v), float(variance_floor))

        # SANOS production construction requires an increasing variance curve.
        for j in range(1, V.size):
            V[j] = max(V[j], V[j - 1] + float(variance_step_floor))
        return V

    # ------------------------------ LP setup -------------------------------
    @staticmethod
    def _build_anchors_from_observed_strikes(
        k_obs: np.ndarray,
        *,
        round_decimals: int = 12,
    ) -> np.ndarray:
        """
        Construct the common SANOS anchor grid from observed strikes.

        Cash strikes are converted to forward-normalized strikes k = K / F(T).
        The anchor set is the sorted union of these normalized strikes across
        all fitted maturities. The unit-forward anchor 1.0 is always included
        so that the unit-mean martingale constraint has a natural support point.

        No artificial padding, equally spaced support, or fixed anchor count
        is imposed.
        """
        k = np.asarray(k_obs, float).ravel()
        k = k[np.isfinite(k) & (k > 0.0)]

        if k.size == 0:
            raise ValueError(
                "Cannot construct SANOS anchors without positive observed "
                "forward-normalized strikes."
            )

        decimals = int(round_decimals)
        if decimals < 0:
            raise ValueError("anchor_round_decimals must be >= 0.")

        anchors = np.unique(np.round(k, decimals=decimals))
        anchors = np.unique(
            np.concatenate([anchors, np.array([1.0], dtype=float)])
        )
        anchors = anchors[np.isfinite(anchors) & (anchors > 0.0)]
        anchors.sort()

        if anchors.size == 0:
            raise ValueError("SANOS anchor construction produced an empty grid.")

        return anchors

    @staticmethod
    def _alpha(u: float, mode: str) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        if mode == "linear":
            return u
        if mode == "smoothstep":
            return u * u * (3.0 - 2.0 * u)
        raise ValueError("time_interpolation must be 'linear' or 'smoothstep'.")

    def fit(
        self,
        K_obs,
        T_obs,
        C_obs,
        x0: Optional[Dict[str, float]] = None,
        bounds=None,
        max_nfev: int = 200,
        **kwargs,
    ) -> FitResult:
        del bounds, max_nfev, kwargs
        cfg = dict(x0 or {})

        smoothness = float(cfg.get("smoothness", 0.25))
        if not (0.0 <= smoothness < 1.0):
            raise ValueError("smoothness must satisfy 0 <= smoothness < 1.")

        anchor_round_decimals = int(
            round(float(cfg.get("anchor_round_decimals", 12)))
        )
        variance_floor = float(cfg.get("variance_floor", 1e-8))
        variance_step_floor = float(cfg.get("variance_step_floor", 1e-8))
        quote_weight_power = float(cfg.get("quote_weight_power", 0.0))
        time_code = int(round(float(cfg.get("time_code", 1.0))))
        time_interpolation = "linear" if time_code == 0 else "smoothstep"

        K = np.asarray(K_obs, float).ravel()
        T = np.asarray(T_obs, float).ravel()
        C = np.asarray(C_obs, float).ravel()
        good = (
            np.isfinite(K) & np.isfinite(T) & np.isfinite(C)
            & (K > 0.0) & (T > 0.0) & (C >= 0.0)
        )
        K, T, C = K[good], T[good], C[good]

        if K.size == 0:
            return FitResult(
                params={
                    "smoothness": smoothness,
                    "n_anchors": 0.0,
                    "n_quotes": 0.0,
                    "anchor_source_code": 1.0,
                },
                success=False,
                info={"message": "SANOS received no valid call quotes."},
            )

        maturities = np.unique(T)
        maturities.sort()
        if maturities.size < 1:
            raise ValueError("No positive maturities available.")

        k_obs, c_obs = self._to_pure(K, T, C)
        anchors = self._build_anchors_from_observed_strikes(
            k_obs,
            round_decimals=anchor_round_decimals,
        )
        M, N, Q = maturities.size, anchors.size, K.size

        variances = self._estimate_variance_backbone(
            K, T, C, maturities,
            variance_floor=variance_floor,
            variance_step_floor=variance_step_floor,
        )

        # Variables: q[M*N], e_plus[Q], e_minus[Q].
        nq = M * N
        nvar = nq + 2 * Q
        objective = np.zeros(nvar, dtype=float)
        quote_weights = np.power(np.maximum(c_obs, 1e-6), -quote_weight_power)
        quote_weights /= max(float(np.mean(quote_weights)), 1e-12)
        objective[nq:nq + Q] = quote_weights
        objective[nq + Q:] = quote_weights

        # Tiny deterministic regularizer discourages unnecessary far-tail mass.
        tail_cost = np.abs(np.log(anchors))
        for j in range(M):
            objective[j * N:(j + 1) * N] = 1e-10 * tail_cost

        # Equalities: fit equations + probability + unit mean per maturity.
        A_eq = np.zeros((Q + 2 * M, nvar), dtype=float)
        b_eq = np.zeros(Q + 2 * M, dtype=float)

        maturity_index = {float(t): j for j, t in enumerate(maturities)}
        for h in range(Q):
            j = maturity_index[float(T[h])]
            basis = _pure_bs_call(anchors, float(k_obs[h]), smoothness * variances[j])
            A_eq[h, j * N:(j + 1) * N] = basis
            A_eq[h, nq + h] = -1.0
            A_eq[h, nq + Q + h] = 1.0
            b_eq[h] = c_obs[h]

        row = Q
        for j in range(M):
            sl = slice(j * N, (j + 1) * N)
            A_eq[row, sl] = 1.0
            b_eq[row] = 1.0
            row += 1
            A_eq[row, sl] = anchors
            b_eq[row] = 1.0
            row += 1

        # Convex-order inequalities on the anchor grid:
        # E[(X_j-k)+] <= E[(X_{j+1}-k)+].
        if M > 1:
            A_ub = np.zeros(((M - 1) * N, nvar), dtype=float)
            b_ub = np.zeros((M - 1) * N, dtype=float)
            rr = 0
            payoff = np.maximum(anchors[:, None] - anchors[None, :], 0.0).T
            # payoff[h, i] = (anchor_i - test_strike_h)+
            for j in range(M - 1):
                for h in range(N):
                    A_ub[rr, j * N:(j + 1) * N] = payoff[h]
                    A_ub[rr, (j + 1) * N:(j + 2) * N] = -payoff[h]
                    rr += 1
        else:
            A_ub = None
            b_ub = None

        lp_bounds = [(0.0, None)] * nvar
        res = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=lp_bounds,
            method="highs",
        )

        if not bool(res.success):
            self._state = None
            return FitResult(
                params={
                    "smoothness": smoothness,
                    "n_anchors": float(N),
                    "n_quotes": float(Q),
                    "anchor_source_code": 1.0,
                    "anchor_round_decimals": float(anchor_round_decimals),
                    "n_maturities": float(M),
                    "lp_status": float(res.status),
                },
                success=False,
                info={"message": str(res.message)},
            )

        weights = np.asarray(res.x[:nq], float).reshape(M, N)
        weights = np.maximum(weights, 0.0)
        weights /= np.sum(weights, axis=1, keepdims=True)

        # Diagnostics at observed quotes.
        fitted = np.empty(Q, dtype=float)
        for h in range(Q):
            j = maturity_index[float(T[h])]
            fitted[h] = float(np.dot(
                weights[j],
                _pure_bs_call(anchors, float(k_obs[h]), smoothness * variances[j]),
            ))
        err = fitted - c_obs
        fit_mae = float(np.mean(np.abs(err)))
        fit_rmse = float(np.sqrt(np.mean(err * err)))

        if M >= 2:
            slope = (variances[-1] - variances[-2]) / (maturities[-1] - maturities[-2])
        else:
            slope = variances[-1] / max(maturities[-1], 1e-12)
        slope = max(float(slope), 0.0)

        self._state = _SANOSState(
            maturities=maturities.copy(),
            anchors=anchors.copy(),
            weights=weights.copy(),
            variances=variances.copy(),
            smoothness=smoothness,
            time_interpolation=time_interpolation,
            long_variance_slope=slope,
            fit_mae_pure=fit_mae,
            fit_rmse_pure=fit_rmse,
            lp_status=int(res.status),
            lp_message=str(res.message),
        )

        # Flat scalar params are intentional: the engine stores the actual fitted
        # arrays on this model instance and preserves its existing FitState design.
        params = {
            "smoothness": smoothness,
            "n_anchors": float(N),
            "n_quotes": float(Q),
            "anchor_source_code": 1.0,
            "anchor_round_decimals": float(anchor_round_decimals),
            "n_maturities": float(M),
            "anchor_min": float(anchors[0]),
            "anchor_max": float(anchors[-1]),
            "variance_min": float(variances[0]),
            "variance_max": float(variances[-1]),
            "long_variance_slope": float(slope),
            "fit_mae_pure": fit_mae,
            "fit_rmse_pure": fit_rmse,
            "lp_status": float(res.status),
            "time_code": float(time_code),
        }
        return FitResult(params=params, success=True, info={"message": str(res.message)})

    # ----------------------------- evaluation -----------------------------
    def _require_state(self) -> _SANOSState:
        if self._state is None:
            raise RuntimeError("SANOS model has not been fitted successfully.")
        return self._state

    def _pure_curve_at(self, k: np.ndarray, T: float) -> np.ndarray:
        st = self._require_state()
        k = np.asarray(k, float).ravel()
        T = float(T)

        if T <= 0.0:
            return np.maximum(1.0 - k, 0.0)

        times = st.maturities
        if T <= times[0]:
            u = T / times[0]
            a = self._alpha(u, st.time_interpolation)
            V = a * st.variances[0]
            c0 = _pure_bs_call(1.0, k, V)
            c1 = np.zeros_like(k)
            for i, anchor in enumerate(st.anchors):
                c1 += st.weights[0, i] * _pure_bs_call(anchor, k, V)
            return (1.0 - a) * c0 + a * c1

        if T > times[-1]:
            V = st.variances[-1] + st.long_variance_slope * (T - times[-1])
            out = np.zeros_like(k)
            for i, anchor in enumerate(st.anchors):
                out += st.weights[-1, i] * _pure_bs_call(anchor, k, st.smoothness * V)
            return out

        # Exact or interior maturity.
        j = int(np.searchsorted(times, T, side="left"))
        if np.isclose(T, times[j], rtol=0.0, atol=1e-14):
            out = np.zeros_like(k)
            for i, anchor in enumerate(st.anchors):
                out += st.weights[j, i] * _pure_bs_call(
                    anchor, k, st.smoothness * st.variances[j]
                )
            return out

        j0, j1 = j - 1, j
        u = (T - times[j0]) / (times[j1] - times[j0])
        a = self._alpha(u, st.time_interpolation)
        V = (1.0 - a) * st.variances[j0] + a * st.variances[j1]

        lo = np.zeros_like(k)
        hi = np.zeros_like(k)
        for i, anchor in enumerate(st.anchors):
            lo += st.weights[j0, i] * _pure_bs_call(anchor, k, st.smoothness * V)
            hi += st.weights[j1, i] * _pure_bs_call(anchor, k, st.smoothness * V)
        return (1.0 - a) * lo + a * hi

    def call_prices(self, K: np.ndarray, T: float, params=None, **kwargs) -> np.ndarray:
        del params, kwargs
        K = np.asarray(K, float).ravel()
        T = float(T)
        if T <= 0.0:
            return np.maximum(self.S0 - K, 0.0)
        F = float(self._forward(T))
        scale = float(self._scale(T))
        return scale * self._pure_curve_at(K / F, T)

    def price_surface(self, K_grid: np.ndarray, T_grid: np.ndarray, params=None, **kwargs) -> np.ndarray:
        del params, kwargs
        K_grid = np.asarray(K_grid, float).ravel()
        T_grid = np.asarray(T_grid, float).ravel()
        out = np.empty((T_grid.size, K_grid.size), dtype=float)
        for j, T in enumerate(T_grid):
            out[j] = self.call_prices(K_grid, float(T))
        return out

    # Optional inspection helper; not used by GlobalSurfacePricer.
    def fitted_surface_state(self) -> Dict[str, np.ndarray | float | str]:
        st = self._require_state()
        return {
            "maturities": st.maturities.copy(),
            "anchors": st.anchors.copy(),
            "weights": st.weights.copy(),
            "variances": st.variances.copy(),
            "smoothness": float(st.smoothness),
            "time_interpolation": st.time_interpolation,
            "long_variance_slope": float(st.long_variance_slope),
            "fit_mae_pure": float(st.fit_mae_pure),
            "fit_rmse_pure": float(st.fit_rmse_pure),
            "lp_status": int(st.lp_status),
            "lp_message": st.lp_message,
        }
