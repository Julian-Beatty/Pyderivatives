from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import eig
from scipy.optimize import lsq_linear

from ..base import MeasureTransform
from ..config import BootstrapSpec, CacheSpec, KeySpec
from ..moments import physical_moments_table
from ..output import (
    build_axis_grids,
    build_transform_output,
    lr_surface_to_k_surface,
    lr_surface_to_r_surface,
)
from ..registry import register_transform
from ..utils import _as_1d, _cdf_from_density, _find_spot, _trapz_normalize_density


@dataclass
class RossRecoveryOutput:
    state_price_matrix: np.ndarray
    state_price_transition: np.ndarray
    physical_transition: np.ndarray
    eigenvalue: float
    eigenfunction: np.ndarray
    ridge_zeta: float
    kl_loss: float
    current_state_index: int
    dt: float
    recovery_grid: np.ndarray
    recovery_grid_size: int
    original_grid_size: int
    recovery_grid_bounds: Tuple[float, float]


@register_transform("ross_recovery")
@register_transform("ross")
class RossRecoveryKernel(MeasureTransform):
    """
    Audrino-style Ross recovery from a single date's state-price surface.

    The transition matrices are estimated on an internal, possibly coarsened,
    log-return grid. The recovered physical densities are then interpolated
    back to the original RND grid so the public output remains compatible with
    the rest of PyDerivatives.

    Parameters
    ----------
    recovery_grid_size:
        Number of internal Ross states. Defaults to 50. Set to ``None`` to use
        the original RND grid without coarsening.

    recovery_grid_bounds:
        Optional ``(lower, upper)`` bounds in log-return units. By default the
        complete original RND grid is used. Bounds must lie inside the original
        grid and contain ``current_state``.

    zeta_grid:
        Candidate ridge penalties. The value minimizing generalized KL loss is
        selected.

    nnls_max_iter:
        Maximum iterations for each nonnegative least-squares problem.
    """

    def __init__(
        self,
        *,
        recovery_grid_size: Optional[int] = 50,
        recovery_grid_bounds: Optional[Tuple[float, float]] = None,
        zeta_grid: Optional[np.ndarray] = None,
        state_price_floor: float = 1e-14,
        transition_floor: float = 1e-14,
        current_state: float = 0.0,
        nnls_max_iter: int = 500,
        key_spec: KeySpec = KeySpec(),
        eps: float = 1e-10,
        verbose: bool = True,
        penalty_value: float = 1e100,
        cache_spec: CacheSpec = CacheSpec(),
        behavioral: bool = False,
        stock_df=None,
        stock_date_col: str = "date",
        volume_col: str = "volume",
        k1: float = 1.0,
        k2: float = 1.2,
        k3: float = 1.0,
        sentiment_alpha: float = 0.05,
        iv_sentiment_alpha: float = 0.05,
        volume_sentiment_alpha: float = 0.05,
        tail_sentiment_alpha: float = 0.05,
        skew_threshold: float = 1.5,
    ):
        super().__init__(
            key_spec=key_spec,
            min_obs=1,
            eps=eps,
            verbose=verbose,
            penalty_value=penalty_value,
            cache_spec=cache_spec,
            behavioral=behavioral,
            stock_df=stock_df,
            stock_date_col=stock_date_col,
            volume_col=volume_col,
            k1=k1,
            k2=k2,
            k3=k3,
            sentiment_alpha=sentiment_alpha,
        )

        if recovery_grid_size is None:
            self.recovery_grid_size = None
        else:
            recovery_grid_size = int(recovery_grid_size)
            if recovery_grid_size < 10:
                raise ValueError("recovery_grid_size must be at least 10 or None.")
            self.recovery_grid_size = recovery_grid_size

        if recovery_grid_bounds is None:
            self.recovery_grid_bounds = None
        else:
            if len(recovery_grid_bounds) != 2:
                raise ValueError("recovery_grid_bounds must contain (lower, upper).")
            lower, upper = map(float, recovery_grid_bounds)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(
                    "recovery_grid_bounds must be finite and strictly increasing."
                )
            self.recovery_grid_bounds = (lower, upper)

        self.zeta_grid = (
            np.asarray(zeta_grid, dtype=float).ravel()
            if zeta_grid is not None
            else np.r_[0.0, np.logspace(-12, 2, 40)]
        )
        if self.zeta_grid.size == 0:
            raise ValueError("zeta_grid cannot be empty.")
        if np.any(~np.isfinite(self.zeta_grid)) or np.any(self.zeta_grid < 0):
            raise ValueError("zeta_grid must contain finite nonnegative values.")

        self.state_price_floor = float(state_price_floor)
        self.transition_floor = float(transition_floor)
        self.current_state = float(current_state)
        self.nnls_max_iter = int(nnls_max_iter)

        if self.state_price_floor <= 0:
            raise ValueError("state_price_floor must be positive.")
        if self.transition_floor <= 0:
            raise ValueError("transition_floor must be positive.")
        if self.nnls_max_iter < 1:
            raise ValueError("nnls_max_iter must be at least 1.")

        # Ross recovery is cross-sectional and does not require historical fit.
        self.is_fitted_ = True

    def fit(self, *args, **kwargs):
        self.is_fitted_ = True
        return self

    def _cache_params(self) -> dict:
        return {
            "recovery_grid_size": self.recovery_grid_size,
            "recovery_grid_bounds": self.recovery_grid_bounds,
            "zeta_grid": self.zeta_grid,
            "state_price_floor": self.state_price_floor,
            "transition_floor": self.transition_floor,
            "current_state": self.current_state,
            "nnls_max_iter": self.nnls_max_iter,
        }

    def transform_info(
        self,
        info: dict,
        *,
        bootstrap: bool = False,
        bootstrap_spec: BootstrapSpec = BootstrapSpec(),
    ) -> dict:
        return self.transform_rnd(
            info,
            bootstrap=bootstrap,
            bootstrap_spec=bootstrap_spec,
        )

    def transform_rnd(
        self,
        info: dict,
        *,
        bootstrap: bool = False,
        bootstrap_spec: BootstrapSpec = BootstrapSpec(),
    ) -> dict:
        original_x, original_rnd, original_cdf, T_grid = self._extract_surfaces(info)
        original_x = _as_1d(original_x)
        T_grid = _as_1d(T_grid)

        self._validate_surface(
            x_grid=original_x,
            surface=original_rnd,
            T_grid=T_grid,
        )

        S0 = _find_spot(info, self.key_spec.spot_keys)

        # 1. Interpolate the RND surface onto the internal recovery grid.
        recovery_x = self._build_recovery_grid(original_x)
        recovery_rnd = self._interpolate_surface(
            source_x=original_x,
            target_x=recovery_x,
            surface=original_rnd,
        )

        # 2. Estimate G and recover P on the smaller grid.
        recovery = self._estimate_recovery_from_surface(
            x_grid=recovery_x,
            rnd_lr_surface=recovery_rnd,
            T_grid=T_grid,
            info=info,
            original_grid_size=original_x.size,
        )

        physical_recovery_surface = self._physical_surface_from_recovery(
            x_grid=recovery_x,
            T_grid=T_grid,
            recovery=recovery,
        )

        # 3. Return the recovered densities to the original public grid.
        physical_lr_surface = self._interpolate_surface(
            source_x=recovery_x,
            target_x=original_x,
            surface=physical_recovery_surface,
        )

        physical_cdf_lr_surface = np.vstack([
            _cdf_from_density(original_x, physical_lr_surface[j], eps=self.eps)
            for j in range(T_grid.size)
        ])

        rnd_lr_surface = np.vstack([
            _trapz_normalize_density(original_x, original_rnd[j], eps=self.eps)
            for j in range(T_grid.size)
        ])

        cdf_lr_surface = self._clean_or_rebuild_cdf_surface(
            x_grid=original_x,
            rnd_lr_surface=rnd_lr_surface,
            supplied_cdf=original_cdf,
        )

        pricing_kernel_surface = (
            rnd_lr_surface / np.maximum(physical_lr_surface, self.eps)
        )
        measure_weight_surface = (
            physical_lr_surface / np.maximum(rnd_lr_surface, self.eps)
        )

        for j in range(T_grid.size):
            expected_kernel = float(
                np.trapezoid(
                    pricing_kernel_surface[j] * physical_lr_surface[j],
                    original_x,
                )
            )
            if np.isfinite(expected_kernel) and expected_kernel > self.eps:
                pricing_kernel_surface[j] /= expected_kernel

        rra_surface = np.vstack([
            self.compute_relative_risk_aversion(
                original_x,
                pricing_kernel_surface[j],
            )
            for j in range(T_grid.size)
        ])

        grid_lr, grid_r, grid_k = build_axis_grids(original_x, S0)

        rnd_r_surface = lr_surface_to_r_surface(rnd_lr_surface, grid_r)
        physical_r_surface = lr_surface_to_r_surface(physical_lr_surface, grid_r)

        rnd_k_surface = lr_surface_to_k_surface(rnd_lr_surface, grid_k)
        physical_k_surface = lr_surface_to_k_surface(physical_lr_surface, grid_k)

        physical_moments = physical_moments_table(
            T_grid=T_grid,
            r_grid=grid_lr,
            physical_lr_surface=physical_lr_surface,
        )
        risk_neutral_moments = physical_moments_table(
            T_grid=T_grid,
            r_grid=grid_lr,
            physical_lr_surface=rnd_lr_surface,
        )

        out = build_transform_output(
            method_name=self.method_name,
            info=info,
            fit_trim_alpha=None,
            T_grid=T_grid,
            matched_T_grid=T_grid,
            status_by_T=[
                {
                    "T": float(T),
                    "matched_T": float(T),
                    "status": "success",
                    "ross_recovery_grid_size": int(recovery_x.size),
                }
                for T in T_grid
            ],
            grid_lr=grid_lr,
            grid_r=grid_r,
            grid_k=grid_k,
            rnd_lr_surface=rnd_lr_surface,
            rnd_r_surface=rnd_r_surface,
            rnd_k_surface=rnd_k_surface,
            cdf_lr_surface=cdf_lr_surface,
            cdf_r_surface=cdf_lr_surface,
            cdf_k_surface=cdf_lr_surface,
            base_physical_lr_surface=physical_lr_surface,
            base_physical_r_surface=physical_r_surface,
            base_physical_k_surface=physical_k_surface,
            base_physical_cdf_lr_surface=physical_cdf_lr_surface,
            base_physical_cdf_r_surface=physical_cdf_lr_surface,
            base_physical_cdf_k_surface=physical_cdf_lr_surface,
            physical_lr_surface=physical_lr_surface,
            physical_r_surface=physical_r_surface,
            physical_k_surface=physical_k_surface,
            physical_cdf_lr_surface=physical_cdf_lr_surface,
            physical_cdf_r_surface=physical_cdf_lr_surface,
            physical_cdf_k_surface=physical_cdf_lr_surface,
            pricing_kernel_surface=pricing_kernel_surface,
            rra_surface=rra_surface,
            measure_weight_surface=measure_weight_surface,
            base_pricing_kernel_surface=pricing_kernel_surface,
            base_measure_weight_surface=measure_weight_surface,
            fit_diagnostics={},
            S0=S0,
            physical_moments=physical_moments,
            risk_neutral_moments=risk_neutral_moments,
            base_physical_moments=physical_moments,
        )

        out["ross_recovery"] = recovery.__dict__
        out["ross_recovery_settings"] = {
            "original_grid_size": int(original_x.size),
            "recovery_grid_size": int(recovery_x.size),
            "requested_recovery_grid_size": self.recovery_grid_size,
            "recovery_grid_bounds": (float(recovery_x[0]), float(recovery_x[-1])),
            "zeta_grid_size": int(self.zeta_grid.size),
            "nnls_max_iter": int(self.nnls_max_iter),
            "interpolated_back_to_original_grid": True,
        }
        out["bootstrap"] = {
            "enabled": False,
            "requested": bool(bootstrap),
            "message": (
                "Ross recovery bootstrap is not implemented."
                if bootstrap
                else None
            ),
        }
        return out

    def _fit_one_maturity(self, *args, **kwargs):
        raise NotImplementedError(
            "RossRecoveryKernel does not use _fit_one_maturity(). "
            "Use transform_rnd(info) instead."
        )

    def _transform_surface_with_model(self, *args, **kwargs):
        raise NotImplementedError(
            "RossRecoveryKernel does not use _transform_surface_with_model(). "
            "Use transform_rnd(info) instead."
        )

    def _validate_surface(self, *, x_grid, surface, T_grid) -> None:
        x = _as_1d(x_grid)
        T = _as_1d(T_grid)
        Z = np.asarray(surface, dtype=float)

        if x.size < 2 or np.any(~np.isfinite(x)):
            raise ValueError("x_grid must contain at least two finite values.")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x_grid must be strictly increasing.")
        if T.size < 3:
            raise ValueError("Ross recovery needs at least 3 maturities.")
        if np.any(~np.isfinite(T)) or np.any(np.diff(T) <= 0):
            raise ValueError("T_grid must be finite and strictly increasing.")
        if Z.shape != (T.size, x.size):
            raise ValueError(
                f"rnd_lr_surface must have shape {(T.size, x.size)}, got {Z.shape}."
            )

    def _build_recovery_grid(self, original_x: np.ndarray) -> np.ndarray:
        x = _as_1d(original_x)

        if self.recovery_grid_bounds is None:
            lower, upper = float(x[0]), float(x[-1])
        else:
            lower, upper = self.recovery_grid_bounds
            if lower < float(x[0]) or upper > float(x[-1]):
                raise ValueError(
                    "recovery_grid_bounds must lie inside the original x grid."
                )

        if not (lower <= self.current_state <= upper):
            raise ValueError("current_state must lie inside recovery_grid_bounds.")

        if self.recovery_grid_size is None:
            recovery_x = x[(x >= lower) & (x <= upper)]
            if recovery_x.size < 10:
                raise ValueError("The selected bounds contain fewer than 10 states.")
            return recovery_x.copy()

        n_states = min(int(self.recovery_grid_size), int(x.size))
        if n_states < 10:
            raise ValueError("The effective recovery grid must have at least 10 states.")

        return np.linspace(lower, upper, n_states, dtype=float)

    def _interpolate_surface(self, *, source_x, target_x, surface) -> np.ndarray:
        source_x = _as_1d(source_x)
        target_x = _as_1d(target_x)
        surface = np.asarray(surface, dtype=float)

        if surface.ndim != 2 or surface.shape[1] != source_x.size:
            raise ValueError(
                "surface must be 2D with one column for each source_x state."
            )

        rows = []
        for row in surface:
            row = np.where(np.isfinite(row) & (row >= 0), row, 0.0)
            interpolated = np.interp(
                target_x,
                source_x,
                row,
                left=0.0,
                right=0.0,
            )
            interpolated = _trapz_normalize_density(
                target_x,
                interpolated,
                eps=self.eps,
            )
            if np.any(~np.isfinite(interpolated)):
                raise ValueError(
                    "A density row could not be normalized after interpolation."
                )
            rows.append(interpolated)

        return np.vstack(rows)

    def _clean_or_rebuild_cdf_surface(
        self,
        *,
        x_grid,
        rnd_lr_surface,
        supplied_cdf,
    ) -> np.ndarray:
        x = _as_1d(x_grid)
        supplied = np.asarray(supplied_cdf, dtype=float)

        if supplied.shape == rnd_lr_surface.shape and np.all(np.isfinite(supplied)):
            cleaned = np.maximum.accumulate(np.clip(supplied, 0.0, 1.0), axis=1)
            spans = cleaned[:, -1] - cleaned[:, 0]
            if np.all(np.isfinite(spans)) and np.all(spans > self.eps):
                cleaned = (cleaned - cleaned[:, [0]]) / spans[:, None]
                cleaned[:, 0] = 0.0
                cleaned[:, -1] = 1.0
                return cleaned

        return np.vstack([
            _cdf_from_density(x, row, eps=self.eps)
            for row in rnd_lr_surface
        ])

    def _state_price_matrix(self, x_grid, rnd_lr_surface, T_grid, info):
        r_rate = float(info.get("r", 0.0))
        if not np.isfinite(r_rate):
            r_rate = 0.0

        x = _as_1d(x_grid)
        T_grid = _as_1d(T_grid)
        dx = np.gradient(x)
        columns = []

        for j, T in enumerate(T_grid):
            f_q = _trapz_normalize_density(x, rnd_lr_surface[j], eps=self.eps)
            if np.any(~np.isfinite(f_q)):
                raise ValueError(f"Invalid RND at maturity index {j}.")

            discount = np.exp(-r_rate * float(T))
            state_prices = discount * f_q * dx
            columns.append(np.maximum(state_prices, self.state_price_floor))

        return np.column_stack(columns)

    def _estimate_recovery_from_surface(
        self,
        *,
        x_grid,
        rnd_lr_surface,
        T_grid,
        info,
        original_grid_size: int,
    ):
        x = _as_1d(x_grid)
        T_grid = _as_1d(T_grid)

        S = self._state_price_matrix(x, rnd_lr_surface, T_grid, info)
        A = S[:, :-1].T
        B = S[:, 1:].T
        i0 = int(np.argmin(np.abs(x - self.current_state)))

        dt = float(np.median(np.diff(T_grid)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Could not determine a positive maturity step.")

        best = None
        for zeta in self.zeta_grid:
            G = self._solve_transition_nnls(A, B, float(zeta))
            U = self._markov_state_prices(G, i0=i0, m=S.shape[1])
            kl = self._generalized_kl(S, U)

            if best is None or kl < best["kl"]:
                best = {
                    "zeta": float(zeta),
                    "G": G,
                    "U": U,
                    "kl": float(kl),
                }

        if best is None:
            raise RuntimeError("Ross recovery failed for every ridge penalty.")

        P, lam, phi = self._ross_recover(best["G"])

        if self.verbose:
            print(
                "[ross] "
                f"original_states={int(original_grid_size)} | "
                f"recovery_states={x.size} | "
                f"maturities={T_grid.size} | "
                f"transitions={T_grid.size - 1} | "
                f"zeta={best['zeta']:.6g} | KL={best['kl']:.6g}"
            )

        return RossRecoveryOutput(
            state_price_matrix=S,
            state_price_transition=best["G"],
            physical_transition=P,
            eigenvalue=float(lam),
            eigenfunction=phi,
            ridge_zeta=float(best["zeta"]),
            kl_loss=float(best["kl"]),
            current_state_index=int(i0),
            dt=dt,
            recovery_grid=x.copy(),
            recovery_grid_size=int(x.size),
            original_grid_size=int(original_grid_size),
            recovery_grid_bounds=(float(x[0]), float(x[-1])),
        )

    def _solve_transition_nnls(self, A, B, zeta):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        n_states = A.shape[1]
        G = np.zeros((n_states, n_states), dtype=float)

        if zeta > 0:
            A_aug = np.vstack([A, np.sqrt(zeta) * np.eye(n_states)])
        else:
            A_aug = A

        for j in range(n_states):
            b = B[:, j]
            b_aug = np.r_[b, np.zeros(n_states)] if zeta > 0 else b

            result = lsq_linear(
                A_aug,
                b_aug,
                bounds=(0.0, np.inf),
                method="trf",
                lsmr_tol="auto",
                max_iter=self.nnls_max_iter,
            )

            if not result.success and self.verbose:
                print(
                    f"[ross warning] NNLS column {j} did not converge: "
                    f"{result.message}"
                )

            G[:, j] = np.maximum(result.x, 0.0)

        return np.maximum(G, self.transition_floor)

    def _markov_state_prices(self, G, *, i0, m):
        n_states = G.shape[0]
        U = np.zeros((n_states, m), dtype=float)
        row = np.zeros(n_states, dtype=float)
        row[i0] = 1.0

        for t in range(1, m + 1):
            row = row @ G
            U[:, t - 1] = row

        return np.maximum(U, self.state_price_floor)

    def _generalized_kl(self, S, U):
        S = np.maximum(np.asarray(S, dtype=float), self.state_price_floor)
        U = np.maximum(np.asarray(U, dtype=float), self.state_price_floor)
        value = np.sum(S * np.log(S / U) - (S - U))
        return float(value) if np.isfinite(value) else np.inf

    def _ross_recover(self, G):
        values, vectors = eig(G)
        finite = np.isfinite(np.real(values)) & np.isfinite(np.imag(values))
        if not np.any(finite):
            raise ValueError("G has no finite eigenvalues.")

        candidates = np.where(finite)[0]
        idx = int(candidates[np.argmax(np.real(values[candidates]))])
        lam = float(np.real(values[idx]))
        if not np.isfinite(lam) or lam <= self.eps:
            raise ValueError("Ross recovery requires a positive dominant eigenvalue.")

        phi = np.real(vectors[:, idx])
        if np.nanmean(phi) < 0:
            phi = -phi
        phi = np.maximum(phi, self.eps)
        phi /= np.mean(phi)

        P = G * (phi[None, :] / np.maximum(lam * phi[:, None], self.eps))
        P = np.where(np.isfinite(P) & (P >= 0), P, 0.0)

        row_sums = P.sum(axis=1, keepdims=True)
        if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= self.eps):
            raise ValueError("Recovered P has invalid row sums.")
        P /= row_sums

        return P, lam, phi

    def _physical_surface_from_recovery(self, *, x_grid, T_grid, recovery):
        x = _as_1d(x_grid)
        T_grid = _as_1d(T_grid)
        dx = np.gradient(x)

        P = recovery.physical_transition
        i0 = recovery.current_state_index
        dt = recovery.dt
        rows = []

        for T in T_grid:
            steps = max(1, int(round(float(T) / dt)))
            probabilities = np.maximum(
                np.linalg.matrix_power(P, steps)[i0, :],
                0.0,
            )
            density = probabilities / np.maximum(dx, self.eps)
            density = _trapz_normalize_density(x, density, eps=self.eps)

            if np.any(~np.isfinite(density)):
                raise ValueError(
                    f"Recovered physical density is invalid at T={float(T):.8g}."
                )
            rows.append(density)

        return np.vstack(rows)
