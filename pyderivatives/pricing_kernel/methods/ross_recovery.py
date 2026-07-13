from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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


@register_transform("ross_recovery")
@register_transform("ross")
class RossRecoveryKernel(MeasureTransform):
    """
    Audrino-style Ross recovery from a single state-price surface.

    Input:
        one RND/state-price surface across maturities

    Steps:
        1. Convert RND columns into state-price vectors S[:, t].
        2. Estimate state-price transition matrix G from

               S_{t+1}' = S_t' G

           using ridge-regularized nonnegative least squares.
        3. Select ridge zeta by generalized KL divergence between
           observed state prices and Markov-implied state prices.
        4. Apply Ross/Perron-Frobenius recovery:

               P_ij = G_ij * phi_j / (lambda * phi_i)

        5. Use the center/current-state row of P^t as the recovered
           physical distribution.
    """

    def __init__(
        self,
        *,
        zeta_grid: Optional[np.ndarray] = None,
        state_price_floor: float = 1e-14,
        transition_floor: float = 1e-14,
        current_state: float = 0.0,
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

        self.zeta_grid = (
            np.asarray(zeta_grid, dtype=float)
            if zeta_grid is not None
            else np.r_[0.0, np.logspace(-12, 2, 40)]
        )
        self.state_price_floor = float(state_price_floor)
        self.transition_floor = float(transition_floor)
        self.current_state = float(current_state)
        self.is_fitted_ = True

    def fit(self, *args, **kwargs):
        self.is_fitted_ = True
        return self

    def _cache_params(self) -> dict:
        return {
            "zeta_grid": self.zeta_grid,
            "state_price_floor": self.state_price_floor,
            "transition_floor": self.transition_floor,
            "current_state": self.current_state,
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
        x_grid, rnd_lr_surface, cdf_lr_surface, T_grid = self._extract_surfaces(info)
        S0 = _find_spot(info, self.key_spec.spot_keys)

        recovery = self._estimate_recovery_from_surface(
            x_grid=x_grid,
            rnd_lr_surface=rnd_lr_surface,
            T_grid=T_grid,
            info=info,
        )

        physical_lr_surface = self._physical_surface_from_recovery(
            x_grid=x_grid,
            T_grid=T_grid,
            recovery=recovery,
        )

        physical_cdf_lr_surface = np.vstack([
            _cdf_from_density(x_grid, physical_lr_surface[j], eps=self.eps)
            for j in range(len(T_grid))
        ])

        rnd_lr_surface = np.vstack([
            _trapz_normalize_density(x_grid, rnd_lr_surface[j], eps=self.eps)
            for j in range(len(T_grid))
        ])

        pricing_kernel_surface = rnd_lr_surface / np.maximum(physical_lr_surface, self.eps)
        measure_weight_surface = physical_lr_surface / np.maximum(rnd_lr_surface, self.eps)

        for j in range(len(T_grid)):
            Em = float(np.trapezoid(pricing_kernel_surface[j] * physical_lr_surface[j], x_grid))
            if np.isfinite(Em) and Em > self.eps:
                pricing_kernel_surface[j] /= Em

        rra_surface = np.vstack([
            self.compute_relative_risk_aversion(x_grid, pricing_kernel_surface[j])
            for j in range(len(T_grid))
        ])

        grid_lr, grid_r, grid_k = build_axis_grids(x_grid, S0)

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
                {"T": float(T), "matched_T": float(T), "status": "success"}
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
        out["bootstrap"] = {"enabled": False}
        return out


    def _fit_one_maturity(self, *args, **kwargs):
        """
        Ross recovery is not fitted maturity-by-maturity.

        This method only exists to satisfy the MeasureTransform abstract
        interface. Use transform_rnd(...) / transform_info(...) instead.
        """
        raise NotImplementedError(
            "RossRecoveryKernel does not use _fit_one_maturity(). "
            "Use transform_rnd(info) instead."
        )

    def _transform_surface_with_model(self, *args, **kwargs):
        """
        Ross recovery transforms the full maturity surface jointly.

        This method only exists to satisfy the MeasureTransform abstract
        interface. Use transform_rnd(...) / transform_info(...) instead.
        """
        raise NotImplementedError(
            "RossRecoveryKernel does not use _transform_surface_with_model(). "
            "Use transform_rnd(info) instead."
        )

    def _state_price_matrix(self, x_grid, rnd_lr_surface, T_grid, info):
        r_rate = float(info.get("r", 0.0))
        x = _as_1d(x_grid)
        dx = np.gradient(x)

        S_cols = []

        for j, T in enumerate(T_grid):
            f_q = _trapz_normalize_density(x, rnd_lr_surface[j], eps=self.eps)
            discount = np.exp(-r_rate * float(T))
            state_prices = discount * f_q * dx
            state_prices = np.maximum(state_prices, self.state_price_floor)
            S_cols.append(state_prices)

        return np.column_stack(S_cols)

    def _estimate_recovery_from_surface(self, *, x_grid, rnd_lr_surface, T_grid, info):
        x = _as_1d(x_grid)
        T_grid = _as_1d(T_grid)

        if len(T_grid) < 3:
            raise ValueError("Ross recovery needs at least 3 maturities.")

        S = self._state_price_matrix(x, rnd_lr_surface, T_grid, info)

        A = S[:, :-1].T
        B = S[:, 1:].T

        i0 = int(np.argmin(np.abs(x - self.current_state)))

        best = None

        dt = float(np.median(np.diff(T_grid)))

        for zeta in self.zeta_grid:
            G = self._solve_transition_nnls(A, B, float(zeta))
            U = self._markov_state_prices(G, i0=i0, m=S.shape[1])
            kl = self._generalized_kl(S, U)

            if best is None or kl < best["kl"]:
                best = {"zeta": float(zeta), "G": G, "U": U, "kl": float(kl)}

        G = best["G"]
        P, lam, phi = self._ross_recover(G)

        return RossRecoveryOutput(
            state_price_matrix=S,
            state_price_transition=G,
            physical_transition=P,
            eigenvalue=float(lam),
            eigenfunction=phi,
            ridge_zeta=float(best["zeta"]),
            kl_loss=float(best["kl"]),
            current_state_index=int(i0),
            dt=dt,
        )

    def _solve_transition_nnls(self, A, B, zeta):
        n = A.shape[1]
        G = np.zeros((n, n), dtype=float)

        if zeta > 0:
            A_aug = np.vstack([A, np.sqrt(zeta) * np.eye(n)])
        else:
            A_aug = A

        for j in range(n):
            b = B[:, j]

            if zeta > 0:
                b_aug = np.r_[b, np.zeros(n)]
            else:
                b_aug = b

            res = lsq_linear(
                A_aug,
                b_aug,
                bounds=(0.0, np.inf),
                method="trf",
                lsmr_tol="auto",
                max_iter=500,
            )

            G[:, j] = np.maximum(res.x, 0.0)

        G = np.maximum(G, self.transition_floor)
        return G

    def _markov_state_prices(self, G, *, i0, m):
        n = G.shape[0]
        U = np.zeros((n, m), dtype=float)

        row = np.zeros(n, dtype=float)
        row[i0] = 1.0

        for t in range(1, m + 1):
            row = row @ G
            U[:, t - 1] = row

        return np.maximum(U, self.state_price_floor)

    def _generalized_kl(self, S, U):
        S = np.maximum(S, self.state_price_floor)
        U = np.maximum(U, self.state_price_floor)

        val = np.sum(S * np.log(S / U) - (S - U))
        return float(val) if np.isfinite(val) else np.inf

    def _ross_recover(self, G):
        vals, vecs = eig(G)

        idx = int(np.argmax(np.real(vals)))
        lam = float(np.real(vals[idx]))
        phi = np.real(vecs[:, idx])

        if np.nanmean(phi) < 0:
            phi = -phi

        phi = np.maximum(phi, self.eps)
        phi = phi / np.mean(phi)

        P = G * (phi[None, :] / np.maximum(lam * phi[:, None], self.eps))
        P = np.where(np.isfinite(P) & (P >= 0), P, 0.0)

        row_sums = P.sum(axis=1, keepdims=True)
        P = P / np.maximum(row_sums, self.eps)

        return P, lam, phi

    def _physical_surface_from_recovery(self, *, x_grid, T_grid, recovery):
        x = _as_1d(x_grid)
        dx = np.gradient(x)

        P = recovery.physical_transition
        i0 = recovery.current_state_index
        dt = recovery.dt

        out = []

        for T in T_grid:
            steps = max(1, int(round(float(T) / dt)))
            P_T = np.linalg.matrix_power(P, steps)
            probs = P_T[i0, :]

            f = probs / np.maximum(dx, self.eps)
            f = _trapz_normalize_density(x, f, eps=self.eps)
            out.append(f)

        return np.vstack(out)