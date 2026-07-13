from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .behavioral import BehavioralOverlayMixin
from .bootstrap import BootstrapMixin
from .cache import _cache_key, _cache_load, _cache_save
from .config import BootstrapSpec, CacheSpec, FitDiagnostics, KeySpec, BehavioralConfig
from .history import HistoryMixin
from .moments import physical_moments_table
from .output import (
    build_axis_grids,
    build_transform_output,
    lr_surface_to_k_surface,
    lr_surface_to_r_surface,
)
from .utils import (
    _as_1d,
    _cdf_from_density,
    _find_spot,
    _trapz_normalize_density,
    _validate_fit_trim_alpha,
)


class MeasureTransform(HistoryMixin, BehavioralOverlayMixin, BootstrapMixin, ABC):
    """
    Base class for transforming risk-neutral density surfaces into
    physical densities, pricing kernels, and relative risk aversion surfaces.
    """

    method_name: str = "base"

    def __init__(
        self,
        *,
        key_spec: KeySpec = KeySpec(),
        fit_trim_alpha: Optional[Tuple[float, float]] = None,
        min_obs: int = 30,
        fit_maturities: Optional[List[float]] = None,
        maturity_match_tol: Optional[float] = None,
        eps: float = 1e-10,
        verbose: bool = True,
        penalty_value: float = 1e100,
        cache_spec: CacheSpec = CacheSpec(),
        behavioral: bool | BehavioralConfig = False,
        stock_df: Optional[pd.DataFrame] = None,
        stock_date_col: str = "date",
        volume_col: str = "volume",
        k1: float = 1.0,
        k2: float = 1.2,
        k3: float = 1.0,
        sentiment_alpha: float = 0.05,
        iv_alpha: Optional[float] = None,
        volume_alpha: Optional[float] = None,
        tail_alpha: Optional[float] = None,
        positive_skew_threshold: float = 1.5,
        negative_skew_threshold: float = -1.5,
        theta_min: float = 0.25,
        theta_max: float = 4.0,
    ):
        self.key_spec = key_spec
        self.fit_trim_alpha = _validate_fit_trim_alpha(fit_trim_alpha)
        self.min_obs = int(min_obs)
        self.fit_maturities = None if fit_maturities is None else [float(x) for x in fit_maturities]
        self.maturity_match_tol = None if maturity_match_tol is None else float(maturity_match_tol)
        self.eps = float(eps)
        self.verbose = bool(verbose)
        self.penalty_value = float(penalty_value)
        self.cache_spec = cache_spec

        # Behavioral overlay configuration.
        # Backward-compatible usage still works:
        #     behavioral=True, stock_df=..., k1=..., sentiment_alpha=...
        # New usage:
        #     behavioral=BehavioralConfig(...)
        if isinstance(behavioral, BehavioralConfig):
            behavior_cfg = behavioral
            if stock_df is not None:
                behavior_cfg.stock_df = stock_df
        else:
            a = float(sentiment_alpha)
            behavior_cfg = BehavioralConfig(
                enabled=bool(behavioral),
                stock_df=stock_df,
                stock_date_col=stock_date_col,
                volume_col=volume_col,
                k1=k1,
                k2=k2,
                k3=k3,
                sentiment_alpha=a,
                iv_alpha=a if iv_alpha is None else float(iv_alpha),
                volume_alpha=a if volume_alpha is None else float(volume_alpha),
                tail_alpha=a if tail_alpha is None else float(tail_alpha),
                positive_skew_threshold=positive_skew_threshold,
                negative_skew_threshold=negative_skew_threshold,
                theta_min=theta_min,
                theta_max=theta_max,
            )

        behavior_cfg.validate()
        self.behavior_config = behavior_cfg

        # Legacy attributes retained so existing mixins/methods keep working.
        self.behavioral = bool(behavior_cfg.enabled)
        self.stock_df = None if behavior_cfg.stock_df is None else behavior_cfg.stock_df.copy()
        self.stock_date_col = str(behavior_cfg.stock_date_col)
        self.volume_col = str(behavior_cfg.volume_col)
        self.k1 = float(behavior_cfg.k1)
        self.k2 = float(behavior_cfg.k2)
        self.k3 = float(behavior_cfg.k3)
        self.sentiment_alpha = float(behavior_cfg.sentiment_alpha)
        self.iv_sentiment_alpha = float(behavior_cfg.iv_alpha)
        self.volume_sentiment_alpha = float(behavior_cfg.volume_alpha)
        self.tail_sentiment_alpha = float(behavior_cfg.tail_alpha)
        self.positive_skew_threshold = float(behavior_cfg.positive_skew_threshold)
        self.negative_skew_threshold = float(behavior_cfg.negative_skew_threshold)
        self.skew_threshold = float(max(abs(self.negative_skew_threshold), abs(self.positive_skew_threshold)))
        self.theta_min = float(behavior_cfg.theta_min)
        self.theta_max = float(behavior_cfg.theta_max)
        self.sentiment_by_date_: Dict[pd.Timestamp, dict] = {}

        self.models_by_T_: Dict[float, Any] = {}
        self.history_by_T_: Dict[float, pd.DataFrame] = {}
        self.fit_history_by_T_: Dict[float, pd.DataFrame] = {}
        self.fit_diagnostics_: Dict[float, FitDiagnostics] = {}
        self.is_fitted_: bool = False

    def fit(
        self,
        rnd_history_dict: Dict[Any, dict],
        stock_df: Optional[pd.DataFrame] = None,
        *,
        price_col: Optional[str] = None,
        adjusted_price_col: Optional[str] = None,
        adjustment_factor_col: Optional[str] = None,
        return_col: Optional[str] = None,
    ):
        if stock_df is not None:
            self.stock_df = stock_df.copy()

        if self.stock_df is None:
            raise ValueError("fit() requires stock_df, or stock_df must be supplied in the constructor.")

        logreturns = self._return_series_from_stock_df(
            self.stock_df,
            price_col=price_col,
            adjusted_price_col=adjusted_price_col,
            adjustment_factor_col=adjustment_factor_col,
            return_col=return_col,
        )

        fit_cache_key = None

        if self.cache_spec.enabled and self.cache_spec.cache_fit:
            fit_cache_key = _cache_key({
                "kind": "fit",
                "method": self.method_name,
                "dataset_tag": self.cache_spec.dataset_tag,
                "fit_trim_alpha": self.fit_trim_alpha,
                "fit_maturities": self.fit_maturities,
                "maturity_match_tol": self.maturity_match_tol,
                "min_obs": self.min_obs,
                "key_spec": self.key_spec,
                "class": self.__class__.__name__,
                "params": self._cache_params(),
                "n_dates": len(rnd_history_dict),
                "stock_n": len(self.stock_df),
                "price_col": price_col,
                "adjusted_price_col": adjusted_price_col,
                "adjustment_factor_col": adjustment_factor_col,
                "return_col": return_col,
                "behavioral": self.behavioral,
                "k1": self.k1,
                "k2": self.k2,
                "k3": self.k3,
                "sentiment_alpha": self.sentiment_alpha,
                "iv_alpha": self.iv_sentiment_alpha,
                "volume_alpha": self.volume_sentiment_alpha,
                "tail_alpha": self.tail_sentiment_alpha,
                "positive_skew_threshold": self.positive_skew_threshold,
                "negative_skew_threshold": self.negative_skew_threshold,
                "theta_min": self.theta_min,
                "theta_max": self.theta_max,
            })

            cached = _cache_load(self.cache_spec.folder, fit_cache_key)
            if cached is not None:
                self.__dict__.update(cached.__dict__)
                if self.verbose:
                    print(
                        f"[cache hit] loaded fitted {self.method_name} model "
                        f"from {self.cache_spec.folder}/{fit_cache_key}.pkl"
                    )
                return self

        history_by_T = self.build_history_by_maturity(rnd_history_dict, logreturns)

        self.history_by_T_ = history_by_T
        self.sentiment_by_date_ = (
            self._build_behavioral_sentiment_by_date(rnd_history_dict)
            if self.behavioral
            else {}
        )

        self.fit_history_by_T_ = {}
        self.models_by_T_ = {}
        self.fit_diagnostics_ = {}

        for T in self._select_fit_maturities(history_by_T):
            hist_T = self._apply_fit_trim(history_by_T[T].copy())
            fit_hist_T = hist_T.loc[hist_T["used_in_fit"]].copy()

            n_total = int(len(hist_T))
            n_used = int(len(fit_hist_T))
            n_dropped = int(n_total - n_used)

            if n_used < self.min_obs:
                diag = FitDiagnostics(
                    maturity=float(T),
                    method=self.method_name,
                    n_total=n_total,
                    n_used=n_used,
                    n_dropped=n_dropped,
                    loss=np.nan,
                    loss_name="not_fit",
                    status="skipped",
                    message=f"Too few observations after trimming: {n_used} < min_obs={self.min_obs}.",
                )
                self.fit_diagnostics_[float(T)] = diag

                if self.verbose:
                    self._print_fit_progress(diag)

                continue

            fitted_model, diag_extra = self._fit_one_maturity(fit_hist_T, T=float(T))

            diag = FitDiagnostics(
                maturity=float(T),
                method=self.method_name,
                n_total=n_total,
                n_used=n_used,
                n_dropped=n_dropped,
                loss=float(diag_extra.get("loss", np.nan)),
                loss_name=str(diag_extra.get("loss_name", "objective")),
                status=str(diag_extra.get("status", "unknown")),
                message=str(diag_extra.get("message", "")),
                params=dict(diag_extra.get("params", {})),
            )

            self.models_by_T_[float(T)] = fitted_model
            self.fit_history_by_T_[float(T)] = fit_hist_T
            self.fit_diagnostics_[float(T)] = diag

            if self.verbose:
                self._print_fit_progress(diag)

        self.is_fitted_ = True

        if self.cache_spec.enabled and self.cache_spec.cache_fit and fit_cache_key is not None:
            path = _cache_save(self.cache_spec.folder, fit_cache_key, self)
            if self.verbose:
                print(f"[cache save] fitted {self.method_name} model -> {path}")

        return self

    def transform_rnd(
        self,
        info: dict,
        *,
        bootstrap: bool = False,
        bootstrap_spec: BootstrapSpec = BootstrapSpec(),
    ) -> dict:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling transform_rnd().")

        transform_cache_key = None

        if self.cache_spec.enabled and self.cache_spec.cache_transform and not bootstrap:
            transform_cache_key = _cache_key({
                "kind": "transform",
                "method": self.method_name,
                "dataset_tag": self.cache_spec.dataset_tag,
                "fit_trim_alpha": self.fit_trim_alpha,
                "fit_maturities": self.fit_maturities,
                "maturity_match_tol": self.maturity_match_tol,
                "key_spec": self.key_spec,
                "class": self.__class__.__name__,
                "params": self._cache_params(),
                "info_date": str(info.get("date", info.get("day", info.get("anchor_date", "unknown")))),
                "T_grid": info.get(self.key_spec.T_grid_key, None),
            })

            cached = _cache_load(self.cache_spec.folder, transform_cache_key)
            if cached is not None:
                if self.verbose:
                    print(
                        f"[cache hit] loaded transformed output from "
                        f"{self.cache_spec.folder}/{transform_cache_key}.pkl"
                    )
                return cached

        result = self._transform_info_no_bootstrap(info)

        if bootstrap:
            result["bootstrap"] = self._bootstrap_transform_info(info, bootstrap_spec)
        else:
            result["bootstrap"] = {"enabled": False}

        if self.cache_spec.enabled and self.cache_spec.cache_transform and not bootstrap and transform_cache_key is not None:
            path = _cache_save(self.cache_spec.folder, transform_cache_key, result)
            if self.verbose:
                print(f"[cache save] transformed output -> {path}")

        return result

    def transform_info(
        self,
        info: dict,
        *,
        bootstrap: bool = False,
        bootstrap_spec: BootstrapSpec = BootstrapSpec(),
    ) -> dict:
        """
        Backward-compatible alias for transform_rnd(...).

        Prefer transform_rnd(...) in new code.
        """
        return self.transform_rnd(
            info,
            bootstrap=bootstrap,
            bootstrap_spec=bootstrap_spec,
        )

    def transform_surface(
        self,
        x_grid: np.ndarray,
        f_q: np.ndarray,
        F_q: np.ndarray,
        *,
        T: float,
        info: Optional[dict] = None,
    ) -> dict:
        T_match = self._match_maturity(float(T))
        fitted = self.models_by_T_.get(T_match)

        if fitted is None:
            raise KeyError(f"No fitted model available for maturity T={T}. Closest matched T={T_match}.")

        return self._transform_surface_with_model(
            fitted,
            _as_1d(x_grid),
            _as_1d(f_q),
            _as_1d(F_q),
            T=T_match,
            info=info or {},
        )

    def _transform_info_no_bootstrap(self, info: dict) -> dict:
        x_grid, rnd_lr_surface, cdf_lr_surface, T_grid = self._extract_surfaces(info)
        S0 = _find_spot(info, self.key_spec.spot_keys)

        nT, nx = rnd_lr_surface.shape

        base_physical_lr_surface = np.full((nT, nx), np.nan)
        base_physical_cdf_lr_surface = np.full((nT, nx), np.nan)
        base_pricing_kernel_surface = np.full((nT, nx), np.nan)
        base_measure_weight_surface = np.full((nT, nx), np.nan)

        physical_lr_surface = np.full((nT, nx), np.nan)
        physical_cdf_lr_surface = np.full((nT, nx), np.nan)
        measure_weight_surface = np.full((nT, nx), np.nan)
        pricing_kernel_surface = np.full((nT, nx), np.nan)
        rra_surface = np.full((nT, nx), np.nan)
        matched_T_grid = np.full(nT, np.nan)

        status_by_T = []

        for j, T in enumerate(T_grid):
            T = float(T)

            try:
                T_fit = self._match_maturity(T)
                fitted = self.models_by_T_[T_fit]

                res = self._transform_surface_with_model(
                    fitted,
                    x_grid,
                    rnd_lr_surface[j, :],
                    cdf_lr_surface[j, :],
                    T=T_fit,
                    info=info,
                )

                f_base = _trapz_normalize_density(x_grid, res["f_p"], eps=self.eps)
                F_base = _cdf_from_density(x_grid, f_base, eps=self.eps)

                kernel_base = rnd_lr_surface[j, :] / np.maximum(f_base, self.eps)
                weight_base = f_base / np.maximum(rnd_lr_surface[j, :], self.eps)

                base_physical_lr_surface[j, :] = f_base
                base_physical_cdf_lr_surface[j, :] = F_base
                base_pricing_kernel_surface[j, :] = kernel_base
                base_measure_weight_surface[j, :] = weight_base

                f_final, F_final, kernel_final, weight_final, overlay_info = (
                    self._apply_behavioral_overlay_if_needed(
                        x_grid=x_grid,
                        f_q=rnd_lr_surface[j, :],
                        f_p=f_base,
                        info=info,
                        T=T,
                    )
                )

                physical_lr_surface[j, :] = f_final
                physical_cdf_lr_surface[j, :] = F_final
                measure_weight_surface[j, :] = weight_final
                pricing_kernel_surface[j, :] = kernel_final

                rra_surface[j, :] = self.compute_relative_risk_aversion(
                    x_grid,
                    kernel_final,
                )

                matched_T_grid[j] = T_fit

                status_by_T.append(
                    {
                        "T": T,
                        "matched_T": T_fit,
                        "status": "success",
                        "behavioral_overlay": overlay_info,
                    }
                )

            except Exception as exc:
                status_by_T.append(
                    {
                        "T": T,
                        "matched_T": np.nan,
                        "status": "failed",
                        "message": str(exc),
                    }
                )

        grid_lr, grid_r, grid_k = build_axis_grids(x_grid, S0)

        rnd_r_surface = lr_surface_to_r_surface(rnd_lr_surface, grid_r)
        base_physical_r_surface = lr_surface_to_r_surface(base_physical_lr_surface, grid_r)
        physical_r_surface = lr_surface_to_r_surface(physical_lr_surface, grid_r)

        rnd_k_surface = lr_surface_to_k_surface(rnd_lr_surface, grid_k)
        base_physical_k_surface = lr_surface_to_k_surface(base_physical_lr_surface, grid_k)
        physical_k_surface = lr_surface_to_k_surface(physical_lr_surface, grid_k)

        cdf_r_surface = cdf_lr_surface
        cdf_k_surface = cdf_lr_surface

        base_physical_cdf_r_surface = base_physical_cdf_lr_surface
        base_physical_cdf_k_surface = base_physical_cdf_lr_surface

        physical_cdf_r_surface = physical_cdf_lr_surface
        physical_cdf_k_surface = physical_cdf_lr_surface

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

        base_physical_moments = physical_moments_table(
            T_grid=T_grid,
            r_grid=grid_lr,
            physical_lr_surface=base_physical_lr_surface,
        )

        return build_transform_output(
            method_name=self.method_name,
            info=info,
            fit_trim_alpha=self.fit_trim_alpha,
            T_grid=T_grid,
            matched_T_grid=matched_T_grid,
            status_by_T=status_by_T,
            grid_lr=grid_lr,
            grid_r=grid_r,
            grid_k=grid_k,
            rnd_lr_surface=rnd_lr_surface,
            rnd_r_surface=rnd_r_surface,
            rnd_k_surface=rnd_k_surface,
            cdf_lr_surface=cdf_lr_surface,
            cdf_r_surface=cdf_r_surface,
            cdf_k_surface=cdf_k_surface,
            base_physical_lr_surface=base_physical_lr_surface,
            base_physical_r_surface=base_physical_r_surface,
            base_physical_k_surface=base_physical_k_surface,
            base_physical_cdf_lr_surface=base_physical_cdf_lr_surface,
            base_physical_cdf_r_surface=base_physical_cdf_r_surface,
            base_physical_cdf_k_surface=base_physical_cdf_k_surface,
            physical_lr_surface=physical_lr_surface,
            physical_r_surface=physical_r_surface,
            physical_k_surface=physical_k_surface,
            physical_cdf_lr_surface=physical_cdf_lr_surface,
            physical_cdf_r_surface=physical_cdf_r_surface,
            physical_cdf_k_surface=physical_cdf_k_surface,
            pricing_kernel_surface=pricing_kernel_surface,
            rra_surface=rra_surface,
            measure_weight_surface=measure_weight_surface,
            base_pricing_kernel_surface=base_pricing_kernel_surface,
            base_measure_weight_surface=base_measure_weight_surface,
            fit_diagnostics=self.fit_diagnostics_,
            S0=S0,
            physical_moments=physical_moments,
            risk_neutral_moments=risk_neutral_moments,
            base_physical_moments=base_physical_moments,
        )

    def _extract_surfaces(self, info):
        ks = self.key_spec

        x_grid = _as_1d(info[ks.x_grid_key])
        rnd_lr_surface = np.asarray(info[ks.pdf_surface_key], dtype=float)
        T_grid = _as_1d(info[ks.T_grid_key])

        cdf_lr_surface = np.vstack([
            _cdf_from_density(x_grid, rnd_lr_surface[j, :], eps=1e-14)
            for j in range(rnd_lr_surface.shape[0])
        ])

        return x_grid, rnd_lr_surface, cdf_lr_surface, T_grid

    def _select_fit_maturities(self, history_by_T: Dict[float, pd.DataFrame]) -> List[float]:
        available = np.asarray(sorted(history_by_T.keys()), dtype=float)

        if available.size == 0:
            return []

        if self.fit_maturities is None:
            return [float(x) for x in available]

        selected = []

        for T_req in self.fit_maturities:
            idx = int(np.argmin(np.abs(available - float(T_req))))
            T_match = float(available[idx])
            err = abs(T_match - float(T_req))

            if self.maturity_match_tol is not None and err > self.maturity_match_tol:
                raise ValueError(
                    f"Requested maturity {T_req} did not match any available maturity within "
                    f"maturity_match_tol={self.maturity_match_tol}. Closest was {T_match}."
                )

            selected.append(T_match)

        out = []
        seen = set()

        for T in selected:
            if T not in seen:
                out.append(float(T))
                seen.add(T)

        return out

    def _match_maturity(self, T: float) -> float:
        if not self.models_by_T_:
            raise RuntimeError("No fitted maturity models available.")

        keys = np.asarray(sorted(self.models_by_T_.keys()), dtype=float)
        idx = int(np.argmin(np.abs(keys - float(T))))
        T_match = float(keys[idx])
        err = abs(T_match - float(T))

        if self.maturity_match_tol is not None and err > self.maturity_match_tol:
            raise ValueError(
                f"Requested maturity {T} is too far from fitted maturities. "
                f"Closest fitted maturity is {T_match}; tolerance is {self.maturity_match_tol}."
            )

        return T_match

    def _apply_fit_trim(self, hist: pd.DataFrame) -> pd.DataFrame:
        hist = hist.copy()

        if self.fit_trim_alpha is None:
            hist["used_in_fit"] = np.isfinite(hist["pit"].astype(float))
            return hist

        a_left, a_right = self.fit_trim_alpha
        pit = hist["pit"].astype(float)

        hist["used_in_fit"] = (
            np.isfinite(pit)
            & (pit >= a_left)
            & (pit <= 1.0 - a_right)
        )

        return hist

    def _print_fit_progress(self, diag: FitDiagnostics):
        loss_str = "nan" if not np.isfinite(diag.loss) else f"{diag.loss:.6g}"

        print(
            f"[fit] T={365.0 * diag.maturity:.0f}d "
            f"({diag.maturity:.6g}y) | method={diag.method} | "
            f"used={diag.n_used}/{diag.n_total} | dropped={diag.n_dropped} | "
            f"{diag.loss_name}={loss_str} | status={diag.status}"
        )

        if diag.message and diag.status not in {"success", "ok"}:
            print(f"      message: {diag.message}")

    def compute_relative_risk_aversion(
        self,
        x_grid: np.ndarray,
        pricing_kernel: np.ndarray,
    ) -> np.ndarray:
        x_grid = _as_1d(x_grid)
        M = _as_1d(pricing_kernel)
        M = np.maximum(np.where(np.isfinite(M), M, np.nan), self.eps)

        if x_grid.size != M.size or x_grid.size < 3:
            return np.full_like(x_grid, np.nan, dtype=float)

        log_M = np.log(M)

        return -np.gradient(log_M, x_grid)

    def _cache_params(self) -> dict:
        return {}

    @abstractmethod
    def _fit_one_maturity(
        self,
        hist_T: pd.DataFrame,
        *,
        T: float,
    ) -> Tuple[Any, Dict[str, Any]]:
        pass

    @abstractmethod
    def _transform_surface_with_model(
        self,
        fitted_model: Any,
        x_grid: np.ndarray,
        f_q: np.ndarray,
        F_q: np.ndarray,
        *,
        T: float,
        info: dict,
    ) -> dict:
        pass