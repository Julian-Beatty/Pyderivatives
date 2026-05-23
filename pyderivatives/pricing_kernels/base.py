from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import copy

import numpy as np
import pandas as pd

from .config import BootstrapSpec, CacheSpec, FitDiagnostics, KeySpec
from .utils import (
    _as_1d,
    _cdf_from_density,
    _safe_interp,
    _find_spot,
    _find_sigma,
    _cache_key,
    _cache_load,
    _cache_save,
    _block_indices_circular,
)
from .moments import physical_moments_table

def _validate_fit_trim_alpha(
    fit_trim_alpha: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """
    Validate trimming specification.

    Parameters
    ----------
    fit_trim_alpha:
        (left_alpha, right_alpha)

    Returns
    -------
    tuple or None
    """
    if fit_trim_alpha is None:
        return None

    if len(fit_trim_alpha) != 2:
        raise ValueError("fit_trim_alpha must have length 2.")

    a_left = float(fit_trim_alpha[0])
    a_right = float(fit_trim_alpha[1])

    if not (0.0 <= a_left < 1.0):
        raise ValueError("Left trim alpha must satisfy 0 <= alpha < 1.")

    if not (0.0 <= a_right < 1.0):
        raise ValueError("Right trim alpha must satisfy 0 <= alpha < 1.")

    if (a_left + a_right) >= 1.0:
        raise ValueError("fit_trim_alpha must satisfy left + right < 1.")

    return (a_left, a_right)
# ============================================================
# Base class
# ============================================================

class MeasureTransform(ABC):
    """
    Base class for transforming risk-neutral density surfaces into
    physical densities, pricing kernels, and RRA surfaces.

    Subclasses implement:
        _fit_one_maturity(history_T)
        _transform_surface_with_model(fitted_model, x_grid, f_q, F_q, T, info)
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

        self.models_by_T_: Dict[float, Any] = {}
        self.history_by_T_: Dict[float, pd.DataFrame] = {}
        self.fit_history_by_T_: Dict[float, pd.DataFrame] = {}
        self.fit_diagnostics_: Dict[float, FitDiagnostics] = {}
        self.is_fitted_: bool = False

    # --------------------------
    # Public API
    # --------------------------

    def fit(self, rnd_history_dict: Dict[Any, dict], logreturns: pd.DataFrame | pd.Series):
        """
        Fit one transformation model per maturity slice.

        Parameters
        ----------
        rnd_history_dict:
            {date: info_dict}, where each info_dict contains RND/CDF surfaces.
        logreturns:
            Time series of adjusted log returns. Either:
                - Series indexed by date
                - DataFrame with DatetimeIndex and one return column
                - DataFrame with columns ['date', return_col]
        """
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
            })
            cached = _cache_load(self.cache_spec.folder, fit_cache_key)
            if cached is not None:
                self.__dict__.update(cached.__dict__)
                if self.verbose:
                    print(f"[cache hit] loaded fitted {self.method_name} model from {self.cache_spec.folder}/{fit_cache_key}.pkl")
                return self

        history_by_T = self.build_history_by_maturity(rnd_history_dict, logreturns)
        self.history_by_T_ = history_by_T
        self.fit_history_by_T_ = {}
        self.models_by_T_ = {}
        self.fit_diagnostics_ = {}

        fit_T_list = self._select_fit_maturities(history_by_T)

        for T in fit_T_list:
            hist_T = history_by_T[T].copy()
            hist_T = self._apply_fit_trim(hist_T)
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

    def transform_info(
        self,
        info: dict,
        *,
        bootstrap: bool = False,
        bootstrap_spec: BootstrapSpec = BootstrapSpec(),
    ) -> dict:
        """
        Apply fitted maturity-specific transformations to an out-of-sample
        RND info dictionary.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling transform_info().")

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
                    print(f"[cache hit] loaded transformed output from {self.cache_spec.folder}/{transform_cache_key}.pkl")
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

    def transform_surface(self, x_grid: np.ndarray, f_q: np.ndarray, F_q: np.ndarray, *, T: float, info: Optional[dict] = None) -> dict:
        """Transform one maturity surface directly."""
        T_match = self._match_maturity(float(T))
        fitted = self.models_by_T_.get(T_match)
        if fitted is None:
            raise KeyError(f"No fitted model available for maturity T={T}. Closest matched T={T_match}.")
        return self._transform_surface_with_model(
            fitted, _as_1d(x_grid), _as_1d(f_q), _as_1d(F_q), T=T_match, info=info or {}
        )

    # --------------------------
    # History construction
    # --------------------------
    def build_history_by_maturity(
        self,
        rnd_history_dict: Dict[Any, dict],
        logreturns: pd.DataFrame | pd.Series,
    ) -> Dict[float, pd.DataFrame]:
        """
        Build calibration / fitting sample by maturity.
    
        Assumption:
            logreturns are daily adjusted log returns. For a maturity of T years,
            horizon_days = round(365*T). The realized return is the sum of daily
            log returns over that horizon after the anchor date.
        """
        ret_series = self._standardize_return_series(logreturns)
    
        keys = sorted(rnd_history_dict.keys(), key=lambda x: pd.Timestamp(x))
        if not keys:
            raise ValueError("rnd_history_dict is empty.")
    
        first_info = rnd_history_dict[keys[0]]
        T_grid = _as_1d(first_info[self.key_spec.T_grid_key])
    
        rows_by_T: Dict[float, list] = {float(T): [] for T in T_grid}
    
        for raw_date in keys:
            date = pd.Timestamp(raw_date).tz_localize(None)
            info = rnd_history_dict[raw_date]
    
            if not info.get("success", True):
                continue
    
            x_grid, rnd_lr_surface, cdf_lr_surface, T_grid_i = self._extract_surfaces(info)
    
            sigma = _find_sigma(info, self.key_spec.sigma_keys, default=1.0)
            S0 = _find_spot(info, self.key_spec.spot_keys)
    
            for j, T in enumerate(T_grid_i):
                T = float(T)
                horizon_days = max(1, int(round(365.0 * T)))
    
                realized_return, end_date = self._realized_horizon_return(
                    ret_series,
                    date,
                    horizon_days,
                )
    
                if not np.isfinite(realized_return):
                    continue
    
                f_q = rnd_lr_surface[j, :]
                F_q = cdf_lr_surface[j, :]
    
                pit = _safe_interp(realized_return, x_grid, F_q)
    
                if not np.isfinite(pit):
                    continue
    
                rows_by_T.setdefault(T, []).append(
                    {
                        "date": date,
                        "end_date": end_date,
                        "T": T,
                        "horizon_days": horizon_days,
                        "realized_return": float(realized_return),
    
                        # This is only the RND percentile of the realized return.
                        # It is used for beta/nonparametric calibration and optional trimming.
                        "pit": float(np.clip(pit, self.eps, 1.0 - self.eps)),
    
                        "sigma": float(sigma),
                        "S0": np.nan if S0 is None else float(S0),
    
                        # Store log-return grid and corresponding RND/CDF slice.
                        "x_grid": x_grid,
                        "f_q": f_q,
                        "F_q": F_q,
                    }
                )
    
        out: Dict[float, pd.DataFrame] = {}
    
        columns = [
            "date",
            "end_date",
            "T",
            "horizon_days",
            "realized_return",
            "pit",
            "sigma",
            "S0",
            "x_grid",
            "f_q",
            "F_q",
        ]
    
        for T, rows in rows_by_T.items():
            if rows:
                out[float(T)] = (
                    pd.DataFrame(rows)
                    .sort_values("date")
                    .reset_index(drop=True)
                )
            else:
                out[float(T)] = pd.DataFrame(columns=columns)
    
        return out

    def _standardize_return_series(self, logreturns: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(logreturns, pd.Series):
            s = logreturns.copy()
        else:
            df = logreturns.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" not in df.columns:
                    raise ValueError("logreturns must have a DatetimeIndex or a 'date' column.")
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise ValueError("logreturns DataFrame must contain at least one numeric return column.")
            s = df[numeric_cols[0]].copy()

        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
        return s

    def _realized_horizon_return(self, ret_series: pd.Series, anchor_date: pd.Timestamp, horizon_days: int) -> Tuple[float, Optional[pd.Timestamp]]:
        if ret_series.empty:
            return np.nan, None
        anchor_date = pd.Timestamp(anchor_date).tz_localize(None)
        target_date = anchor_date + pd.Timedelta(days=int(horizon_days))

        # Use trading-day available returns strictly after anchor date and up to target date.
        window = ret_series.loc[(ret_series.index > anchor_date) & (ret_series.index <= target_date)]
        if window.empty:
            return np.nan, None
        return float(window.sum()), pd.Timestamp(window.index[-1])

    # --------------------------
    # Transform helpers
    # --------------------------

    def _transform_info_no_bootstrap(self, info: dict) -> dict:
        x_grid, rnd_lr_surface, cdf_lr_surface, T_grid = self._extract_surfaces(info)
    
        S0 = _find_spot(info, self.key_spec.spot_keys)
    
        nT, nx = rnd_lr_surface.shape
    
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
    
                physical_lr_surface[j, :] = res["f_p"]
                physical_cdf_lr_surface[j, :] = res["F_p"]
                measure_weight_surface[j, :] = res["weight"]
                pricing_kernel_surface[j, :] = res["pricing_kernel"]
    
                rra_surface[j, :] = self.compute_relative_risk_aversion(
                    x_grid,
                    res["pricing_kernel"],
                )
    
                matched_T_grid[j] = T_fit
    
                status_by_T.append(
                    {
                        "T": T,
                        "matched_T": T_fit,
                        "status": "success",
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
    
        # ============================================================
        # Standardized grids
        # ============================================================
    
        grid_lr = np.asarray(x_grid, float)
        grid_r = np.exp(grid_lr)
    
        if S0 is not None and np.isfinite(S0):
            grid_k = float(S0) * grid_r
        else:
            grid_k = np.full_like(grid_r, np.nan)
    
        # ============================================================
        # Convert densities across axes
        # ============================================================
    
        # Gross return R = exp(lr)
        # f_R(R) = f_lr(lr) * |d lr / dR| = f_lr(lr) / R
        rnd_r_surface = rnd_lr_surface / np.maximum(grid_r[None, :], 1e-300)
        physical_r_surface = physical_lr_surface / np.maximum(grid_r[None, :], 1e-300)
    
        # Terminal price / strike K = S0 * R
        # f_K(K) = f_lr(lr) * |d lr / dK| = f_lr(lr) / K
        rnd_k_surface = rnd_lr_surface / np.maximum(grid_k[None, :], 1e-300)
        physical_k_surface = physical_lr_surface / np.maximum(grid_k[None, :], 1e-300)
    
        # CDF values are invariant under monotone transformations.
        cdf_r_surface = cdf_lr_surface
        cdf_k_surface = cdf_lr_surface
    
        physical_cdf_r_surface = physical_cdf_lr_surface
        physical_cdf_k_surface = physical_cdf_lr_surface
    
        # ============================================================
        # Output dictionary: new naming convention only
        # ============================================================
    
        out = {
            # metadata
            "success": True,
            "method": self.method_name,
            "ticker": info.get("ticker", None),
            "model": info.get("model", None),
            "params": info.get("params", None),
            "meta": info.get("meta", {}),
            "fit_trim_alpha": self.fit_trim_alpha,
    
            # maturity information
            "T_grid": T_grid,
            "matched_T_grid": matched_T_grid,
            "transform_status_by_T": status_by_T,
    
            # grids
            "grid_lr": grid_lr,
            "grid_r": grid_r,
            "grid_k": grid_k,
    
            # risk-neutral density and CDF
            "rnd_lr_surface": rnd_lr_surface,
            "rnd_r_surface": rnd_r_surface,
            "rnd_k_surface": rnd_k_surface,
    
            "cdf_lr_surface": cdf_lr_surface,
            "cdf_r_surface": cdf_r_surface,
            "cdf_k_surface": cdf_k_surface,
    
            # physical density and CDF
            "physical_lr_surface": physical_lr_surface,
            "physical_r_surface": physical_r_surface,
            "physical_k_surface": physical_k_surface,
    
            "physical_cdf_lr_surface": physical_cdf_lr_surface,
            "physical_cdf_r_surface": physical_cdf_r_surface,
            "physical_cdf_k_surface": physical_cdf_k_surface,
    
            # pricing kernel objects
            "pricing_kernel_surface": pricing_kernel_surface,
            "relative_risk_aversion_surface": rra_surface,
            "measure_weight_surface": measure_weight_surface,
    
            # diagnostics
            "fit_diagnostics": self.fit_diagnostics_,
    
            # spot and rates
            "S0": S0,
            "r": info.get("r", np.nan),
            "q": info.get("q", 0.0),
    
            # moments
            "physical_moments": physical_moments_table(
                T_grid=T_grid,
                r_grid=grid_lr,
                physical_lr_surface=physical_lr_surface,
            ),
    
            "risk_neutral_moments": physical_moments_table(
                T_grid=T_grid,
                r_grid=grid_lr,
                physical_lr_surface=rnd_lr_surface,
            ),
        }
    
        return out

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
        """
        Select which maturity slices to fit.

        If self.fit_maturities is None, fit all available maturities.
        Otherwise, each requested maturity is matched to the nearest available
        maturity in history_by_T.
        """
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

        # preserve order and remove duplicates
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

    def compute_relative_risk_aversion(self, x_grid: np.ndarray, pricing_kernel: np.ndarray) -> np.ndarray:
        """
        With x = log(S_T/S0), RRA = - d log M / d x.
        """
        x_grid = _as_1d(x_grid)
        M = _as_1d(pricing_kernel)
        M = np.maximum(np.where(np.isfinite(M), M, np.nan), self.eps)
        if x_grid.size != M.size or x_grid.size < 3:
            return np.full_like(x_grid, np.nan, dtype=float)
        log_M = np.log(M)
        return -np.gradient(log_M, x_grid)

    # --------------------------
    # Bootstrap
    # --------------------------

    def _bootstrap_transform_info(self, info: dict, bootstrap_spec: BootstrapSpec) -> dict:
        if not bootstrap_spec.enabled:
            bootstrap_spec = BootstrapSpec(
                enabled=True,
                B=bootstrap_spec.B,
                block_length=bootstrap_spec.block_length,
                ci_level=bootstrap_spec.ci_level,
                random_state=bootstrap_spec.random_state,
                keep_draws=bootstrap_spec.keep_draws,
            )

        rng = np.random.default_rng(bootstrap_spec.random_state)
        B = int(bootstrap_spec.B)
        alpha = 1.0 - float(bootstrap_spec.ci_level)
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        base = self._transform_info_no_bootstrap(info)
        shape = base["physical_lr_surface"].shape

        f_p_draws = np.full((B, *shape), np.nan)
        F_p_draws = np.full((B, *shape), np.nan)
        kernel_draws = np.full((B, *shape), np.nan)
        rra_draws = np.full((B, *shape), np.nan)

        T_grid = _as_1d(base["T_grid"])
        boot_success = 0
        boot_fail = 0

        # Bootstrap each maturity independently here. If you want strict joint-date
        # resampling across maturities, replace this with panel-level date block sampling.
        for b in range(B):
            boot_model = copy.deepcopy(self)
            boot_model.models_by_T_ = {}
            boot_model.fit_diagnostics_ = {}

            for T, hist_T in self.fit_history_by_T_.items():
                if len(hist_T) < self.min_obs:
                    continue
                idx = _block_indices_circular(len(hist_T), bootstrap_spec.block_length, rng)
                hist_b = hist_T.iloc[idx].reset_index(drop=True)
                try:
                    fitted_b, diag_b = boot_model._fit_one_maturity(hist_b, T=float(T))
                    boot_model.models_by_T_[float(T)] = fitted_b
                except Exception:
                    continue

            boot_model.is_fitted_ = True
            try:
                out_b = boot_model._transform_info_no_bootstrap(info)
                f_p_draws[b] = out_b["physical_lr_surface"]
                F_p_draws[b] = out_b["physical_cdf_lr_surface"]
                kernel_draws[b] = out_b["pricing_kernel_surface"]
                rra_draws[b] = out_b["relative_risk_aversion_surface"]
                boot_success += 1
            except Exception:
                boot_fail += 1

        def ci(A):
            return {
                "lower": np.nanquantile(A, q_lo, axis=0),
                "upper": np.nanquantile(A, q_hi, axis=0),
            }

        boot_out = {
            "enabled": True,
            "B": B,
            "block_length": int(bootstrap_spec.block_length),
            "ci_level": float(bootstrap_spec.ci_level),
            "successes": int(boot_success),
            "failures": int(boot_fail),
        
            "physical_lr_surface_ci": ci(f_p_draws),
            "physical_cdf_lr_surface_ci": ci(F_p_draws),
            "pricing_kernel_surface_ci": ci(kernel_draws),
            "relative_risk_aversion_surface_ci": ci(rra_draws),
        }
        if bootstrap_spec.keep_draws:
            boot_out["draws"] = {
                "physical_lr_surface": f_p_draws,
                "physical_cdf_lr_surface": F_p_draws,
                "pricing_kernel_surface": kernel_draws,
                "relative_risk_aversion_surface": rra_draws,
            }
        return boot_out

    # --------------------------
    # Cache hook
    # --------------------------

    def _cache_params(self) -> dict:
        """Subclasses can override to include method-specific specs in cache keys."""
        return {}

    # --------------------------
    # Method-specific hooks
    # --------------------------

    @abstractmethod
    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float) -> Tuple[Any, Dict[str, Any]]:
        pass

    @abstractmethod
    def _transform_surface_with_model(self, fitted_model: Any, x_grid: np.ndarray, f_q: np.ndarray, F_q: np.ndarray, *, T: float, info: dict) -> dict:
        pass