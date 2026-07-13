from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

from .config import BootstrapSpec, CacheSpec, FitDiagnostics, KeySpec
from .utils import (
    _as_1d,
    _cdf_from_density,
    _trapz_normalize_density,
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
        behavioral: bool = False,
        stock_df: Optional[pd.DataFrame] = None,
        stock_date_col: str = "date",
        volume_col: str = "volume",
        k1: float = 1.0,
        k2: float = 1.2,
        k3: float = 1.0,
        sentiment_alpha: float = 0.05,

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

        # Optional Crisostomo-style behavioral overlay, applied after the chosen transform.
        self.behavioral = bool(behavioral)
        self.stock_df = None if stock_df is None else stock_df.copy()
        self.stock_date_col = str(stock_date_col)
        self.volume_col = str(volume_col)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3)
        self.sentiment_alpha = float(sentiment_alpha)
        self.sentiment_by_date_: Dict[pd.Timestamp, dict] = {}

        self.models_by_T_: Dict[float, Any] = {}
        self.history_by_T_: Dict[float, pd.DataFrame] = {}
        self.fit_history_by_T_: Dict[float, pd.DataFrame] = {}
        self.fit_diagnostics_: Dict[float, FitDiagnostics] = {}
        self.is_fitted_: bool = False

    # --------------------------
    # Public API
    # --------------------------
    def _kde_cdf_value(self, x_now: float, history: np.ndarray) -> float:
        """
        Gaussian KDE estimate of F(x_now) from historical observations.
        Used for Crisostomo IV and volume empirical quantiles.
        """
        from scipy.stats import gaussian_kde
    
        history = np.asarray(history, dtype=float)
        history = history[np.isfinite(history)]
    
        if history.size < 5 or not np.isfinite(x_now):
            return np.nan
    
        sd = float(np.std(history, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return float(np.mean(history <= x_now))
    
        lo = min(float(np.min(history)), float(x_now)) - 4.0 * sd
        hi = max(float(np.max(history)), float(x_now)) + 4.0 * sd
    
        grid = np.linspace(lo, hi, 2000)
    
        try:
            kde = gaussian_kde(history)
            pdf = kde(grid)
        except Exception:
            return float(np.mean(history <= x_now))
    
        pdf = np.where(np.isfinite(pdf) & (pdf >= 0), pdf, 0.0)
    
        cdf = np.empty_like(grid)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(
            0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid)
        )
    
        total = cdf[-1]
        if not np.isfinite(total) or total <= self.eps:
            return float(np.mean(history <= x_now))
    
        cdf = cdf / total
    
        return float(np.clip(np.interp(x_now, grid, cdf), 0.0, 1.0))

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
        """
        Fit one transformation model per maturity slice.

        The stock dataframe is the single source for realized returns and,
        if behavioral=True, trading volume. Realized log returns are built from
        either a return column, an adjusted-price column, or price adjusted by
        an adjustment/share factor.
        """
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
                "stock_n": 0 if self.stock_df is None else len(self.stock_df),
                "price_col": price_col,
                "adjusted_price_col": adjusted_price_col,
                "adjustment_factor_col": adjustment_factor_col,
                "return_col": return_col,
                "behavioral": self.behavioral,
                "k1": self.k1,
                "k2": self.k2,
                "k3": self.k3,
                "sentiment_alpha": self.sentiment_alpha,
            })
            cached = _cache_load(self.cache_spec.folder, fit_cache_key)
            if cached is not None:
                self.__dict__.update(cached.__dict__)
                if self.verbose:
                    print(f"[cache hit] loaded fitted {self.method_name} model from {self.cache_spec.folder}/{fit_cache_key}.pkl")
                return self

        history_by_T = self.build_history_by_maturity(rnd_history_dict, logreturns)
        self.history_by_T_ = history_by_T
        if self.behavioral:
            self.sentiment_by_date_ = self._build_behavioral_sentiment_by_date(rnd_history_dict)
        else:
            self.sentiment_by_date_ = {}
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

    def _return_series_from_stock_df(
        self,
        stock_df: pd.DataFrame,
        *,
        price_col: Optional[str] = None,
        adjusted_price_col: Optional[str] = None,
        adjustment_factor_col: Optional[str] = None,
        return_col: Optional[str] = None,
    ) -> pd.Series:
        """Build daily log returns from the stock dataframe."""
        df = stock_df.copy()

        if self.stock_date_col in df.columns:
            idx = pd.to_datetime(df[self.stock_date_col]).dt.tz_localize(None)
            df = df.set_index(idx)
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)

        df = df.sort_index()

        if return_col is not None and return_col in df.columns:
            s = pd.to_numeric(df[return_col], errors="coerce")
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s.dropna().sort_index()

        if adjusted_price_col is None:
            for c in ("adj_price", "adjusted_price", "adj_close", "adjusted_close", "Adj Close", "adj_prc"):
                if c in df.columns:
                    adjusted_price_col = c
                    break

        if adjusted_price_col is not None and adjusted_price_col in df.columns:
            price = pd.to_numeric(df[adjusted_price_col], errors="coerce")
        else:
            if price_col is None:
                for c in ("price", "close", "Close", "prc", "PRC", "PX_LAST"):
                    if c in df.columns:
                        price_col = c
                        break
            if price_col is None or price_col not in df.columns:
                raise ValueError(
                    "Could not infer a price column. Supply price_col, adjusted_price_col, or return_col."
                )

            price = pd.to_numeric(df[price_col], errors="coerce").abs()

            if adjustment_factor_col is None:
                for c in ("ajexdi", "adj_factor", "adjustment_factor", "cfacpr", "split_factor"):
                    if c in df.columns:
                        adjustment_factor_col = c
                        break

            if adjustment_factor_col is not None and adjustment_factor_col in df.columns:
                factor = pd.to_numeric(df[adjustment_factor_col], errors="coerce")
                # Compustat-style ajexdi usually adjusts price by division.
                price = price / factor.replace(0.0, np.nan)

        price = price.replace([np.inf, -np.inf], np.nan).dropna()
        price = price[price > 0].sort_index()
        ret = np.log(price / price.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        ret.name = "log_return"
        return ret

    # --------------------------
    # Transform helpers
    # --------------------------

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

                f_p_final, F_p_final, kernel_final, weight_final, overlay_info = self._apply_behavioral_overlay_if_needed(
                    x_grid=x_grid,
                    f_q=rnd_lr_surface[j, :],
                    f_p=f_base,
                    info=info,
                    T=T,
                )

                physical_lr_surface[j, :] = f_p_final
                physical_cdf_lr_surface[j, :] = F_p_final
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
        base_physical_r_surface = base_physical_lr_surface / np.maximum(grid_r[None, :], 1e-300)
        physical_r_surface = physical_lr_surface / np.maximum(grid_r[None, :], 1e-300)
    
        # Terminal price / strike K = S0 * R
        # f_K(K) = f_lr(lr) * |d lr / dK| = f_lr(lr) / K
        rnd_k_surface = rnd_lr_surface / np.maximum(grid_k[None, :], 1e-300)
        base_physical_k_surface = base_physical_lr_surface / np.maximum(grid_k[None, :], 1e-300)
        physical_k_surface = physical_lr_surface / np.maximum(grid_k[None, :], 1e-300)
    
        # CDF values are invariant under monotone transformations.
        cdf_r_surface = cdf_lr_surface
        cdf_k_surface = cdf_lr_surface
    
        base_physical_cdf_r_surface = base_physical_cdf_lr_surface
        base_physical_cdf_k_surface = base_physical_cdf_lr_surface

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
    
            # base physical density and CDF before behavioral adjustment
            "base_physical_lr_surface": base_physical_lr_surface,
            "base_physical_r_surface": base_physical_r_surface,
            "base_physical_k_surface": base_physical_k_surface,
            "base_physical_cdf_lr_surface": base_physical_cdf_lr_surface,
            "base_physical_cdf_r_surface": base_physical_cdf_r_surface,
            "base_physical_cdf_k_surface": base_physical_cdf_k_surface,

            # final physical density and CDF
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
            "base_pricing_kernel_surface": base_pricing_kernel_surface,
            "base_measure_weight_surface": base_measure_weight_surface,
    
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

            "base_physical_moments": physical_moments_table(
                T_grid=T_grid,
                r_grid=grid_lr,
                physical_lr_surface=base_physical_lr_surface,
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
    # Optional behavioral overlay
    # --------------------------

    def _date_from_info(self, info: dict) -> pd.Timestamp | None:
        for key in ("date", "day", "anchor_date", "valuation_date"):
            if key in info and info[key] is not None:
                try:
                    return pd.Timestamp(info[key]).tz_localize(None)
                except Exception:
                    return None
        meta = info.get("meta", {}) if isinstance(info.get("meta", {}), dict) else {}
        for key in ("date", "day", "anchor_date", "valuation_date"):
            if key in meta and meta[key] is not None:
                try:
                    return pd.Timestamp(meta[key]).tz_localize(None)
                except Exception:
                    return None
        return None

    def _extract_rn_skew(self, info: dict, T: float) -> float:
        """Use already-computed RND moments. Does not recompute BKM."""
        candidates = [
            info.get("rnd_moments_table"),
            info.get("risk_neutral_moments"),
            info.get("moments"),
        ]
        for tbl in candidates:
            if isinstance(tbl, pd.DataFrame):
                tmp = tbl.copy()
                skew_col = "skew" if "skew" in tmp.columns else ("skew_r" if "skew_r" in tmp.columns else None)
                if skew_col is None:
                    continue
                if "T" in tmp.columns:
                    idx = (tmp["T"].astype(float) - float(T)).abs().idxmin()
                else:
                    idx = tmp.index[0]
                try:
                    val = float(tmp.loc[idx, skew_col])
                    if np.isfinite(val):
                        return val
                except Exception:
                    pass
        return np.nan

    def _standardize_stock_df_for_sentiment(self) -> Optional[pd.DataFrame]:
        if self.stock_df is None:
            return None
        df = self.stock_df.copy()
        if self.stock_date_col in df.columns:
            df["_date"] = pd.to_datetime(df[self.stock_date_col]).dt.tz_localize(None)
            df = df.set_index("_date")
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        return df
    def _build_behavioral_sentiment_by_date(
        self,
        rnd_history_dict: Dict[Any, dict],
    ) -> Dict[pd.Timestamp, dict]:
        """
        Build Crisostomo-style time-varying sentiment inputs by date.
    
        theta1:
            Investor optimism / pessimism from empirical quantiles of IV changes.
    
            Low IV-change quantile  -> excessive optimism  -> theta1 < 0
            High IV-change quantile -> excessive pessimism -> theta1 > 0
    
        theta2:
            Investor confidence from empirical quantiles of volume ratio.
    
            Low volume quantile  -> underconfidence -> theta2 < 1
            High volume quantile -> overconfidence  -> theta2 > 1
    
        theta3:
            Maturity-specific tail sentiment computed later from RND skewness.
        """
        keys = sorted(rnd_history_dict.keys(), key=lambda x: pd.Timestamp(x))
        stock = self._standardize_stock_df_for_sentiment()
    
        records = []
    
        for raw_date in keys:
            date = pd.Timestamp(raw_date).tz_localize(None)
            info = rnd_history_dict[raw_date]
    
            sigma = _find_sigma(info, self.key_spec.sigma_keys, default=np.nan)
    
            volume_ratio = np.nan
            if stock is not None and self.volume_col in stock.columns:
                v = pd.to_numeric(stock[self.volume_col], errors="coerce").sort_index()
    
                # Current one-month trading volume proxy: last 20 trading days up to date.
                cur = v.loc[v.index <= date].tail(20).sum()
    
                # Prior three-month average monthly volume proxy:
                # previous 60 trading days before the current 20-day block.
                hist = v.loc[v.index < date].tail(80)
                prev = hist.iloc[:-20] if len(hist) > 20 else pd.Series(dtype=float)
    
                prev_20d_equiv = prev.mean() * 20.0 if len(prev) > 0 else np.nan
    
                if (
                    np.isfinite(cur)
                    and np.isfinite(prev_20d_equiv)
                    and prev_20d_equiv > 0
                ):
                    volume_ratio = float(cur / prev_20d_equiv)
    
            records.append(
                {
                    "date": date,
                    "sigma": sigma,
                    "volume_ratio": volume_ratio,
                }
            )
    
        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    
        if df.empty:
            return {}
    
        # Crisostomo-style IV change:
        # current IV minus average IV over previous three observations/months.
        df["iv_lag_avg"] = df["sigma"].rolling(3, min_periods=1).mean().shift(1)
        df["iv_change"] = df["sigma"] - df["iv_lag_avg"]
    
        out: Dict[pd.Timestamp, dict] = {}
    
        a = float(getattr(self, "sentiment_alpha", 0.05))
    
        for i, row in df.iterrows():
            date = pd.Timestamp(row["date"])
    
            past_iv = df.loc[: i - 1, "iv_change"].dropna().to_numpy(dtype=float)
            past_vol = df.loc[: i - 1, "volume_ratio"].dropna().to_numpy(dtype=float)
    
            iv_q = np.nan
            vol_q = np.nan
    
            theta1 = 0.0
            theta2 = 1.0
    
            # ------------------------------------------------------------
            # theta1: optimism / pessimism from IV-change quantile
            # ------------------------------------------------------------
            iv_now = float(row["iv_change"]) if np.isfinite(row["iv_change"]) else np.nan
    
            if past_iv.size >= 20 and np.isfinite(iv_now):
                iv_q = self._kde_cdf_value(iv_now, past_iv)   
                
                info_t = (
                rnd_history_dict.get(date)
                or rnd_history_dict.get(str(date.date()))
                or rnd_history_dict.get(pd.Timestamp(date))
                or {}
            )
                if info_t is None:
                    info_t = rnd_history_dict.get(row["date"], {})
    
                try:
                    rate = float(info_t.get("r", 0.0))
                except Exception:
                    rate = 0.0
    
                # Use a one-month risk-free return scale, as in the paper.
                r_month = np.exp(rate / 12.0) - 1.0 if np.isfinite(rate) else 0.0
    
                if iv_q < a:
                    # Low IV change = excessive optimism.
                    # Correct by shifting density left.
                    theta1 = -float(self.k1) * r_month * ((a - iv_q) / a)
    
                elif iv_q > 1.0 - a:
                    # High IV change = excessive pessimism.
                    # Correct by shifting density right.
                    theta1 = float(self.k1) * r_month * ((iv_q - (1.0 - a)) / a)
    
            # ------------------------------------------------------------
            # theta2: confidence / overconfidence from volume quantile
            # ------------------------------------------------------------
            vol_now = (
                float(row["volume_ratio"])
                if np.isfinite(row["volume_ratio"])
                else np.nan
            )
    
            if past_vol.size >= 20 and np.isfinite(vol_now):
                vol_q = self._kde_cdf_value(vol_now, past_vol)
                if vol_q < a:
                    theta2 = float(
                        self.k2 ** ((vol_q - a) / a)
                    )
                
                elif vol_q > 1.0 - a:
                    theta2 = float(
                        self.k2 ** ((vol_q - (1.0 - a)) / a)
                    )
                
                else:
                    theta2 = 1.0
    
            out[date] = {
                "iv_quantile": iv_q,
                "volume_quantile": vol_q,
                "theta1": theta1,
                "theta2": theta2,
                "iv_change": iv_now,
                "volume_ratio": vol_now,
            }
    
        return out

    
    def plot_empirical_vs_kde(
        data,
        *,
        x_now=None,
        title="Empirical distribution vs Gaussian KDE",
        xlabel="Value",
        bins=30,
        grid_n=1000,
    ):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
    
        if data.size < 5:
            raise ValueError("Need at least 5 finite observations.")
    
        sd = np.std(data, ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError("Data has zero or invalid standard deviation.")
    
        lo = min(np.min(data), x_now) if x_now is not None and np.isfinite(x_now) else np.min(data)
        hi = max(np.max(data), x_now) if x_now is not None and np.isfinite(x_now) else np.max(data)
    
        grid = np.linspace(lo - 3 * sd, hi + 3 * sd, grid_n)
    
        kde = gaussian_kde(data)
        kde_pdf = kde(grid)
    
        plt.figure(figsize=(9, 4.5))
    
        plt.hist(
            data,
            bins=bins,
            density=True,
            alpha=0.35,
            label="Empirical histogram",
        )
    
        plt.plot(
            grid,
            kde_pdf,
            linewidth=2.2,
            label="Gaussian KDE",
        )
    
        q05 = np.quantile(data, 0.05)
        q95 = np.quantile(data, 0.95)
    
        plt.axvline(q05, linestyle="--", linewidth=1.4, label="Empirical 5% / 95%")
        plt.axvline(q95, linestyle="--", linewidth=1.4)
    
        if x_now is not None and np.isfinite(x_now):
            plt.axvline(x_now, linestyle="-", linewidth=2.0, label="Current value")
    
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()
    def _tail_theta_from_skew(self, skew: float) -> float:
        if not np.isfinite(skew):
            return 0.0
        if skew < -1.5:
            return float(self.k3 * (-skew - 1.5))
        if skew > 1.5:
            return float(-self.k3 * (skew - 1.5))
        return 0.0
    def _apply_behavioral_overlay_if_needed(self, *, x_grid, f_q, f_p, info: dict, T: float):
        x_grid = _as_1d(x_grid)
        f_q = _trapz_normalize_density(x_grid, f_q, eps=self.eps)
        f_p = _trapz_normalize_density(x_grid, f_p, eps=self.eps)
    
        if not self.behavioral:
            F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)
            kernel = f_q / np.maximum(f_p, self.eps)
            Em = float(np.trapezoid(kernel * f_p, x_grid))
            if np.isfinite(Em) and Em > self.eps:
                kernel = kernel / Em
            weight = f_p / np.maximum(f_q, self.eps)
            return f_p, F_p, kernel, weight, {"enabled": False}
    
        date = self._date_from_info(info)
        s = self.sentiment_by_date_.get(date, {}) if date is not None else {}
    
        theta1 = float(s.get("theta1", 0.0))
        theta2 = float(s.get("theta2", 1.0))
    
        if not np.isfinite(theta1):
            theta1 = 0.0
    
        if not np.isfinite(theta2) or theta2 <= self.eps:
            theta2 = 1.0
    
        skew = self._extract_rn_skew(info, T=float(T))
        theta3 = self._tail_theta_from_skew(skew)
    
        # ------------------------------------------------------------
        # Mean/variance correction, eq. (14)
        # x_tilde = theta1 + theta2*x + (1-theta2)*mu
        #
        # If y = theta1 + theta2*x + (1-theta2)*mu,
        # then x = (y - theta1 - (1-theta2)*mu) / theta2.
        # This is equivalent to:
        # x = (y - mu - theta1) / theta2 + mu.
        # ------------------------------------------------------------
        mu = float(np.trapezoid(x_grid * f_p, x_grid))
    
        x_preimage = (x_grid - mu - theta1) / theta2 + mu
        f_mv = np.interp(
            x_preimage,
            x_grid,
            f_p,
            left=0.0,
            right=0.0,
        ) / theta2
    
        f_mv = _trapz_normalize_density(x_grid, f_mv, eps=self.eps)
    
        # ------------------------------------------------------------
        # Tail sentiment SDF component, eq. (16)
        # Positive theta3:
        #   m_ts is larger in the left tail and smaller in the right tail.
        #   Since density is divided by the SDF, probability moves from
        #   left tail to right tail.
        # ------------------------------------------------------------
        F_mv = _cdf_from_density(x_grid, f_mv, eps=self.eps)
    
        a = float(getattr(self, "sentiment_alpha", 0.05))
    
        if np.all(np.isfinite(F_mv)):
            q_left = float(np.interp(a, F_mv, x_grid))
            q_right = float(np.interp(1.0 - a, F_mv, x_grid))
        else:
            q_left = np.nan
            q_right = np.nan
    
        m_ts = np.ones_like(x_grid, dtype=float)
    
        if (
            np.isfinite(theta3)
            and abs(theta3) > 0
            and np.isfinite(q_left)
            and np.isfinite(q_right)
        ):
            left = x_grid < q_left
            right = x_grid > q_right
    
            m_ts[left] = np.exp(
                np.clip(theta3 * (q_left - x_grid[left]), -700, 700)
            )
    
            m_ts[right] = np.exp(
                np.clip(-theta3 * (x_grid[right] - q_right), -700, 700)
            )
    
        # Eq. (19): density is divided by the SDF component and normalized.
        f_final = f_mv / np.maximum(m_ts, self.eps)
        f_final = _trapz_normalize_density(x_grid, f_final, eps=self.eps)
        F_final = _cdf_from_density(x_grid, f_final, eps=self.eps)
    
        # Final pricing kernel is Q / final P.
        kernel = f_q / np.maximum(f_final, self.eps)
        Em = float(np.trapezoid(kernel * f_final, x_grid))
        if np.isfinite(Em) and Em > self.eps:
            kernel = kernel / Em
    
        weight = f_final / np.maximum(f_q, self.eps)
    
        return f_final, F_final, kernel, weight, {
            "enabled": True,
            "date": None if date is None else str(date.date()),
            "theta1": theta1,
            "theta2": theta2,
            "theta3": theta3,
            "rnd_skew_used": skew,
            "iv_quantile": s.get("iv_quantile", np.nan),
            "volume_quantile": s.get("volume_quantile", np.nan),
            "q_left": q_left,
            "q_right": q_right,
        }
    

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