# models.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .base import ForecastDensity
from .config import EvaluationConfig, TransformCalibrationSpec
from .preprocessing import MarketData
from pyderivatives.pricing_kernel import fit_transform_window, transform_one_date
def _extract_fitted_params(transform, T_actual):
    params = {}

    try:
        T_keys = np.asarray(list(transform.models_by_T_.keys()), dtype=float)
        if T_keys.size == 0:
            return params

        T_match = float(T_keys[np.argmin(np.abs(T_keys - float(T_actual)))])
        fitted = transform.models_by_T_.get(T_match)

        if fitted is None:
            return params

        for k, v in fitted.__dict__.items():
            if isinstance(v, (int, float, str, bool, np.integer, np.floating)):
                params[k] = float(v) if isinstance(v, (np.integer, np.floating)) else v
            elif isinstance(v, np.ndarray):
                params[k] = v.tolist()
            else:
                params[k] = str(v)

        params["matched_fit_T"] = T_match

    except Exception:
        pass

    return params


def _observed_maturities(info):
    day = info.get("day", {})

    if "T_obs" in day:
        T_obs = np.asarray(day["T_obs"], dtype=float)
    elif "matched_T_grid" in info:
        # Precomputed physical dictionaries may retain the matched observed
        # maturities without the original day-level option data.
        T_obs = np.asarray(info["matched_T_grid"], dtype=float)
    else:
        return None

    T_obs = T_obs[np.isfinite(T_obs)]

    if T_obs.size == 0:
        return None

    return np.unique(np.round(T_obs, 12))

def _standardize_dates_dict(d: Dict[Any, dict]) -> Dict[pd.Timestamp, dict]:
    return {pd.Timestamp(k).tz_localize(None): v for k, v in d.items()}


def _cdf_from_pdf(x_grid, pdf):
    x = np.asarray(x_grid, dtype=float)
    f = np.asarray(pdf, dtype=float)
    f = np.where(np.isfinite(f) & (f >= 0), f, 0.0)

    dx = np.diff(x)
    inc = 0.5 * (f[1:] + f[:-1]) * dx

    cdf = np.empty_like(x)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(inc)

    if np.isfinite(cdf[-1]) and cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    return cdf



def _clean_cdf(x_grid, cdf):
    x = np.asarray(x_grid, dtype=float)
    F = np.asarray(cdf, dtype=float)

    if F.shape != x.shape or not np.all(np.isfinite(F)):
        return None

    F = np.maximum.accumulate(np.clip(F, 0.0, np.inf))
    span = float(F[-1] - F[0])

    if not np.isfinite(span) or span <= 0:
        return None

    F = (F - F[0]) / span
    F[0] = 0.0
    F[-1] = 1.0
    return F

def _normalize_pdf(x_grid, pdf):
    x = np.asarray(x_grid, dtype=float)
    f = np.asarray(pdf, dtype=float)
    f = np.where(np.isfinite(f) & (f >= 0), f, 0.0)

    area = float(np.trapezoid(f, x))
    if not np.isfinite(area) or area <= 0:
        return None

    return f / area


def _density_diagnostics(
    x_grid,
    raw_pdf,
    *,
    normalized_pdf=None,
    cdf=None,
    area_tolerance: float = 0.02,
    endpoint_ratio_tolerance: float = 0.01,
):
    """Return lightweight diagnostics for an incoming density surface row.

    Diagnostics are recorded rather than used to reject a forecast, except when
    the density cannot be normalized at all. This preserves backward-compatible
    behavior while making suspicious grids and tails visible in metadata.
    """
    x = np.asarray(x_grid, dtype=float).reshape(-1)
    raw = np.asarray(raw_pdf, dtype=float).reshape(-1)

    warnings = []
    same_shape = x.shape == raw.shape
    finite_x = np.isfinite(x)
    finite_raw = np.isfinite(raw)

    if not same_shape:
        warnings.append("grid_pdf_shape_mismatch")

    grid_increasing = bool(
        len(x) >= 2
        and np.all(finite_x)
        and np.all(np.diff(x) > 0)
    )
    if not grid_increasing:
        warnings.append("grid_not_strictly_increasing")

    negative_count = int(np.sum(finite_raw & (raw < 0)))
    nonfinite_pdf_count = int(np.sum(~finite_raw))
    nonfinite_grid_count = int(np.sum(~finite_x))

    if negative_count:
        warnings.append("negative_pdf_values")
    if nonfinite_pdf_count:
        warnings.append("nonfinite_pdf_values")
    if nonfinite_grid_count:
        warnings.append("nonfinite_grid_values")

    raw_area = np.nan
    if same_shape and len(x) >= 2 and grid_increasing:
        raw_for_area = np.where(finite_raw, raw, 0.0)
        raw_area = float(np.trapezoid(raw_for_area, x))
        if not np.isfinite(raw_area) or raw_area <= 0:
            warnings.append("nonpositive_or_nonfinite_pdf_area")
        elif abs(raw_area - 1.0) > float(area_tolerance):
            warnings.append("pdf_area_not_close_to_one")

    normalized_area = np.nan
    endpoint_left_ratio = np.nan
    endpoint_right_ratio = np.nan
    endpoint_max_ratio = np.nan

    if normalized_pdf is not None:
        normalized = np.asarray(normalized_pdf, dtype=float).reshape(-1)
        if normalized.shape == x.shape and len(x) >= 2 and grid_increasing:
            normalized_area = float(np.trapezoid(normalized, x))
            peak = float(np.nanmax(normalized)) if len(normalized) else np.nan
            if np.isfinite(peak) and peak > 0:
                endpoint_left_ratio = float(max(normalized[0], 0.0) / peak)
                endpoint_right_ratio = float(max(normalized[-1], 0.0) / peak)
                endpoint_max_ratio = float(max(endpoint_left_ratio, endpoint_right_ratio))
                if endpoint_max_ratio > float(endpoint_ratio_tolerance):
                    warnings.append("density_large_at_grid_endpoint")

    cdf_monotone = None
    cdf_start = np.nan
    cdf_end = np.nan
    if cdf is not None:
        F = np.asarray(cdf, dtype=float).reshape(-1)
        if F.shape != x.shape or not np.all(np.isfinite(F)):
            cdf_monotone = False
            warnings.append("invalid_cdf_shape_or_values")
        else:
            cdf_start = float(F[0])
            cdf_end = float(F[-1])
            cdf_monotone = bool(np.all(np.diff(F) >= -1e-10))
            if not cdf_monotone:
                warnings.append("cdf_not_monotone")
            if abs(cdf_start) > 1e-6 or abs(cdf_end - 1.0) > 1e-6:
                warnings.append("cdf_endpoints_not_zero_one")

    warnings = list(dict.fromkeys(warnings))

    return {
        "density_diagnostics_version": 1,
        "grid_size": int(len(x)),
        "grid_min": float(x[0]) if len(x) and np.isfinite(x[0]) else None,
        "grid_max": float(x[-1]) if len(x) and np.isfinite(x[-1]) else None,
        "grid_strictly_increasing": grid_increasing,
        "grid_nonfinite_count": nonfinite_grid_count,
        "pdf_nonfinite_count": nonfinite_pdf_count,
        "pdf_negative_count": negative_count,
        "pdf_raw_area": None if not np.isfinite(raw_area) else float(raw_area),
        "pdf_raw_area_error": None if not np.isfinite(raw_area) else float(raw_area - 1.0),
        "pdf_normalized_area": None if not np.isfinite(normalized_area) else float(normalized_area),
        "endpoint_left_pdf_to_peak": None if not np.isfinite(endpoint_left_ratio) else float(endpoint_left_ratio),
        "endpoint_right_pdf_to_peak": None if not np.isfinite(endpoint_right_ratio) else float(endpoint_right_ratio),
        "endpoint_max_pdf_to_peak": None if not np.isfinite(endpoint_max_ratio) else float(endpoint_max_ratio),
        "cdf_monotone": cdf_monotone,
        "cdf_start": None if not np.isfinite(cdf_start) else float(cdf_start),
        "cdf_end": None if not np.isfinite(cdf_end) else float(cdf_end),
        "area_tolerance": float(area_tolerance),
        "endpoint_ratio_tolerance": float(endpoint_ratio_tolerance),
        "density_anomaly": bool(warnings),
        "density_warnings": warnings,
    }


def _select_maturity_index(info, target_maturity, maturity_match_tol, config=None):
    T_grid = np.asarray(info["T_grid"], dtype=float)

    if T_grid.size == 0:
        return None, None

    maturity_selection = getattr(config, "maturity_selection", "target_grid")

    if maturity_selection == "target_grid":
        j = int(np.argmin(np.abs(T_grid - target_maturity)))
        T_actual = float(T_grid[j])

        if maturity_match_tol is not None:
            if abs(T_actual - target_maturity) > maturity_match_tol:
                return None, None

        return j, T_actual

    if maturity_selection == "nearest_observed":
        tol = getattr(config, "observed_maturity_tol", None)
        if tol is None:
            tol = maturity_match_tol

        T_obs = _observed_maturities(info)

        if T_obs is None or len(T_obs) == 0:
            return None, None

        if tol is not None:
            candidates = T_obs[np.abs(T_obs - target_maturity) <= tol]
        else:
            candidates = T_obs

        if len(candidates) == 0:
            return None, None

        T_observed = float(candidates[np.argmin(np.abs(candidates - target_maturity))])

        j = int(np.argmin(np.abs(T_grid - T_observed)))
        T_actual = float(T_grid[j])

        if abs(T_actual - T_observed) > 1e-8:
            return None, None

        return j, T_actual

    raise ValueError(
        "maturity_selection must be 'target_grid' or 'nearest_observed'."
    )


def _fit_dates(
    dates: Sequence[pd.Timestamp],
    i: int,
    config: EvaluationConfig,
):
    if i < config.reserve_obs:
        return None

    if config.window_type == "expanding":
        out = list(dates[:i])

    elif config.window_type == "rolling":
        if config.window_size is None:
            raise ValueError("window_size must be supplied for rolling windows.")
        out = list(dates[max(0, i - config.window_size):i])

    else:
        raise ValueError("window_type must be 'expanding' or 'rolling'.")

    if len(out) < config.min_fit_obs:
        return None

    return out


@dataclass
class DensityModel(ABC):
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def evaluation_dates(
        self,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> list[pd.Timestamp]:
        raise NotImplementedError

    @abstractmethod
    def forecast_one(
        self,
        *,
        date: pd.Timestamp,
        index: int,
        horizon_days: int,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> ForecastDensity:
        raise NotImplementedError

    def realized_return(
        self,
        *,
        date: pd.Timestamp,
        horizon_days: int,
        market_data: MarketData,
        config: EvaluationConfig,
    ):
        realized, end_date = market_data.realized_after(
            date,
            horizon_days,
            mode=config.realized_horizon_mode,
            tolerance_days=config.realized_match_tol_days,
        )

        if realized is None:
            raise ValueError(
                "Not enough future data to compute realized horizon return."
            )

        return realized, end_date


@dataclass
class RawRNDModel(DensityModel):
    rnd_key: str = "default"

    def _rnd_by_date(self, market_data: MarketData):
        if self.rnd_key not in market_data.rnd_dicts:
            raise KeyError(
                f"rnd_key '{self.rnd_key}' not found in market_data.rnd_dicts."
            )

        return _standardize_dates_dict(market_data.rnd_dicts[self.rnd_key])

    def evaluation_dates(
        self,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> list[pd.Timestamp]:
        return sorted(self._rnd_by_date(market_data).keys())

    def _get_cdf_surface(self, info: dict):
        if "cdf_lr_surface" in info:
            return np.asarray(info["cdf_lr_surface"], dtype=float)

        if "rnd_cdf_surface" in info:
            return np.asarray(info["rnd_cdf_surface"], dtype=float)

        x_grid = np.asarray(info["grid_lr"], dtype=float)
        density = np.asarray(info["rnd_lr_surface"], dtype=float)

        return np.vstack([
            _cdf_from_pdf(x_grid, row)
            for row in density
        ])

    def forecast_one(
        self,
        *,
        date: pd.Timestamp,
        index: int,
        horizon_days: int,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> ForecastDensity:
        
        


        target_maturity = float(getattr(config, "_override_target_maturity", config.target_maturity_for_horizon(horizon_days)))

        rnd_by_date = self._rnd_by_date(market_data)
        info = rnd_by_date[date]

        if not info.get("success", True):
            raise ValueError("RND info marked success=False.")

        j, T_actual = _select_maturity_index(
            info,
            target_maturity,
            config.maturity_match_tol,
            config=config,
        )

        if j is None:
            raise ValueError("No maturity within tolerance.")
        realized_horizon_days = int(round(365 * T_actual))

        realized, end_date = self.realized_return(
            date=date,
            horizon_days=realized_horizon_days,
            market_data=market_data,
            config=config,
        )

        x_grid = np.asarray(info["grid_lr"], dtype=float)

        raw_pdf = np.asarray(info["rnd_lr_surface"], dtype=float)[j, :]
        pdf = _normalize_pdf(x_grid, raw_pdf)

        if pdf is None:
            raise ValueError("Invalid RND pdf.")

        raw_cdf = self._get_cdf_surface(info)[j, :]
        cdf = _clean_cdf(x_grid, raw_cdf)
        if cdf is None:
            cdf = _cdf_from_pdf(x_grid, pdf)

        density_diagnostics = _density_diagnostics(
            x_grid,
            raw_pdf,
            normalized_pdf=pdf,
            cdf=cdf,
        )

        return ForecastDensity(
            date=date,
            model_name=self.name,
            horizon=int(horizon_days),
            x_grid=x_grid,
            pdf=pdf,
            cdf=cdf,
            realized=realized,
            metadata={
                "source": "raw_rnd",
                "rnd_key": self.rnd_key,
                "target_maturity": target_maturity,
                "target_horizon_days": int(horizon_days),
                "T_actual": T_actual,
                "realized_horizon_days": int(realized_horizon_days),
                "maturity_selection": getattr(config, "maturity_selection", "target_grid"),
                "end_date": end_date,
                "density_diagnostics": density_diagnostics,
                "density_anomaly": density_diagnostics["density_anomaly"],
                "density_warnings": density_diagnostics["density_warnings"],
                **self.metadata,
            }
            ,
        )


@dataclass
class PhysicalDensityModel(DensityModel):
    """Backtest a dictionary of already-computed physical densities."""

    physical_key: str = "default"

    def _physical_by_date(self, market_data: MarketData):
        if self.physical_key not in market_data.physical_dicts:
            raise KeyError(
                f"physical_key '{self.physical_key}' not found in "
                "market_data.physical_dicts."
            )

        return _standardize_dates_dict(
            market_data.physical_dicts[self.physical_key]
        )

    def evaluation_dates(
        self,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> list[pd.Timestamp]:
        return sorted(self._physical_by_date(market_data).keys())

    @staticmethod
    def _selected_moments(info: dict, j: int, T_actual: float):
        moments = info.get("physical_moments")
        if moments is None:
            return None

        try:
            if isinstance(moments, pd.DataFrame):
                if len(moments) == 0:
                    return None

                if "T" in moments.columns:
                    T = pd.to_numeric(moments["T"], errors="coerce").to_numpy()
                    k = int(np.nanargmin(np.abs(T - float(T_actual))))
                    row = moments.iloc[k]
                elif len(moments) > j:
                    row = moments.iloc[j]
                else:
                    return None

                out = {}
                for key, value in row.to_dict().items():
                    if pd.isna(value):
                        continue
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    if isinstance(value, (str, int, float, bool)):
                        out[str(key)] = value
                return out or None
        except Exception:
            return None

        return None

    def forecast_one(
        self,
        *,
        date: pd.Timestamp,
        index: int,
        horizon_days: int,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> ForecastDensity:
        target_maturity = float(getattr(config, "_override_target_maturity", config.target_maturity_for_horizon(horizon_days)))

        physical_by_date = self._physical_by_date(market_data)
        info = physical_by_date[date]

        if not info.get("success", True):
            raise ValueError("Physical density info marked success=False.")

        j, T_actual = _select_maturity_index(
            info,
            target_maturity,
            config.maturity_match_tol,
            config=config,
        )

        if j is None:
            raise ValueError("No physical-density maturity within tolerance.")

        x_grid = np.asarray(info["grid_lr"], dtype=float)
        surface = np.asarray(info["physical_lr_surface"], dtype=float)

        if surface.ndim != 2 or j >= surface.shape[0]:
            raise ValueError("Invalid physical_lr_surface shape.")

        raw_pdf = surface[j, :]
        pdf = _normalize_pdf(x_grid, raw_pdf)
        if pdf is None:
            raise ValueError("Invalid physical-density pdf.")

        cdf = None
        if "physical_cdf_lr_surface" in info:
            cdf_surface = np.asarray(info["physical_cdf_lr_surface"], dtype=float)
            if cdf_surface.ndim == 2 and j < cdf_surface.shape[0]:
                cdf = _clean_cdf(x_grid, cdf_surface[j, :])

        if cdf is None:
            cdf = _cdf_from_pdf(x_grid, pdf)

        density_diagnostics = _density_diagnostics(
            x_grid,
            raw_pdf,
            normalized_pdf=pdf,
            cdf=cdf,
        )

        realized_horizon_days = int(round(365 * T_actual))
        realized, end_date = self.realized_return(
            date=date,
            horizon_days=realized_horizon_days,
            market_data=market_data,
            config=config,
        )

        selected_moments = self._selected_moments(info, j, T_actual)

        metadata = {
            "source": "precomputed_physical",
            "physical_key": self.physical_key,
            "method": info.get("method", None),
            "target_maturity": target_maturity,
            "target_horizon_days": int(horizon_days),
            "T_actual": T_actual,
            "realized_horizon_days": int(realized_horizon_days),
            "maturity_selection": getattr(config, "maturity_selection", "target_grid"),
            "end_date": end_date,
            "density_diagnostics": density_diagnostics,
            "density_anomaly": density_diagnostics["density_anomaly"],
            "density_warnings": density_diagnostics["density_warnings"],
            **self.metadata,
        }

        if selected_moments is not None:
            metadata["physical_moments"] = selected_moments

        return ForecastDensity(
            date=date,
            model_name=self.name,
            horizon=int(horizon_days),
            x_grid=x_grid,
            pdf=pdf,
            cdf=cdf,
            realized=realized,
            metadata=metadata,
        )


@dataclass
class TransformRNDModel(DensityModel):
    """Evaluate a Q-to-P transform with model-specific calibration settings.

    Fitted transforms delegate to ``pricing_kernel.transform_one_date`` so the
    standalone pricing-kernel workflow and the density backtest use identical
    window construction, fitting, and transformation logic.
    """

    rnd_key: str = "default"
    transform: Any = None
    requires_fit: bool = True
    calibration: TransformCalibrationSpec = field(
        default_factory=TransformCalibrationSpec
    )

    # Process-local cache used only when calibration.mode == "fixed".
    # Each worker fits its fixed transform once and reuses it for all dates.
    _fixed_fitted_transform: Any = field(
        default=None, init=False, repr=False, compare=False
    )
    _fixed_window_metadata: Optional[Dict[str, Any]] = field(
        default=None, init=False, repr=False, compare=False
    )

    def _rnd_by_date(self, market_data: MarketData):
        if self.rnd_key not in market_data.rnd_dicts:
            raise KeyError(
                f"rnd_key '{self.rnd_key}' not found in market_data.rnd_dicts."
            )
        return _standardize_dates_dict(market_data.rnd_dicts[self.rnd_key])

    def evaluation_dates(
        self,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> list[pd.Timestamp]:
        dates = sorted(self._rnd_by_date(market_data).keys())
        if not self.requires_fit:
            return dates

        self.calibration.validate()
        return dates[int(self.calibration.reserve_obs):]

    @staticmethod
    def _stock_df_for_transform(market_data: MarketData) -> pd.DataFrame:
        if market_data.stock_df is not None:
            return market_data.stock_df

        stock_df = market_data.returns.to_frame("ret_1").reset_index()
        return stock_df.rename(columns={stock_df.columns[0]: "date"})

    @staticmethod
    def _get_physical_density(physical: dict, j: int):
        x_grid = np.asarray(physical["grid_lr"], dtype=float)
        raw_pdf = np.asarray(
            physical["physical_lr_surface"], dtype=float
        )[j, :]
        pdf = _normalize_pdf(x_grid, raw_pdf)

        if pdf is None:
            raise ValueError("Invalid physical pdf.")

        cdf = None
        if "physical_cdf_lr_surface" in physical:
            cdf = _clean_cdf(
                x_grid,
                np.asarray(
                    physical["physical_cdf_lr_surface"], dtype=float
                )[j, :],
            )

        if cdf is None:
            cdf = _cdf_from_pdf(x_grid, pdf)

        diagnostics = _density_diagnostics(
            x_grid,
            raw_pdf,
            normalized_pdf=pdf,
            cdf=cdf,
        )

        return x_grid, pdf, cdf, diagnostics

    @staticmethod
    def _fitted_params_from_output(physical: dict, T_actual: float) -> Dict[str, Any]:
        diagnostics = physical.get("fit_diagnostics", {})
        if not isinstance(diagnostics, dict) or not diagnostics:
            return {}

        try:
            keys = np.asarray(list(diagnostics.keys()), dtype=float)
            T_match = float(keys[np.argmin(np.abs(keys - float(T_actual)))])
            diag = diagnostics.get(T_match)
            if diag is None:
                # Dictionary keys can occasionally be serialized as strings.
                diag = diagnostics.get(str(T_match))

            if diag is None:
                return {}

            if hasattr(diag, "params"):
                params = dict(getattr(diag, "params") or {})
            elif isinstance(diag, dict):
                params = dict(diag.get("params", {}))
            else:
                params = {}

            params["matched_fit_T"] = T_match
            return params
        except Exception:
            return {}

    def _transform_for_date(
        self,
        *,
        date: pd.Timestamp,
        market_data: MarketData,
    ) -> dict:
        if self.transform is None:
            raise ValueError("transform cannot be None.")

        rnd_by_date = self._rnd_by_date(market_data)
        date = pd.Timestamp(date).tz_localize(None)
        info = rnd_by_date[date]

        if not self.requires_fit:
            model = self.transform
            physical = (
                model.transform_rnd(info)
                if hasattr(model, "transform_rnd")
                else model.transform_info(info)
            )
        else:
            cfg = self.calibration
            cfg.validate()
            transform_kwargs = cfg.transform_kwargs()
            stock_df = self._stock_df_for_transform(market_data)

            # When only a ReturnSeries is available, tell MeasureTransform.fit
            # to use the synthetic return column instead of searching for price.
            if market_data.stock_df is None:
                fit_kwargs = dict(transform_kwargs.get("fit_kwargs", {}))
                explicit_data_cols = {
                    "price_col",
                    "adjusted_price_col",
                    "adjustment_factor_col",
                    "return_col",
                }
                if not explicit_data_cols.intersection(fit_kwargs):
                    fit_kwargs["return_col"] = "ret_1"
                transform_kwargs["fit_kwargs"] = fit_kwargs

            if cfg.mode == "fixed":
                if self._fixed_fitted_transform is None:
                    fit_args = dict(transform_kwargs)
                    fit_args.pop("mode", None)

                    fitted, window_metadata = fit_transform_window(
                        transform=self.transform,
                        rnd_history_dict=rnd_by_date,
                        stock_df=stock_df,
                        target_date=None,
                        mode="fixed",
                        verbose=False,
                        **fit_args,
                    )
                    self._fixed_fitted_transform = fitted
                    self._fixed_window_metadata = dict(window_metadata)

                physical = self._fixed_fitted_transform.transform_rnd(info)
                physical.setdefault("window_fit", {})
                physical["window_fit"].update(
                    dict(self._fixed_window_metadata or {})
                )
                physical["window_fit"]["target_date"] = date
                physical["window_fit"]["reused_fitted_transform"] = True
            else:
                physical = transform_one_date(
                    transform=self.transform,
                    rnd_history_dict=rnd_by_date,
                    stock_df=stock_df,
                    target_date=date,
                    verbose=False,
                    **transform_kwargs,
                )

        # Preserve observed maturity information for nearest_observed mode.
        physical.setdefault("day", info.get("day", {}))
        return physical

    def forecast_one(
        self,
        *,
        date: pd.Timestamp,
        index: int,
        horizon_days: int,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> ForecastDensity:
        target_maturity = float(getattr(config, "_override_target_maturity", config.target_maturity_for_horizon(horizon_days)))
        physical = self._transform_for_date(
            date=date,
            market_data=market_data,
        )

        j, T_actual = _select_maturity_index(
            physical,
            target_maturity,
            config.maturity_match_tol,
            config=config,
        )
        if j is None:
            raise ValueError("No transformed maturity within tolerance.")

        fitted_params = self._fitted_params_from_output(physical, T_actual)
        realized_horizon_days = int(round(365 * T_actual))
        realized, end_date = self.realized_return(
            date=date,
            horizon_days=realized_horizon_days,
            market_data=market_data,
            config=config,
        )
        x_grid, pdf, cdf, density_diagnostics = self._get_physical_density(physical, j)

        calibration_metadata = (
            self.calibration.transform_kwargs()
            if self.requires_fit
            else None
        )
        if calibration_metadata is not None:
            calibration_metadata.pop("fit_kwargs", None)

        return ForecastDensity(
            date=date,
            model_name=self.name,
            horizon=int(horizon_days),
            x_grid=x_grid,
            pdf=pdf,
            cdf=cdf,
            realized=realized,
            metadata={
                "source": "transformed_rnd",
                "rnd_key": self.rnd_key,
                "transform_method": physical.get("method", None),
                "target_maturity": target_maturity,
                "target_horizon_days": int(horizon_days),
                "T_actual": T_actual,
                "realized_horizon_days": int(realized_horizon_days),
                "maturity_selection": getattr(
                    config, "maturity_selection", "target_grid"
                ),
                "end_date": end_date,
                "fitted_params": fitted_params,
                "calibration": calibration_metadata,
                "window_fit": physical.get("window_fit"),
                "density_diagnostics": density_diagnostics,
                "density_anomaly": density_diagnostics["density_anomaly"],
                "density_warnings": density_diagnostics["density_warnings"],
                **self.metadata,
            },
        )


def _select_model_calibration_sample(
    series: pd.Series,
    *,
    date: pd.Timestamp,
    calibration: Optional[TransformCalibrationSpec],
    config: EvaluationConfig,
) -> tuple[pd.Series, Dict[str, Any]]:
    """Select a model-specific historical estimation sample.

    When ``calibration`` is supplied, its calendar-time window settings are
    used. When it is ``None``, the legacy EvaluationConfig observation-count
    window is preserved for backward compatibility.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    target = pd.Timestamp(date).tz_localize(None)

    if calibration is None:
        available = s[s.index < target]

        if config.window_type == "expanding":
            sample = available
        elif config.window_type == "rolling":
            if config.window_size is None:
                raise ValueError(
                    "window_size must be supplied for legacy rolling windows."
                )
            sample = available.iloc[-int(config.window_size):]
        else:
            raise ValueError(
                "Legacy model calibration supports only 'expanding' or 'rolling'."
            )

        min_fit = int(config.min_fit_obs)
        metadata = {
            "source": "evaluation_config",
            "mode": config.window_type,
            "window_size": config.window_size,
            "reserve_obs": int(config.reserve_obs),
            "min_fit_dates": min_fit,
        }
    else:
        calibration.validate()
        mode = calibration.mode

        if mode == "expanding":
            sample = s[s.index < target]
            fit_start = None
            fit_end = target

        elif mode == "rolling":
            fit_start = target - pd.Timedelta(days=int(calibration.lookback_days))
            fit_end = target
            sample = s[(s.index >= fit_start) & (s.index < target)]

        elif mode == "fixed":
            fit_end = pd.Timestamp(calibration.fixed_end_date).tz_localize(None)
            fit_start = None
            sample = s[s.index <= fit_end]

        elif mode == "centered":
            fit_start = target - pd.Timedelta(days=int(calibration.lookback_days))
            fit_end = target + pd.Timedelta(days=int(calibration.lookahead_days))
            sample = s[(s.index >= fit_start) & (s.index <= fit_end)]

        else:  # guarded by validate(), retained for defensive clarity
            raise ValueError(f"Unsupported calibration mode: {mode!r}.")

        min_fit = int(calibration.min_fit_dates)
        metadata = {
            "source": "model_calibration",
            "mode": mode,
            "lookback_days": calibration.lookback_days,
            "lookahead_days": int(calibration.lookahead_days),
            "fixed_end_date": calibration.fixed_end_date,
            "reserve_obs": int(calibration.reserve_obs),
            "min_fit_dates": min_fit,
            "fit_start_date": fit_start,
            "fit_end_date": fit_end,
            "ex_post": mode == "centered",
        }

    if len(sample) < min_fit:
        raise ValueError(
            f"Insufficient fit observations: {len(sample)} < {min_fit}."
        )

    metadata["n_fit_obs"] = int(len(sample))
    metadata["sample_first_date"] = (
        None if sample.empty else pd.Timestamp(sample.index.min())
    )
    metadata["sample_last_date"] = (
        None if sample.empty else pd.Timestamp(sample.index.max())
    )
    return sample.astype(float), metadata


@dataclass
class HistoricalKDEModel(DensityModel):
    grid_size: int = 500
    grid_pad: float = 0.25
    calibration: Optional[TransformCalibrationSpec] = None

    def evaluation_dates(
        self,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> list[pd.Timestamp]:
        dates = list(market_data.returns.index)
        reserve = (
            int(self.calibration.reserve_obs)
            if self.calibration is not None
            else int(config.reserve_obs)
        )
        if self.calibration is not None:
            self.calibration.validate()
        return dates[reserve:]

    def _make_density(self, sample):
        x = np.asarray(sample, dtype=float)
        x = x[np.isfinite(x)]

        if len(x) < 20:
            raise ValueError("Too few KDE observations.")

        kde = gaussian_kde(x)

        lo = float(np.min(x))
        hi = float(np.max(x))
        width = hi - lo

        if not np.isfinite(width) or width <= 0:
            raise ValueError("Invalid KDE grid width.")

        x_grid = np.linspace(
            lo - self.grid_pad * width,
            hi + self.grid_pad * width,
            self.grid_size,
        )

        pdf = _normalize_pdf(x_grid, kde(x_grid))

        if pdf is None:
            raise ValueError("Invalid KDE pdf.")

        cdf = _cdf_from_pdf(x_grid, pdf)
        return x_grid, pdf, cdf

    def forecast_one(
        self,
        *,
        date: pd.Timestamp,
        index: int,
        horizon_days: int,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> ForecastDensity:
        reserve = (
            int(self.calibration.reserve_obs)
            if self.calibration is not None
            else int(config.reserve_obs)
        )
        if index < reserve:
            raise ValueError("Forecast date is inside reserve period.")

        realized_horizon_days = int(
            getattr(config, "_override_realized_horizon_days", horizon_days)
        )

        trailing = market_data.return_series.horizon_return(
            realized_horizon_days,
            forward=False,
        )

        sample, calibration_metadata = _select_model_calibration_sample(
            trailing,
            date=date,
            calibration=self.calibration,
            config=config,
        )

        x_grid, pdf, cdf = self._make_density(sample.values)

        realized, end_date = self.realized_return(
            date=date,
            horizon_days=realized_horizon_days,
            market_data=market_data,
            config=config,
        )

        return ForecastDensity(
            date=date,
            model_name=self.name,
            horizon=int(horizon_days),
            x_grid=x_grid,
            pdf=pdf,
            cdf=cdf,
            realized=realized,
            metadata={
                "source": "historical_kde",
                "fit_obs": int(len(sample)),
                "target_horizon_days": int(horizon_days),
                "realized_horizon_days": int(realized_horizon_days),
                "end_date": end_date,
                "calibration": calibration_metadata,
                **self.metadata,
            },
        )


@dataclass
class GARCHModel(DensityModel):
    p: int = 1
    q: int = 1
    distribution: str = "t"
    simulations: int = 20000
    grid_size: int = 500
    grid_pad: float = 0.25
    random_state: Optional[int] = 123
    scale_returns: float = 100.0
    calibration: Optional[TransformCalibrationSpec] = None

    def evaluation_dates(
        self,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> list[pd.Timestamp]:
        dates = list(market_data.returns.index)
        reserve = (
            int(self.calibration.reserve_obs)
            if self.calibration is not None
            else int(config.reserve_obs)
        )
        if self.calibration is not None:
            self.calibration.validate()
        return dates[reserve:]

    def _make_density(self, sample):
        x = np.asarray(sample, dtype=float)
        x = x[np.isfinite(x)]

        if len(x) < 20:
            raise ValueError("Too few GARCH simulation observations.")

        lo = float(np.min(x))
        hi = float(np.max(x))
        width = hi - lo

        if not np.isfinite(width) or width <= 0:
            raise ValueError("Invalid GARCH density grid width.")

        x_grid = np.linspace(
            lo - self.grid_pad * width,
            hi + self.grid_pad * width,
            int(self.grid_size),
        )

        kde = gaussian_kde(x)
        pdf = _normalize_pdf(x_grid, kde(x_grid))

        if pdf is None:
            raise ValueError("Invalid GARCH pdf.")

        cdf = _cdf_from_pdf(x_grid, pdf)
        return x_grid, pdf, cdf

    def _fit_sample(
        self,
        *,
        date: pd.Timestamp,
        market_data: MarketData,
        config: EvaluationConfig,
    ):
        return _select_model_calibration_sample(
            market_data.returns,
            date=date,
            calibration=self.calibration,
            config=config,
        )

    def _simulate_paths(self, fitted, horizon_days: int):
        try:
            sim = fitted.forecast(
                horizon=int(horizon_days),
                method="simulation",
                simulations=int(self.simulations),
                random_state=self.random_state,
            )
        except TypeError:
            sim = fitted.forecast(
                horizon=int(horizon_days),
                method="simulation",
                simulations=int(self.simulations),
            )

        values = sim.simulations.values
        arr = np.asarray(values, dtype=float)

        if arr.ndim == 3:
            arr = arr[-1, :, :]
        elif arr.ndim == 2:
            arr = arr[:, :]
        else:
            raise ValueError(f"Unexpected GARCH simulation shape: {arr.shape}")

        path_returns = np.sum(arr, axis=1)
        return path_returns / float(self.scale_returns)

    def forecast_one(
        self,
        *,
        date: pd.Timestamp,
        index: int,
        horizon_days: int,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> ForecastDensity:
        reserve = (
            int(self.calibration.reserve_obs)
            if self.calibration is not None
            else int(config.reserve_obs)
        )
        if index < reserve:
            raise ValueError("Forecast date is inside reserve period.")

        realized_horizon_days = int(
            getattr(config, "_override_realized_horizon_days", horizon_days)
        )

        sample, calibration_metadata = self._fit_sample(
            date=date,
            market_data=market_data,
            config=config,
        )

        scaled = sample * float(self.scale_returns)

        try:
            from arch import arch_model
        except ImportError as exc:
            raise ImportError(
                "GARCHModel requires the 'arch' package. Install it with: pip install arch"
            ) from exc

        am = arch_model(
            scaled,
            mean="Constant",
            vol="GARCH",
            p=int(self.p),
            q=int(self.q),
            dist=self.distribution,
            rescale=False,
        )

        fitted = am.fit(disp="off")
        simulated_returns = self._simulate_paths(
            fitted,
            horizon_days=realized_horizon_days,
        )

        x_grid, pdf, cdf = self._make_density(simulated_returns)

        realized, end_date = self.realized_return(
            date=date,
            horizon_days=realized_horizon_days,
            market_data=market_data,
            config=config,
        )

        return ForecastDensity(
            date=date,
            model_name=self.name,
            horizon=int(horizon_days),
            x_grid=x_grid,
            pdf=pdf,
            cdf=cdf,
            realized=realized,
            metadata={
                "source": "garch",
                "p": int(self.p),
                "q": int(self.q),
                "distribution": self.distribution,
                "simulations": int(self.simulations),
                "fit_obs": int(len(sample)),
                "target_horizon_days": int(horizon_days),
                "realized_horizon_days": int(realized_horizon_days),
                "end_date": end_date,
                "calibration": calibration_metadata,
                **self.metadata,
            },
        )
