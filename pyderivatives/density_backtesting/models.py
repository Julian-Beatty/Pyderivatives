# models.py

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .base import ForecastDensity
from .config import EvaluationConfig
from .preprocessing import MarketData
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
        
        


        target_maturity = config.target_maturity_for_horizon(horizon_days)

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

        pdf = np.asarray(info["rnd_lr_surface"], dtype=float)[j, :]
        pdf = _normalize_pdf(x_grid, pdf)

        if pdf is None:
            raise ValueError("Invalid RND pdf.")

        cdf = self._get_cdf_surface(info)[j, :]

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
        target_maturity = config.target_maturity_for_horizon(horizon_days)

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

        pdf = _normalize_pdf(x_grid, surface[j, :])
        if pdf is None:
            raise ValueError("Invalid physical-density pdf.")

        cdf = None
        if "physical_cdf_lr_surface" in info:
            cdf_surface = np.asarray(info["physical_cdf_lr_surface"], dtype=float)
            if cdf_surface.ndim == 2 and j < cdf_surface.shape[0]:
                cdf = _clean_cdf(x_grid, cdf_surface[j, :])

        if cdf is None:
            cdf = _cdf_from_pdf(x_grid, pdf)

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
    rnd_key: str = "default"
    transform: Any = None

    requires_fit: bool = True
    clone_transform: bool = True
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)

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

    def _get_physical_density(self, physical: dict, j: int):
        x_grid = np.asarray(physical["grid_lr"], dtype=float)

        pdf = np.asarray(
            physical["physical_lr_surface"],
            dtype=float,
        )[j, :]

        pdf = _normalize_pdf(x_grid, pdf)

        if pdf is None:
            raise ValueError("Invalid physical pdf.")

        if "physical_cdf_lr_surface" in physical:
            cdf = np.asarray(
                physical["physical_cdf_lr_surface"],
                dtype=float,
            )[j, :]
        else:
            cdf = _cdf_from_pdf(x_grid, pdf)

        return x_grid, pdf, cdf

    def _fit_transform(
        self,
        *,
        index: int,
        dates: list[pd.Timestamp],
        market_data: MarketData,
        config: EvaluationConfig,
    ):
        if self.transform is None:
            raise ValueError("transform cannot be None.")

        model = (
            copy.deepcopy(self.transform)
            if self.clone_transform
            else self.transform
        )

        model.verbose = False

        if not self.requires_fit:
            return model

        fit_dates = _fit_dates(dates, index, config)

        if fit_dates is None:
            raise ValueError("Insufficient fit dates.")

        rnd_by_date = self._rnd_by_date(market_data)

        rnd_fit_dict = {
            d: rnd_by_date[d]
            for d in fit_dates
            if d in rnd_by_date and rnd_by_date[d].get("success", True)
        }

        if len(rnd_fit_dict) < config.min_fit_obs:
            raise ValueError("Insufficient successful RND fit observations.")

        stock_df = market_data.stock_df

        if stock_df is None:
            stock_df = market_data.returns.to_frame("ret_1").reset_index()
            stock_df = stock_df.rename(columns={stock_df.columns[0]: "date"})

        model.fit(
            rnd_fit_dict,
            stock_df,
            **self.fit_kwargs,
        )

        return model

    def forecast_one(
        self,
        *,
        date: pd.Timestamp,
        index: int,
        horizon_days: int,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> ForecastDensity:
       
        
        
        


        target_maturity = config.target_maturity_for_horizon(horizon_days)

        dates = self.evaluation_dates(market_data, config)

        model = self._fit_transform(
            index=index,
            dates=dates,
            market_data=market_data,
            config=config,
        )

        rnd_by_date = self._rnd_by_date(market_data)
        info = rnd_by_date[date]

        if hasattr(model, "transform_rnd"):
            physical = model.transform_rnd(info)
            physical.setdefault("day", info.get("day", {}))
        else:
            physical = model.transform_info(info)
        # Preserve observed maturity information for nearest_observed mode.
        physical.setdefault("day", info.get("day", {}))
            
        

        j, T_actual = _select_maturity_index(
            physical,
            target_maturity,
            config.maturity_match_tol,
            config=config,
        )
        
        fitted_params = _extract_fitted_params(model, T_actual)


        if j is None:
            raise ValueError("No transformed maturity within tolerance.")
        
        realized_horizon_days = int(round(365 * T_actual))
        
        realized, end_date = self.realized_return(
            date=date,
            horizon_days=realized_horizon_days,
            market_data=market_data,
            config=config,
        )
        
        x_grid, pdf, cdf = self._get_physical_density(physical, j)
        
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
                "maturity_selection": getattr(config, "maturity_selection", "target_grid"),
                "end_date": end_date,
                "fitted_params": fitted_params,
                **self.metadata,
            },
        )


@dataclass
class HistoricalKDEModel(DensityModel):
    grid_size: int = 500
    grid_pad: float = 0.25

    def evaluation_dates(
        self,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> list[pd.Timestamp]:
        return list(market_data.returns.index)

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
        if index < config.reserve_obs:
            raise ValueError("Forecast date is inside reserve period.")
    
        realized_horizon_days = int(
            getattr(config, "_override_realized_horizon_days", horizon_days)
        )
    
        trailing = market_data.return_series.horizon_return(
            realized_horizon_days,
            forward=False,
        )
    
        available = trailing[trailing.index < pd.Timestamp(date)].dropna()
    
        if config.window_type == "expanding":
            sample = available
        elif config.window_type == "rolling":
            if config.window_size is None:
                raise ValueError("window_size must be supplied for rolling windows.")
            sample = available.iloc[-int(config.window_size):]
        else:
            raise ValueError("window_type must be 'expanding' or 'rolling'.")
    
        if len(sample) < config.min_fit_obs:
            raise ValueError("Insufficient KDE fit dates.")
    
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

    def evaluation_dates(
        self,
        market_data: MarketData,
        config: EvaluationConfig,
    ) -> list[pd.Timestamp]:
        return list(market_data.returns.index)

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
        returns = market_data.returns.sort_index().dropna()
        available = returns[returns.index < pd.Timestamp(date)]

        if config.window_type == "expanding":
            sample = available

        elif config.window_type == "rolling":
            if config.window_size is None:
                raise ValueError("window_size must be supplied for rolling windows.")
            sample = available.iloc[-int(config.window_size):]

        else:
            raise ValueError("window_type must be 'expanding' or 'rolling'.")

        if len(sample) < config.min_fit_obs:
            raise ValueError("Insufficient GARCH fit observations.")

        return sample.astype(float)

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
            # Usually shape: n_origins x simulations x horizon
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
        if index < config.reserve_obs:
            raise ValueError("Forecast date is inside reserve period.")
    
        realized_horizon_days = int(
            getattr(config, "_override_realized_horizon_days", horizon_days)
        )
    
        sample = self._fit_sample(
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
                **self.metadata,
            },
        )