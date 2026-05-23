# logreturns=compute_horizon_returns_backward(
#     stock_df,
#     horizon=1,
#     group_col=None
# )
RND_dict_2013_2025 = slice_dict_by_date(
    USO_RND_dict_2008_2025,
    "2013-09-25",
    "2025-08-29"
)
sliced_df=RND_dict_2013_2025["2019-11-13"]

"""
First-pass implementation of a registry-based measure transformation module.

This file is written as a single self-contained draft so it is easy to inspect.
In the package, you would split it into:

MeasureTransform/
    __init__.py
    registry.py
    config.py
    utils.py
    base.py
    methods/exponential_polynomial.py

Core design:
    - Fit one model per maturity slice.
    - Apply fitted model to any out-of-sample RND info dictionary.
    - Optional fit-time trimming based on PIT/RND tail location.
    - Standardized result dictionary with physical density, pricing kernel, RRA.
    - Verbose progress after each maturity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, List
import copy
import os
import pickle
import hashlib
import json
from dataclasses import asdict, is_dataclass
import numpy as np
import pandas as pd

from scipy.optimize import minimize


# ============================================================
# Registry
# ============================================================

TRANSFORM_REGISTRY: Dict[str, type] = {}


def register_transform(name: str):
    """Register a measure transformation class."""
    def decorator(cls):
        key = str(name).strip().lower()
        if key in TRANSFORM_REGISTRY:
            raise ValueError(f"Transform method '{key}' is already registered.")
        TRANSFORM_REGISTRY[key] = cls
        cls.method_name = key
        return cls
    return decorator


def available_transforms() -> List[str]:
    return sorted(TRANSFORM_REGISTRY.keys())


def get_transform(name: str, **kwargs):
    key = str(name).strip().lower()
    if key not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform '{name}'. Available methods: {available_transforms()}"
        )
    return TRANSFORM_REGISTRY[key](**kwargs)


# ============================================================
# Config objects
# ============================================================

@dataclass(frozen=True)
class ThetaSpec:
    """Specification for exponential-polynomial pricing kernel."""
    N: int = 2
    Ksig: int = 1
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None


@dataclass(frozen=True)
class BootstrapSpec:
    enabled: bool = False
    B: int = 500
    block_length: int = 20
    ci_level: float = 0.95
    random_state: Optional[int] = 123
    keep_draws: bool = False


@dataclass(frozen=True)
class CacheSpec:
    """Optional disk cache for fitted models or transformed outputs."""
    enabled: bool = False
    folder: str = "measure_transform_cache"
    cache_fit: bool = True
    cache_transform: bool = False
    dataset_tag: str = "default"


@dataclass(frozen=True)
class KeySpec:
    x_grid_key: str = "rnd_lr_grid"
    pdf_surface_key: str = "rnd_lr_surface"
    cdf_surface_key: str = "cdf_surface"
    T_grid_key: str = "T_grid"
    spot_keys: Tuple[str, ...] = ("S0", "spot", "se")
    sigma_keys: Tuple[str, ...] = ("atm_vol", "sigma", "vol")


@dataclass
class FitDiagnostics:
    maturity: float
    method: str
    n_total: int
    n_used: int
    n_dropped: int
    loss: float = np.nan
    loss_name: str = "not_applicable"
    status: str = "unknown"
    message: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Cache utilities
# ============================================================

def _stable_for_hash(obj: Any) -> Any:
    if is_dataclass(obj):
        return _stable_for_hash(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _stable_for_hash(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_stable_for_hash(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return {
            "shape": obj.shape,
            "mean": float(np.nanmean(obj)) if obj.size else np.nan,
            "std": float(np.nanstd(obj)) if obj.size else np.nan,
            "min": float(np.nanmin(obj)) if obj.size else np.nan,
            "max": float(np.nanmax(obj)) if obj.size else np.nan,
        }
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


def _cache_key(payload: dict) -> str:
    s = json.dumps(_stable_for_hash(payload), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:24]


def _cache_load(folder: str, key: str):
    path = os.path.join(folder, f"{key}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _cache_save(folder: str, key: str, obj):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


# ============================================================
# Numerical utilities
# ============================================================

def _as_1d(x, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype).ravel()


def _trapz_normalize_density(x: np.ndarray, f: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    x = _as_1d(x)
    f = _as_1d(f)
    f = np.where(np.isfinite(f) & (f >= 0), f, 0.0)
    mass = float(np.trapezoid(f, x)) if x.size >= 2 else np.nan
    if not np.isfinite(mass) or mass <= eps:
        return np.full_like(f, np.nan, dtype=float)
    return f / mass


def _cdf_from_density(x: np.ndarray, f: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    x = _as_1d(x)
    f = _as_1d(f)
    if x.size != f.size or x.size < 2:
        return np.full_like(x, np.nan, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x grid must be strictly increasing.")
    f = np.where(np.isfinite(f) & (f >= 0), f, 0.0)
    dx = np.diff(x)
    area = 0.5 * (f[:-1] + f[1:]) * dx
    cdf = np.empty_like(x, dtype=float)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(area)
    total = float(cdf[-1])
    if not np.isfinite(total) or total <= eps:
        return np.full_like(x, np.nan, dtype=float)
    return cdf / total


def _safe_interp(x_new: float, x: np.ndarray, y: np.ndarray) -> float:
    x = _as_1d(x)
    y = _as_1d(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    xx = x[m]
    yy = y[m]
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    return float(np.interp(float(x_new), xx, yy, left=np.nan, right=np.nan))


def _find_spot(info: dict, spot_keys: Tuple[str, ...]) -> Optional[float]:
    for key in spot_keys:
        if key in info:
            val = info[key]
            try:
                val = float(val)
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    return None


def _find_sigma(info: dict, sigma_keys: Tuple[str, ...], default: float = 1.0) -> float:
    for key in sigma_keys:
        if key in info:
            val = info[key]
            try:
                val = float(val)
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    return float(default)


def _validate_fit_trim_alpha(fit_trim_alpha: Optional[Tuple[float, float]]):
    if fit_trim_alpha is None:
        return None
    a_left, a_right = map(float, fit_trim_alpha)
    if not (0.0 <= a_left < 1.0 and 0.0 <= a_right < 1.0):
        raise ValueError("fit_trim_alpha entries must be in [0, 1).")
    if a_left + a_right >= 1.0:
        raise ValueError("fit_trim_alpha must satisfy left + right < 1.")
    return a_left, a_right


def _block_indices_circular(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap indices."""
    if n <= 0:
        return np.array([], dtype=int)
    L = max(1, int(block_length))
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=n_blocks)
    idx = []
    for s in starts:
        idx.extend([(int(s) + k) % n for k in range(L)])
    return np.asarray(idx[:n], dtype=int)


# ============================================================
# Moment utilities
# ============================================================

def physical_moments_table(
    *,
    T_grid: np.ndarray,
    r_grid: np.ndarray,
    physical_lr_surface: np.ndarray,
) -> pd.DataFrame:
    """
    Compute moments and tail quantities from a log-return density surface.

    Inputs
    ------
    T_grid:
        Maturities, usually in years.
    r_grid:
        Log-return grid.
    physical_lr_surface:
        Density in log-return space with shape (len(T_grid), len(r_grid)).

    Returns
    -------
    DataFrame with mean, variance, volatility, annualized volatility,
    skewness, kurtosis, density area, and selected gross-return tail measures.
    """
    T = _as_1d(T_grid)
    r = _as_1d(r_grid)
    P = np.asarray(physical_lr_surface, dtype=float)

    if P.shape != (T.size, r.size):
        raise ValueError("physical_lr_surface must have shape (len(T_grid), len(r_grid)).")

    rows = []
    G = np.exp(r)

    for j, Tj in enumerate(T):
        f = np.asarray(P[j, :], dtype=float)
        m = np.isfinite(r) & np.isfinite(f) & (f >= 0)

        if m.sum() < 10:
            rows.append({"T": float(Tj), "area": np.nan})
            continue

        rr = r[m]
        ff = f[m]
        GG = G[m]

        area = float(np.trapezoid(ff, rr))
        if not np.isfinite(area) or area <= 0:
            rows.append({"T": float(Tj), "area": area})
            continue

        ff = ff / area
        mu = float(np.trapezoid(rr * ff, rr))
        var = float(np.trapezoid(((rr - mu) ** 2) * ff, rr))
        vol = float(np.sqrt(var)) if var >= 0 else np.nan
        vol_ann = float(vol / np.sqrt(float(Tj))) if np.isfinite(vol) and Tj > 0 else np.nan

        if np.isfinite(var) and var > 0:
            sd = np.sqrt(var)
            skew = float(np.trapezoid(((rr - mu) / sd) ** 3 * ff, rr))
            kurt = float(np.trapezoid(((rr - mu) / sd) ** 4 * ff, rr))
        else:
            skew = np.nan
            kurt = np.nan

        cdf = _cdf_from_density(rr, ff, eps=1e-14)
        if np.all(np.isfinite(cdf)):
            q025_r = float(np.interp(0.025, cdf, rr))
            q05_r = float(np.interp(0.05, cdf, rr))
            q95_r = float(np.interp(0.95, cdf, rr))
            q975_r = float(np.interp(0.975, cdf, rr))

            var95 = 1.0 - float(np.exp(q05_r))
            tailgain95 = float(np.exp(q95_r) - 1.0)

            left = rr <= q025_r
            right = rr >= q975_r

            left_mass = float(np.trapezoid(ff[left], rr[left])) if left.sum() >= 2 else np.nan
            right_mass = float(np.trapezoid(ff[right], rr[right])) if right.sum() >= 2 else np.nan

            if np.isfinite(left_mass) and left_mass > 0:
                cvar97_5 = float(np.trapezoid((1.0 - np.exp(rr[left])) * ff[left], rr[left]) / left_mass)
            else:
                cvar97_5 = np.nan

            if np.isfinite(right_mass) and right_mass > 0:
                ctg97_5 = float(np.trapezoid((np.exp(rr[right]) - 1.0) * ff[right], rr[right]) / right_mass)
            else:
                ctg97_5 = np.nan
        else:
            var95 = cvar97_5 = tailgain95 = ctg97_5 = np.nan

        rows.append(
            {
                "T": float(Tj),
                "mean_r": mu,
                "var_r": var,
                "vol_r": vol,
                "vol_ann_r": vol_ann,
                "skew_r": skew,
                "kurt_r": kurt,
                "area": area,
                "var95": var95,
                "cvar97_5": cvar97_5,
                "tailgain95": tailgain95,
                "ctg97_5": ctg97_5,
            }
        )

    return pd.DataFrame(rows)


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
                F_p_draws[b] = out_b["physical_lr_surface"]
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
            "physical_lr_surface_ci": ci(F_p_draws),
            "pricing_kernel_surface_ci": ci(kernel_draws),
            "relative_risk_aversion_surface_ci": ci(rra_draws),
        }
        if bootstrap_spec.keep_draws:
            boot_out["draws"] = {
                "physical_lr_surface": f_p_draws,
                "physical_lr_surface": F_p_draws,
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


# ============================================================
# Exponential-polynomial method
# ============================================================

def unpack_theta(theta: np.ndarray, N: int, Ksig: int) -> np.ndarray:
    theta = _as_1d(theta)
    return theta.reshape((Ksig + 1, N), order="F")


def c_it(sigma: float, theta_mat: np.ndarray) -> np.ndarray:
    Ksig = theta_mat.shape[0] - 1
    sig_pow = float(sigma) ** np.arange(Ksig + 1, dtype=float)
    return sig_pow @ theta_mat


def g_r_sigma(r: np.ndarray, sigma: float, theta: np.ndarray, *, N: int, Ksig: int) -> np.ndarray:
    theta_mat = unpack_theta(theta, N, Ksig)
    c = c_it(float(sigma), theta_mat)
    powers = np.vstack([_as_1d(r) ** i for i in range(1, N + 1)]).T
    return powers @ c


@dataclass
class ExponentialPolynomialFitted:
    theta_hat: np.ndarray
    theta_spec: ThetaSpec
    T: float
    loss: float
    success: bool
    message: str


@register_transform("exponential_polynomial")
class ExponentialPolynomialKernel(MeasureTransform):
    """
    Exponential-polynomial pricing-kernel transformation.

    Parameterization:
        M(r, sigma; theta) = exp(g(r, sigma; theta))

    Physical density:
        f_P(r) ∝ f_Q(r) / M(r, sigma; theta)
               = f_Q(r) exp(-g(r, sigma; theta))
    """

    def __init__(
        self,
        *,
        theta_spec: ThetaSpec = ThetaSpec(),
        maxiter: int = 400,
        x0: Optional[np.ndarray] = None,
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
        super().__init__(
            key_spec=key_spec,
            fit_trim_alpha=fit_trim_alpha,
            min_obs=min_obs,
            fit_maturities=fit_maturities,
            maturity_match_tol=maturity_match_tol,
            eps=eps,
            verbose=verbose,
            penalty_value=penalty_value,
            cache_spec=cache_spec,
        )
        self.theta_spec = theta_spec
        self.maxiter = int(maxiter)
        self.x0 = None if x0 is None else _as_1d(x0)

    def _cache_params(self) -> dict:
        return {"theta_spec": self.theta_spec, "maxiter": self.maxiter, "x0": self.x0}

    def _theta_dim(self) -> int:
        return int(self.theta_spec.N) * (int(self.theta_spec.Ksig) + 1)

    def _bounds(self):
        p = self._theta_dim()
        if self.theta_spec.bounds is None:
            return None
        lb, ub = self.theta_spec.bounds
        lb = _as_1d(lb)
        ub = _as_1d(ub)
        if lb.size != p or ub.size != p:
            raise ValueError("ThetaSpec.bounds must have lower and upper arrays of length N*(Ksig+1).")
        return list(zip(lb, ub))

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float) -> Tuple[ExponentialPolynomialFitted, Dict[str, Any]]:
        p = self._theta_dim()
        x0 = np.zeros(p, dtype=float) if self.x0 is None else self.x0.copy()
        if x0.size != p:
            raise ValueError(f"x0 has length {x0.size}, expected {p}.")

        bounds = self._bounds()
        N = int(self.theta_spec.N)
        Ksig = int(self.theta_spec.Ksig)
        eps = self.eps

        def nll(theta):
            theta = _as_1d(theta)
            total_ll = 0.0
            penalty = self.penalty_value

            if theta.size != p or not np.all(np.isfinite(theta)):
                return penalty

            for _, row in hist_T.iterrows():
                x_grid = _as_1d(row["x_grid"])

                # Invalid grid means this parameter vector/observation pair is unusable.
                if x_grid.size < 10 or np.any(~np.isfinite(x_grid)) or np.any(np.diff(x_grid) <= 0):
                    return penalty

                f_q = _trapz_normalize_density(x_grid, row["f_q"], eps=eps)
                if f_q.size != x_grid.size or not np.all(np.isfinite(f_q)):
                    return penalty

                sigma = float(row.get("sigma", 1.0))
                realized = float(row["realized_return"])

                if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(realized):
                    return penalty

                # If the realized return is outside the evaluation grid, the likelihood
                # contribution is not trustworthy. Penalize instead of allowing nan/zero.
                if realized < x_grid[0] or realized > x_grid[-1]:
                    return penalty

                g = g_r_sigma(x_grid, sigma, theta, N=N, Ksig=Ksig)
                if g.size != x_grid.size or not np.all(np.isfinite(g)):
                    return penalty

                # exp(-g) enters f_P. Clip to prevent overflow/underflow.
                weight = np.exp(np.clip(-g, -700, 700))
                raw = f_q * weight

                mass = float(np.trapezoid(raw, x_grid))
                if not np.isfinite(mass) or mass <= eps:
                    return penalty

                f_p = raw / mass
                if f_p.size != x_grid.size or not np.all(np.isfinite(f_p)):
                    return penalty

                f_at_realized = _safe_interp(realized, x_grid, f_p)
                if not np.isfinite(f_at_realized) or f_at_realized <= eps:
                    return penalty

                total_ll += np.log(f_at_realized)

            if not np.isfinite(total_ll):
                return penalty

            out = float(-total_ll)
            return out if np.isfinite(out) else penalty

        res = minimize(
            nll,
            x0,
            method="L-BFGS-B" if bounds is not None else "BFGS",
            bounds=bounds,
            options={"maxiter": self.maxiter},
        )

        fitted = ExponentialPolynomialFitted(
            theta_hat=np.asarray(res.x, dtype=float),
            theta_spec=self.theta_spec,
            T=float(T),
            loss=float(res.fun),
            success=bool(res.success),
            message=str(res.message),
        )

        diag = {
            "loss": float(res.fun),
            "loss_name": "negative_log_likelihood",
            "status": "success" if res.success else "failed",
            "message": str(res.message),
            "params": {f"theta_{i}": float(v) for i, v in enumerate(res.x)},
        }
        return fitted, diag

    def _transform_surface_with_model(
        self,
        fitted_model: ExponentialPolynomialFitted,
        x_grid: np.ndarray,
        f_q: np.ndarray,
        F_q: np.ndarray,
        *,
        T: float,
        info: dict,
    ) -> dict:
        x_grid = _as_1d(x_grid)
        f_q = _trapz_normalize_density(x_grid, f_q, eps=self.eps)
        F_q = _as_1d(F_q)

        sigma = _find_sigma(info, self.key_spec.sigma_keys, default=1.0)
        theta = fitted_model.theta_hat
        N = int(fitted_model.theta_spec.N)
        Ksig = int(fitted_model.theta_spec.Ksig)

        g = g_r_sigma(x_grid, sigma, theta, N=N, Ksig=Ksig)

        # Kernel is proportional to exp(g). Normalize so E_P[M] = 1 on the grid.
        raw_kernel = np.exp(np.clip(g, -700, 700))

        # Physical density is f_Q / M, normalized.
        raw_f_p = f_q / np.maximum(raw_kernel, self.eps)
        f_p = _trapz_normalize_density(x_grid, raw_f_p, eps=self.eps)
        F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)

        # dP/dQ
        weight = f_p / np.maximum(f_q, self.eps)

        # normalize M so ∫ M f_P dx = 1
        M = f_q / np.maximum(f_p, self.eps)
        Em = float(np.trapezoid(M * f_p, x_grid))
        if np.isfinite(Em) and Em > self.eps:
            M = M / Em

        return {
            "x_grid": x_grid,
            "f_q": f_q,
            "F_q": F_q,
            "f_p": f_p,
            "F_p": F_p,
            "weight": weight,
            "pricing_kernel": M,
            "g": g,
            "theta_hat": theta,
            "T_fit": float(T),
        }


# ============================================================
# Plotting utilities
# ============================================================

def _slice_surface_bounds(
    T_grid: np.ndarray,
    x_grid: np.ndarray,
    Z: np.ndarray,
    *,
    T_bounds: Optional[Tuple[float, float]] = None,
    x_bounds: Optional[Tuple[float, float]] = None,
):
    T_grid = _as_1d(T_grid)
    x_grid = _as_1d(x_grid)
    Z = np.asarray(Z, dtype=float)

    if Z.shape != (T_grid.size, x_grid.size):
        raise ValueError(
            f"Surface shape mismatch: Z {Z.shape}, expected {(T_grid.size, x_grid.size)}."
        )

    tmask = np.isfinite(T_grid)
    xmask = np.isfinite(x_grid)

    if T_bounds is not None:
        lo, hi = sorted(map(float, T_bounds))
        tmask &= (T_grid >= lo) & (T_grid <= hi)

    if x_bounds is not None:
        lo, hi = sorted(map(float, x_bounds))
        xmask &= (x_grid >= lo) & (x_grid <= hi)

    if not np.any(tmask):
        raise ValueError("T_bounds produced an empty maturity range.")
    if not np.any(xmask):
        raise ValueError("x_bounds produced an empty x-axis range.")

    return T_grid[tmask], x_grid[xmask], Z[np.ix_(tmask, xmask)]


def _get_plot_x_grid(result: dict, x_axis: str = "R") -> Tuple[np.ndarray, str]:
    """
    Choose plotting x-axis from standardized result dictionary.

    x_axis:
        "r"      -> log return
        "R"      -> gross return
        "return" -> simple return
        "K"      -> terminal price / strike grid
    """
    x_axis = str(x_axis).strip()

    if x_axis in {"r", "log", "log_return", "lr"}:
        return _as_1d(result["grid_lr"]), "Log return $r$"

    if x_axis in {"R", "gross", "gross_return"}:
        return _as_1d(result["R_grid"]), "Gross return $R=S_T/S_0$"

    if x_axis in {"return", "simple", "simple_return"}:
        return _as_1d(result["simple_return_grid"]), "Simple return $R-1$"

    if x_axis in {"K", "strike", "terminal", "terminal_price", "ST"}:
        ST = result.get("grid_k", None)
        if ST is None:
            raise ValueError("result['grid_k'] is unavailable because S0 was not stored/found.")
        return _as_1d(ST), "Terminal price $S_T$"

    raise ValueError("x_axis must be one of {'r', 'R', 'return', 'K'}.")


def P_Q_K_multipanel(
    out: dict,
    *,
    title: Optional[str] = None,
    n_panels: Optional[int] = None,
    panel_shape: Tuple[int, int] = (2, 4),
    save: Optional[str] = None,
    dpi: int = 200,
    x_space: str = "gross",
    truncate: bool = True,
    ptail_alpha: Tuple[float, float] = (0.1, 0.0),
    trunc_mode: str = "cdf",
    r_bounds: Optional[Tuple[float, float]] = None,
    clip_trunc_to_support: bool = True,
    kernel_color: str = "black",
    kernel_linestyle: str = "--",
    kernel_yscale: str = "linear",
    kernel_log_eps: float = 1e-300,
    legend_loc: str = "upper center",
):
    import matplotlib.pyplot as plt
    from pathlib import Path

    r = _as_1d(out["grid_lr"])
    R = _as_1d(out["grid_r"])
    T_grid = _as_1d(out["T_grid"])

    q_lr = np.asarray(out["rnd_lr_surface"], dtype=float)
    p_lr = np.asarray(out["physical_lr_surface"], dtype=float)
    M = np.asarray(out["pricing_kernel_surface"], dtype=float)

    if q_lr.shape != (T_grid.size, r.size):
        raise ValueError("rnd_lr_surface must have shape (len(T_grid), len(grid_lr)).")
    if p_lr.shape != q_lr.shape or M.shape != q_lr.shape:
        raise ValueError("physical_lr_surface and pricing_kernel_surface must match rnd_lr_surface.")

    x_space = str(x_space).lower().strip()

    if x_space in {"gross", "r", "return", "returns"}:
        x = R
        q = q_lr / np.maximum(R[None, :], 1e-300)
        p = p_lr / np.maximum(R[None, :], 1e-300)
        xlabel = "Gross return $R=S_T/S_0$"
        ref_x = 1.0

    elif x_space in {"log", "lr", "logreturn", "log_return"}:
        x = r
        q = q_lr
        p = p_lr
        xlabel = r"Log return $r=\log(S_T/S_0)$"
        ref_x = 0.0

    else:
        raise ValueError("x_space must be either 'gross' or 'log'.")

    trunc_mode = str(trunc_mode).lower().strip()
    if trunc_mode not in {"cdf", "rbounds", "cdf+rbounds", "none"}:
        raise ValueError("trunc_mode must be one of {'cdf', 'rbounds', 'cdf+rbounds', 'none'}.")

    nT = T_grid.size
    nrows, ncols = panel_shape
    max_panels = int(nrows * ncols)

    if n_panels is None:
        n_panels = min(max_panels, nT)
    else:
        n_panels = min(int(n_panels), max_panels, nT)

    if n_panels <= 0:
        raise ValueError("n_panels must be positive.")

    j_indices = np.unique(np.round(np.linspace(0, nT - 1, n_panels)).astype(int))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.1 * nrows),
        constrained_layout=True,
    )

    axes = np.asarray(axes).ravel()

    for ax, j in zip(axes, j_indices):
        xj = x.copy()
        qj = q[j, :].copy()
        pj = p[j, :].copy()
        Mj = M[j, :].copy()

        base_mask = (
            np.isfinite(xj)
            & np.isfinite(qj)
            & np.isfinite(pj)
            & np.isfinite(Mj)
        )

        if truncate and r_bounds is not None and trunc_mode in {"rbounds", "cdf+rbounds"}:
            rlo, rhi = sorted(map(float, r_bounds))
            base_mask &= (r >= rlo) & (r <= rhi)

        x_qp = xj[base_mask]
        q_plot = qj[base_mask]
        p_plot = pj[base_mask]

        ax.plot(x_qp, q_plot, linewidth=2.0, label="$q$ / RND")
        ax.plot(x_qp, p_plot, linewidth=2.0, linestyle=":", label="$p$ / physical")
        ax.axvline(ref_x, linewidth=1.0, alpha=0.4)

        ax.set_title(f"T ≈ {365 * T_grid[j]:.0f}d")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.2)

        k_mask = base_mask.copy()

        if truncate and trunc_mode in {"cdf", "cdf+rbounds"}:
            a_left, a_right = map(float, ptail_alpha)

            if not (
                0.0 <= a_left < 1.0
                and 0.0 <= a_right < 1.0
                and a_left + a_right < 1.0
            ):
                raise ValueError(
                    "ptail_alpha must satisfy 0 <= left,right < 1 and left+right < 1."
                )

            if "physical_cdf_lr_surface" in out:
                cdf_p = np.asarray(out["physical_cdf_lr_surface"], dtype=float)[j, :]
            else:
                cdf_p = _cdf_from_density(r, p_lr[j, :], eps=1e-14)

            if np.all(np.isfinite(cdf_p)):
                keep_cdf = (cdf_p >= a_left) & (cdf_p <= 1.0 - a_right)
                k_mask &= keep_cdf
            elif not clip_trunc_to_support:
                raise ValueError("Could not compute physical CDF for kernel truncation.")

        x_k = xj[k_mask]
        M_plot = Mj[k_mask]

        if kernel_yscale == "log":
            M_plot = np.maximum(M_plot, float(kernel_log_eps))

        ax2 = ax.twinx()
        ax2.plot(
            x_k,
            M_plot,
            color=kernel_color,
            linestyle=kernel_linestyle,
            linewidth=2.0,
            label="$M$ / kernel",
        )

        ax2.set_ylabel("Pricing kernel $M$")

        if kernel_yscale in {"linear", "log"}:
            ax2.set_yscale(kernel_yscale)
        else:
            raise ValueError("kernel_yscale must be 'linear' or 'log'.")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc=legend_loc, fontsize=8)

    for ax in axes[len(j_indices):]:
        ax.set_visible(False)

    fig.suptitle(title or "Risk-neutral density, physical density, and pricing kernel")

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {save_path}")

    plt.show()
    return fig


def plot_pricing_kernel_surface(
    result: dict,
    *,
    x_axis: str = "R",
    T_bounds: Optional[Tuple[float, float]] = None,
    x_bounds: Optional[Tuple[float, float]] = None,
    title: str = "Pricing Kernel Surface",
    cmap: str = "viridis",
    elev: float = 28,
    azim: float = -60,
    alpha: float = 0.9,
    log_z: bool = False,
    save: Optional[str] = None,
    dpi: int = 200,
    show: bool = True,
):
    """Plot the pricing-kernel surface M(x,T)."""
    import matplotlib.pyplot as plt

    T_grid = _as_1d(result["T_grid"])
    x_grid, xlabel = _get_plot_x_grid(result, x_axis=x_axis)
    M = np.asarray(result["pricing_kernel_surface"], dtype=float)

    T_plot, x_plot, M_plot = _slice_surface_bounds(
        T_grid,
        x_grid,
        M,
        T_bounds=T_bounds,
        x_bounds=x_bounds,
    )

    if log_z:
        M_plot = np.log(np.maximum(M_plot, 1e-12))
        zlabel = "$\log M$"
    else:
        zlabel = "$M$"

    X, T = np.meshgrid(x_plot, T_plot)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        T,
        M_plot,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=alpha,
    )
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label=zlabel)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maturity $T$ (years)")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, surf


def plot_pricing_kernel_surface(
    result: dict,
    *,
    x_axis: str = "R",
    T_bounds: Optional[Tuple[float, float]] = None,
    x_bounds: Optional[Tuple[float, float]] = None,
    title: str = "Pricing Kernel Surface",
    cmap: str = "viridis",
    elev: float = 28,
    azim: float = -60,
    alpha: float = 0.9,
    log_z: bool = False,
    truncate_by_alpha: bool = False,
    ptail_alpha: Tuple[float, float] = (0.05, 0.05),
    truncation_measure: str = "physical",
    min_points_per_row: int = 5,
    save: Optional[str] = None,
    dpi: int = 200,
    show: bool = True,
):
    """
    Plot the pricing-kernel surface M(x,T), with optional alpha-tail truncation.

    x_axis:
        "r"      -> log return
        "R"      -> gross return
        "return" -> simple return
        "K"      -> terminal price / strike

    truncate_by_alpha:
        If True, hides kernel values outside the CDF interval
        [ptail_alpha[0], 1 - ptail_alpha[1]].

    truncation_measure:
        "physical"     -> truncate using physical CDF F_p
        "risk_neutral" -> truncate using risk-neutral CDF F_q
    """
    import matplotlib.pyplot as plt

    T_grid = _as_1d(result["T_grid"])
    x_grid, xlabel = _get_plot_x_grid(result, x_axis=x_axis)
    r_grid = _as_1d(result["grid_lr"])
    M = np.asarray(result["pricing_kernel_surface"], dtype=float).copy()

    if M.shape != (T_grid.size, x_grid.size):
        raise ValueError("pricing_kernel_surface must have shape (len(T_grid), len(x_grid)).")

    if truncate_by_alpha:
        a_left, a_right = map(float, ptail_alpha)
        if not (0.0 <= a_left < 1.0 and 0.0 <= a_right < 1.0 and a_left + a_right < 1.0):
            raise ValueError("ptail_alpha must satisfy 0 <= left,right < 1 and left+right < 1.")

        truncation_measure = str(truncation_measure).lower().strip()
        if truncation_measure in {"physical", "p", "fp"}:
            F_key = "physical_lr_surface"
            f_key = "physical_lr_surface"
            measure_label = "physical"
        elif truncation_measure in {"risk_neutral", "risk-neutral", "q", "fq"}:
            F_key = "rnd_lr_surface"
            f_key = "rnd_lr_surface"
            measure_label = "risk-neutral"
        else:
            raise ValueError("truncation_measure must be 'physical' or 'risk_neutral'.")

        if F_key in result:
            F_trunc = np.asarray(result[F_key], dtype=float)
        elif f_key in result:
            f_trunc = np.asarray(result[f_key], dtype=float)
            F_trunc = np.vstack([
                _cdf_from_density(r_grid, f_trunc[j, :], eps=1e-14)
                for j in range(f_trunc.shape[0])
            ])
        else:
            raise KeyError(f"Need result['{F_key}'] or result['{f_key}'] for alpha truncation.")

        if F_trunc.shape != M.shape:
            raise ValueError(f"{F_key} must have the same shape as pricing_kernel_surface.")

        keep = (F_trunc >= a_left) & (F_trunc <= 1.0 - a_right)

        for j in range(M.shape[0]):
            if int(np.sum(keep[j, :])) >= int(min_points_per_row):
                M[j, ~keep[j, :]] = np.nan
    else:
        measure_label = None

    T_plot, x_plot, M_plot = _slice_surface_bounds(
        T_grid,
        x_grid,
        M,
        T_bounds=T_bounds,
        x_bounds=x_bounds,
    )

    if T_plot.size < 2 or x_plot.size < 2:
        raise ValueError("Not enough points to plot after bounds/truncation.")

    if log_z:
        M_plot = np.where(np.isfinite(M_plot) & (M_plot > 0), np.log(M_plot), np.nan)
        zlabel = "$\log M$"
    else:
        zlabel = "$M$"

    X, T = np.meshgrid(x_plot, T_plot)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        T,
        M_plot,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=alpha,
    )
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label=zlabel)

    if truncate_by_alpha:
        keep_pct = 100.0 * (1.0 - float(ptail_alpha[0]) - float(ptail_alpha[1]))
        title = f"{title} — inner {keep_pct:.0f}% by {measure_label} CDF"

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maturity $T$ (years)")
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, surf


def plot_kernel_density_overlay(
    result: dict,
    *,
    maturity_index: Optional[int] = None,
    maturity: Optional[float] = None,
    x_axis: str = "R",
    x_bounds: Optional[Tuple[float, float]] = None,
    density_scale: str = "separate_axis",
    title: Optional[str] = None,
    log_kernel: bool = False,
    save: Optional[str] = None,
    dpi: int = 200,
    show: bool = True,
):
    """
    Overlay pricing kernel, risk-neutral density, and physical density for
    one maturity slice.

    density_scale:
        "separate_axis" -> M on left axis, densities on right axis
        "normalize"     -> normalize all three curves to max=1 and use one axis
    """
    import matplotlib.pyplot as plt

    T_grid = _as_1d(result["T_grid"])
    x_grid, xlabel = _get_plot_x_grid(result, x_axis=x_axis)

    if maturity_index is None:
        if maturity is None:
            maturity_index = 0
        else:
            maturity_index = int(np.argmin(np.abs(T_grid - float(maturity))))
    else:
        maturity_index = int(maturity_index)

    if not (0 <= maturity_index < T_grid.size):
        raise ValueError("maturity_index is out of bounds.")

    j = maturity_index
    T_val = float(T_grid[j])

    M = _as_1d(result["pricing_kernel_surface"][j, :])
    f_q = _as_1d(result["rnd_lr_surface"][j, :])
    f_p = _as_1d(result["physical_lr_surface"][j, :])

    mask = np.isfinite(x_grid) & np.isfinite(M) & np.isfinite(f_q) & np.isfinite(f_p)
    if x_bounds is not None:
        lo, hi = sorted(map(float, x_bounds))
        mask &= (x_grid >= lo) & (x_grid <= hi)

    x = x_grid[mask]
    M = M[mask]
    f_q = f_q[mask]
    f_p = f_p[mask]

    if log_kernel:
        M_plot = np.log(np.maximum(M, 1e-12))
        M_label = "$\log M$"
    else:
        M_plot = M
        M_label = "$M$"

    fig, ax1 = plt.subplots(figsize=(9, 5))

    if density_scale == "normalize":
        def norm(y):
            y = np.asarray(y, dtype=float)
            ymax = np.nanmax(np.abs(y))
            return y / ymax if np.isfinite(ymax) and ymax > 0 else y

        ax1.plot(x, norm(M_plot), linewidth=2.0, label=M_label)
        ax1.plot(x, norm(f_q), linewidth=2.0, linestyle="--", label="$f_Q$")
        ax1.plot(x, norm(f_p), linewidth=2.0, linestyle=":", label="$f_P$")
        ax1.set_ylabel("Normalized value")
        ax1.legend(loc="best")

    elif density_scale == "separate_axis":
        line1, = ax1.plot(x, M_plot, linewidth=2.0, label=M_label)
        ax1.set_ylabel(M_label)

        ax2 = ax1.twinx()
        line2, = ax2.plot(x, f_q, linewidth=2.0, linestyle="--", label="$f_Q$")
        line3, = ax2.plot(x, f_p, linewidth=2.0, linestyle=":", label="$f_P$")
        ax2.set_ylabel("Density")

        lines = [line1, line2, line3]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")

    else:
        raise ValueError("density_scale must be 'separate_axis' or 'normalize'.")

    ax1.set_xlabel(xlabel)
    ax1.set_title(title or f"Pricing Kernel vs Risk-Neutral and Physical Densities — T ≈ {365*T_val:.0f}d")
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ============================================================
# Nonparametric calibration method
# ============================================================

@dataclass(frozen=True)
class NonparametricCalibrationSpec:
    """
    Nonparametric calibration specification.

    Method:
        1. Compute PIT values u_t = F_Q,t(r_t^realized).
        2. Transform z_t = Phi^{-1}(u_t).
        3. Estimate h(z) nonparametrically using Gaussian KDE.
        4. Define C(u) = H(Phi^{-1}(u)).

    Density transformation:

        f_P(x) = f_Q(x) * h(z) / phi(z),

    where:

        z = Phi^{-1}(F_Q(x)).

    bandwidth:
        "silverman" -> 1.06 * std(z) * n^(-1/5)
        "scott"     -> std(z) * n^(-1/5)
        float        -> manual bandwidth in z-space
    """
    bandwidth: str | float = "silverman"
    z_grid_size: int = 1000
    z_grid_pad: float = 1.0
    min_bandwidth: float = 1e-3
    max_bandwidth: Optional[float] = None


@dataclass
class NonparametricCalibrationFitted:
    z_obs: np.ndarray
    z_grid: np.ndarray
    h_grid: np.ndarray
    H_grid: np.ndarray
    bandwidth: float
    bandwidth_method: str
    T: float
    loss: float
    success: bool
    message: str


@register_transform("nonparametric")
@register_transform("nonparametric_calibration")
@register_transform("np_calibration")
class NonparametricCalibration(MeasureTransform):
    """
    Nonparametric PIT calibration using Gaussian KDE in normal-score space.

    This is the nonparametric analogue of beta calibration:

        F_P(x) = C(F_Q(x))

    but C is estimated from transformed PITs rather than forced to be beta.
    """

    def __init__(
        self,
        *,
        spec: NonparametricCalibrationSpec = NonparametricCalibrationSpec(),
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
        super().__init__(
            key_spec=key_spec,
            fit_trim_alpha=fit_trim_alpha,
            min_obs=min_obs,
            fit_maturities=fit_maturities,
            maturity_match_tol=maturity_match_tol,
            eps=eps,
            verbose=verbose,
            penalty_value=penalty_value,
            cache_spec=cache_spec,
        )
        self.spec = spec

    def _normal_pdf(self, z):
        z = np.asarray(z, dtype=float)
        return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

    def _normal_ppf(self, u):
        from scipy.stats import norm
        return norm.ppf(u)

    def _select_bandwidth(self, z: np.ndarray) -> Tuple[float, str]:
        z = _as_1d(z)
        n = z.size
        if n < 2:
            return float(self.spec.min_bandwidth), "fallback_min_bandwidth"

        bw = self.spec.bandwidth

        if isinstance(bw, str):
            method = bw.lower().strip()
            sd = float(np.std(z, ddof=1))
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0

            if method == "silverman":
                h = 1.06 * sd * (n ** (-1.0 / 5.0))
            elif method == "scott":
                h = sd * (n ** (-1.0 / 5.0))
            else:
                raise ValueError("bandwidth must be 'silverman', 'scott', or a positive float.")
        else:
            h = float(bw)
            method = "manual"

        if not np.isfinite(h) or h <= 0:
            h = float(self.spec.min_bandwidth)

        h = max(h, float(self.spec.min_bandwidth))
        if self.spec.max_bandwidth is not None:
            h = min(h, float(self.spec.max_bandwidth))

        return float(h), method

    def _kde_gaussian_on_grid(self, z_obs: np.ndarray, z_grid: np.ndarray, h: float) -> np.ndarray:
        z_obs = _as_1d(z_obs)
        z_grid = _as_1d(z_grid)
        h = float(h)

        # Matrix form is fine for moderate n/grid sizes. If n becomes huge,
        # this can be replaced by FFT/binning later.
        U = (z_grid[:, None] - z_obs[None, :]) / h
        h_grid = np.mean(np.exp(-0.5 * U**2) / np.sqrt(2.0 * np.pi), axis=1) / h
        h_grid = np.where(np.isfinite(h_grid) & (h_grid >= 0), h_grid, 0.0)
        return h_grid

    def _cache_params(self) -> dict:
        return {"spec": self.spec}

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float):
        u = np.asarray(hist_T["pit"], dtype=float)
        u = np.clip(u[np.isfinite(u)], self.eps, 1.0 - self.eps)

        if u.size < self.min_obs:
            fitted = NonparametricCalibrationFitted(
                z_obs=np.array([], dtype=float),
                z_grid=np.array([], dtype=float),
                h_grid=np.array([], dtype=float),
                H_grid=np.array([], dtype=float),
                bandwidth=np.nan,
                bandwidth_method="none",
                T=float(T),
                loss=np.nan,
                success=False,
                message="Too few valid PIT observations.",
            )
            diag = {
                "loss": np.nan,
                "loss_name": "negative_log_kde_likelihood",
                "status": "failed",
                "message": "Too few valid PIT observations.",
                "params": {"bandwidth": np.nan, "bandwidth_method": "none"},
            }
            return fitted, diag

        z_obs = self._normal_ppf(u)
        z_obs = z_obs[np.isfinite(z_obs)]

        if z_obs.size < self.min_obs:
            raise RuntimeError("Too few finite normal-score PIT observations.")

        h, method = self._select_bandwidth(z_obs)

        z_min = float(np.min(z_obs) - self.spec.z_grid_pad)
        z_max = float(np.max(z_obs) + self.spec.z_grid_pad)

        # Always include a reasonable normal-score region, because transformation
        # may be evaluated outside the observed z range.
        z_min = min(z_min, -5.0)
        z_max = max(z_max, 5.0)

        z_grid = np.linspace(z_min, z_max, int(self.spec.z_grid_size))
        h_grid = self._kde_gaussian_on_grid(z_obs, z_grid, h)
        h_grid = _trapz_normalize_density(z_grid, h_grid, eps=self.eps)
        H_grid = _cdf_from_density(z_grid, h_grid, eps=self.eps)

        # Pseudo likelihood: leave-in KDE likelihood on z_obs.
        h_at_obs = np.interp(z_obs, z_grid, h_grid, left=self.eps, right=self.eps)
        h_at_obs = np.maximum(h_at_obs, self.eps)
        loss = -float(np.sum(np.log(h_at_obs)))

        fitted = NonparametricCalibrationFitted(
            z_obs=z_obs,
            z_grid=z_grid,
            h_grid=h_grid,
            H_grid=H_grid,
            bandwidth=float(h),
            bandwidth_method=method,
            T=float(T),
            loss=loss,
            success=True,
            message="success",
        )

        diag = {
            "loss": loss,
            "loss_name": "negative_log_kde_likelihood",
            "status": "success",
            "message": "success",
            "params": {
                "bandwidth": float(h),
                "bandwidth_method": method,
                "n_z": int(z_obs.size),
                "z_min": float(z_min),
                "z_max": float(z_max),
            },
        }
        return fitted, diag

    def _transform_surface_with_model(
        self,
        fitted_model: NonparametricCalibrationFitted,
        x_grid: np.ndarray,
        f_q: np.ndarray,
        F_q: np.ndarray,
        *,
        T: float,
        info: dict,
    ) -> dict:
        x_grid = _as_1d(x_grid)
        f_q = _trapz_normalize_density(x_grid, f_q, eps=self.eps)
        F_q = _as_1d(F_q)

        if fitted_model.z_grid.size < 2:
            raise RuntimeError("Nonparametric calibration model is not valid for this maturity.")

        u = np.clip(F_q, self.eps, 1.0 - self.eps)
        z = self._normal_ppf(u)

        h_z = np.interp(
            z,
            fitted_model.z_grid,
            fitted_model.h_grid,
            left=self.eps,
            right=self.eps,
        )
        H_z = np.interp(
            z,
            fitted_model.z_grid,
            fitted_model.H_grid,
            left=0.0,
            right=1.0,
        )

        phi_z = self._normal_pdf(z)
        ratio = h_z / np.maximum(phi_z, self.eps)
        ratio = np.where(np.isfinite(ratio) & (ratio >= 0), ratio, 0.0)

        f_p_raw = f_q * ratio
        f_p = _trapz_normalize_density(x_grid, f_p_raw, eps=self.eps)

        # Use numerical CDF for consistency with normalized f_p.
        F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)

        weight = f_p / np.maximum(f_q, self.eps)

        M = f_q / np.maximum(f_p, self.eps)
        Em = float(np.trapezoid(M * f_p, x_grid))
        if np.isfinite(Em) and Em > self.eps:
            M = M / Em

        return {
            "x_grid": x_grid,
            "f_q": f_q,
            "F_q": F_q,
            "f_p": f_p,
            "F_p": F_p,
            "weight": weight,
            "pricing_kernel": M,
            "z": z,
            "h_z": h_z,
            "H_z": H_z,
            "bandwidth": fitted_model.bandwidth,
            "bandwidth_method": fitted_model.bandwidth_method,
            "T_fit": float(T),
        }


# ============================================================
# Beta calibration method
# ============================================================

@dataclass(frozen=True)
class BetaCalibrationSpec:
    """
    Beta calibration specification.

    Calibration form:

        F_P(x) = BetaCDF(F_Q(x); a, b)

        f_P(x) = BetaPDF(F_Q(x); a, b) f_Q(x)

    Estimation uses PIT values:

        u_t = F_Q,t(r_t^realized)

    and maximizes the beta likelihood of u_t.
    """
    a_bounds: Tuple[float, float] = (1e-4, 100.0)
    b_bounds: Tuple[float, float] = (1e-4, 100.0)
    x0: Tuple[float, float] = (1.0, 1.0)


@dataclass
class BetaCalibrationFitted:
    a: float
    b: float
    T: float
    loss: float
    success: bool
    message: str


@register_transform("beta_calibration")
@register_transform("beta")
class BetaCalibration(MeasureTransform):
    """
    Beta calibration transformation.

    This estimates one beta calibration function per maturity slice.

    For each maturity T:

        u_t = F_Q,t(r_t^realized)

    Then estimate:

        u_t ~ Beta(a_T, b_T)

    Transformation:

        F_P(x) = B(F_Q(x); a_T, b_T)
        f_P(x) = b(F_Q(x); a_T, b_T) f_Q(x)

    The pricing kernel is recovered as:

        M(x) ∝ f_Q(x) / f_P(x).
    """

    def __init__(
        self,
        *,
        spec: BetaCalibrationSpec = BetaCalibrationSpec(),
        maxiter: int = 400,
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
        super().__init__(
            key_spec=key_spec,
            fit_trim_alpha=fit_trim_alpha,
            min_obs=min_obs,
            fit_maturities=fit_maturities,
            maturity_match_tol=maturity_match_tol,
            eps=eps,
            verbose=verbose,
            penalty_value=penalty_value,
            cache_spec=cache_spec,
        )
        self.spec = spec
        self.maxiter = int(maxiter)

    def _cache_params(self) -> dict:
        return {"spec": self.spec, "maxiter": self.maxiter}

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float):
        from scipy.stats import beta as beta_dist

        u = np.asarray(hist_T["pit"], dtype=float)
        u = np.clip(u[np.isfinite(u)], self.eps, 1.0 - self.eps)

        penalty = self.penalty_value

        if u.size < self.min_obs:
            fitted = BetaCalibrationFitted(
                a=np.nan,
                b=np.nan,
                T=float(T),
                loss=np.nan,
                success=False,
                message="Too few valid PIT observations.",
            )
            diag = {
                "loss": np.nan,
                "loss_name": "negative_log_likelihood",
                "status": "failed",
                "message": "Too few valid PIT observations.",
                "params": {"a": np.nan, "b": np.nan},
            }
            return fitted, diag

        def nll(params):
            a, b = map(float, params)
            if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
                return penalty
            ll = beta_dist.logpdf(u, a, b)
            if not np.all(np.isfinite(ll)):
                return penalty
            out = -float(np.sum(ll))
            return out if np.isfinite(out) else penalty

        bounds = [self.spec.a_bounds, self.spec.b_bounds]
        x0 = np.asarray(self.spec.x0, dtype=float)
        x0[0] = np.clip(x0[0], bounds[0][0], bounds[0][1])
        x0[1] = np.clip(x0[1], bounds[1][0], bounds[1][1])

        res = minimize(
            nll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter},
        )

        a_hat, b_hat = map(float, res.x)

        fitted = BetaCalibrationFitted(
            a=a_hat,
            b=b_hat,
            T=float(T),
            loss=float(res.fun),
            success=bool(res.success),
            message=str(res.message),
        )

        diag = {
            "loss": float(res.fun),
            "loss_name": "negative_log_likelihood",
            "status": "success" if res.success else "failed",
            "message": str(res.message),
            "params": {"a": a_hat, "b": b_hat},
        }
        return fitted, diag

    def _transform_surface_with_model(
        self,
        fitted_model: BetaCalibrationFitted,
        x_grid: np.ndarray,
        f_q: np.ndarray,
        F_q: np.ndarray,
        *,
        T: float,
        info: dict,
    ) -> dict:
        from scipy.stats import beta as beta_dist

        x_grid = _as_1d(x_grid)
        f_q = _trapz_normalize_density(x_grid, f_q, eps=self.eps)
        F_q = _as_1d(F_q)

        if not np.isfinite(fitted_model.a) or not np.isfinite(fitted_model.b):
            raise RuntimeError("Beta calibration model is not valid for this maturity.")

        u = np.clip(F_q, self.eps, 1.0 - self.eps)

        weight = beta_dist.pdf(u, fitted_model.a, fitted_model.b)
        weight = np.where(np.isfinite(weight) & (weight >= 0), weight, 0.0)

        f_p_raw = f_q * weight
        f_p = _trapz_normalize_density(x_grid, f_p_raw, eps=self.eps)

        # The analytical calibrated CDF is beta.cdf(F_q), but after grid
        # normalization of f_p, using numerical CDF keeps F_p consistent with f_p.
        F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)

        weight_final = f_p / np.maximum(f_q, self.eps)

        M = f_q / np.maximum(f_p, self.eps)
        Em = float(np.trapezoid(M * f_p, x_grid))
        if np.isfinite(Em) and Em > self.eps:
            M = M / Em

        return {
            "x_grid": x_grid,
            "f_q": f_q,
            "F_q": F_q,
            "f_p": f_p,
            "F_p": F_p,
            "weight": weight_final,
            "pricing_kernel": M,
            "a": fitted_model.a,
            "b": fitted_model.b,
            "T_fit": float(T),
        }


# ============================================================
# Schreindorfer conditional-risk kernel
# ============================================================

@dataclass(frozen=True)
class ConditionalRiskSpec:
    """
    Specification for the Schreindorfer conditional-risk pricing kernel.

    Kernel form:

        M(r, sigma; theta)
            = exp( delta_t + sum_i c_i(sigma) r^i )

    with:

        c_i(sigma) = c_i / sigma^(b*i)

    where delta_t is solved each observation/day so that:

        E_Q[M] = 1

    holds exactly on the grid.
    """
    N: int = 2
    b_bounds: Tuple[float, float] = (-5.0, 5.0)
    c_bounds: Tuple[float, float] = (-50.0, 50.0)


@dataclass
class ConditionalRiskFitted:
    theta_hat: np.ndarray
    spec: ConditionalRiskSpec
    T: float
    loss: float
    success: bool
    message: str


@register_transform("conditional_risk")
class ConditionalRiskKernel(MeasureTransform):
    """
    Schreindorfer-style conditional-risk pricing kernel.

    Parameterization:

        M(r, sigma; theta)
            = exp(delta_t + sum_i c_i(sigma) r^i)

        c_i(sigma)
            = c_i / sigma^(b*i)

    where delta_t is solved per observation so that:

        ∫ M(r,sigma) f_Q(r) dr = 1.
    """

    def __init__(
        self,
        *,
        spec: ConditionalRiskSpec = ConditionalRiskSpec(),
        maxiter: int = 400,
        x0: Optional[np.ndarray] = None,
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
        super().__init__(
            key_spec=key_spec,
            fit_trim_alpha=fit_trim_alpha,
            min_obs=min_obs,
            fit_maturities=fit_maturities,
            maturity_match_tol=maturity_match_tol,
            eps=eps,
            verbose=verbose,
            penalty_value=penalty_value,
            cache_spec=cache_spec,
        )
        self.spec = spec
        self.maxiter = int(maxiter)
        self.x0 = None if x0 is None else _as_1d(x0)

    def _cache_params(self) -> dict:
        return {"spec": self.spec, "maxiter": self.maxiter, "x0": self.x0}

    def _theta_dim(self) -> int:
        # theta = [b, c1, c2, ..., cN]
        return int(self.spec.N) + 1

    def _unpack_theta(self, theta: np.ndarray):
        theta = _as_1d(theta)
        b = float(theta[0])
        c = theta[1:]
        return b, c

    def _bounds(self):
        b_lo, b_hi = self.spec.b_bounds
        c_lo, c_hi = self.spec.c_bounds

        bounds = [(float(b_lo), float(b_hi))]
        bounds.extend([(float(c_lo), float(c_hi)) for _ in range(int(self.spec.N))])
        return bounds

    def _g_no_delta(self, r: np.ndarray, sigma: float, theta: np.ndarray) -> np.ndarray:
        b, c = self._unpack_theta(theta)
        r = _as_1d(r)

        g = np.zeros_like(r, dtype=float)
        sigma = max(float(sigma), self.eps)

        for i in range(1, int(self.spec.N) + 1):
            coeff = c[i - 1] / (sigma ** (b * i))
            g += coeff * (r ** i)

        return g

    def _solve_delta(self, x_grid, f_q, sigma, theta):
        """
        Solve delta_t so that:

            ∫ exp(delta + g(r)) f_Q(r) dr = 1.

        Therefore:

            delta = -log ∫ exp(g(r)) f_Q(r) dr
        """
        g0 = self._g_no_delta(x_grid, sigma, theta)
        mass = np.trapezoid(
            np.exp(np.clip(g0, -700, 700)) * f_q,
            x_grid,
        )

        if not np.isfinite(mass) or mass <= self.eps:
            return np.nan

        return float(-np.log(mass))

    def _kernel(self, x_grid, f_q, sigma, theta):
        delta = self._solve_delta(x_grid, f_q, sigma, theta)
        if not np.isfinite(delta):
            return None, None

        g0 = self._g_no_delta(x_grid, sigma, theta)
        g = delta + g0

        M = np.exp(np.clip(g, -700, 700))
        return M, delta

    def _fit_one_maturity(self, hist_T: pd.DataFrame, *, T: float):
        p = self._theta_dim()
        x0 = np.zeros(p, dtype=float) if self.x0 is None else self.x0.copy()
        if x0.size != p:
            raise ValueError(f"x0 has length {x0.size}, expected {p}.")

        bounds = self._bounds()
        eps = self.eps
        penalty = self.penalty_value

        def nll(theta):
            theta = _as_1d(theta)

            if theta.size != p or not np.all(np.isfinite(theta)):
                return penalty

            total_ll = 0.0

            for _, row in hist_T.iterrows():
                x_grid = _as_1d(row["x_grid"])
                f_q = _trapz_normalize_density(x_grid, row["f_q"], eps=eps)

                sigma = float(row.get("sigma", 1.0))
                realized = float(row["realized_return"])

                if not np.all(np.isfinite(f_q)):
                    return penalty

                M, delta = self._kernel(x_grid, f_q, sigma, theta)
                if M is None:
                    return penalty

                f_p = _trapz_normalize_density(
                    x_grid,
                    f_q / np.maximum(M, eps),
                    eps=eps,
                )

                if not np.all(np.isfinite(f_p)):
                    return penalty

                val = _safe_interp(realized, x_grid, f_p)
                if not np.isfinite(val) or val <= eps:
                    return penalty

                total_ll += np.log(val)

            out = -float(total_ll)
            return out if np.isfinite(out) else penalty

        res = minimize(
            nll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter},
        )

        fitted = ConditionalRiskFitted(
            theta_hat=np.asarray(res.x, dtype=float),
            spec=self.spec,
            T=float(T),
            loss=float(res.fun),
            success=bool(res.success),
            message=str(res.message),
        )

        diag = {
            "loss": float(res.fun),
            "loss_name": "negative_log_likelihood",
            "status": "success" if res.success else "failed",
            "message": str(res.message),
            "params": {
                "b": float(res.x[0]),
                **{f"c{i}": float(v) for i, v in enumerate(res.x[1:], start=1)},
            },
        }

        return fitted, diag

    def _transform_surface_with_model(
        self,
        fitted_model,
        x_grid,
        f_q,
        F_q,
        *,
        T,
        info,
    ):
        x_grid = _as_1d(x_grid)
        f_q = _trapz_normalize_density(x_grid, f_q, eps=self.eps)
        F_q = _as_1d(F_q)

        sigma = _find_sigma(info, self.key_spec.sigma_keys, default=1.0)

        M, delta = self._kernel(
            x_grid,
            f_q,
            sigma,
            fitted_model.theta_hat,
        )

        if M is None:
            raise RuntimeError("Failed to compute pricing kernel.")

        f_p = _trapz_normalize_density(
            x_grid,
            f_q / np.maximum(M, self.eps),
            eps=self.eps,
        )

        F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)
        weight = f_p / np.maximum(f_q, self.eps)

        return {
            "x_grid": x_grid,
            "f_q": f_q,
            "F_q": F_q,
            "f_p": f_p,
            "F_p": F_p,
            "weight": weight,
            "pricing_kernel": M,
            "delta": delta,
            "theta_hat": fitted_model.theta_hat,
            "T_fit": float(T),
        }
def add_axis_named_surfaces(out: dict) -> dict:
    """
    Standardize naming convention.

    lr = log return, r = gross return, k = terminal price / strike units.
    """

    grid_lr = np.asarray(out["grid_lr"], float)
    grid_r = np.exp(grid_lr)

    S0 = out.get("S0", None)
    grid_k = None if S0 is None else float(S0) * grid_r

    rnd_lr = np.asarray(out["rnd_lr_surface"], float)
    physical_lr = np.asarray(out["physical_lr_surface"], float)

    cdf_lr = np.asarray(out["rnd_lr_surface"], float)
    physical_cdf_lr = np.asarray(out["physical_lr_surface"], float)

    # Change of variables:
    # r_gross = exp(lr), so f_R(R) = f_lr(lr) / R
    rnd_r = rnd_lr / np.maximum(grid_r[None, :], 1e-300)
    physical_r = physical_lr / np.maximum(grid_r[None, :], 1e-300)

    if grid_k is not None:
        # K = S0 * exp(lr), so f_K(K) = f_lr(lr) / K
        rnd_k = rnd_lr / np.maximum(grid_k[None, :], 1e-300)
        physical_k = physical_lr / np.maximum(grid_k[None, :], 1e-300)
    else:
        rnd_k = None
        physical_k = None

    out.update(
        {
            # grids
            "grid_lr": grid_lr,
            "grid_r": grid_r,
            "grid_k": grid_k,

            # risk-neutral / RND objects
            "rnd_lr_surface": rnd_lr,
            "cdf_lr_surface": cdf_lr,
            "rnd_r_surface": rnd_r,
            "cdf_r_surface": cdf_lr,
            "rnd_k_surface": rnd_k,
            "cdf_k_surface": cdf_lr,

            # physical objects
            "physical_lr_surface": physical_lr,
            "physical_cdf_lr_surface": physical_cdf_lr,
            "physical_r_surface": physical_r,
            "physical_cdf_r_surface": physical_cdf_lr,
            "physical_k_surface": physical_k,
            "physical_cdf_k_surface": physical_cdf_lr,

            # measure-change objects
            "pricing_kernel_surface": out["pricing_kernel_surface"],
            "relative_risk_aversion_surface": out["relative_risk_aversion_surface"],
            "measure_weight_surface": out["weight_surface"],
        }
    )

    return out

# ============================================================
# Example usage template
# ============================================================



if __name__ == "__main__":
    model = get_transform(
    "beta",
    fit_maturities=[7/365, 14/365],
    maturity_match_tol=0/365,
    fit_trim_alpha=(0.025, 0.025),
    verbose=True,
    )
    model.fit(RND_dict_2013_2025, logreturns)
    result_model = model.transform_info(sliced_df)
    
    f=P_Q_K_multipanel(
    result_model,
    x_space="log",
    r_bounds=(-1.9, 1.9),
    trunc_mode="cdf+rbounds",
    truncate=True,
    ptail_alpha=(0.01, 0.01),
    )
    
    plot_pricing_kernel_surface(
    result_model,
    x_axis="r",
    log_z=False,
    x_bounds=(-0.6,.5),
    T_bounds=(0.05,0.2),
    title="BETA"
    )
    
    

    #####################################################################
    model_beta = get_transform(
    "beta",
    fit_trim_alpha=(0.025, 0.025),
    cache_spec=CacheSpec(
        enabled=True,
        folder="measure_transform_cache",
        cache_fit=True,
        cache_transform=False,
        dataset_tag="beta",
    ),
    verbose=True,
    )
    model_beta.fit(RND_dict_2013_2025, logreturns)
    result_beta = model_beta.transform_info(sliced_df)
    
    f=P_Q_K_multipanel(
    result_beta,
    x_space="log",
    r_bounds=(-1.9, 1.9),
    trunc_mode="cdf+rbounds",
    truncate=True,
    ptail_alpha=(0.01, 0.01),
    )
    
    plot_pricing_kernel_surface(
    result_beta,
    x_axis="r",
    log_z=False,
    x_bounds=(-0.6,.5),
    T_bounds=(0.05,0.2),
    title="BETA"
    )
    
    
    ########################################################################
    model_kde = get_transform(
    "nonparametric",
    fit_trim_alpha=(0.05, 0.05),
    spec=NonparametricCalibrationSpec(bandwidth="scott"),
    cache_spec=CacheSpec(
        enabled=False,
        folder="measure_transform_cache",
        cache_fit=True,
        cache_transform=False,
        dataset_tag="kde",
    ),
    verbose=True,
    )
    model_kde.fit(RND_dict_2013_2025, logreturns)
    result_kde = model_kde.transform_info(sliced_df)
    
    plot_pricing_kernel_surface(
    result_kde,
    x_axis="r",
    log_z=False,
    x_bounds=(-0.6,.25),
    T_bounds=(0.05,0.2),
    title="Exponential polynomial"
    )

    f=P_Q_K_multipanel(
    result_kde,
    x_space="log",
    r_bounds=(-1.9, 1.9),
    trunc_mode="cdf+rbounds",
    truncate=True,
    ptail_alpha=(0.01, 0.01),
    )
    

    
    
    ####################################################################
    model_exponential = get_transform(
        "exponential_polynomial",
        fit_maturities=[7/365, 15/365],
        maturity_match_tol=0/365,
        theta_spec=ThetaSpec(N=2, Ksig=2),
        cache_spec=CacheSpec(
            enabled=False,
            folder="measure_transform_cache",
            cache_fit=False,
            cache_transform=False,
            dataset_tag="exponentialn2k1",
        ),
        fit_trim_alpha=(0.05, 0.05),
        verbose=True)
    
    model_exponential.fit(RND_dict_2013_2025, logreturns)
    result_exp = model_exponential.transform_info(sliced_df)
    
    f=P_Q_K_multipanel(
    result_exp,
    x_space="log",
    r_bounds=(-0.6, 0.6),
    trunc_mode="cdf+rbounds",
    truncate=True,
    ptail_alpha=(0.01, 0.01),
    )
    
    plot_pricing_kernel_surface(
    result_exp,
    x_axis="r",
    log_z=False,
    x_bounds=(-0.2,0.2),
    T_bounds=(0.04,0.1),
    title="Exponential polynomial"
    )
    
    
    ##################################################
    model_cond = get_transform(
        "conditional_risk",
        spec=ConditionalRiskSpec(N=2),
        fit_maturities=[7/365, 15/365],
        maturity_match_tol=0/365,
        cache_spec=CacheSpec(
            enabled=True,
            folder="measure_transform_cache",
            cache_fit=True,
            cache_transform=False,
            dataset_tag="conditional_n2",
        ),
        fit_trim_alpha=(0.05, 0.05),
        verbose=True,
    )
    model_cond.fit(RND_dict_2013_2025, logreturns)
    result_cond = model_cond.transform_info(sliced_df)

    
    f=P_Q_K_multipanel(
    result_cond,
    x_space="log",
    r_bounds=(-1.9, 1.9),
    trunc_mode="cdf+rbounds",
    truncate=True,
    ptail_alpha=(0.01, 0.01),
    )
    
    plot_pricing_kernel_surface(
    result_cond,
    x_axis="r",
    log_z=False,
    x_bounds=(-0.2,0.2),
    T_bounds=(0.04,0.1),
    title="conditional risk kernel"
    )
    
    pass
