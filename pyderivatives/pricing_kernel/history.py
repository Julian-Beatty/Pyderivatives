from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from typing import Any, Dict, Literal, Optional
import copy

from .utils import (
    _as_1d,
    _safe_interp,
    _find_spot,
    _find_sigma,
)


class HistoryMixin:
    """
    Mixin for building realized-return calibration samples.

    It converts a history of RND dictionaries plus stock returns into
    maturity-specific fitting datasets used by all pricing-kernel transforms.
    """

    def build_history_by_maturity(
        self,
        rnd_history_dict: Dict[Any, dict],
        logreturns: pd.DataFrame | pd.Series,
    ) -> Dict[float, pd.DataFrame]:
        """
        Build the calibration sample by maturity.

        For maturity T, the realized return is the cumulative log return from
        the anchor date to approximately 365*T calendar days ahead.
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
                        "pit": float(np.clip(pit, self.eps, 1.0 - self.eps)),
                        "sigma": float(sigma),
                        "S0": np.nan if S0 is None else float(S0),
                        # Preserve the daily Q-model state for stochastic-dynamics
                        # measure transforms. Existing density-only transforms simply
                        # ignore these additional columns.
                        "model": info.get("model", None),
                        "params": info.get("params", None),
                        "r": info.get("r", np.nan),
                        "q": info.get("q", np.nan),
                        "meta": info.get("meta", {}),
                        "x_grid": x_grid,
                        "f_q": f_q,
                        "F_q": F_q,
                    }
                )

        columns = [
            "date",
            "end_date",
            "T",
            "horizon_days",
            "realized_return",
            "pit",
            "sigma",
            "S0",
            "model",
            "params",
            "r",
            "q",
            "meta",
            "x_grid",
            "f_q",
            "F_q",
        ]

        out: Dict[float, pd.DataFrame] = {}

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

    def _standardize_return_series(
        self,
        logreturns: pd.DataFrame | pd.Series,
    ) -> pd.Series:
        """
        Convert a return series/dataframe into a clean DatetimeIndex Series.
        """
        if isinstance(logreturns, pd.Series):
            s = logreturns.copy()
        else:
            df = logreturns.copy()

            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" not in df.columns:
                    raise ValueError("logreturns must have a DatetimeIndex or a 'date' column.")

                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            numeric_cols = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
            ]

            if not numeric_cols:
                raise ValueError("logreturns DataFrame must contain at least one numeric return column.")

            s = df[numeric_cols[0]].copy()

        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = pd.to_numeric(s, errors="coerce").dropna().sort_index()

        return s

    def _realized_horizon_return(
        self,
        ret_series: pd.Series,
        anchor_date: pd.Timestamp,
        horizon_days: int,
    ) -> Tuple[float, Optional[pd.Timestamp]]:
        """
        Sum available trading-day log returns after anchor_date and up to
        anchor_date + horizon_days.
        """
        if ret_series.empty:
            return np.nan, None

        anchor_date = pd.Timestamp(anchor_date).tz_localize(None)
        target_date = anchor_date + pd.Timedelta(days=int(horizon_days))

        window = ret_series.loc[
            (ret_series.index > anchor_date)
            & (ret_series.index <= target_date)
        ]

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
        """
        Build daily log returns from stock data.

        Preference order:
            1. explicit return_col
            2. adjusted price column
            3. raw price column adjusted by factor
            4. raw price column
        """
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
                price = price / factor.replace(0.0, np.nan)

        price = price.replace([np.inf, -np.inf], np.nan).dropna()
        price = price[price > 0].sort_index()

        ret = np.log(price / price.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        ret.name = "log_return"

        return ret
    
WindowMode = Literal[
    "fixed",
    "expanding",
    "rolling",
    "centered",
]


def _normalize_date_dict(
    data: Dict[Any, dict],
) -> Dict[pd.Timestamp, dict]:
    """
    Convert dictionary keys to timezone-naive pandas Timestamps and sort them.
    """
    out: Dict[pd.Timestamp, dict] = {}

    for raw_date, value in data.items():
        date = pd.Timestamp(raw_date)

        if date.tzinfo is not None:
            date = date.tz_localize(None)

        out[date] = value

    return dict(sorted(out.items()))


def _slice_date_dict(
    data: Dict[pd.Timestamp, dict],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Dict[pd.Timestamp, dict]:
    """
    Inclusively slice a Timestamp-keyed dictionary.
    """
    return {
        date: value
        for date, value in data.items()
        if (start is None or date >= start)
        and (end is None or date <= end)
    }


def _window_bounds(
    target_date: pd.Timestamp,
    *,
    mode: WindowMode,
    first_date: pd.Timestamp,
    lookback_days: Optional[int],
    lookahead_days: int,
    fixed_end_date: Optional[pd.Timestamp],
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Return inclusive calibration-window bounds for a target valuation date.

    Modes
    -----
    fixed:
        [first_date, fixed_end_date]

    expanding:
        [first_date, target_date]

    rolling:
        [target_date - lookback_days, target_date]

    centered:
        [target_date - lookback_days, target_date + lookahead_days]
    """
    target_date = pd.Timestamp(target_date)

    if mode == "fixed":
        if fixed_end_date is None:
            raise ValueError(
                "fixed_end_date is required when mode='fixed'."
            )

        return first_date, pd.Timestamp(fixed_end_date)

    if mode == "expanding":
        return first_date, target_date

    if mode == "rolling":
        if lookback_days is None or lookback_days <= 0:
            raise ValueError(
                "lookback_days must be positive when mode='rolling'."
            )

        start = target_date - pd.Timedelta(days=int(lookback_days))
        return start, target_date

    if mode == "centered":
        if lookback_days is None or lookback_days < 0:
            raise ValueError(
                "lookback_days must be nonnegative when mode='centered'."
            )

        if lookahead_days < 0:
            raise ValueError(
                "lookahead_days must be nonnegative when mode='centered'."
            )

        start = target_date - pd.Timedelta(days=int(lookback_days))
        end = target_date + pd.Timedelta(days=int(lookahead_days))

        return start, end

    raise ValueError(
        "mode must be one of "
        "'fixed', 'expanding', 'rolling', or 'centered'."
    )



def _normalize_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    """Convert a date-like value to a timezone-naive Timestamp."""
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{name} must be date-like; received {value!r}.") from exc

    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    return ts


def _validate_transform_window_inputs(
    *,
    reserve_obs: int,
    min_fit_dates: int,
    on_error: Literal["raise", "skip"],
) -> None:
    if reserve_obs < 0:
        raise ValueError("reserve_obs cannot be negative.")

    if min_fit_dates < 1:
        raise ValueError("min_fit_dates must be at least 1.")

    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'.")


def _window_fit_metadata(
    *,
    mode: WindowMode,
    target_date: pd.Timestamp,
    fit_start: Optional[pd.Timestamp],
    fit_end: Optional[pd.Timestamp],
    n_fit_dates: int,
    reserve_obs: int,
    lookback_days: Optional[int],
    lookahead_days: int,
    refit_every: int,
    fixed_end_date: Optional[pd.Timestamp],
) -> dict:
    """Build the common fit-window metadata attached to every output."""
    return {
        "mode": mode,
        "target_date": target_date,
        "fit_start_date": fit_start,
        "fit_end_date": fit_end,
        "n_fit_dates": int(n_fit_dates),
        "reserve_obs": int(reserve_obs),
        "lookback_days": None if lookback_days is None else int(lookback_days),
        "lookahead_days": int(lookahead_days),
        "refit_every": int(refit_every),
        "fixed_end_date": fixed_end_date,
        "ex_post": mode == "centered",
    }



def fit_transform_window(
    transform,
    rnd_history_dict: Dict[Any, dict],
    stock_df: pd.DataFrame,
    *,
    target_date: Optional[Any] = None,
    mode: WindowMode = "expanding",
    lookback_days: Optional[int] = None,
    lookahead_days: int = 0,
    fixed_end_date: Optional[Any] = None,
    fit_kwargs: Optional[dict] = None,
    min_fit_dates: int = 20,
    reserve_obs: int = 0,
    verbose: bool = True,
):
    """Fit a transform for one calibration window and return it for reuse.

    Parameters are identical to the fitting portion of :func:`transform_one_date`.
    For ``mode='fixed'``, ``target_date`` is optional because the calibration
    window does not depend on the output date. For all other modes it is
    required.

    Returns
    -------
    fitted_transform, window_metadata
        A deep-copied fitted transform and metadata describing the fitting
        window. The fitted object can be applied repeatedly with
        ``fitted_transform.transform_rnd(info)``.
    """
    _validate_transform_window_inputs(
        reserve_obs=int(reserve_obs),
        min_fit_dates=int(min_fit_dates),
        on_error="raise",
    )

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    rnd = _normalize_date_dict(rnd_history_dict)

    if not rnd:
        raise ValueError("rnd_history_dict is empty.")

    first_date = next(iter(rnd))
    fixed_end_ts = (
        None
        if fixed_end_date is None
        else _normalize_timestamp(fixed_end_date, name="fixed_end_date")
    )

    if target_date is None:
        if mode != "fixed":
            raise ValueError("target_date is required unless mode='fixed'.")
        target_ts = first_date
    else:
        target_ts = _normalize_timestamp(target_date, name="target_date")
        if target_ts not in rnd:
            raise KeyError(
                f"target_date {target_ts.date()} was not found in rnd_history_dict."
            )

    fit_start, fit_end = _window_bounds(
        target_ts,
        mode=mode,
        first_date=first_date,
        lookback_days=lookback_days,
        lookahead_days=int(lookahead_days),
        fixed_end_date=fixed_end_ts,
    )

    fit_rnd = _slice_date_dict(rnd, fit_start, fit_end)
    if len(fit_rnd) < int(min_fit_dates):
        raise ValueError(
            f"Too few fitting dates: {len(fit_rnd)} "
            f"< min_fit_dates={int(min_fit_dates)}."
        )

    fitted_transform = copy.deepcopy(transform)
    fitted_transform.fit(
        fit_rnd,
        stock_df=stock_df,
        **fit_kwargs,
    )

    metadata = _window_fit_metadata(
        mode=mode,
        target_date=target_ts,
        fit_start=fit_start,
        fit_end=fit_end,
        n_fit_dates=len(fit_rnd),
        reserve_obs=int(reserve_obs),
        lookback_days=lookback_days,
        lookahead_days=int(lookahead_days),
        refit_every=1,
        fixed_end_date=fixed_end_ts,
    )

    if verbose:
        print(
            f"[fit window] mode={mode} | "
            f"window={fit_start.date() if fit_start is not None else 'beginning'} "
            f"to {fit_end.date() if fit_end is not None else 'end'} | "
            f"n_dates={len(fit_rnd)}"
        )

    return fitted_transform, metadata

def transform_one_date(
    transform,
    rnd_history_dict: Dict[Any, dict],
    stock_df: pd.DataFrame,
    target_date: Any,
    *,
    mode: WindowMode = "expanding",
    lookback_days: Optional[int] = None,
    lookahead_days: int = 0,
    fixed_end_date: Optional[Any] = None,
    fit_kwargs: Optional[dict] = None,
    min_fit_dates: int = 20,
    reserve_obs: int = 0,
    verbose: bool = True,
) -> dict:
    """Fit and transform one target date independently.

    This is the cluster-friendly primitive underlying ``transform_history``.
    It constructs the target date's calibration window, deep-copies and fits
    the supplied transform, transforms that date's RND, and attaches the same
    ``window_fit`` metadata used by the historical wrapper.

    Notes
    -----
    ``reserve_obs`` is metadata only here. The history wrapper uses it to
    choose eligible output dates; a direct single-date call is allowed whenever
    the requested target exists and its fitting window contains enough dates.
    """
    _validate_transform_window_inputs(
        reserve_obs=int(reserve_obs),
        min_fit_dates=int(min_fit_dates),
        on_error="raise",
    )

    rnd = _normalize_date_dict(rnd_history_dict)
    target_ts = _normalize_timestamp(target_date, name="target_date")
    if target_ts not in rnd:
        raise KeyError(f"target_date {target_ts.date()} was not found in rnd_history_dict.")

    fitted_transform, window_metadata = fit_transform_window(
        transform=transform,
        rnd_history_dict=rnd,
        stock_df=stock_df,
        target_date=target_ts,
        mode=mode,
        lookback_days=lookback_days,
        lookahead_days=lookahead_days,
        fixed_end_date=fixed_end_date,
        fit_kwargs=fit_kwargs,
        min_fit_dates=min_fit_dates,
        reserve_obs=reserve_obs,
        verbose=False,
    )

    transformed = fitted_transform.transform_rnd(rnd[target_ts])
    transformed.setdefault("window_fit", {})
    transformed["window_fit"].update(window_metadata)
    transformed["window_fit"]["target_date"] = target_ts

    if verbose:
        print(
            f"[fit+transform] target={target_ts.date()} | mode={mode} | "
            f"window={window_metadata['fit_start_date'].date()} "
            f"to {window_metadata['fit_end_date'].date()} | "
            f"n_dates={window_metadata['n_fit_dates']}"
        )

    return transformed


def transform_history(
    transform,
    rnd_history_dict: Dict[Any, dict],
    stock_df: pd.DataFrame,
    *,
    mode: WindowMode = "expanding",
    lookback_days: Optional[int] = None,
    lookahead_days: int = 0,
    fixed_end_date: Optional[Any] = None,
    reserve_obs: int = 0,
    refit_every: int = 1,
    fit_kwargs: Optional[dict] = None,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    min_fit_dates: int = 20,
    on_error: Literal["raise", "skip"] = "skip",
    verbose: bool = True,
) -> Dict[pd.Timestamp, dict]:
    """Construct a date-keyed physical-density history.

    With the normal setting ``refit_every=1``, each eligible target date is
    delegated to :func:`transform_one_date`, making the serial and cluster
    workflows use the same implementation. Values greater than one retain the
    prior debugging behavior by reusing a fitted transform between refits.
    """
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1.")

    _validate_transform_window_inputs(
        reserve_obs=int(reserve_obs),
        min_fit_dates=int(min_fit_dates),
        on_error=on_error,
    )

    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    rnd = _normalize_date_dict(rnd_history_dict)
    all_dates = list(rnd)

    if not all_dates:
        return {}

    fixed_end_ts = (
        None
        if fixed_end_date is None
        else _normalize_timestamp(fixed_end_date, name="fixed_end_date")
    )
    start_ts = None if start_date is None else _normalize_timestamp(start_date, name="start_date")
    end_ts = None if end_date is None else _normalize_timestamp(end_date, name="end_date")

    target_dates = all_dates[int(reserve_obs):]
    if start_ts is not None:
        target_dates = [date for date in target_dates if date >= start_ts]
    if end_ts is not None:
        target_dates = [date for date in target_dates if date <= end_ts]

    physical_dict: Dict[pd.Timestamp, dict] = {}

    # Normal production path: every date is an independent fit-and-transform.
    if int(refit_every) == 1:
        for target_date in target_dates:
            try:
                physical_dict[target_date] = transform_one_date(
                    transform,
                    rnd,
                    stock_df,
                    target_date,
                    mode=mode,
                    lookback_days=lookback_days,
                    lookahead_days=int(lookahead_days),
                    fixed_end_date=fixed_end_ts,
                    fit_kwargs=fit_kwargs,
                    min_fit_dates=int(min_fit_dates),
                    reserve_obs=int(reserve_obs),
                    verbose=verbose,
                )
            except Exception as exc:
                if on_error == "raise":
                    raise
                if verbose:
                    print(
                        f"[date failed] target={target_date.date()} | "
                        f"{type(exc).__name__}: {exc}"
                    )
        return physical_dict

    # Debugging path: preserve fitted-model reuse when refit_every > 1.
    first_date = all_dates[0]
    fitted_transform = None
    last_fit_bounds: Optional[tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = None
    last_fit_n_dates: Optional[int] = None

    for target_number, target_date in enumerate(target_dates):
        fit_start, fit_end = _window_bounds(
            target_date,
            mode=mode,
            first_date=first_date,
            lookback_days=lookback_days,
            lookahead_days=int(lookahead_days),
            fixed_end_date=fixed_end_ts,
        )
        fit_rnd = _slice_date_dict(rnd, fit_start, fit_end)

        should_refit = fitted_transform is None or target_number % int(refit_every) == 0
        if mode == "fixed" and fitted_transform is not None:
            should_refit = False

        if should_refit:
            if len(fit_rnd) < int(min_fit_dates):
                if verbose:
                    print(
                        f"[skip fit] target={target_date.date()} | "
                        f"fit dates={len(fit_rnd)} | minimum={min_fit_dates}"
                    )
                continue

            candidate_transform = copy.deepcopy(transform)
            try:
                candidate_transform.fit(fit_rnd, stock_df=stock_df, **fit_kwargs)
            except Exception as exc:
                if on_error == "raise":
                    raise
                if verbose:
                    print(
                        f"[fit failed] target={target_date.date()} | "
                        f"{type(exc).__name__}: {exc}"
                    )
                continue

            fitted_transform = candidate_transform
            last_fit_bounds = (fit_start, fit_end)
            last_fit_n_dates = len(fit_rnd)

            if verbose:
                print(
                    f"[fit] target={target_date.date()} | mode={mode} | "
                    f"window={fit_start.date() if fit_start is not None else 'beginning'} "
                    f"to {fit_end.date() if fit_end is not None else 'end'} | "
                    f"n_dates={len(fit_rnd)}"
                )

        if fitted_transform is None or last_fit_bounds is None or last_fit_n_dates is None:
            continue

        try:
            transformed = fitted_transform.transform_rnd(rnd[target_date])
            transformed.setdefault("window_fit", {})
            transformed["window_fit"].update(
                _window_fit_metadata(
                    mode=mode,
                    target_date=target_date,
                    fit_start=last_fit_bounds[0],
                    fit_end=last_fit_bounds[1],
                    n_fit_dates=last_fit_n_dates,
                    reserve_obs=int(reserve_obs),
                    lookback_days=lookback_days,
                    lookahead_days=int(lookahead_days),
                    refit_every=int(refit_every),
                    fixed_end_date=fixed_end_ts,
                )
            )
            physical_dict[target_date] = transformed
        except Exception as exc:
            if on_error == "raise":
                raise
            if verbose:
                print(
                    f"[transform failed] target={target_date.date()} | "
                    f"{type(exc).__name__}: {exc}"
                )

    return physical_dict
