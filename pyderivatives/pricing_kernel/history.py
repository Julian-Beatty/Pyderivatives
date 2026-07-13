from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

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