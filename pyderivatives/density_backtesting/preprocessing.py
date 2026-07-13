from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnConfig:
    date_col: str = "date"

    return_col: Optional[str] = None
    adjusted_price_col: Optional[str] = None
    price_col: Optional[str] = "price"
    adjustment_factor_col: Optional[str] = "ajexdi"

    return_type: str = "log"  # "log" or "simple"
    adjustment_method: str = "multiply"  # "multiply" or "divide"

    dropna: bool = True


@dataclass
class ReturnSeries:
    returns: pd.Series
    adjusted_prices: Optional[pd.Series] = None
    raw_prices: Optional[pd.Series] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def horizon_return(self, horizon_days: int, *, forward: bool = True) -> pd.Series:
        """
        If forward=True:
            value at date t is sum of returns after t over the next horizon_days observations.

        If forward=False:
            value at date t is trailing horizon_days return ending at t.
        """
        h = int(horizon_days)
        if h < 1:
            raise ValueError("horizon_days must be >= 1.")

        r = self.returns.sort_index()

        if forward:
            out = (
                r.shift(-1)
                .rolling(h)
                .sum()
                .shift(-(h - 1))
            )
        else:
            out = r.rolling(h).sum()

        return out.dropna()

    def realized_after(
        self,
        date: Any,
        horizon_days: int,
        *,
        mode: str = "trading",
        tolerance_days: int = 3,
    ):
        date = pd.Timestamp(date).tz_localize(None)
        h = int(horizon_days)

        if h < 1:
            raise ValueError("horizon_days must be >= 1.")

        r = self.returns.sort_index().dropna()

        if mode == "trading":
            future = r[r.index > date]

            if len(future) < h:
                return None, None

            realized = float(future.iloc[:h].sum())
            end_date = future.index[h - 1]

            return realized, end_date

        if mode == "calendar":
            if self.adjusted_prices is None:
                raise ValueError(
                    "Calendar realized returns require adjusted_prices."
                )

            prices = self.adjusted_prices.sort_index().dropna()
            prices.index = pd.to_datetime(prices.index).tz_localize(None)

            if date not in prices.index:
                prior = prices[prices.index <= date]
                if prior.empty:
                    return None, None
                start_date = prior.index[-1]
            else:
                start_date = date

            target = date + pd.Timedelta(days=h)

            if prices.empty:
                return None, None

            idx = prices.index.get_indexer([target], method="nearest")[0]

            if idx < 0:
                return None, None

            end_date = prices.index[idx]
            err_days = abs((end_date - target).days)

            if err_days > int(tolerance_days):
                return None, None

            p0 = float(prices.loc[start_date])
            p1 = float(prices.loc[end_date])

            if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0 or p1 <= 0:
                return None, None

            realized = float(np.log(p1 / p0))

            return realized, end_date

        raise ValueError("mode must be 'trading' or 'calendar'.")

    @classmethod
    def from_stock_df(
        cls,
        stock_df: pd.DataFrame | pd.Series,
        config: ReturnConfig = ReturnConfig(),
    ) -> "ReturnSeries":
        if isinstance(stock_df, pd.Series):
            s = stock_df.copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s = s.sort_index().astype(float)

            if config.dropna:
                s = s.dropna()

            return cls(
                returns=s,
                metadata={
                    "source": "series",
                    "return_type": "supplied",
                },
            )

        df = stock_df.copy()

        if config.date_col in df.columns:
            df[config.date_col] = pd.to_datetime(df[config.date_col]).dt.tz_localize(None)
            df = df.sort_values(config.date_col).set_index(config.date_col)
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()

        df = df[~df.index.duplicated(keep="last")].sort_index()

        raw_prices = None
        adjusted_prices = None

        if config.return_col is not None and config.return_col in df.columns:
            returns = pd.to_numeric(df[config.return_col], errors="coerce").astype(float)

            source = "return_col"

        elif config.adjusted_price_col is not None and config.adjusted_price_col in df.columns:
            adjusted_prices = pd.to_numeric(df[config.adjusted_price_col], errors="coerce").astype(float)

            returns = _returns_from_price(
                adjusted_prices,
                return_type=config.return_type,
            )

            source = "adjusted_price_col"

        elif (
            config.price_col is not None
            and config.price_col in df.columns
            and config.adjustment_factor_col is not None
            and config.adjustment_factor_col in df.columns
        ):
            raw_prices = pd.to_numeric(df[config.price_col], errors="coerce").astype(float)
            adj_factor = pd.to_numeric(df[config.adjustment_factor_col], errors="coerce").astype(float)

            if config.adjustment_method == "multiply":
                adjusted_prices = raw_prices * adj_factor
            elif config.adjustment_method == "divide":
                adjusted_prices = raw_prices / adj_factor
            else:
                raise ValueError("adjustment_method must be 'multiply' or 'divide'.")

            returns = _returns_from_price(
                adjusted_prices,
                return_type=config.return_type,
            )

            source = "price_plus_adjustment_factor"

        else:
            raise ValueError(
                "Could not construct returns. Supply one of: "
                "return_col, adjusted_price_col, or price_col + adjustment_factor_col."
            )

        returns = returns.astype(float).sort_index()

        if config.dropna:
            returns = returns.dropna()
            if adjusted_prices is not None:
                adjusted_prices = adjusted_prices.dropna()
            if raw_prices is not None:
                raw_prices = raw_prices.dropna()

        return cls(
            returns=returns,
            adjusted_prices=adjusted_prices,
            raw_prices=raw_prices,
            metadata={
                "source": source,
                "return_type": config.return_type,
                "price_col": config.price_col,
                "adjusted_price_col": config.adjusted_price_col,
                "adjustment_factor_col": config.adjustment_factor_col,
                "adjustment_method": config.adjustment_method,
                "n_returns": int(len(returns)),
            },
        )


@dataclass
class MarketData:
    stock_df: Optional[pd.DataFrame] = None
    return_series: Optional[ReturnSeries] = None
    rnd_dicts: Dict[str, Dict[Any, dict]] = field(default_factory=dict)
    physical_dicts: Dict[str, Dict[Any, dict]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_inputs(
        cls,
        *,
        stock_df: Optional[pd.DataFrame] = None,
        return_config: ReturnConfig = ReturnConfig(),
        rnd_dicts: Optional[Dict[str, Dict[Any, dict]]] = None,
        physical_dicts: Optional[Dict[str, Dict[Any, dict]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MarketData":
        rs = None

        if stock_df is not None:
            rs = ReturnSeries.from_stock_df(stock_df, return_config)

        return cls(
            stock_df=stock_df.copy() if stock_df is not None else None,
            return_series=rs,
            rnd_dicts={} if rnd_dicts is None else dict(rnd_dicts),
            physical_dicts={} if physical_dicts is None else dict(physical_dicts),
            metadata={} if metadata is None else dict(metadata),
        )

    @property
    def returns(self) -> pd.Series:
        if self.return_series is None:
            raise ValueError("MarketData has no return_series.")
        return self.return_series.returns

    def realized_after(
        self,
        date: Any,
        horizon_days: int,
        *,
        mode: str = "trading",
        tolerance_days: int = 3,
    ):
        if self.return_series is None:
            raise ValueError("MarketData has no return_series.")

        return self.return_series.realized_after(
            date,
            horizon_days,
            mode=mode,
            tolerance_days=tolerance_days,
        )


def _returns_from_price(price: pd.Series, *, return_type: str = "log") -> pd.Series:
    price = pd.to_numeric(price, errors="coerce").astype(float).sort_index()

    if return_type == "log":
        return np.log(price).diff()

    if return_type == "simple":
        return price.pct_change()

    raise ValueError("return_type must be 'log' or 'simple'.")