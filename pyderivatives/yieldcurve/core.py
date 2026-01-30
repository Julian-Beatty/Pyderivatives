from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

from .utils import parse_maturity_to_years, maturity_sort_key
from .registry import get_model  # this also triggers model imports/registration


@dataclass
class create_yield_curve:
    """
    Container around your wide DataFrame:
      Date | 1M | 2M | 3M | 6M | 1Y | 2Y | ...

    Assumes maturity columns are already standardized (1M, 2Y, ...).
    """
    df: pd.DataFrame

    def __post_init__(self):
        if "Date" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'Date' column.")

        df = self.df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        mats = []
        taus = []
        for c in df.columns:
            if c == "Date":
                continue
            t = parse_maturity_to_years(c)
            if t is not None:
                mats.append(c)
                taus.append(t)

        if not mats:
            raise ValueError("No maturity columns detected (expected like '1M','6M','1Y',...).")

        # sort by maturity
        order = np.argsort(np.array(taus))
        self.maturity_cols = [mats[i] for i in order]
        self.taus_years = np.array([taus[i] for i in order], float)

        # store cleaned df
        self.df = df

    def fit(self, model: str, *args, **kwargs):
        """
        Generic entry point:
            yc.fit("nelson_siegel", grid_days=(1,365), fit_days_window=(1,365*5). Fits on first 5 of observed yield qoutes. Evaluates on a grid of 1 day to 365 days.

        Each model function takes (curve=self, ...) and returns a DataFrame
        or (fitted_df, params_df).
        """
        fn = get_model(model)
        return fn(self, *args, **kwargs)
