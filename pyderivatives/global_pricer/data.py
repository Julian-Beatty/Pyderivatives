from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CallSurfaceDay:
    """
    Single day of call quotes (possibly sparse in K,T).
    All arrays are length N (quotes), not a rectangular grid.
    """
    S0: float
    r: float
    q: float
    K_obs: np.ndarray
    T_obs: np.ndarray
    C_obs: np.ndarray
    ticker: str = "Unknown"
    date: Optional[pd.Timestamp] = None

    def __post_init__(self):
        object.__setattr__(self, "S0", float(self.S0))
        object.__setattr__(self, "r", float(self.r))
        object.__setattr__(self, "q", float(self.q))
        object.__setattr__(self, "K_obs", np.asarray(self.K_obs, float).ravel())
        object.__setattr__(self, "T_obs", np.asarray(self.T_obs, float).ravel())
        object.__setattr__(self, "C_obs", np.asarray(self.C_obs, float).ravel())

        # --- normalize ticker ---
        t = self.ticker
        if t is None:
            t = "Unknown"
        t = str(t).strip()
        if t == "" or t.lower() == "nan":
            t = "Unknown"
        object.__setattr__(self, "ticker", t)

        # --- normalize date ---
        d = self.date
        if d is not None:
            d = pd.Timestamp(d)
        object.__setattr__(self, "date", d)

        if not (self.K_obs.size == self.T_obs.size == self.C_obs.size):
            raise ValueError("K_obs, T_obs, C_obs must have same length.")

        m = (
            np.isfinite(self.K_obs)
            & np.isfinite(self.T_obs)
            & np.isfinite(self.C_obs)
            & (self.K_obs > 0)
            & (self.T_obs >= 0)
            & (self.C_obs >= 0)
        )

        if not np.all(m):
            object.__setattr__(self, "K_obs", self.K_obs[m])
            object.__setattr__(self, "T_obs", self.T_obs[m])
            object.__setattr__(self, "C_obs", self.C_obs[m])