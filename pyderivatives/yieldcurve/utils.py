from __future__ import annotations

import re
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


def parse_maturity_to_years(col: str) -> Optional[float]:
    """
    Convert standardized columns like '1M','3M','6M','1Y','2Y','10Y' -> years.
    Return None if not a recognized maturity label.
    """
    if not isinstance(col, str):
        return None
    s = col.strip().upper()
    if s == "DATE":
        return None
    if s.endswith("M") and s[:-1].isdigit():
        return int(s[:-1]) / 12.0
    if s.endswith("Y") and s[:-1].isdigit():
        return float(int(s[:-1]))
    return None


def maturity_sort_key(label: str) -> float:
    """
    Sort maturities (1M < 3M < 6M < 1Y < 2Y < ...).
    """
    s = str(label).strip().upper()
    m = re.match(r"^(\d+)(M|Y)$", s)
    if not m:
        return float("inf")
    n = int(m.group(1))
    unit = m.group(2)
    return float(n) if unit == "M" else float(12 * n)


def detect_date_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() == "date":
            return c
    return df.columns[0]


def build_day_grid(grid_days: Tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """
    grid_days=(start,end) inclusive, in days.
    Returns tau_grid (years) and column labels like '1/365', '2/365', ...
    """
    start_d, end_d = grid_days
    if start_d < 1 or end_d < start_d:
        raise ValueError("grid_days must satisfy 1 <= start <= end.")

    days = np.arange(start_d, end_d + 1, dtype=int)
    tau_grid = days.astype(float) / 365.0
    col_labels = [f"{d}/365" for d in days]
    return tau_grid, col_labels


def make_fit_window_mask(
    taus_years: np.ndarray,
    fit_days_window: Optional[Tuple[int, int]],
) -> np.ndarray:
    """
    If fit_days_window=(lo,hi), select maturities in [lo/365, hi/365].
    Otherwise select all.
    """
    if fit_days_window is None:
        return np.ones_like(taus_years, dtype=bool)

    lo, hi = fit_days_window
    if lo < 1 or hi < lo:
        raise ValueError("fit_days_window must satisfy 1 <= lo <= hi.")
    lo_tau = lo / 365.0
    hi_tau = hi / 365.0
    return (taus_years >= lo_tau) & (taus_years <= hi_tau)
