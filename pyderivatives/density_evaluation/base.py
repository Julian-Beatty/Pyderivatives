# pyderivatives/density_evaluation/base.py

from dataclasses import dataclass
import numpy as np


@dataclass
class ForecastDensity:
    date: object
    horizon: int
    x_grid: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    realized: float
    model_name: str

    @property
    def pit(self):
        val = np.interp(self.realized, self.x_grid, self.cdf)
        return float(np.clip(val, 1e-10, 1 - 1e-10))

    @property
    def log_score(self):
        pdf_val = np.interp(self.realized, self.x_grid, self.pdf)
        return np.log(max(pdf_val, 1e-300))