# base.py

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class ForecastDensity:
    date: Any
    horizon: int
    x_grid: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    realized: float
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pit(self):
        val = np.interp(self.realized, self.x_grid, self.cdf)
        return float(np.clip(val, 1e-10, 1 - 1e-10))

    @property
    def log_score(self):
        pdf_val = np.interp(self.realized, self.x_grid, self.pdf)
        return float(np.log(max(pdf_val, 1e-300)))

    def scores(self, config=None) -> Dict[str, float]:
        """Return all configured proper and diagnostic density scores."""
        from .scoring import score_density
        return score_density(
            x_grid=self.x_grid,
            pdf=self.pdf,
            cdf=self.cdf,
            realized=self.realized,
            config=config,
        )


@dataclass
class EvaluationError:
    date: Any
    model_name: str
    stage: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    test_id: str
    test_name: str
    model_name: Optional[str]
    statistic: Optional[float]
    pvalue: Optional[float]
    reject: Optional[bool]

    category: str = "generic"
    null: Optional[str] = None
    alternative: Optional[str] = None
    distribution: Optional[str] = None
    degrees_of_freedom: Optional[int] = None
    sample_size: Optional[int] = None
    effect_size: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
