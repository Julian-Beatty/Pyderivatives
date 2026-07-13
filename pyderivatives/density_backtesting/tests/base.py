# tests/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..base import TestResult
from ..forecast import ForecastDataset


@dataclass(frozen=True)
class DensityTest(ABC):
    test_id: str
    test_name: str
    category: str = "generic"
    alpha: float = 0.05
    null: Optional[str] = None
    alternative: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def evaluate(self, dataset: ForecastDataset) -> list[TestResult]:
        raise NotImplementedError

    def result(
        self,
        *,
        model_name: Optional[str],
        statistic: Optional[float],
        pvalue: Optional[float],
        metadata: Optional[Dict[str, Any]] = None,
        distribution: Optional[str] = None,
        degrees_of_freedom: Optional[int] = None,
        sample_size: Optional[int] = None,
        effect_size: Optional[float] = None,
    ) -> TestResult:
        reject = None if pvalue is None else bool(pvalue < self.alpha)

        return TestResult(
            test_id=self.test_id,
            test_name=self.test_name,
            model_name=model_name,
            statistic=statistic,
            pvalue=pvalue,
            reject=reject,
            category=self.category,
            null=self.null,
            alternative=self.alternative,
            distribution=distribution,
            degrees_of_freedom=degrees_of_freedom,
            sample_size=sample_size,
            effect_size=effect_size,
            metadata={
                "alpha": self.alpha,
                **self.metadata,
                **({} if metadata is None else metadata),
            },
        )