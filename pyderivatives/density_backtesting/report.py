# report.py

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .base import TestResult
from .forecast import ForecastDataset


@dataclass
class DensityEvaluationReport:
    dataset: ForecastDataset
    test_results: List[TestResult] = field(default_factory=list)
    adjusted_results: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(
        self,
        tests: Optional[Sequence[Any]] = None,
    ) -> "DensityEvaluationReport":
        if tests is None:
            tests = [] if self.dataset.config is None else self.dataset.config.tests

        results = []

        for test in tests:
            res = test.evaluate(self.dataset)

            if isinstance(res, list):
                results.extend(res)
            else:
                results.append(res)

        self.test_results = results

        postprocessors = []
        if self.dataset.config is not None:
            postprocessors = getattr(self.dataset.config, "postprocessors", [])

        tf = self.tests_frame()

        self.adjusted_results = {}
        for pp in postprocessors:
            self.adjusted_results[pp.__class__.__name__] = pp.adjust(tf)

        self.metadata["n_tests_run"] = len(tests)
        self.metadata["n_test_results"] = len(results)
        self.metadata["n_adjustment_methods"] = len(self.adjusted_results)

        return self

    def tests_frame(self) -> pd.DataFrame:
        rows = []

        for r in self.test_results:
            rows.append({
                "test_id": r.test_id,
                "test_name": r.test_name,
                "model": r.model_name,
                "statistic": r.statistic,
                "pvalue": r.pvalue,
                "reject": r.reject,
                "category": r.category,
                "null": r.null,
                "alternative": r.alternative,
                "distribution": r.distribution,
                "degrees_of_freedom": r.degrees_of_freedom,
                "sample_size": r.sample_size,
                "effect_size": r.effect_size,
                **r.metadata,
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, Any]:
        return {
            "scores": self.dataset.score_summary(),
            "pit": self.dataset.pit_summary(),
            "success": self.dataset.success_rate(),
            "tests": self.tests_frame(),
            "adjusted_tests": self.adjusted_results,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        return path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


def evaluate_dataset(
    dataset: ForecastDataset,
    *,
    tests: Optional[Sequence[Any]] = None,
) -> DensityEvaluationReport:
    report = DensityEvaluationReport(dataset=dataset)
    return report.evaluate(tests=tests)