
# tests/staggered.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .base import DensityTest
from ..base import TestResult
from ..forecast import ForecastDataset
from ..staggered_paths import build_staggered_paths


def holm_bonferroni_decision(pvalues, alpha=0.05):
    """Apply Holm's step-down family-wise error-rate correction."""
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p)

    if valid.sum() == 0:
        return {
            "reject_any": None,
            "n_tests": 0,
            "n_rejected": 0,
            "sorted_pvalues": [],
            "critical_values": [],
            "rejected_sorted": [],
            "adjusted_pvalues_sorted": [],
        }

    sorted_p = np.sort(p[valid])
    m = len(sorted_p)
    critical = np.array([alpha / (m - k) for k in range(m)], dtype=float)

    rejected = np.zeros(m, dtype=bool)
    for k in range(m):
        if sorted_p[k] <= critical[k]:
            rejected[k] = True
        else:
            break

    adjusted = np.maximum.accumulate(
        np.array([(m - k) * sorted_p[k] for k in range(m)], dtype=float)
    )
    adjusted = np.clip(adjusted, 0.0, 1.0)

    return {
        "reject_any": bool(rejected.any()),
        "n_tests": int(m),
        "n_rejected": int(rejected.sum()),
        "sorted_pvalues": sorted_p.tolist(),
        "critical_values": critical.tolist(),
        "rejected_sorted": rejected.tolist(),
        "adjusted_pvalues_sorted": adjusted.tolist(),
    }


def _comparison_models(base_test: Any) -> Optional[tuple[str, str]]:
    model_a = getattr(base_test, "model_a", None)
    model_b = getattr(base_test, "model_b", None)

    if (
        isinstance(model_a, str)
        and model_a.strip()
        and isinstance(model_b, str)
        and model_b.strip()
    ):
        return model_a, model_b

    return None


def _available_horizons(
    dataset: ForecastDataset,
    model_names: Sequence[str],
) -> list[int]:
    horizon_sets = []

    for model_name in model_names:
        frame = dataset.get_model(model_name, require_nonempty=False)
        if frame.empty or "horizon" not in frame.columns:
            return []
        horizon_sets.append(
            {int(value) for value in frame["horizon"].dropna().unique()}
        )

    common = set.intersection(*horizon_sets) if horizon_sets else set()
    return sorted(common)


def _matching_results(
    results: Iterable[TestResult],
    *,
    expected_model_name: str,
    horizon: int,
) -> list[TestResult]:
    matched = []

    for result in results:
        result_horizon = result.metadata.get("horizon", horizon)
        try:
            same_horizon = int(result_horizon) == int(horizon)
        except (TypeError, ValueError):
            same_horizon = True

        if result.model_name == expected_model_name and same_horizon:
            matched.append(result)

    return matched


@dataclass(frozen=True)
class StaggeredNonOverlap(DensityTest):
    """
    Apply a one-model or two-model test to public, inspectable staggered paths,
    then apply Holm correction across path-level p-values.
    """

    base_test: Any = None
    min_subsample_obs: int = 5

    test_id: str = "staggered_nonoverlap"
    test_name: str = "Staggered non-overlap Holm-Bonferroni"
    category: str = "multiple_testing"
    null: str = (
        "All staggered non-overlapping paths fail to reject after "
        "Holm-Bonferroni correction."
    )
    alternative: str = (
        "At least one staggered path rejects after Holm-Bonferroni correction."
    )

    def _evaluate_family(
        self,
        dataset: ForecastDataset,
        *,
        model_names: Sequence[str],
        expected_model_name: str,
        horizon: int,
    ) -> TestResult:
        paths = build_staggered_paths(
            dataset,
            horizon=int(horizon),
            model_names=tuple(model_names),
            min_obs=int(self.min_subsample_obs),
        )

        pvalues = []
        statistics = []
        sample_sizes = []
        effect_sizes = []
        path_metadata = []

        for path in paths:
            sub_dataset = path.to_dataset(
                dataset,
                model_names=model_names,
            )

            counts = {
                model_name: len(
                    sub_dataset.get_model(
                        model_name,
                        horizon=int(horizon),
                        require_nonempty=False,
                    )
                )
                for model_name in model_names
            }

            if any(count != len(path) for count in counts.values()):
                continue

            results = self.base_test.evaluate(sub_dataset)
            matched = _matching_results(
                results,
                expected_model_name=expected_model_name,
                horizon=int(horizon),
            )
            if not matched:
                continue

            result = matched[0]
            if result.pvalue is None or not np.isfinite(result.pvalue):
                continue

            pvalues.append(float(result.pvalue))
            statistics.append(
                None if result.statistic is None else float(result.statistic)
            )
            sample_sizes.append(
                int(result.sample_size or result.metadata.get("n", len(path)))
            )
            effect_sizes.append(
                None if result.effect_size is None else float(result.effect_size)
            )

            path_metadata.append(
                {
                    "subsample": int(path.path_id),
                    "n": int(len(path)),
                    "model_counts": counts,
                    "first_date": str(path.first_date.date()),
                    "last_date": str(path.last_date.date()),
                    "first_end_date": str(path.first_end_date.date()),
                    "last_end_date": str(path.last_end_date.date()),
                    "dates": [
                        str(value.date())
                        for value in path.pairs["date"].tolist()
                    ],
                    "end_dates": [
                        str(value.date())
                        for value in path.pairs["end_date"].tolist()
                    ],
                    "pvalue": float(result.pvalue),
                    "statistic": (
                        None
                        if result.statistic is None
                        else float(result.statistic)
                    ),
                    "effect_size": (
                        None
                        if result.effect_size is None
                        else float(result.effect_size)
                    ),
                    "base_result_metadata": dict(result.metadata),
                }
            )

        holm = holm_bonferroni_decision(pvalues, alpha=self.alpha)

        common_metadata = {
            "alpha": self.alpha,
            "horizon": int(horizon),
            "base_test_id": self.base_test.test_id,
            "base_test_name": self.base_test.test_name,
            "base_test_category": getattr(self.base_test, "category", None),
            "required_models": list(model_names),
            "is_comparison_test": len(model_names) == 2,
            "min_subsample_obs": int(self.min_subsample_obs),
            "path_summary": paths.summary(),
            "path_validation": paths.validate().__dict__,
        }

        if not pvalues:
            return self.result(
                model_name=expected_model_name,
                statistic=None,
                pvalue=None,
                sample_size=None,
                metadata={
                    **common_metadata,
                    "message": "No valid staggered paths.",
                    "n_candidate_subsamples": int(len(paths)),
                },
            )

        finite_effects = [
            value
            for value in effect_sizes
            if value is not None and np.isfinite(value)
        ]

        return TestResult(
            test_id=f"staggered_{self.base_test.test_id}",
            test_name=f"Staggered {self.base_test.test_name}",
            model_name=expected_model_name,
            statistic=None,
            pvalue=float(np.min(pvalues)),
            reject=holm["reject_any"],
            category="multiple_testing",
            null=self.null,
            alternative=self.alternative,
            distribution=None,
            degrees_of_freedom=None,
            sample_size=None,
            effect_size=(
                float(np.mean(finite_effects)) if finite_effects else None
            ),
            metadata={
                **common_metadata,
                "n_candidate_subsamples": int(len(paths)),
                "n_valid_subsamples": int(len(pvalues)),
                "subsample_ns": sample_sizes,
                "mean_subsample_n": float(np.mean(sample_sizes)),
                "median_subsample_n": float(np.median(sample_sizes)),
                "min_subsample_n": int(np.min(sample_sizes)),
                "max_subsample_n": int(np.max(sample_sizes)),
                "subsample_pvalues": pvalues,
                "subsample_statistics": statistics,
                "subsample_effect_sizes": effect_sizes,
                "subsample_summary": path_metadata,
                "subsample_nonrejection_rate": float(
                    1.0 - np.mean([pvalue < self.alpha for pvalue in pvalues])
                ),
                "holm": holm,
            },
        )

    def evaluate(self, dataset: ForecastDataset) -> list[TestResult]:
        if self.base_test is None:
            raise ValueError("base_test must be supplied.")

        if dataset.to_frame().empty:
            return []

        comparison = _comparison_models(self.base_test)

        if comparison is not None:
            model_a, model_b = comparison
            expected_name = f"{model_a} vs {model_b}"

            missing = [
                name
                for name in (model_a, model_b)
                if name not in dataset.models
            ]
            if missing:
                return [
                    self.result(
                        model_name=expected_name,
                        statistic=None,
                        pvalue=None,
                        sample_size=0,
                        metadata={
                            "message": f"Missing models: {missing}",
                            "base_test_id": self.base_test.test_id,
                        },
                    )
                ]

            return [
                self._evaluate_family(
                    dataset,
                    model_names=(model_a, model_b),
                    expected_model_name=expected_name,
                    horizon=horizon,
                )
                for horizon in _available_horizons(
                    dataset,
                    (model_a, model_b),
                )
            ]

        output = []
        for model_name in dataset.models:
            for horizon in _available_horizons(dataset, (model_name,)):
                output.append(
                    self._evaluate_family(
                        dataset,
                        model_names=(model_name,),
                        expected_model_name=model_name,
                        horizon=horizon,
                    )
                )

        return output
