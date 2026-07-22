# backtest.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .config import EvaluationConfig
from .forecast import ForecastDataset
from .preprocessing import MarketData
from .report import DensityEvaluationReport, evaluate_dataset
from .runner import run_backtest
from .cluster import (
    BacktestJob, ClusterBundle, build_cluster_plan,
    merge_cluster_jobs, run_cluster_job,
)


@dataclass
class DensityBacktest:
    market_data: MarketData
    models: Sequence
    config: EvaluationConfig

    def run(
        self,
        *,
        start_i: Optional[int] = None,
        end_i: Optional[int] = None,
        evaluation_plan=None,
        model_names: Optional[Sequence[str]] = None,
        progress_every: int = 25,
        verbose: bool = True,
    ) -> ForecastDataset:
        return run_backtest(
            market_data=self.market_data,
            models=self.models,
            config=self.config,
            start_i=start_i,
            end_i=end_i,
            evaluation_plan=evaluation_plan,
            model_names=model_names,
            progress_every=progress_every,
            verbose=verbose,
        )

    def evaluate(
        self,
        dataset: Optional[ForecastDataset] = None,
        *,
        tests=None,
        run_if_needed: bool = True,
        **run_kwargs,
    ) -> DensityEvaluationReport:
        if dataset is None:
            if not run_if_needed:
                raise ValueError("dataset must be supplied when run_if_needed=False.")
            dataset = self.run(**run_kwargs)

        return evaluate_dataset(dataset, tests=tests)

    def plan_jobs(self, **kwargs):
        """Backward-compatible alias for :meth:`plan_cluster`."""
        return self.plan_cluster(**kwargs)

    def run_job(
        self,
        job: BacktestJob,
        *,
        save: bool = True,
        **kwargs,
    ) -> ForecastDataset:
        dataset = self.run(
            evaluation_plan=job.evaluations,
            **kwargs,
        )

        if save and job.output_path is not None:
            dataset.save(job.output_path)

        return dataset

    def plan_cluster(self, **kwargs):
        return build_cluster_plan(self, **kwargs)

    def save_cluster_bundle(self, path, *, plan, metadata=None):
        bundle = ClusterBundle(
            backtest=self,
            plan=plan,
            metadata={} if metadata is None else dict(metadata),
        )
        return bundle.save(path)

    @staticmethod
    def run_cluster_job(bundle_path, job_id, **kwargs):
        return run_cluster_job(bundle_path, job_id, **kwargs)

    @staticmethod
    def merge_cluster_jobs(bundle_path, paths_or_dir, **kwargs):
        return merge_cluster_jobs(bundle_path, paths_or_dir, **kwargs)

    @staticmethod
    def merge_jobs(paths_or_datasets) -> ForecastDataset:
        parts = []

        for x in paths_or_datasets:
            if isinstance(x, ForecastDataset):
                parts.append(x)
            else:
                parts.append(ForecastDataset.load(x))

        return ForecastDataset.combine(parts)