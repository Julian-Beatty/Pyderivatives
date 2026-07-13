# backtest.py

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .config import EvaluationConfig
from .forecast import ForecastDataset
from .preprocessing import MarketData
from .report import DensityEvaluationReport, evaluate_dataset
from .runner import run_backtest


@dataclass(frozen=True)
class BacktestJob:
    job_id: int
    start_i: int
    end_i: int
    output_path: Optional[str] = None


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

    def plan_jobs(
        self,
        *,
        n_jobs: int,
        output_dir: Optional[str] = None,
        prefix: str = "density_backtest_part",
    ) -> list[BacktestJob]:
        if n_jobs < 1:
            raise ValueError("n_jobs must be >= 1.")

        max_dates = 0

        for model in self.models:
            n = len(model.evaluation_dates(self.market_data, self.config))
            max_dates = max(max_dates, n)

        if max_dates == 0:
            return []

        chunk = int(math.ceil(max_dates / n_jobs))
        jobs = []

        for j in range(n_jobs):
            start = j * chunk
            end = min((j + 1) * chunk, max_dates)

            if start >= end:
                continue

            output_path = None

            if output_dir is not None:
                output_path = str(
                    Path(output_dir)
                    / f"{prefix}_{j:04d}_{start:06d}_{end:06d}.pkl"
                )

            jobs.append(
                BacktestJob(
                    job_id=j,
                    start_i=start,
                    end_i=end,
                    output_path=output_path,
                )
            )

        return jobs

    def run_job(
        self,
        job: BacktestJob,
        *,
        save: bool = True,
        **kwargs,
    ) -> ForecastDataset:
        dataset = self.run(
            start_i=job.start_i,
            end_i=job.end_i,
            **kwargs,
        )

        if save and job.output_path is not None:
            dataset.save(job.output_path)

        return dataset

    @staticmethod
    def merge_jobs(paths_or_datasets) -> ForecastDataset:
        parts = []

        for x in paths_or_datasets:
            if isinstance(x, ForecastDataset):
                parts.append(x)
            else:
                parts.append(ForecastDataset.load(x))

        return ForecastDataset.combine(parts)