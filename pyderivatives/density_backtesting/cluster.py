from __future__ import annotations

import copy
import os
import pickle
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence
from pathlib import PurePosixPath

import pandas as pd

from .forecast import ForecastDataset, stable_hash
from .models import RawRNDModel, PhysicalDensityModel, TransformRNDModel, _select_maturity_index


def _clean_date(x):
    return pd.Timestamp(x).tz_localize(None).normalize()


@dataclass(frozen=True)
class PlannedEvaluation:
    date: Any
    horizon_days: int
    target_maturity: float
    T_actual: float
    realized_horizon_days: int
    end_date: Any


@dataclass(frozen=True)
class BacktestJob:
    job_id: int
    evaluations: tuple[PlannedEvaluation, ...]
    output_path: Optional[str] = None

    @property
    def evaluation_dates(self):
        return tuple(x.date for x in self.evaluations)


@dataclass(frozen=True)
class ClusterPlan:
    reference_model: str
    required_models: tuple[str, ...]
    horizon_days: int
    evaluations: tuple[PlannedEvaluation, ...]
    jobs: tuple[BacktestJob, ...]
    plan_hash: str
    evaluation_strategy: str

    @property
    def n_jobs(self) -> int:
        return len(self.jobs)


@dataclass(frozen=True)
class ClusterRunStatus:
    expected_job_ids: tuple[int, ...]
    completed_job_ids: tuple[int, ...]
    missing_job_ids: tuple[int, ...]
    invalid_files: tuple[str, ...]
    duplicate_job_ids: tuple[int, ...]

    @property
    def n_expected(self) -> int:
        return len(self.expected_job_ids)

    @property
    def n_completed(self) -> int:
        return len(self.completed_job_ids)

    @property
    def is_complete(self) -> bool:
        return (
            not self.missing_job_ids
            and not self.invalid_files
            and not self.duplicate_job_ids
        )

    def summary(self) -> str:
        def fmt(values):
            return ", ".join(map(str, values)) if values else "none"

        return "\n".join([
            f"Expected jobs : {self.n_expected}",
            f"Completed     : {self.n_completed}",
            f"Missing       : {fmt(self.missing_job_ids)}",
            f"Invalid files : {fmt(self.invalid_files)}",
            f"Duplicate IDs : {fmt(self.duplicate_job_ids)}",
        ])


@dataclass
class ClusterBundle:
    backtest: Any
    plan: ClusterPlan
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


def _reference_info(model, market_data, date):
    if isinstance(model, RawRNDModel):
        return model._rnd_by_date(market_data)[_clean_date(date)]
    if isinstance(model, PhysicalDensityModel):
        return model._physical_by_date(market_data)[_clean_date(date)]
    if isinstance(model, TransformRNDModel):
        return model._rnd_by_date(market_data)[_clean_date(date)]
    raise TypeError(
        "The reference model must be RawRNDModel, PhysicalDensityModel, "
        "or TransformRNDModel so its maturity can be planned without fitting."
    )


def build_cluster_plan(
    backtest,
    *,
    n_jobs: int,
    reference_model: Optional[str] = None,
    required_models: Optional[Sequence[str]] = None,
    output_dir: Optional[str] = None,
    prefix: str = "density_backtest_part",
) -> ClusterPlan:
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    config = backtest.config
    config.validate()
    horizons = config.resolved_horizons()
    if len(horizons) != 1:
        
        raise ValueError(
    "Precomputed cluster planning currently requires exactly one horizon."
)
        
    horizon = int(horizons[0])

    model_by_name = {m.name: m for m in backtest.models}
    reference_model = reference_model or config.path_reference_model
    if reference_model not in model_by_name:
        raise ValueError(f"Unknown reference model: {reference_model!r}.")
    required = tuple(required_models or tuple(model_by_name))
    missing = [m for m in required if m not in model_by_name]
    if missing:
        raise ValueError(f"Required models are not present: {missing}.")

    model_date_sets = {
        name: {_clean_date(d) for d in model_by_name[name].evaluation_dates(backtest.market_data, config)}
        for name in required
    }
    candidates = sorted(set.intersection(*model_date_sets.values())) if model_date_sets else []
    ref_model = model_by_name[reference_model]
    target = config.target_maturity_for_horizon(horizon)

    feasible = []
    for date in candidates:
        try:
            info = _reference_info(ref_model, backtest.market_data, date)
            if not info.get("success", True):
                continue
            j, T_actual = _select_maturity_index(
                info, target, config.maturity_match_tol, config=config
            )
            if j is None:
                continue
            realized_h = int(round(365.0 * float(T_actual)))
            realized, end_date = backtest.market_data.realized_after(
                date,
                realized_h,
                mode=config.realized_horizon_mode,
                tolerance_days=config.realized_match_tol_days,
            )
            if realized is None or end_date is None:
                continue
            feasible.append(PlannedEvaluation(
                date=date,
                horizon_days=horizon,
                target_maturity=float(target),
                T_actual=float(T_actual),
                realized_horizon_days=realized_h,
                end_date=_clean_date(end_date),
            ))
        except Exception:
            continue



    selected = []
    last_end = None
    
    feasible = sorted(
        feasible,
        key=lambda x: _clean_date(x.date),
    )
    
    if config.evaluation_strategy == "all":
        # Keep every feasible forecast date.
        # StaggeredNonOverlap will construct the non-overlapping
        # subsamples later during testing.
        selected = feasible
    
    elif config.evaluation_strategy in {"shared_path", "single_path"}:
        # Construct one greedy globally non-overlapping path.
        selected = []
        last_end = None
    
        for item in feasible:
            start = _clean_date(item.date)
    
            if last_end is not None and start <= last_end:
                continue
    
            selected.append(item)
            last_end = _clean_date(item.end_date)
    
    else:
        raise ValueError(
            "Unsupported evaluation_strategy for cluster planning: "
            f"{config.evaluation_strategy!r}"
        )
    
    if not selected:
        raise RuntimeError(
            "No feasible evaluations remained after applying "
            f"evaluation_strategy={config.evaluation_strategy!r}."
        )
        
    

    # Treat n_jobs as the requested number of nonempty jobs, capped only by
    # the number of planned evaluations. Distribute the remainder evenly so
    # job sizes differ by at most one evaluation.
    actual_n_jobs = min(int(n_jobs), len(selected)) if selected else 0
    jobs = []

    if actual_n_jobs > 0:
        base_size, remainder = divmod(len(selected), actual_n_jobs)
        start = 0

        for job_id in range(actual_n_jobs):
            size = base_size + (1 if job_id < remainder else 0)
            chunk = selected[start:start + size]
            start += size

            output_path = None
            if output_dir is not None:
                output_path = str(Path(output_dir) / f"{prefix}_{job_id:04d}.pkl")
                output_path = str(
                                    PurePosixPath(output_dir)
                                    / f"{prefix}_{job_id:04d}.pkl"
                                )

            jobs.append(BacktestJob(job_id, tuple(chunk), output_path))

    payload = {
        "reference_model": reference_model,
        "required_models": required,
        "horizon": horizon,
        "evaluation_strategy": config.evaluation_strategy,
        "evaluations": [x.__dict__ for x in selected],
    }
    print("Evaluation strategy:", config.evaluation_strategy)
    print("Feasible:", len(feasible))
    print("Selected:", len(selected))
    return ClusterPlan(
        reference_model=reference_model,
        required_models=required,
        horizon_days=horizon,
        evaluations=tuple(selected),
        jobs=tuple(jobs),
        plan_hash=stable_hash(payload),
        evaluation_strategy=config.evaluation_strategy,
    )

def run_cluster_job(bundle_path, job_id: int, *, overwrite=False, verbose=True, progress_every=25):
    bundle = ClusterBundle.load(bundle_path)
    job_id = int(job_id)

    if job_id < 0 or job_id >= bundle.plan.n_jobs:
        if verbose:
            print(
                f"[Worker {job_id}] No work assigned; "
                f"plan contains {bundle.plan.n_jobs} jobs. Exiting successfully.",
                flush=True,
            )
        return None

    job = bundle.plan.jobs[job_id]
    if job.job_id != job_id:
        jobs = {j.job_id: j for j in bundle.plan.jobs}
        if job_id not in jobs:
            if verbose:
                print(
                    f"[Worker {job_id}] No work assigned. Exiting successfully.",
                    flush=True,
                )
            return None
        job = jobs[job_id]

    if job.output_path is None:
        raise ValueError("The selected job has no output_path.")

    output = Path(job.output_path)
    if output.exists() and not overwrite:
        if verbose:
            print(
                f"[Worker {job_id}] Output already exists: {output}. Skipping.",
                flush=True,
            )
        return output

    worker_config = copy.copy(bundle.backtest.config)
    object.__setattr__(worker_config, "evaluation_strategy", "all")
    object.__setattr__(worker_config, "date_alignment", "model_specific")

    dataset = bundle.backtest.__class__(
        market_data=bundle.backtest.market_data,
        models=bundle.backtest.models,
        config=worker_config,
    ).run(
        evaluation_plan=job.evaluations,
        progress_every=progress_every,
        verbose=verbose,
    )
    dataset.config = bundle.backtest.config
    dataset.metadata.update({
        "cluster_plan_hash": bundle.plan.plan_hash,
        "cluster_job_id": int(job.job_id),
        "cluster_job_status": "complete",
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "planned_dates": [str(_clean_date(x.date).date()) for x in job.evaluations],
    })
    dataset.with_metadata()

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")

    try:
        dataset.save(tmp)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    return output


def inspect_cluster_run(bundle_path, paths_or_dir) -> ClusterRunStatus:
    bundle = ClusterBundle.load(bundle_path)

    if isinstance(paths_or_dir, (str, Path)):
        root = Path(paths_or_dir)
        if root.is_dir():
            paths = sorted(root.glob("*.pkl"))
        elif root.exists():
            paths = [root]
        else:
            paths = []
    else:
        paths = [Path(p) for p in paths_or_dir]

    expected_ids = tuple(sorted(j.job_id for j in bundle.plan.jobs))
    completed_ids = []
    invalid_files = []

    for path in paths:
        try:
            part = ForecastDataset.load(path)
            if part.metadata.get("cluster_plan_hash") != bundle.plan.plan_hash:
                raise ValueError("plan hash mismatch")
            raw_id = part.metadata.get("cluster_job_id")
            if raw_id is None:
                raise ValueError("missing cluster_job_id")
            job_id = int(raw_id)
            if job_id not in expected_ids:
                raise ValueError(f"unexpected cluster_job_id={job_id}")
            if part.metadata.get("cluster_job_status") != "complete":
                raise ValueError("job status is not complete")
            completed_ids.append(job_id)
        except Exception as exc:
            invalid_files.append(f"{path}: {exc}")

    counts = {}
    for job_id in completed_ids:
        counts[job_id] = counts.get(job_id, 0) + 1

    duplicate_ids = tuple(sorted(k for k, v in counts.items() if v > 1))
    unique_completed = tuple(sorted(counts))
    missing_ids = tuple(sorted(set(expected_ids) - set(unique_completed)))

    return ClusterRunStatus(
        expected_job_ids=expected_ids,
        completed_job_ids=unique_completed,
        missing_job_ids=missing_ids,
        invalid_files=tuple(invalid_files),
        duplicate_job_ids=duplicate_ids,
    )

def merge_cluster_jobs(bundle_path, paths_or_dir, *, require_complete=True, verbose=True) -> ForecastDataset:
    bundle = ClusterBundle.load(bundle_path)
    status = inspect_cluster_run(bundle_path, paths_or_dir)

    if verbose:
        print(status.summary(), flush=True)

    if status.invalid_files:
        raise ValueError(
            "Invalid cluster part files were found:\n"
            + "\n".join(status.invalid_files)
        )

    if status.duplicate_job_ids:
        raise ValueError(
            f"Duplicate cluster job outputs found: {list(status.duplicate_job_ids)}"
        )

    if require_complete and status.missing_job_ids:
        raise ValueError(
            f"Missing cluster jobs: {list(status.missing_job_ids)}"
        )

    if isinstance(paths_or_dir, (str, Path)) and Path(paths_or_dir).is_dir():
        paths = sorted(Path(paths_or_dir).glob("*.pkl"))
    else:
        paths = list(paths_or_dir)

    parts = []
    for path in paths:
        part = ForecastDataset.load(path)
        if part.metadata.get("cluster_plan_hash") == bundle.plan.plan_hash:
            parts.append(part)

    if not parts:
        raise ValueError("No valid cluster parts were found to merge.")

    missing = list(status.missing_job_ids)
    merged = ForecastDataset.combine(parts)
    required = set(bundle.plan.required_models)
    planned = {_clean_date(x.date) for x in bundle.plan.evaluations}
    success_by_model = {
        m: {_clean_date(d) for d in merged.get_model(m, horizon=bundle.plan.horizon_days, require_nonempty=False)["date"]}
        for m in required
    }
    common = set.intersection(*success_by_model.values()) if success_by_model else set()
    keep = planned & common
    merged.forecasts = [
        f for f in merged.forecasts
        if f.model_name in required
        and int(f.horizon) == bundle.plan.horizon_days
        and _clean_date(f.date) in keep
    ]
    merged.metadata.update({
        "cluster_plan_hash": bundle.plan.plan_hash,
        "cluster_merge_complete": not missing,
        "missing_job_ids": missing,
        "planned_path_dates": len(planned),
        "complete_path_dates": len(keep),
        "path_reference_model": bundle.plan.reference_model,
        "path_required_models": list(bundle.plan.required_models),
        "evaluation_strategy_applied": "precomputed_shared_path",
    })
    return merged.with_metadata()
