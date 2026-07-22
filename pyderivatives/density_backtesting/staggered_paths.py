
# staggered_paths.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from .forecast import ForecastDataset


def _clean_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).tz_localize(None).normalize()


def _normalize_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a forecast-date/end-date frame."""
    if frame.empty:
        return pd.DataFrame(columns=["date", "end_date"])

    missing = {"date", "end_date"}.difference(frame.columns)
    if missing:
        raise KeyError(
            "Staggered path construction requires columns "
            f"'date' and 'end_date'; missing {sorted(missing)}."
        )

    out = frame[["date", "end_date"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out["end_date"] = pd.to_datetime(
        out["end_date"], errors="coerce"
    ).dt.tz_localize(None)

    return (
        out.dropna(subset=["date", "end_date"])
        .drop_duplicates(subset=["date", "end_date"])
        .sort_values(["date", "end_date"])
        .reset_index(drop=True)
    )


def _pairs_from_dataset(
    dataset: ForecastDataset,
    *,
    model_names: Sequence[str],
    horizon: int,
) -> pd.DataFrame:
    """Return forecast/end-date pairs common to all requested models."""
    if not model_names:
        raise ValueError("model_names cannot be empty for a ForecastDataset.")

    pair_sets: list[set[tuple[pd.Timestamp, pd.Timestamp]]] = []

    for model_name in model_names:
        frame = dataset.get_model(
            model_name,
            horizon=int(horizon),
            require_nonempty=False,
        )
        pairs = _normalize_pair_frame(frame)

        if pairs.empty:
            return pd.DataFrame(columns=["date", "end_date"])

        pair_sets.append(
            {
                (_clean_date(row.date), _clean_date(row.end_date))
                for row in pairs.itertuples(index=False)
            }
        )

    common = set.intersection(*pair_sets) if pair_sets else set()

    if not common:
        return pd.DataFrame(columns=["date", "end_date"])

    return pd.DataFrame(
        sorted(common, key=lambda item: (item[0], item[1])),
        columns=["date", "end_date"],
    )


def _pairs_from_plan(plan: Any, *, horizon: Optional[int]) -> pd.DataFrame:
    """
    Extract pairs from a ClusterPlan-like object.

    A plan is useful for pre-run inspection because PlannedEvaluation already
    stores date and end_date. These paths describe the planned master sample.
    """
    evaluations = getattr(plan, "evaluations", None)
    if evaluations is None:
        raise TypeError(
            "source must be a ForecastDataset, ClusterPlan-like object, "
            "or a dataframe with date/end_date columns."
        )

    rows = []
    for item in evaluations:
        item_horizon = getattr(item, "horizon_days", horizon)
        if horizon is not None and int(item_horizon) != int(horizon):
            continue
        rows.append(
            {
                "date": getattr(item, "date"),
                "end_date": getattr(item, "end_date"),
            }
        )

    return _normalize_pair_frame(pd.DataFrame(rows))


def _source_pairs(
    source: Any,
    *,
    horizon: Optional[int],
    model_names: Optional[Sequence[str]],
) -> tuple[pd.DataFrame, int, tuple[str, ...], str]:
    if isinstance(source, ForecastDataset):
        names = tuple(model_names or source.models)
        if not names:
            raise ValueError("The ForecastDataset contains no models.")

        horizons = source.horizons
        if horizon is None:
            if len(horizons) != 1:
                raise ValueError(
                    "horizon must be supplied when the dataset contains "
                    f"multiple horizons: {horizons}."
                )
            horizon = int(horizons[0])

        return (
            _pairs_from_dataset(
                source,
                model_names=names,
                horizon=int(horizon),
            ),
            int(horizon),
            names,
            "forecast_dataset",
        )

    if isinstance(source, pd.DataFrame):
        if horizon is None:
            raise ValueError("horizon must be supplied for a dataframe source.")
        return (
            _normalize_pair_frame(source),
            int(horizon),
            tuple(model_names or ()),
            "dataframe",
        )

    plan_horizon = getattr(source, "horizon_days", horizon)
    if horizon is None:
        if plan_horizon is None:
            raise ValueError(
                "horizon could not be inferred from the plan; supply it explicitly."
            )
        horizon = int(plan_horizon)

    required = tuple(
        model_names
        or getattr(source, "required_models", ())
    )

    return (
        _pairs_from_plan(source, horizon=int(horizon)),
        int(horizon),
        required,
        "cluster_plan",
    )


@dataclass(frozen=True)
class StaggeredPath:
    path_id: int
    horizon_days: int
    pairs: pd.DataFrame
    model_names: tuple[str, ...] = ()
    source_type: str = "unknown"

    def __len__(self) -> int:
        return int(len(self.pairs))

    @property
    def first_date(self) -> Optional[pd.Timestamp]:
        return None if self.pairs.empty else pd.Timestamp(self.pairs["date"].iloc[0])

    @property
    def last_date(self) -> Optional[pd.Timestamp]:
        return None if self.pairs.empty else pd.Timestamp(self.pairs["date"].iloc[-1])

    @property
    def first_end_date(self) -> Optional[pd.Timestamp]:
        return None if self.pairs.empty else pd.Timestamp(self.pairs["end_date"].iloc[0])

    @property
    def last_end_date(self) -> Optional[pd.Timestamp]:
        return None if self.pairs.empty else pd.Timestamp(self.pairs["end_date"].iloc[-1])

    def is_non_overlapping(self, *, allow_touching: bool = False) -> bool:
        if len(self.pairs) <= 1:
            return True

        starts = pd.to_datetime(self.pairs["date"]).tolist()
        ends = pd.to_datetime(self.pairs["end_date"]).tolist()

        if allow_touching:
            return all(starts[i] >= ends[i - 1] for i in range(1, len(starts)))
        return all(starts[i] > ends[i - 1] for i in range(1, len(starts)))

    def to_frame(self) -> pd.DataFrame:
        out = self.pairs.copy()
        out.insert(0, "path_id", int(self.path_id))
        out.insert(1, "position", np.arange(1, len(out) + 1, dtype=int))
        out["horizon_days"] = int(self.horizon_days)
        out["source_type"] = self.source_type
        out["model_names"] = " | ".join(self.model_names)
        return out

    def to_dataset(
        self,
        dataset: ForecastDataset,
        *,
        model_names: Optional[Sequence[str]] = None,
    ) -> ForecastDataset:
        """Select all requested model forecasts belonging to this path."""
        names = tuple(model_names or self.model_names or dataset.models)
        wanted = {
            (_clean_date(row.date), _clean_date(row.end_date))
            for row in self.pairs.itertuples(index=False)
        }

        forecasts = []
        for forecast in dataset.forecasts:
            if forecast.model_name not in names:
                continue
            if int(forecast.horizon) != int(self.horizon_days):
                continue

            pair = (
                _clean_date(forecast.date),
                _clean_date(forecast.metadata.get("end_date", pd.NaT)),
            )
            if pair in wanted:
                forecasts.append(forecast)

        return ForecastDataset(
            forecasts=forecasts,
            errors=[],
            config=dataset.config,
            metadata={
                **dataset.metadata,
                "staggered_path_id": int(self.path_id),
                "staggered_horizon_days": int(self.horizon_days),
                "staggered_model_names": list(names),
                "staggered_source_type": self.source_type,
            },
        )

    def __repr__(self) -> str:
        if self.pairs.empty:
            span = "empty"
        else:
            span = (
                f"{self.first_date.date()} -> {self.last_date.date()}"
            )
        return (
            f"StaggeredPath(path_id={self.path_id}, n={len(self)}, "
            f"horizon_days={self.horizon_days}, span={span})"
        )


@dataclass(frozen=True)
class StaggeredPathValidation:
    valid: bool
    n_paths: int
    total_assigned: int
    unique_pairs: int
    source_pairs: int
    duplicate_pairs: int
    unused_pairs: int
    overlap_failures: tuple[int, ...]
    chronological_failures: tuple[int, ...]

    def summary(self) -> str:
        status = "PASS" if self.valid else "FAIL"
        return "\n".join(
            [
                "Staggered Path Validation",
                "=========================",
                f"Status:                 {status}",
                f"Paths:                  {self.n_paths}",
                f"Source pairs:           {self.source_pairs}",
                f"Assigned pairs:         {self.total_assigned}",
                f"Unique assigned pairs:  {self.unique_pairs}",
                f"Duplicate pairs:        {self.duplicate_pairs}",
                f"Unused pairs:           {self.unused_pairs}",
                f"Overlap failures:       {list(self.overlap_failures) or 'none'}",
                (
                    "Chronology failures:    "
                    f"{list(self.chronological_failures) or 'none'}"
                ),
            ]
        )


@dataclass(frozen=True)
class StaggeredPathCollection:
    paths: tuple[StaggeredPath, ...]
    horizon_days: int
    source_pairs: pd.DataFrame
    model_names: tuple[str, ...] = ()
    source_type: str = "unknown"
    min_obs: int = 1

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[StaggeredPath]:
        return iter(self.paths)

    def __getitem__(self, item):
        return self.paths[item]

    @property
    def n_forecasts(self) -> int:
        return int(sum(len(path) for path in self.paths))

    @property
    def path_lengths(self) -> list[int]:
        return [len(path) for path in self.paths]

    def to_frame(self) -> pd.DataFrame:
        frames = [path.to_frame() for path in self.paths]
        if not frames:
            return pd.DataFrame(
                columns=[
                    "path_id",
                    "position",
                    "date",
                    "end_date",
                    "horizon_days",
                    "source_type",
                    "model_names",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def to_csv(self, path: str | Path, **kwargs) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output, index=False, **kwargs)
        return output

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for path in self.paths:
            rows.append(
                {
                    "path_id": path.path_id,
                    "n": len(path),
                    "first_date": path.first_date,
                    "last_date": path.last_date,
                    "first_end_date": path.first_end_date,
                    "last_end_date": path.last_end_date,
                    "non_overlapping": path.is_non_overlapping(),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lengths = self.path_lengths
        validation = self.validate()

        if lengths:
            length_lines = [
                f"Mean path length:       {np.mean(lengths):.2f}",
                f"Median path length:     {np.median(lengths):.2f}",
                f"Min path length:        {min(lengths)}",
                f"Max path length:        {max(lengths)}",
            ]
        else:
            length_lines = [
                "Mean path length:       n/a",
                "Median path length:     n/a",
                "Min path length:        n/a",
                "Max path length:        n/a",
            ]

        return "\n".join(
            [
                "Staggered Path Summary",
                "======================",
                f"Source type:            {self.source_type}",
                f"Horizon days:           {self.horizon_days}",
                f"Models:                 {list(self.model_names)}",
                f"Number of paths:        {len(self.paths)}",
                f"Source forecast pairs:  {len(self.source_pairs)}",
                f"Assigned forecasts:     {self.n_forecasts}",
                *length_lines,
                f"Validation:             {'PASS' if validation.valid else 'FAIL'}",
            ]
        )

    def validate(self) -> StaggeredPathValidation:
        source = {
            (_clean_date(row.date), _clean_date(row.end_date))
            for row in self.source_pairs.itertuples(index=False)
        }

        assigned_list = [
            (_clean_date(row.date), _clean_date(row.end_date))
            for path in self.paths
            for row in path.pairs.itertuples(index=False)
        ]
        assigned = set(assigned_list)

        overlap_failures = tuple(
            path.path_id
            for path in self.paths
            if not path.is_non_overlapping()
        )

        chronological_failures = []
        for path in self.paths:
            dates = pd.to_datetime(path.pairs["date"])
            if not dates.is_monotonic_increasing:
                chronological_failures.append(path.path_id)

        duplicates = len(assigned_list) - len(assigned)
        unused = len(source.difference(assigned))

        valid = (
            duplicates == 0
            and not overlap_failures
            and not chronological_failures
        )

        return StaggeredPathValidation(
            valid=bool(valid),
            n_paths=len(self.paths),
            total_assigned=len(assigned_list),
            unique_pairs=len(assigned),
            source_pairs=len(source),
            duplicate_pairs=int(duplicates),
            unused_pairs=int(unused),
            overlap_failures=overlap_failures,
            chronological_failures=tuple(chronological_failures),
        )

    def print_paths(self, *, max_paths: Optional[int] = None) -> None:
        paths = self.paths if max_paths is None else self.paths[: int(max_paths)]

        for path in paths:
            print("=" * 80)
            print(path)
            print(path.to_frame()[["position", "date", "end_date"]].to_string(index=False))


def build_staggered_paths(
    source: Any,
    *,
    horizon: Optional[int] = None,
    model_names: Optional[Sequence[str]] = None,
    min_obs: int = 1,
) -> StaggeredPathCollection:
    """
    Build the exact Serrano-style paths used by StaggeredNonOverlap.

    Parameters
    ----------
    source:
        ForecastDataset, ClusterPlan-like object, or dataframe containing
        date and end_date.

    horizon:
        Forecast horizon. Inferred when the source contains exactly one.

    model_names:
        For ForecastDataset input, paths are built from date/end-date pairs
        common to these models. For ClusterPlan input this is descriptive,
        because the plan already represents the required-model intersection.

    min_obs:
        Minimum observations required for a path to be retained.
    """
    if int(min_obs) < 1:
        raise ValueError("min_obs must be at least 1.")

    pairs, resolved_horizon, names, source_type = _source_pairs(
        source,
        horizon=horizon,
        model_names=model_names,
    )

    if pairs.empty:
        return StaggeredPathCollection(
            paths=(),
            horizon_days=int(resolved_horizon),
            source_pairs=pairs,
            model_names=names,
            source_type=source_type,
            min_obs=int(min_obs),
        )



    work = pairs.copy()
    work["row_id"] = np.arange(len(work), dtype=int)
    
    first_start = pd.Timestamp(work["date"].min()).normalize()
    last_allowed_first_start = (
        first_start + pd.Timedelta(days=int(resolved_horizon))
    )
    
    claimed_rows: set[int] = set()
    claimed_forecast_dates: set[pd.Timestamp] = set()
    claimed_realization_dates: set[pd.Timestamp] = set()
    paths: list[StaggeredPath] = []
    
    while True:
        selected_ids: list[int] = []
        last_end: Optional[pd.Timestamp] = None
        first_path_start: Optional[pd.Timestamp] = None
    
        for row in work.itertuples(index=False):
            row_id = int(row.row_id)
    
            if row_id in claimed_rows:
                continue
    
            start = _clean_date(row.date)
            end = _clean_date(row.end_date)
    
            if start is None or end is None or end <= start:
                continue
    
            if start in claimed_forecast_dates:
                continue
    
            if end in claimed_realization_dates:
                continue
    
            if last_end is not None and start <= last_end:
                continue
    
            if first_path_start is None:
                if start >= last_allowed_first_start:
                    continue
                first_path_start = start
    
            selected_ids.append(row_id)
            last_end = end
    
        if len(selected_ids) < int(min_obs):
            break
    
        selected = (
            work[work["row_id"].isin(selected_ids)]
            .drop(columns=["row_id"])
            .sort_values(["date", "end_date"])
            .reset_index(drop=True)
        )
    
        path = StaggeredPath(
            path_id=len(paths),
            horizon_days=int(resolved_horizon),
            pairs=selected,
            model_names=names,
            source_type=source_type,
        )
    
        paths.append(path)
    
        for row_id in selected_ids:
            row = work.loc[work["row_id"] == row_id].iloc[0]
    
            claimed_rows.add(int(row_id))
            claimed_forecast_dates.add(
                _clean_date(row["date"])
            )
            claimed_realization_dates.add(
                _clean_date(row["end_date"])
            )

    return StaggeredPathCollection(
        paths=tuple(paths),
        horizon_days=int(resolved_horizon),
        source_pairs=pairs,
        model_names=names,
        source_type=source_type,
        min_obs=int(min_obs),
    )
