# tests/staggered.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .base import DensityTest
from ..base import TestResult
from ..forecast import ForecastDataset


def holm_bonferroni_decision(pvalues, alpha=0.05):
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
        }

    sorted_p = np.sort(p[valid])
    m = len(sorted_p)

    critical = np.array(
        [alpha / (m - k) for k in range(m)],
        dtype=float,
    )

    rejected = np.zeros(m, dtype=bool)

    for k in range(m):
        if sorted_p[k] <= critical[k]:
            rejected[k] = True
        else:
            break

    return {
        "reject_any": bool(rejected.any()),
        "n_tests": int(m),
        "n_rejected": int(rejected.sum()),
        "sorted_pvalues": sorted_p.tolist(),
        "critical_values": critical.tolist(),
        "rejected_sorted": rejected.tolist(),
    }


def _as_clean_date(x):
    if pd.isna(x):
        return None
    return pd.Timestamp(x).tz_localize(None).normalize()


def _build_claimed_nonoverlap_subsamples(
    df_mh: pd.DataFrame,
    *,
    horizon_days: int,
    min_obs: int,
) -> list[pd.DataFrame]:
    """
    Build Serrano-style staggered non-overlapping subsamples.

    Each row is a forecast-realization pair:
        forecast date     = date
        realization date  = end_date

    Rules:
      1. end_date must exist.
      2. Within each subsample, windows cannot overlap:
             next forecast date > previous realization date
      3. Across subsamples, forecast dates and realization dates claimed
         by earlier subsamples cannot be reused.
      4. A new subsample must start within the first horizon_days
         calendar days of the first valid forecast date.
    """
    if df_mh.empty:
        return []

    if "end_date" not in df_mh.columns:
        raise ValueError(
            "StaggeredNonOverlap requires forecast metadata column 'end_date'. "
            "Use realized_horizon_mode='calendar' when running the backtest."
        )

    df = df_mh.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.tz_localize(None)

    df = df.dropna(subset=["date", "end_date"])
    df = df.sort_values(["date", "end_date"]).reset_index(drop=True)

    if df.empty:
        return []

    df["_row_id"] = np.arange(len(df), dtype=int)

    first_start = df["date"].min().normalize()
    last_allowed_first_start = first_start + pd.Timedelta(days=int(horizon_days))

    claimed_rows: set[int] = set()
    claimed_forecast_dates: set[pd.Timestamp] = set()
    claimed_realization_dates: set[pd.Timestamp] = set()

    subsamples: list[pd.DataFrame] = []

    while True:
        sub_row_ids: list[int] = []
        last_end = None
        sub_first_start = None

        for _, row in df.iterrows():
            row_id = int(row["_row_id"])

            if row_id in claimed_rows:
                continue

            start = _as_clean_date(row["date"])
            end = _as_clean_date(row["end_date"])

            if start is None or end is None:
                continue

            if start in claimed_forecast_dates:
                continue

            if end in claimed_realization_dates:
                continue

            if end <= start:
                continue

            if last_end is not None and start <= last_end:
                continue

            if sub_first_start is None:
                if start >= last_allowed_first_start:
                    continue
                sub_first_start = start

            sub_row_ids.append(row_id)
            last_end = end

        if len(sub_row_ids) < int(min_obs):
            break

        sub = (
            df[df["_row_id"].isin(sub_row_ids)]
            .copy()
            .sort_values("date")
            .drop(columns=["_row_id"])
            .reset_index(drop=True)
        )

        subsamples.append(sub)

        for row_id in sub_row_ids:
            row = df.loc[df["_row_id"] == row_id].iloc[0]
            claimed_rows.add(int(row_id))
            claimed_forecast_dates.add(_as_clean_date(row["date"]))
            claimed_realization_dates.add(_as_clean_date(row["end_date"]))

    return subsamples


@dataclass(frozen=True)
class StaggeredNonOverlap(DensityTest):
    base_test: Any = None
    min_subsample_obs: int = 5

    test_id: str = "staggered_nonoverlap"
    test_name: str = "Staggered non-overlap Holm-Bonferroni"
    category: str = "multiple_testing"
    null: str = (
        "All staggered non-overlapping subsamples fail to reject "
        "after Holm-Bonferroni correction."
    )
    alternative: str = (
        "At least one staggered subsample rejects after "
        "Holm-Bonferroni correction."
    )

    def evaluate(self, dataset: ForecastDataset) -> list[TestResult]:
        if self.base_test is None:
            raise ValueError("base_test must be supplied.")

        out = []
        df_full = dataset.to_frame()

        if df_full.empty:
            return out

        for model in dataset.models:
            df_model = dataset.get_model(model, require_nonempty=False)

            if df_model.empty:
                continue

            for horizon in sorted(df_model["horizon"].dropna().unique()):
                horizon = int(horizon)

                df_mh = dataset.get_model(
                    model,
                    horizon=horizon,
                    require_nonempty=False,
                )

                if df_mh.empty:
                    continue

                subsamples = _build_claimed_nonoverlap_subsamples(
                    df_mh,
                    horizon_days=horizon,
                    min_obs=int(self.min_subsample_obs),
                )

                pvalues = []
                stats_ = []
                ns = []
                sub_meta = []

                for sub_idx, sub in enumerate(subsamples):
                    sub_pairs = {
                        (
                            _as_clean_date(row["date"]),
                            _as_clean_date(row["end_date"]),
                        )
                        for _, row in sub.iterrows()
                    }

                    sub_forecasts = []

                    for f in dataset.forecasts:
                        if f.model_name != model:
                            continue

                        if int(f.horizon) != horizon:
                            continue

                        f_start = _as_clean_date(f.date)
                        f_end = _as_clean_date(f.metadata.get("end_date", pd.NaT))

                        if (f_start, f_end) in sub_pairs:
                            sub_forecasts.append(f)

                    sub_dataset = ForecastDataset(
                        forecasts=sub_forecasts,
                        errors=[],
                        config=dataset.config,
                        metadata={
                            **dataset.metadata,
                            "staggered_subsample": int(sub_idx),
                            "staggered_horizon": int(horizon),
                            "staggered_method": "claimed_forecast_realization_pairs",
                        },
                    )

                    res = self.base_test.evaluate(sub_dataset)
                    res = [
                        r for r in res
                        if r.model_name == model
                        and int(r.metadata.get("horizon", horizon)) == horizon
                    ]

                    if not res:
                        continue

                    r = res[0]

                    if r.pvalue is not None and np.isfinite(r.pvalue):
                        pvalues.append(float(r.pvalue))
                        stats_.append(
                            None if r.statistic is None else float(r.statistic)
                        )
                        ns.append(
                            int(r.sample_size or r.metadata.get("n", len(sub)))
                        )
                        sub_meta.append({
                            "subsample": int(sub_idx),
                            "n": int(len(sub)),
                            "first_date": str(sub["date"].min().date()),
                            "last_date": str(sub["date"].max().date()),
                            "first_end_date": str(sub["end_date"].min().date()),
                            "last_end_date": str(sub["end_date"].max().date()),
                        })

                holm = holm_bonferroni_decision(
                    pvalues,
                    alpha=self.alpha,
                )

                if len(pvalues) == 0:
                    out.append(self.result(
                        model_name=model,
                        statistic=None,
                        pvalue=None,
                        sample_size=None,
                        metadata={
                            "horizon": int(horizon),
                            "base_test_id": self.base_test.test_id,
                            "base_test_name": self.base_test.test_name,
                            "message": "No valid staggered subsamples.",
                            "staggered_method": "claimed_forecast_realization_pairs",
                            "min_subsample_obs": int(self.min_subsample_obs),
                        },
                    ))
                    continue

                reject_flags = [p < self.alpha for p in pvalues]
                fraction_nonrejected = float(1.0 - np.mean(reject_flags))

                out.append(TestResult(
                    test_id=f"staggered_{self.base_test.test_id}",
                    test_name=f"Staggered {self.base_test.test_name}",
                    model_name=model,
                    statistic=None,
                    pvalue=float(np.min(pvalues)),
                    reject=holm["reject_any"],
                    category="multiple_testing",
                    null=self.null,
                    alternative=self.alternative,
                    distribution=None,
                    degrees_of_freedom=None,
                    sample_size=None,
                    effect_size=None,
                    metadata={
                        "alpha": self.alpha,
                        "horizon": int(horizon),
                        "base_test_id": self.base_test.test_id,
                        "base_test_name": self.base_test.test_name,
                        "staggered_method": "claimed_forecast_realization_pairs",
                    
                        "first_subsample_start_window_days": int(horizon),
                        "min_subsample_obs": int(self.min_subsample_obs),
                    
                        "n_subsamples": len(ns),
                    
                        "subsample_ns": ns,
                        "mean_subsample_n": float(np.mean(ns)),
                        "median_subsample_n": float(np.median(ns)),
                        "min_subsample_n": int(np.min(ns)),
                        "max_subsample_n": int(np.max(ns)),
                    
                        "subsample_pvalues": pvalues,
                        "subsample_statistics": stats_,
                        "subsample_summary": sub_meta,
                    
                        "subsample_nonrejection_rate": fraction_nonrejected,
                    
                        "holm": holm,
                    },
                ))

        return out