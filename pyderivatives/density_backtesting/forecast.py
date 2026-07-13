# forecast.py

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .base import EvaluationError, ForecastDensity
from .provenance import runtime_metadata


def _stable_for_hash(obj):
    if is_dataclass(obj):
        return _stable_for_hash(asdict(obj))

    if isinstance(obj, dict):
        return {
            str(k): _stable_for_hash(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(obj, (list, tuple)):
        return [_stable_for_hash(x) for x in obj]

    return str(obj)


def stable_hash(obj) -> str:
    s = json.dumps(
        _stable_for_hash(obj),
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(s).hexdigest()[:16]


@dataclass
class ForecastDataset:
    forecasts: List[ForecastDensity] = field(default_factory=list)
    errors: List[EvaluationError] = field(default_factory=list)
    config: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, tests=None):
        from .report import evaluate_dataset
        return evaluate_dataset(self, tests=tests)

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

    def to_frame(self) -> pd.DataFrame:
        rows = []

        for f in self.forecasts:
            rows.append({
                "date": pd.Timestamp(f.date),
                "model": f.model_name,
                "horizon": int(f.horizon),
                "realized": f.realized,
                "pit": f.pit,
                "log_score": f.log_score,
                **f.metadata,
            })

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .sort_values(["model", "horizon", "date"])
            .reset_index(drop=True)
        )

    def with_metadata(self) -> "ForecastDataset":
        self.metadata = {
            **runtime_metadata(),
            **self.metadata,
        }

        if self.config is not None:
            self.metadata.setdefault("config_hash", stable_hash(self.config))

        self.metadata.setdefault("n_forecasts", len(self.forecasts))
        self.metadata.setdefault("n_errors", len(self.errors))
        self.metadata.setdefault("models", self.models)
        self.metadata.setdefault("horizons", self.horizons)

        return self

    @property
    def models(self) -> list[str]:
        df = self.to_frame()

        if df.empty:
            return []

        return sorted(
            df["model"]
            .dropna()
            .unique()
            .tolist()
        )

    @property
    def horizons(self) -> list[int]:
        df = self.to_frame()

        if df.empty or "horizon" not in df.columns:
            return []

        return sorted(
            int(h)
            for h in df["horizon"].dropna().unique().tolist()
        )

    def get_model(
        self,
        model: str,
        *,
        horizon: Optional[int] = None,
        require_nonempty: bool = True,
    ) -> pd.DataFrame:
        df = self.to_frame()

        if df.empty:
            if require_nonempty:
                raise ValueError("ForecastDataset is empty.")
            return df

        out = df[df["model"] == model]

        if horizon is not None:
            out = out[out["horizon"] == int(horizon)]

        out = (
            out.sort_values(["horizon", "date"])
            .reset_index(drop=True)
        )

        if require_nonempty and out.empty:
            msg = f"No forecasts found for model '{model}'"
            if horizon is not None:
                msg += f" at horizon={int(horizon)}"
            msg += "."
            raise ValueError(msg)

        return out

    def compare(
        self,
        model_a: str,
        model_b: str,
        *,
        horizon: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        common_dates_only: bool = True,
    ) -> pd.DataFrame:
        a = self.get_model(model_a, horizon=horizon)
        b = self.get_model(model_b, horizon=horizon)

        if columns is None:
            base_cols = [c for c in a.columns if c in b.columns]
        else:
            base_cols = ["date"] + [c for c in columns if c != "date"]

        if "date" not in base_cols:
            base_cols = ["date"] + base_cols

        if "horizon" not in base_cols and "horizon" in a.columns and "horizon" in b.columns:
            base_cols = ["horizon"] + base_cols

        base_cols = list(dict.fromkeys(base_cols))

        missing_a = [c for c in base_cols if c not in a.columns]
        missing_b = [c for c in base_cols if c not in b.columns]

        if missing_a:
            raise KeyError(f"Columns missing from model_a frame: {missing_a}")

        if missing_b:
            raise KeyError(f"Columns missing from model_b frame: {missing_b}")

        merge_keys = ["date"]
        if "horizon" in base_cols:
            merge_keys = ["date", "horizon"]

        a = a[base_cols].rename(
            columns={
                c: f"{c}_a"
                for c in base_cols
                if c not in merge_keys
            }
        )

        b = b[base_cols].rename(
            columns={
                c: f"{c}_b"
                for c in base_cols
                if c not in merge_keys
            }
        )

        how = "inner" if common_dates_only else "outer"

        out = (
            pd.merge(a, b, on=merge_keys, how=how)
            .sort_values(merge_keys)
        )

        if common_dates_only:
            out = out.dropna()

        return out.reset_index(drop=True)

    def errors_frame(self) -> pd.DataFrame:
        rows = []

        for e in self.errors:
            rows.append({
                "date": pd.Timestamp(e.date) if e.date is not None else None,
                "model": e.model_name,
                "stage": e.stage,
                "message": e.message,
                **e.metadata,
            })

        if not rows:
            return pd.DataFrame()

        sort_cols = ["model"]
        if "horizon_days" in rows[0]:
            sort_cols.append("horizon_days")
        sort_cols.extend(["date", "stage"])

        return (
            pd.DataFrame(rows)
            .sort_values([c for c in sort_cols if c in pd.DataFrame(rows).columns])
            .reset_index(drop=True)
        )

    def n_forecasts(
        self,
        model: str,
        *,
        horizon: Optional[int] = None,
    ) -> int:
        df = self.get_model(
            model,
            horizon=horizon,
            require_nonempty=False,
        )

        return int(len(df))

    def n_errors(
        self,
        model: str,
        *,
        horizon: Optional[int] = None,
    ) -> int:
        e = self.errors_frame()

        if e.empty:
            return 0

        out = e[e["model"] == model]

        if horizon is not None and "horizon_days" in out.columns:
            out = out[out["horizon_days"] == int(horizon)]

        return int(len(out))

    def failure_rate(
        self,
        model: str,
        *,
        horizon: Optional[int] = None,
    ) -> float:
        nf = self.n_forecasts(model, horizon=horizon)
        ne = self.n_errors(model, horizon=horizon)
        total = nf + ne

        return float(ne / total) if total > 0 else float("nan")

    def testable_models(
        self,
        min_obs: int = 10,
        *,
        horizon: Optional[int] = None,
    ) -> list[str]:
        out = []

        for m in self.models:
            n = len(
                self.get_model(
                    m,
                    horizon=horizon,
                    require_nonempty=False,
                )
            )

            if n >= min_obs:
                out.append(m)

        return out

    def common_dates(
        self,
        models: Sequence[str],
        *,
        horizon: Optional[int] = None,
    ) -> list[pd.Timestamp]:
        common = None

        for m in models:
            dates = set(
                self.get_model(
                    m,
                    horizon=horizon,
                )["date"]
            )

            common = dates if common is None else common & dates

        return sorted(common or [])

    def success_rate(self) -> pd.DataFrame:
        f = self.to_frame()
        e = self.errors_frame()

        if f.empty and e.empty:
            return pd.DataFrame()

        group_cols_f = ["model", "horizon"]
        group_cols_e = ["model"]

        if not e.empty and "horizon_days" in e.columns:
            e = e.rename(columns={"horizon_days": "horizon"})
            group_cols_e = ["model", "horizon"]

        forecast_counts = (
            f.groupby(group_cols_f)
            .size()
            .rename("n_forecasts")
            if not f.empty
            else pd.Series(dtype=float, name="n_forecasts")
        )

        error_counts = (
            e.groupby(group_cols_e)
            .size()
            .rename("n_errors")
            if not e.empty
            else pd.Series(dtype=float, name="n_errors")
        )

        out = pd.concat(
            [forecast_counts, error_counts],
            axis=1,
        ).fillna(0)

        out["n_total_attempted"] = out["n_forecasts"] + out["n_errors"]
        out["success_rate"] = (
            out["n_forecasts"] / out["n_total_attempted"]
        )

        return out.reset_index()

    def score_summary(self) -> pd.DataFrame:
        df = self.to_frame()

        if df.empty:
            return pd.DataFrame()

        return (
            df.groupby(["model", "horizon"])["log_score"]
            .agg(["sum", "mean", "median", "std", "count"])
            .sort_values(["horizon", "mean"], ascending=[True, False])
            .reset_index()
        )

    def pit_summary(self) -> pd.DataFrame:
        df = self.to_frame()

        if df.empty:
            return pd.DataFrame()

        return (
            df.groupby(["model", "horizon"])["pit"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

    def subset(
        self,
        *,
        models: Optional[Sequence[str]] = None,
        horizons: Optional[Sequence[int]] = None,
        start_date: Any = None,
        end_date: Any = None,
    ) -> "ForecastDataset":
        allowed_models = None if models is None else set(models)
        allowed_horizons = None if horizons is None else {int(h) for h in horizons}

        start = None if start_date is None else pd.Timestamp(start_date)
        end = None if end_date is None else pd.Timestamp(end_date)

        forecasts = []

        for f in self.forecasts:
            d = pd.Timestamp(f.date)

            if allowed_models is not None and f.model_name not in allowed_models:
                continue

            if allowed_horizons is not None and int(f.horizon) not in allowed_horizons:
                continue

            if start is not None and d < start:
                continue

            if end is not None and d > end:
                continue

            forecasts.append(f)

        errors = []

        for e in self.errors:
            d = None if e.date is None else pd.Timestamp(e.date)

            if allowed_models is not None and e.model_name not in allowed_models:
                continue

            if allowed_horizons is not None:
                h = e.metadata.get("horizon_days", None)
                if h is not None and int(h) not in allowed_horizons:
                    continue

            if d is not None:
                if start is not None and d < start:
                    continue
                if end is not None and d > end:
                    continue

            errors.append(e)

        return ForecastDataset(
            forecasts=forecasts,
            errors=errors,
            config=self.config,
            metadata={
                **self.metadata,
                "subset": True,
                "subset_models": None if models is None else list(models),
                "subset_horizons": None if horizons is None else list(horizons),
                "subset_start_date": None if start_date is None else str(start_date),
                "subset_end_date": None if end_date is None else str(end_date),
            },
        ).with_metadata()

    @classmethod
    def combine(cls, parts):
        forecasts = []
        errors = []
        config = None
        metadata = {"n_parts": len(parts)}

        for part in parts:
            forecasts.extend(part.forecasts)
            errors.extend(part.errors)

            if config is None:
                config = part.config
            elif part.config != config:
                raise ValueError(
                    "Cannot combine ForecastDataset parts with different configs."
                )

        forecasts = sorted(
            forecasts,
            key=lambda f: (
                str(f.model_name),
                int(f.horizon),
                pd.Timestamp(f.date),
            ),
        )

        return cls(
            forecasts=forecasts,
            errors=errors,
            config=config,
            metadata=metadata,
        ).with_metadata()