# tests/comparison.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from .base import DensityTest


def _newey_west_variance(x, max_lag: int):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    if n < 2:
        return np.nan

    xc = x - np.mean(x)

    gamma0 = np.sum(xc * xc) / n
    lrv = gamma0

    for lag in range(1, int(max_lag) + 1):
        cov = np.sum(xc[lag:] * xc[:-lag]) / n
        weight = 1.0 - lag / (max_lag + 1.0)
        lrv += 2.0 * weight * cov

    return lrv


def _comparison_stat(
    d,
    *,
    horizon: int = 1,
    correction: str = "newey-west",
    max_lag: Optional[int] = None,
):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]

    n = len(d)
    if n < 5:
        return np.nan, np.nan, n, np.nan

    mean_d = float(np.mean(d))

    if max_lag is None:
        max_lag = max(0, int(horizon) - 1)

    if correction in {"newey-west", "hac"}:
        lrv = _newey_west_variance(d, max_lag=max_lag)
    elif correction in {"none", "iid"}:
        lrv = float(np.var(d, ddof=1))
    else:
        raise ValueError("correction must be 'newey-west', 'hac', 'none', or 'iid'.")

    if not np.isfinite(lrv) or lrv <= 0:
        return np.nan, np.nan, n, mean_d

    se = np.sqrt(lrv / n)
    stat = mean_d / se
    pvalue = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))

    return float(stat), float(pvalue), n, mean_d


def _tail_weights_from_pit(pit, *, alpha_level: float, side: str):
    pit = np.asarray(pit, dtype=float)

    if side == "left":
        return (pit <= alpha_level).astype(float)

    if side == "right":
        return (pit >= 1.0 - alpha_level).astype(float)

    if side == "two-sided":
        return (
            (pit <= alpha_level / 2.0)
            | (pit >= 1.0 - alpha_level / 2.0)
        ).astype(float)

    if side == "center":
        return (
            (pit > alpha_level)
            & (pit < 1.0 - alpha_level)
        ).astype(float)

    raise ValueError("side must be 'left', 'right', 'two-sided', or 'center'.")


def _comparison_frames(dataset, model_a: str, model_b: str, *, columns):
    horizons = sorted(
        set(dataset.get_model(model_a, require_nonempty=False)["horizon"].dropna().unique())
        & set(dataset.get_model(model_b, require_nonempty=False)["horizon"].dropna().unique())
    )

    for horizon in horizons:
        df = dataset.compare(
            model_a,
            model_b,
            horizon=int(horizon),
            columns=columns,
            common_dates_only=True,
        )

        if not df.empty:
            yield int(horizon), df


@dataclass(frozen=True)
class DieboldMariano(DensityTest):
    model_a: str = ""
    model_b: str = ""
    correction: str = "newey-west"
    max_lag: Optional[int] = None

    test_id: str = "diebold_mariano"
    test_name: str = "Diebold-Mariano log-score comparison"
    category: str = "comparison"
    null: str = "Equal predictive accuracy."
    alternative: str = "Unequal predictive accuracy."

    def evaluate(self, dataset):
        out = []

        any_result = False

        for horizon, df in _comparison_frames(
            dataset,
            self.model_a,
            self.model_b,
            columns=["log_score", "horizon"],
        ):
            any_result = True

            d = df["log_score_a"].values - df["log_score_b"].values

            stat, pvalue, n, mean_d = _comparison_stat(
                d,
                horizon=horizon,
                correction=self.correction,
                max_lag=self.max_lag,
            )

            out.append(self.result(
                model_name=f"{self.model_a} vs {self.model_b}",
                statistic=None if not np.isfinite(stat) else stat,
                pvalue=None if not np.isfinite(pvalue) else pvalue,
                distribution="normal",
                sample_size=n,
                effect_size=mean_d,
                metadata={
                    "horizon": horizon,
                    "model_a": self.model_a,
                    "model_b": self.model_b,
                    "mean_score_diff": mean_d,
                    "positive_stat_means": f"{self.model_a} better",
                    "correction": self.correction,
                    "max_lag": self.max_lag,
                },
            ))

        if not any_result:
            out.append(self.result(
                model_name=f"{self.model_a} vs {self.model_b}",
                statistic=None,
                pvalue=None,
                sample_size=0,
                metadata={"message": "No common dates."},
            ))

        return out


@dataclass(frozen=True)
class AmisanoGiacomini(DensityTest):
    model_a: str = ""
    model_b: str = ""
    correction: str = "newey-west"
    max_lag: Optional[int] = None

    test_id: str = "amisano_giacomini"
    test_name: str = "Amisano-Giacomini weighted likelihood ratio test"
    category: str = "comparison"
    null: str = "Equal average weighted log score."
    alternative: str = "Unequal average weighted log score."

    def evaluate(self, dataset):
        out = []
        any_result = False

        for horizon, df in _comparison_frames(
            dataset,
            self.model_a,
            self.model_b,
            columns=["log_score", "horizon"],
        ):
            any_result = True

            d = df["log_score_a"].values - df["log_score_b"].values

            stat, pvalue, n, mean_d = _comparison_stat(
                d,
                horizon=horizon,
                correction=self.correction,
                max_lag=self.max_lag,
            )

            out.append(self.result(
                model_name=f"{self.model_a} vs {self.model_b}",
                statistic=None if not np.isfinite(stat) else stat,
                pvalue=None if not np.isfinite(pvalue) else pvalue,
                distribution="normal",
                sample_size=n,
                effect_size=mean_d,
                metadata={
                    "horizon": horizon,
                    "model_a": self.model_a,
                    "model_b": self.model_b,
                    "mean_weighted_score_diff": mean_d,
                    "weight_type": "uniform",
                    "positive_stat_means": f"{self.model_a} better",
                    "correction": self.correction,
                    "max_lag": self.max_lag,
                },
            ))

        if not any_result:
            out.append(self.result(
                model_name=f"{self.model_a} vs {self.model_b}",
                statistic=None,
                pvalue=None,
                sample_size=0,
                metadata={"message": "No common dates."},
            ))

        return out


@dataclass(frozen=True)
class TailWeightedAmisanoGiacomini(DensityTest):
    model_a: str = ""
    model_b: str = ""
    alpha_level: float = 0.10
    side: str = "left"
    weight_on: str = "model_a_pit"
    correction: str = "newey-west"
    max_lag: Optional[int] = None

    test_id: str = "tail_weighted_ag"
    test_name: str = "Tail-weighted Amisano-Giacomini test"
    category: str = "comparison"
    null: str = "Equal average tail-weighted log score."
    alternative: str = "Unequal average tail-weighted log score."

    def evaluate(self, dataset):
        out = []
        any_result = False

        for horizon, df in _comparison_frames(
            dataset,
            self.model_a,
            self.model_b,
            columns=["log_score", "pit", "horizon"],
        ):
            any_result = True

            if self.weight_on == "model_a_pit":
                pit_ref = df["pit_a"].values
            elif self.weight_on == "model_b_pit":
                pit_ref = df["pit_b"].values
            elif self.weight_on == "average_pit":
                pit_ref = 0.5 * (df["pit_a"].values + df["pit_b"].values)
            else:
                raise ValueError(
                    "weight_on must be 'model_a_pit', 'model_b_pit', or 'average_pit'."
                )

            w = _tail_weights_from_pit(
                pit_ref,
                alpha_level=float(self.alpha_level),
                side=self.side,
            )

            raw_d = df["log_score_a"].values - df["log_score_b"].values
            d = w * raw_d

            stat, pvalue, n, mean_d = _comparison_stat(
                d,
                horizon=horizon,
                correction=self.correction,
                max_lag=self.max_lag,
            )

            out.append(self.result(
                model_name=f"{self.model_a} vs {self.model_b}",
                statistic=None if not np.isfinite(stat) else stat,
                pvalue=None if not np.isfinite(pvalue) else pvalue,
                distribution="normal",
                sample_size=n,
                effect_size=mean_d,
                metadata={
                    "horizon": horizon,
                    "model_a": self.model_a,
                    "model_b": self.model_b,
                    "mean_tail_weighted_score_diff": mean_d,
                    "mean_raw_score_diff": float(np.mean(raw_d)),
                    "weight_type": "indicator",
                    "weight_on": self.weight_on,
                    "alpha_level": float(self.alpha_level),
                    "side": self.side,
                    "n_weighted_obs": int(np.sum(w > 0)),
                    "weighted_obs_share": float(np.mean(w > 0)),
                    "positive_stat_means": f"{self.model_a} better in weighted region",
                    "correction": self.correction,
                    "max_lag": self.max_lag,
                },
            ))

        if not any_result:
            out.append(self.result(
                model_name=f"{self.model_a} vs {self.model_b}",
                statistic=None,
                pvalue=None,
                sample_size=0,
                metadata={"message": "No common dates."},
            ))

        return out