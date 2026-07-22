
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from .base import DensityTest
from .bootstrap_inference import (
    BootstrapStorageSpec,
    store_bootstrap_distribution,
    two_sided_centered_mean_cbb,
)


def _newey_west_variance(x, max_lag: int):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan

    centered = x - np.mean(x)
    lrv = np.sum(centered * centered) / n

    for lag in range(1, int(max_lag) + 1):
        covariance = np.sum(centered[lag:] * centered[:-lag]) / n
        weight = 1.0 - lag / (max_lag + 1.0)
        lrv += 2.0 * weight * covariance

    return float(lrv)


def _comparison_inference(
    difference,
    *,
    horizon: int,
    correction: str,
    max_lag: Optional[int],
    block_length: Optional[int],
    bootstrap_reps: int,
    random_state: Optional[int],
):
    difference = np.asarray(difference, dtype=float)
    difference = difference[np.isfinite(difference)]
    n = len(difference)

    if n < 5:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "n": n,
            "mean_difference": np.nan,
            "standard_error": np.nan,
            "bootstrap": None,
        }

    mean_difference = float(np.mean(difference))

    if correction in {"circular-block-bootstrap", "cbb"}:
        resolved_block = (
            max(1, int(horizon) - 1)
            if block_length is None
            else int(block_length)
        )
        bootstrap = two_sided_centered_mean_cbb(
            difference,
            block_length=resolved_block,
            bootstrap_reps=bootstrap_reps,
            random_state=random_state,
        )
        return {
            "statistic": bootstrap.observed_statistic,
            "pvalue": bootstrap.pvalue,
            "n": bootstrap.n,
            "mean_difference": mean_difference,
            "standard_error": bootstrap.bootstrap_standard_error,
            "bootstrap": bootstrap,
        }

    if max_lag is None:
        max_lag = max(0, int(horizon) - 1)

    if correction in {"newey-west", "hac"}:
        long_run_variance = _newey_west_variance(
            difference,
            max_lag=max_lag,
        )
    elif correction in {"none", "iid"}:
        long_run_variance = float(np.var(difference, ddof=1))
    else:
        raise ValueError(
            "correction must be 'newey-west', 'hac', 'none', 'iid', "
            "'circular-block-bootstrap', or 'cbb'."
        )

    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "n": n,
            "mean_difference": mean_difference,
            "standard_error": np.nan,
            "bootstrap": None,
        }

    standard_error = float(np.sqrt(long_run_variance / n))
    statistic = float(mean_difference / standard_error)
    pvalue = float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))

    return {
        "statistic": statistic,
        "pvalue": pvalue,
        "n": n,
        "mean_difference": mean_difference,
        "standard_error": standard_error,
        "bootstrap": None,
    }


def _comparison_frames(dataset, model_a: str, model_b: str, *, columns):
    horizons = sorted(
        set(
            dataset.get_model(
                model_a,
                require_nonempty=False,
            )["horizon"].dropna().unique()
        )
        & set(
            dataset.get_model(
                model_b,
                require_nonempty=False,
            )["horizon"].dropna().unique()
        )
    )

    for horizon in horizons:
        frame = dataset.compare(
            model_a,
            model_b,
            horizon=int(horizon),
            columns=columns,
            common_dates_only=True,
        )
        if not frame.empty:
            yield int(horizon), frame


def _direction_metadata(
    *,
    model_a: str,
    model_b: str,
    score_a,
    score_b,
    weighted_a=None,
    weighted_b=None,
):
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)

    if weighted_a is None:
        weighted_a = score_a
    if weighted_b is None:
        weighted_b = score_b

    weighted_a = np.asarray(weighted_a, dtype=float)
    weighted_b = np.asarray(weighted_b, dtype=float)

    mean_a = float(np.nanmean(score_a))
    mean_b = float(np.nanmean(score_b))
    mean_weighted_a = float(np.nanmean(weighted_a))
    mean_weighted_b = float(np.nanmean(weighted_b))
    difference = float(mean_weighted_a - mean_weighted_b)

    if difference > 0:
        winner, loser = model_a, model_b
    elif difference < 0:
        winner, loser = model_b, model_a
    else:
        winner, loser = None, None

    return {
        "mean_log_score_model_a": mean_a,
        "mean_log_score_model_b": mean_b,
        "mean_raw_score_diff_a_minus_b": float(mean_a - mean_b),
        "mean_weighted_log_score_model_a": mean_weighted_a,
        "mean_weighted_log_score_model_b": mean_weighted_b,
        "mean_weighted_score_diff_a_minus_b": difference,
        "winner": winner,
        "loser": loser,
        "direction_rule": (
            "Positive A-minus-B score difference means model_a is better."
        ),
    }


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

    raise ValueError(
        "side must be 'left', 'right', 'two-sided', or 'center'."
    )


@dataclass(frozen=True)
class _ComparisonBase(DensityTest):
    model_a: str = ""
    model_b: str = ""
    correction: str = "newey-west"
    max_lag: Optional[int] = None
    block_length: Optional[int] = None
    bootstrap_reps: int = 2_000
    random_state: Optional[int] = None
    bootstrap_storage: BootstrapStorageSpec = field(
        default_factory=BootstrapStorageSpec
    )

    def _result_from_difference(
        self,
        *,
        horizon: int,
        difference,
        metadata: dict,
    ):
        label = f"{self.model_a} vs {self.model_b}"

        inference = _comparison_inference(
            difference,
            horizon=horizon,
            correction=self.correction,
            max_lag=self.max_lag,
            block_length=self.block_length,
            bootstrap_reps=self.bootstrap_reps,
            random_state=self.random_state,
        )

        result_metadata = {
            **metadata,
            "correction": self.correction,
            "max_lag": self.max_lag,
            "block_length": self.block_length,
            "bootstrap_reps": (
                self.bootstrap_reps
                if self.correction in {"circular-block-bootstrap", "cbb"}
                else None
            ),
            "bootstrap_standard_error": inference["standard_error"],
        }

        bootstrap = inference["bootstrap"]
        if bootstrap is not None:
            result_metadata = store_bootstrap_distribution(
                storage=self.bootstrap_storage,
                statistics=bootstrap.bootstrap_statistics,
                metadata=result_metadata,
                test_id=self.test_id,
                model_name=label,
                horizon=horizon,
            )

        return self.result(
            model_name=label,
            statistic=(
                None
                if not np.isfinite(inference["statistic"])
                else float(inference["statistic"])
            ),
            pvalue=(
                None
                if not np.isfinite(inference["pvalue"])
                else float(inference["pvalue"])
            ),
            distribution=(
                "bootstrap"
                if self.correction in {"circular-block-bootstrap", "cbb"}
                else "normal"
            ),
            sample_size=int(inference["n"]),
            effect_size=float(inference["mean_difference"]),
            metadata=result_metadata,
        )


@dataclass(frozen=True)
class DieboldMariano(_ComparisonBase):
    test_id: str = "diebold_mariano"
    test_name: str = "Diebold-Mariano log-score comparison"
    category: str = "comparison"
    null: str = "Equal predictive accuracy."
    alternative: str = "Unequal predictive accuracy."

    def evaluate(self, dataset):
        output = []
        found = False

        for horizon, frame in _comparison_frames(
            dataset,
            self.model_a,
            self.model_b,
            columns=["log_score", "horizon"],
        ):
            found = True
            score_a = frame["log_score_a"].to_numpy(dtype=float)
            score_b = frame["log_score_b"].to_numpy(dtype=float)

            output.append(
                self._result_from_difference(
                    horizon=horizon,
                    difference=score_a - score_b,
                    metadata={
                        "horizon": horizon,
                        "model_a": self.model_a,
                        "model_b": self.model_b,
                        **_direction_metadata(
                            model_a=self.model_a,
                            model_b=self.model_b,
                            score_a=score_a,
                            score_b=score_b,
                        ),
                    },
                )
            )

        if not found:
            output.append(
                self.result(
                    model_name=f"{self.model_a} vs {self.model_b}",
                    statistic=None,
                    pvalue=None,
                    sample_size=0,
                    metadata={"message": "No common dates."},
                )
            )

        return output


@dataclass(frozen=True)
class AmisanoGiacomini(DieboldMariano):
    test_id: str = "amisano_giacomini"
    test_name: str = "Amisano-Giacomini weighted likelihood ratio test"
    null: str = "Equal average weighted log score."
    alternative: str = "Unequal average weighted log score."


@dataclass(frozen=True)
class TailWeightedAmisanoGiacomini(_ComparisonBase):
    alpha_level: float = 0.10
    side: str = "left"
    weight_on: str = "model_a_pit"

    test_id: str = "tail_weighted_ag"
    test_name: str = "Tail-weighted Amisano-Giacomini test"
    category: str = "comparison"
    null: str = "Equal average tail-weighted log score."
    alternative: str = "Unequal average tail-weighted log score."

    def evaluate(self, dataset):
        output = []
        found = False

        for horizon, frame in _comparison_frames(
            dataset,
            self.model_a,
            self.model_b,
            columns=["log_score", "pit", "horizon"],
        ):
            found = True

            if self.weight_on == "model_a_pit":
                pit_reference = frame["pit_a"].to_numpy(dtype=float)
            elif self.weight_on == "model_b_pit":
                pit_reference = frame["pit_b"].to_numpy(dtype=float)
            elif self.weight_on == "average_pit":
                pit_reference = 0.5 * (
                    frame["pit_a"].to_numpy(dtype=float)
                    + frame["pit_b"].to_numpy(dtype=float)
                )
            else:
                raise ValueError(
                    "weight_on must be 'model_a_pit', "
                    "'model_b_pit', or 'average_pit'."
                )

            weights = _tail_weights_from_pit(
                pit_reference,
                alpha_level=float(self.alpha_level),
                side=self.side,
            )

            score_a = frame["log_score_a"].to_numpy(dtype=float)
            score_b = frame["log_score_b"].to_numpy(dtype=float)
            weighted_a = weights * score_a
            weighted_b = weights * score_b

            output.append(
                self._result_from_difference(
                    horizon=horizon,
                    difference=weighted_a - weighted_b,
                    metadata={
                        "horizon": horizon,
                        "model_a": self.model_a,
                        "model_b": self.model_b,
                        "weight_type": "indicator",
                        "weight_on": self.weight_on,
                        "alpha_level": float(self.alpha_level),
                        "side": self.side,
                        "n_weighted_obs": int(np.sum(weights > 0)),
                        "weighted_obs_share": float(np.mean(weights > 0)),
                        **_direction_metadata(
                            model_a=self.model_a,
                            model_b=self.model_b,
                            score_a=score_a,
                            score_b=score_b,
                            weighted_a=weighted_a,
                            weighted_b=weighted_b,
                        ),
                    },
                )
            )

        if not found:
            output.append(
                self.result(
                    model_name=f"{self.model_a} vs {self.model_b}",
                    statistic=None,
                    pvalue=None,
                    sample_size=0,
                    metadata={"message": "No common dates."},
                )
            )

        return output
