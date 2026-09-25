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
from ..scoring import score_direction, stationarity_diagnostics


def _newey_west_variance(x, max_lag: int):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.nan

    centered = x - np.mean(x)
    lrv = np.sum(centered * centered) / n

    for lag in range(1, min(int(max_lag), n - 1) + 1):
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
            dataset.get_model(model_a, require_nonempty=False)["horizon"]
            .dropna().unique()
        )
        & set(
            dataset.get_model(model_b, require_nonempty=False)["horizon"]
            .dropna().unique()
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


def _oriented_scores(frame, score: str):
    a = frame[f"{score}_a"].to_numpy(dtype=float)
    b = frame[f"{score}_b"].to_numpy(dtype=float)
    direction = score_direction(score)
    return direction * a, direction * b, a, b, direction


def _direction_metadata(
    *,
    model_a: str,
    model_b: str,
    score_name: str,
    oriented_a,
    oriented_b,
    raw_a,
    raw_b,
    weighted_a=None,
    weighted_b=None,
):
    oriented_a = np.asarray(oriented_a, dtype=float)
    oriented_b = np.asarray(oriented_b, dtype=float)
    raw_a = np.asarray(raw_a, dtype=float)
    raw_b = np.asarray(raw_b, dtype=float)

    if weighted_a is None:
        weighted_a = oriented_a
    if weighted_b is None:
        weighted_b = oriented_b

    weighted_a = np.asarray(weighted_a, dtype=float)
    weighted_b = np.asarray(weighted_b, dtype=float)
    difference = float(np.nanmean(weighted_a) - np.nanmean(weighted_b))

    if difference > 0:
        winner, loser = model_a, model_b
    elif difference < 0:
        winner, loser = model_b, model_a
    else:
        winner, loser = None, None

    return {
        "score_name": score_name,
        "score_higher_is_better": bool(score_direction(score_name) > 0),
        "mean_raw_value_model_a": float(np.nanmean(raw_a)),
        "mean_raw_value_model_b": float(np.nanmean(raw_b)),
        "mean_oriented_score_model_a": float(np.nanmean(oriented_a)),
        "mean_oriented_score_model_b": float(np.nanmean(oriented_b)),
        "mean_oriented_score_diff_a_minus_b": float(
            np.nanmean(oriented_a) - np.nanmean(oriented_b)
        ),
        "mean_weighted_oriented_score_model_a": float(np.nanmean(weighted_a)),
        "mean_weighted_oriented_score_model_b": float(np.nanmean(weighted_b)),
        "mean_weighted_score_diff_a_minus_b": difference,
        "winner": winner,
        "loser": loser,
        "direction_rule": "Positive A-minus-B means model_a is better.",
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
    score: str = "log_score"
    correction: str = "newey-west"
    max_lag: Optional[int] = None
    block_length: Optional[int] = None
    bootstrap_reps: int = 2_000
    random_state: Optional[int] = None
    bootstrap_storage: BootstrapStorageSpec = field(
        default_factory=BootstrapStorageSpec
    )
    check_stationarity: bool = False
    stationarity_alpha: float = 0.05
    stationarity_min_obs: int = 20
    stationarity_action: str = "warn"  # warn, fail, or ignore

    def _result_from_difference(
        self,
        *,
        horizon: int,
        difference,
        metadata: dict,
        score_a=None,
        score_b=None,
    ):
        label = f"{self.model_a} vs {self.model_b}"
        difference = np.asarray(difference, dtype=float)

        stationarity = None
        assumption_met = None
        if self.check_stationarity:
            stationarity = {
                "model_a_score": stationarity_diagnostics(
                    score_a if score_a is not None else difference,
                    alpha=self.stationarity_alpha,
                    min_obs=self.stationarity_min_obs,
                ),
                "model_b_score": stationarity_diagnostics(
                    score_b if score_b is not None else difference,
                    alpha=self.stationarity_alpha,
                    min_obs=self.stationarity_min_obs,
                ),
                "score_differential": stationarity_diagnostics(
                    difference,
                    alpha=self.stationarity_alpha,
                    min_obs=self.stationarity_min_obs,
                ),
            }
            assumption_met = stationarity["score_differential"]["stationary"]

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
            "stationarity_checked": bool(self.check_stationarity),
            "stationarity_action": self.stationarity_action,
            "stationarity_assumption_met": assumption_met,
            "stationarity_diagnostics": stationarity,
            "stationarity_note": (
                "AG asymptotics require a covariance-stationary score differential; "
                "ADF and KPSS are reported as complementary diagnostics."
                if self.check_stationarity else None
            ),
        }

        invalid_for_stationarity = (
            self.check_stationarity
            and self.stationarity_action == "fail"
            and assumption_met is not True
        )
        if self.stationarity_action not in {"warn", "fail", "ignore"}:
            raise ValueError("stationarity_action must be 'warn', 'fail', or 'ignore'.")

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
                None if invalid_for_stationarity or not np.isfinite(inference["statistic"])
                else float(inference["statistic"])
            ),
            pvalue=(
                None if invalid_for_stationarity or not np.isfinite(inference["pvalue"])
                else float(inference["pvalue"])
            ),
            distribution=(
                "bootstrap"
                if self.correction in {"circular-block-bootstrap", "cbb"}
                else "normal"
            ),
            sample_size=int(inference["n"]),
            effect_size=(
                None if not np.isfinite(inference["mean_difference"])
                else float(inference["mean_difference"])
            ),
            metadata=result_metadata,
        )

    def _evaluate_score(self, dataset):
        output = []
        found = False

        for horizon, frame in _comparison_frames(
            dataset,
            self.model_a,
            self.model_b,
            columns=[self.score, "horizon"],
        ):
            found = True
            oriented_a, oriented_b, raw_a, raw_b, _ = _oriented_scores(
                frame, self.score
            )
            output.append(
                self._result_from_difference(
                    horizon=horizon,
                    difference=oriented_a - oriented_b,
                    score_a=oriented_a,
                    score_b=oriented_b,
                    metadata={
                        "horizon": horizon,
                        "model_a": self.model_a,
                        "model_b": self.model_b,
                        **_direction_metadata(
                            model_a=self.model_a,
                            model_b=self.model_b,
                            score_name=self.score,
                            oriented_a=oriented_a,
                            oriented_b=oriented_b,
                            raw_a=raw_a,
                            raw_b=raw_b,
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
                    metadata={
                        "message": "No common dates or requested score is unavailable.",
                        "score_name": self.score,
                    },
                )
            )
        return output


@dataclass(frozen=True)
class DieboldMariano(_ComparisonBase):
    test_id: str = "diebold_mariano"
    test_name: str = "Diebold-Mariano score comparison"
    category: str = "comparison"
    null: str = "Equal predictive accuracy."
    alternative: str = "Unequal predictive accuracy."

    def evaluate(self, dataset):
        return self._evaluate_score(dataset)


@dataclass(frozen=True)
class AmisanoGiacomini(_ComparisonBase):
    check_stationarity: bool = True

    test_id: str = "amisano_giacomini"
    test_name: str = "Amisano-Giacomini score comparison"
    category: str = "comparison"
    null: str = "Equal average oriented score."
    alternative: str = "Unequal average oriented score."

    def evaluate(self, dataset):
        return self._evaluate_score(dataset)


@dataclass(frozen=True)
class TailWeightedAmisanoGiacomini(_ComparisonBase):
    alpha_level: float = 0.10
    side: str = "left"
    weight_on: str = "model_a_pit"
    check_stationarity: bool = True

    test_id: str = "tail_weighted_ag"
    test_name: str = "Tail-weighted Amisano-Giacomini score comparison"
    category: str = "comparison"
    null: str = "Equal average tail-weighted oriented score."
    alternative: str = "Unequal average tail-weighted oriented score."

    def evaluate(self, dataset):
        output = []
        found = False

        for horizon, frame in _comparison_frames(
            dataset,
            self.model_a,
            self.model_b,
            columns=[self.score, "pit", "horizon"],
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

            oriented_a, oriented_b, raw_a, raw_b, _ = _oriented_scores(
                frame, self.score
            )
            weighted_a = weights * oriented_a
            weighted_b = weights * oriented_b

            output.append(
                self._result_from_difference(
                    horizon=horizon,
                    difference=weighted_a - weighted_b,
                    score_a=weighted_a,
                    score_b=weighted_b,
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
                            score_name=self.score,
                            oriented_a=oriented_a,
                            oriented_b=oriented_b,
                            raw_a=raw_a,
                            raw_b=raw_b,
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
                    metadata={
                        "message": "No common dates or requested score is unavailable.",
                        "score_name": self.score,
                    },
                )
            )

        return output
