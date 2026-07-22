from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from .base import DensityTest


def _clean_pit(values) -> np.ndarray:
    pit = np.asarray(values, dtype=float)
    return pit[np.isfinite(pit)]


def _model_horizon_frames(dataset):
    for model in dataset.models:
        model_frame = dataset.get_model(
            model,
            require_nonempty=False,
        )

        if model_frame.empty:
            continue

        for horizon in sorted(
            model_frame["horizon"]
            .dropna()
            .unique()
        ):
            frame = dataset.get_model(
                model,
                horizon=int(horizon),
                require_nonempty=False,
            )

            if not frame.empty:
                yield model, int(horizon), frame


def _interval_violations(
    pit,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    pit = _clean_pit(pit)

    if upper == 1.0:
        inside = (
            (pit >= lower)
            & (pit <= upper)
        )
    else:
        inside = (
            (pit >= lower)
            & (pit < upper)
        )

    return (~inside).astype(float)


def _block_violation_sums(
    violations: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, int]:
    violations = np.asarray(
        violations,
        dtype=float,
    )

    n_blocks = len(violations) // int(block_size)

    if n_blocks <= 0:
        return np.empty(
            0,
            dtype=float,
        ), 0

    trimmed = violations[
        : n_blocks * int(block_size)
    ]

    sums = trimmed.reshape(
        n_blocks,
        int(block_size),
    ).sum(axis=1)

    return sums.astype(float), n_blocks


def _binomial_orthonormal_polynomials(
    *,
    block_size: int,
    probability: float,
    maximum_order: int,
) -> np.ndarray:
    """
    Numerically construct the Krawtchouk basis by weighted
    Gram-Schmidt under Binomial(N, probability).
    """
    N = int(block_size)
    m = int(maximum_order)

    if N < 1:
        raise ValueError(
            "block_size must be positive."
        )

    if not (0.0 < probability < 1.0):
        raise ValueError(
            "probability must lie strictly between 0 and 1."
        )

    if not (1 <= m < N):
        raise ValueError(
            "Require 1 <= maximum_order < block_size."
        )

    support = np.arange(
        N + 1,
        dtype=float,
    )

    weights = stats.binom.pmf(
        support,
        N,
        probability,
    )

    monomials = np.column_stack(
        [
            support ** order
            for order in range(m + 1)
        ]
    )

    basis = np.zeros_like(
        monomials,
        dtype=float,
    )

    for order in range(m + 1):
        vector = monomials[:, order].copy()

        for previous in range(order):
            projection = np.sum(
                weights
                * vector
                * basis[:, previous]
            )

            vector -= (
                projection
                * basis[:, previous]
            )

        norm = np.sqrt(
            np.sum(
                weights
                * vector ** 2
            )
        )

        if (
            not np.isfinite(norm)
            or norm <= 1e-14
        ):
            raise RuntimeError(
                "Could not construct the binomial "
                "orthonormal basis."
            )

        basis[:, order] = vector / norm

        first_nonzero = np.flatnonzero(
            np.abs(
                basis[:, order]
            ) > 1e-12
        )

        if (
            len(first_nonzero)
            and basis[
                first_nonzero[0],
                order,
            ] < 0
        ):
            basis[:, order] *= -1.0

    return basis


def _gmm_scores(
    block_sums: np.ndarray,
    *,
    block_size: int,
    probability: float,
    moment_orders: int,
) -> np.ndarray:
    basis = _binomial_orthonormal_polynomials(
        block_size=block_size,
        probability=probability,
        maximum_order=moment_orders,
    )

    indices = np.asarray(
        block_sums,
        dtype=int,
    )

    if (
        np.any(indices < 0)
        or np.any(indices > block_size)
    ):
        raise ValueError(
            "Block sums fall outside the binomial support."
        )

    return basis[
        indices,
        1 : moment_orders + 1,
    ]


def dumitrescu_interval_gmm_statistic(
    violations,
    *,
    nominal_violation_probability: float,
    block_size: int = 25,
    moment_orders: int = 2,
    hypothesis: Literal[
        "uc",
        "ind",
        "cc",
    ] = "cc",
) -> tuple[float, int, dict]:
    violations = np.asarray(
        violations,
        dtype=float,
    )

    violations = violations[
        np.isfinite(violations)
    ]

    if hypothesis not in {
        "uc",
        "ind",
        "cc",
    }:
        raise ValueError(
            "hypothesis must be 'uc', 'ind', or 'cc'."
        )

    block_sums, H = _block_violation_sums(
        violations,
        block_size=block_size,
    )

    if H == 0:
        return np.nan, 0, {
            "message": (
                "No complete violation blocks."
            ),
        }

    if hypothesis == "uc":
        probability = float(
            nominal_violation_probability
        )
        orders = 1
        degrees_of_freedom = 1

    elif hypothesis == "ind":
        probability = float(
            np.mean(violations)
        )
        orders = int(moment_orders)
        degrees_of_freedom = orders - 1

        if degrees_of_freedom < 1:
            raise ValueError(
                "The independence test requires "
                "moment_orders >= 2."
            )

    else:
        probability = float(
            nominal_violation_probability
        )
        orders = int(moment_orders)
        degrees_of_freedom = orders

    probability = float(
        np.clip(
            probability,
            1e-10,
            1.0 - 1e-10,
        )
    )

    scores = _gmm_scores(
        block_sums,
        block_size=int(block_size),
        probability=probability,
        moment_orders=orders,
    )

    mean_scores = np.mean(
        scores,
        axis=0,
    )

    statistic = float(
        H
        * np.dot(
            mean_scores,
            mean_scores,
        )
    )

    return statistic, degrees_of_freedom, {
        "hypothesis": hypothesis,
        "block_size": int(block_size),
        "n_blocks": int(H),
        "moment_orders": int(orders),
        "nominal_violation_probability": float(
            nominal_violation_probability
        ),
        "polynomial_probability": float(
            probability
        ),
        "observed_violation_probability": float(
            np.mean(violations)
        ),
        "n_violations": int(
            np.sum(violations)
        ),
        "n_observations": int(
            len(violations)
        ),
        "used_observations": int(
            H * block_size
        ),
        "dropped_final_observations": int(
            len(violations)
            - H * block_size
        ),
        "block_sums": (
            block_sums
            .astype(int)
            .tolist()
        ),
        "mean_polynomial_scores": (
            mean_scores.tolist()
        ),
    }


@dataclass(frozen=True, kw_only=True)
class DumitrescuIntervalGMM(DensityTest):
    lower: float = 0.05
    upper: float = 0.95
    block_size: int = 25
    moment_orders: int = 2
    hypothesis: Literal[
        "uc",
        "ind",
        "cc",
    ] = "cc"

    test_id: str = "dumitrescu_interval_gmm"
    test_name: str = (
        "Dumitrescu-Hurlin-Madkour interval GMM"
    )
    category: str = "coverage"
    null: str = (
        "The selected interval-forecast "
        "validity restrictions hold."
    )
    alternative: str = (
        "At least one selected interval "
        "restriction fails."
    )

    def __post_init__(self):
        lo = float(self.lower)
        hi = float(self.upper)
        N = int(self.block_size)
        m = int(self.moment_orders)

        if not (
            0.0 <= lo < hi <= 1.0
        ):
            raise ValueError(
                "Require 0 <= lower < upper <= 1."
            )

        if N < 2:
            raise ValueError(
                "block_size must be at least 2."
            )

        if not (1 <= m < N):
            raise ValueError(
                "Require 1 <= moment_orders "
                "< block_size."
            )

        if self.hypothesis not in {
            "uc",
            "ind",
            "cc",
        }:
            raise ValueError(
                "hypothesis must be "
                "'uc', 'ind', or 'cc'."
            )

        if (
            self.hypothesis == "ind"
            and m < 2
        ):
            raise ValueError(
                "The independence test requires "
                "moment_orders >= 2."
            )

        interval = (
            f"{lo:.3f}_{hi:.3f}"
            .replace(".", "p")
        )

        object.__setattr__(
            self,
            "test_id",
            (
                "dumitrescu_interval_gmm_"
                f"{self.hypothesis}_"
                f"{interval}_N{N}_m{m}"
            ),
        )

        object.__setattr__(
            self,
            "test_name",
            (
                "Dumitrescu-Hurlin-Madkour "
                f"{self.hypothesis.upper()} "
                f"interval GMM "
                f"[{lo:.2f}, {hi:.2f}]"
            ),
        )

        nulls = {
            "uc": (
                "The unconditional violation "
                "probability equals the nominal "
                "interval violation probability."
            ),
            "ind": (
                "Violations are independent, "
                "allowing their unconditional "
                "probability to be estimated."
            ),
            "cc": (
                "Violation-block sums follow "
                "Binomial(N, alpha), implying "
                "correct conditional coverage "
                "under the paper's assumptions."
            ),
        }

        object.__setattr__(
            self,
            "null",
            nulls[self.hypothesis],
        )

    def evaluate(self, dataset):
        output = []

        nominal_probability = (
            float(self.lower)
            + 1.0
            - float(self.upper)
        )

        for (
            model,
            horizon,
            frame,
        ) in _model_horizon_frames(dataset):
            violations = _interval_violations(
                frame[
                    "pit"
                ].to_numpy(dtype=float),
                lower=float(self.lower),
                upper=float(self.upper),
            )

            (
                statistic,
                degrees_of_freedom,
                metadata,
            ) = dumitrescu_interval_gmm_statistic(
                violations,
                nominal_violation_probability=(
                    nominal_probability
                ),
                block_size=int(self.block_size),
                moment_orders=int(
                    self.moment_orders
                ),
                hypothesis=self.hypothesis,
            )

            pvalue = (
                None
                if not np.isfinite(statistic)
                else float(
                    stats.chi2.sf(
                        statistic,
                        df=degrees_of_freedom,
                    )
                )
            )

            output.append(
                self.result(
                    model_name=model,
                    statistic=(
                        None
                        if not np.isfinite(
                            statistic
                        )
                        else float(statistic)
                    ),
                    pvalue=pvalue,
                    distribution=(
                        "chi2"
                        if pvalue is not None
                        else None
                    ),
                    degrees_of_freedom=(
                        int(degrees_of_freedom)
                        if pvalue is not None
                        else None
                    ),
                    sample_size=int(
                        len(violations)
                    ),
                    effect_size=(
                        float(
                            np.mean(violations)
                            - nominal_probability
                        )
                        if len(violations)
                        else None
                    ),
                    metadata={
                        "horizon": int(horizon),
                        "lower": float(
                            self.lower
                        ),
                        "upper": float(
                            self.upper
                        ),
                        **metadata,
                    },
                )
            )

        return output
