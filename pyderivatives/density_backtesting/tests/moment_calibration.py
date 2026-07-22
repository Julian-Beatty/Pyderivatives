from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats
from scipy.special import ndtri

from .base import DensityTest


def _clean_pit(values, eps: float = 1e-10) -> np.ndarray:
    pit = np.asarray(values, dtype=float)
    pit = pit[np.isfinite(pit)]
    return np.clip(pit, eps, 1.0 - eps)


def _model_horizon_frames(dataset):
    for model in dataset.models:
        frame = dataset.get_model(model, require_nonempty=False)
        if frame.empty:
            continue
        for horizon in sorted(frame["horizon"].dropna().unique()):
            subframe = dataset.get_model(
                model, horizon=int(horizon), require_nonempty=False
            )
            if not subframe.empty:
                yield model, int(horizon), subframe


def _biased_autocovariances(x: np.ndarray) -> np.ndarray:
    """gamma_hat(j) for j=0,...,n-1 using divisor n."""
    centered = np.asarray(x, dtype=float) - float(np.mean(x))
    n = len(centered)
    return np.array([
        float(np.dot(centered[lag:], centered[: n - lag]) / n)
        for lag in range(n)
    ])


def _lobato_f_hat(autocov: np.ndarray, power: int) -> float:
    # Sum over negative and positive lags:
    # gamma(0)^k + 2 * sum_{j=1}^{n-1} gamma(j)^k.
    value = autocov[0] ** power
    if len(autocov) > 1:
        value += 2.0 * float(np.sum(autocov[1:] ** power))
    return float(value)


def lobato_velasco_statistic(values) -> tuple[float, dict]:
    """
    Generalized skewness-kurtosis statistic of Lobato and Velasco (2004).

    Tests the one-dimensional marginal Gaussian restrictions:
    mu_3 = 0 and mu_4 = 3 * mu_2^2,
    while allowing serial correlation under a stationary Gaussian-process null.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan, {"message": "Too few observations."}

    centered = x - float(np.mean(x))
    m2 = float(np.mean(centered ** 2))
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))

    autocov = _biased_autocovariances(x)
    f3 = _lobato_f_hat(autocov, 3)
    f4 = _lobato_f_hat(autocov, 4)

    if not np.isfinite(f3) or not np.isfinite(f4) or f3 <= 0 or f4 <= 0:
        return np.nan, {
            "message": "Non-positive Lobato-Velasco variance estimate.",
            "f3_hat": float(f3),
            "f4_hat": float(f4),
        }

    skew_component = n * (m3 ** 2) / (6.0 * f3)
    kurt_component = n * ((m4 - 3.0 * m2 ** 2) ** 2) / (24.0 * f4)
    statistic = float(skew_component + kurt_component)

    return statistic, {
        "mean": float(np.mean(x)),
        "variance_mle": m2,
        "skewness": float(m3 / max(m2, 1e-15) ** 1.5),
        "kurtosis": float(m4 / max(m2, 1e-15) ** 2),
        "f3_hat": float(f3),
        "f4_hat": float(f4),
        "skew_component": float(skew_component),
        "kurtosis_component": float(kurt_component),
    }


@dataclass(frozen=True)
class LobatoVelasco(DensityTest):
    use_normal_scores: bool = True

    test_id: str = "lobato_velasco"
    test_name: str = "Lobato-Velasco serial-correlation-robust normality"
    category: str = "calibration"
    null: str = (
        "The transformed PIT series is a stationary Gaussian process; "
        "in particular its marginal skewness is zero and kurtosis is three."
    )
    alternative: str = (
        "The transformed PIT marginal has nonzero skewness or non-Gaussian kurtosis."
    )

    def evaluate(self, dataset):
        output = []
        for model, horizon, frame in _model_horizon_frames(dataset):
            pit = _clean_pit(frame["pit"].to_numpy(dtype=float))
            values = ndtri(pit) if self.use_normal_scores else pit
            n = len(values)
            statistic, metadata = lobato_velasco_statistic(values)

            pvalue = (
                None if not np.isfinite(statistic)
                else float(stats.chi2.sf(statistic, df=2))
            )
            output.append(self.result(
                model_name=model,
                statistic=None if not np.isfinite(statistic) else statistic,
                pvalue=pvalue,
                distribution="chi2" if pvalue is not None else None,
                degrees_of_freedom=2 if pvalue is not None else None,
                sample_size=n,
                metadata={
                    "horizon": horizon,
                    "n": n,
                    "transform": "inverse_normal_pit" if self.use_normal_scores else "pit",
                    **metadata,
                },
            ))
        return output


def _qs_weight(x: float) -> float:
    if x == 0.0:
        return 1.0
    z = 6.0 * np.pi * x / 5.0
    return float(
        25.0 / (12.0 * np.pi ** 2 * x ** 2)
        * (np.sin(z) / z - np.cos(z))
    )


def _kernel_weight(lag: int, bandwidth: float, kernel: str) -> float:
    if lag == 0:
        return 1.0
    x = lag / bandwidth
    if kernel == "bartlett":
        return max(0.0, 1.0 - x) if lag <= bandwidth else 0.0
    if kernel == "qs":
        return _qs_weight(x)
    raise ValueError("kernel must be 'qs' or 'bartlett'.")


def _default_bandwidth(n: int, kernel: str) -> float:
    # Stable rule-of-thumb defaults. Users may supply bandwidth explicitly.
    if kernel == "qs":
        return max(1.0, 1.3221 * n ** 0.2)
    return max(1.0, np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def long_run_covariance(
    scores: np.ndarray,
    *,
    kernel: str = "qs",
    bandwidth: Optional[float] = None,
    demean: bool = False,
) -> tuple[np.ndarray, float]:
    """
    HAC long-run covariance.

    For Knüppel's null-based estimator, demean=False is intentional because
    each score has expectation zero under H0.
    """
    x = np.asarray(scores, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, _ = x.shape
    if demean:
        x = x - np.mean(x, axis=0, keepdims=True)

    bw = float(_default_bandwidth(n, kernel) if bandwidth is None else bandwidth)
    if bw <= 0:
        raise ValueError("bandwidth must be positive.")

    omega = (x.T @ x) / n
    max_lag = n - 1 if kernel == "qs" else min(n - 1, int(np.floor(bw)))
    for lag in range(1, max_lag + 1):
        weight = _kernel_weight(lag, bw, kernel)
        if abs(weight) < 1e-12:
            continue
        gamma = (x[lag:].T @ x[:-lag]) / n
        omega += weight * (gamma + gamma.T)

    omega = (omega + omega.T) / 2.0
    return omega, bw


def _uniform_raw_moment(order: int) -> float:
    # Y = sqrt(12) * (U - 1/2) ~ Uniform(-sqrt(3), sqrt(3)).
    if order % 2 == 1:
        return 0.0
    return float((3.0 ** (order / 2.0)) / (order + 1.0))


def _wald(moment_diff: np.ndarray, omega: np.ndarray, n: int) -> float:
    inverse = np.linalg.pinv(omega, hermitian=True)
    return float(n * moment_diff.T @ inverse @ moment_diff)


def knuppel_statistic(
    pit,
    *,
    moments: Sequence[int] = (1, 2, 3, 4),
    split_odd_even: bool = True,
    kernel: str = "qs",
    bandwidth: Optional[float] = None,
) -> tuple[float, dict]:
    u = _clean_pit(pit)
    n = len(u)
    orders = tuple(sorted({int(order) for order in moments}))
    if n < max(20, 5 * len(orders)):
        return np.nan, {"message": "Too few observations."}
    if not orders or min(orders) < 1:
        raise ValueError("moments must contain positive integers.")

    y = np.sqrt(12.0) * (u - 0.5)
    expected = np.array([_uniform_raw_moment(r) for r in orders])
    powers = np.column_stack([y ** r for r in orders])
    empirical = np.mean(powers, axis=0)
    diff = empirical - expected
    scores = powers - expected[None, :]

    if split_odd_even and any(r % 2 for r in orders) and any(r % 2 == 0 for r in orders):
        statistic = 0.0
        block_details = {}
        used_bw = None
        for label, selector in (
            ("odd", np.array([r % 2 == 1 for r in orders])),
            ("even", np.array([r % 2 == 0 for r in orders])),
        ):
            omega, used_bw = long_run_covariance(
                scores[:, selector],
                kernel=kernel,
                bandwidth=bandwidth,
                demean=False,
            )
            component = _wald(diff[selector], omega, n)
            statistic += component
            block_details[f"{label}_component"] = float(component)
            block_details[f"{label}_condition_number"] = float(np.linalg.cond(omega))
        method = "alpha0_split_odd_even"
    else:
        omega, used_bw = long_run_covariance(
            scores, kernel=kernel, bandwidth=bandwidth, demean=False
        )
        statistic = _wald(diff, omega, n)
        block_details = {"condition_number": float(np.linalg.cond(omega))}
        method = "joint_alpha"

    return float(statistic), {
        "moments": list(orders),
        "expected_raw_moments": expected.tolist(),
        "empirical_raw_moments": empirical.tolist(),
        "moment_differences": diff.tolist(),
        "spit_mean": float(np.mean(y)),
        "spit_variance_mle": float(np.mean(y ** 2) - np.mean(y) ** 2),
        "kernel": kernel,
        "bandwidth": float(used_bw),
        "covariance_centering": "null_zero_not_sample_demeaned",
        "variant": method,
        **block_details,
    }


@dataclass(frozen=True)
class KnuppelRawMoments(DensityTest):
    moments: Sequence[int] = field(default_factory=lambda: (1, 2, 3, 4))
    split_odd_even: bool = True
    kernel: str = "qs"
    bandwidth: Optional[float] = None

    test_id: str = "knuppel_raw_moments"
    test_name: str = "Knüppel standardized-PIT raw-moments calibration"
    category: str = "calibration"
    null: str = (
        "The selected raw moments of standardized PITs equal their "
        "Uniform(-sqrt(3), sqrt(3)) values."
    )
    alternative: str = "At least one selected standardized-PIT raw moment is incorrect."

    def evaluate(self, dataset):
        output = []
        df = len(tuple(sorted({int(r) for r in self.moments})))
        for model, horizon, frame in _model_horizon_frames(dataset):
            pit = _clean_pit(frame["pit"].to_numpy(dtype=float))
            statistic, metadata = knuppel_statistic(
                pit,
                moments=self.moments,
                split_odd_even=self.split_odd_even,
                kernel=self.kernel,
                bandwidth=self.bandwidth,
            )
            pvalue = (
                None if not np.isfinite(statistic)
                else float(stats.chi2.sf(statistic, df=df))
            )
            output.append(self.result(
                model_name=model,
                statistic=None if not np.isfinite(statistic) else statistic,
                pvalue=pvalue,
                distribution="chi2" if pvalue is not None else None,
                degrees_of_freedom=df if pvalue is not None else None,
                sample_size=len(pit),
                metadata={"horizon": horizon, "n": len(pit), **metadata},
            ))
        return output
