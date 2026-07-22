from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class BootstrapStorageSpec:
    """
    Controls where complete bootstrap sampling distributions are stored.

    mode="none"
        Do not retain the full sampling distribution.

    mode="metadata"
        Store NumPy arrays directly in TestResult.metadata. This makes the
        saved report self-contained, but report pickle files may be large.

    mode="external"
        Store each sampling distribution in a compressed NPZ file and retain
        the path in TestResult.metadata.
    """

    mode: str = "metadata"
    output_dir: Optional[str] = None
    prefix: str = "density_bootstrap"

    def validate(self) -> None:
        if self.mode not in {"none", "metadata", "external"}:
            raise ValueError(
                "mode must be 'none', 'metadata', or 'external'."
            )

        if self.mode == "external" and not self.output_dir:
            raise ValueError(
                "output_dir is required for external storage."
            )


@dataclass(frozen=True)
class BootstrapInferenceResult:
    observed_statistic: float
    pvalue: float
    bootstrap_statistics: np.ndarray
    bootstrap_standard_error: Optional[float]
    n: int
    block_length: int
    bootstrap_reps: int
    centered: bool
    null_method: str


def clean_finite(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def circular_block_indices(
    n: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive.")

    if int(block_length) < 1:
        raise ValueError("block_length must be positive.")

    block_length = min(int(block_length), n)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_length, dtype=int)

    return (
        (starts[:, None] + offsets[None, :]) % n
    ).reshape(-1)[:n]


def circular_block_bootstrap_statistics(
    null_series,
    *,
    statistic_fn: Callable[[np.ndarray], float],
    block_length: int,
    bootstrap_reps: int,
    random_state: Optional[int],
) -> np.ndarray:
    x = clean_finite(null_series)
    n = len(x)

    if n < 5:
        raise ValueError(
            "At least five finite observations are required."
        )

    if int(bootstrap_reps) < 100:
        raise ValueError(
            "bootstrap_reps must be at least 100."
        )

    rng = np.random.default_rng(random_state)
    output = np.empty(int(bootstrap_reps), dtype=float)

    for replication in range(int(bootstrap_reps)):
        indices = circular_block_indices(
            n,
            block_length=block_length,
            rng=rng,
        )

        output[replication] = float(
            statistic_fn(x[indices])
        )

    return output


def two_sided_centered_mean_cbb(
    values,
    *,
    block_length: int,
    bootstrap_reps: int = 2_000,
    random_state: Optional[int] = None,
) -> BootstrapInferenceResult:
    x = clean_finite(values)
    n = len(x)

    if n < 5:
        raise ValueError(
            "At least five finite observations are required."
        )

    observed_mean = float(np.mean(x))
    null_series = x - observed_mean

    bootstrap_means = circular_block_bootstrap_statistics(
        null_series,
        statistic_fn=lambda sample: float(np.mean(sample)),
        block_length=block_length,
        bootstrap_reps=bootstrap_reps,
        random_state=random_state,
    )

    bootstrap_se = float(
        np.std(bootstrap_means, ddof=1)
    )

    statistic = (
        float(observed_mean / bootstrap_se)
        if np.isfinite(bootstrap_se) and bootstrap_se > 0
        else np.nan
    )

    pvalue = float(
        (
            1
            + np.sum(
                np.abs(bootstrap_means)
                >= abs(observed_mean)
            )
        )
        / (len(bootstrap_means) + 1)
    )

    return BootstrapInferenceResult(
        observed_statistic=statistic,
        pvalue=pvalue,
        bootstrap_statistics=bootstrap_means,
        bootstrap_standard_error=bootstrap_se,
        n=n,
        block_length=min(int(block_length), n),
        bootstrap_reps=int(bootstrap_reps),
        centered=True,
        null_method="centered_score_differential",
    )


def uniform_rank_null(
    pit,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    u = clean_finite(pit)
    n = len(u)

    if n == 0:
        return u

    ranks = rankdata(
        u,
        method="average",
    )

    null_u = (ranks - 0.5) / n

    return np.clip(
        null_u,
        eps,
        1.0 - eps,
    )


def _slug(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() else "_"
        for character in str(value)
    )

    return "_".join(
        part
        for part in cleaned.split("_")
        if part
    )


def store_bootstrap_distribution(
    *,
    storage: BootstrapStorageSpec,
    statistics: np.ndarray,
    metadata: dict,
    test_id: str,
    model_name: str,
    horizon: int,
    extra_arrays: Optional[
        dict[str, np.ndarray]
    ] = None,
    observed_statistic: Optional[float] = None,
) -> dict:
    storage.validate()

    values = np.asarray(
        statistics,
        dtype=float,
    ).reshape(-1)

    values = values[np.isfinite(values)]

    if observed_statistic is None:
        observed_statistic = metadata.get(
            "observed_statistic"
        )

    output_metadata = {
        **metadata,
        "observed_statistic": (
            None
            if observed_statistic is None
            else float(observed_statistic)
        ),
        "finite_bootstrap_reps": int(len(values)),
        "bootstrap_quantiles": (
            {
                str(q): float(
                    np.quantile(values, q)
                )
                for q in (
                    0.01,
                    0.025,
                    0.05,
                    0.50,
                    0.95,
                    0.975,
                    0.99,
                )
            }
            if len(values)
            else {}
        ),
    }

    arrays = {
        "bootstrap_statistics": values,
    }

    if extra_arrays:
        arrays.update(
            {
                key: np.asarray(value)
                for key, value
                in extra_arrays.items()
            }
        )

    if storage.mode == "none":
        return {
            **output_metadata,
            "bootstrap_storage": "none",
        }

    if storage.mode == "metadata":
        return {
            **output_metadata,
            **arrays,
            "bootstrap_storage": "metadata",
            "bootstrap_array_names": sorted(arrays),
        }

    folder = Path(
        str(storage.output_dir)
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{_slug(storage.prefix)}_"
        f"{_slug(test_id)}_"
        f"{_slug(model_name)}_"
        f"h{int(horizon)}.npz"
    )

    path = folder / filename

    np.savez_compressed(
        path,
        observed_statistic=np.asarray(
            [
                np.nan
                if observed_statistic is None
                else float(observed_statistic)
            ],
            dtype=float,
        ),
        **arrays,
    )

    return {
        **output_metadata,
        "bootstrap_storage": "external",
        "bootstrap_path": str(path),
        "bootstrap_statistics_path": str(path),
        "bootstrap_distribution_path": str(path),
        "bootstrap_array_names": sorted(arrays),
    }