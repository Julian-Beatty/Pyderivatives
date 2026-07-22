from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import ndtri

from .base import DensityTest
from .bootstrap_inference import (
    BootstrapStorageSpec,
    circular_block_bootstrap_statistics,
    store_bootstrap_distribution,
    uniform_rank_null,
)


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


def _cvm_statistic(pit) -> float:
    return float(stats.cramervonmises(pit, "uniform").statistic)


def _ks_statistic(pit) -> float:
    return float(stats.kstest(pit, "uniform").statistic)


def _bootstrap_gof_result(
    test,
    *,
    model: str,
    horizon: int,
    pit: np.ndarray,
    statistic_fn,
):
    n = len(pit)
    observed = float(statistic_fn(pit))
    null_pit = uniform_rank_null(pit)

    bootstrap_statistics = circular_block_bootstrap_statistics(
        null_pit,
        statistic_fn=statistic_fn,
        block_length=test.block_length,
        bootstrap_reps=test.bootstrap_reps,
        random_state=test.random_state,
    )
    bootstrap_statistics = bootstrap_statistics[
        np.isfinite(bootstrap_statistics)
    ]

    if len(bootstrap_statistics) == 0:
        return test.result(
            model_name=model,
            statistic=observed,
            pvalue=None,
            sample_size=n,
            metadata={
                "horizon": horizon,
                "n": n,
                "message": "No finite bootstrap statistics.",
            },
        )

    pvalue = float(
        (1 + np.sum(bootstrap_statistics >= observed))
        / (len(bootstrap_statistics) + 1)
    )

    metadata = {
        "horizon": horizon,
        "n": n,
        "inference": "circular-block-bootstrap",
        "block_length": min(int(test.block_length), n),
        "bootstrap_reps": int(test.bootstrap_reps),
        "finite_bootstrap_reps": int(len(bootstrap_statistics)),
        "null_method": "rank_uniformization",
        "rank_plotting_position": "(rank - 0.5) / n",
        "bootstrap_quantiles": {
            str(q): float(np.quantile(bootstrap_statistics, q))
            for q in (0.01, 0.025, 0.5, 0.95, 0.975, 0.99)
        },
    }
    metadata = store_bootstrap_distribution(
        storage=test.bootstrap_storage,
        statistics=bootstrap_statistics,
        metadata=metadata,
        test_id=test.test_id,
        model_name=model,
        horizon=horizon,
        observed_statistic=observed,
    )

    return test.result(
        model_name=model,
        statistic=observed,
        pvalue=pvalue,
        distribution="bootstrap",
        sample_size=n,
        metadata=metadata,
    )


@dataclass(frozen=True)
class KolmogorovSmirnovCircularBootstrap(DensityTest):
    block_length: int = 21
    bootstrap_reps: int = 2_000
    random_state: Optional[int] = None
    bootstrap_storage: BootstrapStorageSpec = field(
        default_factory=BootstrapStorageSpec
    )

    test_id: str = "ks_circular_block_bootstrap"
    test_name: str = "Kolmogorov-Smirnov circular-block bootstrap"
    category: str = "calibration"
    null: str = (
        "PITs have a Uniform(0,1) marginal under the retained local "
        "serial rank-dependence structure."
    )
    alternative: str = "The PIT marginal distribution is not Uniform(0,1)."

    def evaluate(self, dataset):
        output = []
        for model, horizon, frame in _model_horizon_frames(dataset):
            pit = _clean_pit(frame["pit"].to_numpy(dtype=float))
            if len(pit) < 10:
                output.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=len(pit),
                    metadata={
                        "horizon": horizon,
                        "n": len(pit),
                        "message": "Too few observations.",
                    },
                ))
                continue
            output.append(_bootstrap_gof_result(
                self,
                model=model,
                horizon=horizon,
                pit=pit,
                statistic_fn=_ks_statistic,
            ))
        return output


@dataclass(frozen=True)
class CramerVonMisesCircularBootstrap(DensityTest):
    block_length: int = 21
    bootstrap_reps: int = 2_000
    random_state: Optional[int] = None
    bootstrap_storage: BootstrapStorageSpec = field(
        default_factory=BootstrapStorageSpec
    )

    test_id: str = "cvm_circular_block_bootstrap"
    test_name: str = "Cramér-von Mises circular-block bootstrap"
    category: str = "calibration"
    null: str = (
        "PITs have a Uniform(0,1) marginal under the retained local "
        "serial rank-dependence structure."
    )
    alternative: str = "The PIT marginal distribution is not Uniform(0,1)."

    def evaluate(self, dataset):
        output = []
        for model, horizon, frame in _model_horizon_frames(dataset):
            pit = _clean_pit(frame["pit"].to_numpy(dtype=float))
            if len(pit) < 10:
                output.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=len(pit),
                    metadata={
                        "horizon": horizon,
                        "n": len(pit),
                        "message": "Too few observations.",
                    },
                ))
                continue
            output.append(_bootstrap_gof_result(
                self,
                model=model,
                horizon=horizon,
                pit=pit,
                statistic_fn=_cvm_statistic,
            ))
        return output


def _berkowitz_lr3_statistic(z) -> tuple[float, dict]:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 10:
        return np.nan, {"success": False, "message": "Too few observations."}

    lagged = z[:-1]
    current = z[1:]

    def negative_log_likelihood(parameters):
        mu, log_sigma, raw_rho = parameters
        sigma = np.exp(log_sigma)
        rho = np.tanh(raw_rho)
        conditional_mean = mu + rho * (lagged - mu)
        residual = current - conditional_mean
        ll = stats.norm.logpdf(residual, loc=0.0, scale=sigma)
        return 1e100 if not np.all(np.isfinite(ll)) else -float(np.sum(ll))

    initial = np.array([
        np.mean(z),
        np.log(max(np.std(z, ddof=1), 1e-8)),
        0.0,
    ])
    fitted = minimize(negative_log_likelihood, initial, method="BFGS")
    alternative_ll = -float(fitted.fun)
    null_ll = float(np.sum(stats.norm.logpdf(current, loc=0.0, scale=1.0)))
    statistic = max(0.0, float(-2.0 * (null_ll - alternative_ll)))
    mu_hat, log_sigma_hat, raw_rho_hat = fitted.x
    return statistic, {
        "mu_hat": float(mu_hat),
        "sigma_hat": float(np.exp(log_sigma_hat)),
        "rho_hat": float(np.tanh(raw_rho_hat)),
        "success": bool(fitted.success),
        "message": str(fitted.message),
    }


@dataclass(frozen=True)
class BerkowitzLR3CircularBootstrap(DensityTest):
    block_length: int = 21
    bootstrap_reps: int = 2_000
    random_state: Optional[int] = None
    bootstrap_storage: BootstrapStorageSpec = field(
        default_factory=BootstrapStorageSpec
    )

    test_id: str = "berkowitz_lr3_circular_block_bootstrap"
    test_name: str = "Berkowitz LR3 circular-block bootstrap"
    category: str = "calibration"
    null: str = (
        "Normal-score PITs have an N(0,1) marginal under the retained "
        "local serial rank-dependence structure."
    )
    alternative: str = "The normal-score PIT process violates calibration."

    def evaluate(self, dataset):
        output = []
        for model, horizon, frame in _model_horizon_frames(dataset):
            pit = _clean_pit(frame["pit"].to_numpy(dtype=float))
            z = ndtri(pit)
            n = len(z)
            if n < 20:
                output.append(self.result(
                    model_name=model, statistic=None, pvalue=None,
                    sample_size=n,
                    metadata={"horizon": horizon, "n": n,
                              "message": "Too few observations."},
                ))
                continue

            observed, fit_metadata = _berkowitz_lr3_statistic(z)
            null_z = ndtri(uniform_rank_null(pit))
            bootstrap_statistics = circular_block_bootstrap_statistics(
                null_z,
                statistic_fn=lambda sample: _berkowitz_lr3_statistic(sample)[0],
                block_length=self.block_length,
                bootstrap_reps=self.bootstrap_reps,
                random_state=self.random_state,
            )
            finite = bootstrap_statistics[np.isfinite(bootstrap_statistics)]
            if len(finite) == 0:
                output.append(self.result(
                    model_name=model,
                    statistic=None if not np.isfinite(observed) else float(observed),
                    pvalue=None,
                    sample_size=n,
                    metadata={"horizon": horizon, "n": n,
                              "message": "No finite bootstrap LR3 statistics.",
                              **fit_metadata},
                ))
                continue

            pvalue = float((1 + np.sum(finite >= observed)) / (len(finite) + 1))
            metadata = {
                "horizon": horizon,
                "n": n,
                "inference": "circular-block-bootstrap",
                "block_length": min(self.block_length, n),
                "bootstrap_reps": self.bootstrap_reps,
                "finite_bootstrap_reps": len(finite),
                "null_method": "rank_gaussianization",
                **fit_metadata,
            }
            metadata = store_bootstrap_distribution(
                storage=self.bootstrap_storage,
                statistics=finite,
                metadata=metadata,
                test_id=self.test_id,
                model_name=model,
                horizon=horizon,
                observed_statistic=observed,
            )
            output.append(self.result(
                model_name=model,
                statistic=None if not np.isfinite(observed) else float(observed),
                pvalue=pvalue,
                distribution="bootstrap",
                sample_size=n,
                metadata=metadata,
            ))
        return output
