# tests/coverage.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .base import DensityTest


def _clean_pit(pit):
    pit = np.asarray(pit, dtype=float)
    return pit[np.isfinite(pit)]


def _hits_from_pit(pit, *, alpha: float, side: str):
    pit = _clean_pit(pit)

    if side == "left":
        return (pit <= alpha).astype(int), alpha

    if side == "right":
        return (pit >= 1.0 - alpha).astype(int), alpha

    if side == "two-sided":
        return ((pit <= alpha / 2.0) | (pit >= 1.0 - alpha / 2.0)).astype(int), alpha

    raise ValueError("side must be 'left', 'right', or 'two-sided'.")


def _model_horizon_frames(dataset):
    for model in dataset.models:
        df_model = dataset.get_model(model, require_nonempty=False)

        if df_model.empty:
            continue

        for horizon in sorted(df_model["horizon"].dropna().unique()):
            df = dataset.get_model(model, horizon=int(horizon), require_nonempty=False)

            if not df.empty:
                yield model, int(horizon), df


@dataclass(frozen=True)
class Kupiec(DensityTest):
    alpha_level: float = 0.05
    side: str = "left"

    test_id: str = "kupiec"
    test_name: str = "Kupiec unconditional coverage"
    category: str = "coverage"
    null: str = "Hit probability equals nominal probability."
    alternative: str = "Hit probability differs from nominal probability."

    def evaluate(self, dataset):
        out = []

        for model, horizon, df in _model_horizon_frames(dataset):
            hits, p = _hits_from_pit(
                df["pit"].values,
                alpha=float(self.alpha_level),
                side=self.side,
            )

            n = int(len(hits))
            x = int(hits.sum())

            if n == 0:
                out.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=0,
                    metadata={
                        "horizon": horizon,
                        "message": "No observations.",
                    },
                ))
                continue

            pi_hat = x / n
            eps = 1e-12

            ll_null = (
                (n - x) * np.log(max(1.0 - p, eps))
                + x * np.log(max(p, eps))
            )

            ll_alt = (
                (n - x) * np.log(max(1.0 - pi_hat, eps))
                + x * np.log(max(pi_hat, eps))
            )

            lr = -2.0 * (ll_null - ll_alt)
            pvalue = 1.0 - stats.chi2.cdf(lr, df=1)

            out.append(self.result(
                model_name=model,
                statistic=float(lr),
                pvalue=float(pvalue),
                distribution="chi2",
                degrees_of_freedom=1,
                sample_size=n,
                effect_size=float(pi_hat - p),
                metadata={
                    "horizon": horizon,
                    "alpha_level": float(self.alpha_level),
                    "side": self.side,
                    "expected_prob": float(p),
                    "observed_prob": float(pi_hat),
                    "hits": x,
                    "n": n,
                },
            ))

        return out


@dataclass(frozen=True)
class ChristoffersenIndependence(DensityTest):
    alpha_level: float = 0.05
    side: str = "left"

    test_id: str = "christoffersen_independence"
    test_name: str = "Christoffersen independence"
    category: str = "coverage"
    null: str = "Hits are serially independent."
    alternative: str = "Hits are serially dependent."

    def evaluate(self, dataset):
        out = []

        for model, horizon, df in _model_horizon_frames(dataset):
            hits, p = _hits_from_pit(
                df["pit"].values,
                alpha=float(self.alpha_level),
                side=self.side,
            )

            n = int(len(hits))

            if n < 3:
                out.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=n,
                    metadata={
                        "horizon": horizon,
                        "message": "Too few observations.",
                    },
                ))
                continue

            h0 = hits[:-1]
            h1 = hits[1:]

            n00 = int(np.sum((h0 == 0) & (h1 == 0)))
            n01 = int(np.sum((h0 == 0) & (h1 == 1)))
            n10 = int(np.sum((h0 == 1) & (h1 == 0)))
            n11 = int(np.sum((h0 == 1) & (h1 == 1)))

            eps = 1e-12

            pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
            pi0 = n01 / max(n00 + n01, 1)
            pi1 = n11 / max(n10 + n11, 1)

            ll_ind = (
                (n00 + n10) * np.log(max(1.0 - pi, eps))
                + (n01 + n11) * np.log(max(pi, eps))
            )

            ll_dep = (
                n00 * np.log(max(1.0 - pi0, eps))
                + n01 * np.log(max(pi0, eps))
                + n10 * np.log(max(1.0 - pi1, eps))
                + n11 * np.log(max(pi1, eps))
            )

            lr = -2.0 * (ll_ind - ll_dep)
            pvalue = 1.0 - stats.chi2.cdf(lr, df=1)

            out.append(self.result(
                model_name=model,
                statistic=float(lr),
                pvalue=float(pvalue),
                distribution="chi2",
                degrees_of_freedom=1,
                sample_size=n,
                effect_size=float(pi1 - pi0),
                metadata={
                    "horizon": horizon,
                    "alpha_level": float(self.alpha_level),
                    "side": self.side,
                    "n00": n00,
                    "n01": n01,
                    "n10": n10,
                    "n11": n11,
                    "pi": float(pi),
                    "pi0": float(pi0),
                    "pi1": float(pi1),
                },
            ))

        return out


@dataclass(frozen=True)
class ChristoffersenConditionalCoverage(DensityTest):
    alpha_level: float = 0.05
    side: str = "left"

    test_id: str = "christoffersen_cc"
    test_name: str = "Christoffersen conditional coverage"
    category: str = "coverage"
    null: str = "Correct unconditional coverage and independent hits."
    alternative: str = "Incorrect coverage and/or dependent hits."

    def evaluate(self, dataset):
        kupiec = Kupiec(
            alpha=self.alpha,
            alpha_level=self.alpha_level,
            side=self.side,
        )

        indep = ChristoffersenIndependence(
            alpha=self.alpha,
            alpha_level=self.alpha_level,
            side=self.side,
        )

        kupiec_results = {
            (r.model_name, r.metadata.get("horizon")): r
            for r in kupiec.evaluate(dataset)
        }

        indep_results = {
            (r.model_name, r.metadata.get("horizon")): r
            for r in indep.evaluate(dataset)
        }

        out = []

        for model, horizon, df in _model_horizon_frames(dataset):
            k = kupiec_results.get((model, horizon))
            ci = indep_results.get((model, horizon))

            if (
                k is None
                or ci is None
                or k.statistic is None
                or ci.statistic is None
            ):
                out.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=None,
                    metadata={
                        "horizon": horizon,
                        "message": "Could not compute component tests.",
                    },
                ))
                continue

            lr = float(k.statistic + ci.statistic)
            pvalue = 1.0 - stats.chi2.cdf(lr, df=2)

            out.append(self.result(
                model_name=model,
                statistic=lr,
                pvalue=float(pvalue),
                distribution="chi2",
                degrees_of_freedom=2,
                sample_size=k.sample_size,
                metadata={
                    "horizon": horizon,
                    "alpha_level": float(self.alpha_level),
                    "side": self.side,
                    "kupiec_stat": k.statistic,
                    "kupiec_pvalue": k.pvalue,
                    "independence_stat": ci.statistic,
                    "independence_pvalue": ci.pvalue,
                },
            ))

        return out
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True, kw_only=True)
class PattonIntervalHit(DensityTest):
    lower: float = 0.0
    upper: float = 0.2

    test_id: str = "patton_interval_hit"
    test_name: str = "Patton interval hit test"
    category: str = "coverage"
    null: str = "Correct interval hit probability and no dependence on recent hits."
    alternative: str = "Incorrect interval coverage and/or predictable interval hits."

    def __post_init__(self):
        lo = float(self.lower)
        hi = float(self.upper)

        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("Require 0 <= lower < upper <= 1.")

        object.__setattr__(
            self,
            "test_id",
            f"patton_interval_hit_{lo:.3f}_{hi:.3f}".replace(".", "p"),
        )

        object.__setattr__(
            self,
            "test_name",
            f"Patton interval hit test [{lo:.2f}, {hi:.2f}]",
        )

    def evaluate(self, dataset):
        out = []

        lo = float(self.lower)
        hi = float(self.upper)
        p = hi - lo
        eps = 1e-12

        for model, horizon, df in _model_horizon_frames(dataset):
            pit = _clean_pit(df["pit"].values)

            n = int(len(pit))

            if n < 15:
                out.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=n,
                    metadata={
                        "horizon": horizon,
                        "lower": lo,
                        "upper": hi,
                        "message": "Too few observations.",
                    },
                ))
                continue

            hits = ((pit >= lo) & (pit < hi)).astype(float)

            # Include right endpoint for final interval.
            if hi == 1.0:
                hits = ((pit >= lo) & (pit <= hi)).astype(float)

            y = hits[10:]

            lag1 = hits[9:-1]
            lag5 = np.array([hits[t - 5:t].sum() for t in range(10, n)])
            lag10 = np.array([hits[t - 10:t].sum() for t in range(10, n)])

            Z = np.column_stack([
                np.ones_like(y),
                lag1,
                lag5,
                lag10,
            ])

            if len(y) == 0:
                continue

            logit_p = np.log(max(p, eps) / max(1.0 - p, eps))

            def loglik(beta):
                beta = np.asarray(beta, dtype=float)
                eta = logit_p + Z @ beta
                prob = expit(np.clip(eta, -700, 700))
                prob = np.clip(prob, eps, 1.0 - eps)
                return float(np.sum(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)))

            ll0 = loglik(np.zeros(4))

            def obj(beta):
                val = loglik(beta)
                if not np.isfinite(val):
                    return 1e100
                return -val

            res = minimize(
                obj,
                np.zeros(4),
                method="BFGS",
            )

            if not res.success or not np.isfinite(res.fun):
                out.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=n,
                    metadata={
                        "horizon": horizon,
                        "lower": lo,
                        "upper": hi,
                        "message": f"Optimization failed: {res.message}",
                    },
                ))
                continue

            ll1 = -float(res.fun)
            lr = max(0.0, -2.0 * (ll0 - ll1))
            pvalue = 1.0 - stats.chi2.cdf(lr, df=4)

            out.append(self.result(
                model_name=model,
                statistic=float(lr),
                pvalue=float(pvalue),
                distribution="chi2",
                degrees_of_freedom=4,
                sample_size=int(len(y)),
                effect_size=float(hits.mean() - p),
                metadata={
                    "horizon": horizon,
                    "lower": lo,
                    "upper": hi,
                    "expected_prob": float(p),
                    "observed_prob": float(hits.mean()),
                    "hits": int(hits.sum()),
                    "n": int(n),
                    "effective_n": int(len(y)),
                    "beta_hat": np.asarray(res.x, dtype=float).tolist(),
                    "ll_restricted": float(ll0),
                    "ll_unrestricted": float(ll1),
                },
            ))

        return out