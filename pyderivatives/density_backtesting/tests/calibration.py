# tests/calibration.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import ndtri
from statsmodels.stats.diagnostic import acorr_ljungbox

from .base import DensityTest


def _clean_pit(pit, eps=1e-10):
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    return np.clip(pit, eps, 1.0 - eps)


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
class KolmogorovSmirnov(DensityTest):
    test_id: str = "ks"
    test_name: str = "Kolmogorov-Smirnov PIT uniformity"
    category: str = "calibration"

    def evaluate(self, dataset):
        out = []

        for model, horizon, df in _model_horizon_frames(dataset):
            pit = _clean_pit(df["pit"].values)

            if len(pit) < 5:
                out.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=int(len(pit)),
                    metadata={
                        "horizon": horizon,
                        "n": int(len(pit)),
                        "message": "Too few observations.",
                    },
                ))
                continue

            stat, pvalue = stats.kstest(pit, "uniform")

            out.append(self.result(
                model_name=model,
                statistic=float(stat),
                pvalue=float(pvalue),
                sample_size=int(len(pit)),
                metadata={
                    "horizon": horizon,
                    "n": int(len(pit)),
                },
            ))

        return out


@dataclass(frozen=True)
class CramerVonMises(DensityTest):
    test_id: str = "cvm"
    test_name: str = "Cramer-von Mises PIT uniformity"
    category: str = "calibration"

    def evaluate(self, dataset):
        out = []

        for model, horizon, df in _model_horizon_frames(dataset):
            pit = _clean_pit(df["pit"].values)

            if len(pit) < 5:
                out.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=int(len(pit)),
                    metadata={
                        "horizon": horizon,
                        "n": int(len(pit)),
                        "message": "Too few observations.",
                    },
                ))
                continue

            res = stats.cramervonmises(pit, "uniform")

            out.append(self.result(
                model_name=model,
                statistic=float(res.statistic),
                pvalue=float(res.pvalue),
                sample_size=int(len(pit)),
                metadata={
                    "horizon": horizon,
                    "n": int(len(pit)),
                },
            ))

        return out


@dataclass(frozen=True)
class JarqueBera(DensityTest):
    test_id: str = "jb"
    test_name: str = "Jarque-Bera normal-score PIT test"
    category: str = "calibration"

    def evaluate(self, dataset):
        out = []

        for model, horizon, df in _model_horizon_frames(dataset):
            pit = _clean_pit(df["pit"].values)
            z = ndtri(pit)

            if len(z) < 5:
                out.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=int(len(z)),
                    metadata={
                        "horizon": horizon,
                        "n": int(len(z)),
                        "message": "Too few observations.",
                    },
                ))
                continue

            stat, pvalue = stats.jarque_bera(z)

            out.append(self.result(
                model_name=model,
                statistic=float(stat),
                pvalue=float(pvalue),
                sample_size=int(len(z)),
                metadata={
                    "horizon": horizon,
                    "n": int(len(z)),
                    "z_mean": float(np.mean(z)),
                    "z_std": float(np.std(z, ddof=1)),
                    "z_skew": float(stats.skew(z)),
                    "z_kurtosis": float(stats.kurtosis(z, fisher=False)),
                },
            ))

        return out


@dataclass(frozen=True)
class BerkowitzLR3(DensityTest):
    test_id: str = "berkowitz_lr3"
    test_name: str = "Berkowitz LR3"
    category: str = "calibration"

    def evaluate(self, dataset):
        out = []

        for model, horizon, df in _model_horizon_frames(dataset):
            pit = _clean_pit(df["pit"].values)
            z = ndtri(pit)

            if len(z) < 10:
                out.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=int(len(z)),
                    metadata={
                        "horizon": horizon,
                        "n": int(len(z)),
                        "message": "Too few observations.",
                    },
                ))
                continue

            z_lag = z[:-1]
            z_now = z[1:]

            def nll(params):
                mu, log_sigma, rho_raw = params
                sigma = np.exp(log_sigma)
                rho = np.tanh(rho_raw)

                mean = mu + rho * (z_lag - mu)
                resid = z_now - mean

                ll = stats.norm.logpdf(resid, loc=0.0, scale=sigma)

                if not np.all(np.isfinite(ll)):
                    return 1e100

                return -float(np.sum(ll))

            x0 = np.array([
                np.mean(z),
                np.log(max(np.std(z, ddof=1), 1e-8)),
                0.0,
            ])

            res = minimize(nll, x0, method="BFGS")

            ll_alt = -float(res.fun)
            ll_null = float(np.sum(stats.norm.logpdf(z_now, loc=0.0, scale=1.0)))

            lr3 = -2.0 * (ll_null - ll_alt)
            pvalue = 1.0 - stats.chi2.cdf(lr3, df=3)

            mu_hat, log_sigma_hat, rho_raw_hat = res.x

            out.append(self.result(
                model_name=model,
                statistic=float(lr3),
                pvalue=float(pvalue),
                distribution="chi2",
                degrees_of_freedom=3,
                sample_size=int(len(z)),
                metadata={
                    "horizon": horizon,
                    "n": int(len(z)),
                    "mu_hat": float(mu_hat),
                    "sigma_hat": float(np.exp(log_sigma_hat)),
                    "rho_hat": float(np.tanh(rho_raw_hat)),
                    "success": bool(res.success),
                    "message": str(res.message),
                },
            ))

        return out


@dataclass(frozen=True)
class BerkowitzLR1(DensityTest):
    test_id: str = "berkowitz_lr1"
    test_name: str = "Berkowitz LR1 independence"
    category: str = "independence"

    def evaluate(self, dataset):
        out = []

        for model, horizon, df in _model_horizon_frames(dataset):
            pit = _clean_pit(df["pit"].values)
            z = ndtri(pit)

            if len(z) < 10:
                out.append(self.result(
                    model_name=model,
                    statistic=None,
                    pvalue=None,
                    sample_size=int(len(z)),
                    metadata={
                        "horizon": horizon,
                        "n": int(len(z)),
                        "message": "Too few observations.",
                    },
                ))
                continue

            z_lag = z[:-1]
            z_now = z[1:]

            def nll_unrestricted(params):
                mu, log_sigma, rho_raw = params
                sigma = np.exp(log_sigma)
                rho = np.tanh(rho_raw)
                mean = mu + rho * (z_lag - mu)
                resid = z_now - mean
                return -float(np.sum(stats.norm.logpdf(resid, 0.0, sigma)))

            def nll_restricted(params):
                mu, log_sigma = params
                sigma = np.exp(log_sigma)
                resid = z_now - mu
                return -float(np.sum(stats.norm.logpdf(resid, 0.0, sigma)))

            x0_u = np.array([
                np.mean(z),
                np.log(max(np.std(z, ddof=1), 1e-8)),
                0.0,
            ])

            x0_r = np.array([
                np.mean(z),
                np.log(max(np.std(z, ddof=1), 1e-8)),
            ])

            res_u = minimize(nll_unrestricted, x0_u, method="BFGS")
            res_r = minimize(nll_restricted, x0_r, method="BFGS")

            ll_u = -float(res_u.fun)
            ll_r = -float(res_r.fun)

            lr1 = -2.0 * (ll_r - ll_u)
            pvalue = 1.0 - stats.chi2.cdf(lr1, df=1)

            out.append(self.result(
                model_name=model,
                statistic=float(lr1),
                pvalue=float(pvalue),
                distribution="chi2",
                degrees_of_freedom=1,
                sample_size=int(len(z)),
                metadata={
                    "horizon": horizon,
                    "n": int(len(z)),
                    "rho_hat": float(np.tanh(res_u.x[2])),
                    "success": bool(res_u.success and res_r.success),
                    "message": f"unrestricted={res_u.message}; restricted={res_r.message}",
                },
            ))

        return out


@dataclass(frozen=True)
class LjungBox(DensityTest):
    lags: Sequence[int] = field(default_factory=lambda: (6, 12))
    moments: Sequence[int] = field(default_factory=lambda: (1, 2, 3, 4))

    test_id: str = "ljung_box"
    test_name: str = "Ljung-Box normal-score moment autocorrelation"
    category: str = "independence"

    def evaluate(self, dataset):
        out = []

        for model, horizon, df in _model_horizon_frames(dataset):
            pit = _clean_pit(df["pit"].values)
            z = ndtri(pit)

            for moment in self.moments:
                series = z ** int(moment)

                for lag in self.lags:
                    if len(series) <= lag + 2:
                        out.append(self.result(
                            model_name=model,
                            statistic=None,
                            pvalue=None,
                            sample_size=int(len(series)),
                            metadata={
                                "horizon": horizon,
                                "n": int(len(series)),
                                "moment": int(moment),
                                "lag": int(lag),
                                "message": "Too few observations.",
                            },
                        ))
                        continue

                    lb = acorr_ljungbox(
                        series,
                        lags=[int(lag)],
                        return_df=True,
                    )

                    stat = float(lb["lb_stat"].iloc[0])
                    pvalue = float(lb["lb_pvalue"].iloc[0])

                    out.append(self.result(
                        model_name=model,
                        statistic=stat,
                        pvalue=pvalue,
                        sample_size=int(len(series)),
                        metadata={
                            "horizon": horizon,
                            "n": int(len(series)),
                            "moment": int(moment),
                            "lag": int(lag),
                        },
                    ))

        return out
    
