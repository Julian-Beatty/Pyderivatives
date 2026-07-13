import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import ndtri
from statsmodels.stats.diagnostic import acorr_ljungbox

def _clean_pit(pit, eps=1e-10):
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    return np.clip(pit, eps, 1.0 - eps)


def pit_uniformity_test(pit):
    pit = _clean_pit(pit)

    stat, pvalue = stats.kstest(pit, "uniform")

    return {
        "test": "KS PIT uniformity",
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "n": int(len(pit)),
    }


def z_diagnostics(pit):
    pit = _clean_pit(pit)
    z = ndtri(pit)

    jb_stat, jb_pvalue = stats.jarque_bera(z)
    ks_stat, ks_pvalue = stats.kstest(z, "norm")

    return {
        "test": "z diagnostics",
        "n": int(len(z)),
        "z_mean": float(np.mean(z)),
        "z_var": float(np.var(z, ddof=1)),
        "z_std": float(np.std(z, ddof=1)),
        "z_skew": float(stats.skew(z)),
        "z_kurtosis": float(stats.kurtosis(z, fisher=False)),
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pvalue),
        "ks_z_stat": float(ks_stat),
        "ks_z_pvalue": float(ks_pvalue),
    }


def berkowitz_lr3_test(pit):
    pit = _clean_pit(pit)
    z = ndtri(pit)

    if len(z) < 10:
        return {
            "test": "Berkowitz LR3",
            "statistic": np.nan,
            "pvalue": np.nan,
            "n": int(len(z)),
        }

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

    return {
        "test": "Berkowitz LR3",
        "statistic": float(lr3),
        "pvalue": float(pvalue),
        "n": int(len(z)),
        "mu_hat": float(mu_hat),
        "sigma_hat": float(np.exp(log_sigma_hat)),
        "rho_hat": float(np.tanh(rho_raw_hat)),
        "success": bool(res.success),
        "message": str(res.message),
    }


def pit_summary(pit):
    pit = _clean_pit(pit)

    return {
        "n": int(len(pit)),
        "mean": float(np.mean(pit)),
        "std": float(np.std(pit, ddof=1)),
        "uniform_mean": 0.5,
        "uniform_std": float(np.sqrt(1.0 / 12.0)),
        "min": float(np.min(pit)),
        "max": float(np.max(pit)),
    }


def evaluate_pit_tests(pit):
    return {
        "pit_summary": pit_summary(pit),
        "ks_uniform": pit_uniformity_test(pit),
        "z_diagnostics": z_diagnostics(pit),
        "berkowitz_lr3": berkowitz_lr3_test(pit),
    }

def autocorrelation_moment_tests(
    pit,
    lags=(6, 12),
):
    """
    Autocorrelation tests for z_t moments.

    Tests:
        z_t
        z_t^2
        z_t^3
        z_t^4

    using Ljung-Box statistics.
    """

    pit = np.asarray(pit, dtype=float)

    pit = np.clip(pit, 1e-10, 1 - 1e-10)

    z = stats.norm.ppf(pit)

    moments = {
        "1st Moment": z,
        "2nd Moment": z**2,
        "3rd Moment": z**3,
        "4th Moment": z**4,
    }

    out = {}

    for moment_name, series in moments.items():

        out[moment_name] = {}

        for lag in lags:

            lb = acorr_ljungbox(
                series,
                lags=[lag],
                return_df=True,
            )

            out[moment_name][lag] = {
                "lb_stat": float(lb["lb_stat"].iloc[0]),
                "pvalue": float(lb["lb_pvalue"].iloc[0]),
            }

    return out