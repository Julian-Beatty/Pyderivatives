import numpy as np
import pandas as pd
from scipy import stats


def align_model_scores(df, model_a, model_b):
    """
    Align two model log score series by date.
    """

    a = (
        df[df["model"] == model_a]
        [["date", "log_score"]]
        .rename(columns={"log_score": "score_a"})
    )

    b = (
        df[df["model"] == model_b]
        [["date", "log_score"]]
        .rename(columns={"log_score": "score_b"})
    )

    merged = pd.merge(a, b, on="date", how="inner")

    return merged.sort_values("date")


def diebold_mariano_test(
    df,
    model_a,
    model_b,
    horizon=1,
):
    """
    Diebold-Mariano test using log scores.

    Null:
        equal predictive accuracy

    Positive statistic:
        model_a better than model_b
    """

    aligned = align_model_scores(df, model_a, model_b)

    if len(aligned) < 5:
        return {
            "model_a": model_a,
            "model_b": model_b,
            "n": len(aligned),
            "statistic": np.nan,
            "pvalue": np.nan,
        }

    d = (
        aligned["score_a"].values
        -
        aligned["score_b"].values
    )

    mean_d = np.mean(d)

    # HAC variance estimate
    n = len(d)

    gamma0 = np.var(d, ddof=1)

    var = gamma0

    for lag in range(1, horizon):
        cov = np.cov(d[:-lag], d[lag:])[0, 1]
        weight = 1.0 - lag / horizon
        var += 2.0 * weight * cov

    var = var / n

    if var <= 0 or not np.isfinite(var):
        return {
            "model_a": model_a,
            "model_b": model_b,
            "n": n,
            "statistic": np.nan,
            "pvalue": np.nan,
        }

    dm_stat = mean_d / np.sqrt(var)

    pvalue = 2.0 * (
        1.0 - stats.norm.cdf(abs(dm_stat))
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "n": n,
        "mean_score_diff": float(mean_d),
        "statistic": float(dm_stat),
        "pvalue": float(pvalue),
    }


def pairwise_dm_table(df, horizon=1):
    models = sorted(df["model"].unique())
    rows = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            res = diebold_mariano_test(
                df,
                models[i],
                models[j],
                horizon=horizon,
            )
            rows.append(res)

    return pd.DataFrame(rows)