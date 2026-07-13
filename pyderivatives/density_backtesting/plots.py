# plots.py

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def density_overlay_frame(
    dataset,
    *,
    date,
    horizon,
    models=None,
):
    import pandas as pd

    date = pd.Timestamp(date).tz_localize(None)

    rows = []

    for f in dataset.forecasts:
        if pd.Timestamp(f.date).tz_localize(None) != date:
            continue

        if int(f.horizon) != int(horizon):
            continue

        if models is not None and f.model_name not in models:
            continue

        for x, pdf, cdf in zip(f.x_grid, f.pdf, f.cdf):
            rows.append({
                "date": pd.Timestamp(f.date),
                "model": f.model_name,
                "horizon": int(f.horizon),
                "x": float(x),
                "pdf": float(pdf),
                "cdf": float(cdf),
                "realized": float(f.realized),
                **f.metadata,
            })

    if not rows:
        raise ValueError("No matching forecast densities found.")

    return pd.DataFrame(rows)

def plot_density_overlay(
    dataset,
    *,
    date,
    horizon,
    models=None,
    xlim=None,
    mark_realized=True,
    title=None,
):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = density_overlay_frame(
        dataset,
        date=date,
        horizon=horizon,
        models=models,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for model, g in df.groupby("model"):
        g = g.sort_values("x")
        ax.plot(g["x"], g["pdf"], label=model)

    if mark_realized:
        realized = df["realized"].iloc[0]
        ax.axvline(
            realized,
            linestyle="--",
            linewidth=1.5,
            label=f"Realized = {realized:.4f}",
        )

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_title(title or f"Forecast densities on {pd.Timestamp(date).date()} ({horizon}d)")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.25)

    return fig, ax

def _get_df(report_or_dataset):
    if hasattr(report_or_dataset, "dataset"):
        return report_or_dataset.dataset.to_frame()
    return report_or_dataset.to_frame()


def plot_pit_histogram(
    report_or_dataset,
    *,
    model: str,
    horizon: Optional[int] = None,
    bins: int = 20,
    ax=None,
):
    df = _get_df(report_or_dataset)
    df = df[df["model"] == model]

    if horizon is not None:
        df = df[df["horizon"] == int(horizon)]

    if df.empty:
        raise ValueError("No matching forecasts found.")

    pit = df["pit"].dropna().values

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.hist(pit, bins=bins, density=True, edgecolor="black", alpha=0.75)
    ax.axhline(1.0, linestyle="--", linewidth=1.5)

    title = f"PIT Histogram: {model}"
    if horizon is not None:
        title += f" ({horizon}d)"

    ax.set_title(title)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)

    return fig, ax


def plot_pit_ecdf(
    report_or_dataset,
    *,
    model: str,
    horizon: Optional[int] = None,
    ax=None,
):
    df = _get_df(report_or_dataset)
    df = df[df["model"] == model]

    if horizon is not None:
        df = df[df["horizon"] == int(horizon)]

    if df.empty:
        raise ValueError("No matching forecasts found.")

    pit = np.sort(df["pit"].dropna().values)
    y = np.arange(1, len(pit) + 1) / len(pit)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ax.plot(pit, y, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)

    title = f"PIT ECDF: {model}"
    if horizon is not None:
        title += f" ({horizon}d)"

    ax.set_title(title)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return fig, ax


def plot_pit_qq(
    report_or_dataset,
    *,
    model: str,
    horizon: Optional[int] = None,
    ax=None,
):
    df = _get_df(report_or_dataset)
    df = df[df["model"] == model]

    if horizon is not None:
        df = df[df["horizon"] == int(horizon)]

    if df.empty:
        raise ValueError("No matching forecasts found.")

    pit = np.sort(df["pit"].dropna().values)
    n = len(pit)
    theo = (np.arange(1, n + 1) - 0.5) / n

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ax.scatter(theo, pit, s=18)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)

    title = f"PIT QQ Plot: {model}"
    if horizon is not None:
        title += f" ({horizon}d)"

    ax.set_title(title)
    ax.set_xlabel("Uniform quantiles")
    ax.set_ylabel("Empirical PIT quantiles")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return fig, ax


def plot_normal_score_qq(
    report_or_dataset,
    *,
    model: str,
    horizon: Optional[int] = None,
    ax=None,
):
    df = _get_df(report_or_dataset)
    df = df[df["model"] == model]

    if horizon is not None:
        df = df[df["horizon"] == int(horizon)]

    if df.empty:
        raise ValueError("No matching forecasts found.")

    pit = df["pit"].clip(1e-10, 1 - 1e-10).dropna().values
    z = stats.norm.ppf(pit)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    stats.probplot(z, dist="norm", plot=ax)

    title = f"Normal-Score PIT QQ: {model}"
    if horizon is not None:
        title += f" ({horizon}d)"

    ax.set_title(title)

    return fig, ax


def plot_logscore_time_series(
    report_or_dataset,
    *,
    models: Optional[Sequence[str]] = None,
    horizon: Optional[int] = None,
    cumulative: bool = False,
    ax=None,
):
    df = _get_df(report_or_dataset)

    if models is not None:
        df = df[df["model"].isin(models)]

    if horizon is not None:
        df = df[df["horizon"] == int(horizon)]

    if df.empty:
        raise ValueError("No matching forecasts found.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    for model, g in df.groupby("model"):
        g = g.sort_values("date")
        y = g["log_score"].astype(float)

        if cumulative:
            y = y.cumsum()

        ax.plot(g["date"], y, label=model, linewidth=1.6)

    title = "Cumulative Log Score" if cumulative else "Log Score"
    if horizon is not None:
        title += f" ({horizon}d)"

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log score")
    ax.legend()
    fig.autofmt_xdate()

    return fig, ax


def plot_cumulative_logscore_difference(
    report_or_dataset,
    *,
    model_a: str,
    model_b: str,
    horizon: Optional[int] = None,
    ax=None,
):
    dataset = report_or_dataset.dataset if hasattr(report_or_dataset, "dataset") else report_or_dataset

    df = dataset.compare(
        model_a,
        model_b,
        horizon=horizon,
        columns=["log_score", "horizon"],
        common_dates_only=True,
    )

    if df.empty:
        raise ValueError("No common forecasts found.")

    diff = df["log_score_a"].astype(float) - df["log_score_b"].astype(float)
    cumdiff = diff.cumsum()

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    ax.plot(df["date"], cumdiff, linewidth=2)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)

    title = f"Cumulative Log-Score Difference: {model_a} - {model_b}"
    if horizon is not None:
        title += f" ({horizon}d)"

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative difference")
    fig.autofmt_xdate()

    return fig, ax


def plot_hit_sequence(
    report_or_dataset,
    *,
    model: str,
    horizon: Optional[int] = None,
    alpha_level: float = 0.05,
    side: str = "left",
    ax=None,
):
    df = _get_df(report_or_dataset)
    df = df[df["model"] == model]

    if horizon is not None:
        df = df[df["horizon"] == int(horizon)]

    if df.empty:
        raise ValueError("No matching forecasts found.")

    df = df.sort_values("date")
    pit = df["pit"].astype(float)

    if side == "left":
        hits = (pit <= alpha_level).astype(int)
    elif side == "right":
        hits = (pit >= 1.0 - alpha_level).astype(int)
    elif side == "two-sided":
        hits = ((pit <= alpha_level / 2.0) | (pit >= 1.0 - alpha_level / 2.0)).astype(int)
    else:
        raise ValueError("side must be 'left', 'right', or 'two-sided'.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3))
    else:
        fig = ax.figure

    ax.step(df["date"], hits, where="post", linewidth=1.5)
    ax.set_ylim(-0.1, 1.1)

    title = f"Hit Sequence: {model}, {side}, alpha={alpha_level}"
    if horizon is not None:
        title += f" ({horizon}d)"

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Hit")
    fig.autofmt_xdate()

    return fig, ax


def plot_test_pvalues(
    report,
    *,
    category: Optional[str] = None,
    horizon: Optional[int] = None,
    ax=None,
):
    df = report.tests_frame()

    if df.empty:
        raise ValueError("No test results found.")

    if category is not None and "category" in df.columns:
        df = df[df["category"] == category]

    if horizon is not None and "horizon" in df.columns:
        df = df[df["horizon"] == int(horizon)]

    df = df.dropna(subset=["pvalue"])

    if df.empty:
        raise ValueError("No matching p-values found.")

    df = df.copy()
    df["label"] = df["model"].astype(str) + " | " + df["test_id"].astype(str)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(df))))
    else:
        fig = ax.figure

    y = np.arange(len(df))
    ax.barh(y, df["pvalue"].astype(float))
    ax.axvline(0.05, linestyle="--", linewidth=1.2)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("p-value")
    ax.set_title("Test p-values")
    ax.set_xlim(0, 1)

    return fig, ax


def plot_holm(
    report,
    *,
    method: str = "HolmBonferroni",
    group_filter: Optional[dict] = None,
    ax=None,
):
    if not getattr(report, "adjusted_results", None):
        raise ValueError("Report has no adjusted results.")

    if method not in report.adjusted_results:
        raise ValueError(f"Adjustment method '{method}' not found.")

    df = report.adjusted_results[method].copy()

    if group_filter is not None:
        for key, value in group_filter.items():
            if key in df.columns:
                df = df[df[key] == value]

    if df.empty:
        raise ValueError("No matching Holm results found.")

    df = df.dropna(subset=["holm_rank", "pvalue", "holm_critical"])
    df = df.sort_values("holm_rank")

    if df.empty:
        raise ValueError("No ranked Holm results found.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    ax.scatter(df["holm_rank"], df["pvalue"], label="p-values")
    ax.plot(df["holm_rank"], df["holm_critical"], linestyle="--", label="Holm critical values")

    ax.set_title("Holm-Bonferroni Step-Down Plot")
    ax.set_xlabel("Sorted rank")
    ax.set_ylabel("p-value")
    ax.legend()

    return fig, ax