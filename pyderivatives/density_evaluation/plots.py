import numpy as np
import matplotlib.pyplot as plt


def plot_pit_histogram(
    pit,
    bins=20,
    title="PIT Histogram",
    density=True,
    ax=None,
):
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
        pit,
        bins=bins,
        density=density,
    )

    ax.axhline(1.0, linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.set_title(title)

    return ax

from scipy.special import ndtri
from scipy import stats


def plot_pit_qq(
    pit,
    title="PIT Normal QQ Plot",
    ax=None,
):
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    pit = np.clip(pit, 1e-10, 1 - 1e-10)

    z = ndtri(pit)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    stats.probplot(z, dist="norm", plot=ax)

    ax.set_title(title)

    return ax

import pandas as pd


def plot_cumulative_log_scores(
    df,
    ax=None,
):
    """
    df must contain:
        date
        model
        log_score
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    df = df.copy()
    df = df.sort_values("date")

    for model, sub in df.groupby("model"):
        sub = sub.sort_values("date")

        cumulative = sub["log_score"].cumsum()

        ax.plot(
            sub["date"],
            cumulative,
            label=model,
        )

    ax.set_title("Cumulative Log Scores")
    ax.set_ylabel("Cumulative Log Score")
    ax.legend()

    return ax

def plot_forecast_density(
    forecast,
    show_realized=True,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        forecast.x_grid,
        forecast.pdf,
        label=forecast.model_name,
    )

    if show_realized:
        ax.axvline(
            forecast.realized,
            linestyle="--",
            label="Realized",
        )

    ax.set_title(
        f"{forecast.model_name} | {forecast.date}"
    )

    ax.legend()

    return ax

def plot_density_comparison(
    forecasts,
    show_realized=True,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for f in forecasts:
        ax.plot(
            f.x_grid,
            f.pdf,
            label=f.model_name,
        )

    if show_realized and len(forecasts) > 0:
        ax.axvline(
            forecasts[0].realized,
            linestyle="--",
            label="Realized",
        )

    ax.legend()
    ax.set_title("Density Comparison")

    return ax
