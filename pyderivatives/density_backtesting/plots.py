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
    date = pd.Timestamp(
        date
    ).tz_localize(None)

    rows = []

    for forecast in dataset.forecasts:
        if (
            pd.Timestamp(
                forecast.date
            ).tz_localize(None)
            != date
        ):
            continue

        if int(forecast.horizon) != int(horizon):
            continue

        if (
            models is not None
            and forecast.model_name not in models
        ):
            continue

        for x, pdf, cdf in zip(
            forecast.x_grid,
            forecast.pdf,
            forecast.cdf,
        ):
            rows.append(
                {
                    "date": pd.Timestamp(
                        forecast.date
                    ),
                    "model": (
                        forecast.model_name
                    ),
                    "horizon": int(
                        forecast.horizon
                    ),
                    "x": float(x),
                    "pdf": float(pdf),
                    "cdf": float(cdf),
                    "realized": float(
                        forecast.realized
                    ),
                    **forecast.metadata,
                }
            )

    if not rows:
        raise ValueError(
            "No matching forecast densities found."
        )

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
    df = density_overlay_frame(
        dataset,
        date=date,
        horizon=horizon,
        models=models,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for model, group in df.groupby(
        "model"
    ):
        group = group.sort_values("x")

        ax.plot(
            group["x"],
            group["pdf"],
            label=model,
        )

    if mark_realized:
        realized = df[
            "realized"
        ].iloc[0]

        ax.axvline(
            realized,
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Realized = "
                f"{realized:.4f}"
            ),
        )

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_title(
        title
        or (
            "Forecast densities on "
            f"{pd.Timestamp(date).date()} "
            f"({horizon}d)"
        )
    )

    ax.set_xlabel(
        "Log return"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    return fig, ax


def _get_df(report_or_dataset):
    if hasattr(
        report_or_dataset,
        "dataset",
    ):
        return (
            report_or_dataset
            .dataset
            .to_frame()
        )

    return report_or_dataset.to_frame()


def plot_pit_histogram(
    report_or_dataset,
    *,
    model: str,
    horizon: Optional[int] = None,
    bins: int = 20,
    ax=None,
):
    df = _get_df(
        report_or_dataset
    )

    df = df[
        df["model"] == model
    ]

    if horizon is not None:
        df = df[
            df["horizon"] == int(horizon)
        ]

    if df.empty:
        raise ValueError(
            "No matching forecasts found."
        )

    pit = df[
        "pit"
    ].dropna().values

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(7, 4)
        )
    else:
        fig = ax.figure

    ax.hist(
        pit,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.75,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
    )

    title = (
        f"PIT Histogram: {model}"
    )

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
    df = _get_df(
        report_or_dataset
    )

    df = df[
        df["model"] == model
    ]

    if horizon is not None:
        df = df[
            df["horizon"] == int(horizon)
        ]

    if df.empty:
        raise ValueError(
            "No matching forecasts found."
        )

    pit = np.sort(
        df["pit"]
        .dropna()
        .values
    )

    y = (
        np.arange(
            1,
            len(pit) + 1,
        )
        / len(pit)
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(5, 5)
        )
    else:
        fig = ax.figure

    ax.plot(
        pit,
        y,
        linewidth=2,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
    )

    title = (
        f"PIT ECDF: {model}"
    )

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
    df = _get_df(
        report_or_dataset
    )

    df = df[
        df["model"] == model
    ]

    if horizon is not None:
        df = df[
            df["horizon"] == int(horizon)
        ]

    if df.empty:
        raise ValueError(
            "No matching forecasts found."
        )

    pit = np.sort(
        df["pit"]
        .dropna()
        .values
    )

    n = len(pit)

    theoretical = (
        np.arange(1, n + 1) - 0.5
    ) / n

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(5, 5)
        )
    else:
        fig = ax.figure

    ax.scatter(
        theoretical,
        pit,
        s=18,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
    )

    title = (
        f"PIT QQ Plot: {model}"
    )

    if horizon is not None:
        title += f" ({horizon}d)"

    ax.set_title(title)
    ax.set_xlabel(
        "Uniform quantiles"
    )
    ax.set_ylabel(
        "Empirical PIT quantiles"
    )
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
    df = _get_df(
        report_or_dataset
    )

    df = df[
        df["model"] == model
    ]

    if horizon is not None:
        df = df[
            df["horizon"] == int(horizon)
        ]

    if df.empty:
        raise ValueError(
            "No matching forecasts found."
        )

    pit = (
        df["pit"]
        .clip(
            1e-10,
            1 - 1e-10,
        )
        .dropna()
        .values
    )

    z = stats.norm.ppf(pit)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(5, 5)
        )
    else:
        fig = ax.figure

    stats.probplot(
        z,
        dist="norm",
        plot=ax,
    )

    title = (
        f"Normal-Score PIT QQ: {model}"
    )

    if horizon is not None:
        title += f" ({horizon}d)"

    ax.set_title(title)

    return fig, ax


def plot_logscore_time_series(
    report_or_dataset,
    *,
    models: Optional[
        Sequence[str]
    ] = None,
    horizon: Optional[int] = None,
    cumulative: bool = False,
    ax=None,
):
    df = _get_df(
        report_or_dataset
    )

    if models is not None:
        df = df[
            df["model"].isin(models)
        ]

    if horizon is not None:
        df = df[
            df["horizon"] == int(horizon)
        ]

    if df.empty:
        raise ValueError(
            "No matching forecasts found."
        )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(9, 4.5)
        )
    else:
        fig = ax.figure

    for model, group in df.groupby(
        "model"
    ):
        group = group.sort_values(
            "date"
        )

        y = group[
            "log_score"
        ].astype(float)

        if cumulative:
            y = y.cumsum()

        ax.plot(
            group["date"],
            y,
            label=model,
            linewidth=1.6,
        )

    title = (
        "Cumulative Log Score"
        if cumulative
        else "Log Score"
    )

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
    dataset = (
        report_or_dataset.dataset
        if hasattr(
            report_or_dataset,
            "dataset",
        )
        else report_or_dataset
    )

    df = dataset.compare(
        model_a,
        model_b,
        horizon=horizon,
        columns=[
            "log_score",
            "horizon",
        ],
        common_dates_only=True,
    )

    if df.empty:
        raise ValueError(
            "No common forecasts found."
        )

    difference = (
        df["log_score_a"].astype(float)
        - df["log_score_b"].astype(float)
    )

    cumulative_difference = (
        difference.cumsum()
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(9, 4.5)
        )
    else:
        fig = ax.figure

    ax.plot(
        df["date"],
        cumulative_difference,
        linewidth=2,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )

    title = (
        "Cumulative Log-Score Difference: "
        f"{model_a} - {model_b}"
    )

    if horizon is not None:
        title += f" ({horizon}d)"

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(
        "Cumulative difference"
    )

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
    df = _get_df(
        report_or_dataset
    )

    df = df[
        df["model"] == model
    ]

    if horizon is not None:
        df = df[
            df["horizon"] == int(horizon)
        ]

    if df.empty:
        raise ValueError(
            "No matching forecasts found."
        )

    df = df.sort_values(
        "date"
    )

    pit = df[
        "pit"
    ].astype(float)

    if side == "left":
        hits = (
            pit <= alpha_level
        ).astype(int)

    elif side == "right":
        hits = (
            pit >= 1.0 - alpha_level
        ).astype(int)

    elif side == "two-sided":
        hits = (
            (
                pit
                <= alpha_level / 2.0
            )
            | (
                pit
                >= 1.0
                - alpha_level / 2.0
            )
        ).astype(int)

    else:
        raise ValueError(
            "side must be 'left', "
            "'right', or 'two-sided'."
        )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(9, 3)
        )
    else:
        fig = ax.figure

    ax.step(
        df["date"],
        hits,
        where="post",
        linewidth=1.5,
    )

    ax.set_ylim(
        -0.1,
        1.1,
    )

    title = (
        f"Hit Sequence: {model}, "
        f"{side}, alpha={alpha_level}"
    )

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
        raise ValueError(
            "No test results found."
        )

    if (
        category is not None
        and "category" in df.columns
    ):
        df = df[
            df["category"] == category
        ]

    if (
        horizon is not None
        and "horizon" in df.columns
    ):
        df = df[
            df["horizon"] == int(horizon)
        ]

    df = df.dropna(
        subset=["pvalue"]
    )

    if df.empty:
        raise ValueError(
            "No matching p-values found."
        )

    df = df.copy()

    df["label"] = (
        df["model"].astype(str)
        + " | "
        + df["test_id"].astype(str)
    )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(
                10,
                max(
                    4,
                    0.3 * len(df),
                ),
            )
        )
    else:
        fig = ax.figure

    y = np.arange(len(df))

    ax.barh(
        y,
        df["pvalue"].astype(float),
    )

    ax.axvline(
        0.05,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        df["label"]
    )

    ax.set_xlabel(
        "p-value"
    )

    ax.set_title(
        "Test p-values"
    )

    ax.set_xlim(
        0,
        1,
    )

    return fig, ax


def plot_holm(
    report,
    *,
    method: str = "HolmBonferroni",
    group_filter: Optional[dict] = None,
    ax=None,
):
    if not getattr(
        report,
        "adjusted_results",
        None,
    ):
        raise ValueError(
            "Report has no adjusted results."
        )

    if (
        method
        not in report.adjusted_results
    ):
        raise ValueError(
            f"Adjustment method "
            f"{method!r} not found."
        )

    df = report.adjusted_results[
        method
    ].copy()

    if group_filter is not None:
        for key, value in (
            group_filter.items()
        ):
            if key in df.columns:
                df = df[
                    df[key] == value
                ]

    if df.empty:
        raise ValueError(
            "No matching Holm results found."
        )

    df = df.dropna(
        subset=[
            "holm_rank",
            "pvalue",
            "holm_critical",
        ]
    )

    df = df.sort_values(
        "holm_rank"
    )

    if df.empty:
        raise ValueError(
            "No ranked Holm results found."
        )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(7, 4.5)
        )
    else:
        fig = ax.figure

    ax.scatter(
        df["holm_rank"],
        df["pvalue"],
        label="p-values",
    )

    ax.plot(
        df["holm_rank"],
        df["holm_critical"],
        linestyle="--",
        label="Holm critical values",
    )

    ax.set_title(
        "Holm-Bonferroni Step-Down Plot"
    )

    ax.set_xlabel(
        "Sorted rank"
    )

    ax.set_ylabel(
        "p-value"
    )

    ax.legend()

    return fig, ax


def _forecast_lookup(
    dataset,
    *,
    date,
    horizon,
):
    target = (
        pd.Timestamp(date)
        .tz_localize(None)
        .normalize()
    )

    lookup = {}

    for forecast in dataset.forecasts:
        forecast_date = (
            pd.Timestamp(
                forecast.date
            )
            .tz_localize(None)
            .normalize()
        )

        if forecast_date != target:
            continue

        if int(forecast.horizon) != int(horizon):
            continue

        lookup[
            forecast.model_name
        ] = forecast

    return lookup


def _inverse_cdf_bounds(
    forecast,
    *,
    tail_alpha: float,
) -> tuple[float, float]:
    """
    Return [Q(alpha), Q(1-alpha)] from this forecast's own CDF.
    """
    alpha = float(tail_alpha)

    if not (
        0.0 <= alpha < 0.5
    ):
        raise ValueError(
            "tail_alpha must satisfy "
            "0 <= tail_alpha < 0.5."
        )

    x = np.asarray(
        forecast.x_grid,
        dtype=float,
    )

    cdf = np.asarray(
        forecast.cdf,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(cdf)
    )

    x = x[valid]
    cdf = cdf[valid]

    if len(x) < 2:
        raise ValueError(
            f"Forecast {forecast.model_name!r} "
            "has too few finite CDF points."
        )

    order = np.argsort(x)
    x = x[order]
    cdf = cdf[order]

    cdf = np.clip(
        cdf,
        0.0,
        1.0,
    )

    cdf = np.maximum.accumulate(cdf)

    unique_cdf, unique_indices = (
        np.unique(
            cdf,
            return_index=True,
        )
    )

    unique_x = x[
        unique_indices
    ]

    lower_probability = alpha
    upper_probability = 1.0 - alpha

    if (
        lower_probability
        < unique_cdf[0]
        or upper_probability
        > unique_cdf[-1]
    ):
        raise ValueError(
            f"Forecast {forecast.model_name!r} "
            "does not span the requested "
            f"CDF probabilities "
            f"[{lower_probability:.6g}, "
            f"{upper_probability:.6g}]."
        )

    lower_x, upper_x = np.interp(
        [
            lower_probability,
            upper_probability,
        ],
        unique_cdf,
        unique_x,
    )

    return (
        float(lower_x),
        float(upper_x),
    )


def physical_tail_alpha_bounds(
    dataset,
    *,
    date,
    horizon,
    physical_models=None,
    tail_alpha: float = 0.01,
) -> dict[str, tuple[float, float]]:
    """
    Calculate model-specific central probability regions from each
    physical density's own CDF.

    alpha is removed from EACH tail. For alpha=0.01, every model is
    retained on its own central 98% probability region.
    """
    lookup = _forecast_lookup(
        dataset,
        date=date,
        horizon=horizon,
    )

    if physical_models is None:
        physical_models = list(
            lookup
        )

    bounds = {}

    for model in physical_models:
        if model not in lookup:
            continue

        bounds[model] = _inverse_cdf_bounds(
            lookup[model],
            tail_alpha=tail_alpha,
        )

    if not bounds:
        raise ValueError(
            "No requested physical models "
            "were available."
        )

    return bounds


def pricing_kernel_overlay_frame(
    dataset,
    *,
    date,
    horizon,
    rnd_model: str = "Raw RND",
    physical_models=None,
    normalization: Optional[str] = "at_zero",
    minimum_physical_density: float = 1e-10,
    truncate_by_physical_tail_alpha: Optional[
        float
    ] = None,
):
    """
    Compute m(x)=q(x)/p(x), truncating each pricing kernel by its
    corresponding physical density's own probability tails.
    """
    lookup = _forecast_lookup(
        dataset,
        date=date,
        horizon=horizon,
    )

    if rnd_model not in lookup:
        raise ValueError(
            f"Risk-neutral model "
            f"{rnd_model!r} is unavailable."
        )

    if physical_models is None:
        physical_models = [
            model
            for model in lookup
            if model != rnd_model
        ]

    if (
        truncate_by_physical_tail_alpha
        is not None
    ):
        physical_bounds = (
            physical_tail_alpha_bounds(
                dataset,
                date=date,
                horizon=horizon,
                physical_models=(
                    physical_models
                ),
                tail_alpha=float(
                    truncate_by_physical_tail_alpha
                ),
            )
        )
    else:
        physical_bounds = {}

    rnd = lookup[rnd_model]

    q_grid = np.asarray(
        rnd.x_grid,
        dtype=float,
    )

    q_pdf = np.asarray(
        rnd.pdf,
        dtype=float,
    )

    rows = []

    for model in physical_models:
        if model not in lookup:
            continue

        physical = lookup[model]

        x = np.asarray(
            physical.x_grid,
            dtype=float,
        )

        p_pdf = np.asarray(
            physical.pdf,
            dtype=float,
        )

        q_interpolated = np.interp(
            x,
            q_grid,
            q_pdf,
            left=np.nan,
            right=np.nan,
        )

        valid = (
            np.isfinite(x)
            & np.isfinite(p_pdf)
            & np.isfinite(
                q_interpolated
            )
            & (
                p_pdf
                > float(
                    minimum_physical_density
                )
            )
            & (
                q_interpolated >= 0.0
            )
        )

        model_bounds = physical_bounds.get(
            model
        )

        if model_bounds is not None:
            valid &= (
                (x >= model_bounds[0])
                & (x <= model_bounds[1])
            )

        kernel = np.full(
            len(x),
            np.nan,
            dtype=float,
        )

        kernel[valid] = (
            q_interpolated[valid]
            / p_pdf[valid]
        )

        if (
            normalization == "at_zero"
            and np.any(valid)
        ):
            valid_indices = np.flatnonzero(
                valid
            )

            anchor = valid_indices[
                np.argmin(
                    np.abs(
                        x[valid_indices]
                    )
                )
            ]

            scale = kernel[anchor]

            if (
                np.isfinite(scale)
                and scale != 0.0
            ):
                kernel[valid] /= scale

        elif (
            normalization == "mean"
            and np.any(valid)
        ):
            scale = np.nanmean(
                kernel[valid]
            )

            if (
                np.isfinite(scale)
                and scale != 0.0
            ):
                kernel[valid] /= scale

        elif normalization not in {
            None,
            "at_zero",
            "mean",
        }:
            raise ValueError(
                "normalization must be "
                "None, 'at_zero', or 'mean'."
            )

        for (
            x_value,
            physical_pdf,
            risk_neutral_pdf,
            pricing_kernel,
            is_valid,
        ) in zip(
            x,
            p_pdf,
            q_interpolated,
            kernel,
            valid,
        ):
            if not is_valid:
                continue

            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "horizon": int(horizon),
                    "model": model,
                    "rnd_model": rnd_model,
                    "x": float(x_value),
                    "physical_pdf": float(
                        physical_pdf
                    ),
                    "risk_neutral_pdf": float(
                        risk_neutral_pdf
                    ),
                    "pricing_kernel": float(
                        pricing_kernel
                    ),
                    "normalization": (
                        normalization
                    ),
                    "truncate_by_physical_tail_alpha": (
                        None
                        if (
                            truncate_by_physical_tail_alpha
                            is None
                        )
                        else float(
                            truncate_by_physical_tail_alpha
                        )
                    ),
                    "physical_lower_x": (
                        None
                        if model_bounds is None
                        else float(
                            model_bounds[0]
                        )
                    ),
                    "physical_upper_x": (
                        None
                        if model_bounds is None
                        else float(
                            model_bounds[1]
                        )
                    ),
                }
            )

    if not rows:
        raise ValueError(
            "No compatible pricing kernels "
            "were available after filtering."
        )

    return pd.DataFrame(rows)


def plot_density_pricing_kernel_overlay(
    dataset,
    *,
    date,
    horizon,
    models=None,
    rnd_model: str = "Raw RND",
    physical_models=None,
    xlim=None,
    truncate_by_physical_tail_alpha: Optional[float] = None,
    density_ylim=None,
    pricing_kernel_ylim=None,
    mark_realized: bool = True,
    normalization: Optional[str] = "at_zero",
    minimum_physical_density: float = 1e-10,
    show_densities: bool = True,
    show_pricing_kernels: bool = True,
    title=None,
    figsize=(10, 6),
):
    """Overlay forecast densities and/or pricing kernels.

    Parameters
    ----------
    show_densities
        Plot the risk-neutral and physical density curves.
    show_pricing_kernels
        Plot q(x) / p(x) for the selected physical models.

    At least one panel type must be enabled. When both are enabled, densities
    use the left y-axis and pricing kernels use the right y-axis. The function
    always returns ``(fig, (density_ax, kernel_ax))``; a disabled axis is None.
    """
    if not (show_densities or show_pricing_kernels):
        raise ValueError(
            "At least one of show_densities or "
            "show_pricing_kernels must be True."
        )

    if xlim is not None and truncate_by_physical_tail_alpha is not None:
        raise ValueError(
            "Specify either xlim or truncate_by_physical_tail_alpha, not both."
        )

    density_frame = density_overlay_frame(
        dataset,
        date=date,
        horizon=horizon,
        models=models,
    )

    lookup = _forecast_lookup(dataset, date=date, horizon=horizon)

    if physical_models is None:
        physical_models = [
            model
            for model in (models if models is not None else lookup.keys())
            if model != rnd_model
        ]
    physical_models = list(physical_models)

    if truncate_by_physical_tail_alpha is not None:
        model_bounds = physical_tail_alpha_bounds(
            dataset,
            date=date,
            horizon=horizon,
            physical_models=physical_models,
            tail_alpha=float(truncate_by_physical_tail_alpha),
        )

        if not model_bounds:
            raise ValueError("No physical-density bounds were available.")

        physical_union_lower = min(bound[0] for bound in model_bounds.values())
        physical_union_upper = max(bound[1] for bound in model_bounds.values())
        retained_parts = []

        for model, group in density_frame.groupby("model", sort=False):
            if model in model_bounds:
                lower, upper = model_bounds[model]
                group = group[(group["x"] >= lower) & (group["x"] <= upper)]
            elif model == rnd_model:
                group = group[
                    (group["x"] >= physical_union_lower)
                    & (group["x"] <= physical_union_upper)
                ]
            elif model in lookup:
                lower, upper = _inverse_cdf_bounds(
                    lookup[model],
                    tail_alpha=float(truncate_by_physical_tail_alpha),
                )
                group = group[(group["x"] >= lower) & (group["x"] <= upper)]

            if not group.empty:
                retained_parts.append(group)

        if not retained_parts:
            raise ValueError("Physical-tail truncation removed all plotted curves.")

        density_frame = pd.concat(retained_parts, ignore_index=True)
        plotting_xlim = (physical_union_lower, physical_union_upper)
    else:
        plotting_xlim = xlim

    kernel_frame = None
    if show_pricing_kernels:
        kernel_frame = pricing_kernel_overlay_frame(
            dataset,
            date=date,
            horizon=horizon,
            rnd_model=rnd_model,
            physical_models=physical_models,
            normalization=normalization,
            minimum_physical_density=minimum_physical_density,
            truncate_by_physical_tail_alpha=truncate_by_physical_tail_alpha,
        )

    fig, base_ax = plt.subplots(figsize=figsize)

    if show_densities:
        density_ax = base_ax
        kernel_ax = base_ax.twinx() if show_pricing_kernels else None
    else:
        density_ax = None
        kernel_ax = base_ax

    if show_densities:
        density_order = (
            list(models)
            if models is not None
            else list(density_frame["model"].drop_duplicates())
        )

        for model in density_order:
            group = density_frame[density_frame["model"] == model].sort_values("x")
            if group.empty:
                continue
            density_ax.plot(
                group["x"],
                group["pdf"],
                linewidth=1.8,
                linestyle="-",
                label=f"{model} density",
            )

    if show_pricing_kernels:
        for model in physical_models:
            group = kernel_frame[kernel_frame["model"] == model].sort_values("x")
            if group.empty:
                continue
            kernel_ax.plot(
                group["x"],
                group["pricing_kernel"],
                linewidth=1.8,
                linestyle="--",
                label=f"{model} pricing kernel",
            )

    display_ax = density_ax if density_ax is not None else kernel_ax

    if mark_realized:
        realized = float(density_frame["realized"].iloc[0])
        display_ax.axvline(
            realized,
            linestyle=":",
            linewidth=1.5,
            label=f"Realized = {realized:.4f}",
        )

    if plotting_xlim is not None:
        display_ax.set_xlim(*plotting_xlim)

    if show_densities and density_ylim is not None:
        density_ax.set_ylim(*density_ylim)

    if show_pricing_kernels and pricing_kernel_ylim is not None:
        kernel_ax.set_ylim(*pricing_kernel_ylim)

    display_ax.set_xlabel("Log return")

    if show_densities:
        density_ax.set_ylabel("Density")

    if show_pricing_kernels:
        kernel_label = "Pricing kernel q(x) / p(x)"
        if normalization == "at_zero":
            kernel_label += " (normalized to 1 at zero)"
        elif normalization == "mean":
            kernel_label += " (mean normalized)"
        kernel_ax.set_ylabel(kernel_label)

    display_ax.grid(alpha=0.25)

    handles = []
    labels = []
    for axis in (density_ax, kernel_ax):
        if axis is None:
            continue
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)

    if handles:
        display_ax.legend(handles, labels, loc="best", fontsize=8, ncol=2)

    if title is None:
        if show_densities and show_pricing_kernels:
            heading = "Forecast densities and pricing kernels"
        elif show_densities:
            heading = "Forecast densities"
        else:
            heading = "Pricing kernels"

        title = (
            f"{heading}\n"
            f"{pd.Timestamp(date).date()} ({int(horizon)}d)"
        )

        if truncate_by_physical_tail_alpha is not None:
            alpha = float(truncate_by_physical_tail_alpha)
            title += (
                "\nEach physical density truncated to its central "
                f"{100.0 * (1.0 - 2.0 * alpha):.1f}%"
            )

    display_ax.set_title(title)
    fig.tight_layout()

    return fig, (density_ax, kernel_ax)
