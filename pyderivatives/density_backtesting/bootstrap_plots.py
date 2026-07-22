from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _metadata_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        for parser in (
            json.loads,
            ast.literal_eval,
        ):
            try:
                parsed = parser(value)

                if isinstance(parsed, dict):
                    return parsed

            except Exception:
                pass

    return {}


def bootstrap_statistics_from_result(
    result_or_row,
) -> tuple[np.ndarray, Optional[float]]:
    if hasattr(result_or_row, "metadata"):
        metadata = _metadata_dict(
            result_or_row.metadata
        )

        observed = getattr(
            result_or_row,
            "statistic",
            None,
        )

    else:
        # report.tests_frame() flattens TestResult.metadata into columns, so
        # support both a nested metadata object and a flattened pandas row.
        metadata = _metadata_dict(result_or_row.get("metadata"))
        try:
            flattened = result_or_row.to_dict()
        except AttributeError:
            flattened = dict(result_or_row)
        metadata = {**flattened, **metadata}

        observed = result_or_row.get("statistic")

    observed = metadata.get(
        "observed_statistic",
        observed,
    )

    inline = metadata.get(
        "bootstrap_statistics"
    )

    if inline is not None:
        values = np.asarray(
            inline,
            dtype=float,
        ).reshape(-1)

        return (
            values[np.isfinite(values)],
            observed,
        )

    path = (
        metadata.get(
            "bootstrap_statistics_path"
        )
        or metadata.get(
            "bootstrap_distribution_path"
        )
        or metadata.get(
            "bootstrap_path"
        )
    )

    if not path:
        raise ValueError(
            "This result does not contain stored "
            "bootstrap statistics."
        )

    path = Path(path)

    if path.suffix.lower() == ".npz":
        archive = np.load(
            path,
            allow_pickle=False,
        )

        for key in (
            "bootstrap_statistics",
            "statistics",
            "values",
            "arr_0",
        ):
            if key in archive:
                values = np.asarray(
                    archive[key],
                    dtype=float,
                )
                break

        else:
            raise KeyError(
                "No bootstrap statistic array "
                f"found in {path}."
            )

        if (
            observed is None
            and "observed_statistic" in archive
        ):
            candidate = np.asarray(
                archive[
                    "observed_statistic"
                ],
                dtype=float,
            ).reshape(-1)

            if (
                len(candidate)
                and np.isfinite(candidate[0])
            ):
                observed = float(
                    candidate[0]
                )

    elif path.suffix.lower() == ".npy":
        values = np.load(
            path,
            allow_pickle=False,
        )

    else:
        raise ValueError(
            "Bootstrap plot loader supports "
            ".npz and .npy files."
        )

    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    values = values[
        np.isfinite(values)
    ]

    return values, observed


def plot_bootstrap_sampling_distribution(
    result_or_row,
    *,
    bins: int = 50,
    show_kde: bool = True,
    show_observed: bool = True,
    show_critical_values: bool = True,
    two_sided: Optional[bool] = None,
    title: Optional[str] = None,
    ax=None,
):
    (
        values,
        observed,
    ) = bootstrap_statistics_from_result(
        result_or_row
    )

    if len(values) == 0:
        raise ValueError(
            "No finite bootstrap statistics found."
        )

    if hasattr(result_or_row, "metadata"):
        metadata = _metadata_dict(
            result_or_row.metadata
        )

        test_id = getattr(
            result_or_row,
            "test_id",
            "bootstrap test",
        )

        model = getattr(
            result_or_row,
            "model_name",
            "",
        )

    else:
        metadata = _metadata_dict(result_or_row.get("metadata"))
        try:
            flattened = result_or_row.to_dict()
        except AttributeError:
            flattened = dict(result_or_row)
        metadata = {**flattened, **metadata}

        test_id = result_or_row.get(
            "test_id",
            "bootstrap test",
        )

        model = result_or_row.get(
            "model",
            result_or_row.get(
                "model_name",
                "",
            ),
        )

    if two_sided is None:
        two_sided = bool(
            metadata.get(
                "bootstrap_two_sided",
                False,
            )
        )

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(8.5, 5)
        )
    else:
        fig = ax.figure

    ax.hist(
        values,
        bins=bins,
        density=True,
        alpha=0.65,
        edgecolor="black",
        label="Bootstrap statistics",
    )

    if (
        show_kde
        and len(np.unique(values)) > 2
    ):
        grid = np.linspace(
            values.min(),
            values.max(),
            500,
        )

        kde = stats.gaussian_kde(values)

        ax.plot(
            grid,
            kde(grid),
            linewidth=2,
            label="Bootstrap KDE",
        )

    if show_critical_values:
        if two_sided:
            low, high = np.quantile(
                values,
                [0.025, 0.975],
            )

            ax.axvline(
                low,
                linestyle=":",
                linewidth=1.5,
                label=(
                    "2.5% / 97.5% critical values"
                ),
            )

            ax.axvline(
                high,
                linestyle=":",
                linewidth=1.5,
            )

        else:
            high = float(
                np.quantile(
                    values,
                    0.95,
                )
            )

            ax.axvline(
                high,
                linestyle=":",
                linewidth=1.5,
                label="95% critical value",
            )

    if (
        show_observed
        and observed is not None
        and np.isfinite(
            float(observed)
        )
    ):
        ax.axvline(
            float(observed),
            linestyle="--",
            linewidth=2.2,
            label=(
                "Observed = "
                f"{float(observed):.4g}"
            ),
        )

    ax.set_title(
        title
        or (
            "Bootstrap sampling distribution\n"
            f"{test_id} | {model}"
        )
    )

    ax.set_xlabel(
        "Test statistic"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    return fig, ax


def plot_all_bootstrap_sampling_distributions(
    report,
    *,
    output_dir,
    bins: int = 50,
    show_kde: bool = True,
    show_observed: bool = True,
) -> list[Path]:
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = report.tests_frame()
    saved = []

    for (
        row_number,
        (_, row),
    ) in enumerate(
        frame.iterrows(),
        start=1,
    ):
        try:
            fig, _ = (
                plot_bootstrap_sampling_distribution(
                    row,
                    bins=bins,
                    show_kde=show_kde,
                    show_observed=show_observed,
                )
            )

        except (
            ValueError,
            FileNotFoundError,
            KeyError,
        ):
            continue

        test_id = str(
            row.get(
                "test_id",
                "bootstrap",
            )
        )

        model = str(
            row.get(
                "model",
                row.get(
                    "model_name",
                    "model",
                ),
            )
        )

        def safe(text):
            return "".join(
                character
                if (
                    character.isalnum()
                    or character in {
                        "-",
                        "_",
                    }
                )
                else "_"
                for character in text
            ).strip("_")

        destination = output_dir / (
            f"{row_number:04d}__"
            f"{safe(test_id)}__"
            f"{safe(model)}.png"
        )

        fig.savefig(
            destination,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        saved.append(destination)

    return saved
