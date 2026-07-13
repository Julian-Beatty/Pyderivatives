# tables.py

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


def test_table(
    report,
    *,
    category: Optional[str] = None,
    horizon: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = report.tests_frame()

    if df.empty:
        return df

    if category is not None and "category" in df.columns:
        df = df[df["category"] == category]

    if horizon is not None and "horizon" in df.columns:
        df = df[df["horizon"] == int(horizon)]

    if models is not None:
        df = df[df["model"].isin(models)]

    cols = [
        c for c in [
            "model",
            "horizon",
            "test_id",
            "test_name",
            "statistic",
            "pvalue",
            "reject",
            "sample_size",
            "degrees_of_freedom",
        ]
        if c in df.columns
    ]

    return df[cols].reset_index(drop=True)


def score_table(
    report,
    *,
    horizon: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = report.dataset.score_summary()

    if df.empty:
        return df

    if horizon is not None and "horizon" in df.columns:
        df = df[df["horizon"] == int(horizon)]

    if models is not None:
        df = df[df["model"].isin(models)]

    return df.reset_index(drop=True)


def pit_table(
    report,
    *,
    horizon: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = report.dataset.pit_summary()

    if df.empty:
        return df

    if horizon is not None and "horizon" in df.columns:
        df = df[df["horizon"] == int(horizon)]

    if models is not None:
        df = df[df["model"].isin(models)]

    return df.reset_index(drop=True)


def coverage_table(
    report,
    *,
    horizon: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    return test_table(
        report,
        category="coverage",
        horizon=horizon,
        models=models,
    )


def calibration_table(
    report,
    *,
    horizon: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = report.tests_frame()

    if df.empty:
        return df

    if "category" in df.columns:
        df = df[df["category"].isin(["calibration", "independence"])]

    if horizon is not None and "horizon" in df.columns:
        df = df[df["horizon"] == int(horizon)]

    if models is not None:
        df = df[df["model"].isin(models)]

    cols = [
        c for c in [
            "model",
            "horizon",
            "test_id",
            "statistic",
            "pvalue",
            "reject",
            "sample_size",
        ]
        if c in df.columns
    ]

    return df[cols].reset_index(drop=True)


def comparison_table(
    report,
    *,
    horizon: Optional[int] = None,
) -> pd.DataFrame:
    return test_table(
        report,
        category="comparison",
        horizon=horizon,
    )


def adjusted_table(
    report,
    *,
    method: Optional[str] = None,
) -> pd.DataFrame:
    if not getattr(report, "adjusted_results", None):
        return pd.DataFrame()

    if method is None:
        frames = []
        for name, df in report.adjusted_results.items():
            tmp = df.copy()
            tmp["adjustment_method"] = name
            frames.append(tmp)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return report.adjusted_results.get(method, pd.DataFrame()).copy()


def format_pvalues(
    df: pd.DataFrame,
    *,
    pvalue_cols: Sequence[str] = ("pvalue", "holm_adjusted_pvalue", "bonferroni_adjusted_pvalue"),
    digits: int = 4,
    stars: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    def fmt(x):
        if pd.isna(x):
            return ""
        x = float(x)
        s = f"{x:.{digits}f}"
        if stars:
            if x < 0.01:
                s += "***"
            elif x < 0.05:
                s += "**"
            elif x < 0.10:
                s += "*"
        return s

    for col in pvalue_cols:
        if col in out.columns:
            out[col] = out[col].map(fmt)

    return out


def latex_table(
    df: pd.DataFrame,
    *,
    index: bool = False,
    float_format: str = "%.4f",
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    if df.empty:
        return ""

    return df.to_latex(
        index=index,
        float_format=float_format,
        caption=caption,
        label=label,
        escape=False,
    )


def latex_test_table(
    report,
    *,
    category: Optional[str] = None,
    horizon: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    df = test_table(
        report,
        category=category,
        horizon=horizon,
        models=models,
    )

    df = format_pvalues(df)

    return latex_table(
        df,
        caption=caption,
        label=label,
    )


def latex_score_table(
    report,
    *,
    horizon: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    df = score_table(
        report,
        horizon=horizon,
        models=models,
    )

    return latex_table(
        df,
        caption=caption,
        label=label,
    )