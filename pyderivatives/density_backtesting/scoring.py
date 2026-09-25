from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence
import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass(frozen=True)
class ScoreConfig:
    """Configuration for grid-based density scoring rules.

    Every reported ``*_score`` is oriented so that larger values are better.
    The corresponding ``*_loss`` columns retain the conventional lower-is-better
    orientation where applicable.
    """

    quantile_levels: Sequence[float] = (
        0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99
    )
    left_event_threshold: float = -0.10
    right_event_threshold: float = 0.10
    weighted_crps_left_threshold: float = -0.05
    weighted_crps_right_threshold: float = 0.05
    pdf_floor: float = 1e-300
    derivative_floor: float = 1e-12

    def validate(self) -> "ScoreConfig":
        q = np.asarray(self.quantile_levels, dtype=float)
        if q.ndim != 1 or len(q) == 0 or np.any(~np.isfinite(q)):
            raise ValueError("quantile_levels must be a nonempty finite sequence.")
        if np.any((q <= 0.0) | (q >= 1.0)):
            raise ValueError("quantile_levels must lie strictly between 0 and 1.")
        if float(self.pdf_floor) <= 0.0:
            raise ValueError("pdf_floor must be positive.")
        if float(self.derivative_floor) <= 0.0:
            raise ValueError("derivative_floor must be positive.")
        return self


def _clean_density(x_grid, pdf, cdf):
    x = np.asarray(x_grid, dtype=float).reshape(-1)
    f = np.asarray(pdf, dtype=float).reshape(-1)
    F = np.asarray(cdf, dtype=float).reshape(-1)

    if not (x.shape == f.shape == F.shape) or len(x) < 3:
        raise ValueError("x_grid, pdf, and cdf must have the same length >= 3.")

    mask = np.isfinite(x) & np.isfinite(f) & np.isfinite(F)
    x, f, F = x[mask], f[mask], F[mask]
    order = np.argsort(x)
    x, f, F = x[order], f[order], F[order]

    unique = np.r_[True, np.diff(x) > 0]
    x, f, F = x[unique], f[unique], F[unique]
    if len(x) < 3:
        raise ValueError("Density grid has fewer than three unique points.")

    f = np.maximum(f, 0.0)
    area = float(np.trapezoid(f, x))
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Density has nonpositive or nonfinite area.")
    f = f / area

    F = np.maximum.accumulate(np.clip(F, 0.0, 1.0))
    span = float(F[-1] - F[0])
    if not np.isfinite(span) or span <= 0.0:
        dx = np.diff(x)
        F = np.r_[0.0, np.cumsum(0.5 * (f[1:] + f[:-1]) * dx)]
        F /= F[-1]
    else:
        F = (F - F[0]) / span
    F[0], F[-1] = 0.0, 1.0
    return x, f, F


def _integral(values, x):
    return float(np.trapezoid(np.asarray(values, dtype=float), x))


def _crps_components(x, F, realized):
    indicator = (x >= float(realized)).astype(float)
    squared_error = (F - indicator) ** 2
    return squared_error, _integral(squared_error, x)


def _quantile_from_cdf(x, F, levels):
    F_unique, index = np.unique(F, return_index=True)
    x_unique = x[index]
    return np.interp(levels, F_unique, x_unique)


def _pinball_loss(realized, quantiles, levels):
    error = float(realized) - np.asarray(quantiles, dtype=float)
    levels = np.asarray(levels, dtype=float)
    return float(np.mean(np.where(error >= 0.0, levels * error, (levels - 1.0) * error)))


def _dawid_sebastiani_loss(x, f, realized):
    mean = _integral(x * f, x)
    variance = _integral((x - mean) ** 2 * f, x)
    variance = max(float(variance), 1e-15)
    return float((float(realized) - mean) ** 2 / variance + np.log(variance)), mean, variance


def _hyvarinen_loss(x, f, realized, floor):
    log_f = np.log(np.maximum(f, float(floor)))
    first = np.gradient(log_f, x, edge_order=2)
    second = np.gradient(first, x, edge_order=2)
    first_y = float(np.interp(realized, x, first))
    second_y = float(np.interp(realized, x, second))
    return float(2.0 * second_y + first_y ** 2)


def _brier_loss(F, x, realized, threshold, side):
    p_left = float(np.interp(threshold, x, F))
    if side == "left":
        probability = p_left
        outcome = float(realized <= threshold)
    elif side == "right":
        probability = 1.0 - p_left
        outcome = float(realized >= threshold)
    else:
        raise ValueError("side must be 'left' or 'right'.")
    return float((probability - outcome) ** 2), probability, outcome


def _censored_log_score(F, x, f, realized, threshold, side, floor):
    density_y = max(float(np.interp(realized, x, f)), float(floor))
    cdf_threshold = float(np.interp(threshold, x, F))
    if side == "left":
        if realized <= threshold:
            return float(np.log(density_y))
        return float(np.log(max(1.0 - cdf_threshold, floor)))
    if side == "right":
        if realized >= threshold:
            return float(np.log(density_y))
        return float(np.log(max(cdf_threshold, floor)))
    raise ValueError("side must be 'left' or 'right'.")


def score_density(
    *,
    x_grid,
    pdf,
    cdf,
    realized: float,
    config: ScoreConfig | None = None,
) -> Dict[str, float]:
    """Calculate the complete univariate density-score suite."""
    config = (config or ScoreConfig()).validate()
    x, f, F = _clean_density(x_grid, pdf, cdf)
    y = float(realized)

    pdf_y = max(float(np.interp(y, x, f)), float(config.pdf_floor))
    log_score = float(np.log(pdf_y))

    crps_integrand, crps_loss = _crps_components(x, F, y)
    left_weight = (x <= float(config.weighted_crps_left_threshold)).astype(float)
    right_weight = (x >= float(config.weighted_crps_right_threshold)).astype(float)
    tail_weight = np.maximum(left_weight, right_weight)

    left_twcrps_loss = _integral(left_weight * crps_integrand, x)
    right_twcrps_loss = _integral(right_weight * crps_integrand, x)
    tail_twcrps_loss = _integral(tail_weight * crps_integrand, x)

    dss_loss, predictive_mean, predictive_variance = _dawid_sebastiani_loss(x, f, y)
    hyvarinen_loss = _hyvarinen_loss(x, f, y, config.derivative_floor)

    levels = np.asarray(config.quantile_levels, dtype=float)
    quantiles = _quantile_from_cdf(x, F, levels)
    quantile_loss = _pinball_loss(y, quantiles, levels)

    left_brier_loss, left_event_probability, left_event_outcome = _brier_loss(
        F, x, y, float(config.left_event_threshold), "left"
    )
    right_brier_loss, right_event_probability, right_event_outcome = _brier_loss(
        F, x, y, float(config.right_event_threshold), "right"
    )

    left_censored_log_score = _censored_log_score(
        F, x, f, y, float(config.left_event_threshold), "left", float(config.pdf_floor)
    )
    right_censored_log_score = _censored_log_score(
        F, x, f, y, float(config.right_event_threshold), "right", float(config.pdf_floor)
    )

    return {
        "log_score": log_score,
        "crps_loss": crps_loss,
        "crps_score": -crps_loss,
        "energy_loss": crps_loss,
        "energy_score": -crps_loss,
        "weighted_crps_left_loss": left_twcrps_loss,
        "weighted_crps_left_score": -left_twcrps_loss,
        "weighted_crps_right_loss": right_twcrps_loss,
        "weighted_crps_right_score": -right_twcrps_loss,
        "weighted_crps_tail_loss": tail_twcrps_loss,
        "weighted_crps_tail_score": -tail_twcrps_loss,
        "dawid_sebastiani_loss": dss_loss,
        "dawid_sebastiani_score": -dss_loss,
        "hyvarinen_loss": hyvarinen_loss,
        "hyvarinen_score": -hyvarinen_loss,
        "quantile_loss": quantile_loss,
        "quantile_score": -quantile_loss,
        "brier_left_loss": left_brier_loss,
        "brier_left_score": -left_brier_loss,
        "brier_right_loss": right_brier_loss,
        "brier_right_score": -right_brier_loss,
        "censored_log_left_score": left_censored_log_score,
        "censored_log_right_score": right_censored_log_score,
        "predictive_mean": predictive_mean,
        "predictive_variance": predictive_variance,
        "left_event_probability": left_event_probability,
        "left_event_outcome": left_event_outcome,
        "right_event_probability": right_event_probability,
        "right_event_outcome": right_event_outcome,
    }


def score_direction(column: str) -> int:
    """Return +1 when larger is better and -1 when smaller is better."""
    name = str(column)
    if name.endswith("_loss"):
        return -1
    return 1


def stationarity_diagnostics(values, *, alpha: float = 0.05, min_obs: int = 20) -> Dict[str, Any]:
    """Run complementary ADF and KPSS checks on a scalar time series."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    out: Dict[str, Any] = {
        "n": int(len(x)),
        "alpha": float(alpha),
        "adf_pvalue": None,
        "kpss_pvalue": None,
        "adf_reject_unit_root": None,
        "kpss_reject_stationarity": None,
        "stationary": None,
        "message": None,
    }
    if len(x) < int(min_obs):
        out["message"] = f"Too few observations for stationarity diagnostics; require {min_obs}."
        return out
    if np.nanstd(x) <= 1e-14:
        out.update({
            "adf_pvalue": 0.0,
            "kpss_pvalue": 1.0,
            "adf_reject_unit_root": True,
            "kpss_reject_stationarity": False,
            "stationary": True,
            "message": "Series is numerically constant.",
        })
        return out

    try:
        adf_result = adfuller(x, regression="c", autolag="AIC")
        adf_pvalue = float(adf_result[1])
        out["adf_pvalue"] = adf_pvalue
        out["adf_reject_unit_root"] = bool(adf_pvalue < alpha)
    except Exception as exc:
        out["message"] = f"ADF failed: {exc}"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(x, regression="c", nlags="auto")
        kpss_pvalue = float(kpss_result[1])
        out["kpss_pvalue"] = kpss_pvalue
        out["kpss_reject_stationarity"] = bool(kpss_pvalue < alpha)
    except Exception as exc:
        previous = out.get("message")
        out["message"] = (previous + "; " if previous else "") + f"KPSS failed: {exc}"

    if out["adf_reject_unit_root"] is not None and out["kpss_reject_stationarity"] is not None:
        out["stationary"] = bool(
            out["adf_reject_unit_root"] and not out["kpss_reject_stationarity"]
        )
        if not out["stationary"] and out["message"] is None:
            out["message"] = "ADF/KPSS do not jointly support covariance stationarity."
    return out
