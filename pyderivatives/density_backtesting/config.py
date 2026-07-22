# config.py

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from .preprocessing import ReturnConfig


@dataclass(frozen=True)
class TransformCalibrationSpec:
    """Per-model calibration settings for Q-to-P density transforms.

    The window is expressed in calendar time so it matches
    ``pricing_kernel.transform_one_date`` exactly.  Each TransformRNDModel
    may therefore use a different calibration protocol inside one backtest.
    """

    mode: str = "expanding"
    lookback_days: Optional[int] = None
    lookahead_days: int = 0
    fixed_end_date: Optional[Any] = None

    reserve_obs: int = 252
    min_fit_dates: int = 30
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if self.mode not in {"fixed", "expanding", "rolling", "centered"}:
            raise ValueError(
                "mode must be 'fixed', 'expanding', 'rolling', or 'centered'."
            )

        if self.mode == "fixed" and self.fixed_end_date is None:
            raise ValueError("fixed_end_date is required when mode='fixed'.")

        if self.mode == "rolling":
            if self.lookback_days is None or int(self.lookback_days) <= 0:
                raise ValueError(
                    "lookback_days must be positive when mode='rolling'."
                )

        if self.mode == "centered":
            if self.lookback_days is None or int(self.lookback_days) < 0:
                raise ValueError(
                    "lookback_days must be nonnegative when mode='centered'."
                )
            if int(self.lookahead_days) < 0:
                raise ValueError(
                    "lookahead_days must be nonnegative when mode='centered'."
                )

        if int(self.reserve_obs) < 0:
            raise ValueError("reserve_obs cannot be negative.")

        if int(self.min_fit_dates) < 1:
            raise ValueError("min_fit_dates must be at least 1.")

        return self

    def transform_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments forwarded to transform_one_date()."""
        self.validate()
        return {
            "mode": self.mode,
            "lookback_days": self.lookback_days,
            "lookahead_days": int(self.lookahead_days),
            "fixed_end_date": self.fixed_end_date,
            "reserve_obs": int(self.reserve_obs),
            "min_fit_dates": int(self.min_fit_dates),
            "fit_kwargs": dict(self.fit_kwargs),
        }


@dataclass(frozen=True)
class EvaluationConfig:
    target_maturity: float = 30 / 365
    horizon_days: Optional[int] = None

    horizons: Optional[Sequence[int]] = None
    target_maturities: Optional[Sequence[float]] = None
    maturity_selection: str = "target_grid"  # "target_grid" or "nearest_observed"
    observed_maturity_tol: Optional[float] = None
    path_reference_model: str = "Raw RND"
    shared_path_align_realized_horizon: bool = True

    maturity_match_tol: Optional[float] = None
    return_config: ReturnConfig = field(default_factory=ReturnConfig)
    realized_horizon_mode: str = "trading"   # "trading" or "calendar"
    realized_match_tol_days: int = 3

    evaluation_strategy: str = "all"         # "all", "single_path", or "shared_path"
    date_alignment: str = "model_specific"    # "model_specific", "intersection", or "reference"
    date_alignment_reference_model: Optional[str] = None

    window_type: str = "expanding"
    window_size: Optional[int] = None
    reserve_obs: int = 252
    min_fit_obs: int = 30

    eval_step: int = 1
    overlap: bool = True
    overlap_correction: str = "newey-west"

    tests: Sequence[Any] = field(default_factory=list)
    postprocessors: Sequence[Any] = field(default_factory=list)
    multiple_testing: Optional[Any] = None

    def resolved_horizon_days(self) -> int:
        if self.horizon_days is not None:
            return int(self.horizon_days)
        return int(round(365 * self.target_maturity))

    def resolved_horizons(self) -> list[int]:
        if self.horizons is not None:
            return [int(h) for h in self.horizons]

        if self.target_maturities is not None:
            return [int(round(365 * float(T))) for T in self.target_maturities]

        return [self.resolved_horizon_days()]

    def target_maturity_for_horizon(self, horizon_days: int) -> float:
        return float(horizon_days) / 365.0

    def validate(self):
        if self.window_type not in {"expanding", "rolling"}:
            raise ValueError("window_type must be 'expanding' or 'rolling'.")

        if self.window_type == "rolling" and self.window_size is None:
            raise ValueError("window_size must be supplied for rolling windows.")

        if self.eval_step < 1:
            raise ValueError("eval_step must be >= 1.")

        if self.overlap_correction not in {"none", "newey-west", "block-bootstrap"}:
            raise ValueError("Invalid overlap_correction.")

        if self.horizons is not None and self.target_maturities is not None:
            raise ValueError("Use either horizons or target_maturities, not both.")

        for h in self.resolved_horizons():
            if int(h) < 1:
                raise ValueError("All horizons must be >= 1.")
        if self.realized_horizon_mode not in {"trading", "calendar"}:
            raise ValueError("realized_horizon_mode must be 'trading' or 'calendar'.")
        
        if int(self.realized_match_tol_days) < 0:
            raise ValueError("realized_match_tol_days must be >= 0.")
        
            
        if self.date_alignment not in {"model_specific", "intersection", "reference"}:
            raise ValueError(
                "date_alignment must be 'model_specific', 'intersection', or 'reference'."
            )

        if self.date_alignment == "reference":
            ref = self.date_alignment_reference_model or self.path_reference_model
            if not ref:
                raise ValueError(
                    "date_alignment='reference' requires date_alignment_reference_model "
                    "or path_reference_model."
                )

        if self.evaluation_strategy not in {"all", "single_path", "shared_path"}:
            raise ValueError(
                "evaluation_strategy must be 'all', 'single_path', or 'shared_path'."
            )
        


        return self