# config.py

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .preprocessing import ReturnConfig


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