from .config import EvaluationConfig
from .preprocessing import ReturnConfig, ReturnSeries, MarketData
from .base import ForecastDensity, EvaluationError, TestResult
from .forecast import ForecastDataset
from .models import (
    DensityModel,
    RawRNDModel,
    PhysicalDensityModel,
    TransformRNDModel,
    HistoricalKDEModel,
    GARCHModel,
)
from .runner import run_backtest
from .report import DensityEvaluationReport, evaluate_dataset
from .backtest import DensityBacktest, BacktestJob
from .registry import ModelRegistry, TestRegistry

from .specs import (
    RawRND,
    PhysicalDensity,
    CRRA,
    RossRecovery,
    BetaCalibration,
    ExponentialPolynomial,
    NonparametricCalibration,
    HistoricalKDE,
    GARCH,
)

from .tests import (
    DensityTest,
    KolmogorovSmirnov,
    CramerVonMises,
    JarqueBera,
    BerkowitzLR3,
    BerkowitzLR1,
    LjungBox,
    Kupiec,
    ChristoffersenIndependence,
    ChristoffersenConditionalCoverage,
    DieboldMariano,
    AmisanoGiacomini,
    TailWeightedAmisanoGiacomini,
    StaggeredNonOverlap,
    holm_bonferroni_decision,
    HolmBonferroni,
    Bonferroni,
    PattonIntervalHit
)

from .tables import (
    test_table,
    score_table,
    pit_table,
    coverage_table,
    calibration_table,
    comparison_table,
    adjusted_table,
    latex_table,
    latex_test_table,
    latex_score_table,
)

from .plots import (
    plot_pit_histogram,
    plot_pit_ecdf,
    plot_pit_qq,
    plot_normal_score_qq,
    plot_logscore_time_series,
    plot_cumulative_logscore_difference,
    plot_hit_sequence,
    plot_test_pvalues,
    plot_holm,
    density_overlay_frame,
    plot_density_overlay

)


__all__ = [
    "EvaluationConfig",
    "ReturnConfig",
    "ReturnSeries",
    "MarketData",
    "ForecastDensity",
    "EvaluationError",
    "TestResult",
    "ForecastDataset",
    "DensityModel",
    "RawRNDModel",
    "PhysicalDensityModel",
    "TransformRNDModel",
    "HistoricalKDEModel",
    "run_backtest",
    "DensityEvaluationReport",
    "evaluate_dataset",
    "DensityBacktest",
    "BacktestJob",
    "ModelRegistry",
    "TestRegistry",
    "RawRND",
    "PhysicalDensity",
    "CRRA",
    "RossRecovery",
    "BetaCalibration",
    "ExponentialPolynomial",
    "NonparametricCalibration",
    "HistoricalKDE",
    "DensityTest",
    "KolmogorovSmirnov",
    "CramerVonMises",
    "JarqueBera",
    "BerkowitzLR3",
    "BerkowitzLR1",
    "LjungBox",
    "Kupiec",
    "ChristoffersenIndependence",
    "ChristoffersenConditionalCoverage",
    "DieboldMariano",
    "AmisanoGiacomini",
    "TailWeightedAmisanoGiacomini",
    "StaggeredNonOverlap",
    "holm_bonferroni_decision",
    "HolmBonferroni",
    "Bonferroni",
    "test_table",
    "score_table",
    "pit_table",
    "coverage_table",
    "calibration_table",
    "comparison_table",
    "adjusted_table",
    "latex_table",
    "latex_test_table",
    "latex_score_table",
    "plot_pit_histogram",
    "plot_pit_ecdf",
    "plot_pit_qq",
    "plot_normal_score_qq",
    "plot_logscore_time_series",
    "plot_cumulative_logscore_difference",
    "plot_hit_sequence",
    "plot_test_pvalues",
    "plot_holm",
    "GARCHModel",
    "GARCH",
    "density_overlay_frame",
    "plot_density_overlay",
    "PattonIntervalHit"
]