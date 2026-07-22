from .config import (
    EvaluationConfig,
    TransformCalibrationSpec,
)

from .preprocessing import (
    ReturnConfig,
    ReturnSeries,
    MarketData,
)

from .base import (
    ForecastDensity,
    EvaluationError,
    TestResult,
)

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

from .staggered_paths import (
    StaggeredPath,
    StaggeredPathCollection,
    StaggeredPathValidation,
    build_staggered_paths,
)

from .report import (
    DensityEvaluationReport,
    evaluate_dataset,
)

from .backtest import (
    DensityBacktest,
    BacktestJob,
)

from .registry import (
    ModelRegistry,
    TestRegistry,
)

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
    PattonIntervalHit,
    BootstrapInferenceResult,
    BootstrapStorageSpec,
    KolmogorovSmirnovCircularBootstrap,
    CramerVonMisesCircularBootstrap,
    BerkowitzLR3CircularBootstrap,
    KnuppelRawMoments,
    LobatoVelasco,
    DumitrescuIntervalGMM,
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
    plot_density_overlay,
    physical_tail_alpha_bounds,
    pricing_kernel_overlay_frame,
    plot_density_pricing_kernel_overlay,
)

from .bootstrap_plots import (
    bootstrap_statistics_from_result,
    plot_bootstrap_sampling_distribution,
    plot_all_bootstrap_sampling_distributions,
)


__all__ = [
    # Configuration
    "EvaluationConfig",
    "TransformCalibrationSpec",

    # Preprocessing
    "ReturnConfig",
    "ReturnSeries",
    "MarketData",

    # Base objects
    "ForecastDensity",
    "EvaluationError",
    "TestResult",
    "ForecastDataset",

    # Models
    "DensityModel",
    "RawRNDModel",
    "PhysicalDensityModel",
    "TransformRNDModel",
    "HistoricalKDEModel",
    "GARCHModel",

    # Runner and reports
    "run_backtest",
    "DensityEvaluationReport",
    "evaluate_dataset",
    "DensityBacktest",
    "BacktestJob",

    # Registries
    "ModelRegistry",
    "TestRegistry",

    # Model specifications
    "RawRND",
    "PhysicalDensity",
    "CRRA",
    "RossRecovery",
    "BetaCalibration",
    "ExponentialPolynomial",
    "NonparametricCalibration",
    "HistoricalKDE",
    "GARCH",

    # Calibration tests
    "DensityTest",
    "KolmogorovSmirnov",
    "CramerVonMises",
    "JarqueBera",
    "BerkowitzLR3",
    "BerkowitzLR1",
    "LjungBox",
    "KnuppelRawMoments",
    "LobatoVelasco",

    # Coverage tests
    "Kupiec",
    "ChristoffersenIndependence",
    "ChristoffersenConditionalCoverage",
    "PattonIntervalHit",
    "DumitrescuIntervalGMM",

    # Comparison tests
    "DieboldMariano",
    "AmisanoGiacomini",
    "TailWeightedAmisanoGiacomini",

    # Staggered-path inference
    "StaggeredNonOverlap",
    "holm_bonferroni_decision",
    "StaggeredPath",
    "StaggeredPathCollection",
    "StaggeredPathValidation",
    "build_staggered_paths",

    # Multiple testing
    "HolmBonferroni",
    "Bonferroni",

    # Bootstrap inference
    "BootstrapInferenceResult",
    "BootstrapStorageSpec",
    "KolmogorovSmirnovCircularBootstrap",
    "CramerVonMisesCircularBootstrap",
    "BerkowitzLR3CircularBootstrap",

    # Tables
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

    # Standard plots
    "plot_pit_histogram",
    "plot_pit_ecdf",
    "plot_pit_qq",
    "plot_normal_score_qq",
    "plot_logscore_time_series",
    "plot_cumulative_logscore_difference",
    "plot_hit_sequence",
    "plot_test_pvalues",
    "plot_holm",
    "density_overlay_frame",
    "plot_density_overlay",

    # Pricing-kernel plots
    "physical_tail_alpha_bounds",
    "pricing_kernel_overlay_frame",
    "plot_density_pricing_kernel_overlay",

    # Bootstrap plots
    "bootstrap_statistics_from_result",
    "plot_bootstrap_sampling_distribution",
    "plot_all_bootstrap_sampling_distributions",
]