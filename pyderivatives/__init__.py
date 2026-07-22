# pyderivatives/__init__.py

# =========================================================
# Core packages
# =========================================================

from . import demodata
from . import yieldcurve
from . import global_pricer
from . import option_market_standardizer
from . import arbitrage_repair
from . import post_estimation
from . import pricing_kernel
from . import density_evaluation
from . import dealer_positioning

# =========================================================
# Yield curve
# =========================================================

from .yieldcurve.core import create_yield_curve
from .yieldcurve.build_yield_curve import build_yield_dataframe
from .yieldcurve.plotting_functions import (
    plot_yield_curve,
    plot_yield_surface,
)


# =========================================================
# Global pricer
# =========================================================

from .global_pricer.global_surface_pricer import GlobalSurfacePricer
from .global_pricer.plotting import surfaces, panels
from .global_pricer.io import make_day_from_df

from .global_pricer.postprocess.rnd import SafetyClipConfig
from .global_pricer.postprocess.iv import IVConfig


# =========================================================
# Option market standardizer
# =========================================================

from .option_market_standardizer import OptionMarketStandardizer

from .option_market_standardizer.utils import (
    summarize_put_call_parity_diff,
)

from .option_market_standardizer.core import (
    put_call_parity,
)

from .option_market_standardizer.registry import (
    VENDOR_REGISTRY,
)


# =========================================================
# Arbitrage repair
# =========================================================

from .arbitrage_repair import (
    RepairConfig,
    CallSurfaceArbRepair,
    repair_arb,
)

from .arbitrage_repair import (
    plot_surface,
    plot_panels,
    plot_perturb,
    plot_term,
    plot_heatmap,
)


# =========================================================
# Density evaluation
# =========================================================
from .density_backtesting import *

# from .density_evaluation import (

#     # Evaluators
#     ForecastDensity,
#     DensityEvaluationResults,
#     OptionImpliedDensityEvaluator,
#     HistoricalKDEDensityEvaluator,
#     GARCHDensityEvaluator,
#     RNDDensityEvaluator,

#     # Utilities
#     get_fit_dates,
#     load_forecast_part,
#     merge_forecast_parts,

#     # PIT tests
#     pit_summary,
#     pit_uniformity_test,
#     z_diagnostics,
#     berkowitz_lr3_test,
#     evaluate_pit_tests,
#     autocorrelation_moment_tests,

#     # Hit tests
#     hit_test,
#     standard_hit_tests,
#     patton_hit_test,
#     standard_patton_hit_tests,
#     patton_union_hit_test,
#     paper_hit_test_table,
#     paper_hit_test_pvalue_table,

#     # Model comparison
#     diebold_mariano_test,
#     pairwise_dm_table,

#     # Plots
#     plot_pit_histogram,
#     plot_pit_qq,
#     plot_cumulative_log_scores,
#     plot_forecast_density,
#     plot_density_comparison,
# )


# =========================================================
# Pricing kernels
# =========================================================
# =========================================================
# Pricing kernel
# =========================================================

from . import pricing_kernel

from .pricing_kernel import (

    # Registry
    get_transform,
    available_transforms,
    register_transform,

    # Configs
    ThetaSpec,
    ExponentialSpec,
    BetaCalibrationSpec,
    NonparametricCalibrationSpec,
    BootstrapSpec,
    CacheSpec,
    KeySpec,
    FitDiagnostics,


    # Methods
    ExponentialKernel,
    ExponentialPolynomialKernel,
    BetaCalibration,
    NonparametricCalibration,
    CRRAKernel,

    # Plots
    plot_surface,
    plot_surface_panels,
    plot_pqk_multipanel,
    plot_pricing_kernel_surface,
    plot_physical_density_surface,
    plot_rnd_surface,
    plot_rra_surface,
    plot_physical_density_panels,
    plot_rnd_panels,
    plot_pricing_kernel_panels,
    plot_rra_panels,
    plot_surface_3d_by_T,
    plot_pit_calibration_panels,
    plot_pqk_time_panels
)
# =========================================================
# Dealer positioning
# =========================================================

from .dealer_positioning import *

# =========================================================
# Post estimation
# =========================================================

from .post_estimation.multiplots_error_diagn import *
from .post_estimation.quantilereg import *
from .post_estimation.TVP_QSVAR import *
from .post_estimation.wavelets import *
from .post_estimation.tex import *
from .post_estimation.utils import *
from .post_estimation.generalizedquantilesreg import *

# =========================================================
# Useful
# =========================================================
from .Useful_functions.merging_helpers import *


# =========================================================
# Automatic export list
# =========================================================

__all__ = [
    name
    for name in globals()
    if not name.startswith("_")
]
