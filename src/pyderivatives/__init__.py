# pyderivatives/__init__.py
# Top-level convenience API for your project.
# Keep imports explicit to avoid circular-import hell.

# -------------------------
# yieldcurve (your current exports)
# -------------------------
from . import yieldcurve
from .yieldcurve.core import create_yield_curve
from .yieldcurve.build_yield_curve import build_yield_dataframe
from .yieldcurve.plotting_functions import plot_yield_curve, plot_yield_surface

# -------------------------
# conditional_pricing_kernel
# -------------------------
from . import conditional_pricing_kernel  # <-- ADD THIS

from .conditional_pricing_kernel.config import *
from .conditional_pricing_kernel.data import *
from .conditional_pricing_kernel.eval import *
from .conditional_pricing_kernel.fit import *
from .conditional_pricing_kernel.kernel import *
from .conditional_pricing_kernel.moments import *
from .conditional_pricing_kernel.bootstrap import *
from .conditional_pricing_kernel.cache import *
from .conditional_pricing_kernel.panel_plots import *
from .conditional_pricing_kernel.eval import evaluate_anchor_surfaces_with_theta_master

# (optional but common)
# -------------------------
# global_pricer
# -------------------------
from . import global_pricer
from .global_pricer.global_surface_pricer import GlobalSurfacePricer
from .global_pricer.plotting import surfaces, panels

# (optional common postprocess configs you use a lot)
from .global_pricer.postprocess.rnd import SafetyClipConfig
from .global_pricer.postprocess.iv import IVConfig

# -------------------------
# option_market_standardizer
# -------------------------
from . import option_market_standardizer
from .option_market_standardizer import OptionMarketStandardizer
from .option_market_standardizer.utils import summarize_put_call_parity_diff
from .option_market_standardizer.core import put_call_parity
from .option_market_standardizer.registry import VENDOR_REGISTRY

#
#
from .global_pricer.io import make_day_from_df

#
# -------------------------
# arbitrage_repair
# -------------------------
from . import arbitrage_repair
from .arbitrage_repair import RepairConfig, CallSurfaceArbRepair, repair_arb
###Post estimation
from . import post_estimation
from .post_estimation.post_estimation_functions import *



# (optional) also export plotters at the top-level convenience API
from .arbitrage_repair import plot_surface, plot_panels, plot_perturb, plot_term, plot_heatmap
# __all__ = [
#     # subpackages
#     "yieldcurve",
#     "global_pricer",
#     "option_market_standardizer",
#     "arbitrage_repair",

#     # yieldcurve exports
#     "create_yield_curve",
#     "build_yield_dataframe",
#     "plot_yield_curve",
#     "plot_yield_surface",

#     # pricingkernel exports

#     "conditional_pricing_kernel",
#     "ThetaSpec", "BootstrapSpec", "CacheSpec", "EvalSpec",
#     "estimate_pricing_kernel_global", "evaluate_anchor_surfaces_with_theta_master",

#     # global pricer exports
#     "GlobalSurfacePricer",
#     "surfaces",
#     "panels",
#     "SafetyClipConfig",
#     "IVConfig",
    
#     #io import
#     "make_day_from_df",

#     # option market standardizer exports
#     "OptionMarketStandardizer",
#     "summarize_put_call_parity_diff",
#     "put_call_parity",
#     "VENDOR_REGISTRY",

#     # arbitrage repair exports
#     "arbitrage_repair",
#     "RepairConfig",
#     "CallSurfaceArbRepair",
#     "repair_arb",
#     "plot_surface",
#     "plot_panels",
#     "plot_perturb",
#     "plot_term",
#     "plot_heatmap",
# ]
