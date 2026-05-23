from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np


# ============================================================
# Exponential-polynomial kernel specification
# ============================================================

@dataclass(frozen=True)
class ThetaSpec:
    """
    Exponential-polynomial pricing kernel specification.

    Kernel:

        M(r, sigma)
        =
        exp(
            sum_i sum_k theta_{i,k} * r^i * sigma^{-k}
        )

    Parameters
    ----------
    N:
        Highest polynomial order in return r.

    Ksig:
        Highest sigma interaction order.
    """
    N: int = 2
    Ksig: int = 1
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None


# ============================================================
# Conditional-risk kernel specification
# ============================================================

@dataclass(frozen=True)
class ConditionalRiskSpec:
    """
    Conditional-risk pricing kernel specification.

    Kernel:

        M(r,sigma)
        =
        exp(
            delta
            +
            sum_i c_i / sigma^(b i) * r^i
        )
    """
    N: int = 2

    b_bounds: Tuple[float, float] = (-2.0, 2.0)
    c_bounds: Tuple[float, float] = (-25.0, 25.0)

    enforce_convexity: bool = False


# ============================================================
# Beta calibration specification
# ============================================================

@dataclass(frozen=True)
class BetaCalibrationSpec:
    a0: float = 1.0
    b0: float = 1.0

    a_bounds: Tuple[float, float] = (1e-3, 100.0)
    b_bounds: Tuple[float, float] = (1e-3, 100.0)

    @property
    def x0(self):
        return np.array([self.a0, self.b0], dtype=float)


# ============================================================
# Nonparametric calibration specification
# ============================================================

@dataclass(frozen=True)
class NonparametricCalibrationSpec:
    """
    Nonparametric calibration specification.

    Method:
        1. Compute PIT values u_t = F_Q,t(r_t^realized).
        2. Transform z_t = Phi^{-1}(u_t).
        3. Estimate h(z) nonparametrically using Gaussian KDE.
        4. Define C(u) = H(Phi^{-1}(u)).

    Density transformation:

        f_P(x) = f_Q(x) * h(z) / phi(z),

    where:

        z = Phi^{-1}(F_Q(x)).

    bandwidth:
        "silverman" -> 1.06 * std(z) * n^(-1/5)
        "scott"     -> std(z) * n^(-1/5)
        float       -> manual bandwidth in z-space
    """
    bandwidth: str | float = "silverman"

    z_grid_size: int = 1000
    z_grid_pad: float = 1.0

    min_bandwidth: float = 1e-3
    max_bandwidth: Optional[float] = None


# ============================================================
# Bootstrap specification
# ============================================================

@dataclass(frozen=True)
class BootstrapSpec:
    enabled: bool = False

    B: int = 500
    block_length: int = 20

    ci_level: float = 0.95

    random_state: Optional[int] = 123

    keep_draws: bool = False


# ============================================================
# Cache specification
# ============================================================

@dataclass(frozen=True)
class CacheSpec:
    """
    Optional disk cache for fitted models or transformed outputs.
    """
    enabled: bool = False

    folder: str = "measure_transform_cache"

    cache_fit: bool = True
    cache_transform: bool = False

    dataset_tag: str = "default"


# ============================================================
# Input dictionary key specification
# ============================================================

@dataclass(frozen=True)
class KeySpec:
    """
    Naming convention for RND surface dictionaries.
    """

    # x-axis grid
    x_grid_key: str = "grid_lr"

    # density surface on log-return grid
    pdf_surface_key: str = "rnd_lr_surface"

    # CDF surface
    cdf_surface_key: str = "rnd_cdf_surface"

    # maturity grid
    T_grid_key: str = "T_grid"

    # spot aliases
    spot_keys: Tuple[str, ...] = (
        "S0",
        "spot",
        "s0",
    )

    # volatility aliases
    sigma_keys: Tuple[str, ...] = (
        "atm_vol",
        "sigma",
        "vol",
    )


# ============================================================
# Fit diagnostics
# ============================================================

@dataclass
class FitDiagnostics:
    maturity: float

    method: str

    n_total: int
    n_used: int
    n_dropped: int

    loss: float = np.nan
    loss_name: str = "not_applicable"

    status: str = "unknown"

    message: str = ""

    params: Dict[str, Any] = field(default_factory=dict)