from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np



@dataclass
class BehavioralConfig:
    """
    Configuration for the forward-looking behavioral overlay.

    enabled:
        Turn the overlay on/off.

    k1, k2, k3:
        Loading coefficients for optimism, confidence, and tail sentiment.

    iv_alpha:
        Percentile threshold for IV-change sentiment. Example: 0.05 uses 5/95.

    volume_alpha:
        Percentile threshold for volume/confidence sentiment. Example: 0.05 uses 5/95.

    tail_alpha:
        Tail probability used to define the left/right density tail regions.

    positive_skew_threshold, negative_skew_threshold:
        Fixed risk-neutral skewness thresholds for the tail-sentiment channel.
    """
    enabled: bool = False

    # Data used by the behavioral overlay.
    stock_df: Any = None
    stock_date_col: str = "date"
    volume_col: str = "volume"

    # Loading coefficients.
    k1: float = 1.0
    k2: float = 1.2
    k3: float = 1.0

    # Backward-compatible common threshold.
    sentiment_alpha: float = 0.05

    # Separate thresholds.
    iv_alpha: float = 0.05
    volume_alpha: float = 0.05
    tail_alpha: float = 0.05

    # Skew thresholds.
    positive_skew_threshold: float = 1.5
    negative_skew_threshold: float = -1.5

    # Safety clipping for multiplicative/dispersion adjustments.
    theta_min: float = 0.25
    theta_max: float = 4.0

    def validate(self):
        for name in ("sentiment_alpha", "iv_alpha", "volume_alpha", "tail_alpha"):
            val = float(getattr(self, name))
            if not (0.0 < val < 0.5):
                raise ValueError(f"{name} must lie in (0, 0.5).")

        if float(self.negative_skew_threshold) >= float(self.positive_skew_threshold):
            raise ValueError("negative_skew_threshold must be less than positive_skew_threshold.")

        if float(self.theta_min) <= 0:
            raise ValueError("theta_min must be positive.")

        if float(self.theta_max) <= float(self.theta_min):
            raise ValueError("theta_max must exceed theta_min.")

        return self


# ============================================================
# Transform specifications
# ============================================================

@dataclass(frozen=True)
class ThetaSpec:
    """
    Specification for the flexible exponential-polynomial pricing kernel.

    Kernel form:

        M(r, sigma)
            =
        exp(
            sum_i sum_k theta_{i,k} r^i sigma^{-k}
        )

    Parameters
    ----------
    N:
        Highest polynomial order in log return r.

    Ksig:
        Highest inverse-volatility interaction order.

    bounds:
        Optional lower and upper parameter bounds.
    """
    N: int = 2
    Ksig: int = 1
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None


@dataclass(frozen=True)
class ExponentialSpec:
    """
    Specification for the volatility-conditioned exponential pricing kernel.

    Kernel form:

        M(r, sigma)
            =
        exp(
            delta
            +
            sum_i c_i sigma^{-(b i)} r^i
        )

    The intercept delta is solved internally so that E_Q[M] = 1.
    """
    N: int = 2

    b_bounds: Tuple[float, float] = (-2.0, 2.0)
    c_bounds: Tuple[float, float] = (-25.0, 25.0)

    enforce_convexity: bool = False


@dataclass(frozen=True)
class BetaCalibrationSpec:
    """
    Specification for beta PIT calibration.

    The PIT values u_t = F_Q,t(r_t) are modeled as beta distributed.
    """
    a0: float = 1.0
    b0: float = 1.0

    a_bounds: Tuple[float, float] = (1e-3, 100.0)
    b_bounds: Tuple[float, float] = (1e-3, 100.0)

    @property
    def x0(self) -> np.ndarray:
        return np.array([self.a0, self.b0], dtype=float)


@dataclass(frozen=True)
class NonparametricCalibrationSpec:
    """
    Specification for nonparametric PIT calibration.

    Method:
        1. Compute PIT values u_t = F_Q,t(r_t).
        2. Transform to normal scores z_t = Phi^{-1}(u_t).
        3. Estimate the density h(z) by Gaussian KDE.
        4. Define the calibration map C(u) = H(Phi^{-1}(u)).

    Density transformation:

        f_P(x)
            =
        f_Q(x) h(z) / phi(z),

    where z = Phi^{-1}(F_Q(x)).
    """
    bandwidth: str | float = "silverman"

    z_grid_size: int = 1000
    z_grid_pad: float = 1.0

    min_bandwidth: float = 1e-3
    max_bandwidth: Optional[float] = None


# ============================================================
# Runtime and infrastructure specifications
# ============================================================

@dataclass(frozen=True)
class BootstrapSpec:
    """
    Bootstrap settings for transformed physical-density outputs.
    """
    enabled: bool = False

    B: int = 500
    block_length: int = 20

    ci_level: float = 0.95

    random_state: Optional[int] = 123

    keep_draws: bool = False


@dataclass(frozen=True)
class CacheSpec:
    """
    Optional disk-cache settings for fitted models and transformed outputs.
    """
    enabled: bool = False

    folder: str = "measure_transform_cache"

    cache_fit: bool = False
    cache_transform: bool = False

    dataset_tag: str = "default"


@dataclass(frozen=True)
class KeySpec:
    """
    Key names used to read RND dictionaries.

    This allows the transform layer to work with output dictionaries that
    use slightly different naming conventions.
    """

    x_grid_key: str = "grid_lr"
    pdf_surface_key: str = "rnd_lr_surface"
    cdf_surface_key: str = "rnd_cdf_surface"
    T_grid_key: str = "T_grid"

    spot_keys: Tuple[str, ...] = (
        "S0",
        "spot",
        "s0",
    )

    sigma_keys: Tuple[str, ...] = (
        "atm_vol",
        "sigma",
        "vol",
    )


# ============================================================
# Diagnostics
# ============================================================

@dataclass
class FitDiagnostics:
    """
    Summary of a single maturity-specific transform fit.
    """
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