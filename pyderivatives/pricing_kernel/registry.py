from __future__ import annotations

from typing import Dict, List


TRANSFORM_REGISTRY: Dict[str, type] = {}


def register_transform(name: str):
    """Register a measure transformation class."""
    def decorator(cls):
        key = str(name).strip().lower()
        if key in TRANSFORM_REGISTRY:
            raise ValueError(f"Transform method '{key}' is already registered.")
        TRANSFORM_REGISTRY[key] = cls
        cls.method_name = key
        return cls
    return decorator


def get_transform(name: str, **kwargs):
    """
Construct a registered pricing-kernel transform.

Parameters
----------
name : str
    Transform name. Available transforms:

    ------------------------------------------------------------------
    "crra"
    ------------------------------------------------------------------

    CRRAKernel(
        gamma: float = 2.0,
        behavioral: bool = False,
        stock_df: pd.DataFrame | None = None,
        volume_col: str = "volume",
        k1: float = 1.0,
        k2: float = 1.2,
        k3: float = 1.0,
        sentiment_alpha: float = 0.05,
        ...
    )

    Example
    -------
    get_transform(
        "crra",
        gamma=2.0,
    )

    ------------------------------------------------------------------
    "exponential"
    ------------------------------------------------------------------

    ExponentialKernel(
        spec: ExponentialSpec = ExponentialSpec(N=2),
        min_obs: int = 30,
        fit_trim_alpha: tuple | None = None,
        fit_maturities: list[float] | None = None,
        maturity_match_tol: float | None = None,
        behavioral: bool = False,
        stock_df: pd.DataFrame | None = None,
        volume_col: str = "volume",
        k1: float = 1.0,
        k2: float = 1.2,
        k3: float = 1.0,
        sentiment_alpha: float = 0.05,
        ...
    )

    Example
    -------
    get_transform(
        "exponential",
        spec=ExponentialSpec(N=2),
    )

    ------------------------------------------------------------------
    "exponential_polynomial"
    ------------------------------------------------------------------

    ExponentialPolynomialKernel(
        theta_spec: ThetaSpec = ThetaSpec(N=2, Ksig=1),
        min_obs: int = 30,
        fit_trim_alpha: tuple | None = None,
        fit_maturities: list[float] | None = None,
        maturity_match_tol: float | None = None,
        behavioral: bool = False,
        stock_df: pd.DataFrame | None = None,
        volume_col: str = "volume",
        k1: float = 1.0,
        k2: float = 1.2,
        k3: float = 1.0,
        sentiment_alpha: float = 0.05,
        ...
    )

    Example
    -------
    get_transform(
        "exponential_polynomial",
        theta_spec=ThetaSpec(N=2, Ksig=1),
    )

    ------------------------------------------------------------------
    "beta"
    ------------------------------------------------------------------

    BetaCalibration(
        spec: BetaCalibrationSpec = BetaCalibrationSpec(),
        min_obs: int = 30,
        fit_trim_alpha: tuple | None = None,
        fit_maturities: list[float] | None = None,
        maturity_match_tol: float | None = None,
        behavioral: bool = False,
        ...
    )

    Example
    -------
    get_transform(
        "beta",
        spec=BetaCalibrationSpec(),
    )

    ------------------------------------------------------------------
    "nonparametric"
    ------------------------------------------------------------------

    NonparametricCalibration(
        spec: NonparametricCalibrationSpec
            = NonparametricCalibrationSpec(
                bandwidth="silverman"
            ),
        min_obs: int = 30,
        fit_trim_alpha: tuple | None = None,
        fit_maturities: list[float] | None = None,
        maturity_match_tol: float | None = None,
        behavioral: bool = False,
        ...
    )

    Example
    -------
    get_transform(
        "nonparametric",
        spec=NonparametricCalibrationSpec(
            bandwidth="silverman"
        ),
    )

    ------------------------------------------------------------------
    Aliases
    ------------------------------------------------------------------

    "beta_calibration"       -> "beta"
    "np_calibration"         -> "nonparametric"
    "nonparametric_calibration"
                            -> "nonparametric"
    "crra_kernel"           -> "crra"

Returns
-------
MeasureTransform
    Instantiated transform object.

Examples
--------
>>> get_transform("crra", gamma=2)

>>> get_transform(
...     "exponential",
...     spec=ExponentialSpec(N=2)
... )

>>> get_transform(
...     "exponential_polynomial",
...     theta_spec=ThetaSpec(N=2, Ksig=1)
... )
"""
    key = str(name).strip().lower()
    if key not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform '{name}'. Available methods: {available_transforms()}"
        )
    return TRANSFORM_REGISTRY[key](**kwargs)


def available_transforms() -> List[str]:
    return sorted(TRANSFORM_REGISTRY.keys())