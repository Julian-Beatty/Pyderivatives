# specs.py

from __future__ import annotations

from typing import Optional, Dict, Any

from pyderivatives.pricing_kernel import get_transform
from .models import (
    RawRNDModel,
    PhysicalDensityModel,
    TransformRNDModel,
    HistoricalKDEModel,
)


def RawRND(*, rnd_key: str, name: Optional[str] = None, **metadata):
    return RawRNDModel(
        name=name or f"Raw RND [{rnd_key}]",
        rnd_key=rnd_key,
        metadata=metadata,
    )


def PhysicalDensity(
    *,
    physical_key: str,
    name: Optional[str] = None,
    **metadata,
):
    return PhysicalDensityModel(
        name=name or f"Physical Density [{physical_key}]",
        physical_key=physical_key,
        metadata=metadata,
    )


def CRRA(
    *,
    rnd_key: str,
    gamma: float = 2.0,
    name: Optional[str] = None,
    behavioral: bool = False,
    transform_kwargs: Optional[Dict[str, Any]] = None,
    **metadata,
):
    kwargs = {} if transform_kwargs is None else dict(transform_kwargs)
    transform = get_transform(
        "crra",
        gamma=gamma,
        behavioral=behavioral,
        **kwargs,
    )

    return TransformRNDModel(
        name=name or f"CRRA gamma={gamma} [{rnd_key}]",
        rnd_key=rnd_key,
        transform=transform,
        requires_fit=True,
        clone_transform=True,
        metadata=metadata,
    )


def RossRecovery(
    *,
    rnd_key: str,
    name: Optional[str] = None,
    transform_kwargs: Optional[Dict[str, Any]] = None,
    **metadata,
):
    transform = get_transform(
        "ross_recovery",
        **({} if transform_kwargs is None else dict(transform_kwargs)),
    )

    return TransformRNDModel(
        name=name or f"Ross Recovery [{rnd_key}]",
        rnd_key=rnd_key,
        transform=transform,
        requires_fit=False,
        clone_transform=True,
        metadata=metadata,
    )


def BetaCalibration(
    *,
    rnd_key: str,
    name: Optional[str] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    transform_kwargs: Optional[Dict[str, Any]] = None,
    **metadata,
):
    transform = get_transform(
        "beta",
        **({} if transform_kwargs is None else dict(transform_kwargs)),
    )

    return TransformRNDModel(
        name=name or f"Beta Calibration [{rnd_key}]",
        rnd_key=rnd_key,
        transform=transform,
        requires_fit=True,
        clone_transform=True,
        fit_kwargs={} if fit_kwargs is None else dict(fit_kwargs),
        metadata=metadata,
    )


def ExponentialPolynomial(
    *,
    rnd_key: str,
    name: Optional[str] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    transform_kwargs: Optional[Dict[str, Any]] = None,
    **metadata,
):
    transform = get_transform(
        "exponential_polynomial",
        **({} if transform_kwargs is None else dict(transform_kwargs)),
    )

    return TransformRNDModel(
        name=name or f"Exponential Polynomial [{rnd_key}]",
        rnd_key=rnd_key,
        transform=transform,
        requires_fit=True,
        clone_transform=True,
        fit_kwargs={} if fit_kwargs is None else dict(fit_kwargs),
        metadata=metadata,
    )


def NonparametricCalibration(
    *,
    rnd_key: str,
    name: Optional[str] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    transform_kwargs: Optional[Dict[str, Any]] = None,
    **metadata,
):
    transform = get_transform(
        "nonparametric",
        **({} if transform_kwargs is None else dict(transform_kwargs)),
    )

    return TransformRNDModel(
        name=name or f"Nonparametric Calibration [{rnd_key}]",
        rnd_key=rnd_key,
        transform=transform,
        requires_fit=True,
        clone_transform=True,
        fit_kwargs={} if fit_kwargs is None else dict(fit_kwargs),
        metadata=metadata,
    )


def HistoricalKDE(
    *,
    name: str = "Historical KDE",
    grid_size: int = 500,
    grid_pad: float = 0.25,
    **metadata,
):
    return HistoricalKDEModel(
        name=name,
        grid_size=grid_size,
        grid_pad=grid_pad,
        metadata=metadata,
    )
def GARCH(
    *,
    name: str = "GARCH(1,1)-t",
    p: int = 1,
    q: int = 1,
    distribution: str = "t",
    simulations: int = 20000,
    grid_size: int = 500,
    grid_pad: float = 0.25,
    random_state: Optional[int] = 123,
    scale_returns: float = 100.0,
    **metadata,
):
    from .models import GARCHModel

    return GARCHModel(
        name=name,
        p=p,
        q=q,
        distribution=distribution,
        simulations=simulations,
        grid_size=grid_size,
        grid_pad=grid_pad,
        random_state=random_state,
        scale_returns=scale_returns,
        metadata=metadata,
    )