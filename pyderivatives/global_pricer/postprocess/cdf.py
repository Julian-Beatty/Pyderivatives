from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CDFConfig:
    """
    If normalize=True, enforce CDF(T, last_K)=1 by renormalizing each row.
    clamp=True clamps CDF into [0,1] (good for numerical wiggles).
    """
    normalize: bool = True
    clamp: bool = True
    eps_area: float = 1e-16


def cdf_from_pdf_surface(
    pdf: np.ndarray,
    *,
    K_grid: np.ndarray,
    cfg: CDFConfig = CDFConfig(),
) -> np.ndarray:
    """
    Build CDF surface row-by-row:
      F(K) = ∫_{K_min}^K q(u) du

    Assumes pdf is q_T(K) over strike K.
    Returns array shape (nT, nK).
    """
    pdf = np.asarray(pdf, float)
    K_grid = np.asarray(K_grid, float).ravel()

    if pdf.ndim != 2:
        raise ValueError("pdf must be 2D (nT, nK).")
    if pdf.shape[1] != K_grid.size:
        raise ValueError("pdf.shape[1] must equal len(K_grid).")
    if np.any(np.diff(K_grid) <= 0):
        raise ValueError("K_grid must be strictly increasing.")

    nT, nK = pdf.shape
    out = np.full((nT, nK), np.nan, dtype=float)

    for i in range(nT):
        qi = np.asarray(pdf[i, :], float)
        if not np.all(np.isfinite(qi)):
            # still try: replace nonfinite with 0
            qi = np.where(np.isfinite(qi), qi, 0.0)

        # cumulative trapezoid
        cdf = np.zeros(nK, dtype=float)
        for j in range(1, nK):
            dx = K_grid[j] - K_grid[j - 1]
            cdf[j] = cdf[j - 1] + 0.5 * dx * (qi[j] + qi[j - 1])

        if cfg.normalize:
            area = float(cdf[-1])
            if np.isfinite(area) and area > cfg.eps_area:
                cdf = cdf / area

        if cfg.clamp:
            cdf = np.clip(cdf, 0.0, 1.0)

        out[i, :] = cdf

    return out
