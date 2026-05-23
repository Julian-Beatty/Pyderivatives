from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

@dataclass(frozen=True)
class ThetaSpec:
    N: int = 2
    Ksig: int = 1
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (lb, ub) length p

@dataclass(frozen=True)
class BootstrapSpec:
    enabled: bool = False
    n_boot: int = 10
    block_len: int = 21
    ci_levels: Tuple[float, float] = (0.05, 0.95)
    random_state: Optional[int] = 123
    keep_draws: bool = True


from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class EvalSpec:
    r_bounds: Tuple[float, float] = (-4.75, 1.75)  # log-return bounds
    r_grid_size: int = 350

    @classmethod
    def from_R_bounds(cls, R_bounds: Tuple[float, float], r_grid_size: int = 350) -> "EvalSpec":
        Rmin, Rmax = map(float, R_bounds)
        if not (Rmin > 0.0 and Rmax > 0.0 and Rmax > Rmin):
            raise ValueError("R_bounds must be positive with Rmax > Rmin.")
        return cls(r_bounds=(float(np.log(Rmin)), float(np.log(Rmax))), r_grid_size=int(r_grid_size))


@dataclass(frozen=True)
class CacheSpec:
    use_disk: bool = True
    folder: str = "pk_cache"

@dataclass(frozen=True)
class SafetyClipSpec:
    """Post-processing safety clip for the *physical* density.

    Behavior:
      - Find the mode (argmax) on the evaluation grid
      - Moving left from the mode: enforce non-increasing (toward tails)
      - Moving right from the mode: enforce non-increasing (toward tails)
      - Renormalize so the density integrates to 1 on the grid (in the same measure)

    Notes:
      - This is intentionally a *post* step: it does NOT change M or theta fitting.
      - Set enabled=False to disable.
    """
    enabled: bool = False
    # optional floor to avoid exact zeros before renormalization
    floor: float = 0.0
