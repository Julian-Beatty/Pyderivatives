from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd

from .core import RepairConfig, CallSurfaceArbRepair


def repair_arb(
    df_date: pd.DataFrame,
    *,
    cfg: Optional[RepairConfig] = None,
    perturb_mode: str = "absolute",
    perturb_eps: float = 1e-10,
) -> Dict[str, Any]:
    """
    Convenience functional wrapper.

    Example
    -------
    out = repair_arb(option_day_df)
    plot_surface(out["plot_data"], save="surface.png")
    """
    if cfg is None:
        cfg = RepairConfig()
    rep = CallSurfaceArbRepair(cfg)
    return rep.repair_one_date(df_date, perturb_mode=perturb_mode, perturb_eps=perturb_eps)
