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
    repair_mode: Optional[str] = None,
    objective: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience functional wrapper.

    Repairs European calls and puts jointly by default using fixed-carry parity.
    ``repair_mode`` overrides ``cfg.repair_mode``; ``calls``/``puts`` filter the
    input. Missing option-right columns retain legacy all-call behavior.
    ``C_rep`` always denotes a call equivalent; ``price_rep`` denotes the row's
    actual right. Plotters automatically separate observed call and put quotes.
    ``objective='bid_ask'`` selects Cohen's soft-spread L1-BA objective, requiring
    finite bid < reference < ask (column names configurable in RepairConfig).
    ``objective='hybrid'`` additionally retains bid == reference == ask rows
    using normalized L1 penalties. Other invalid spread rows still raise.
    Omit it to retain the configured objective, default normalized_l1.

    Example
    -------
    out = repair_arb(option_day_df)
    plot_surface(out["plot_data"], save="surface.png")
    """
    if cfg is None:
        cfg = RepairConfig()
    rep = CallSurfaceArbRepair(cfg)
    return rep.repair_one_date(df_date, perturb_mode=perturb_mode, perturb_eps=perturb_eps, repair_mode=repair_mode, objective=objective)
