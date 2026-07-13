# tests/multiple_testing.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HolmBonferroni:
    alpha: float = 0.05
    group_cols: Optional[Sequence[str]] = None
    pvalue_col: str = "pvalue"

    def adjust(self, results) -> pd.DataFrame:
        df = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)

        if df.empty:
            return df

        group_cols = list(self.group_cols or ["test_id", "model"])

        out = []

        for keys, sub in df.groupby(group_cols, dropna=False):
            tmp = sub.copy()
            p = pd.to_numeric(tmp[self.pvalue_col], errors="coerce")

            valid = p.notna() & np.isfinite(p)
            tmp["holm_rank"] = np.nan
            tmp["holm_critical"] = np.nan
            tmp["holm_reject"] = False
            tmp["holm_adjusted_pvalue"] = np.nan

            if valid.sum() == 0:
                out.append(tmp)
                continue

            idx = p[valid].sort_values().index
            m = len(idx)

            prev_adj = 0.0
            stopped = False

            for rank, ix in enumerate(idx, start=1):
                raw_p = float(p.loc[ix])
                crit = self.alpha / (m - rank + 1)

                adj_p = min(1.0, max(prev_adj, (m - rank + 1) * raw_p))
                prev_adj = adj_p

                reject = (raw_p <= crit) and not stopped
                if not reject:
                    stopped = True

                tmp.loc[ix, "holm_rank"] = rank
                tmp.loc[ix, "holm_critical"] = crit
                tmp.loc[ix, "holm_adjusted_pvalue"] = adj_p
                tmp.loc[ix, "holm_reject"] = bool(reject)

            out.append(tmp)

        return pd.concat(out, axis=0).sort_index()


@dataclass(frozen=True)
class Bonferroni:
    alpha: float = 0.05
    group_cols: Optional[Sequence[str]] = None
    pvalue_col: str = "pvalue"

    def adjust(self, results) -> pd.DataFrame:
        df = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(results)

        if df.empty:
            return df

        group_cols = list(self.group_cols or ["test_id", "model"])
        out = []

        for keys, sub in df.groupby(group_cols, dropna=False):
            tmp = sub.copy()
            p = pd.to_numeric(tmp[self.pvalue_col], errors="coerce")
            m = int(np.isfinite(p).sum())

            tmp["bonferroni_adjusted_pvalue"] = np.nan
            tmp["bonferroni_reject"] = False

            if m > 0:
                tmp["bonferroni_adjusted_pvalue"] = np.minimum(1.0, p * m)
                tmp["bonferroni_reject"] = tmp["bonferroni_adjusted_pvalue"] <= self.alpha

            out.append(tmp)

        return pd.concat(out, axis=0).sort_index()