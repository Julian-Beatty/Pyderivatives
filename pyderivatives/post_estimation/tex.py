import numpy as np
import pandas as pd
from pathlib import Path
from typing import Mapping, Sequence


def export_asym_qr_latex_pvals(
    asym_dict: Mapping[int, Mapping],
    *,
    eqs: Sequence[str] = ("A_var", "B_skew", "C_kurt"),
    out_dir: str | Path = "tables/asym_qr_tex",
    file_prefix: str = "btc_asym_qr",
    # formatting
    coef_decimals: int = 4,
    pval_decimals: int = 4,
    show_stars: bool = True,
    # optional: rename rows (regressors) for nicer LaTeX labels
    row_name_map: dict[str, str] | None = None,
) -> dict[int, dict[str, str]]:
    """
    Builds LaTeX tables like the screenshot:
      - columns = quantiles (taus)
      - rows = regressors
      - line 1: coefficient (optionally with stars)
      - line 2: (p-value)

    Uses keys from run_asym_quantreg_with_controls output:
      params      : eqd["params"]
      p-values    : eqd["p_boot_norm"]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def star_from_p(p: float) -> str:
        if (not show_stars) or (not np.isfinite(p)):
            return ""
        if p < 0.01:
            return r"\sym{***}"
        if p < 0.05:
            return r"\sym{**}"
        if p < 0.10:
            return r"\sym{*}"
        return ""

    latex_out: dict[int, dict[str, str]] = {}

    for h, hdict in asym_dict.items():
        latex_out[h] = {}

        for eq in eqs:
            eqd = hdict[eq]

            # Your actual keys (from screenshot)
            params = eqd["params"].copy()
            pvals  = eqd["p_boot_norm"].copy()

            if not (isinstance(params, pd.DataFrame) and isinstance(pvals, pd.DataFrame)):
                raise TypeError(f"h={h}, eq={eq}: expected DataFrames for params and p_boot_norm")

            if params.shape != pvals.shape:
                raise ValueError(f"h={h}, eq={eq}: params shape {params.shape} != pvals shape {pvals.shape}")

            # In your objects:
            #   rows = taus (0.05, 0.1, ...)
            #   cols = regressors
            # We want: rows=regressors, cols=taus
            params = params.T
            pvals  = pvals.T

            # Optional nicer regressor names
            if row_name_map is not None:
                params = params.rename(index=row_name_map)
                pvals  = pvals.rename(index=row_name_map)

            # Ensure columns are nicely formatted (taus)
            def _fmt_tau(x):
                try:
                    return f"{float(x):.2f}"
                except Exception:
                    return str(x)
            params.columns = [_fmt_tau(c) for c in params.columns]
            pvals.columns  = params.columns

            # ---- Build two-line per regressor table ----
            coef_tbl = pd.DataFrame(index=params.index, columns=params.columns, dtype=object)
            pval_tbl = pd.DataFrame(index=params.index, columns=params.columns, dtype=object)

            for r in params.index:
                for c in params.columns:
                    b = float(params.loc[r, c])
                    p = float(pvals.loc[r, c])
                    coef_tbl.loc[r, c] = f"{b:.{coef_decimals}f}{star_from_p(p)}"
                    pval_tbl.loc[r, c] = f"({p:.{pval_decimals}f})"

            rows, idx = [], []
            for r in coef_tbl.index:
                rows.append(coef_tbl.loc[r].tolist()); idx.append(r)
                rows.append(pval_tbl.loc[r].tolist()); idx.append("")

            table = pd.DataFrame(rows, index=idx, columns=coef_tbl.columns)

            # ---- LaTeX wrapper to look like paper tables ----
            body = table.to_latex(
                escape=False,
                na_rep="",
                column_format="l" + "c" * len(table.columns),
            )

            latex = (
                r"\begin{table}[!htbp]\centering" "\n"
                r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}" "\n"
                f"\\caption{{Estimates using quantile regressions ({eq}), horizon = {h} days.}}\n"
                f"\\label{{tab:{file_prefix}_{eq}_h{h}}}\n"
                + body
                + "\n\\begin{flushleft}\n\\footnotesize "
                  "Notes: P-values in parentheses. "
                  "\\sym{*} $p<0.10$, \\sym{**} $p<0.05$, \\sym{***} $p<0.01$.\n"
                  "\\end{flushleft}\n"
                + "\\end{table}\n"
            )

            latex_out[h][eq] = latex

            # Save to .tex
            path = out_dir / f"{file_prefix}_h{h}_{eq}.tex"
            path.write_text(latex, encoding="utf-8")

    return latex_out
