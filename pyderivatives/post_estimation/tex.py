import numpy as np
import pandas as pd
from pathlib import Path
from typing import Mapping, Sequence, Optional
import re

import re

def _pretty_regressor_name(x: str) -> str:
    s = str(x).strip()

    # Constant
    if s.lower() in {"const", "constant", "intercept"}:
        return "Constant"

    # Ret+ / Ret-
    if s == "ret_pos":
        return r"$Ret^{+}$"
    if s == "ret_neg":
        return r"$Ret^{-}$"

    # ret_pos_L1, ret_neg_L2, ...
    m = re.match(r"^(ret_pos|ret_neg)_L(\d+)$", s)
    if m:
        base = m.group(1)
        k = int(m.group(2))
        sup = "+" if base == "ret_pos" else "-"
        return rf"$Ret^{{{sup}}}_{{t-{k}}}$"

    # d_var_L1, d_skew_L2, d_kurt_L1
    m = re.match(r"^(d_var|d_skew|d_kurt)_L(\d+)$", s)
    if m:
        mom = m.group(1)
        k = int(m.group(2))
        pretty = {"d_var": r"\Delta Var", "d_skew": r"\Delta Skew", "d_kurt": r"\Delta Kurt"}[mom]
        return rf"${pretty}_{{t-{k}}}$"

    # If you ever have contemporaneous deltas without lag:
    if s in {"d_var", "d_skew", "d_kurt"}:
        pretty = {"d_var": r"\Delta Var", "d_skew": r"\Delta Skew", "d_kurt": r"\Delta Kurt"}[s]
        return rf"${pretty}$"

    # fallback: raw
    return s.replace("_", r"\_")



def export_asym_qr_latex_pvals(
    asym_dict: Mapping[int, Mapping],
    *,
    eqs: Sequence[str] = ("A_var", "B_skew", "C_kurt"),
    out_dir: str | Path = "tables/asym_qr_tex",
    file_prefix: str = "btc_asym_qr",
    table_number: int = 2,                 # <-- for "Table 2:"
    coef_decimals: int = 4,
    pval_decimals: int = 4,
    show_stars: bool = True,
    transpose: bool = True,                # keep taus as columns
    custom_row_map: Optional[dict[str, str]] = None,  # overrides automatic labels
) -> dict[int, dict[str, str]]:

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

    def eq_title(eq: str) -> str:
        # show (A_var) exactly like screenshot; you can change to "Volatility" etc if you want
        return rf"({eq})"

    latex_out: dict[int, dict[str, str]] = {}

    for h, hdict in asym_dict.items():
        latex_out[h] = {}

        for eq in eqs:
            eqd = hdict[eq]

            params = eqd["params"].copy()
            pvals  = eqd["p_boot_norm"].copy()

            # rows=taus, cols=regressors -> we want rows=regressors, cols=taus
            if transpose:
                params = params.T
                pvals  = pvals.T

            # format tau columns like 0.05, 0.10, ...
            def _fmt_tau(x):
                try:
                    return f"{float(x):.2f}"
                except Exception:
                    return str(x)

            params.columns = [_fmt_tau(c) for c in params.columns]
            pvals.columns  = params.columns

            # row labels
            labels = []
            for r in params.index:
                r0 = str(r)
                if custom_row_map and r0 in custom_row_map:
                    labels.append(custom_row_map[r0])
                else:
                    labels.append(_pretty_regressor_name(r0))

            params.index = labels
            pvals.index  = labels

            # build coef line + pvalue line
            coef_tbl = pd.DataFrame(index=params.index, columns=params.columns, dtype=object)
            pv_tbl   = pd.DataFrame(index=params.index, columns=params.columns, dtype=object)

            for r in params.index:
                for c in params.columns:
                    b = float(params.loc[r, c])
                    p = float(pvals.loc[r, c])
                    coef_tbl.loc[r, c] = f"{b:.{coef_decimals}f}{star_from_p(p)}"
                    pv_tbl.loc[r, c]   = f"({p:.{pval_decimals}f})"

            # interleave rows
            rows, idx = [], []
            for r in coef_tbl.index:
                rows.append(coef_tbl.loc[r].tolist()); idx.append(r)
                rows.append(pv_tbl.loc[r].tolist());   idx.append("")

            table = pd.DataFrame(rows, index=idx, columns=coef_tbl.columns)

            # table body
            body = table.to_latex(
                escape=False,
                na_rep="",
                column_format="l" + "c" * len(table.columns),
            )

            # caption like screenshot (fix title)
            caption = rf"Estimates using quantile regressions ({eq}), horizon = {h} days."

            latex = (
                r"\begin{table}[!htbp]\centering" "\n"
                r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}" "\n"
                rf"\caption{{{caption}}}" "\n"
                rf"\label{{tab:{file_prefix}_{eq}_h{h}}}" "\n"
                + body +
                "\n\\begin{flushleft}\n\\footnotesize "
                "Notes: P-values in parentheses. "
                "* $p<0.10$, ** $p<0.05$, *** $p<0.01$.\n"
                "\\end{flushleft}\n"
                "\\end{table}\n"
            )

            latex_out[h][eq] = latex

            (out_dir / f"{file_prefix}_h{h}_{eq}.tex").write_text(latex, encoding="utf-8")

    return latex_out
