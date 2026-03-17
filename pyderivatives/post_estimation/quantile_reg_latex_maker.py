from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Union


# =========================
# Parsing: quantile QR .tex
# =========================

_TABULAR_BEGIN_RE = re.compile(r"\\begin\{tabular\}\{.*?\}", re.I)
_TABULAR_END_RE   = re.compile(r"\\end\{tabular\}", re.I)

# header row like: "& 0.05 & 0.10 & ... \\"
_HEADER_RE = re.compile(r"^\s*&\s*([0-9].*?)\\\\\s*$")

# detect "(0.1234)" style cells
_P_IN_PARENS_RE = re.compile(r"^\(?\s*[-+]?\d+(?:\.\d+)?\s*\)?$")

def _strip_row_end(s: str) -> str:
    return re.sub(r"\\\\\s*$", "", s.strip())

def _split_ampersands(line: str) -> List[str]:
    s = _strip_row_end(line)
    return [p.strip() for p in s.split("&")]

def _is_rule_line(ln: str) -> bool:
    s = ln.strip()
    return (
        not s
        or s.startswith("\\hline")
        or s in {"\\toprule", "\\midrule", "\\bottomrule"}
    )

def _norm_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    return s

def _looks_like_paren_row(line: str, n_q: int) -> bool:
    """
    True if line starts with '&' and has n_q cells that look like '(...)' numbers.
    Works for p-values or SEs.
    """
    parts = _split_ampersands(line)
    if len(parts) != n_q + 1:
        return False
    if parts[0] != "":
        return False
    # allow "(0.0000)" or "0.0000" inside parentheses already stripped later
    for c in parts[1:]:
        c2 = c.strip()
        c2 = c2.strip()
        # accept "(0.0000)" or "( 0.0000 )"
        c2 = c2.strip()
        c2_inner = c2.strip("()").strip()
        if not _P_IN_PARENS_RE.match(c2_inner):
            return False
    return True

def parse_quantreg_tex(
    tex_path: Union[str, Path],
) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, Optional[str]]]]]:
    """
    Returns:
      quantiles: ["0.05","0.10",...]
      data[var] = {
        "coef": {q: coef_cell_str},
        "p":    {q: p_str_without_parens OR None if missing}
      }

    Stars are preserved because coef_cell_str is taken verbatim from the LaTeX cell.
    """
    tex_path = Path(tex_path)
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # Grab first tabular environment
    in_tabular = False
    tab_lines: List[str] = []
    for ln in lines:
        if _TABULAR_BEGIN_RE.search(ln):
            in_tabular = True
        if in_tabular:
            tab_lines.append(ln)
        if in_tabular and _TABULAR_END_RE.search(ln):
            break
    if not tab_lines:
        raise ValueError(f"No tabular found in: {tex_path}")

    # Quantile header
    quantiles: Optional[List[str]] = None
    header_idx: Optional[int] = None
    for i, ln in enumerate(tab_lines):
        if _HEADER_RE.match(ln):
            parts = _split_ampersands(ln)  # ["", "0.05", ...]
            quantiles = [p for p in parts[1:] if p]
            header_idx = i
            break
    if not quantiles:
        raise ValueError(f"Could not find quantile header row in: {tex_path}")

    data: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    i = (header_idx or 0) + 1

    while i < len(tab_lines):
        ln = tab_lines[i].strip()
        if _TABULAR_END_RE.search(ln):
            break
        if _is_rule_line(ln):
            i += 1
            continue

        # coef row: "Var & ... \\"
        parts = _split_ampersands(ln)
        if len(parts) < 2 or parts[0] == "":
            i += 1
            continue

        var = parts[0]
        coef_cells = parts[1:]
        if len(coef_cells) != len(quantiles):
            i += 1
            continue

        # Default: p-values missing
        p_map: Dict[str, Optional[str]] = {q: None for q in quantiles}

        # If next line looks like a paren row, capture it
        if i + 1 < len(tab_lines):
            nxt = tab_lines[i + 1].strip()
            if _looks_like_paren_row(nxt, n_q=len(quantiles)):
                p_parts = _split_ampersands(nxt)[1:]  # drop leading ""
                # strip parentheses
                p_cells = [c.strip().strip("()").strip() for c in p_parts]
                p_map = dict(zip(quantiles, p_cells))
                i += 1  # consume paren row

        data[var] = {
            "coef": dict(zip(quantiles, coef_cells)),
            "p": p_map,
        }
        i += 1

    return quantiles, data


# =========================
# Builder: stacked by horizon
# =========================

def build_stacked_horizon_quantreg_table(
    tex_files_by_horizon: Dict[int, Union[str, Path]],
    *,
    include_vars: Sequence[str],
    caption: str,
    label: str,
    out_path: Union[str, Path],
    var_labels: Optional[Dict[str, str]] = None,
    horizon_unit: str = "days",
    show_pvalues: bool = True,
    add_sym_def: bool = True,
    use_booktabs: bool = True,
    notes: Optional[str] = None,
) -> str:
    """
    Builds a table like your screenshot, but lets you:
    - choose which variables/controls appear (include_vars)
    - hide p-value rows (show_pvalues=False) to save space
    - keep stars in coefficient cells
    """
    var_labels = var_labels or {}

    # Parse each file
    parsed: Dict[int, Tuple[List[str], Dict[str, Dict[str, Dict[str, Optional[str]]]]]] = {}
    base_quantiles: Optional[List[str]] = None

    for H, fp in tex_files_by_horizon.items():
        q, d = parse_quantreg_tex(fp)
        parsed[H] = (q, d)
        if base_quantiles is None:
            base_quantiles = q
        elif q != base_quantiles:
            raise ValueError(f"Quantile columns differ across horizons. H={H} has {q}, expected {base_quantiles}.")

    assert base_quantiles is not None
    quantiles = base_quantiles
    horizons_sorted = sorted(parsed.keys())

    # normalized name lookup per horizon (robust matching)
    perH_lookup: Dict[int, Dict[str, str]] = {}
    for H in horizons_sorted:
        _, d = parsed[H]
        perH_lookup[H] = {_norm_key(k): k for k in d.keys()}

    include_norm = [_norm_key(v) for v in include_vars]
    labels_norm = {_norm_key(k): v for k, v in var_labels.items()}

    # out_path may be directory or file
    outp = Path(out_path)
    if outp.suffix.lower() != ".tex":
        outp.mkdir(parents=True, exist_ok=True)
        outp = outp / "stacked_horizon_quantreg.tex"
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Column spec: Horizon | Variable | quantiles
    colspec = "ll" + ("c" * len(quantiles))

    lines: List[str] = []
    lines.append(r"\begin{table}[!htbp]\centering")
    if add_sym_def:
        lines.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")

    if use_booktabs:
        lines.append(r"\toprule")
    else:
        lines.append(r"\hline")

    lines.append(" & ".join(["Horizon", "Variable"] + quantiles) + r" \\")
    lines.append(r"\midrule" if use_booktabs else r"\hline")

    for h_i, H in enumerate(horizons_sorted):
        q, d = parsed[H]
        lut = perH_lookup[H]

        first_row_for_h = True
        for vnorm in include_norm:
            if vnorm not in lut:
                avail = list(d.keys())
                raise ValueError(
                    f"Variable '{vnorm}' not found in horizon {H}. "
                    f"First few available: {avail[:12]}"
                )
            vname = lut[vnorm]
            vdisp = labels_norm.get(vnorm, vname)

            hcell = f"{H} {horizon_unit}" if first_row_for_h else ""
            first_row_for_h = False

            # coefficient row (stars preserved)
            coef_cells = [d[vname]["coef"][qq] for qq in q]
            lines.append(" & ".join([hcell, vdisp] + coef_cells) + r" \\")

            # optional p-value row (if present in source and requested)
            if show_pvalues:
                p_cells = []
                for qq in q:
                    pv = d[vname]["p"].get(qq, None)
                    p_cells.append(f"({pv})" if pv is not None else "")
                lines.append(" & ".join(["", ""] + p_cells) + r" \\")

        if h_i < len(horizons_sorted) - 1:
            lines.append(r"\midrule" if use_booktabs else r"\hline")

    lines.append(r"\bottomrule" if use_booktabs else r"\hline")
    lines.append(r"\end{tabular}")

    if notes:
        lines.append(r"\vspace{0.3em}")
        lines.append(r"\begin{minipage}{0.95\linewidth}")
        lines.append(r"\footnotesize")
        lines.append(r"\textit{Notes:} " + notes)
        lines.append(r"\end{minipage}")

    lines.append(r"\end{table}")

    outp.write_text("\n".join(lines), encoding="utf-8")
    return str(outp)


# tex_files = {
#     7:  r"C:\Users\beatt\Spyder directory\State Price Density\economics of bitcoin research\tables\asym_qr_mom_tex\btc_quant_qr_h7_C_kurt.tex",
#     14:  r"C:\Users\beatt\Spyder directory\State Price Density\economics of bitcoin research\tables\asym_qr_mom_tex\btc_quant_qr_h14_C_kurt.tex",
#     21:  r"C:\Users\beatt\Spyder directory\State Price Density\economics of bitcoin research\tables\asym_qr_mom_tex\btc_quant_qr_h21_C_kurt.tex",
#     28:  r"C:\Users\beatt\Spyder directory\State Price Density\economics of bitcoin research\tables\asym_qr_mom_tex\btc_quant_qr_h28_C_kurt.tex",
#     35:  r"C:\Users\beatt\Spyder directory\State Price Density\economics of bitcoin research\tables\asym_qr_mom_tex\btc_quant_qr_h35_C_kurt.tex",
#     60:  r"C:\Users\beatt\Spyder directory\State Price Density\economics of bitcoin research\tables\asym_qr_mom_tex\btc_quant_qr_h60_C_kurt.tex",
# }

# out = build_stacked_horizon_quantreg_table(
#     tex_files,
#     include_vars=[
#         r"$Ret^{+}$",
#         r"$Ret^{-}$",
#         r"$Ret^{+}_{t-1}$",
#         r"$Ret^{-}_{t-1}$",
#         r"$\Delta Var_{t-1}$",
#         r"$\Delta Skew_{t-1}$",
#         r"$\Delta Kurt_{t-1}$",

#     ],
#     caption="Quantile regression estimates (selected controls) across horizons.",
#     label="tab:selected_controls_B_skew",
#     out_path=r"C:\Users\beatt\Spyder directory\State Price Density\economics of bitcoin research\qr\btc_qr_kurt_all.tex",
#     show_pvalues=False,
#     notes=r"Stars denote significance: $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.",
# )
# print("Wrote:", out)
