from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ----------------------------
# Regex patterns
# ----------------------------

_PANEL_RE = re.compile(
    r"Panel\s+[A-Z]\s*:\s*Quantile.*?\\tau\s*=?\s*([0-9]*\.?[0-9]+)",
    flags=re.IGNORECASE | re.DOTALL,
)

# Header row like: "& BTC & GLD & ... & From others \\"
_HDR_RE = re.compile(
    r"^\s*&\s*BTC\s*&.*?From\s+others\s*\\\\\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

_TO_ROW_RE = re.compile(r"^\s*TO\s*&\s*(.*?)\s*\\\\\s*$", flags=re.IGNORECASE | re.MULTILINE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Generic "ASSET & ... \\"
def _make_asset_row_re(asset: str) -> re.Pattern:
    return re.compile(
        rf"^\s*{re.escape(asset)}\s*&\s*(.*?)\s*\\\\\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )


# ----------------------------
# Helpers
# ----------------------------

def _split_ampersand_row(line: str) -> List[str]:
    s = line.strip()
    s = re.sub(r"(\\hline|\\toprule|\\midrule|\\bottomrule)", "", s, flags=re.I).strip()
    s = re.sub(r"\\\\\s*$", "", s).strip()
    cells = [c.strip() for c in s.split("&")]
    if cells and cells[0] == "":
        cells = cells[1:]
    return cells


def _find_panels(text: str) -> List[Tuple[float, int, int]]:
    hits = [(float(m.group(1)), m.start()) for m in _PANEL_RE.finditer(text)]
    hits.sort(key=lambda x: x[1])
    if not hits:
        raise ValueError("No quantile panels found. Expected: Panel A: Quantile ($\\tau=0.10$).")

    out: List[Tuple[float, int, int]] = []
    for i, (tau, start) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        out.append((tau, start, end))
    return out


def _extract_numbers(rhs: str) -> List[float]:
    return [float(x) for x in _NUM_RE.findall(rhs)]


def _nearest_key(d: Dict[float, List[float]], target: float) -> float:
    if target in d:
        return target
    return min(d.keys(), key=lambda k: abs(k - target))


# ----------------------------
# Generic parser
# ----------------------------

def parse_focus_and_to_by_quantile(
    tex_path: str,
    *,
    focus_asset: str,
) -> Tuple[List[str], Dict[float, List[float]], Dict[float, List[float]]]:
    """
    Returns:
      columns: ["BTC","GLD","SLV","SPY","XLE","From others"]  (from header if present)
      focus_by_tau[tau] = [BTC,GLD,SLV,SPY,XLE,From others]  (row = focus_asset)
      to_by_tau[tau]    = [BTC,GLD,SLV,SPY,XLE]              (TO row across asset columns)
    """
    text = Path(tex_path).read_text(encoding="utf-8", errors="ignore")

    # Columns
    columns: Optional[List[str]] = None
    hm = _HDR_RE.search(text)
    if hm:
        columns = _split_ampersand_row(hm.group(0))
    if not columns:
        columns = ["BTC", "GLD", "SLV", "SPY", "XLE", "From others"]

    asset_cols = columns[:-1]
    expected_focus_len = len(columns)
    expected_to_len = len(asset_cols)

    focus_re = _make_asset_row_re(focus_asset)

    focus_by_tau: Dict[float, List[float]] = {}
    to_by_tau: Dict[float, List[float]] = {}

    for tau, start, end in _find_panels(text):
        panel_text = text[start:end]

        fm = focus_re.search(panel_text)
        if not fm:
            raise ValueError(f"{focus_asset} row not found for tau={tau} in: {tex_path}")
        focus_nums = _extract_numbers(fm.group(1))
        if len(focus_nums) != expected_focus_len:
            raise ValueError(
                f"{focus_asset} numeric count mismatch in {tex_path} tau={tau}: "
                f"got {len(focus_nums)}, expected {expected_focus_len}."
            )
        focus_by_tau[tau] = focus_nums

        tm = _TO_ROW_RE.search(panel_text)
        if not tm:
            raise ValueError(f"TO row not found for tau={tau} in: {tex_path}")
        to_nums = _extract_numbers(tm.group(1))
        if len(to_nums) < expected_to_len:
            raise ValueError(
                f"TO numeric count mismatch in {tex_path} tau={tau}: "
                f"got {len(to_nums)}, expected >= {expected_to_len}."
            )
        to_by_tau[tau] = to_nums[:expected_to_len]

    return columns, focus_by_tau, to_by_tau


# ----------------------------
# Table builder (generalized)
# ----------------------------

def build_focus_only_table_with_to_net(
    tex_files_by_horizon: Dict[int, str],
    *,
    focus_asset: str,                 # e.g. "GLD"
    caption: str,
    label: str,
    out_path: str,
    moment_name: str = "volatility",
    taus: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
    float_fmt: str = "{:.2f}",
) -> str:
    r"""
    Creates a focus-asset table with columns:

      Horizon | (asset columns...) | From others | TO | Net

    Where:
      From others = last entry in focus row
      TO          = focus asset's entry in the TO row (based on column position)
      Net         = TO - From others
    """
    parsed: Dict[int, Tuple[List[str], Dict[float, List[float]], Dict[float, List[float]]]] = {}
    base_cols: Optional[List[str]] = None

    for H, fp in tex_files_by_horizon.items():
        cols, focus_by_tau, to_by_tau = parse_focus_and_to_by_quantile(fp, focus_asset=focus_asset)
        parsed[H] = (cols, focus_by_tau, to_by_tau)

        if base_cols is None:
            base_cols = cols
        elif cols != base_cols:
            raise ValueError(f"Column headers differ across horizons. H={H} has {cols}, expected {base_cols}.")

    assert base_cols is not None
    cols = base_cols[:]
    asset_cols = cols[:-1]  # e.g. BTC..XLE
    if focus_asset.upper() not in [c.upper() for c in asset_cols]:
        raise ValueError(
            f"focus_asset='{focus_asset}' not found in asset columns {asset_cols}. "
            "Make sure it matches the table header (case-insensitive)."
        )

    # Find focus asset column index in TO row
    focus_idx = [c.upper() for c in asset_cols].index(focus_asset.upper())

    final_cols = asset_cols + ["From others", "TO", "Net"]
    horizons_sorted = sorted(tex_files_by_horizon.keys())

    def fmt(x: float) -> str:
        return float_fmt.format(x)

    lines: List[str] = []
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(final_cols) + r"}",
        r"\hline",
        "Horizon & " + " & ".join(final_cols) + r" \\",
        r"\hline",
    ]

    for tau in taus:
        lines.append(
            rf"\multicolumn{{{1+len(final_cols)}}}{{l}}{{\textbf{{Panel: Quantile $\tau = {tau:.2f}$}}}} \\"
        )

        for H in horizons_sorted:
            _, focus_by_tau, to_by_tau = parsed[H]

            tau_f = _nearest_key(focus_by_tau, tau)
            tau_t = _nearest_key(to_by_tau, tau)

            focus_vals = focus_by_tau[tau_f]  # [BTC,GLD,SLV,SPY,XLE,From others]
            to_vals = to_by_tau[tau_t]        # [BTC,GLD,SLV,SPY,XLE]

            from_others = focus_vals[-1]
            to_focus = to_vals[focus_idx]
            net = to_focus - from_others

            row = [f"{focus_asset.upper()} ({H}d)"]
            row += [fmt(x) for x in focus_vals[:-1]]  # across asset columns
            row += [fmt(from_others), fmt(to_focus), fmt(net)]
            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\hline")

    lines += [
        r"\end{tabular}",
        r"\vspace{0.3em}",
        r"\begin{minipage}{0.95\linewidth}",
        r"\footnotesize",
        rf"\textit{{Notes:}} Entries report generalized forecast error variance decompositions for {focus_asset.upper()} "
        rf"{moment_name} from quantile VAR models. Columns denote the source of shocks. "
        r"\emph{From others} is the total share of the focus asset's forecast error variance explained by the other assets. "
        r"\emph{TO} denotes directional spillovers transmitted by the focus asset to the rest of the system "
        r"(the focus asset’s entry in the \texttt{TO} row, matched by column). "
        r"\emph{Net} equals \emph{TO} minus \emph{From others}.",
        r"\end{minipage}",
        r"\end{table}",
    ]

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines), encoding="utf-8")
    return str(outp)


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    tex_files = {
            7:  r"C:/Users/beatt/Spyder directory/State Price Density/economics of bitcoin research/figures/tables/ConnectednessTable_phys_kurt_H7.tex",
            14: r"C:/Users/beatt/Spyder directory/State Price Density/economics of bitcoin research/figures/tables/ConnectednessTable_phys_kurt_H14.tex",
            21: r"C:/Users/beatt/Spyder directory/State Price Density/economics of bitcoin research/figures/tables/ConnectednessTable_phys_kurt_H21.tex",
            28: r"C:/Users/beatt/Spyder directory/State Price Density/economics of bitcoin research/figures/tables/ConnectednessTable_phys_kurt_H28.tex",
            35: r"C:/Users/beatt/Spyder directory/State Price Density/economics of bitcoin research/figures/tables/ConnectednessTable_phys_kurt_H35.tex",
            60: r"C:/Users/beatt/Spyder directory/State Price Density/economics of bitcoin research/figures/tables/ConnectednessTable_phys_kurt_H60.tex",
        }


    out = build_focus_only_table_with_to_net(
        tex_files,
        focus_asset="GLD",
        caption="Gold Kurtosis Spillovers Across Investment Horizons",
        label="tab:gld_kurt_spillovers_allH",
        out_path=r"C:\Users\beatt\Spyder directory\State Price Density\economics of bitcoin research\data\GLD_only_phys_kurt_allH.tex",
        moment_name="kurtosis",
    )
    print("Wrote:", out)

    
