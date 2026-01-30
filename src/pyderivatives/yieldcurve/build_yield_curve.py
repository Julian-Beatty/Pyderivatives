import re
from pathlib import Path
from typing import List, Optional, Dict, Union
import pandas as pd


# =========================
# Helpers
# =========================
def _parse_maturity_to_years(col: str) -> Optional[float]:
    if not isinstance(col, str):
        return None
    s = col.strip().upper()
    if s == "DATE":
        return None
    if s.endswith("M") and s[:-1].isdigit():
        return int(s[:-1]) / 12.0
    if s.endswith("Y") and s[:-1].isdigit():
        return float(int(s[:-1]))
    return None




# =========================
# Maturity parsing helpers
# =========================
def _standardize_maturity(col: str) -> Optional[str]:
    if col is None:
        return None

    s = str(col).strip()
    if s.lower() in {"date", "time", "timestamp"}:
        return None

    s = re.sub(r"[,_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # months
    m = re.match(r"^(\d+)\s*(mo|mos|month|months|m)$", s, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))}M"

    # years
    m = re.match(r"^(\d+)\s*(yr|yrs|year|years|y)$", s, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))}Y"

    return None


def _maturity_sort_key(m: str) -> float:
    n = int(re.findall(r"\d+", m)[0])
    return n if m.endswith("M") else 12 * n


def _detect_date_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).lower() == "date":
            return c
    return df.columns[0]


# =========================
# Read ONE file (path-aware)
# =========================
def read_yield_file(
    file: Union[str, Path]
) -> pd.DataFrame:
    """
    Read a single yield CSV.
    - `file` can be a filename or a full/relative path.
    """
    path = Path(file).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    date_col = _detect_date_column(df)

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col], errors="coerce")

    maturity_map: Dict[str, str] = {}
    for c in df.columns:
        if c == date_col:
            continue
        std = _standardize_maturity(c)
        if std is not None:
            maturity_map[c] = std

    if not maturity_map:
        raise ValueError(f"No maturity columns detected in {path.name}")

    for raw, std in maturity_map.items():
        out[std] = pd.to_numeric(df[raw], errors="coerce")

    out = out.dropna(subset=["Date"])

    # If duplicate dates exist, keep last non-null
    out = (
        out.sort_values("Date")
        .groupby("Date", as_index=False)
        .last()
    )

    return out


# =========================
# Read MANY files & merge
# =========================
def build_yield_dataframe(
    files: List[Union[str, Path]],
    how: str = "outer"
) -> pd.DataFrame:
    """
    Read multiple yield CSVs (filenames or paths) and merge on Date. Input paths of treasury yields from
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives (pre 2022)
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2025 (post 2022)
    
    eg. files=[daily-treasury-rates (1),daily-treasury-rates (2),daily-treasury-rates (3),par-yield-curve-rates-1990-2022.csv] 
    then build_yield_dataframe(files)
    """
    merged = None

    for f in files:
        df = read_yield_file(f)

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="Date", how=how, suffixes=("", "_new"))

            # resolve overlapping maturity columns
            for c in list(merged.columns):
                if c.endswith("_new"):
                    base = c[:-4]
                    merged[base] = merged[base].combine_first(merged[c])
                    merged.drop(columns=c, inplace=True)

    maturity_cols = [c for c in merged.columns if c != "Date"]
    maturity_cols = sorted(maturity_cols, key=_maturity_sort_key)

    merged = merged[["Date"] + maturity_cols]
    merged = merged.sort_values("Date").reset_index(drop=True)

    return merged

###Demo
# files = [
#     "daily-treasury-rates (1).csv",
#     "daily-treasury-rates (2).csv",
#     "daily-treasury-rates (3).csv",
#     "par-yield-curve-rates-1990-2022.csv"
# ]

# df = build_yield_dataframe(files)


