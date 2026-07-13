from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ============================================================
# Basic helpers
# ============================================================

def _as_1d(x) -> np.ndarray:
    """
    Convert input to a flattened float array.
    """
    return np.asarray(x, dtype=float).ravel()


def _maybe_save(fig, save=None, dpi: int = 200) -> None:
    """
    Save a matplotlib figure if a path is supplied.
    """
    if save is None:
        return

    save = Path(save)
    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=int(dpi), bbox_inches="tight")
    print(f"[saved] {save}")


def _get_meta_value(result: dict, *keys, default=None):
    """
    Look for metadata at the top level first, then inside result['meta'].
    """
    meta = result.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}

    for key in keys:
        if key in result and result[key] is not None:
            return result[key]

        if key in meta and meta[key] is not None:
            return meta[key]

    return default


def _title_suffix(result: dict) -> str:
    """
    Create a useful title suffix such as ticker/date.
    """
    ticker = _get_meta_value(result, "ticker", default="")

    date = _get_meta_value(
        result,
        "date",
        "valuation_date",
        "anchor_key_used",
        "anchor_date_used",
        default="",
    )

    pieces = [str(x) for x in (ticker, date) if x not in ("", None)]
    return " ".join(pieces)


# ============================================================
# Axis helpers
# ============================================================

def _standardize_x_axis(x_axis: str) -> str:
    """
    Canonicalize the x-axis choice.

    Returns one of:
        "r"      log return
        "R"      gross return
        "return" simple return R - 1
        "K"      strike / terminal price
    """
    key = str(x_axis).strip()

    aliases = {
        "r": "r",
        "log": "r",
        "lr": "r",
        "log_return": "r",
        "log-return": "r",

        "R": "R",
        "gross": "R",
        "gross_return": "R",

        "return": "return",
        "simple": "return",
        "simple_return": "return",

        "K": "K",
        "k": "K",
        "strike": "K",
        "terminal": "K",
        "terminal_price": "K",
        "ST": "K",
        "st": "K",
    }

    if key not in aliases:
        raise ValueError("x_axis must be one of {'r', 'R', 'return', 'K'}.")

    return aliases[key]


def _get_plot_x_grid(result: dict, x_axis: str = "R"):
    """
    Return the requested x-grid and its axis label.
    """
    x_axis = _standardize_x_axis(x_axis)

    if x_axis == "r":
        return _as_1d(result["grid_lr"]), r"Log return $r=\log(S_T/S_0)$"

    if x_axis == "R":
        return _as_1d(result["grid_r"]), r"Gross return $R=S_T/S_0$"

    if x_axis == "return":
        return _as_1d(result["grid_r"]) - 1.0, r"Simple return $R-1$"

    if x_axis == "K":
        return _as_1d(result["grid_k"]), r"Terminal price / strike $K=S_T$"

    raise RuntimeError("Unreachable x_axis.")


# ============================================================
# Surface and CDF key helpers
# ============================================================

def _surface_key(kind: str, x_axis: str) -> tuple[str, str, str]:
    """
    Map a surface type and x-axis to the correct result dictionary key.

    Returns
    -------
    surface_key:
        Dictionary key for the surface.

    zlabel:
        Axis label for the plotted surface.

    default_title:
        Default plot title.
    """
    kind = str(kind).lower().strip()
    x_axis = _standardize_x_axis(x_axis)

    suffix = {
        "r": "lr",
        "R": "r",
        "return": "r",
        "K": "k",
    }[x_axis]

    if kind in {"physical", "p"}:
        return (
            f"physical_{suffix}_surface",
            {
                "r": r"$f_P(r)$",
                "R": r"$p_P(R)$",
                "return": r"$p_P(R)$",
                "K": r"$p_P(K)$",
            }[x_axis],
            "Physical Density Surface",
        )

    if kind in {"rnd", "risk_neutral", "risk-neutral", "q"}:
        return (
            f"rnd_{suffix}_surface",
            {
                "r": r"$f_Q(r)$",
                "R": r"$p_Q(R)$",
                "return": r"$p_Q(R)$",
                "K": r"$p_Q(K)$",
            }[x_axis],
            "Risk-Neutral Density Surface",
        )

    if kind in {"pricing_kernel", "kernel", "m"}:
        return (
            "pricing_kernel_surface",
            r"$M$",
            "Pricing Kernel Surface",
        )

    if kind in {"rra", "relative_risk_aversion", "risk_aversion"}:
        return (
            "relative_risk_aversion_surface",
            "Relative risk aversion",
            "Relative Risk Aversion Surface",
        )

    raise ValueError("kind must be one of {'physical', 'rnd', 'pricing_kernel', 'rra'}.")


def _cdf_key(kind: str, x_axis: str) -> str:
    """
    Return the CDF key corresponding to a density surface.
    """
    kind = str(kind).lower().strip()
    x_axis = _standardize_x_axis(x_axis)

    suffix = {
        "r": "lr",
        "R": "r",
        "return": "r",
        "K": "k",
    }[x_axis]

    if kind in {"physical", "p"}:
        return f"physical_cdf_{suffix}_surface"

    if kind in {"rnd", "risk_neutral", "risk-neutral", "q"}:
        return f"cdf_{suffix}_surface"

    raise ValueError("CDF is only available for physical or risk-neutral density.")


def _cdf_from_density(x, f, eps: float = 1e-14) -> np.ndarray:
    """
    Compute a normalized CDF from a density using trapezoidal integration.

    This local copy keeps plotting utilities self-contained.
    """
    x = _as_1d(x)
    f = _as_1d(f)

    out = np.full_like(x, np.nan, dtype=float)

    good = np.isfinite(x) & np.isfinite(f) & (f >= 0)
    if good.sum() < 3:
        return out

    xx = x[good]
    ff = f[good]

    order = np.argsort(xx)
    xx = xx[order]
    ff = ff[order]

    inc = 0.5 * (ff[1:] + ff[:-1]) * np.diff(xx)

    cdf = np.empty_like(xx)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(inc)

    total = cdf[-1]
    if not np.isfinite(total) or total <= eps:
        return out

    cdf = cdf / total
    out[good] = np.interp(x[good], xx, cdf, left=np.nan, right=np.nan)

    return out


def _get_cdf_surface(result: dict, *, kind: str, x_axis: str) -> np.ndarray:
    """
    Fetch a stored CDF surface, or compute it row-by-row from the density surface.
    """
    cdf_key = _cdf_key(kind, x_axis)

    if cdf_key in result:
        return np.asarray(result[cdf_key], dtype=float)

    density_key, _, _ = _surface_key(kind, x_axis)

    if density_key not in result:
        raise KeyError(f"Need result['{cdf_key}'] or result['{density_key}'].")

    x_grid, _ = _get_plot_x_grid(result, x_axis=x_axis)
    density = np.asarray(result[density_key], dtype=float)

    return np.vstack([
        _cdf_from_density(x_grid, density[j, :])
        for j in range(density.shape[0])
    ])


# ============================================================
# Surface slicing / panel helpers
# ============================================================

def _slice_surface_bounds(
    T_grid,
    x_grid,
    Z,
    *,
    T_bounds=None,
    x_bounds=None,
):
    """
    Slice a surface by maturity and x-axis bounds.

    Assumes:
        Z.shape == (len(T_grid), len(x_grid))
    """
    T_grid = _as_1d(T_grid)
    x_grid = _as_1d(x_grid)
    Z = np.asarray(Z, dtype=float)

    if Z.shape != (T_grid.size, x_grid.size):
        raise ValueError(
            f"Surface shape mismatch. Expected {(T_grid.size, x_grid.size)}, got {Z.shape}."
        )

    tmask = np.isfinite(T_grid)
    xmask = np.isfinite(x_grid)

    if T_bounds is not None:
        lo, hi = sorted(map(float, T_bounds))
        tmask &= (T_grid >= lo) & (T_grid <= hi)

    if x_bounds is not None:
        lo, hi = sorted(map(float, x_bounds))
        xmask &= (x_grid >= lo) & (x_grid <= hi)

    if not np.any(tmask):
        raise ValueError("T_bounds produced an empty maturity grid.")

    if not np.any(xmask):
        raise ValueError("x_bounds produced an empty x-grid.")

    return T_grid[tmask], x_grid[xmask], Z[np.ix_(tmask, xmask)], tmask, xmask


def _pick_panel_indices(T_grid, n_panels: int) -> np.ndarray:
    """
    Pick approximately evenly spaced maturity indices.
    """
    T_grid = _as_1d(T_grid)

    if T_grid.size == 0:
        raise ValueError("Cannot select panels from an empty T_grid.")

    n_panels = int(min(max(1, n_panels), T_grid.size))

    return np.unique(
        np.linspace(0, T_grid.size - 1, n_panels)
        .round()
        .astype(int)
    )


def _quantiles_from_cdf(X, F, probs):
    """
    Convert a CDF row into x-axis quantile locations.
    """
    X = np.asarray(X, float)
    F = np.asarray(F, float)
    probs = np.asarray(probs, float)

    good = np.isfinite(X) & np.isfinite(F)

    if good.sum() < 3:
        return np.full(probs.shape, np.nan)

    Xg = X[good]
    Fg = F[good]

    order = np.argsort(Xg)
    Xg = Xg[order]
    Fg = Fg[order]

    # Enforce monotonicity before inverting the CDF.
    Fg = np.maximum.accumulate(Fg)

    unique_F, unique_idx = np.unique(Fg, return_index=True)
    unique_X = Xg[unique_idx]

    return np.interp(probs, unique_F, unique_X, left=np.nan, right=np.nan)