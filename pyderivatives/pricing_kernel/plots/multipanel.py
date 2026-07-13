from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np

from .utils import (
    _as_1d,
    _maybe_save,
    _standardize_x_axis,
    _get_plot_x_grid,
    _surface_key,
    _cdf_key,
    _get_cdf_surface,
    _pick_panel_indices,
    _quantiles_from_cdf,
)


def plot_pqk_multipanel(
    out_dict: Dict[str, dict],
    *,
    rnd_dict: Optional[dict] = None,
    title: Optional[str] = None,
    n_panels: Optional[int] = None,
    panel_shape: Tuple[int, int] = (2, 4),
    snap_percentiles_to_traded_strikes: bool = True,
    target_maturities: Optional[Tuple[float, ...]] = None,
    maturity_tol: Optional[float] = None,
    maturity_units: Literal["years", "days"] = "years",
    show_expiry_dates: bool = False,
    valuation_date: Optional[Union[str, "pd.Timestamp"]] = None,
    x_axis: str = "R",
    x_bounds: Optional[Tuple[float, float]] = None,
    truncate_kernel: bool = True,
    ptail_alpha: Tuple[float, float] = (0.10, 0.00),
    truncation_measure: str = "physical",
    kernel_yscale: Literal["linear", "log"] = "linear",
    kernel_log_eps: float = 1e-300,
    kernel_linestyle: str = "--",
    lw_density: float = 1.8,
    lw_kernel: float = 1.5,
    alpha_density: float = 0.95,
    alpha_kernel: float = 0.90,
    show_percentiles: bool = False,
    percentiles: Tuple[float, ...] = (0.05, 0.50, 0.95),
    percentile_measures: Tuple[str, ...] = ("rnd", "physical"),
    percentile_linestyle_rnd: str = "-",
    percentile_linestyle_physical: str = "--",
    percentile_alpha_rnd: float = 0.90,
    percentile_alpha_physical: float = 0.45,
    percentile_linewidth_rnd: float = 1.8,
    percentile_linewidth_physical: float = 1.5,
    snap_maturity_tol: float = 1.0 / 365.0,
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    legend_loc: str = "upper right",
    show: bool = True,
):
    """
    Plot RND, physical density, and pricing kernel across selected maturities.

    Parameters
    ----------
    out_dict:
        Dictionary of transformed outputs, usually:
            {"Model name": physical_result}

    rnd_dict:
        Optional original RND dictionary. Used only when snapping percentile
        markers to traded strikes.

    target_maturities:
        Optional target maturities. Interpreted as years by default, or days
        when maturity_units="days".

    x_axis:
        One of {"r", "R", "return", "K"}.
    """
    import matplotlib.pyplot as plt

    if show_expiry_dates:
        import pandas as pd

    if not isinstance(out_dict, dict) or len(out_dict) == 0:
        raise ValueError("out_dict must be a non-empty dictionary.")

    x_axis = _standardize_x_axis(x_axis)

    kernel_yscale = str(kernel_yscale).lower().strip()
    if kernel_yscale not in {"linear", "log"}:
        raise ValueError("kernel_yscale must be 'linear' or 'log'.")

    nrows, ncols = panel_shape
    max_panels = int(nrows) * int(ncols)

    first_label = next(iter(out_dict))
    first = out_dict[first_label]

    T_ref = _as_1d(first["T_grid"])
    X_ref, xlabel = _get_plot_x_grid(first, x_axis=x_axis)

    q_key, q_label, _ = _surface_key("rnd", x_axis)
    p_key, p_label, _ = _surface_key("physical", x_axis)
    M_key = "pricing_kernel_surface"

    cdf_q_key = _cdf_key("rnd", x_axis)
    cdf_p_key = _cdf_key("physical", x_axis)

    # ------------------------------------------------------------
    # Select maturity panels.
    # ------------------------------------------------------------
    if target_maturities is not None:
        targets = np.asarray(target_maturities, dtype=float).ravel()

        if maturity_units == "days":
            targets = targets / 365.0
            tol_eff = None if maturity_tol is None else float(maturity_tol) / 365.0
        elif maturity_units == "years":
            tol_eff = None if maturity_tol is None else float(maturity_tol)
        else:
            raise ValueError("maturity_units must be either 'years' or 'days'.")

        idxs = []

        for target in targets:
            if not np.isfinite(target):
                continue

            j = int(np.argmin(np.abs(T_ref - target)))
            dist = abs(float(T_ref[j]) - float(target))

            if tol_eff is not None and dist > tol_eff:
                continue

            if j not in idxs:
                idxs.append(j)

        if len(idxs) == 0:
            raise ValueError("No requested target_maturities were found within maturity_tol.")

        idxs = np.asarray(idxs, dtype=int)

        if idxs.size > max_panels:
            idxs = idxs[:max_panels]

    else:
        if n_panels is None:
            n_panels = max_panels

        n_panels = min(int(n_panels), max_panels, T_ref.size)
        idxs = _pick_panel_indices(T_ref, n_panels)

    # ------------------------------------------------------------
    # Select x-axis region.
    # ------------------------------------------------------------
    if x_bounds is not None:
        lo, hi = sorted(map(float, x_bounds))
        xmask_ref = np.isfinite(X_ref) & (X_ref >= lo) & (X_ref <= hi)
    else:
        xmask_ref = np.isfinite(X_ref)

    if not np.any(xmask_ref):
        raise ValueError("x_bounds produced an empty x-grid.")

    X_plot = X_ref[xmask_ref]

    # ------------------------------------------------------------
    # Kernel truncation settings.
    # ------------------------------------------------------------
    if truncate_kernel:
        aL, aR = map(float, ptail_alpha)

        if not (0.0 <= aL < 1.0 and 0.0 <= aR < 1.0 and aL + aR < 1.0):
            raise ValueError("ptail_alpha must satisfy 0 <= left,right < 1 and left+right < 1.")

        truncation_measure = str(truncation_measure).lower().strip()

        if truncation_measure in {"physical", "p"}:
            trunc_kind = "physical"
        elif truncation_measure in {"risk_neutral", "risk-neutral", "rnd", "q"}:
            trunc_kind = "rnd"
        else:
            raise ValueError("truncation_measure must be 'physical' or 'risk_neutral'.")
    else:
        aL = aR = None
        trunc_kind = None

    # ------------------------------------------------------------
    # Percentile marker settings.
    # ------------------------------------------------------------
    if show_percentiles:
        probs = np.asarray(percentiles, dtype=float)

        if np.any((probs <= 0) | (probs >= 1)):
            raise ValueError("percentiles must lie strictly between 0 and 1.")

        percentile_measures = tuple(str(x).lower().strip() for x in percentile_measures)
    else:
        probs = np.asarray([], dtype=float)
        percentile_measures = tuple()

    # ------------------------------------------------------------
    # Local helpers.
    # ------------------------------------------------------------
    def _cdf_prob_at_x(xgrid, cdf_vals, x):
        xgrid = np.asarray(xgrid, float)
        cdf_vals = np.asarray(cdf_vals, float)

        good = np.isfinite(xgrid) & np.isfinite(cdf_vals)

        if good.sum() < 2:
            return np.nan

        return float(
            np.interp(
                x,
                xgrid[good],
                cdf_vals[good],
                left=np.nan,
                right=np.nan,
            )
        )

    def _interp_row_to_ref(X, row):
        X = np.asarray(X, float).ravel()
        row = np.asarray(row, float).ravel()

        good = np.isfinite(X) & np.isfinite(row)

        if good.sum() < 2:
            return np.full_like(X_ref, np.nan, dtype=float)

        return np.interp(
            X_ref,
            X[good],
            row[good],
            left=np.nan,
            right=np.nan,
        )

    def _format_xmark(xmark):
        if x_axis == "K":
            return f"K={xmark:.2f}"
        if x_axis == "R":
            return f"R={xmark:.4f}"
        if x_axis == "r":
            return f"r={xmark:.4f}"
        if x_axis == "return":
            return f"R-1={xmark:.4f}"
        return f"x={xmark:.4f}"

    def _format_panel_title(T_years):
        days = float(T_years) * 365.0

        if days < 1:
            t_label = "Same day"
        elif abs(days - round(days)) < 0.05:
            d_int = int(round(days))
            t_label = f"T={d_int} day" if d_int == 1 else f"T={d_int} days"
        else:
            t_label = f"T={days:.1f} days"

        if show_expiry_dates:
            if valuation_date is None:
                raise ValueError("valuation_date must be supplied when show_expiry_dates=True.")

            expiry_date = pd.Timestamp(valuation_date) + pd.to_timedelta(days, unit="D")
            expiry_str = expiry_date.strftime("%Y-%m-%d")

            return f"{expiry_str} ({t_label})"

        return t_label

    def _get_single_rnd_day():
        if rnd_dict is None:
            return None

        if not isinstance(rnd_dict, dict) or len(rnd_dict) == 0:
            return None

        # Case 1: user passed one RND result directly.
        if "day" in rnd_dict:
            return rnd_dict["day"]

        # Case 2: user passed a one-date dictionary.
        if len(rnd_dict) == 1:
            date_key = next(iter(rnd_dict))
            entry = rnd_dict[date_key]

            if isinstance(entry, dict) and "day" in entry:
                return entry["day"]

            raise ValueError("rnd_dict entry must contain key 'day'.")

        # Case 3: user passed a many-date dictionary.
        if valuation_date is None:
            raise ValueError("rnd_dict has multiple date keys, so valuation_date must be supplied.")

        import pandas as pd

        val_date = pd.Timestamp(valuation_date).normalize()

        normalized_lookup = {}

        for k in rnd_dict.keys():
            try:
                k_date = pd.Timestamp(k).normalize()
            except Exception:
                continue

            normalized_lookup[k_date] = k

        if val_date not in normalized_lookup:
            raise ValueError(f"valuation_date {val_date.date()} was not found in rnd_dict.")

        date_key = normalized_lookup[val_date]
        entry = rnd_dict[date_key]

        if not isinstance(entry, dict) or "day" not in entry:
            raise ValueError("rnd_dict[valuation_date] must contain key 'day'.")

        return entry["day"]

    rnd_day = _get_single_rnd_day()

    def _snap_to_traded_x(xmark, T_target_ref):
        """
        Snap an x-axis percentile marker to the nearest traded strike.
        """
        if rnd_day is None or not np.isfinite(xmark):
            return xmark

        if isinstance(rnd_day, dict):
            K_obs = np.asarray(rnd_day["K_obs"], float).ravel()
            T_obs = np.asarray(rnd_day["T_obs"], float).ravel()
            S0_day = rnd_day.get("S0", None)
        else:
            K_obs = np.asarray(rnd_day.K_obs, float).ravel()
            T_obs = np.asarray(rnd_day.T_obs, float).ravel()
            S0_day = getattr(rnd_day, "S0", None)

        valid = (
            np.isfinite(K_obs)
            & np.isfinite(T_obs)
            & (K_obs > 0)
            & (T_obs >= 0)
        )

        K_obs = K_obs[valid]
        T_obs = T_obs[valid]

        if K_obs.size == 0:
            return xmark

        tmask = np.abs(T_obs - float(T_target_ref)) <= float(snap_maturity_tol)
        K_use = K_obs[tmask] if np.any(tmask) else K_obs

        # Convert the x-axis marker into a strike before snapping.
        S0 = first.get("S0", S0_day)

        if S0 is not None and np.isfinite(float(S0)):
            S0 = float(S0)

            if x_axis == "K":
                K_target = float(xmark)
            elif x_axis == "R":
                K_target = float(xmark) * S0
            elif x_axis == "r":
                K_target = float(np.exp(xmark) * S0)
            elif x_axis == "return":
                K_target = float((1.0 + xmark) * S0)
            else:
                K_target = float(xmark)
        else:
            K_target = float(xmark)

        K_snap = float(K_use[np.argmin(np.abs(K_use - K_target))])

        # Convert snapped strike back to the plotting axis.
        if S0 is None or not np.isfinite(float(S0)):
            return K_snap

        if x_axis == "K":
            return K_snap
        if x_axis == "R":
            return K_snap / S0
        if x_axis == "r":
            return float(np.log(K_snap / S0))
        if x_axis == "return":
            return K_snap / S0 - 1.0

        return K_snap

    # ------------------------------------------------------------
    # Create figure.
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.8 * ncols, 4.1 * nrows),
        sharex=True,
    )

    axes = np.asarray(axes).ravel()

    # ------------------------------------------------------------
    # Draw panels.
    # ------------------------------------------------------------
    for k, j_ref in enumerate(idxs):
        ax = axes[k]
        ax2 = ax.twinx()

        local_legend_seen = set()
        T_target_ref = float(T_ref[j_ref])

        for model_label, result in out_dict.items():
            T = _as_1d(result["T_grid"])
            X, _ = _get_plot_x_grid(result, x_axis=x_axis)

            q_surface = np.asarray(result[q_key], dtype=float)
            p_surface = np.asarray(result[p_key], dtype=float)
            M_surface = np.asarray(result[M_key], dtype=float)

            j = int(np.argmin(np.abs(T - T_target_ref)))

            same_x_grid = (
                X.shape == X_ref.shape
                and np.allclose(X, X_ref, equal_nan=True)
            )

            if same_x_grid:
                q_ref = q_surface[j, :]
                p_ref = p_surface[j, :]
                M_ref = M_surface[j, :]
            else:
                q_ref = _interp_row_to_ref(X, q_surface[j, :])
                p_ref = _interp_row_to_ref(X, p_surface[j, :])
                M_ref = _interp_row_to_ref(X, M_surface[j, :])

            q_plot = q_ref[xmask_ref]
            p_plot = p_ref[xmask_ref]
            M_plot = M_ref[xmask_ref]

            q_legend = f"{q_label} ({model_label})"
            p_legend = f"{p_label} ({model_label})"
            m_legend = f"$M$ ({model_label})"

            q_line, = ax.plot(
                X_plot,
                q_plot,
                linewidth=lw_density + 0.2,
                alpha=alpha_density,
                label=q_legend if q_legend not in local_legend_seen else "_nolegend_",
            )

            local_legend_seen.add(q_legend)
            color = q_line.get_color()

            ax.plot(
                X_plot,
                p_plot,
                linewidth=lw_density,
                alpha=0.72,
                color=color,
                linestyle="-",
                label=p_legend if p_legend not in local_legend_seen else "_nolegend_",
            )

            local_legend_seen.add(p_legend)

            # --------------------------------------------------------
            # Optional percentile markers.
            # --------------------------------------------------------
            if show_percentiles:
                if any(m in percentile_measures for m in {"rnd", "q", "risk_neutral"}):
                    if cdf_q_key in result:
                        cdf_q_surface = np.asarray(result[cdf_q_key], dtype=float)
                    else:
                        cdf_q_surface = _get_cdf_surface(result, kind="rnd", x_axis=x_axis)

                    cdf_q_ref = (
                        cdf_q_surface[j, :]
                        if same_x_grid
                        else _interp_row_to_ref(X, cdf_q_surface[j, :])
                    )

                    q_marks = _quantiles_from_cdf(X_ref, cdf_q_ref, probs)

                    for prob, xmark_raw in zip(probs, q_marks):
                        if not np.isfinite(xmark_raw):
                            continue

                        if snap_percentiles_to_traded_strikes:
                            xmark = _snap_to_traded_x(xmark_raw, T_target_ref)
                            label_prob = _cdf_prob_at_x(X_ref, cdf_q_ref, xmark)
                        else:
                            xmark = xmark_raw
                            label_prob = prob

                        legend_label = f"RND p{100 * label_prob:.1f}: {_format_xmark(xmark)}"

                        ax.axvline(
                            xmark,
                            color=color,
                            linestyle=percentile_linestyle_rnd,
                            alpha=percentile_alpha_rnd,
                            linewidth=percentile_linewidth_rnd,
                            label=legend_label,
                        )

                if any(m in percentile_measures for m in {"physical", "p"}):
                    if cdf_p_key in result:
                        cdf_p_surface = np.asarray(result[cdf_p_key], dtype=float)
                    else:
                        cdf_p_surface = _get_cdf_surface(result, kind="physical", x_axis=x_axis)

                    cdf_p_ref = (
                        cdf_p_surface[j, :]
                        if same_x_grid
                        else _interp_row_to_ref(X, cdf_p_surface[j, :])
                    )

                    p_marks = _quantiles_from_cdf(X_ref, cdf_p_ref, probs)

                    for prob, xmark_raw in zip(probs, p_marks):
                        if not np.isfinite(xmark_raw):
                            continue

                        if snap_percentiles_to_traded_strikes:
                            xmark = _snap_to_traded_x(xmark_raw, T_target_ref)
                            label_prob = _cdf_prob_at_x(X_ref, cdf_p_ref, xmark)
                        else:
                            xmark = xmark_raw
                            label_prob = prob

                        legend_label = f"Physical p{100 * label_prob:.1f}: {_format_xmark(xmark)}"

                        ax.axvline(
                            xmark,
                            color=color,
                            linestyle=percentile_linestyle_physical,
                            alpha=percentile_alpha_physical,
                            linewidth=percentile_linewidth_physical,
                            label=legend_label,
                        )

            # --------------------------------------------------------
            # Kernel line on secondary axis.
            # --------------------------------------------------------
            if truncate_kernel:
                if trunc_kind == "physical":
                    F_surface = (
                        np.asarray(result[cdf_p_key], dtype=float)
                        if cdf_p_key in result
                        else _get_cdf_surface(result, kind="physical", x_axis=x_axis)
                    )
                else:
                    F_surface = (
                        np.asarray(result[cdf_q_key], dtype=float)
                        if cdf_q_key in result
                        else _get_cdf_surface(result, kind="rnd", x_axis=x_axis)
                    )

                F_ref = (
                    F_surface[j, :]
                    if same_x_grid
                    else _interp_row_to_ref(X, F_surface[j, :])
                )

                F_plot = F_ref[xmask_ref]

                kmask = (
                    np.isfinite(M_plot)
                    & np.isfinite(F_plot)
                    & (F_plot >= float(aL))
                    & (F_plot <= 1.0 - float(aR))
                )
            else:
                kmask = np.isfinite(M_plot)

            Xk = X_plot[kmask]
            Mk = M_plot[kmask]

            if kernel_yscale == "log":
                good = np.isfinite(Mk) & (Mk > 0)
                Xk = Xk[good]
                Mk = np.maximum(Mk[good], kernel_log_eps)

            if Xk.size > 1:
                ax2.plot(
                    Xk,
                    Mk,
                    linestyle=kernel_linestyle,
                    linewidth=lw_kernel,
                    alpha=alpha_kernel,
                    color=color,
                    label=m_legend if m_legend not in local_legend_seen else "_nolegend_",
                )

                local_legend_seen.add(m_legend)

        ax.set_title(_format_panel_title(T_target_ref))
        ax.grid(True, alpha=0.25)

        if k % ncols == 0:
            ax.set_ylabel("Density")

        if k >= len(idxs) - ncols:
            ax.set_xlabel(xlabel)

        if k % ncols == ncols - 1:
            ax2.set_ylabel("Pricing kernel")

        ax2.set_yscale(kernel_yscale)

        if x_bounds is not None:
            ax.set_xlim(X_plot[0], X_plot[-1])

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()

        handles = h1 + h2
        labels = l1 + l2

        unique = {}

        for h, lab in zip(handles, labels):
            if lab != "_nolegend_" and lab not in unique:
                unique[lab] = h

        ax.legend(
            unique.values(),
            unique.keys(),
            loc=legend_loc,
            fontsize=8,
            frameon=True,
        )

    for k in range(len(idxs), axes.size):
        axes[k].axis("off")

    if title is None:
        title = "Risk-Neutral Density, Physical Density, and Pricing Kernel"

    fig.suptitle(title, y=0.995, fontsize=14)

    fig.subplots_adjust(
        top=0.90,
        wspace=0.28,
        hspace=0.34,
    )

    _maybe_save(fig, save, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def M_Q_K_multipanel_multi(*args, **kwargs):
    """
    Backward-compatible alias for the older multipanel plotting name.
    """
    return plot_pqk_multipanel(*args, **kwargs)