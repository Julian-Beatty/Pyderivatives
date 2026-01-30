import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, Tuple

def _col_to_years(col: str) -> Optional[float]:
    """
    Convert standardized maturity labels like '1M','3M','1Y','10Y' -> years.
    Return None if not a maturity label.
    """
    if not isinstance(col, str):
        return None
    s = col.strip().upper()
    if s.endswith("M") and s[:-1].isdigit():
        return int(s[:-1]) / 12.0
    if s.endswith("Y") and s[:-1].isdigit():
        return float(int(s[:-1]))
    return None


def _surface_cols_to_days(surface: pd.DataFrame) -> Tuple[np.ndarray, Sequence[str]]:
    """
    For a fitted surface with columns like '1/365', '2/365', ... return day grid + col names.
    """
    cols = [c for c in surface.columns if c != "Date"]
    days = []
    for c in cols:
        if isinstance(c, str) and "/365" in c:
            d = int(c.split("/")[0])
            days.append(d)
        else:
            # ignore unknown columns
            pass
    # sort by day
    order = np.argsort(days)
    days_sorted = np.array([days[i] for i in order], dtype=int)
    cols_sorted = [cols[i] for i in order]
    return days_sorted, cols_sorted


def plot_yield_curve(
    yield_dict: Dict[str, pd.DataFrame],
    df_obs: pd.DataFrame,
    *,
    n_dates: int = 6,
    random_state: Optional[int] = 0,
    date_list: Optional[Sequence[pd.Timestamp]] = None,
    single_date: Optional[pd.Timestamp] = None,
    maturities_days_window: Optional[Tuple[int, int]] = None,
    y_in_percent: bool = True,
    title: str = "Observed vs Fitted Yield Curves",
    figsize: Tuple[int, int] = (12, 8),
    single_figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
):
    """
    Plot observed yield curve points vs fitted curves (Nelson–Siegel, Svensson, etc.).

    Modes
    -----
    • Panel mode (default): plots multiple dates in a grid
    • Single-date mode: set `single_date=...` to get one large plot
    """

    if "Date" not in df_obs.columns:
        raise ValueError("df_obs must contain a 'Date' column.")
    df_obs = df_obs.copy()
    df_obs["Date"] = pd.to_datetime(df_obs["Date"])

    # --------------------------------------------------
    # Determine observed maturity columns and day grid
    # --------------------------------------------------
    obs_cols = [c for c in df_obs.columns if c != "Date"]
    obs_years, obs_keep = [], []

    for c in obs_cols:
        t = _col_to_years(c)
        if t is not None:
            obs_keep.append(c)
            obs_years.append(t)

    if not obs_keep:
        raise ValueError("No observed maturity columns like '1M','1Y' found in df_obs.")

    obs_days = 365.0 * np.array(obs_years, float)

    order = np.argsort(obs_days)
    obs_days = obs_days[order]
    obs_keep = [obs_keep[i] for i in order]

    if maturities_days_window is not None:
        lo, hi = maturities_days_window
        m = (obs_days >= lo) & (obs_days <= hi)
        obs_days_plot = obs_days[m]
        obs_cols_plot = [c for c, ok in zip(obs_keep, m) if ok]
    else:
        obs_days_plot = obs_days
        obs_cols_plot = obs_keep

    if len(obs_cols_plot) == 0:
        raise ValueError("maturities_days_window filtered out all observed maturities.")

    # --------------------------------------------------
    # Determine common dates
    # --------------------------------------------------
    common_dates = set(df_obs["Date"].dropna().unique())
    for name, surf in yield_dict.items():
        if "Date" not in surf.columns:
            raise ValueError(f"Surface '{name}' must contain a 'Date' column.")
        surf = surf.copy()
        surf["Date"] = pd.to_datetime(surf["Date"])
        common_dates &= set(surf["Date"].dropna().unique())

    common_dates = sorted(common_dates)
    if len(common_dates) == 0:
        raise ValueError("No common dates between df_obs and fitted surfaces.")

    # --------------------------------------------------
    # Single-date mode
    # --------------------------------------------------
    if single_date is not None:
        date = pd.to_datetime(single_date)
        if date not in common_dates:
            raise ValueError("single_date not found in common dates.")

        fig, ax = plt.subplots(figsize=single_figsize, dpi=dpi)

        # observed
        row_obs = df_obs.loc[df_obs["Date"] == date]
        y_obs = row_obs.iloc[0][obs_cols_plot].to_numpy(float)
        ax.plot(obs_days_plot, y_obs, "o", label="Observed")

        # fitted curves
        for name, surf in yield_dict.items():
            surf = surf.copy()
            surf["Date"] = pd.to_datetime(surf["Date"])
            row_fit = surf.loc[surf["Date"] == date]
            if row_fit.empty:
                continue

            days_grid, cols_grid = _surface_cols_to_days(surf)

            if maturities_days_window is not None:
                lo, hi = maturities_days_window
                m = (days_grid >= lo) & (days_grid <= hi)
                x = days_grid[m]
                cols_use = [c for c, ok in zip(cols_grid, m) if ok]
            else:
                x = days_grid
                cols_use = cols_grid

            y_fit = row_fit.iloc[0][cols_use].to_numpy(float)
            ax.plot(x, y_fit, label=name)

        ax.set_title(str(date.date()))
        ax.set_xlabel("Maturity (days)")
        ax.set_ylabel("Yield (%)" if y_in_percent else "Yield")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.suptitle(title)
        fig.tight_layout()

        return fig

    # --------------------------------------------------
    # Panel mode (existing behavior)
    # --------------------------------------------------
    if date_list is not None:
        plot_dates = [pd.to_datetime(d) for d in date_list if pd.to_datetime(d) in common_dates]
        if len(plot_dates) == 0:
            raise ValueError("date_list contains no common dates.")
    else:
        rng = np.random.default_rng(random_state)
        n = min(n_dates, len(common_dates))
        plot_dates = rng.choice(common_dates, size=n, replace=False)
        plot_dates = sorted(plot_dates)

    n = len(plot_dates)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.array(axes).reshape(-1)

    model_grids = {}
    for name, surf in yield_dict.items():
        days_grid, cols_grid = _surface_cols_to_days(surf)
        model_grids[name] = (days_grid, cols_grid, surf.copy())

    for ax, date in zip(axes, plot_dates):
        row_obs = df_obs.loc[df_obs["Date"] == date]
        if row_obs.empty:
            ax.set_visible(False)
            continue

        y_obs = row_obs.iloc[0][obs_cols_plot].to_numpy(float)
        ax.plot(obs_days_plot, y_obs, "o", label="Observed")

        for name, (days_grid, cols_grid, surf) in model_grids.items():
            row_fit = surf.loc[surf["Date"] == date]
            if row_fit.empty:
                continue

            if maturities_days_window is not None:
                lo, hi = maturities_days_window
                m = (days_grid >= lo) & (days_grid <= hi)
                x = days_grid[m]
                cols_use = [c for c, ok in zip(cols_grid, m) if ok]
            else:
                x = days_grid
                cols_use = cols_grid

            y_fit = row_fit.iloc[0][cols_use].to_numpy(float)
            ax.plot(x, y_fit, label=name)

        ax.set_title(str(pd.to_datetime(date).date()))
        ax.set_xlabel("Maturity (days)")
        ax.set_ylabel("Yield (%)" if y_in_percent else "Yield")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(plot_dates):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig



import numpy as np
import pandas as pd

def plot_yield_surface(
    yield_dict,
    *,
    maturities_days_window=None,
    dates_window=None,
    date_step=1,
    max_dates=200,
    opacity=0.65,
    interactive=False,
    zlim=None,
    title="Yield Surface(s)",
    figsize=(10, 7),
    dpi=150,
):
    """
    Plot yield surfaces contained in yield_dict.

    - If interactive=False → static Matplotlib 3D plot
    - If interactive=True  → interactive Plotly 3D plot
      and ALL surfaces in yield_dict are overlaid.

    Plotly note:
      Surfaces default to a heatmap colorscale. We override that to make each surface a
      SOLID color so multiple overlaid surfaces remain distinguishable.

      Plotly does not reliably show legends for go.Surface traces.
      We add a proxy Scatter3d trace per model and use legendgroup so
      clicking the legend toggles BOTH the proxy and the surface.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(yield_dict, dict) or len(yield_dict) == 0:
        raise ValueError("yield_dict must be a non-empty dict of {name: DataFrame}.")

    # ------------------------
    # INTERACTIVE (Plotly)
    # ------------------------
    if interactive:
        import plotly.graph_objects as go
        from itertools import cycle

        # A simple qualitative palette (solid colors, good separation)
        # (No external deps; these are common Plotly-like hues.)
        palette = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # olive
            "#17becf",  # cyan
        ]
        color_cycle = cycle(palette)

        fig = go.Figure()
        any_added = False

        for model_name, surf in yield_dict.items():
            surf = surf.copy()
            if "Date" not in surf.columns:
                continue

            surf["Date"] = pd.to_datetime(surf["Date"], errors="coerce")
            surf = surf.dropna(subset=["Date"]).sort_values("Date")

            # date window
            if dates_window is not None:
                d0, d1 = pd.to_datetime(dates_window[0]), pd.to_datetime(dates_window[1])
                surf = surf[(surf["Date"] >= d0) & (surf["Date"] <= d1)]

            # subsample dates
            surf = surf.iloc[::max(1, date_step)]
            if max_dates is not None and len(surf) > max_dates:
                surf = surf.iloc[-max_dates:]

            if len(surf) == 0:
                continue

            # maturity grid: columns like '1/365'
            mat_cols = [c for c in surf.columns if c != "Date" and isinstance(c, str) and "/365" in c]
            if len(mat_cols) == 0:
                continue

            days = []
            cols = []
            for c in mat_cols:
                try:
                    d = int(c.split("/")[0])
                    days.append(d)
                    cols.append(c)
                except Exception:
                    pass

            if len(cols) == 0:
                continue

            days = np.array(days, dtype=int)
            order = np.argsort(days)
            days = days[order]
            cols = [cols[i] for i in order]

            if maturities_days_window is not None:
                lo, hi = maturities_days_window
                m = (days >= lo) & (days <= hi)
                days = days[m]
                cols = [c for c, ok in zip(cols, m) if ok]

            if len(cols) == 0 or len(days) == 0:
                continue

            Z = surf[cols].to_numpy(float)
            dates = surf["Date"].to_numpy()

            color = next(color_cycle)

            # ---- SOLID-COLOR surface (no heatmap) ----
            # Use a constant surfacecolor and a 2-point colorscale mapping both ends to same color.
            surfacecolor = np.zeros_like(Z)
            fig.add_trace(
                go.Surface(
                    x=days,
                    y=dates,
                    z=Z,
                    surfacecolor=surfacecolor,
                    colorscale=[[0.0, color], [1.0, color]],
                    cmin=0,
                    cmax=1,
                    opacity=opacity,
                    showscale=False,
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=False,  # surfaces don't legend reliably
                )
            )

            # Proxy trace to create a clickable legend item that toggles the group
            fig.add_trace(
                go.Scatter3d(
                    x=[int(days[0])],
                    y=[dates[0]],
                    z=[float(Z[0, 0])],
                    mode="markers",
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=True,
                    marker=dict(size=7, color=color),
                    hoverinfo="skip",
                    visible="legendonly",
                )
            )

            any_added = True

        if not any_added:
            raise ValueError("No surfaces could be plotted (check Date column and 'k/365' columns).")

        fig.update_layout(
            title=title,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="black",
                borderwidth=1,
                groupclick="toggleitem",
            ),
            scene=dict(
                xaxis_title="Maturity (days)",
                yaxis_title="Date",
                zaxis_title="Yield (%)",
                zaxis=dict(range=zlim) if zlim else {},
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        return fig

    # ------------------------
    # STATIC (Matplotlib)
    # ------------------------
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from itertools import cycle
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    legend_handles = []
    any_added = False

    # Stable public way to get the color cycle (no private attributes)
    color_cycle = cycle(mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"]))

    for model_name, surf in yield_dict.items():
        surf = surf.copy()
        if "Date" not in surf.columns:
            continue

        surf["Date"] = pd.to_datetime(surf["Date"], errors="coerce")
        surf = surf.dropna(subset=["Date"]).sort_values("Date")

        if dates_window is not None:
            d0, d1 = pd.to_datetime(dates_window[0]), pd.to_datetime(dates_window[1])
            surf = surf[(surf["Date"] >= d0) & (surf["Date"] <= d1)]

        surf = surf.iloc[::max(1, date_step)]
        if max_dates is not None and len(surf) > max_dates:
            surf = surf.iloc[-max_dates:]

        if len(surf) == 0:
            continue

        mat_cols = [c for c in surf.columns if c != "Date" and isinstance(c, str) and "/365" in c]
        if len(mat_cols) == 0:
            continue

        days = []
        cols = []
        for c in mat_cols:
            try:
                d = int(c.split("/")[0])
                days.append(d)
                cols.append(c)
            except Exception:
                pass

        if len(cols) == 0:
            continue

        days = np.array(days, dtype=int)
        order = np.argsort(days)
        days = days[order]
        cols = [cols[i] for i in order]

        if maturities_days_window is not None:
            lo, hi = maturities_days_window
            m = (days >= lo) & (days <= hi)
            days = days[m]
            cols = [c for c, ok in zip(cols, m) if ok]

        if len(cols) == 0 or len(days) == 0:
            continue

        Z = surf[cols].to_numpy(float)
        date_nums = mpl.dates.date2num(pd.to_datetime(surf["Date"]))

        X, Y = np.meshgrid(days.astype(float), date_nums.astype(float))

        color = next(color_cycle)
        ax.plot_surface(X, Y, Z, alpha=opacity, color=color)
        legend_handles.append(Patch(facecolor=color, edgecolor="k", label=model_name))

        any_added = True

    if not any_added:
        raise ValueError("No surfaces could be plotted (check Date column and 'k/365' columns).")

    ax.set_xlabel("Maturity (days)")
    ax.set_ylabel("Date")
    ax.set_zlabel("Yield (%)")
    ax.yaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))

    if zlim is not None:
        ax.set_zlim(*zlim)

    ax.set_title(title)
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)
    fig.tight_layout()

    return fig
