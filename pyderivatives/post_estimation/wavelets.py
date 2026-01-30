#########
# ============================================================
# Wavelet Coherence + Monte Carlo (Phase-Randomized) Significance
# Works with pycwt 0.4.0b1.dev10+g3343016af.d20251028
# ============================================================
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


def _phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Phase-randomized Fourier surrogate.
    Preserves power spectrum (and thus autocorrelation), destroys phase relations.
    """
    x = np.asarray(x, float)
    n = x.size
    mu = x.mean()
    z = x - mu

    Zf = np.fft.rfft(z)
    amp = np.abs(Zf)
    ph = np.angle(Zf)

    # random phases in [0, 2pi)
    rand_ph = rng.uniform(0.0, 2.0 * np.pi, size=ph.shape)

    # keep DC exactly; keep Nyquist if even length
    rand_ph[0] = ph[0]
    if n % 2 == 0:
        rand_ph[-1] = ph[-1]

    Zf_surr = amp * np.exp(1j * rand_ph)
    zs = np.fft.irfft(Zf_surr, n=n)

    return zs + mu


def _mc_phase_significance_threshold(
    xret: np.ndarray,
    yret: np.ndarray,
    *,
    dt: float,
    dj: float,
    s0: float,
    J: int,
    mother,
    period_min: float,
    period_max: float,
    B: int,
    alpha: float,
    seed: int,
):
    """
    Monte Carlo phase-randomized significance thresholds for WTC.
    Returns:
      period_k : (n_keep,) periods used
      thr_k    : (n_keep,) threshold coherence by period (global-by-period)
    Notes:
      - We compute the mean coherence over time for each period in each surrogate,
        then take the alpha-quantile across surrogates. This yields stable contours.
    """
    import pycwt as wavelet

    rng = np.random.default_rng(seed)

    # First run (no sig) to get the exact period grid that pycwt will use
    W0, _, _, freq0, _ = wavelet.wct(
        xret, yret, dt,
        dj=dj, s0=s0, J=J,
        sig=False,
        wavelet=mother
    )
    period0 = 1.0 / freq0
    keep0 = (period0 >= period_min) & (period0 <= period_max)
    period_k = period0[keep0]
    n_keep = period_k.size

    # Monte Carlo draws: store time-averaged coherence by period
    mc_vals = np.empty((B, n_keep), dtype=float)

    for b in range(B):
        y_surr = _phase_randomize(yret, rng)

        Wb, _, _, freqb, _ = wavelet.wct(
            xret, y_surr, dt,
            dj=dj, s0=s0, J=J,
            sig=False,
            wavelet=mother
        )
        periodb = 1.0 / freqb
        keepb = (periodb >= period_min) & (periodb <= period_max)

        # robust assumption: same keep mask length (should match for fixed params)
        Wb_k = Wb[keepb, :]
        mc_vals[b, :] = Wb_k.mean(axis=1)

    thr_k = np.quantile(mc_vals, alpha, axis=0)  # (n_keep,)
    return period_k, thr_k


def plot_wtc(
    x,
    y,
    *,
    x_name: str = "X",
    y_name: str = "Y",
    value_col_x: str | None = None,
    value_col_y: str | None = None,
    date_col: str = "date",
    data_col: str = "value",

    # window
    min_date: str | pd.Timestamp | None = None,
    max_date: str | pd.Timestamp | None = None,

    # preprocessing
    transform: str = "none",     # {"none","diff","logdiff","pct"}
    detrend: bool = False,
    standardize: bool = True,
    min_n: int = 120,

    # wavelet params
    w0: float = 6.0,
    dj: float = 1/12,
    s0: float | None = None,
    J: int | None = None,

    period_min: float = 2.0,
    period_max: float = 128.0,
    period_ticks: tuple = (4, 8, 16, 32, 64, 128),

    # arrows
    arrows_step_t: int = 22,
    arrows_step_s: int = 10,
    arrow_min_coh: float = 0.5,
    arrow_only_signif: bool = True,
    arrow_len_axes: float = 0.022,
    arrow_lw: float = 0.8,
    arrow_mutation_scale: float = 10,

    # significance
    sig: bool = True,
    sig_method: str = "mc-phase",     # {"ar1","mc-phase"}
    significance_level: float = 0.95, # AR(1) only
    mc_B: int = 300,                  # MC only
    mc_alpha: float = 0.95,           # MC only
    mc_seed: int = 123,               # MC only

    # plot
    figsize=(10.5, 4.8),
    cmap: str = "jet",
    title_fontsize: int = 11,
    label_fontsize: int = 9,
    tick_fontsize: int = 8,
    show: bool = True,
    debug: bool = False,
):
    """
    Wavelet coherence for two generic time series signals, with significance contour.

    Inputs x,y:
      - pd.Series with DateTimeIndex
      - pd.DataFrame with DateTimeIndex (choose column via value_col_x/value_col_y, or single-col)
      - pd.DataFrame with columns [date_col, data_col] (old mode)

    transform:
      - "none"   : use levels as given (good for skew, variance, returns)
      - "diff"   : first difference
      - "pct"    : percent change
      - "logdiff": log-difference (ONLY for strictly positive price levels)

    sig_method:
      - "mc-phase": Monte Carlo phase-randomized surrogates (recommended for finance)
      - "ar1"     : pycwt AR(1) red-noise (may fail on short/trended series)
    """
    try:
        import pycwt as wavelet
    except ImportError as e:
        raise ImportError("Install pycwt: pip install pycwt") from e

    def _to_series(obj, *, value_col: str | None, fallback: str) -> pd.Series:
        if isinstance(obj, pd.Series):
            s = obj.copy()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s.sort_index()
            s.name = s.name or fallback
            return s.astype(float)

        if isinstance(obj, pd.DataFrame):
            df = obj.copy()

            if isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df = df.sort_index()
                if value_col is None:
                    if df.shape[1] == 1:
                        value_col = df.columns[0]
                    else:
                        raise ValueError("DataFrame has multiple columns; pass value_col_x/value_col_y.")
                s = df[value_col].astype(float).copy()
                s.name = str(value_col)
                return s

            if date_col in df.columns:
                tmp = df.copy()
                tmp[date_col] = pd.to_datetime(tmp[date_col]).dt.tz_localize(None).dt.normalize()
                tmp = tmp.sort_values(date_col)
                if value_col is None:
                    value_col = data_col
                if value_col not in tmp.columns:
                    raise ValueError(f"Could not find value column {value_col!r} in DataFrame.")
                s = tmp.set_index(date_col)[value_col].astype(float).copy()
                s.name = str(value_col)
                return s

            raise ValueError("DataFrame must have a DateTimeIndex or a date column.")
        raise TypeError("x/y must be a pandas Series or DataFrame.")

    def _apply_transform(s: pd.Series, how: str) -> pd.Series:
        if how == "none":
            return s
        if how == "diff":
            return s.diff()
        if how == "pct":
            return s.pct_change()
        if how == "logdiff":
            if (s <= 0).any():
                bad = int((s <= 0).sum())
                raise ValueError(f"logdiff requires strictly positive series; found {bad} non-positive values.")
            return np.log(s).diff()
        raise ValueError("transform must be one of {'none','diff','pct','logdiff'}.")

    def _detrend_linear(arr: np.ndarray) -> np.ndarray:
        t = np.arange(arr.size, dtype=float)
        b, a = np.polyfit(t, arr, 1)
        return arr - (b * t + a)

    def _paper_axis_style(ax):
        ax.set_yscale("log")
        ax.set_ylim(period_min, period_max)
        ax.invert_yaxis()
        ax.set_yticks(np.array(period_ticks, dtype=float))
        ax.set_yticklabels([str(int(t)) for t in period_ticks], fontsize=tick_fontsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_ylabel("Period", fontsize=label_fontsize)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _draw_fixed_phase_arrows(ax, dates, period, phase, mask):
        x0, x1 = dates[0], dates[-1]
        span = (x1 - x0) / np.timedelta64(1, "ns")

        def xfrac(xdt):
            return float(((xdt - x0) / np.timedelta64(1, "ns")) / span)

        lp0, lp1 = np.log(period_min), np.log(period_max)

        def yfrac(p):
            frac = (np.log(p) - lp0) / (lp1 - lp0)
            return float(1.0 - frac)

        for j in range(0, phase.shape[0], arrows_step_s):
            p = float(period[j])
            if (p < period_min) or (p > period_max):
                continue
            y = yfrac(p)

            for i in range(0, phase.shape[1], arrows_step_t):
                if not mask[j, i]:
                    continue
                x = xfrac(dates[i])
                ang = float(phase[j, i])
                dx = arrow_len_axes * np.cos(ang)
                dy = arrow_len_axes * np.sin(ang) * 0.55

                ax.annotate(
                    "",
                    xy=(x + dx, y + dy),
                    xytext=(x, y),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=arrow_lw,
                        color="black",
                        mutation_scale=arrow_mutation_scale,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=20,
                )

    # --- coerce inputs to Series ---
    sx = _to_series(x, value_col=value_col_x, fallback=x_name).rename("x")
    sy = _to_series(y, value_col=value_col_y, fallback=y_name).rename("y")

    # --- align on overlap + window ---
    xy = pd.concat([sx, sy], axis=1, join="inner").dropna()
    if xy.empty:
        raise ValueError("No overlapping timestamps between x and y after alignment.")

    if min_date is not None:
        xy = xy.loc[xy.index >= pd.Timestamp(min_date)]
    if max_date is not None:
        xy = xy.loc[xy.index <= pd.Timestamp(max_date)]
    if xy.empty:
        raise ValueError("Empty after applying min_date/max_date window.")

    # --- transform ---
    tx = _apply_transform(xy["x"], transform)
    ty = _apply_transform(xy["y"], transform)
    rxy = pd.concat([tx.rename("x"), ty.rename("y")], axis=1).dropna()

    if len(rxy) < min_n:
        raise ValueError(f"Not enough overlapping observations after transform. n={len(rxy)} (min_n={min_n})")

    dates = rxy.index
    dt = 1.0

    xret = rxy["x"].to_numpy(float)
    yret = rxy["y"].to_numpy(float)

    if detrend:
        xret = _detrend_linear(xret)
        yret = _detrend_linear(yret)

    if standardize:
        xs = xret.std(ddof=1)
        ys = yret.std(ddof=1)
        if xs == 0 or ys == 0:
            raise ValueError("One series has zero variance after preprocessing.")
        xret = (xret - xret.mean()) / xs
        yret = (yret - yret.mean()) / ys

    if debug:
        print("Final window:", dates.min(), "to", dates.max(), "n=", len(dates))

    mother = wavelet.Morlet(w0)
    _s0 = s0 if s0 is not None else 2 * dt
    if J is None:
        J = int(np.ceil(np.log2((period_max * 1.2) / _s0) / dj))

    # --- WTC (no significance inside pycwt; we'll do MC or AR1 ourselves) ---
    WCT, aWCT, coi, freq, _ = wavelet.wct(
        xret, yret, dt,
        dj=dj, s0=_s0, J=J,
        sig=False,
        wavelet=mother
    )

    period = 1.0 / freq
    keep = (period >= period_min) & (period <= period_max)

    period_k = period[keep]
    WCT_k = WCT[keep, :]
    aWCT_k = aWCT[keep, :]
    coi_k = np.clip(coi, period_min, period_max)

    phase = np.angle(aWCT_k) if np.iscomplexobj(aWCT_k) else np.asarray(aWCT_k, float)

    # --- significance (AR1 or MC-phase) ---
    sig_used = bool(sig)
    signif_plot = None
    sig_mask = np.ones_like(WCT_k, dtype=bool)

    if sig_used:
        if sig_method == "mc-phase":
            # Monte Carlo thresholds by period (global-by-period)
            period_thr, thr_k = _mc_phase_significance_threshold(
                xret, yret,
                dt=dt, dj=dj, s0=_s0, J=J,
                mother=mother,
                period_min=period_min,
                period_max=period_max,
                B=mc_B, alpha=mc_alpha,
                seed=mc_seed
            )
            # sanity: should match period_k closely
            if period_thr.shape != period_k.shape or np.max(np.abs(period_thr - period_k)) > 1e-9:
                # not fatal; just align by index length
                if debug:
                    print("[warn] MC period grid differs slightly; using computed thresholds as-is.")
            signif_plot = thr_k
            sig_mask = (WCT_k >= thr_k[:, None])

        elif sig_method == "ar1":
            # Try pycwt AR(1) red-noise significance.
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", message="Cannot place an upperbound*")
                    _, _, _, _, signif = wavelet.wct(
                        xret, yret, dt,
                        dj=dj, s0=_s0, J=J,
                        sig=True,
                        significance_level=significance_level,
                        wavelet=mother
                    )
                if signif is not None:
                    signif_arr = np.asarray(signif)
                    if signif_arr.ndim == 1:
                        signif_plot = signif_arr[keep]
                        sig_mask = (WCT_k >= signif_plot[:, None])
                    elif signif_arr.ndim == 2:
                        signif_plot = signif_arr[keep, :]
                        sig_mask = (WCT_k >= signif_plot)
                    else:
                        sig_used = False
                        signif_plot = None
            except Warning:
                sig_used = False
                signif_plot = None

        else:
            raise ValueError("sig_method must be {'mc-phase','ar1'}")

    # --- plot ---
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    levels = np.linspace(0, 1, 21)
    im = ax.contourf(dates, period_k, WCT_k, levels=levels, cmap=cmap, extend="both")

    # black contour for significant coherence regions
    if sig_used and (signif_plot is not None):
        ax.contour(
            dates, period_k,
            sig_mask.astype(float),
            levels=[0.5],          # <-- IMPORTANT: boundary of {0,1} mask
            colors="black",
            linewidths=1.2,
            zorder=25,
        )

    # COI shading
    ax.fill_between(dates, coi_k, period_max, facecolor="white", edgecolor="none", alpha=0.35, zorder=3)

    # arrows (optionally only in significant areas)
    mask = (WCT_k >= arrow_min_coh)
    if arrow_only_signif and sig_used and (signif_plot is not None):
        mask = mask & sig_mask

    _draw_fixed_phase_arrows(ax, dates, period_k, phase, mask)
    _paper_axis_style(ax)

    title_extra = ""
    if sig and not sig_used:
        title_extra = " (sig failed → no contour)"
    if sig_used and sig_method == "mc-phase":
        title_extra = f" (MC phase, B={mc_B}, α={mc_alpha:.2f})"

    ax.set_title(
        f"{x_name} vs {y_name} wavelet coherence{title_extra}\n{dates.min().date()} to {dates.max().date()}",
        fontsize=title_fontsize,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("Coherence", fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    def _abbr(name, n=6):
        """Abbreviate variable name for legend."""
        return name[:n]
        
    x_lab = _abbr(x_name)
    y_lab = _abbr(y_name)
    if show:
        # --- custom legend (phase interpretation + significance) ---
        
        legend_handles = [
            FancyArrowPatch((0, 0), (0.3, 0),
                             arrowstyle="-|>",
                             mutation_scale=12,
                             lw=1.2,
                             color="black",
                             label="→ Positive correlation"),
        
            FancyArrowPatch((0, 0), (-0.3, 0),
                             arrowstyle="-|>",
                             mutation_scale=12,
                             lw=1.2,
                             color="black",
                             label="← Negative correlation"),
        
            FancyArrowPatch((0, 0), (0, 0.3),
                             arrowstyle="-|>",
                             mutation_scale=12,
                             lw=1.2,
                             color="black",
                             label=f"↑ {y_lab} leads {x_lab}"),
        
            FancyArrowPatch((0, 0), (0, -0.3),
                             arrowstyle="-|>",
                             mutation_scale=12,
                             lw=1.2,
                             color="black",
                             label=f"↓ {x_lab} leads {y_lab}"),
        
            Line2D([0], [0],
                   color="black",
                   lw=1.5,
                   label="Significant coherence"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),   # <-- BELOW the x-axis
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.95,
            fontsize=8,
            handlelength=2.0,
            labelspacing=0.4,
            ncol=3,                        # compact horizontal layout
        )
        plt.show()

    return fig, ax