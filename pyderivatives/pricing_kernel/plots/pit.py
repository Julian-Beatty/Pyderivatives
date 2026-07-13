from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def plot_pit_calibration_panels(
    model,
    *,
    n_panels: int = 6,
    bins: int = 20,
    panel_shape: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    save=None,
    dpi: int = 200,
    show: bool = True,
):
    """
    Plot PIT histograms overlaid with the fitted calibration density.

    Works for:
        - BetaCalibration
        - NonparametricCalibration

    Requires a fitted model with:
        model.fit_history_by_T_
        model.models_by_T_

    Notes
    -----
    For beta calibration, the fitted density is Beta(a, b).

    For nonparametric calibration, the fitted density of PIT values is

        g(u) = h(Phi^{-1}(u)) / phi(Phi^{-1}(u)),

    where h is the KDE density in normal-score space.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import beta as beta_dist
    from scipy.stats import norm

    if not getattr(model, "is_fitted_", False):
        raise RuntimeError("Model must be fitted before plotting PIT calibration panels.")

    if not hasattr(model, "fit_history_by_T_") or not hasattr(model, "models_by_T_"):
        raise AttributeError("model must have fit_history_by_T_ and models_by_T_ attributes.")

    Ts = sorted(model.models_by_T_.keys())

    if len(Ts) == 0:
        raise RuntimeError("No fitted maturity models found.")

    n_panels = int(max(1, min(int(n_panels), len(Ts))))

    idxs = np.unique(
        np.linspace(0, len(Ts) - 1, n_panels)
        .round()
        .astype(int)
    )

    Ts_plot = [Ts[i] for i in idxs]

    if panel_shape is None:
        ncols = min(3, len(Ts_plot))
        nrows = int(np.ceil(len(Ts_plot) / ncols))
    else:
        nrows, ncols = map(int, panel_shape)

        if nrows <= 0 or ncols <= 0:
            raise ValueError("panel_shape must contain positive integers.")

        max_axes = nrows * ncols
        if len(Ts_plot) > max_axes:
            Ts_plot = Ts_plot[:max_axes]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.2 * ncols, 3.8 * nrows),
        sharex=True,
        sharey=False,
    )

    axes = np.asarray(axes).ravel()

    eps = float(getattr(model, "eps", 1e-10))
    u_grid = np.linspace(eps, 1.0 - eps, 1000)

    for ax, T in zip(axes, Ts_plot):
        hist = model.fit_history_by_T_.get(T)

        if hist is None or hist.empty or "pit" not in hist.columns:
            ax.set_title(f"T ≈ {365.0 * float(T):.0f}d: no PITs")
            ax.axis("off")
            continue

        u = np.asarray(hist["pit"], dtype=float)
        u = u[np.isfinite(u)]
        u = np.clip(u, eps, 1.0 - eps)

        if u.size == 0:
            ax.set_title(f"T ≈ {365.0 * float(T):.0f}d: no finite PITs")
            ax.axis("off")
            continue

        fitted = model.models_by_T_[T]

        ax.hist(
            u,
            bins=int(bins),
            range=(0.0, 1.0),
            density=True,
            alpha=0.35,
            edgecolor="black",
            label="PIT histogram",
        )

        # ------------------------------------------------------------
        # Beta calibration: PIT density is Beta(a,b).
        # ------------------------------------------------------------
        if hasattr(fitted, "a") and hasattr(fitted, "b"):
            g = beta_dist.pdf(u_grid, fitted.a, fitted.b)

            fitted_label = (
                rf"Beta fit: $a={float(fitted.a):.3g}$, "
                rf"$b={float(fitted.b):.3g}$"
            )

        # ------------------------------------------------------------
        # Nonparametric calibration:
        # z = Phi^{-1}(u), g(u) = h(z) / phi(z).
        # ------------------------------------------------------------
        elif hasattr(fitted, "z_grid") and hasattr(fitted, "h_grid"):
            z = norm.ppf(u_grid)

            h_z = np.interp(
                z,
                fitted.z_grid,
                fitted.h_grid,
                left=np.nan,
                right=np.nan,
            )

            phi_z = norm.pdf(z)
            g = h_z / np.maximum(phi_z, eps)
            g = np.where(np.isfinite(g) & (g >= 0), g, np.nan)

            fitted_label = (
                f"KDE fit: h={float(fitted.bandwidth):.3g}, "
                f"{fitted.bandwidth_method}"
            )

        else:
            raise TypeError(
                "Unsupported fitted model. Expected beta parameters "
                "(a,b) or nonparametric KDE fields (z_grid,h_grid)."
            )

        ax.plot(u_grid, g, linewidth=2.2, label=fitted_label)

        # Uniform PIT benchmark. If the model is calibrated, density should be near 1.
        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.2,
            label="Uniform benchmark",
        )

        ax.set_title(f"T ≈ {365.0 * float(T):.0f}d | n={len(u)}")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    for ax in axes[len(Ts_plot):]:
        ax.axis("off")

    if title is None:
        title = f"PIT Calibration Density Panels — {getattr(model, 'method_name', 'model')}"

    fig.suptitle(title)
    fig.tight_layout()

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=int(dpi), bbox_inches="tight")
        print(f"[saved] {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig