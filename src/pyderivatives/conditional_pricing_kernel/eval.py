from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from .kernel import g_r_sigma
from .moments import physical_moments_table
from .config import SafetyClipSpec

import numpy as np
import matplotlib.pyplot as plt
def plot_kernel_bootstrap_quantiles_panel(
    *,
    anchor_day: dict,
    pk_fit: dict,
    theta_spec,
    eval_spec,
    j_indices=None,                  # list of maturity indices to plot
    n_panels: int = 6,                # if j_indices=None, auto-pick this many
    panel_shape: tuple[int, int] = (2, 3),
    safety_clip=None,
    q_line: float = 0.95,
    q_band=(0.025, 0.975),
    R_bounds=None,
    r_bounds=None,
    logy: bool = True,
    title: str | None = None,
):
    """
    Panel plot of bootstrap kernel quantiles for multiple maturities
    on a single anchor date.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ----- choose maturities -----
    T_grid = np.asarray(pk_fit["T_grid"], float)
    nT = T_grid.size

    if j_indices is None:
        # evenly spaced maturities
        j_indices = np.linspace(0, nT - 1, n_panels, dtype=int)
    else:
        j_indices = np.asarray(j_indices, int)

    # ----- base evaluation (point estimate + grids) -----
    base = evaluate_anchor_surfaces_with_theta_master(
        anchor_day,
        theta_master=pk_fit["theta_master"],
        theta_spec=theta_spec,
        eval_spec=eval_spec,
        safety_clip=safety_clip,
    )

    x = np.asarray(base["R_common"], float)
    r = np.asarray(base["r_common"], float)

    # ----- truncation mask (shared across panels) -----
    mask = np.ones_like(x, dtype=bool)
    if R_bounds is not None:
        Rmin, Rmax = map(float, R_bounds)
        mask &= (x >= Rmin) & (x <= Rmax)
    if r_bounds is not None:
        rmin, rmax = map(float, r_bounds)
        mask &= (r >= rmin) & (r <= rmax)

    xm = x[mask]

    # ----- setup figure -----
    nrows, ncols = panel_shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()

    for ax, j in zip(axes, j_indices):
        M_hat = np.asarray(base["anchor_surfaces"]["M_surface"][j], float)

        # bootstrap draws
        draws_by_T = pk_fit["theta_boot_draws_by_T"]
        if draws_by_T is None or draws_by_T[j] is None:
            ax.set_visible(False)
            continue

        theta_draws = np.asarray(draws_by_T[j], float)
        B = theta_draws.shape[0]

        M_boot = np.empty((B, x.size), float)
        for b in range(B):
            theta_tmp = np.array(pk_fit["theta_master"], float, copy=True)
            theta_tmp[j, :] = theta_draws[b, :]

            out_b = evaluate_anchor_surfaces_with_theta_master(
                anchor_day,
                theta_master=theta_tmp,
                theta_spec=theta_spec,
                eval_spec=eval_spec,
                safety_clip=safety_clip,
            )
            M_boot[b, :] = np.asarray(out_b["anchor_surfaces"]["M_surface"][j], float)

        # quantiles
        qcurve = np.quantile(M_boot, q_line, axis=0)
        lo = np.quantile(M_boot, q_band[0], axis=0)
        hi = np.quantile(M_boot, q_band[1], axis=0)

        # apply truncation
        Mh = M_hat[mask]
        qc = qcurve[mask]
        lom = lo[mask]
        him = hi[mask]

        # plot
        ax.plot(xm, Mh, linewidth=2, label="M")
        ax.fill_between(xm, lom, him, alpha=0.25)
        ax.plot(xm, qc, linewidth=2, linestyle="--")

        ax.set_title(f"T ≈ {T_grid[j]*365:.0f}d")
        if logy:
            ax.set_yscale("log")

    # clean unused axes
    for ax in axes[len(j_indices):]:
        ax.set_visible(False)

    # labels & legend
    axes[0].legend(["M (point)", f"{int((q_band[1]-q_band[0])*100)}% band", f"{int(q_line*100)}th pct"], loc="best")
    fig.suptitle(title or "Bootstrap pricing kernel quantiles (by maturity)")
    fig.supxlabel("Gross return R")
    fig.supylabel("Pricing kernel M(R)")
    fig.tight_layout()
    plt.show()

def plot_kernel_bootstrap_quantiles_single_anchor(
    *,
    anchor_day: dict,
    pk_fit: dict,
    theta_spec,
    eval_spec,
    j: int,
    safety_clip=None,
    q_line: float = 0.95,              # single quantile curve (e.g. 95th percentile)
    q_band=(0.025, 0.975),             # shaded pointwise band (e.g. 95% band)
    logy: bool = True,
    title: str | None = None,
    R_bounds=None,                     # tuple (Rmin, Rmax) in gross return space
    r_bounds=None,                     # tuple (rmin, rmax) in log-return space
):
    """
    Single anchor date, single maturity index j.

    Uses pk_fit["theta_boot_draws_by_T"][j] to form a pointwise bootstrap
    distribution of M(R) for that maturity, then plots:
      - point estimate M(R)
      - optional shaded band [q_band[0], q_band[1]]
      - optional quantile curve at q_line
    """
    import numpy as np
    import matplotlib.pyplot as plt

    j = int(j)

    # ----- 1) Point estimate (and grids) -----
    base = evaluate_anchor_surfaces_with_theta_master(
        anchor_day,
        theta_master=pk_fit["theta_master"],
        theta_spec=theta_spec,
        eval_spec=eval_spec,
        safety_clip=safety_clip,
    )

    x = np.asarray(base["R_common"], float)      # gross return grid
    r = np.asarray(base["r_common"], float)      # log return grid
    M_hat = np.asarray(base["anchor_surfaces"]["M_surface"][j], float)

    # ----- 2) Bootstrap draws of theta for this maturity -----
    draws_by_T = pk_fit.get("theta_boot_draws_by_T", None)
    if draws_by_T is None or j >= len(draws_by_T) or draws_by_T[j] is None:
        raise ValueError(
            "No theta bootstrap draws found for this maturity. "
            "Fit with bootstrap.enabled=True and keep_draws=True."
        )

    theta_draws = np.asarray(draws_by_T[j], float)  # (B, p)
    B = theta_draws.shape[0]

    # ----- 3) Re-evaluate M(R) for each bootstrap draw -----
    M_boot = np.empty((B, x.size), float)
    for b in range(B):
        theta_tmp = np.array(pk_fit["theta_master"], float, copy=True)
        theta_tmp[j, :] = theta_draws[b, :]

        out_b = evaluate_anchor_surfaces_with_theta_master(
            anchor_day,
            theta_master=theta_tmp,
            theta_spec=theta_spec,
            eval_spec=eval_spec,
            safety_clip=safety_clip,
        )
        M_boot[b, :] = np.asarray(out_b["anchor_surfaces"]["M_surface"][j], float)

    # ----- 4) Pointwise quantiles -----
    qcurve = np.quantile(M_boot, q_line, axis=0) if q_line is not None else None
    lo = np.quantile(M_boot, q_band[0], axis=0) if q_band is not None else None
    hi = np.quantile(M_boot, q_band[1], axis=0) if q_band is not None else None

    # ----- 5) Optional truncation mask (apply at the end) -----
    mask = np.ones_like(x, dtype=bool)

    if R_bounds is not None:
        Rmin, Rmax = map(float, R_bounds)
        mask &= (x >= Rmin) & (x <= Rmax)

    if r_bounds is not None:
        rmin, rmax = map(float, r_bounds)
        mask &= (r >= rmin) & (r <= rmax)

    xm = x[mask]
    Mh = M_hat[mask]
    qc = qcurve[mask] if qcurve is not None else None
    lom = lo[mask] if lo is not None else None
    him = hi[mask] if hi is not None else None

    # ----- 6) Plot -----
    plt.figure()
    plt.plot(xm, Mh, linewidth=2, label="M (point)")

    if lom is not None and him is not None:
        band_pct = int(round(100.0 * (q_band[1] - q_band[0])))
        plt.fill_between(xm, lom, him, alpha=0.25, label=f"{band_pct}% band")

    #if qc is not None:
        #plt.plot(xm, qc, linewidth=2, label=f"{int(round(100*q_line))}th percentile")

    if logy:
        plt.yscale("log")

    plt.xlabel("Gross return R")
    plt.ylabel("Pricing kernel M(R)")
    plt.legend()
    plt.title(title or f"Kernel bootstrap quantiles (j={j})")
    plt.show()


def _safety_clip_from_mode(y: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    """Enforce unimodal / monotone tails around the mode.

    Given y on a 1D grid:
      - Let m = argmax(y)
      - For i=m-1..0: y[i] = min(y[i], y[i+1])
      - For i=m+1..end: y[i] = min(y[i], y[i-1])

    Returns a *new* array.
    """
    yy = np.asarray(y, float).copy()
    if yy.size == 0:
        return yy
    if not np.any(np.isfinite(yy)):
        return yy
    # replace non-finite with 0 for clipping pass
    yy[~np.isfinite(yy)] = 0.0
    if float(floor) > 0.0:
        yy = np.maximum(yy, float(floor))

    m = int(np.argmax(yy))
    # left
    for i in range(m - 1, -1, -1):
        if yy[i] > yy[i + 1]:
            yy[i] = yy[i + 1]
    # right
    for i in range(m + 1, yy.size):
        if yy[i] > yy[i - 1]:
            yy[i] = yy[i - 1]
    return yy


def _renormalize_density_on_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return y
    area = np.trapz(y[mask], x[mask])
    if not np.isfinite(area) or area <= 0:
        return y
    out = y.copy()
    out[mask] = out[mask] / area
    return out


def evaluate_anchor_surfaces_with_theta_master(
    anchor_dict: dict,
    *,
    theta_master: np.ndarray,      # (nT, p)
    theta_spec,                    # ThetaSpec
    eval_spec,                     # EvalSpec
    safety_clip: Optional[SafetyClipSpec] = None,
) -> Dict[str, Any]:
    """Produces an output dict for ONE anchor date.

    If safety_clip.enabled=True, we post-process pR_surface[T,:] by clipping from
    the mode outward and renormalizing each maturity slice. Moments are computed
    from the *clipped* physical density.

    Returns both:
      - strike-space densities (qK_surface, pK_surface, mK_surface)
      - gross-return densities (qR_surface, pR_surface)
      - log-return densities (fQ_r_surface, fP_r_surface)
    """
    T_grid = np.asarray(anchor_dict["T_grid"], float).ravel()
    K_grid = np.asarray(anchor_dict["K_grid"], float).ravel()
    qK = np.asarray(anchor_dict["rnd_surface"], float)  # (nT, nK)
    S0 = float(anchor_dict["S0"])
    sigma = float(anchor_dict["atm_vol"])
    r_rf = float(anchor_dict.get("r", 0.0))

    r_lo, r_hi = eval_spec.r_bounds
    r_common = np.linspace(r_lo, r_hi, int(eval_spec.r_grid_size))
    R_common = np.exp(r_common)

    nT = T_grid.size
    nR = r_common.size

    # --- surfaces on r-grid (common) ---
    qR_surf  = np.full((nT, nR), np.nan)
    pR_surf  = np.full((nT, nR), np.nan)
    fQ_surf  = np.full((nT, nR), np.nan)   # log-return RND  f_Q(r)
    fP_surf  = np.full((nT, nR), np.nan)   # log-return phys f_P(r)
    M_surf   = np.full((nT, nR), np.nan)
    RRA_surf = np.full((nT, nR), np.nan)

    # Build fQ(r) from q(K) row by row
    for j in range(nT):
        theta_j = np.asarray(theta_master[j], float)
        if not np.all(np.isfinite(theta_j)):
            continue

        qK_row = np.asarray(qK[j, :], float)
        K = np.asarray(K_grid, float)
        mask = np.isfinite(K) & np.isfinite(qK_row) & (K > 0)
        if mask.sum() < 10:
            continue

        K2 = K[mask]
        qK2 = qK_row[mask]

        # log-return grid implied by available strikes
        rK = np.log(K2 / S0)
        Rk = np.exp(rK)

        # ---- risk-neutral log-return density f_Q(r) ----
        # q_R(R) = q_K(K)*S0  and  f_Q(r) = q_R(R)*R
        # => f_Q(r) = q_K(S0*R) * S0 * R
        fQ = (qK2 * S0) * Rk
        fQ_grid = np.interp(r_common, rK, fQ, left=0.0, right=0.0)

        # ---- physical log-return density f_P(r) via weighting ----
        g = g_r_sigma(r_common, sigma, theta_j, N=theta_spec.N, Ksig=theta_spec.Ksig)

        w = fQ_grid * np.exp(-g)
        mass = np.trapz(w, r_common)
        if not np.isfinite(mass) or mass <= 0:
            continue

        fP_grid = w / mass

        # Pricing kernel in levels up to normalization: M = exp(delta + g)
        # with delta = log(mass) (old code convention)
        delta = np.log(mass)
        M_grid = np.exp(delta + g)

        # ---- convert log-return densities to gross-return densities ----
        # f_R(R) = f_r(r) / R
        qR = fQ_grid / R_common
        pR = fP_grid / R_common

        # --- optional safety clip on *physical* density in R-space ---
        if safety_clip is not None and bool(safety_clip.enabled):
            pR = _safety_clip_from_mode(pR, floor=float(getattr(safety_clip, "floor", 0.0)))
            pR = _renormalize_density_on_grid(R_common, pR)

            # keep log-return physical density consistent with clipped pR:
            # fP(r) = pR(R) * R
            fP_grid = pR * R_common

        # ---- store surfaces ----
        qR_surf[j, :]  = qR
        pR_surf[j, :]  = pR
        fQ_surf[j, :]  = fQ_grid
        fP_surf[j, :]  = fP_grid
        M_surf[j, :]   = M_grid

        # Relative risk aversion proxy often computed from log M slope in r:
        # RRA(R) = - d log M / d log R = - d log M / dr
        dlogM_dr = np.gradient(np.log(np.maximum(M_grid, 1e-300)), r_common)
        RRA_surf[j, :] = -dlogM_dr

    # ---- derived strike-space surfaces on common grid ----
    K_common = S0 * R_common

    qK_surface = qR_surf / S0
    pK_surface = pR_surf / S0
    mK_surface = qK_surface / np.maximum(pK_surface, 1e-300)

    # moments want pr_surface on r-grid: pr(r) = pR(R) * R
    pr_surf = pR_surf * R_common[None, :]
    moments_df = physical_moments_table(T_grid=T_grid, r_common=r_common, pr_surface=pr_surf)

    dict_result = {
        "T_anchor": T_grid,
        "R_common": R_common,
        "r_common": r_common,
        "K_common": K_common,
        "sigma_value": sigma,
        "anchor_surfaces": {
            # gross-return densities
            "qR_surface": qR_surf,
            "pR_surface": pR_surf,

            # log-return densities (NEW)
            "fQ_r_surface": fQ_surf,   # risk-neutral log-return density
            "fP_r_surface": fP_surf,   # physical log-return density

            # pricing kernel objects
            "M_surface": M_surf,
            "RRA_surface": RRA_surf,

            # strike densities on the same common grid
            "qK_surface": qK_surface,
            "pK_surface": pK_surface,
            "mK_surface": mK_surface,
        },
        "physical_moments_table": moments_df,
        "meta": {"S0": S0, "r": r_rf},
        "safety_clip": {
            "enabled": bool(safety_clip.enabled) if safety_clip is not None else False,
            "floor": float(getattr(safety_clip, "floor", 0.0)) if safety_clip is not None else 0.0,
        },
    }

    return dict_result
