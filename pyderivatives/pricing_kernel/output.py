from __future__ import annotations

import numpy as np


def build_axis_grids(x_grid, S0):
    """
    Build standardized log-return, gross-return, and strike grids.

    Input x_grid is assumed to be the log-return grid.
    """
    grid_lr = np.asarray(x_grid, dtype=float).ravel()
    grid_r = np.exp(grid_lr)

    if S0 is not None and np.isfinite(S0):
        grid_k = float(S0) * grid_r
    else:
        grid_k = np.full_like(grid_r, np.nan)

    return grid_lr, grid_r, grid_k


def lr_surface_to_r_surface(lr_surface, grid_r):
    """
    Convert log-return density to gross-return density.

    R = exp(r), so f_R(R) = f_r(r) / R.
    """
    return np.asarray(lr_surface, dtype=float) / np.maximum(grid_r[None, :], 1e-300)


def lr_surface_to_k_surface(lr_surface, grid_k):
    """
    Convert log-return density to terminal-price/strike density.

    K = S0 * exp(r), so f_K(K) = f_r(r) / K.
    """
    return np.asarray(lr_surface, dtype=float) / np.maximum(grid_k[None, :], 1e-300)

def build_transform_output(
    *,
    method_name,
    info,
    fit_trim_alpha,
    T_grid,
    matched_T_grid,
    status_by_T,
    grid_lr,
    grid_r,
    grid_k,
    rnd_lr_surface,
    rnd_r_surface,
    rnd_k_surface,
    cdf_lr_surface,
    cdf_r_surface,
    cdf_k_surface,
    base_physical_lr_surface,
    base_physical_r_surface,
    base_physical_k_surface,
    base_physical_cdf_lr_surface,
    base_physical_cdf_r_surface,
    base_physical_cdf_k_surface,
    physical_lr_surface,
    physical_r_surface,
    physical_k_surface,
    physical_cdf_lr_surface,
    physical_cdf_r_surface,
    physical_cdf_k_surface,
    pricing_kernel_surface,
    rra_surface,
    measure_weight_surface,
    base_pricing_kernel_surface,
    base_measure_weight_surface,
    fit_diagnostics,
    S0,
    physical_moments,
    risk_neutral_moments,
    base_physical_moments,
):
    return {
        "success": True,
        "method": method_name,
        "ticker": info.get("ticker", None),
        "model": info.get("model", None),
        "params": info.get("params", None),
        "meta": info.get("meta", {}),
        "fit_trim_alpha": fit_trim_alpha,

        "T_grid": T_grid,
        "matched_T_grid": matched_T_grid,
        "transform_status_by_T": status_by_T,

        "grid_lr": grid_lr,
        "grid_r": grid_r,
        "grid_k": grid_k,

        "rnd_lr_surface": rnd_lr_surface,
        "rnd_r_surface": rnd_r_surface,
        "rnd_k_surface": rnd_k_surface,

        "cdf_lr_surface": cdf_lr_surface,
        "cdf_r_surface": cdf_r_surface,
        "cdf_k_surface": cdf_k_surface,

        "base_physical_lr_surface": base_physical_lr_surface,
        "base_physical_r_surface": base_physical_r_surface,
        "base_physical_k_surface": base_physical_k_surface,
        "base_physical_cdf_lr_surface": base_physical_cdf_lr_surface,
        "base_physical_cdf_r_surface": base_physical_cdf_r_surface,
        "base_physical_cdf_k_surface": base_physical_cdf_k_surface,

        "physical_lr_surface": physical_lr_surface,
        "physical_r_surface": physical_r_surface,
        "physical_k_surface": physical_k_surface,
        "physical_cdf_lr_surface": physical_cdf_lr_surface,
        "physical_cdf_r_surface": physical_cdf_r_surface,
        "physical_cdf_k_surface": physical_cdf_k_surface,

        "pricing_kernel_surface": pricing_kernel_surface,
        "relative_risk_aversion_surface": rra_surface,
        "measure_weight_surface": measure_weight_surface,
        "base_pricing_kernel_surface": base_pricing_kernel_surface,
        "base_measure_weight_surface": base_measure_weight_surface,

        "fit_diagnostics": fit_diagnostics,

        "S0": S0,
        "r": info.get("r", np.nan),
        "q": info.get("q", 0.0),

        "physical_moments": physical_moments,
        "risk_neutral_moments": risk_neutral_moments,
        "base_physical_moments": base_physical_moments,
    }