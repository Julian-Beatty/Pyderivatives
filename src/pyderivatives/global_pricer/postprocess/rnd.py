from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class SafetyClipConfig:
    """
    Spike guardrail applied AFTER Breeden–Litzenberger.

    center:
      - "spot": use S0
      - "mode": argmax(q)
      - "meanS": compute mean under q (after normalization)
    """
    enabled: bool = False

    center: str = "mode"          # "spot" | "mode" | "meanS"
    left_jump: float = np.e
    right_jump: float = np.e
    clip_left: bool = True
    clip_right: bool = False

    eps: float = 1e-30            # value to set in clipped region (matches your archive)
    min_floor: float = 1e-30      # ratio guard floor


def breeden_litzenberger_pdf(
    C_fit: np.ndarray,
    *,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    r: float,
    floor: float = 1e-12,   # tiny positive mass
) -> np.ndarray:
    """
    Pure Breeden–Litzenberger on a rectangular grid:
      f_T(K) = e^{rT} * d^2 C(K,T) / dK^2
    """
    C_fit = np.asarray(C_fit, float)
    K_grid = np.asarray(K_grid, float).ravel()
    T_grid = np.asarray(T_grid, float).ravel()

    if C_fit.shape != (T_grid.size, K_grid.size):
        raise ValueError("C_fit must have shape (len(T_grid), len(K_grid)).")
    if np.any(np.diff(K_grid) <= 0):
        raise ValueError("K_grid must be strictly increasing.")

    dC_dK = np.gradient(C_fit, K_grid, axis=1, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, K_grid, axis=1, edge_order=2)

    scale = np.exp(float(r) * T_grid).reshape(-1, 1)
    q = scale * d2C_dK2

    # ---- CLIP NEGATIVE DENSITY ----
    q = np.maximum(q, floor)
    Z = np.trapz(q, K_grid, axis=1).reshape(-1, 1)
    q = q / Z
    return q



def _clip_tails_if_exp_jump_row(
    qi: np.ndarray,
    K: np.ndarray,
    *,
    S0: float,
    eps: float,
    center: str,
    left_jump: float,
    right_jump: float,
    clip_left: bool,
    clip_right: bool,
    min_floor: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Row version of your archived method. Returns (q_clipped, info_dict).
    """
    qi = np.asarray(qi, float).copy()
    K = np.asarray(K, float)

    n = qi.size
    info: Dict[str, Any] = {
        "used": False,
        "clipped_left": False, "left_cut_idx": None, "left_K_cut": None, "left_ratio": None,
        "clipped_right": False, "right_cut_idx": None, "right_K_cut": None, "right_ratio": None,
        "center": center,
        "left_jump": float(left_jump),
        "right_jump": float(right_jump),
    }
    if n < 5:
        return qi, info

    # ---- choose center index ----
    if center == "spot":
        ic = int(np.argmin(np.abs(K - float(S0))))
    elif center == "mode":
        ic = int(np.argmax(qi))
    elif center == "meanS":
        area = float(np.trapz(qi, K))
        if not (np.isfinite(area) and area > 0):
            return qi, info
        qn = qi / area
        meanS = float(np.trapz(K * qn, K))
        ic = int(np.argmin(np.abs(K - meanS)))
        info["meanS"] = meanS
    else:
        raise ValueError("center must be 'spot', 'mode', or 'meanS'")

    # ---- left scan: compare q[j] to q[j+1] ----
    if clip_left:
        for j in range(ic - 1, -1, -1):
            right = max(qi[j + 1], min_floor)
            left = max(qi[j], min_floor)
            ratio = left / right
            if ratio > left_jump:
                qi[: j + 1] = eps
                info.update({
                    "used": True,
                    "clipped_left": True,
                    "left_cut_idx": j,
                    "left_K_cut": float(K[j]),
                    "left_ratio": float(ratio),
                })
                break

    # ---- right scan: compare q[j] to q[j-1] ----
    if clip_right:
        for j in range(ic + 1, n):
            left_neighbor = max(qi[j - 1], min_floor)
            cur = max(qi[j], min_floor)
            ratio = cur / left_neighbor
            if ratio > right_jump:
                qi[j:] = eps
                info.update({
                    "used": True,
                    "clipped_right": True,
                    "right_cut_idx": j,
                    "right_K_cut": float(K[j]),
                    "right_ratio": float(ratio),
                })
                break

    return qi, info


def apply_safety_clip_surface(
    pdf: np.ndarray,
    *,
    K_grid: np.ndarray,
    S0: float,
    cfg: SafetyClipConfig,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Apply tail spike clip row-by-row (per maturity).
    Returns (pdf_clipped, info_list_per_row).
    """
    pdf = np.asarray(pdf, float)
    K_grid = np.asarray(K_grid, float).ravel()
    if pdf.shape[1] != K_grid.size:
        raise ValueError("pdf must have shape (nT, len(K_grid)).")

    if not cfg.enabled:
        info = [{"used": False, "center": cfg.center, "left_jump": cfg.left_jump, "right_jump": cfg.right_jump}
                for _ in range(pdf.shape[0])]
        return pdf, info

    out = pdf.copy()
    infos: List[Dict[str, Any]] = []
    for i in range(pdf.shape[0]):
        qi, info = _clip_tails_if_exp_jump_row(
            out[i, :], K_grid,
            S0=float(S0),
            eps=float(cfg.eps),
            center=str(cfg.center),
            left_jump=float(cfg.left_jump),
            right_jump=float(cfg.right_jump),
            clip_left=bool(cfg.clip_left),
            clip_right=bool(cfg.clip_right),
            min_floor=float(cfg.min_floor),
        )
        out[i, :] = qi
        infos.append(info)
    return out, infos
def strike_rnd_to_return_density(
    qS: np.ndarray,
    *,
    K_grid: np.ndarray,
    S0: float,
    return_type: str = "log",   # "log" or "gross"
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    qS = np.asarray(qS, float)
    K = np.asarray(K_grid, float).ravel()

    if np.any(K <= 0):
        raise ValueError("K_grid must be strictly positive.")
    if qS.shape[-1] != K.size:
        raise ValueError("Last dimension of qS must match len(K_grid).")

    S0 = float(S0)

    if return_type == "log":
        grid = np.log(K / S0)          # r grid
        jac = S0 * np.exp(grid)        # dS/dr
        q_ret = qS * jac
    elif return_type == "gross":
        grid = K / S0                  # R grid
        jac = S0                       # dS/dR
        q_ret = qS * jac
    else:
        raise ValueError("return_type must be 'log' or 'gross'")

    if normalize:
        if q_ret.ndim == 1:
            Z = np.trapz(q_ret, grid)
            q_ret = q_ret / Z
        else:
            Z = np.trapz(q_ret, grid, axis=1).reshape(-1, 1)
            q_ret = q_ret / Z

    return grid, q_ret

