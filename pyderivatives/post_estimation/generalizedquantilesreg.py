import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from typing import Optional, Sequence, Dict, Tuple, Union, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from typing import Optional, Sequence, Dict, Tuple, Union, Any

import os
from pathlib import Path
from math import erf, sqrt
from typing import Optional, Sequence, Dict, Tuple, Union, Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from dataclasses import dataclass
# ============================================================
# FULL COPY-PASTE MODULE
# Diagnostics + optional test-stat sampling-distribution plots
# (hist + QQ), and storage of bootstrap estimators + Wald cov mats.
# ============================================================


from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.stattools import jarque_bera

# ============================================================
# FULL COPY-PASTE MODULE (ALL TESTS 1–6 KEPT)
#
# Your requested changes (implemented):
# ✅ Histograms ALWAYS mark the observed statistic (T_obs / Obs) with a vertical line
# ✅ Each panel also annotates p-value (when available) so you can eyeball it
# ✅ Tail-extremeness-vs-middle (test 6) now ALSO produces bootstrap distributions
#    and saves histogram plots for:
#       - High tail vs Middle (mean|beta_high| - mean|beta_mid|)
#       - Low tail vs Middle  (mean|beta_low|  - mean|beta_mid|)
#
# Still included:
# ✅ No QQ plots
# ✅ 1x4 hist panels (Global | Low | Mid | High) when possible
# ✅ Journal-friendly titles (no underscores)
# ✅ Organized save tree:
#    <plot_dir>/<prefix>/<horizon>/<equation>/<test group>/<stat>.png
# ============================================================


from math import erf, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg


from math import erf, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# ============================================================
# Bootstrap utilities
# ============================================================

def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap indices of length n."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    if block_len == 1:
        return rng.integers(0, n, size=n)

    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx: List[int] = []
    for s in starts:
        idx.extend([(s + k) % n for k in range(block_len)])
    return np.asarray(idx[:n], dtype=int)


def _make_bootstrap_index_matrix(
    n: int,
    *,
    B: int,
    bootstrap: str,
    block_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pre-generate bootstrap indices; shape (B, n).
    SAME draw b for all taus so cross-quantile tests are valid.
    """
    if B < 1:
        raise ValueError("B must be >= 1")
    idx_mat = np.empty((B, n), dtype=int)
    for b in range(B):
        if bootstrap == "iid":
            idx = rng.integers(0, n, size=n)
        elif bootstrap == "block":
            idx = _block_bootstrap_indices(n, block_len=block_len, rng=rng)
        else:
            raise ValueError("bootstrap must be one of {'iid','block'}")
        idx_mat[b, :] = idx
    return idx_mat


# ============================================================
# Aligned bootstrap QuantReg fit
# ============================================================

def _quantreg_fit_bootstrap(
    y: np.ndarray,
    X: np.ndarray,
    *,
    taus: Tuple[float, ...],
    B: int,
    bootstrap: str,
    block_len: int,
    seed: Optional[int],
    fit_kwargs: Optional[dict],
    max_retries_per_draw: int = 3,
    verbose: bool = False,
    progress_every: int = 25,
) -> Tuple[
    Dict[float, np.ndarray],   # params
    Dict[float, np.ndarray],   # se_boot
    Dict[float, Any],          # results
    Dict[float, np.ndarray],   # boot_params
    Dict[str, Any],            # boot_meta
]:
    """
    Fits QuantReg for each tau and computes bootstrap SEs by resampling rows.

    CRITICAL ALIGNMENT GUARANTEE:
      Draw j corresponds to the SAME resampled dataset for ALL taus.
      We therefore NEVER "fallback-resample" a new dataset inside the tau loop.
      If a draw fails for a tau, we retry solver attempts on the SAME dataset,
      otherwise leave that (j,tau) as NaN.
    """
    fit_kwargs = fit_kwargs or {}
    rng = np.random.default_rng(seed)

    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y and X must have same number of rows")

    idx_mat = _make_bootstrap_index_matrix(
        n, B=B, bootstrap=bootstrap, block_len=block_len, rng=rng
    )

    params: Dict[float, np.ndarray] = {}
    se_boot: Dict[float, np.ndarray] = {}
    results: Dict[float, Any] = {}
    boot_params: Dict[float, np.ndarray] = {}

    fail_count_by_tau: Dict[float, int] = {}
    nan_rows_by_tau: Dict[float, int] = {}

    n_taus = len(taus)

    if verbose:
        print(f"[BOOT] start | n={n}, p={p}, B={B}, n_taus={n_taus}, bootstrap={bootstrap}, block_len={block_len}, seed={seed}")

    for tau_i, tau in enumerate(taus, start=1):
        if verbose:
            print(f"[BOOT] tau {tau_i}/{n_taus} = {tau:.2f} | fitting original sample")

        res = QuantReg(y, X).fit(q=tau, **fit_kwargs)
        results[tau] = res
        params[tau] = np.asarray(res.params, dtype=float)

        bmat = np.full((B, p), np.nan, float)
        fails = 0

        for j in range(B):
            idx = idx_mat[j]
            yb = y[idx]
            Xb = X[idx, :]

            ok = False
            for _ in range(max_retries_per_draw):
                try:
                    rb = QuantReg(yb, Xb).fit(q=tau, **fit_kwargs)
                    bmat[j, :] = np.asarray(rb.params, dtype=float)
                    ok = True
                    break
                except Exception:
                    fails += 1
                    continue

            if not ok:
                bmat[j, :] = np.nan

            if verbose:
                done = j + 1
                if (done == 1) or (done % progress_every == 0) or (done == B):
                    pct = 100.0 * done / B
                    print(
                        f"[BOOT] tau {tau_i}/{n_taus} = {tau:.2f} | "
                        f"draw {done}/{B} ({pct:5.1f}%) | fails={fails}"
                    )

        boot_params[tau] = bmat
        se_boot[tau] = np.nanstd(bmat, axis=0, ddof=1)

        fail_count_by_tau[tau] = int(fails)
        nan_rows_by_tau[tau] = int(np.sum(~np.isfinite(bmat).all(axis=1)))

        if verbose:
            print(
                f"[BOOT] tau {tau_i}/{n_taus} = {tau:.2f} done | "
                f"nan_rows={nan_rows_by_tau[tau]} | fails={fail_count_by_tau[tau]}"
            )

    boot_meta = {
        "aligned_draws_across_taus": True,
        "B": int(B),
        "n": int(n),
        "p": int(p),
        "bootstrap": bootstrap,
        "block_len": int(block_len),
        "seed": seed,
        "fail_count_by_tau": fail_count_by_tau,
        "nan_rows_by_tau": nan_rows_by_tau,
        "max_retries_per_draw": int(max_retries_per_draw),
        "note": "No fallback-resampling inside tau loop; NaNs mark failed tau/draw fits.",
    }

    if verbose:
        print("[BOOT] finished")

    return params, se_boot, results, boot_params, boot_meta


# ============================================================
# Quantile pseudo-R2 + NW default lags
# ============================================================

def _rho_check(u: np.ndarray, tau: float) -> np.ndarray:
    """Check loss rho_tau(u) = u*(tau - 1{u<0})."""
    u = np.asarray(u, dtype=float)
    return u * (tau - (u < 0.0).astype(float))


def _quantile_pseudo_r2(y: np.ndarray, X: np.ndarray, beta: np.ndarray, tau: float) -> float:
    """
    Koenker–Machado pseudo-R^2:
        1 - sum rho_tau(y - Xb) / sum rho_tau(y - q_tau(y))
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    beta = np.asarray(beta, dtype=float)

    resid = y - X @ beta
    num = float(np.sum(_rho_check(resid, tau)))

    q = float(np.quantile(y, tau))
    den = float(np.sum(_rho_check(y - q, tau)))

    if den <= 0.0 or not np.isfinite(den):
        return np.nan
    return float(1.0 - num / den)


def _newey_west_default_lags(n: int) -> int:
    """Automatic NW/HAC bandwidth: floor(4*(n/100)^(2/9))."""
    if n <= 0:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _pval_from_t_normal(tt: float) -> float:
    Phi = 0.5 * (1.0 + erf(abs(float(tt)) / sqrt(2.0)))
    return 2.0 * (1.0 - Phi)


# ============================================================
# Segment helpers
# ============================================================

def _split_taus_into_thirds(taus_sorted: List[float]) -> Dict[str, List[float]]:
    """Split ordered taus into ~3 equal contiguous segments."""
    K = len(taus_sorted)
    i1 = K // 3
    i2 = (2 * K) // 3
    return {
        "low": taus_sorted[:i1],
        "mid": taus_sorted[i1:i2],
        "high": taus_sorted[i2:],
    }


# ============================================================
# Joint tests (non-dominance tests retained)
# ============================================================

def _joint_significance_everywhere(
    g_b: np.ndarray,
    *,
    alpha: float = 0.05,
    eps: float = 1e-12,
    return_boot_dist: bool = False,
) -> Dict[str, Any]:
    """
    Everywhere significant test:
      Reject iff min_tau |t_obs| > c_{1-alpha}, where c from bootstrap max-|t*|.
    """
    g_b = np.asarray(g_b, float)
    g_hat = np.nanmean(g_b, axis=0)
    se_hat = np.nanstd(g_b, axis=0, ddof=1) + eps

    t_obs = g_hat / se_hat
    S_obs = float(np.nanmin(np.abs(t_obs)))

    t_star = (g_b - g_hat) / se_hat
    T_star = np.nanmax(np.abs(t_star), axis=1)

    c = float(np.nanquantile(T_star, 1.0 - alpha))
    reject = bool(S_obs > c)
    p = float(np.nanmean(T_star >= S_obs))

    out = {
        "stat": "Everywhere significant (max-|t| critical value)",
        "alpha": float(alpha),
        "crit": c,
        "obs_value": S_obs,
        "p_value": p,
        "reject": reject,
        "x_label": "Bootstrap max-|t*|",
    }
    if return_boot_dist:
        out["boot_draws"] = T_star
    return out


def _maxabs_t_omnibus_pvalue(
    g_b: np.ndarray,
    eps: float = 1e-12,
    *,
    return_boot_dist: bool = False,
) -> Dict[str, Any]:
    """
    SOMEWHERE omnibus:
      T_obs = max_tau |t_obs|
      T*_b  = max_tau |(g_b - g_hat)/se_hat|
    """
    g_b = np.asarray(g_b, float)
    g_hat = np.nanmean(g_b, axis=0)
    se_hat = np.nanstd(g_b, axis=0, ddof=1) + eps

    t_obs = g_hat / se_hat
    T_obs = float(np.nanmax(np.abs(t_obs)))

    t_star = (g_b - g_hat) / se_hat
    T_star = np.nanmax(np.abs(t_star), axis=1)
    p = float(np.nanmean(T_star >= T_obs))

    out = {
        "stat": "Somewhere significant (max-|t|)",
        "obs_value": T_obs,
        "p_value": p,
        "x_label": "Bootstrap max-|t*|",
    }
    if return_boot_dist:
        out["boot_draws"] = T_star
    return out


def _l2_t2_omnibus_pvalue(
    g_b: np.ndarray,
    eps: float = 1e-12,
    *,
    return_boot_dist: bool = False,
) -> Dict[str, Any]:
    """
    OVERALL omnibus:
      T_obs = sum_tau t_obs^2
      T*_b  = sum_tau t*_b^2
    """
    g_b = np.asarray(g_b, float)
    g_hat = np.nanmean(g_b, axis=0)
    se_hat = np.nanstd(g_b, axis=0, ddof=1) + eps

    t_obs = g_hat / se_hat
    T_obs = float(np.nansum(t_obs**2))

    t_star = (g_b - g_hat) / se_hat
    T_star = np.nansum(t_star**2, axis=1)
    p = float(np.nanmean(T_star >= T_obs))

    out = {
        "stat": "Overall significance (sum of t-squared)",
        "obs_value": T_obs,
        "p_value": p,
        "x_label": "Bootstrap sum of t*²",
    }
    if return_boot_dist:
        out["boot_draws"] = T_star
    return out


def _wald_omnibus_pvalue(
    g_b: np.ndarray,
    *,
    ridge: float = 1e-8,
    use_pinv: bool = True,
    return_boot_dist: bool = False,
    store_mats: bool = False,
) -> Dict[str, Any]:
    """
    Wald omnibus using bootstrap covariance across taus.
    """
    g_b = np.asarray(g_b, float)
    g_hat = np.nanmean(g_b, axis=0)
    g_star = g_b - g_hat

    Sigma = np.cov(g_star, rowvar=False, ddof=1)

    K = Sigma.shape[0]
    tr = float(np.trace(Sigma))
    lam = float(ridge * (tr / K if (K > 0 and np.isfinite(tr) and tr > 0) else 1.0))
    Sigma_reg = Sigma + lam * np.eye(K)

    if use_pinv:
        Sigma_inv = np.linalg.pinv(Sigma_reg)
    else:
        Sigma_inv = np.linalg.inv(Sigma_reg)

    T_obs = float(g_hat.T @ Sigma_inv @ g_hat)
    T_star = np.einsum("bi,ij,bj->b", g_star, Sigma_inv, g_star)
    p = float(np.nanmean(T_star >= T_obs))

    try:
        cond = float(np.linalg.cond(Sigma_reg))
    except Exception:
        cond = np.nan

    out: Dict[str, Any] = {
        "stat": "Overall significance (Wald with bootstrap covariance)",
        "obs_value": T_obs,
        "p_value": p,
        "ridge": float(ridge),
        "lambda_used": lam,
        "use_pinv": bool(use_pinv),
        "Sigma_cond": cond,
        "x_label": "Bootstrap Wald statistic",
    }

    if store_mats:
        out["Sigma"] = Sigma
        out["Sigma_reg"] = Sigma_reg
        out["Sigma_inv"] = Sigma_inv

    if return_boot_dist:
        out["boot_draws"] = T_star

    return out


# ============================================================
# NEW dominance tests
# ============================================================

def _uniform_positive_dominance_test(
    g_b: np.ndarray,
    *,
    eps: float = 1e-12,
    return_boot_dist: bool = False,
) -> Dict[str, Any]:
    """
    Uniform positive dominance test (studentized):
        H0: min_tau g(tau) <= 0
        H1: min_tau g(tau) > 0

    Test statistic:
        t(tau)    = g_hat(tau) / se_hat(tau)
        T_U       = min_tau t(tau)          [observed]

    Bootstrap (centered, studentized):
        t*_b(tau) = (g_b(tau) - g_hat(tau)) / se_hat(tau)
        T*_U,b    = min_tau t*_b(tau)

    p = P*(T*_U,b >= T_U)

    Studentizing before taking the min means se differences across taus
    don't distort which tau is binding.  The centering g_b - g_hat imposes
    the boundary null g(tau)=0 for all tau (conservative but standard for
    composite inequality nulls).
    """
    g_b = np.asarray(g_b, float)           # (B, K)  K = number of taus
    g_hat = np.nanmean(g_b, axis=0)        # (K,)
    se_hat = np.nanstd(g_b, axis=0, ddof=1) + eps  # (K,)

    # Observed studentized path, then take min
    t_obs = g_hat / se_hat                 # (K,)
    T_obs = float(np.nanmin(t_obs))

    # Bootstrap studentized path (centered), then take min per draw
    t_star = (g_b - g_hat) / se_hat       # (B, K)
    T_star = np.nanmin(t_star, axis=1)    # (B,)
    T_star = T_star[np.isfinite(T_star)]

    out = {
        "stat": "Uniform positive dominance (min-t over tau, studentized)",
        "obs_value": T_obs,
        "x_label": "Bootstrap min_tau t*_b(tau)",
    }

    if T_star.size == 0 or not np.isfinite(T_obs):
        out["p_value"] = np.nan
        out["note"] = "No finite bootstrap draws."
        if return_boot_dist:
            out["boot_draws"] = T_star
        return out

    p = float((np.sum(T_star >= T_obs) + 1) / (T_star.size + 1))
    out["p_value"] = p

    if return_boot_dist:
        out["boot_draws"] = T_star
    return out


def _somewhere_positive_dominance_test(
    g_b: np.ndarray,
    *,
    eps: float = 1e-12,
    return_boot_dist: bool = False,
) -> Dict[str, Any]:
    """
    Average positive dominance test (studentized average over tau):
        H0: average positive dominance is not present
        H1: average dominance signal is positive

    Define
        g(tau) = |beta_cont(tau)| - |beta_lag(tau)|

    Studentized path:
        t(tau) = g_hat(tau) / se_hat(tau)

    Observed statistic:
        T_obs = mean_tau t(tau)

    Bootstrap statistic:
        t*_b(tau) = (g_b(tau) - g_hat(tau)) / se_hat(tau)
        T*_b      = mean_tau t*_b(tau)

    One-sided p-value:
        p = P*(T*_b >= T_obs)

    This is less conservative than max-t because it aggregates evidence
    across quantiles instead of focusing on the single most extreme tau.
    """
    g_b = np.asarray(g_b, float)
    g_hat = np.nanmean(g_b, axis=0)
    se_hat = np.nanstd(g_b, axis=0, ddof=1) + eps

    valid = np.isfinite(g_hat) & np.isfinite(se_hat)
    out = {
        "stat": "Average positive dominance (mean t over tau, studentized)",
        "x_label": "Bootstrap mean_tau t*_b(tau)",
    }

    if not np.any(valid):
        out["obs_value"] = np.nan
        out["p_value"] = np.nan
        out["note"] = "No finite taus."
        if return_boot_dist:
            out["boot_draws"] = np.array([])
        return out

    t_obs = g_hat[valid] / se_hat[valid]
    T_obs = float(np.nanmean(t_obs))

    t_star = (g_b[:, valid] - g_hat[valid].reshape(1, -1)) / se_hat[valid].reshape(1, -1)
    T_star = np.nanmean(t_star, axis=1)
    T_star = T_star[np.isfinite(T_star)]

    out["obs_value"] = T_obs

    if T_star.size == 0 or not np.isfinite(T_obs):
        out["p_value"] = np.nan
        out["note"] = "No finite bootstrap draws."
        if return_boot_dist:
            out["boot_draws"] = T_star
        return out

    out["p_value"] = float((np.sum(T_star >= T_obs) + 1) / (T_star.size + 1))

    if return_boot_dist:
        out["boot_draws"] = T_star
    return out


# ============================================================
# Tail extremeness vs middle
# ============================================================

def _tail_extremeness_vs_middle_centered_test(
    *,
    coef_b_full: np.ndarray,         # (B,K) bootstrap draws of beta(tau)
    beta_hat: np.ndarray,            # (K,) point estimate beta(tau)
    taus_sorted: List[float],
    segments: Dict[str, List[float]],
    middle_key: str = "mid",
    return_boot_dist: bool = False,
) -> Dict[str, Any]:
    """
    Tail extremeness vs middle, for one coefficient path beta(tau).

    Delta = mean_{tau in tail} |beta(tau)| - mean_{tau in mid} |beta(tau)|

    H0: Delta <= 0
    H1: Delta > 0

    Centered bootstrap:
      E_b = Delta_b - Delta_hat  ~ sampling error under boundary null Delta=0

    One-sided p-value (correct):
      p = P*( E_b >= Delta_hat )
    """

    taus = list(taus_sorted)
    tau_to_j = {t: j for j, t in enumerate(taus)}

    def _idx(seg_taus: List[float]) -> List[int]:
        return [tau_to_j[t] for t in seg_taus if t in tau_to_j]

    idx_mid = _idx(segments.get(middle_key, []))
    idx_low = _idx(segments.get("low", []))
    idx_high = _idx(segments.get("high", []))

    out: Dict[str, Any] = {
        "stat": "Tail extremeness vs middle (centered bootstrap, mean |beta|)",
        "middle_key": middle_key,
        "segments_def": {k: list(v) for k, v in segments.items()},
    }

    if len(idx_mid) == 0:
        out["error"] = f"middle segment '{middle_key}' is empty or missing."
        return out

    coef_b_full = np.asarray(coef_b_full, float)
    beta_hat = np.asarray(beta_hat, float).reshape(-1)

    if coef_b_full.ndim != 2:
        raise ValueError("coef_b_full must be 2D (B,K)")

    B, K = coef_b_full.shape
    if beta_hat.shape[0] != K:
        raise ValueError(f"beta_hat must have length K={K}, got {beta_hat.shape[0]}")

    abs_b = np.abs(coef_b_full)
    abs_hat = np.abs(beta_hat)

    mid_mean_b = np.nanmean(abs_b[:, idx_mid], axis=1)
    mid_mean_hat = float(np.nanmean(abs_hat[idx_mid]))

    def _pack(idx_tail: List[int], label: str) -> Dict[str, Any]:
        if len(idx_tail) == 0:
            return {"p_value": np.nan, "note": "tail segment empty"}

        tail_mean_hat = float(np.nanmean(abs_hat[idx_tail]))
        Delta_hat = float(tail_mean_hat - mid_mean_hat)

        tail_mean_b = np.nanmean(abs_b[:, idx_tail], axis=1)
        Delta_b = tail_mean_b - mid_mean_b

        E_b = (Delta_b - Delta_hat)
        E_b = E_b[np.isfinite(E_b)]

        if E_b.size == 0 or (not np.isfinite(Delta_hat)):
            return {"p_value": np.nan, "obs_value": Delta_hat, "note": "no finite draws"}

        p = float((np.sum(E_b >= Delta_hat) + 1) / (E_b.size + 1))

        pack: Dict[str, Any] = {
            "stat": label,
            "obs_value": Delta_hat,
            "p_value": p,
            "x_label": "Centered bootstrap error (Delta* - Delta_obs)",
        }
        if return_boot_dist:
            pack["boot_draws"] = E_b
        return pack

    out["high_gt_mid"] = _pack(idx_high, "High tail > Middle (mean |beta|)")
    out["low_gt_mid"]  = _pack(idx_low,  "Low tail > Middle (mean |beta|)")
    return out


# ============================================================
# Plotting utilities
# ============================================================

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_stem(s: str) -> str:
    """Safe path stem: letters/numbers/- only (replace everything else with '-')."""
    out = []
    for ch in str(s):
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "."):
            out.append("-" if ch == " " else ch)
        else:
            out.append("-")
    stem = "".join(out)
    while "--" in stem:
        stem = stem.replace("--", "-")
    return stem.strip("-").lower()


def _pretty(s: str) -> str:
    """Make a journal-ish label (no underscores; basic cleanup)."""
    s = str(s).replace("_", " ")
    s = " ".join(s.split())
    return s


def _hist_panel_grid(
    *,
    panels: List[Dict[str, Any]],
    title: str,
    bins: int,
    dpi: int,
    savepath: Union[str, Path],
    max_cols: int = 4,
    obs_line_color: str = "red",
    obs_line_width: float = 3.0,
    obs_line_style: str = "--",
    obs_text_fontsize: int = 10,
) -> None:
    """
    General histogram panel grid.
    Each panel dict keys:
      - name: str
      - draws: array-like
      - obs: float or None
      - p_value: float or None
      - xlabel: str
    """
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)

    use = []
    for p in panels:
        x = np.asarray(p.get("draws", []), float)
        x = x[np.isfinite(x)]
        if x.size >= 5:
            use.append({**p, "draws": x})
    if len(use) == 0:
        return

    k = len(use)
    cols = min(max_cols, k)
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.6 * rows), squeeze=False)
    axes_flat = axes.ravel()

    allx = np.concatenate([p["draws"] for p in use])
    xlo, xhi = float(np.nanmin(allx)), float(np.nanmax(allx))
    if np.isfinite(xlo) and np.isfinite(xhi) and xhi > xlo:
        pad = 0.04 * (xhi - xlo)
        xlo -= pad
        xhi += pad
    else:
        xlo, xhi = None, None

    for ax, p in zip(axes_flat, use):
        ax.hist(p["draws"], bins=bins)

        obs = p.get("obs", None)
        pv = p.get("p_value", None)

        if obs is not None and np.isfinite(obs):
            ax.axvline(
                float(obs),
                linewidth=obs_line_width,
                linestyle=obs_line_style,
                color=obs_line_color,
                zorder=10,
            )

            if pv is not None and np.isfinite(pv):
                txt = f"Obs = {float(obs):.3f}\np = {float(pv):.3f}"
            else:
                txt = f"Obs = {float(obs):.3f}"

            ax.text(
                0.98, 0.95, txt,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=obs_text_fontsize,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, linewidth=0),
            )

        ax.set_title(_pretty(p.get("name", "")))
        ax.set_xlabel(_pretty(p.get("xlabel", "Bootstrap statistic")))
        ax.set_ylabel("Frequency")
        if xlo is not None:
            ax.set_xlim(xlo, xhi)

    for j in range(len(use), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(_pretty(title), y=1.02)
    fig.tight_layout()
    fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_distributions_from_tests(
    tests: Dict[str, Any],
    *,
    base_dir: Union[str, Path],
    prefix: str,
    horizon_label: str,
    equation_name: str,
    dpi: int = 200,
    bins: int = 40,
) -> List[Path]:
    """
    Creates histogram figures.

    - For tests with global+segments: produces up to 4 panels (Global|Low|Mid|High).
    - For tail-extremeness (test 6): produces a small grid of available contrasts
      (High>Mid, Low>Mid) per regressor (ret_pos, ret_neg).
    """
    base_dir = Path(base_dir)
    saved: List[Path] = []

    STAT_KEYS = [
        "somewhere",
        "overall_l2",
        "overall_wald",
        "everywhere",
        "uniform_positive_dominance",
        "somewhere_positive_dominance",
    ]

    def _seg_nodes(v: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        out: List[Tuple[str, Dict[str, Any]]] = []
        if isinstance(v.get("global", None), dict):
            out.append(("Global", v["global"]))
        segs = v.get("segments", {})
        if isinstance(segs, dict):
            for seg_name in ["low", "mid", "high"]:
                node = segs.get(seg_name, None)
                if isinstance(node, dict):
                    out.append((seg_name.capitalize(), node))
        return out

    def _extract_panel(node: Dict[str, Any], stat_key: str, panel_name: str) -> Optional[Dict[str, Any]]:
        d = node.get(stat_key, None)
        if not isinstance(d, dict):
            return None
        draws = d.get("boot_draws", None)
        if draws is None:
            return None
        return {
            "name": panel_name,
            "draws": np.asarray(draws, float),
            "obs": d.get("obs_value", None),
            "p_value": d.get("p_value", None),
            "xlabel": d.get("x_label", "Bootstrap statistic"),
            "stat_title": d.get("stat", stat_key),
        }

    def _plot_group(group_key: str, v: Dict[str, Any]) -> None:
        segs = _seg_nodes(v)
        if len(segs) == 0:
            return

        for stat_key in STAT_KEYS:
            panels: List[Dict[str, Any]] = []
            stat_title = None
            for seg_label, node in segs:
                p = _extract_panel(node, stat_key, seg_label)
                if p is None:
                    continue
                stat_title = stat_title or p["stat_title"]
                panels.append(p)

            if len(panels) == 0:
                continue

            title = " | ".join([_pretty(prefix), _pretty(horizon_label), _pretty(equation_name), _pretty(group_key), _pretty(stat_title)])
            out_dir = _ensure_dir(base_dir / _safe_stem(prefix) / _safe_stem(horizon_label) / _safe_stem(equation_name) / _safe_stem(group_key))
            savepath = out_dir / (_safe_stem(stat_title) + ".png")

            _hist_panel_grid(panels=panels, title=title, bins=bins, dpi=dpi, savepath=savepath, max_cols=4)
            saved.append(savepath)

    def _plot_tail_extremeness(test6_key: str, test6_val: Dict[str, Any]) -> None:
        for coef_name, pack in test6_val.items():
            if not isinstance(pack, dict):
                continue

            panels: List[Dict[str, Any]] = []
            for side_key in ["high_gt_mid", "low_gt_mid"]:
                d = pack.get(side_key, None)
                if not isinstance(d, dict):
                    continue
                draws = d.get("boot_draws", None)
                if draws is None:
                    continue
                panels.append({
                    "name": d.get("stat", side_key),
                    "draws": np.asarray(draws, float),
                    "obs": d.get("obs_value", None),
                    "p_value": d.get("p_value", None),
                    "xlabel": d.get("x_label", "Bootstrap statistic"),
                })

            if len(panels) == 0:
                continue

            group_key = f"{test6_key}: {coef_name}"
            title = " | ".join([_pretty(prefix), _pretty(horizon_label), _pretty(equation_name), _pretty(group_key)])

            out_dir = _ensure_dir(base_dir / _safe_stem(prefix) / _safe_stem(horizon_label) / _safe_stem(equation_name) / _safe_stem(group_key))
            savepath = out_dir / "tail-extremeness-diffs.png"

            _hist_panel_grid(panels=panels, title=title, bins=bins, dpi=dpi, savepath=savepath, max_cols=4)
            saved.append(savepath)

    for k, v in tests.items():
        if k == "meta":
            continue
        if not isinstance(v, dict):
            continue

        if "(6)" in str(k):
            _plot_tail_extremeness(k, v)
            continue

        if "global" in v:
            _plot_group(k, v)
        else:
            for kk, vv in v.items():
                if isinstance(vv, dict) and "global" in vv:
                    _plot_group(f"{k}: {kk}", vv)

    return saved


# ============================================================
# Compute tests for an equation
# ============================================================

def _compute_tests_for_equation(
    *,
    dep: str,
    X_cols: List[str],
    taus: Tuple[float, ...],
    params: pd.DataFrame,  # used for beta_hat (point estimates)
    boot_params: Dict[float, np.ndarray],
    segments: Optional[Dict[str, List[float]]] = None,
    alpha: float = 0.05,
    return_test_boot_dist: bool = False,
    store_wald_mats: bool = False,
) -> dict:
    taus_sorted = sorted(list(map(float, taus)))
    if segments is None:
        segments = _split_taus_into_thirds(taus_sorted)

    col_to_i = {c: i for i, c in enumerate(X_cols)}

    def _have(c: str) -> bool:
        return c in col_to_i

    def _coef_b(col: str, taus_subset: List[float]) -> np.ndarray:
        i = col_to_i[col]
        mats = []
        for t in taus_subset:
            b = np.asarray(boot_params[t], float)[:, i]
            mats.append(b.reshape(-1, 1))
        out = np.concatenate(mats, axis=1)
        B_min = min(m.shape[0] for m in mats)
        if out.shape[0] != B_min:
            out = out[:B_min, :]
        return out  # (B,Ksub)

    def _beta_hat_path(col: str, taus_subset: List[float]) -> np.ndarray:
        return np.asarray([float(params.loc[float(t), col]) for t in taus_subset], float)

    def _joint_suite(g_b: np.ndarray) -> Dict[str, Any]:
        return {
            "somewhere": _maxabs_t_omnibus_pvalue(g_b, return_boot_dist=return_test_boot_dist),
            "overall_l2": _l2_t2_omnibus_pvalue(g_b, return_boot_dist=return_test_boot_dist),
            "overall_wald": _wald_omnibus_pvalue(
                g_b,
                ridge=1e-8,
                use_pinv=True,
                return_boot_dist=return_test_boot_dist,
                store_mats=store_wald_mats,
            ),
            "everywhere": _joint_significance_everywhere(
                g_b,
                alpha=alpha,
                return_boot_dist=return_test_boot_dist,
            ),
        }

    def _apply_joint_global_and_segments_for_col(col: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["global"] = _joint_suite(_coef_b(col, taus_sorted))

        seg_out: Dict[str, Any] = {}
        for nm, seg_taus in segments.items():
            if not seg_taus:
                seg_out[nm] = {"note": "empty segment"}
                continue
            seg_out[nm] = _joint_suite(_coef_b(col, seg_taus))
        out["segments"] = seg_out
        out["segments_def"] = {k: list(v) for k, v in segments.items()}
        return out

    def _apply_joint_global_and_segments_from_builder(builder) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["global"] = _joint_suite(builder(taus_sorted))

        seg_out: Dict[str, Any] = {}
        for nm, seg_taus in segments.items():
            if not seg_taus:
                seg_out[nm] = {"note": "empty segment"}
                continue
            seg_out[nm] = _joint_suite(builder(seg_taus))

        out["segments"] = seg_out
        out["segments_def"] = {k: list(v) for k, v in segments.items()}
        return out

    def _sorted_lag_cols(prefix: str) -> List[str]:
        cols = [c for c in X_cols if c.startswith(prefix)]

        def _key(c: str) -> int:
            try:
                return int(c.split("_L")[-1])
            except Exception:
                return 999

        return sorted(cols, key=_key)

    tests: Dict[str, Any] = {
        "meta": {
            "dep": dep,
            "taus": taus_sorted,
            "alpha": float(alpha),
            "segments_def": {k: list(v) for k, v in segments.items()},
            "return_test_boot_dist": bool(return_test_boot_dist),
            "store_wald_mats": bool(store_wald_mats),
        }
    }

    # (1) Contemporaneous ret_pos / ret_neg
    if _have("ret_pos"):
        tests["(1) Contemporaneous positive returns"] = _apply_joint_global_and_segments_for_col("ret_pos")
    if _have("ret_neg"):
        tests["(1) Contemporaneous negative returns"] = _apply_joint_global_and_segments_for_col("ret_neg")

    # (2) Lagged returns (each lag)
    lag_tests: Dict[str, Any] = {}
    for c in _sorted_lag_cols("ret_pos_L"):
        lag_tests[f"Positive return lag {c.split('_L')[-1]}"] = _apply_joint_global_and_segments_for_col(c)
    for c in _sorted_lag_cols("ret_neg_L"):
        lag_tests[f"Negative return lag {c.split('_L')[-1]}"] = _apply_joint_global_and_segments_for_col(c)
    tests["(2) Lagged returns"] = lag_tests

    # (3) Dominance: ONLY uniform + somewhere positive dominance
    def _absdiff_builder(col_now: str, col_l1: str):
        if (not _have(col_now)) or (not _have(col_l1)):
            return None
        i_now, i_l1 = col_to_i[col_now], col_to_i[col_l1]

        def _builder(taus_subset: List[float]) -> np.ndarray:
            mats = []
            for t in taus_subset:
                b = np.asarray(boot_params[t], float)  # (B,p)
                d = np.abs(b[:, i_now]) - np.abs(b[:, i_l1])  # (B,)
                mats.append(d.reshape(-1, 1))
            out = np.concatenate(mats, axis=1)  # (B,Ksub)
            B_min = min(m.shape[0] for m in mats)
            if out.shape[0] != B_min:
                out = out[:B_min, :]
            return out

        return _builder

    def _apply_dominance(builder) -> Dict[str, Any]:
        """
        For g(tau) = |beta_now(tau)| - |beta_lag1(tau)|, compute only:

        1) Uniform positive dominance:
           H0: min_tau g(tau) <= 0
           H1: min_tau g(tau) > 0

        2) Somewhere positive dominance:
           H0: max_tau g(tau) <= 0
           H1: max_tau g(tau) > 0
        """
        out: Dict[str, Any] = {}

        g_all = builder(taus_sorted)
        out["global"] = {
            "uniform_positive_dominance": _uniform_positive_dominance_test(
                g_all,
                return_boot_dist=return_test_boot_dist,
            ),
            "somewhere_positive_dominance": _somewhere_positive_dominance_test(
                g_all,
                return_boot_dist=return_test_boot_dist,
            ),
        }

        seg_out: Dict[str, Any] = {}
        for nm, seg_taus in segments.items():
            if not seg_taus:
                seg_out[nm] = {"note": "empty segment"}
                continue

            g_seg = builder(seg_taus)
            seg_out[nm] = {
                "uniform_positive_dominance": _uniform_positive_dominance_test(
                    g_seg,
                    return_boot_dist=return_test_boot_dist,
                ),
                "somewhere_positive_dominance": _somewhere_positive_dominance_test(
                    g_seg,
                    return_boot_dist=return_test_boot_dist,
                ),
            }

        out["segments"] = seg_out
        out["segments_def"] = {k: list(v) for k, v in segments.items()}
        return out

    b_pos = _absdiff_builder("ret_pos", "ret_pos_L1")
    if b_pos is not None:
        tests["(3) Dominance: |positive return| > |lag 1 positive return|"] = _apply_dominance(b_pos)

    b_neg = _absdiff_builder("ret_neg", "ret_neg_L1")
    if b_neg is not None:
        tests["(3) Dominance: |negative return| > |lag 1 negative return|"] = _apply_dominance(b_neg)

    # (4) dep_L1
    dep_l1 = f"{dep}_L1"
    if _have(dep_l1):
        tests["(4) Lag 1 dependent variable"] = _apply_joint_global_and_segments_for_col(dep_l1)

    # (5) Asymmetry: |ret_pos| != |ret_neg|
    if _have("ret_pos") and _have("ret_neg"):
        i_p, i_n = col_to_i["ret_pos"], col_to_i["ret_neg"]

        def _abs_pos_minus_abs_neg_builder(taus_subset: List[float]) -> np.ndarray:
            mats = []
            for t in taus_subset:
                b = np.asarray(boot_params[t], float)
                d = np.abs(b[:, i_p]) - np.abs(b[:, i_n])
                mats.append(d.reshape(-1, 1))
            out = np.concatenate(mats, axis=1)
            B_min = min(m.shape[0] for m in mats)
            if out.shape[0] != B_min:
                out = out[:B_min, :]
            return out

        tests["(5) Asymmetry: |positive return| differs from |negative return|"] = _apply_joint_global_and_segments_from_builder(
            _abs_pos_minus_abs_neg_builder
        )

    # (6) Tail extremeness vs middle
    tail_ext: Dict[str, Any] = {}
    for col in ["ret_pos", "ret_neg"]:
        if not _have(col):
            continue
        coef_full = _coef_b(col, taus_sorted)
        beta_hat = _beta_hat_path(col, taus_sorted)
        tail_ext[_pretty(col)] = _tail_extremeness_vs_middle_centered_test(
            coef_b_full=coef_full,
            beta_hat=beta_hat,
            taus_sorted=taus_sorted,
            segments=segments,
            middle_key="mid",
            return_boot_dist=return_test_boot_dist,
        )
    tests["(6) Tail extremeness relative to middle (magnitude)"] = tail_ext

    return tests


# ============================================================
# Main regression runner
# ============================================================

def run_asym_quantreg_with_controls(
    *,
    r_df: pd.DataFrame,
    var_s: pd.Series,
    skew_s: pd.Series,
    kurt_s: pd.Series,
    controls_df: Optional[pd.DataFrame] = None,
    controls_cols: Union[str, Sequence[str]] = "all",
    controls_diff: bool = False,
    n_controls_lags: int = 0,
    date_col: str = "date",
    ret_col: str = "ret_30",
    horizon_label: str = "30d",
    n_ret_lags: int = 2,
    n_mom_lags: int = 2,
    taus: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    add_const: bool = True,
    bootstrap: str = "block",
    B: int = 1000,
    block_len: int = 10,
    seed: Optional[int] = 123,
    fit_kwargs: Optional[dict] = None,
    dropna: bool = True,
    X_A_extra: Optional[Sequence[str]] = None,
    X_B_extra: Optional[Sequence[str]] = None,
    X_C_extra: Optional[Sequence[str]] = None,
    nw_lags: Optional[int] = None,
    mom_cross_contemporaneous: bool = True,
    segments: Optional[Dict[str, List[float]]] = None,
    alpha_tests: float = 0.05,
    equations_to_run: Optional[Sequence[str]] = None,
    store_boot_params: bool = True,
    store_test_boot_dist: bool = False,
    store_wald_mats: bool = False,
    plot_test_dist: bool = False,
    plot_test_dist_dir: Optional[Union[str, Path]] = None,
    plot_test_dist_prefix: Optional[str] = None,
    plot_bins: int = 40,
    plot_dpi: int = 200,
) -> dict:
    """
    Quantile regressions with aligned bootstrap draws across taus, plus joint tests and diagnostics.

    Plotting rule:
      If plot_test_dist=True, we auto-enable store_test_boot_dist=True.
    """
    if plot_test_dist and (not store_test_boot_dist):
        store_test_boot_dist = True

    r = r_df[[date_col, ret_col]].copy()
    r[date_col] = pd.to_datetime(r[date_col])

    def _series_to_df(s: pd.Series, name: str) -> pd.DataFrame:
        ss = s.copy()
        ss.name = name
        if not isinstance(ss.index, pd.DatetimeIndex):
            ss.index = pd.to_datetime(ss.index)
        return ss.to_frame().reset_index().rename(columns={"index": date_col})

    var_df = _series_to_df(var_s, "var")
    skew_df = _series_to_df(skew_s, "skew")
    kurt_df = _series_to_df(kurt_s, "kurt")

    df = (
        r.merge(var_df, on=date_col, how="inner")
         .merge(skew_df, on=date_col, how="inner")
         .merge(kurt_df, on=date_col, how="inner")
         .sort_values(date_col)
         .reset_index(drop=True)
    )

    control_names: List[str] = []
    if controls_df is not None:
        cdf = controls_df.copy()
        cdf[date_col] = pd.to_datetime(cdf[date_col])

        if controls_cols == "all":
            control_names = [c for c in cdf.columns if c != date_col]
        else:
            control_names = list(controls_cols)

        keep = [date_col] + control_names
        df = df.merge(cdf[keep], on=date_col, how="left")

    df["d_var"] = df["var"].diff()
    df["d_skew"] = df["skew"].diff()
    df["d_kurt"] = df["kurt"].diff()

    df["ret_pos"] = df[ret_col].clip(lower=0.0)
    df["ret_neg"] = df[ret_col].clip(upper=0.0)

    for L in range(1, n_ret_lags + 1):
        df[f"ret_pos_L{L}"] = df["ret_pos"].shift(L)
        df[f"ret_neg_L{L}"] = df["ret_neg"].shift(L)

    for L in range(1, n_mom_lags + 1):
        df[f"d_var_L{L}"] = df["d_var"].shift(L)
        df[f"d_skew_L{L}"] = df["d_skew"].shift(L)
        df[f"d_kurt_L{L}"] = df["d_kurt"].shift(L)

    controls_terms: List[str] = []
    if control_names:
        if controls_diff:
            for c in control_names:
                dc = f"d_{c}"
                df[dc] = df[c].astype(float).diff()
                controls_terms.append(dc)
                for L in range(1, n_controls_lags + 1):
                    nm = f"{dc}_L{L}"
                    df[nm] = df[dc].shift(L)
                    controls_terms.append(nm)
        else:
            for c in control_names:
                controls_terms.append(c)
                for L in range(1, n_controls_lags + 1):
                    nm = f"{c}_L{L}"
                    df[nm] = df[c].shift(L)
                    controls_terms.append(nm)

    df_model = df.dropna().copy() if dropna else df.copy()

    ret_terms = (
        ["ret_pos", "ret_neg"]
        + [f"ret_pos_L{L}" for L in range(1, n_ret_lags + 1)]
        + [f"ret_neg_L{L}" for L in range(1, n_ret_lags + 1)]
    )

    def _mom_terms_for_eq(dep_mom: str) -> List[str]:
        moms = ["d_var", "d_skew", "d_kurt"]
        terms: List[str] = []
        for m in moms:
            if m == dep_mom:
                terms += [f"{m}_L{L}" for L in range(1, n_mom_lags + 1)]
            else:
                if mom_cross_contemporaneous:
                    terms.append(m)
                terms += [f"{m}_L{L}" for L in range(1, n_mom_lags + 1)]
        return terms

    def _run_eq(dep: str, X_cols: List[str], equation_name: str) -> dict:
        y = df_model[dep].to_numpy(float)
        Xdf = df_model[X_cols].astype(float)

        if add_const:
            Xdf = sm.add_constant(Xdf, has_constant="add")

        Xmat = Xdf.to_numpy(float)
        colnames = list(Xdf.columns)
        n_used = int(len(df_model))
        taus_t = tuple(float(t) for t in taus)

        params_dict, se_dict, results, boot_params, boot_meta = _quantreg_fit_bootstrap(
            y=y,
            X=Xmat,
            taus=taus_t,
            B=B,
            bootstrap=bootstrap,
            block_len=block_len,
            seed=seed,
            fit_kwargs=fit_kwargs,
            verbose=True,
            progress_every=25,
        )

        params = pd.DataFrame([params_dict[t] for t in taus_t], index=taus_t, columns=colnames)
        se_boot = pd.DataFrame([se_dict[t] for t in taus_t], index=taus_t, columns=colnames)
        t_boot = params / se_boot
        p_boot_norm = t_boot.applymap(_pval_from_t_normal)

        q_pseudo_r2: Dict[float, float] = {}
        q_prsquared: Dict[float, float] = {}
        for tau in taus_t:
            beta = params_dict[tau]
            q_pseudo_r2[tau] = _quantile_pseudo_r2(y=y, X=Xmat, beta=beta, tau=tau)
            res_tau = results.get(tau, None)
            q_prsquared[tau] = getattr(res_tau, "prsquared", np.nan) if res_tau is not None else np.nan

        q_pseudo_r2_s = pd.Series(q_pseudo_r2, name="pseudo_r2")
        q_prsquared_s = pd.Series(q_prsquared, name="prsquared_sm")

        maxlags = _newey_west_default_lags(n_used) if nw_lags is None else int(nw_lags)
        ols_res = sm.OLS(y, Xmat).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags, "use_correction": True},
        )

        ols_pack = {
            "params": pd.Series(ols_res.params, index=colnames, name="params"),
            "se_hac": pd.Series(ols_res.bse, index=colnames, name="se_hac"),
            "t_hac": pd.Series(ols_res.tvalues, index=colnames, name="t_hac"),
            "p_hac": pd.Series(ols_res.pvalues, index=colnames, name="p_hac"),
            "r2": float(getattr(ols_res, "rsquared", np.nan)),
            "adj_r2": float(getattr(ols_res, "rsquared_adj", np.nan)),
            "nw_lags": int(maxlags),
            "result": ols_res,
        }

        out_eq: Dict[str, Any] = {
            "dep": dep,
            "X_cols": colnames,
            "params": params,
            "se_boot": se_boot,
            "t_boot": t_boot,
            "p_boot_norm": p_boot_norm,
            "q_pseudo_r2": q_pseudo_r2_s,
            "q_prsquared_sm": q_prsquared_s,
            "results": results,
            "ols": ols_pack,
            "bootstrap_diagnostics": boot_meta,
            "meta": {
                "taus": taus_t,
                "bootstrap": bootstrap,
                "B": int(B),
                "block_len": int(block_len),
                "n_used": n_used,
                "horizon_label": horizon_label,
                "n_ret_lags": int(n_ret_lags),
                "n_mom_lags": int(n_mom_lags),
                "controls": control_names,
                "controls_diff": bool(controls_diff),
                "n_controls_lags": int(n_controls_lags),
                "ret_col": ret_col,
                "nw_lags": int(maxlags),
                "mom_cross_contemporaneous": bool(mom_cross_contemporaneous),
            },
        }

        if store_boot_params:
            out_eq["boot_params"] = boot_params

        out_eq["tests"] = _compute_tests_for_equation(
            dep=dep,
            X_cols=colnames,
            taus=taus_t,
            params=params,
            boot_params=boot_params,
            segments=segments,
            alpha=alpha_tests,
            return_test_boot_dist=store_test_boot_dist,
            store_wald_mats=store_wald_mats,
        )

        if plot_test_dist:
            outdir = plot_test_dist_dir
            if outdir is None:
                outdir = Path.cwd() / "test_sampling_dists"

            pref = plot_test_dist_prefix or "Results"
            saved = plot_sampling_distributions_from_tests(
                out_eq["tests"],
                base_dir=outdir,
                prefix=pref,
                horizon_label=horizon_label,
                equation_name=equation_name,
                dpi=plot_dpi,
                bins=plot_bins,
            )
            out_eq["test_dist_plots"] = [str(p) for p in saved]

        return out_eq

    X_A = ret_terms + _mom_terms_for_eq("d_var") + controls_terms + (list(X_A_extra) if X_A_extra else [])
    X_B = ret_terms + _mom_terms_for_eq("d_skew") + controls_terms + (list(X_B_extra) if X_B_extra else [])
    X_C = ret_terms + _mom_terms_for_eq("d_kurt") + controls_terms + (list(X_C_extra) if X_C_extra else [])

    eqs = list(equations_to_run) if equations_to_run is not None else ["A_var", "B_skew", "C_kurt"]
    bad = sorted(set(eqs) - {"A_var", "B_skew", "C_kurt"})
    if bad:
        raise ValueError(f"Unknown equations_to_run entries: {bad}")

    out = {
        "data": df_model,
        "controls_terms": controls_terms,
    }

    if "A_var" in eqs:
        out["A_var"] = _run_eq("d_var", X_A, equation_name="Delta variance")
    if "B_skew" in eqs:
        out["B_skew"] = _run_eq("d_skew", X_B, equation_name="Delta skewness")
    if "C_kurt" in eqs:
        out["C_kurt"] = _run_eq("d_kurt", X_C, equation_name="Delta kurtosis")

    return out

# ============================================================
# Example call
# ============================================================
# out = run_asym_quantreg_with_controls(
#     r_df=logreturns,
#     var_s=moments_premia_dict[h]["phys_vol_ann"],
#     skew_s=moments_premia_dict[h]["phys_skew"],
#     kurt_s=moments_premia_dict[h]["phys_kurt"],
#     ret_col="ret_1",
#     horizon_label="30d",
#     n_controls_lags=2,
#     n_ret_lags=2,
#     n_mom_lags=2,
#     B=500,
#     block_len=25,
#     store_boot_params=True,
#     store_wald_mats=False,
#     plot_test_dist=True,
#     plot_test_dist_dir="C:/tmp/test_dists",
#     plot_test_dist_prefix="USO",
# )
# ============================================================

# ============================================================
# Example call (for your loop)
# ============================================================
#
# out = run_asym_quantreg_with_controls(
#     r_df=logreturns,
#     var_s=moments_premia_dict[h]["phys_vol_ann"],
#     skew_s=moments_premia_dict[h]["phys_skew"],
#     kurt_s=moments_premia_dict[h]["phys_kurt"],
#     ret_col="ret_1",
#     horizon_label="30d",
#     n_controls_lags=2,
#     n_ret_lags=2,
#     n_mom_lags=2,
#     B=500,
#     block_len=25,
#     store_boot_params=True,
#     store_wald_mats=False,        # True can be huge
#     plot_test_dist=True,
#     plot_test_dist_dir="C:/tmp/test_dists",
#     plot_test_dist_prefix="USO",
# )
# ============================================================

# ============================================================
# Usage reminder (your call)
#   - remove bh_q from your call (no BH)
#   - plotting now works even if you forget store_test_boot_dist
# ============================================================
#
# out = run_asym_quantreg_with_controls(
#     r_df=logreturns,
#     var_s=moments_premia_dict[h]["phys_vol_ann"],
#     skew_s=moments_premia_dict[h]["phys_skew"],
#     kurt_s=moments_premia_dict[h]["phys_kurt"],
#     ret_col="ret_1",
#     n_controls_lags=2,
#     n_ret_lags=2,
#     n_mom_lags=2,
#     B=50,
#     block_len=25,
#     store_boot_params=True,
#     store_wald_mats=True,
#     plot_test_dist=True,
#     plot_test_dist_dir="C:/tmp/test_dists",
#     plot_test_dist_prefix="USO",
# )
# ============================================================

# ============================================================
# Example usage:
# out = run_asym_quantreg_with_controls(
#     r_df=logreturns,
#     var_s=moments_premia_dict[h]["rnd_var"],  # or whatever you pass
#     skew_s=moments_premia_dict[h]["rnd_skew"],
#     kurt_s=moments_premia_dict[h]["rnd_kurt"],
#     ret_col="ret_30",
#     horizon_label="30d",
#     B=600,
#     block_len=25,
#     store_boot_params=True,
#     store_test_boot_dist=True,   # required if you want plots
#     store_wald_mats=True,        # store Sigma/Sigma_inv in the Wald results
#     plot_test_dist=True,
#     plot_test_dist_dir="C:/tmp/test_dists",
#     plot_test_dist_prefix="BTC",
# )
# ============================================================

# ============================================================
# LaTeX table builder for hypothesis testing saver (all horizons/taus by default)
# ============================================================

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def asym_tests_to_latex_bundle(
    asym_results: Union[dict, Dict[str, dict], List[dict]],
    *,
    out_dir: str = "latex_tests",
    file_prefix: str = "tests",
    bh_q: Optional[float] = None,
    equations: Tuple[str, ...] = ("A_var", "B_skew", "C_kurt"),
    equation_titles: Optional[Dict[str, str]] = None,
    table_env: str = "table",
    size_cmd: str = r"\small",
    add_notes: bool = True,
    tail_reject_symbol: str = "T",
    tail_fail_symbol: str = "F",
) -> Dict[str, Any]:
    """
    ONE compact academic-style LaTeX table combining:
      - BH rejection counts across quantiles (cells show #rej/#taus)
      - Tail-vs-median magnitude tests as 4 additional columns with:
            R = reject, F = fail to reject (BH-adjusted within the 4 tail tests)

    Arranged by moment then horizon:
      V|7, V|14, ..., S|7, ..., K|60.

    Saves:
      - <file_prefix>_main.tex
      - <file_prefix>_master.tex
    """

    equation_titles = equation_titles or {"A_var": "V", "B_skew": "S", "C_kurt": "K"}

    # -----------------------
    # Input normalization
    # -----------------------
    def _is_single_output(d: dict) -> bool:
        return isinstance(d, dict) and all(k in d for k in ["A_var", "B_skew", "C_kurt"]) and "data" in d

    def _get_horizon_label(out: dict) -> Optional[str]:
        try:
            return out["A_var"]["meta"].get("horizon_label", None)
        except Exception:
            return None

    def _normalize_inputs(inp) -> List[Tuple[str, dict]]:
        if isinstance(inp, list):
            outs = []
            for j, d in enumerate(inp):
                if not _is_single_output(d):
                    raise ValueError(f"List element {j} does not look like a regression output dict.")
                h = _get_horizon_label(d) or f"h{j+1}"
                outs.append((h, d))
            return outs

        if isinstance(inp, dict) and _is_single_output(inp):
            h = _get_horizon_label(inp) or "horizon"
            return [(h, inp)]

        if isinstance(inp, dict):
            outs = []
            for k, d in inp.items():
                if not _is_single_output(d):
                    raise ValueError(f"Entry '{k}' does not look like a regression output dict.")
                outs.append((str(k), d))
            return outs

        raise ValueError("Unsupported asym_results input type.")

    def _get_tests_block(eq_out: dict) -> dict:
        return eq_out.get("tests", {}) or {}

    def _bhq_from(eq_out: dict) -> float:
        if bh_q is not None:
            return float(bh_q)
        tests = _get_tests_block(eq_out)
        if "bh_q" in tests:
            return float(tests["bh_q"])
        return float(eq_out.get("meta", {}).get("bh_q", 0.05))

    def _count_reject(rej: pd.Series) -> Tuple[int, int]:
        if rej is None or len(rej) == 0:
            return (0, 0)
        return (int(np.sum(rej.astype(bool))), int(len(rej)))

    def _parse_horizon_num(h: str) -> float:
        """
        Extract a numeric horizon from keys like '7d', '30d', '30', '30d (etc)'.
        Falls back to NaN -> sorts last within moment.
        """
        if h is None:
            return float("nan")
        s = str(h)
        m = re.search(r"(\d+(\.\d+)?)", s)
        if m:
            return float(m.group(1))
        return float("nan")

    outs = _normalize_inputs(asym_results)

    # -----------------------
    # Build compact summary rows
    # -----------------------
    rows: List[dict] = []

    for horizon_label, out in outs:
        for eq_key in equations:
            if eq_key not in out:
                continue

            eq_out = out[eq_key]
            tests = _get_tests_block(eq_out)
            if not tests:
                continue

            bq = _bhq_from(eq_out)
            mom_short = equation_titles.get(eq_key, eq_key)  # V/S/K

            # ---- contemporaneous across taus (BH) ----
            sig_contemp = tests.get("sig_contemp_across_quantiles", None)

            def _summ_from_sigdf(df: Optional[pd.DataFrame], col: str) -> str:
                if df is None or df.empty:
                    return ""
                rj = df.get(f"reject_bh_{col}", None)
                if rj is None:
                    return ""
                nrej, m = _count_reject(rj)
                return f"{nrej}/{m}"

            Cpos = _summ_from_sigdf(sig_contemp, "ret_pos")
            Cneg = _summ_from_sigdf(sig_contemp, "ret_neg")

            # ---- lagged returns across taus (BH), use L1 if present ----
            lag_dict = tests.get("sig_lagged_ret_across_quantiles", {}) or {}

            def _summ_lag(name: str) -> str:
                df = lag_dict.get(name, None)
                if df is None or df.empty:
                    return ""
                rj = df["reject_bh"]
                nrej, m = _count_reject(rj)
                return f"{nrej}/{m}"

            Lpos1 = _summ_lag("ret_pos_L1")
            Lneg1 = _summ_lag("ret_neg_L1")

            # ---- dep lag1 across taus (BH) ----
            dep_l1 = tests.get("sig_dep_lag1_across_quantiles", None)
            if isinstance(dep_l1, pd.DataFrame) and not dep_l1.empty:
                rj = dep_l1["reject_bh"]
                nrej, m = _count_reject(rj)
                Dep1 = f"{nrej}/{m}"
            else:
                Dep1 = ""

            # ---- abs(contemp) > abs(lag1) contrasts (BH across taus) ----
            cvl = tests.get("contemp_vs_lag1", {}) or {}

            def _summ_contrast(df: Optional[pd.DataFrame]) -> str:
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    return ""
                if not isinstance(df, pd.DataFrame) or "reject_bh" not in df.columns:
                    return ""
                rj = df["reject_bh"]
                nrej, m = _count_reject(rj)
                return f"{nrej}/{m}"

            AbsCpos = _summ_contrast(cvl.get("abs_ret_pos_gt_abs_ret_pos_L1", None))
            AbsCneg = _summ_contrast(cvl.get("abs_ret_neg_gt_abs_ret_neg_L1", None))

            # ---- ret_pos != ret_neg (BH across taus) ----
            pmn = tests.get("ret_pos_minus_ret_neg", None)
            if isinstance(pmn, pd.DataFrame) and not pmn.empty:
                rj = pmn["reject_bh"]
                nrej, m = _count_reject(rj)
                Asym = f"{nrej}/{m}"
            else:
                Asym = ""

            # ---- Tail tests: 4 columns with R/F (or blank if missing) ----
            # We will set each to:
            #   R if reject_bh True
            #   F if reject_bh False
            #   "" if that row is not present in tail table
            tail_cols = {"T+U": "", "T+L": "", "T-U": "", "T-L": ""}
            tail = tests.get("tail_vs_median_abs", None)
            if isinstance(tail, pd.DataFrame) and not tail.empty:
                # Initialize as F for those that exist; then overwrite to R where rejected
                # We detect existence by scanning rows.
                exist = {"T+U": False, "T+L": False, "T-U": False, "T-L": False}
                for _, r in tail.iterrows():
                    coef = r.get("coef", "")
                    lab = r.get("tail_label", "")
                    key = None
                    if coef == "ret_pos" and lab == "upper_vs_median":
                        key = "T+U"
                    elif coef == "ret_pos" and lab == "lower_vs_median":
                        key = "T+L"
                    elif coef == "ret_neg" and lab == "upper_vs_median":
                        key = "T-U"
                    elif coef == "ret_neg" and lab == "lower_vs_median":
                        key = "T-L"
                    if key is None:
                        continue

                    exist[key] = True
                    rej = bool(r.get("reject_bh", False))
                    tail_cols[key] = tail_reject_symbol if rej else tail_fail_symbol

                # If a key never existed, keep it blank
                for k, ex in exist.items():
                    if not ex:
                        tail_cols[k] = ""

            rows.append(
                dict(
                    H=f"{_parse_horizon_num(horizon_label):g}" if np.isfinite(_parse_horizon_num(horizon_label)) else str(horizon_label),
                    H_num=_parse_horizon_num(horizon_label),
                    Mom=mom_short,
                    Mom_order={"V": 0, "S": 1, "K": 2}.get(mom_short, 99),
                    q=f"{bq:.2f}",
                    Cpos=Cpos, Cneg=Cneg,
                    Lpos1=Lpos1, Lneg1=Lneg1,
                    Dep1=Dep1,
                    AbsCpos=AbsCpos, AbsCneg=AbsCneg,
                    Asym=Asym,
                    **tail_cols,
                )
            )

    df_main = pd.DataFrame(rows)

    # Sort by moment then horizon
    if not df_main.empty:
        df_main = df_main.sort_values(["Mom_order", "H_num", "Mom"]).reset_index(drop=True)

    # -----------------------
    # LaTeX rendering
    # -----------------------
    def _latex_preamble() -> str:
        return (
            "% Required packages:\n"
            "% \\usepackage{booktabs}\n"
            "% \\usepackage{threeparttable}\n"
            "% \\usepackage{adjustbox}\n"
        )

    def _escape_latex(s: str) -> str:
        return (s.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#"))

    def _render_main(df: pd.DataFrame) -> str:
        if df.empty:
            return "% (no test data found)\n"
    
        cols = [
            "Mom", "H",
            "Cpos", "Cneg",
            "Lpos1", "Lneg1",
            "Dep1",
            "AbsCpos", "AbsCneg",
            "Asym",
            "T+U", "T+L", "T-U", "T-L",
        ]
    
        df2 = df[cols].copy().replace({np.nan: ""})
    
        # Original math-style headers (no 'rej')
        headers = [
            "Moment", "Horizon",
            r"$Ret^{+}_t$",
            r"$Ret^{-}_t$",
            r"$Ret^{+}_{t-1}$",
            r"$Ret^{-}_{t-1}$",
            r"$Dep_{t-1}$",
            r"$|Ret^{+}_t|>|Ret^{+}_{t-1}|$",
            r"$|Ret^{-}_t|>|Ret^{-}_{t-1}|$",
            r"$Ret^{+}-Ret^{-}$",
            r"$|Ret^{+}_{U}|>|Ret^{+}_{0.50}|$",
            r"$|Ret^{+}_{L}|>|Ret^{+}_{0.50}|$",
            r"$|Ret^{-}_{U}|>|Ret^{-}_{0.50}|$",
            r"$|Ret^{-}_{L}|>|Ret^{-}_{0.50}|$",
        ]
    
        body_lines = []
        for _, row in df2.iterrows():
            cells = [str(row[c]) for c in cols]
            cells = [s.replace("&", "\\&") for s in cells]
            body_lines.append(" & ".join(cells) + r" \\")
        body = "\n".join(body_lines)
    
        caption = (
            r"BH-FDR test summary arranged by moment (V,S,K) and horizon. "
            r"Cells in the first nine columns report the number of quantiles rejecting the null "
            r"out of the total number of quantiles (BH across $\tau$). "
            r"Tail columns report R (reject) or F (fail to reject) "
            r"for BH-adjusted tail amplification tests."
        )
    
        colspec = "ll" + "c" * (len(cols) - 2)
    
        out = []
        out.append(r"\begin{table}[!htbp]")
        out.append(r"\centering")
        out.append(r"\small")
        out.append(r"\begin{threeparttable}")
        out.append(r"\begin{adjustbox}{max width=\textwidth}")
        out.append(r"\begin{tabular}{" + colspec + r"}")
        out.append(r"\toprule")
        out.append(" & ".join(headers) + r" \\")
        out.append(r"\midrule")
        out.append(body)
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{adjustbox}")
        out.append(r"\caption{" + caption + r"}")
        out.append(r"\label{tab:main_tests}")
        out.append(r"\end{threeparttable}")
        out.append(r"\end{table}")
        return "\n".join(out) + "\n"

    # -----------------------
    # Save files
    # -----------------------
    os.makedirs(out_dir, exist_ok=True)

    latex_main = _latex_preamble() + "\n" + _render_main(df_main)

    main_path = os.path.join(out_dir, f"{file_prefix}_main.tex")
    master_path = os.path.join(out_dir, f"{file_prefix}_master.tex")

    with open(main_path, "w", encoding="utf-8") as f:
        f.write(latex_main)

    master = (
        _latex_preamble()
        + "\n"
        + "% Master include file\n"
        + f"\\input{{{os.path.basename(main_path)}}}\n"
    )
    with open(master_path, "w", encoding="utf-8") as f:
        f.write(master)

    return {
        "paths": {"main": main_path, "master": master_path},
        "latex": {"main": latex_main, "master": master},
        "data": {"main_df": df_main},
    }


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
##########################PLotting 
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_qrm_across_quantiles_selectcoef(
    res_by_key,
    *,
    eq_key="A_var",
    coefs=("ret_pos", "ret_neg"),
    keys_order=None,
    ci=0.95,
    scale=1.0,
    title=None,
    figsize=(10, 8),
    legend=True,
    show_ols=True,
    sharex=True,

    marker="o",
    linewidth=2.0,
    ols_linestyle="--",
    ols_linewidth=1.4,
    ci_alpha=0.15,
    grid_alpha=0.25,

    zero_line=True,
    zero_line_kwargs=None,

    labels_map=None,
    colors=None,

    skip_missing=False,
    skip_missing_ols=True,

    # NEW
    save_path=None,
    dpi=300,
    bbox_inches="tight",
    close_after_save=False,
):
    """
    Plot quantile regression coefficients across quantiles.

    Supports multiple horizons and multiple coefficients.

    Parameters
    ----------
    res_by_key : dict
        Example:
        {
            "14d": results[14],
            "21d": results[21],
        }

    eq_key : str
        One of {"A_var","B_skew","C_kurt"}

    coefs : tuple or str
        Coefficients to plot.

    save_path : str or Path, optional
        If provided, figure will be saved.

    dpi : int
        Resolution when saving figure.

    close_after_save : bool
        If True closes the figure after saving (useful in loops).
    """

    # -----------------------------
    # normalize inputs
    # -----------------------------
    items = list(res_by_key.items())

    if isinstance(coefs, str):
        coefs = [coefs]
    else:
        coefs = list(coefs)

    if keys_order is not None:
        order_map = {k: i for i, k in enumerate(keys_order)}
        items.sort(key=lambda kv: order_map.get(kv[0], 1e9))

    n_panels = len(coefs)

    # -----------------------------
    # find common quantiles
    # -----------------------------
    tau_sets = []

    for label, res in items:
        taus = np.asarray(res[eq_key]["params"].index, dtype=float)
        tau_sets.append(set(np.round(taus, 10)))

    common = sorted(set.intersection(*tau_sets))
    taus_common = np.asarray(common)

    # -----------------------------
    # CI critical value
    # -----------------------------
    z = None
    if ci is not None:
        from scipy.stats import norm
        z = float(norm.ppf(0.5 + ci / 2.0))

    # -----------------------------
    # figure
    # -----------------------------
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=figsize,
        sharex=sharex
    )

    if n_panels == 1:
        axes = [axes]

    if zero_line_kwargs is None:
        zero_line_kwargs = dict(color="black", alpha=0.4, linewidth=1)

    # -----------------------------
    # helper functions
    # -----------------------------
    def get_label(k):
        if labels_map is None:
            return str(k)
        return labels_map.get(k, str(k))

    def get_color(k):
        if colors is None:
            return None
        return colors.get(k)

    def get_ols_coef(res, coef):
        eq = res[eq_key]

        if "ols" not in eq:
            if skip_missing_ols:
                return None
            raise KeyError("OLS results missing")

        params = eq["ols"]["params"]

        if coef not in params.index:
            if skip_missing_ols:
                return None
            raise KeyError(f"{coef} not found in OLS params")

        return float(params.loc[coef])

    # -----------------------------
    # plotting
    # -----------------------------
    for ax, coef in zip(axes, coefs):

        for label_raw, res in items:

            eq = res[eq_key]
            params = eq["params"]

            if coef not in params.columns:
                if skip_missing:
                    continue
                raise KeyError(f"{coef} not in params")

            params_al = params.loc[taus_common]

            beta = params_al[coef].to_numpy() * scale

            label = get_label(label_raw)
            color = get_color(label_raw)

            line = ax.plot(
                taus_common,
                beta,
                marker=marker,
                linewidth=linewidth,
                label=label,
                color=color
            )

            line_color = line[0].get_color()

            # -----------------
            # CI
            # -----------------
            if ci is not None and "se_boot" in eq:

                se_df = eq["se_boot"]

                if coef in se_df.columns:

                    se = se_df.loc[taus_common][coef].to_numpy() * scale

                    lower = beta - z * se
                    upper = beta + z * se

                    ax.fill_between(
                        taus_common,
                        lower,
                        upper,
                        alpha=ci_alpha,
                        color=line_color
                    )

            # -----------------
            # OLS line
            # -----------------
            if show_ols:

                b_ols = get_ols_coef(res, coef)

                if b_ols is not None:

                    ax.axhline(
                        b_ols * scale,
                        linestyle=ols_linestyle,
                        linewidth=ols_linewidth,
                        color=line_color,
                        alpha=0.9
                    )

        if zero_line:
            ax.axhline(0, **zero_line_kwargs)

        ax.grid(True, alpha=grid_alpha)

        ylabel = f"{coef} β(τ)"
        if scale != 1.0:
            ylabel += " (scaled)"

        ax.set_ylabel(ylabel)

    # -----------------------------
    # titles
    # -----------------------------
    if title is None:
        title = f"{eq_key}: coefficients across quantiles"

    axes[0].set_title(title)
    axes[-1].set_xlabel("Quantile (τ)")

    if legend:
        axes[0].legend(title="Horizon")

    fig.tight_layout()

    # -----------------------------
    # SAVE
    # -----------------------------
    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches=bbox_inches
        )

        print(f"Saved figure → {save_path}")

        if close_after_save:
            plt.close(fig)

    return fig, axes
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm


def plot_qrm_by_quantile_across_frequencies_selectcoef(
    res_by_key,
    *,
    eq_key: str = "A_var",
    coefs=("ret_pos", "ret_neg"),
    taus_to_plot=(0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95),
    keys_order=None,
    ncols: int = 2,
    ci: float | None = None,
    scale: float = 1.0,
    title: str | None = None,
    figsize_per_panel=(5.4, 3.2),
    legend: bool = True,
    show_ols: bool = True,
    ols_hac_lags: int | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    bbox_inches: str = "tight",
    close_after_save: bool = False,
):
    """
    Generalized Fig-4 style small multiples:
      - each subplot: a quantile τ
      - x-axis: horizon/frequency label (keys of res_by_key)
      - within each subplot: one line per coefficient in `coefs`

    OLS:
      - dashed horizontal line per coef (OLS slope), constant across horizons (computed per res)
        We compute OLS for each label separately; if you prefer a single pooled OLS, adjust accordingly.

    Save options:
      - save_path: file path to save figure
      - dpi: resolution for saved figure
      - bbox_inches: passed to fig.savefig
      - close_after_save: whether to close fig after saving
    """
    items = list(res_by_key.items()) if isinstance(res_by_key, dict) else list(res_by_key)

    if keys_order is not None:
        order_map = {k: i for i, k in enumerate(keys_order)}
        items.sort(key=lambda kv: order_map.get(kv[0], 10**9))

    labels = [str(k) for k, _ in items]

    if isinstance(coefs, str):
        coefs = [coefs]
    else:
        coefs = list(coefs)

    def _nearest_tau(taus_arr, t):
        taus_arr = np.array(taus_arr, dtype=float)
        j = int(np.argmin(np.abs(taus_arr - float(t))))
        return float(taus_arr[j])

    z = None
    if ci is not None:
        try:
            from scipy.stats import norm
            z = float(norm.ppf(0.5 + ci / 2.0))
        except Exception:
            z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96

    def _ols_slope_for_coef(res: dict, coef_name: str) -> float:
        eq = res[eq_key]

        # prefer stored OLS if available
        if "ols" in eq and "params" in eq["ols"]:
            ols_params = eq["ols"]["params"]
            if coef_name not in ols_params.index:
                raise KeyError(
                    f"coef '{coef_name}' not in stored OLS params: {list(ols_params.index)}"
                )
            return float(ols_params.loc[coef_name])

        # fallback: refit if raw data exists
        if "data" not in res:
            raise KeyError(
                "No stored OLS params found and res['data'] is unavailable."
            )

        df = res["data"]
        dep = eq["dep"]
        X_cols = list(eq["X_cols"])

        has_const = "const" in X_cols
        X_cols_wo = [c for c in X_cols if c != "const"]

        X = df[X_cols_wo].astype(float)
        if has_const:
            X = sm.add_constant(X, has_constant="add")

        y = df[dep].astype(float).to_numpy(float)

        model = sm.OLS(y, X.to_numpy(float))
        if ols_hac_lags is None:
            fit = model.fit()
        else:
            fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(ols_hac_lags)})

        cols = list(X.columns)
        if coef_name not in cols:
            raise KeyError(f"coef '{coef_name}' not in OLS design columns: {cols}")
        return float(fit.params[cols.index(coef_name)])

    taus_panels = list(taus_to_plot)
    n_panels = len(taus_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    extracted = []
    for label, res in items:
        eq = res[eq_key]
        taus_av = np.array(eq["params"].index, dtype=float)
        params = eq["params"]
        se = eq.get("se_boot", None)
        extracted.append((str(label), res, taus_av, params, se))

    for i, t_req in enumerate(taus_panels):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        x = np.arange(len(labels), dtype=float)

        for coef_name in coefs:
            y = np.zeros(len(labels), float)
            e = np.zeros(len(labels), float) if (ci is not None) else None

            for j, (lab, res, taus_av, params, se) in enumerate(extracted):
                t_use = _nearest_tau(taus_av, t_req)

                if coef_name not in params.columns:
                    raise KeyError(
                        f"'{coef_name}' not in params columns for label={lab}. "
                        f"Available: {list(params.columns)}"
                    )

                y[j] = float(params.loc[t_use, coef_name]) * scale

                if ci is not None and se is not None and coef_name in se.columns:
                    e[j] = float(se.loc[t_use, coef_name]) * scale

            ax.plot(x, y, marker="o", linewidth=2, label=coef_name)

            if ci is not None:
                ax.errorbar(x, y, yerr=z * e, fmt="none", capsize=3, alpha=0.8)

            if show_ols:
                ols_vals = np.array(
                    [_ols_slope_for_coef(res, coef_name) for _, res, *_ in extracted],
                    dtype=float
                ) * scale
                ax.plot(x, ols_vals, linestyle="--", linewidth=1.5, alpha=0.9)

        ax.set_title(f"{eq_key}: τ≈{t_req:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("Response" + (" (%)" if scale == 100.0 else ""))
        ax.grid(True, alpha=0.25)
        if legend:
            ax.legend(loc="best")

    for k in range(n_panels, nrows * ncols):
        rr = k // ncols
        cc = k % ncols
        axes[rr, cc].axis("off")

    if title is not None:
        fig.suptitle(title, y=0.995)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved figure -> {save_path}")
        if close_after_save:
            plt.close(fig)

    return fig, axes
####################
import os
import re
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Any, Sequence, Tuple, Optional, List, Union



def _sig_stars(p: float) -> str:
    if p is None or (isinstance(p, float) and (not np.isfinite(p))):
        return ""
    if p < 0.01:
        return r"\sym{***}"
    if p < 0.05:
        return r"\sym{**}"
    if p < 0.10:
        return r"\sym{*}"
    return ""


def _fmt_num(x: float, decimals: int) -> str:
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return ""
    return f"{float(x):.{decimals}f}"


def _fmt_p(p: float, decimals: int = 3) -> str:
    if p is None or (isinstance(p, float) and (not np.isfinite(p))):
        return ""
    return f"{float(p):.{decimals}f}"


def _latex_escape_basic(s: str) -> str:
    return s.replace("_", r"\_")


def _is_latex(s: str) -> bool:
    return ("\\" in s) or ("{" in s) or ("}" in s) or ("^" in s) or ("$" in s)


def _get_asym_by_horizon(asym: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "A_var" in asym and "B_skew" in asym and "C_kurt" in asym:
        hz = asym.get("A_var", {}).get("meta", {}).get("horizon_label", "H")
        return {str(hz): asym}
    return {str(k): v for k, v in asym.items()}


def _infer_dep_name_from_eq_key(eq_key: str) -> str:
    ek = (eq_key or "").lower()
    if "var" in ek:
        return "var"
    if "skew" in ek:
        return "skew"
    if "kurt" in ek:
        return "kurt"
    return "dep"


def _build_default_row_vars(first_cols: List[str], *, dep_name: str, include_const: bool) -> List[str]:
    cols_set = set(first_cols)

    return_candidates = [
        "ret_pos", "ret_neg",
        "ret_pos_L1", "ret_neg_L1",
        "ret_pos_L2", "ret_neg_L2",
        "rt_pos", "rt_neg",
        "rt_pos_L1", "rt_neg_L1",
        "rt_pos_L2", "rt_neg_L2",
        "ret_L1", "ret_L2", "rt_L1", "rt_L2",
    ]
    keep: List[str] = [c for c in return_candidates if c in cols_set]

    dep = dep_name.lower()
    dep_lag_candidates = [
        f"d_{dep}_L1", f"d_{dep}_L2",
        f"{dep}_L1", f"{dep}_L2",
        f"d{dep}_L1", f"d{dep}_L2",
        f"delta_{dep}_L1", f"delta_{dep}_L2",
    ]
    keep += [c for c in dep_lag_candidates if c in cols_set]

    if include_const:
        for c in first_cols:
            if c.lower() in {"const", "intercept"}:
                keep.append(c)
                break

    seen = set()
    out = []
    for c in keep:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def asym_to_latex_table(
    asym: Dict[str, Any],
    *,
    eq_key: str = "A_var",
    caption: str = "Quantile regression estimates across horizons.",
    label: str = "tab:asym_quant",
    horizons: Optional[Sequence[str]] = None,
    taus: Optional[Sequence[float]] = None,
    drop_missing_horizons: bool = True,
    drop_missing_taus: bool = True,
    include_vars: Optional[Sequence[str]] = None,
    drop_missing_vars: bool = True,
    include_const: bool = False,
    var_rename: Optional[Dict[str, str]] = None,
    include_ols: bool = True,
    show_pvalues_below: bool = False,
    stars_on_estimates: bool = True,
    decimals: int = 4,
    p_decimals: int = 3,
    size_cmd: str = r"\tiny",
    include_r2_row: bool = True,
    r2_decimals: int = 3,
    save_path: Optional[Union[str, Path]] = None,
    make_dirs: bool = True,
    encoding: str = "utf-8",
) -> str:
    var_rename = var_rename or {}

    asym_by_hz = _get_asym_by_horizon(asym)
    try:
        available_hz = sorted(asym_by_hz.keys(), key=lambda x: float(x))
    except Exception:
        available_hz = list(asym_by_hz.keys())

    if horizons is None:
        hz_list = available_hz
    else:
        hz_list = [str(h) for h in horizons]
        missing_h = [h for h in hz_list if h not in asym_by_hz]
        if missing_h:
            if drop_missing_horizons:
                hz_list = [h for h in hz_list if h in asym_by_hz]
            else:
                raise KeyError(f"horizons contains missing keys: {missing_h}")

    if not hz_list:
        raise ValueError("No horizons selected after filtering.")

    first_eq = asym_by_hz[hz_list[0]][eq_key]
    avail_taus = [float(t) for t in first_eq["meta"]["taus"]]

    if taus is None:
        taus_list = avail_taus
    else:
        taus_list = [float(t) for t in taus]
        missing_t = [t for t in taus_list if t not in set(avail_taus)]
        if missing_t:
            if drop_missing_taus:
                taus_list = [t for t in taus_list if t in set(avail_taus)]
            else:
                raise KeyError(f"taus contains values not in the fitted model: {missing_t}")

    if not taus_list:
        raise ValueError("No taus selected after filtering.")

    tau_headers = [f"{t:.2f}" for t in taus_list]
    first_cols_all = list(first_eq["params"].columns)

    first_cols = first_cols_all[:]
    if not include_const:
        first_cols = [c for c in first_cols if c.lower() not in {"const", "intercept"}]

    dep_name = _infer_dep_name_from_eq_key(eq_key)

    if include_vars is None:
        row_vars = _build_default_row_vars(first_cols_all, dep_name=dep_name, include_const=include_const)
        if not include_const:
            row_vars = [v for v in row_vars if v.lower() not in {"const", "intercept"}]
    else:
        row_vars = [v for v in include_vars if (include_const or v.lower() not in {"const", "intercept"})]
        missing_v = [v for v in row_vars if v not in first_cols_all]
        if missing_v:
            if drop_missing_vars:
                row_vars = [v for v in row_vars if v in first_cols_all]
            else:
                raise KeyError(f"include_vars contains variables not found in design matrix: {missing_v}")

    col_spec = "ll" + ("c" if include_ols else "") + ("c" * len(taus_list))

    header_cells = ["Horizon", "Variable"]
    if include_ols:
        header_cells.append("OLS")
    header_cells += tau_headers
    header = " & ".join(header_cells) + r" \\"

    lines: List[str] = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(size_cmd)
    lines.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for h_i, hz in enumerate(hz_list):
        eq = asym_by_hz[hz][eq_key]

        q_params: pd.DataFrame = eq["params"]
        q_pvals: pd.DataFrame = eq["p_boot_norm"]
        q_pseudo_r2 = eq.get("q_pseudo_r2", pd.Series(dtype=float))

        ols = eq.get("ols", {}) if include_ols else {}
        ols_params = ols.get("params", pd.Series(dtype=float))
        ols_pvals = ols.get("p_hac", pd.Series(dtype=float))
        ols_r2 = float(ols.get("r2", np.nan)) if include_ols else np.nan

        for r_i, v in enumerate(row_vars):
            hz_cell = hz if r_i == 0 else ""
            v_disp = var_rename.get(v, v)
            v_cell = v_disp if _is_latex(v_disp) else _latex_escape_basic(v_disp)

            row = [hz_cell, v_cell]

            if include_ols:
                if v in ols_params.index:
                    b = float(ols_params.loc[v])
                    p = float(ols_pvals.loc[v]) if v in ols_pvals.index else np.nan
                    cell = _fmt_num(b, decimals) + (_sig_stars(p) if stars_on_estimates else "")
                else:
                    cell = ""
                row.append(cell)

            for t in taus_list:
                b = float(q_params.loc[t, v]) if (t in q_params.index and v in q_params.columns) else np.nan
                p = float(q_pvals.loc[t, v]) if (t in q_pvals.index and v in q_pvals.columns) else np.nan
                cell = _fmt_num(b, decimals) + (_sig_stars(p) if stars_on_estimates else "")
                row.append(cell)

            lines.append(" & ".join(row) + r" \\")

            if show_pvalues_below:
                prow = ["", ""]
                if include_ols:
                    p = float(ols_pvals.loc[v]) if v in ols_pvals.index else np.nan
                    prow.append(rf"({_fmt_p(p, p_decimals)})" if np.isfinite(p) else "")
                for t in taus_list:
                    p = float(q_pvals.loc[t, v]) if (t in q_pvals.index and v in q_pvals.columns) else np.nan
                    prow.append(rf"({_fmt_p(p, p_decimals)})" if np.isfinite(p) else "")
                lines.append(" & ".join(prow) + r" \\")

        if include_r2_row:
            rrow = ["", r"$R^2$"]
            if include_ols:
                rrow.append(_fmt_num(ols_r2, r2_decimals))
            for t in taus_list:
                val = np.nan
                if isinstance(q_pseudo_r2, pd.Series) and (t in q_pseudo_r2.index):
                    val = float(q_pseudo_r2.loc[t])
                rrow.append(_fmt_num(val, r2_decimals))
            lines.append(r"\addlinespace[0.25em]")
            lines.append(" & ".join(rrow) + r" \\")

        if h_i < len(hz_list) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{0.5em}")
    lines.append(r"\begin{minipage}{0.98\linewidth}")
    if show_pvalues_below:
        lines.append(
            r"\footnotesize \textit{Notes:} P-values are shown in parentheses below point estimates. "
            r"Stars denote significance: \sym{***} $p<0.01$, \sym{**} $p<0.05$, \sym{*} $p<0.10$."
        )
    else:
        lines.append(
            r"\footnotesize \textit{Notes:} Stars denote significance: "
            r"\sym{***} $p<0.01$, \sym{**} $p<0.05$, \sym{*} $p<0.10$."
        )
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)

    if save_path is not None:
        path = Path(save_path)
        if make_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tex, encoding=encoding)

    return tex


@dataclass
class LatexTableSpec:
    title: str
    caption: str
    label: str
    filename: str


def _fmt_p_star(
    x: Any,
    *,
    decimals: int = 3,
    star_levels: Tuple[float, float, float] = (0.10, 0.05, 0.01),
) -> str:
    try:
        p = float(x)
    except Exception:
        return ""
    if not np.isfinite(p):
        return ""

    s = f"{p:.{decimals}f}"

    p10, p05, p01 = star_levels
    stars = ""
    if p <= p01:
        stars = r"\sym{***}"
    elif p <= p05:
        stars = r"\sym{**}"
    elif p <= p10:
        stars = r"\sym{*}"
    return s + stars


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def _moment_name_from_eq(eq_key: str) -> str:
    m = {"A_var": "Variance", "B_skew": "Skewness", "C_kurt": "Kurtosis"}
    return m.get(eq_key, str(eq_key))


def _moment_abbrev(moment_full: str) -> str:
    m = {"Variance": "V", "Skewness": "S", "Kurtosis": "K"}
    return m.get(moment_full, moment_full[:1].upper())


def _parse_momh(s: str) -> Tuple[str, float]:
    try:
        mom, hz = s.split("|")
        return mom.strip(), float(hz)
    except Exception:
        return str(s), float("inf")


def _sort_by_momh(df: pd.DataFrame, momh_col: str = "Mom|H") -> pd.DataFrame:
    if momh_col not in df.columns:
        return df
    order = {"V": 0, "S": 1, "K": 2}

    def _key(v: Any) -> Tuple[int, float, str]:
        mom, hz = _parse_momh(str(v))
        return (order.get(mom, 999), hz, str(v))

    return df.sort_values(by=momh_col, key=lambda s: s.map(_key))


def _insert_moment_separators(
    df: pd.DataFrame,
    cols: List[str],
    *,
    momh_col: str = "Mom|H",
    marker: str = "__MIDRULE__",
) -> pd.DataFrame:
    rows: List[dict] = []
    last_mom = None
    for _, r in df.iterrows():
        mom, _hz = _parse_momh(str(r.get(momh_col, "")))
        if last_mom is not None and mom != last_mom:
            sep = {c: "" for c in cols}
            sep[momh_col] = marker
            rows.append(sep)
        rows.append({c: r.get(c, "") for c in cols})
        last_mom = mom
    return pd.DataFrame(rows, columns=cols)


def _normalize_out_to_horizon_map(out: dict) -> Dict[str, dict]:
    if not isinstance(out, dict):
        return {}

    if any(k in out for k in ("A_var", "B_skew", "C_kurt")):
        hz = None
        try:
            hz = out.get("A_var", {}).get("meta", {}).get("horizon_label", None)
        except Exception:
            hz = None
        if not hz:
            hz = "H"
        return {str(hz): out}

    horizon_map: Dict[str, dict] = {}
    for k, v in out.items():
        if isinstance(v, dict) and any(kk in v for kk in ("A_var", "B_skew", "C_kurt")):
            horizon_map[str(k)] = v
    return horizon_map


def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\s\-\(\)\:\|]+", " ", s)
    return s


def _get_test_block(eq_out: dict, test_key: str) -> Optional[dict]:
    if not isinstance(eq_out, dict):
        return None
    tests = eq_out.get("tests", None)
    if not isinstance(tests, dict):
        return None

    if test_key in tests:
        return tests[test_key]

    tgt = _canon(test_key)
    for k in tests.keys():
        if _canon(k) == tgt:
            return tests[k]

    for k in tests.keys():
        if tgt in _canon(k) or _canon(k) in tgt:
            return tests[k]

    return None


def _unwrap_block(blk: Any) -> Any:
    if not isinstance(blk, dict):
        return blk
    if ("global" in blk) or ("segments" in blk):
        return blk
    for _k, v in blk.items():
        if isinstance(v, dict) and (("global" in v) or ("segments" in v)):
            return v
    return blk


def _extract_everywhere_only(blk: Any) -> Dict[str, float]:
    blk = _unwrap_block(blk)

    def _p(d: Any, *path) -> float:
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return np.nan
            cur = cur[k]
        try:
            return float(cur)
        except Exception:
            return np.nan

    return {
        "every_global": _p(blk, "global", "everywhere", "p_value"),
        "every_low": _p(blk, "segments", "low", "everywhere", "p_value"),
        "every_mid": _p(blk, "segments", "mid", "everywhere", "p_value"),
        "every_high": _p(blk, "segments", "high", "everywhere", "p_value"),
    }


def _extract_uniform_dom_only(blk: Any) -> Dict[str, float]:
    blk = _unwrap_block(blk)

    def _p(d: Any, *path) -> float:
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return np.nan
            cur = cur[k]
        try:
            return float(cur)
        except Exception:
            return np.nan

    return {
        "uni_global": _p(blk, "global", "uniform_positive_dominance", "p_value"),
        "uni_low": _p(blk, "segments", "low", "uniform_positive_dominance", "p_value"),
        "uni_mid": _p(blk, "segments", "mid", "uniform_positive_dominance", "p_value"),
        "uni_high": _p(blk, "segments", "high", "uniform_positive_dominance", "p_value"),
    }


def _extract_tail_extremes(blk: Any) -> Dict[str, float]:
    def _p(d: Any, key: str) -> float:
        if not isinstance(d, dict):
            return np.nan
        node = d.get(key, None)
        if not isinstance(node, dict):
            return np.nan
        try:
            return float(node.get("p_value", np.nan))
        except Exception:
            return np.nan

    return {
        "high_gt_mid": _p(blk, "high_gt_mid"),
        "low_gt_mid": _p(blk, "low_gt_mid"),
    }


def _header_type_from_cols(cols: List[str]) -> str:
    joined = " | ".join(cols)

    if len(cols) == 9 and "p_{\\mathrm{Every}}(Ret^+" in joined and "p_{\\mathrm{Every}}(Ret^-" in joined:
        return "merged_every"
    if len(cols) == 9 and "p_{\\mathrm{Uni+}}(Ret^+" in joined and "p_{\\mathrm{Uni+}}(Ret^-" in joined:
        return "merged_uni"
    if len(cols) == 5 and "p_{\\mathrm{Every}}" in joined:
        return "single_every"
    if len(cols) == 5 and "Hi$>$Mid" in joined:
        return "tail"
    return "default"


def _df_to_booktabs_latex_with_title_and_caption_below(
    df: pd.DataFrame,
    *,
    title: str,
    caption: str,
    label: str,
    escape: bool = False,
    resize_to_textwidth: bool = True,
    midrule_marker: str = "__MIDRULE__",
) -> str:
    cols = list(df.columns)
    htype = _header_type_from_cols(cols)

    # Add vertical separator between + and - blocks for merged tables
    if htype in {"merged_every", "merged_uni"}:
        col_spec = "lcccc|cccc"
    else:
        col_spec = "l" + "c" * (len(cols) - 1)

    def _esc(s: Any) -> str:
        s = "" if s is None else str(s)
        if not escape:
            return s
        s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
        s = s.replace("_", r"\_").replace("$", r"\$")
        return s

    lines: List[str] = []
    lines.append(r"\begin{table}[!htbp] \centering")
    lines.append(r"\tiny")
    lines.append(r"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}")
    lines.append(rf"\textbf{{{title}}}\par")
    lines.append(r"\vspace{2pt}")

    if label:
        lines.append(rf"\label{{{label}}}")

    if resize_to_textwidth:
        lines.append(r"\resizebox{\textwidth}{!}{%")

    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    if htype == "merged_every":
        lines.append(
            r"Mom|H & \multicolumn{4}{c|}{P-values for $Ret^{+}$} & \multicolumn{4}{c}{P-values for $Ret^{-}$} \\"
        )
        lines.append(
            r"& All & Low & Mid & High & All & Low & Mid & High \\"
        )
    elif htype == "merged_uni":
        lines.append(
            r"Mom|H & \multicolumn{4}{c|}{P-values for $Ret^{+}$ dominance} & \multicolumn{4}{c}{P-values for $Ret^{-}$ dominance} \\"
        )
        lines.append(
            r"& All & Low & Mid & High & All & Low & Mid & High \\"
        )
    elif htype == "single_every":
        lines.append(
            r"Mom|H & \multicolumn{4}{c}{P-values} \\"
        )
        lines.append(
            r"& All & Low & Mid & High \\"
        )
    elif htype == "tail":
        lines.append(
            r"Mom|H & \multicolumn{2}{c|}{$Ret^{+}$} & \multicolumn{2}{c}{$Ret^{-}$} \\"
        )
        lines.append(
            r"& Hi$>$Mid & Lo$>$Mid & Hi$>$Mid & Lo$>$Mid \\"
        )
    else:
        lines.append(" & ".join(cols) + r" \\")

    lines.append(r"\midrule")

    for _, row in df.iterrows():
        if str(row.get(cols[0], "")) == midrule_marker:
            lines.append(r"\midrule")
            continue
        vals = [_esc(row.get(c, "")) for c in cols]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if resize_to_textwidth:
        lines.append(r"}")

    lines.append(r"\vspace{2pt}")
    lines.append(rf"\caption*{{{caption}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_asym_quantreg_tests_latex(
    out: dict,
    *,
    base_dir: Optional[Union[str, Path]] = None,
    decimals: int = 3,
    file_prefix: str = "asymqreg_joint_tests",
    include_eqs: Tuple[str, ...] = ("A_var", "B_skew", "C_kurt"),
    lag_for_pos: int = 1,
    lag_for_neg: int = 1,
    star_levels: Tuple[float, float, float] = (0.10, 0.05, 0.01),
) -> Dict[str, Dict[str, str]]:
    base_path = Path(base_dir).expanduser().resolve() if base_dir is not None else None
    if base_path is not None:
        base_path.mkdir(parents=True, exist_ok=True)

    horizon_map = _normalize_out_to_horizon_map(out)

    def pstar(v: Any) -> str:
        return _fmt_p_star(v, decimals=decimals, star_levels=star_levels)

    COL_EPOS_ALL  = r"$p_{\mathrm{Every}}(Ret^+,\mathrm{All})$"
    COL_EPOS_LOW  = r"$p_{\mathrm{Every}}(Ret^+,\mathrm{Low})$"
    COL_EPOS_MID  = r"$p_{\mathrm{Every}}(Ret^+,\mathrm{Mid})$"
    COL_EPOS_HIGH = r"$p_{\mathrm{Every}}(Ret^+,\mathrm{High})$"

    COL_ENEG_ALL  = r"$p_{\mathrm{Every}}(Ret^-,\mathrm{All})$"
    COL_ENEG_LOW  = r"$p_{\mathrm{Every}}(Ret^-,\mathrm{Low})$"
    COL_ENEG_MID  = r"$p_{\mathrm{Every}}(Ret^-,\mathrm{Mid})$"
    COL_ENEG_HIGH = r"$p_{\mathrm{Every}}(Ret^-,\mathrm{High})$"

    COL_UPOS_ALL  = r"$p_{\mathrm{Uni+}}(Ret^+,\mathrm{All})$"
    COL_UPOS_LOW  = r"$p_{\mathrm{Uni+}}(Ret^+,\mathrm{Low})$"
    COL_UPOS_MID  = r"$p_{\mathrm{Uni+}}(Ret^+,\mathrm{Mid})$"
    COL_UPOS_HIGH = r"$p_{\mathrm{Uni+}}(Ret^+,\mathrm{High})$"

    COL_UNEG_ALL  = r"$p_{\mathrm{Uni+}}(Ret^-,\mathrm{All})$"
    COL_UNEG_LOW  = r"$p_{\mathrm{Uni+}}(Ret^-,\mathrm{Low})$"
    COL_UNEG_MID  = r"$p_{\mathrm{Uni+}}(Ret^-,\mathrm{Mid})$"
    COL_UNEG_HIGH = r"$p_{\mathrm{Uni+}}(Ret^-,\mathrm{High})$"

    KEY_RET_POS_NEW = "(1) Contemporaneous positive returns"
    KEY_RET_NEG_NEW = "(1) Contemporaneous negative returns"
    KEY_LAGGED_NEW = "(2) Lagged returns"
    KEY_DOM_POS_NEW = "(3) Dominance: |positive return| > |lag 1 positive return|"
    KEY_DOM_NEG_NEW = "(3) Dominance: |negative return| > |lag 1 negative return|"

    KEY_DEP_CANDIDATES_NEW = (
        "(4) Lag 1 dependent variable",
        "(4) Lagged dependent variable",
        "(4) Own-lag dependence",
        "(4) Autoregressive dependence",
    )

    KEY_ASYM_NEW = "(5) Asymmetry: |positive return| differs from |negative return|"
    KEY_TAIL_NEW = "(6) Tail extremeness relative to middle (magnitude)"

    KEY_RET_POS_OLD = "(1)_ret_pos_significance"
    KEY_RET_NEG_OLD = "(1)_ret_neg_significance"
    KEY_LAGGED_OLD = "(2)_lagged_returns_significance"
    KEY_DOM_POS_OLD = "(3)_abs_ret_pos_gt_abs_ret_pos_L1"
    KEY_DOM_NEG_OLD = "(3)_abs_ret_neg_gt_abs_ret_neg_L1"
    KEY_DEP_L1_OLD = "(4)_dep_L1_significance"
    KEY_ASYM_OLD = "(5)_abs_ret_pos_diff_abs_ret_neg"
    KEY_TAIL_OLD = "(6)_tail_extremeness_vs_middle"

    def _get_block_multi(eq_out: dict, keys: Sequence[str]) -> Optional[dict]:
        for k in keys:
            blk = _get_test_block(eq_out, k)
            if isinstance(blk, dict):
                return blk
        return None

    def _rows_merged_everywhere(keys_pos: Sequence[str], keys_neg: Sequence[str]) -> List[dict]:
        rows: List[dict] = []
        for hz, hout in horizon_map.items():
            for eq_key in include_eqs:
                eq_out = hout.get(eq_key, None)
                if not isinstance(eq_out, dict):
                    continue
                moment_full = _moment_name_from_eq(eq_key)
                mom = _moment_abbrev(moment_full)

                blk_pos = _get_block_multi(eq_out, keys_pos)
                blk_neg = _get_block_multi(eq_out, keys_neg)

                ex_pos = _extract_everywhere_only(blk_pos)
                ex_neg = _extract_everywhere_only(blk_neg)

                rows.append(
                    {
                        "Mom|H": f"{mom}|{hz}",
                        COL_EPOS_ALL: pstar(ex_pos["every_global"]),
                        COL_EPOS_LOW: pstar(ex_pos["every_low"]),
                        COL_EPOS_MID: pstar(ex_pos["every_mid"]),
                        COL_EPOS_HIGH: pstar(ex_pos["every_high"]),
                        COL_ENEG_ALL: pstar(ex_neg["every_global"]),
                        COL_ENEG_LOW: pstar(ex_neg["every_low"]),
                        COL_ENEG_MID: pstar(ex_neg["every_mid"]),
                        COL_ENEG_HIGH: pstar(ex_neg["every_high"]),
                        "__moment_full__": moment_full,
                        "__horizon__": str(hz),
                    }
                )
        return rows

    def _rows_merged_lagged(lag_pos: int, lag_neg: int) -> List[dict]:
        rows: List[dict] = []
        for hz, hout in horizon_map.items():
            for eq_key in include_eqs:
                eq_out = hout.get(eq_key, None)
                if not isinstance(eq_out, dict):
                    continue
                moment_full = _moment_name_from_eq(eq_key)
                mom = _moment_abbrev(moment_full)

                lag_blk_new = _get_test_block(eq_out, KEY_LAGGED_NEW) or {}
                lag_blk_old = _get_test_block(eq_out, KEY_LAGGED_OLD) or {}

                key_pos_new = f"Positive return lag {int(lag_pos)}"
                key_neg_new = f"Negative return lag {int(lag_neg)}"
                key_pos_old = f"ret_pos_L{int(lag_pos)}"
                key_neg_old = f"ret_neg_L{int(lag_neg)}"

                blk_pos = None
                blk_neg = None

                if isinstance(lag_blk_new, dict) and key_pos_new in lag_blk_new:
                    blk_pos = lag_blk_new.get(key_pos_new)
                elif isinstance(lag_blk_old, dict) and key_pos_old in lag_blk_old:
                    blk_pos = lag_blk_old.get(key_pos_old)

                if isinstance(lag_blk_new, dict) and key_neg_new in lag_blk_new:
                    blk_neg = lag_blk_new.get(key_neg_new)
                elif isinstance(lag_blk_old, dict) and key_neg_old in lag_blk_old:
                    blk_neg = lag_blk_old.get(key_neg_old)

                ex_pos = _extract_everywhere_only(blk_pos)
                ex_neg = _extract_everywhere_only(blk_neg)

                rows.append(
                    {
                        "Mom|H": f"{mom}|{hz}",
                        COL_EPOS_ALL: pstar(ex_pos["every_global"]),
                        COL_EPOS_LOW: pstar(ex_pos["every_low"]),
                        COL_EPOS_MID: pstar(ex_pos["every_mid"]),
                        COL_EPOS_HIGH: pstar(ex_pos["every_high"]),
                        COL_ENEG_ALL: pstar(ex_neg["every_global"]),
                        COL_ENEG_LOW: pstar(ex_neg["every_low"]),
                        COL_ENEG_MID: pstar(ex_neg["every_mid"]),
                        COL_ENEG_HIGH: pstar(ex_neg["every_high"]),
                        "__moment_full__": moment_full,
                        "__horizon__": str(hz),
                    }
                )
        return rows

    def _rows_merged_uniform_dominance(keys_pos: Sequence[str], keys_neg: Sequence[str]) -> List[dict]:
        rows: List[dict] = []
        for hz, hout in horizon_map.items():
            for eq_key in include_eqs:
                eq_out = hout.get(eq_key, None)
                if not isinstance(eq_out, dict):
                    continue
                moment_full = _moment_name_from_eq(eq_key)
                mom = _moment_abbrev(moment_full)

                blk_pos = _get_block_multi(eq_out, keys_pos)
                blk_neg = _get_block_multi(eq_out, keys_neg)

                ex_pos = _extract_uniform_dom_only(blk_pos)
                ex_neg = _extract_uniform_dom_only(blk_neg)

                rows.append(
                    {
                        "Mom|H": f"{mom}|{hz}",
                        COL_UPOS_ALL: pstar(ex_pos["uni_global"]),
                        COL_UPOS_LOW: pstar(ex_pos["uni_low"]),
                        COL_UPOS_MID: pstar(ex_pos["uni_mid"]),
                        COL_UPOS_HIGH: pstar(ex_pos["uni_high"]),
                        COL_UNEG_ALL: pstar(ex_neg["uni_global"]),
                        COL_UNEG_LOW: pstar(ex_neg["uni_low"]),
                        COL_UNEG_MID: pstar(ex_neg["uni_mid"]),
                        COL_UNEG_HIGH: pstar(ex_neg["uni_high"]),
                        "__moment_full__": moment_full,
                        "__horizon__": str(hz),
                    }
                )
        return rows

    def _rows_from_testkeys_everywhere(keys: Sequence[str]) -> List[dict]:
        rows: List[dict] = []
        for hz, hout in horizon_map.items():
            for eq_key in include_eqs:
                eq_out = hout.get(eq_key, None)
                if not isinstance(eq_out, dict):
                    continue
                moment_full = _moment_name_from_eq(eq_key)
                mom = _moment_abbrev(moment_full)

                blk = _get_block_multi(eq_out, keys)
                ex = _extract_everywhere_only(blk)

                rows.append(
                    {
                        "Mom|H": f"{mom}|{hz}",
                        r"$p_{\mathrm{Every}}$(All)": pstar(ex["every_global"]),
                        r"$p_{\mathrm{Every}}$(Low)": pstar(ex["every_low"]),
                        r"$p_{\mathrm{Every}}$(Mid)": pstar(ex["every_mid"]),
                        r"$p_{\mathrm{Every}}$(High)": pstar(ex["every_high"]),
                        "__moment_full__": moment_full,
                        "__horizon__": str(hz),
                    }
                )
        return rows

    def _rows_tail_extremes_combined() -> List[dict]:
        rows: List[dict] = []
        for hz, hout in horizon_map.items():
            for eq_key in include_eqs:
                eq_out = hout.get(eq_key, None)
                if not isinstance(eq_out, dict):
                    continue

                moment_full = _moment_name_from_eq(eq_key)
                mom = _moment_abbrev(moment_full)

                blk_all = _get_block_multi(eq_out, [KEY_TAIL_NEW, KEY_TAIL_OLD]) or {}

                blk_pos = None
                blk_neg = None
                if isinstance(blk_all, dict):
                    blk_pos = blk_all.get("ret pos", None) or blk_all.get("ret_pos", None)
                    blk_neg = blk_all.get("ret neg", None) or blk_all.get("ret_neg", None)

                ex_pos = _extract_tail_extremes(blk_pos)
                ex_neg = _extract_tail_extremes(blk_neg)

                rows.append(
                    {
                        "Mom|H": f"{mom}|{hz}",
                        r"Hi$>$Mid ($|Ret^+|$)": pstar(ex_pos["high_gt_mid"]),
                        r"Lo$>$Mid ($|Ret^+|$)": pstar(ex_pos["low_gt_mid"]),
                        r"Hi$>$Mid ($|Ret^-|$)": pstar(ex_neg["high_gt_mid"]),
                        r"Lo$>$Mid ($|Ret^-|$)": pstar(ex_neg["low_gt_mid"]),
                        "__moment_full__": moment_full,
                        "__horizon__": str(hz),
                    }
                )
        return rows

    def _finalize_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = _ensure_cols(df, cols + ["__moment_full__", "__horizon__"])
        df = _sort_by_momh(df, momh_col="Mom|H")
        df = _insert_moment_separators(
            df,
            cols + ["__moment_full__", "__horizon__"],
            momh_col="Mom|H",
            marker="__MIDRULE__",
        )
        return df.reset_index(drop=True)

    def _caption_everywhere_only(desc: str) -> str:
        return (
            desc
            + " Reported p-values correspond to the uniform significance test, which evaluates "
            + r"$H_0:\exists\tau\ \text{ with }\beta(\tau)=0$ vs $H_1:\beta(\tau)\neq 0\ \forall\tau$."
        )

    tables: Dict[str, Dict[str, str]] = {}

    def _emit(spec: LatexTableSpec, df: pd.DataFrame):
        latex = _df_to_booktabs_latex_with_title_and_caption_below(
            df.drop(columns=["__moment_full__", "__horizon__"]),
            title=spec.title,
            caption=spec.caption,
            label=spec.label,
            escape=False,
            resize_to_textwidth=True,
            midrule_marker="__MIDRULE__",
        )

        tables[spec.label] = {
            "latex": latex,
            "filename": spec.filename,
            "title": spec.title,
            "caption": spec.caption,
            "label": spec.label,
        }

        if base_path is not None:
            p_flat = base_path / "_all_tables"
            p_flat.mkdir(parents=True, exist_ok=True)
            (p_flat / spec.filename).write_text(latex, encoding="utf-8")

    # H1 + H2 merged
    cols_cont = [
        "Mom|H",
        COL_EPOS_ALL, COL_EPOS_LOW, COL_EPOS_MID, COL_EPOS_HIGH,
        COL_ENEG_ALL, COL_ENEG_LOW, COL_ENEG_MID, COL_ENEG_HIGH,
    ]
    df = _finalize_df(
        pd.DataFrame(
            _rows_merged_everywhere(
                [KEY_RET_POS_NEW, KEY_RET_POS_OLD],
                [KEY_RET_NEG_NEW, KEY_RET_NEG_OLD],
            )
        ),
        cols_cont,
    )
    _emit(
        LatexTableSpec(
            title="Uniform significance of contemporaneous positive and negative returns",
            caption=(
                r"Uniform significance tests for contemporaneous positive and negative returns "
                r"($Ret^+$ and $Ret^-$). Reported p-values correspond to "
                r"$H_0:\exists\tau\ \text{ with }\beta(\tau)=0$ vs "
                r"$H_1:\beta(\tau)\neq 0\ \forall\tau$."
            ),
            label=f"tab:{file_prefix}_H1H2_contemporaneous",
            filename=f"{file_prefix}__H1H2_contemporaneous.tex",
        ),
        df,
    )

    # H3 + H4 merged
    df = _finalize_df(
        pd.DataFrame(_rows_merged_lagged(int(lag_for_pos), int(lag_for_neg))),
        cols_cont,
    )
    _emit(
        LatexTableSpec(
            title=f"Uniform significance of lagged positive and negative returns (lag {lag_for_pos}/{lag_for_neg})",
            caption=(
                rf"Uniform significance tests for lagged positive and negative returns "
                rf"($Ret^+_{{t-{lag_for_pos}}}$ and $Ret^-_{{t-{lag_for_neg}}}$). "
                r"Reported p-values correspond to "
                r"$H_0:\exists\tau\ \text{ with }\beta(\tau)=0$ vs "
                r"$H_1:\beta(\tau)\neq 0\ \forall\tau$."
            ),
            label=f"tab:{file_prefix}_H3H4_lagged",
            filename=f"{file_prefix}__H3H4_lagged.tex",
        ),
        df,
    )

    # H5 merged dominance
    cols_dom = [
        "Mom|H",
        COL_UPOS_ALL, COL_UPOS_LOW, COL_UPOS_MID, COL_UPOS_HIGH,
        COL_UNEG_ALL, COL_UNEG_LOW, COL_UNEG_MID, COL_UNEG_HIGH,
    ]
    df = _finalize_df(
        pd.DataFrame(
            _rows_merged_uniform_dominance(
                [KEY_DOM_POS_NEW, KEY_DOM_POS_OLD],
                [KEY_DOM_NEG_NEW, KEY_DOM_NEG_OLD],
            )
        ),
        cols_dom,
    )
    _emit(
        LatexTableSpec(
            title="Uniform dominance of contemporaneous over lag 1 returns",
            caption=(
                r"Uniform dominance tests for contemporaneous over lag 1 returns, reported separately for "
                r"positive and negative return components. For each sign, "
                r"$d(\tau)=|Ret_t(\tau)|-|Ret_{t-1}(\tau)|$, and the reported p-values correspond to "
                r"$H_0:\exists\tau\ \text{ with } d(\tau)\le 0$ vs $H_1:d(\tau)>0\ \forall\tau$."
            ),
            label=f"tab:{file_prefix}_H5_dominance",
            filename=f"{file_prefix}__H5_dominance.tex",
        ),
        df,
    )

    # H6
    dep_keys = list(KEY_DEP_CANDIDATES_NEW) + [KEY_DEP_L1_OLD]
    cols_single = [
        "Mom|H",
        r"$p_{\mathrm{Every}}$(All)",
        r"$p_{\mathrm{Every}}$(Low)",
        r"$p_{\mathrm{Every}}$(Mid)",
        r"$p_{\mathrm{Every}}$(High)",
    ]
    df = _finalize_df(
        pd.DataFrame(_rows_from_testkeys_everywhere(dep_keys)),
        cols_single,
    )
    _emit(
        LatexTableSpec(
            title="Uniform significance of own-lag dependence",
            caption=_caption_everywhere_only(
                r"Uniform significance tests for own-lag dependence (lagged $\Delta Moment$ term; naming may vary across implementations)."
            ),
            label=f"tab:{file_prefix}_H6_dep_lag",
            filename=f"{file_prefix}__H6_dep_lag.tex",
        ),
        df,
    )

    # H7
    df = _finalize_df(
        pd.DataFrame(_rows_from_testkeys_everywhere([KEY_ASYM_NEW, KEY_ASYM_OLD])),
        cols_single,
    )
    _emit(
        LatexTableSpec(
            title="Uniform significance of asymmetry between positive and negative returns",
            caption=_caption_everywhere_only(
                r"Asymmetry test using $g(\tau)=|Ret^+(\tau)|-|Ret^-(\tau)|$."
            ),
            label=f"tab:{file_prefix}_H7_abs_pos_minus_abs_neg",
            filename=f"{file_prefix}__H7_abs_pos_minus_abs_neg.tex",
        ),
        df,
    )

    # H8 unchanged
    tail_cols = [
        "Mom|H",
        r"Hi$>$Mid ($|Ret^+|$)",
        r"Lo$>$Mid ($|Ret^+|$)",
        r"Hi$>$Mid ($|Ret^-|$)",
        r"Lo$>$Mid ($|Ret^-|$)",
    ]
    df = _finalize_df(pd.DataFrame(_rows_tail_extremes_combined()), tail_cols)
    _emit(
        LatexTableSpec(
            title="Tail extremeness relative to the middle segment",
            caption=(
                r"Tail extremeness vs. middle segment (one-sided) for contemporaneous returns. "
                r"For each sign, the test compares the tail segment average magnitude to the middle segment average magnitude: "
                r"$H_0:\Delta\le 0$ vs $H_1:\Delta>0$, reported separately for upper (Hi$>$Mid) and lower (Lo$>$Mid) tails."
            ),
            label=f"tab:{file_prefix}_H8_tail_ext_combined",
            filename=f"{file_prefix}__H8_tail_ext_combined.tex",
        ),
        df,
    )

    if base_path is not None:
        all_tables_dir = base_path / "_all_tables"
        all_tables_dir.mkdir(parents=True, exist_ok=True)

        idx = base_path / f"{file_prefix}__INDEX.tex"
        lines = [
            "% Auto-generated table index",
            "% Requires LaTeX packages: booktabs, graphicx, caption",
            "% Captions are placed below via caption* (caption package).",
            "",
        ]
        for _k, pack in tables.items():
            lines.append(rf"\input{{_all_tables/{pack['filename']}}}")
            lines.append("")

        idx.write_text("\n".join(lines), encoding="utf-8")
        tables["__index__"] = {
            "latex": idx.read_text(encoding="utf-8"),
            "filename": idx.name,
            "title": "Index",
            "caption": "Index",
            "label": "",
        }

    return tables
###Plot
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, List, Literal


import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Union, List, Literal, Optional, Tuple


def plot_return_and_option_moments(
    returns_df: pd.DataFrame,
    moments_dict: dict,
    *,
    return_col: str = "ret_1",
    date_col: str = "date",
    horizons: Union[int, List[int]] = 7,
    moment_type: Literal["rnd", "phys", "both"] = "rnd",
    truncate_to_moments: bool = True,
    truncate_mode: Literal["intersection", "union", "per_horizon"] = "intersection",
    # plotting
    figsize: Tuple[float, float] = (14, 10),
    title: Optional[str] = None,
    linewidth: float = 1.2,
    alpha: float = 0.9,
    # saving
    save: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    bbox_inches: str = "tight",
    show: bool = True,
):
    """
    Plot returns and option-implied moments for selected horizons, with optional truncation
    to the available option-implied moment date ranges, and optional saving.

    Assumes moments_dict[horizon] contains pandas Series with DateTimeIndex (or coercible),
    with keys like:
        rnd_vol_ann, phys_vol_ann, rnd_skew, phys_skew, rnd_kurt, phys_kurt

    Parameters
    ----------
    returns_df : pd.DataFrame
        Must contain date_col and return_col.
    moments_dict : dict
        moments_dict[h] -> dict of series.
    horizons : int or list[int]
        One or more horizons to plot.
    moment_type : {"rnd","phys","both"}
        Which moments to plot.
    truncate_to_moments : bool
        If True, truncate ALL panels (including returns) to the dates supported by moments.
    truncate_mode : {"intersection","union","per_horizon"}
        - "intersection": keep only dates common to all selected horizons *and* selected moment series
                          (most strict; best for clean comparisons across horizons).
        - "union": keep dates covered by at least one selected horizon/series (less strict).
        - "per_horizon": truncate returns separately per horizon when plotting (returns panel uses union),
                          while moment panels plot each series on its own support (least strict).
                          (Note: returns panel will be union-truncated, moments naturally have their own dates.)
    save : str or Path or None
        If provided, saves the figure to this path.
    show : bool
        If True, calls plt.show(). If False, closes figure after saving (useful in scripts).
    """

    # -------------------- inputs --------------------
    if isinstance(horizons, int):
        horizons = [horizons]
    horizons = list(horizons)

    for h in horizons:
        if h not in moments_dict:
            raise ValueError(f"Horizon {h} not found in moments_dict")

    df = returns_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # -------------------- helper: pull relevant series --------------------
    key_map = {
        "var": {"rnd": "rnd_vol_ann", "phys": "phys_vol_ann"},
        "skew": {"rnd": "rnd_skew", "phys": "phys_skew"},
        "kurt": {"rnd": "rnd_kurt", "phys": "phys_kurt"},
    }

    def _coerce_series(s: pd.Series) -> pd.Series:
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    # collect series that will be plotted
    series_by_h = {h: {} for h in horizons}
    for h in horizons:
        block = moments_dict[h]
        for moment in ("var", "skew", "kurt"):
            if moment_type in ("rnd", "both"):
                k = key_map[moment]["rnd"]
                if k in block and block[k] is not None:
                    series_by_h[h][f"rnd_{moment}"] = _coerce_series(block[k])
            if moment_type in ("phys", "both"):
                k = key_map[moment]["phys"]
                if k in block and block[k] is not None:
                    series_by_h[h][f"phys_{moment}"] = _coerce_series(block[k])

    # sanity: ensure we have at least one moment series
    any_series = any(len(d) > 0 for d in series_by_h.values())
    if not any_series:
        raise ValueError("No moment series found to plot for the requested moment_type/horizons.")

    # -------------------- truncation --------------------
    # Determine date support across all series we will plot
    supports = []
    for h, d in series_by_h.items():
        for _, s in d.items():
            if len(s) > 0:
                supports.append((s.index.min(), s.index.max()))

    if truncate_to_moments and supports:
        mins = [a for a, b in supports]
        maxs = [b for a, b in supports]

        if truncate_mode == "intersection":
            t0 = max(mins)
            t1 = min(maxs)
            if t0 > t1:
                raise ValueError(
                    "Truncation 'intersection' produced an empty window. "
                    "Try truncate_mode='union' or reduce horizons/moment_type."
                )
            df_plot = df.loc[t0:t1]

        elif truncate_mode == "union":
            t0 = min(mins)
            t1 = max(maxs)
            df_plot = df.loc[t0:t1]

        elif truncate_mode == "per_horizon":
            # returns panel uses union window; each moment series plotted on its own dates
            t0 = min(mins)
            t1 = max(maxs)
            df_plot = df.loc[t0:t1]

        else:
            raise ValueError("truncate_mode must be one of {'intersection','union','per_horizon'}")
    else:
        df_plot = df

    # -------------------- plotting --------------------
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    ax_ret, ax_var, ax_skew, ax_kurt = axes

    # returns
    ax_ret.plot(
        df_plot.index,
        df_plot[return_col],
        label="Return",
        linewidth=linewidth,
        alpha=alpha,
    )
    ax_ret.set_title("Return")
    ax_ret.legend()

    # moments panels
    def _plot_moment(ax, moment: Literal["var", "skew", "kurt"], title_str: str):
        for h in horizons:
            d = series_by_h[h]

            # per_horizon mode: do not reindex to df_plot; use native index
            # other modes: align to df_plot index for consistent x-axis
            if truncate_mode == "per_horizon":
                idx = None
            else:
                idx = df_plot.index

            # rnd
            if moment_type in ("rnd", "both"):
                key = f"rnd_{moment}"
                if key in d:
                    s = d[key]
                    if truncate_to_moments and truncate_mode in ("intersection", "union"):
                        s = s.loc[df_plot.index.min() : df_plot.index.max()]
                    if idx is not None:
                        s = s.reindex(idx)
                    ax.plot(
                        s.index,
                        s.values,
                        label=f"RND {title_str} (h={h})",
                        linewidth=linewidth,
                        alpha=alpha,
                    )

            # phys
            if moment_type in ("phys", "both"):
                key = f"phys_{moment}"
                if key in d:
                    s = d[key]
                    if truncate_to_moments and truncate_mode in ("intersection", "union"):
                        s = s.loc[df_plot.index.min() : df_plot.index.max()]
                    if idx is not None:
                        s = s.reindex(idx)
                    ax.plot(
                        s.index,
                        s.values,
                        linestyle="--",
                        label=f"Phys {title_str} (h={h})",
                        linewidth=linewidth,
                        alpha=alpha,
                    )

        ax.set_title(title_str)
        ax.legend()

    _plot_moment(ax_var, "var", "Volatility")
    _plot_moment(ax_skew, "skew", "Skewness")
    _plot_moment(ax_kurt, "kurt", "Kurtosis")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    # -------------------- save/show --------------------
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches=bbox_inches)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes