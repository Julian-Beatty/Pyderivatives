from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from typing import Tuple, Optional

from scipy.optimize import minimize

from .cache import make_key, load, save
from .config import ThetaSpec, BootstrapSpec, EvalSpec, CacheSpec
from .data import build_obs_by_maturity
from .kernel import g_r_sigma
from .eval import evaluate_anchor_surfaces_with_theta_master
from .moments import physical_moments_table
from .bootstrap import bootstrap_theta


@dataclass
class ThetaFit:
    theta_hat: np.ndarray
    success: bool
    fun: float
    n_obs: int
    message: str
    theta_ci_low: Optional[np.ndarray] = None
    theta_ci_high: Optional[np.ndarray] = None


def _bounds(theta_spec: ThetaSpec, p: int):
    if theta_spec.bounds is None:
        return None
    lb, ub = theta_spec.bounds
    lb = np.asarray(lb, float).ravel()
    ub = np.asarray(ub, float).ravel()
    if lb.size != p or ub.size != p:
        raise ValueError("ThetaSpec.bounds must be (lb, ub) each length p.")
    return list(zip(lb, ub))


def _log_phys_density_at_R(
    *,
    R_real: float,
    K_grid: np.ndarray,
    qK_row: np.ndarray,
    S0: float,
    sigma: float,
    theta: np.ndarray,
    theta_spec: ThetaSpec,
    r_grid: np.ndarray,
) -> float:
    """
    Constructs f_Q(r) from q(K) and computes f_P at realized R under
      f_P(r) ∝ f_Q(r) * exp(-g(r; sigma))
    with normalization over r_grid.

    NOTE: This matches your old “mass trick” style normalization.
    """
    # Build r(K)=log(K/S0)
    K = np.asarray(K_grid, float).ravel()
    qK = np.asarray(qK_row, float).ravel()
    mask = np.isfinite(K) & np.isfinite(qK) & (K > 0)
    K = K[mask]
    qK = qK[mask]
    if K.size < 10:
        return -np.inf

    rK = np.log(K / float(S0))

    # q_R(R) = q_K(K) * dK/dR = qK * S0 ; then f_Q(r)=q_R(R)*R
    R = np.exp(rK)
    fQ = (qK * float(S0)) * R

    # interpolate fQ onto r_grid
    fQ_grid = np.interp(r_grid, rK, fQ, left=0.0, right=0.0)

    g = g_r_sigma(r_grid, float(sigma), theta, N=theta_spec.N, Ksig=theta_spec.Ksig)
    w = fQ_grid * np.exp(-g)

    mass = np.trapezoid(w, r_grid)
    if not np.isfinite(mass) or mass <= 0:
        return -np.inf

    fP_grid = w / mass

    r_real = float(R_real)
    fP_at = np.interp(r_real, r_grid, fP_grid, left=0.0, right=0.0)
    if not np.isfinite(fP_at) or fP_at <= 0:
        return -np.inf

    return float(np.log(fP_at))


def _fit_theta_for_one_maturity(
    obs_list: List[dict],
    theta_spec: ThetaSpec,
    *,
    x0: Optional[np.ndarray] = None,
    maxiter: int = 400,
    r_bounds=(-1.0, 1.0),
    r_grid_size=350,
) -> ThetaFit:
    N, Ksig = int(theta_spec.N), int(theta_spec.Ksig)
    p = N * (Ksig + 1)
    if x0 is None:
        x0 = np.zeros(p, float)

    bounds = _bounds(theta_spec, p)
    r_grid = np.linspace(float(r_bounds[0]), float(r_bounds[1]), int(r_grid_size))

    def nll(x):
        ll = 0.0
        theta = np.asarray(x, float)

        for t, ob in enumerate(obs_list):
            #t = 0
            ob = obs_list[t]
            
            K = np.asarray(ob["K_grid"], float).ravel()
            qK = np.asarray(ob["qK_row"], float).ravel()
            
            # print("K sorted strictly increasing?", bool(np.all(np.diff(K) > 0)))
            # print("K sorted nondecreasing?", bool(np.all(np.diff(K) >= 0)))
            # print("min diff(K):", float(np.min(np.diff(np.sort(K)))))
            # print("any duplicates?", bool(np.any(np.diff(np.sort(K)) == 0)))
            
            # print("qK min:", float(np.nanmin(qK)))
            # print("qK has negatives?", bool(np.any(qK < 0)))
            # print("this Ran")
            ll_i = _log_phys_density_at_R(
                R_real=ob["R_real"],
                K_grid=ob["K_grid"],
                qK_row=ob["qK_row"],
                S0=ob["S0"],
                sigma=ob["sigma"],
                theta=theta,
                theta_spec=theta_spec,
                r_grid=r_grid,
            )
            if not np.isfinite(ll_i):
                return 1e100   # ← PENALTY, do NOT raise
            # if not np.isfinite(ll_i):
            #     R_real = float(ob["R_real"])
            #     gross = float(np.exp(R_real))
            
            #     raise RuntimeError(
            #         "\n[PK DIAGNOSTIC FAILURE]\n"
            #         f"  obs index t = {t}\n"
            #         f"  anchor_key = {ob.get('anchor_key')}\n"
            #         f"  d0 = {ob.get('anchor_date')}\n"
            #         f"  end_target = {ob.get('end_date_target')}\n"
            #         f"  end_matched = {ob.get('end_date_matched')}\n"
            #         f"  horizon_days = {ob.get('horizon_days')}\n"
            #         f"  actual_days = {(pd.Timestamp(ob.get('end_date_matched')) - pd.Timestamp(ob.get('anchor_date'))).days if ob.get('end_date_matched') is not None else 'NA'}\n"
            #         f"  R_real (log) = {R_real}\n"
            #         f"  gross = {gross}\n"
            #         f"  S0_ret(adj) = {ob.get('S0_ret')}\n"
            #         f"  S1_ret(adj) = {ob.get('S1_ret')}\n"
            #         f"  px0(raw) = {ob.get('px0')}  aj0 = {ob.get('aj0')}\n"
            #         f"  px1(raw) = {ob.get('px1')}  aj1 = {ob.get('aj1')}\n"
            #         f"  option S0 (dd['S0']) = {ob.get('S0')}\n"
            #         f"  r_grid = [{r_grid[0]}, {r_grid[-1]}]\n"
                # )
            # if not np.isfinite(ll_i):
            #     R_real = float(ob["R_real"])
            #     Kg = np.asarray(ob["K_grid"], float)
            #     S0 = float(ob["S0"])
            
            #     gross = np.exp(R_real)
            
            #     raise RuntimeError(
            #         "\n[PK DIAGNOSTIC FAILURE]\n"
            #         f"  obs index t = {t}\n"
            #         f"  R_real (log-return) = {R_real}\n"
            #         f"  exp(R_real) (gross return) = {gross}\n"
            #         f"  r_grid = [{r_grid[0]}, {r_grid[-1]}]\n"
            #         f"  R_real in grid? {r_grid[0] <= R_real <= r_grid[-1]}\n"
            #         f"  S0 = {S0}\n"
            #         f"  implied S_end = {S0 * gross}\n"
            #         f"  K_grid = [{Kg.min()}, {Kg.max()}]  (n={Kg.size})\n"
            #         f"  implied strike = {S0 * gross}\n"
            #         f"  qK_row finite? {np.all(np.isfinite(ob['qK_row']))}\n"
            #     )
            # if not np.isfinite(ll_i):
            #     R_real = float(ob["R_real"])
            #     Kg = np.asarray(ob["K_grid"], float)
            #     S0 = float(ob["S0"])

            #     # Helpful: realized return relative to evaluation grid + strike span
            #     raise RuntimeError(
            #         "Non-finite log-likelihood.\n"
            #         f"  obs index t={t}\n"
            #         f"  R_real={R_real}\n"
            #         f"  r_grid=[{float(r_grid[0])}, {float(r_grid[-1])}]\n"
            #         f"  S0={S0}\n"
            #         f"  K_grid=[{float(np.nanmin(Kg))}, {float(np.nanmax(Kg))}] (n={Kg.size})\n"
            #         f"  qK_row finite? {bool(np.all(np.isfinite(ob['qK_row'])))}\n"
            #     )

            ll += ll_i

        return -ll

    res = minimize(
        nll,
        x0=np.asarray(x0, float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter)},
    )

    return ThetaFit(
        theta_hat=np.asarray(res.x, float),
        success=bool(res.success),
        fun=float(res.fun),
        n_obs=int(len(obs_list)),
        message=str(res.message),
    )


def _block_boot_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Moving-block bootstrap indices of length n using blocks of size block_len.
    """
    if n <= 0:
        return np.array([], dtype=int)
    L = max(1, int(block_len))
    starts = rng.integers(0, max(1, n - L + 1), size=int(np.ceil(n / L)))
    idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
    return idx


def estimate_pricing_kernel_global(
    result_dict: Dict[str, dict],
    stock_df: pd.DataFrame,
    *,
    spot_col: str = "Close",
    dataset_tag: str = "default",
    theta_spec: ThetaSpec = ThetaSpec(N=2, Ksig=1),
    eval_spec: EvalSpec = EvalSpec(),
    bootstrap: BootstrapSpec = BootstrapSpec(enabled=False),
    cache: CacheSpec = CacheSpec(use_disk=True, folder="pk_cache"),
    maxiter: int = 400,
    min_obs_per_T: int = 30,
    on_theta_fit: Optional[Callable[[int, float, ThetaFit], None]] = None,
    diag_build_obs: bool = False,
    rnd_tail_alpha: Optional[Tuple[float, float]] = None,

) -> Dict[str, Any]:
    """
    GLOBAL only:
      - Fit theta(T_j) separately for each maturity slice j (Eq. 9 spec).
      - Optional block bootstrap (block_len=21).
      - Cache the fitted theta_master (and CI) to disk.
      - Return a single OUT DICTIONARY including physical_moments_table.
    """
    key = make_key(dataset_tag=dataset_tag, theta_spec=theta_spec, boot_spec=bootstrap)

    if cache.use_disk:
        hit = load(cache.folder, key)
        if hit is not None:
            return hit
    drop_log = {"counts": {}, "examples": {}} if diag_build_obs else None

    # Build per-maturity observation lists
    T_grid, obs_by_T = build_obs_by_maturity(result_dict, stock_df, spot_col=spot_col)
    
    T_grid, obs_by_T = build_obs_by_maturity(
    result_dict,
    stock_df,
    spot_col=spot_col,
    drop_log=drop_log,
    rnd_tail_alpha=rnd_tail_alpha,
)

    nT = len(obs_by_T)
    p = int(theta_spec.N) * (int(theta_spec.Ksig) + 1)

    theta_master = np.full((nT, p), np.nan, float)
    success_by_T = np.zeros(nT, bool)
    n_obs_by_T = np.zeros(nT, int)
    fun_by_T = np.full(nT, np.nan, float)

    # --- Fit each maturity slice ---
    for j in range(nT):
        obs = obs_by_T[j]
        Tj = float(T_grid[j])
    
        if len(obs) < int(min_obs_per_T):
            print(f"[PK] skip T={Tj:.4f}y  (n_obs={len(obs)} < {min_obs_per_T})")
            continue
    
        print(f"[PK] fitting theta for T={Tj:.4f}y  (n_obs={len(obs)})")
    
        fit = _fit_theta_for_one_maturity(
            obs,
            theta_spec,
            maxiter=maxiter,
            r_bounds=eval_spec.r_bounds,
            r_grid_size=eval_spec.r_grid_size,
        )
    
        theta_master[j, :] = fit.theta_hat
        success_by_T[j] = fit.success
        n_obs_by_T[j] = fit.n_obs
        fun_by_T[j] = fit.fun
    
        status = "OK" if fit.success else "FAIL"
        if on_theta_fit is not None:
            try:
                on_theta_fit(j, Tj, fit)
            except Exception as _e:
                print(f"[PK] on_theta_fit callback raised: {_e}")
        print(
            f"[PK] done  T={Tj:.4f}y  "
            f"status={status}  "
            f"fun={fit.fun:.3e}"
        )


    # --- Bootstrap CIs (per maturity slice) ---
    
    theta_ci_low = None
    theta_ci_high = None
    theta_boot_draws_by_T = None
    
    if bootstrap.enabled:
        theta_ci_low = np.full_like(theta_master, np.nan)
        theta_ci_high = np.full_like(theta_master, np.nan)
        theta_boot_draws_by_T = [None] * nT
    
        for j in range(nT):
            obs = obs_by_T[j]
            Tj = float(T_grid[j])
            if len(obs) < int(min_obs_per_T):
                continue
    
            def fit_once(obs_b):
                fit_b = _fit_theta_for_one_maturity(
                    obs_b,
                    theta_spec,
                    maxiter=maxiter,
                    r_bounds=eval_spec.r_bounds,
                    r_grid_size=eval_spec.r_grid_size,
                )
                return {"success": bool(fit_b.success), "theta_hat": fit_b.theta_hat}
    
            boot_out = bootstrap_theta(obs, fit_once=fit_once, boot=bootstrap)
    
            if "theta_ci_low" in boot_out:
                theta_ci_low[j, :] = boot_out["theta_ci_low"]
                theta_ci_high[j, :] = boot_out["theta_ci_high"]
    
            if bootstrap.keep_draws and "theta_boot_draws" in boot_out:
                theta_boot_draws_by_T[j] = boot_out["theta_boot_draws"]
    
            print(f"[PK][boot] T={Tj:.4f}y succ={boot_out.get('boot_successes',0)}/{bootstrap.n_boot}")



    # theta_ci_low = None
    # theta_ci_high = None
    # if bootstrap.enabled:
    #     rng = np.random.default_rng(bootstrap.random_state)
    #     alpha_lo, alpha_hi = bootstrap.ci_levels
    #     theta_ci_low = np.full_like(theta_master, np.nan)
    #     theta_ci_high = np.full_like(theta_master, np.nan)

    #     for j in range(nT):
    #         obs = obs_by_T[j]
    #         if len(obs) < int(min_obs_per_T):
    #             continue
    #         # bootstrap distribution of theta for slice j
    #         thetas = []
    #         n = len(obs)
    #         for _ in range(int(bootstrap.n_boot)):
    #             idx = _block_boot_indices(n, bootstrap.block_len, rng)
    #             obs_b = [obs[i] for i in idx]
    #             fit_b = _fit_theta_for_one_maturity(
    #                 obs_b,
    #                 theta_spec,
    #                 maxiter=maxiter,
    #                 r_bounds=eval_spec.r_bounds,
    #                 r_grid_size=eval_spec.r_grid_size,
    #             )
    #             if fit_b.success and np.all(np.isfinite(fit_b.theta_hat)):
    #                 thetas.append(fit_b.theta_hat)

    #         if len(thetas) >= max(10, int(0.2 * bootstrap.n_boot)):
    #             A = np.vstack(thetas)
    #             theta_ci_low[j, :] = np.quantile(A, alpha_lo, axis=0)
    #             theta_ci_high[j, :] = np.quantile(A, alpha_hi, axis=0)

    # --- Evaluate surfaces on an example anchor date? (we keep evaluation separate) ---
    # Your plots typically take an "out" per anchor date, so we return:
    #   - fitted theta surface (theta_master + CI)
    #   - plus a helper-ready "global_fit" block
    # You can evaluate any date using eval.evaluate_anchor_surfaces_with_theta_master(...)

    out = {
        "eval_spec": {"r_bounds": tuple(eval_spec.r_bounds), "r_grid_size": int(eval_spec.r_grid_size)},
        "theta_boot_draws_by_T": theta_boot_draws_by_T,
        "obs_drop_log": drop_log,


        "model": "projected_pricing_kernel_global",
        "dataset_tag": dataset_tag,
        "theta_spec": {"N": int(theta_spec.N), "Ksig": int(theta_spec.Ksig)},
        "bootstrap": {
            "enabled": bool(bootstrap.enabled),
            "n_boot": int(bootstrap.n_boot),
            "block_len": int(bootstrap.block_len),
            "ci_levels": tuple(bootstrap.ci_levels),
            "random_state": bootstrap.random_state,
        },
        "cache_key": key,
        "T_grid": np.asarray(T_grid, float),
        "theta_master": theta_master,
        "theta_success_by_T": success_by_T,
        "theta_n_obs_by_T": n_obs_by_T,
        "theta_fun_by_T": fun_by_T,
        "theta_ci_low": theta_ci_low,
        "theta_ci_high": theta_ci_high,
        # convenience: a moments table computed from theta_master at a chosen anchor date
        # (you’ll call physical_moments_table on the evaluated surfaces per date)
    }

    if cache.use_disk:
        save(cache.folder, key, out)

    return out
