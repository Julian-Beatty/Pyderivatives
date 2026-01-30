from __future__ import annotations
import numpy as np
from .config import BootstrapSpec  # <- adjust if your module is config.py

def _block_idx(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    L = max(1, int(block_len))
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, max(1, n - L + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
    return idx.astype(int)

def bootstrap_theta(obs_list: list[dict], *, fit_once, boot: BootstrapSpec) -> dict:
    if not boot.enabled:
        return {}

    rng = np.random.default_rng(boot.random_state)
    draws = []
    succ = 0

    for _ in range(int(boot.n_boot)):
        idx = _block_idx(len(obs_list), boot.block_len, rng)
        obs_b = [obs_list[i] for i in idx]
        r = fit_once(obs_b)
        if r.get("success", True) and r.get("theta_hat", None) is not None:
            draws.append(np.asarray(r["theta_hat"], float))
            succ += 1

    if len(draws) == 0:
        return {"boot_successes": succ, "boot_failures": int(boot.n_boot) - succ}

    A = np.vstack(draws)
    lo, hi = boot.ci_levels
    out = {
        "boot_successes": succ,
        "boot_failures": int(boot.n_boot) - succ,
        "theta_ci_low": np.quantile(A, lo, axis=0),
        "theta_ci_high": np.quantile(A, hi, axis=0),
    }
    if getattr(boot, "keep_draws", False):
        out["theta_boot_draws"] = A
    return out
