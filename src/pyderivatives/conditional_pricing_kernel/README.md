# conditional_pricing_kernel

Small package to fit a *global* projected pricing-kernel parameter surface `theta(T)` from a
dictionary of per-date RND surfaces, and then evaluate physical densities on a chosen anchor date.

## Key features
- GLOBAL theta(T) fit (one theta per maturity slice)
- Optional disk caching of fitted theta_master
- Optional moving-block bootstrap CIs for theta(T)
- **Optional post-processing safety clip for physical density** (after estimation):
  enforce non-increasing tails away from the mode (left and right), per maturity slice,
  then renormalize.
- The fitter prints progress ("talks") after fitting each `theta(T)`; you can also pass a callback.

## Minimal usage

```python
from conditional_pricing_kernel import (
    ThetaSpec, BootstrapSpec, EvalSpec, CacheSpec, SafetyClipSpec,
    estimate_pricing_kernel_global, evaluate_anchor_surfaces_with_theta_master,
)

# Inputs
# result_dict: dict[str(date)] -> dict with keys like "T_grid","K_grid","rnd_surface","S0","atm_vol","r"
# stock_df: DataFrame with DatetimeIndex (or "date" column) and a spot column (default "Close")

pk_fit = estimate_pricing_kernel_global(
    result_dict,
    stock_df,
    spot_col="Close",
    dataset_tag="moderna",
    theta_spec=ThetaSpec(N=2, Ksig=1),
    bootstrap=BootstrapSpec(enabled=False),
    cache=CacheSpec(use_disk=True, folder="pk_cache"),
    maxiter=400,
    min_obs_per_T=30,
)

anchor_key = sorted(result_dict.keys())[0]
anchor_out = evaluate_anchor_surfaces_with_theta_master(
    result_dict[anchor_key],
    theta_master=pk_fit["theta_master"],
    theta_spec=ThetaSpec(**pk_fit["theta_spec"]),
    eval_spec=EvalSpec(),
    safety_clip=SafetyClipSpec(enabled=True, floor=0.0),
)

pR = anchor_out["anchor_surfaces"]["pR_surface"]       # clipped if enabled
mom = anchor_out["physical_moments_table"]
```

## Directory structure

```
conditional_pricing_kernel/
  __init__.py
  cache.py
  config.py
  data.py
  eval.py
  fit.py
  kernel.py
  moments.py
```
