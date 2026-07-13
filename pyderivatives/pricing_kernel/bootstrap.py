from __future__ import annotations

from typing import Dict
import copy

import numpy as np

from .config import BootstrapSpec
from .utils import _as_1d, _block_indices_circular


class BootstrapMixin:
    """
    Mixin for block-bootstrap uncertainty around transformed density outputs.
    """

    def _bootstrap_transform_info(
        self,
        info: dict,
        bootstrap_spec: BootstrapSpec,
    ) -> dict:
        """
        Block-bootstrap fitted maturity models and re-transform the same RND.
        """
        if not bootstrap_spec.enabled:
            bootstrap_spec = BootstrapSpec(
                enabled=True,
                B=bootstrap_spec.B,
                block_length=bootstrap_spec.block_length,
                ci_level=bootstrap_spec.ci_level,
                random_state=bootstrap_spec.random_state,
                keep_draws=bootstrap_spec.keep_draws,
            )

        rng = np.random.default_rng(bootstrap_spec.random_state)

        B = int(bootstrap_spec.B)
        alpha = 1.0 - float(bootstrap_spec.ci_level)
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0

        base = self._transform_info_no_bootstrap(info)
        shape = base["physical_lr_surface"].shape

        f_p_draws = np.full((B, *shape), np.nan)
        F_p_draws = np.full((B, *shape), np.nan)
        kernel_draws = np.full((B, *shape), np.nan)
        rra_draws = np.full((B, *shape), np.nan)

        boot_success = 0
        boot_fail = 0

        for b in range(B):
            boot_model = copy.deepcopy(self)
            boot_model.models_by_T_ = {}
            boot_model.fit_diagnostics_ = {}

            for T, hist_T in self.fit_history_by_T_.items():
                if len(hist_T) < self.min_obs:
                    continue

                idx = _block_indices_circular(
                    len(hist_T),
                    bootstrap_spec.block_length,
                    rng,
                )

                hist_b = hist_T.iloc[idx].reset_index(drop=True)

                try:
                    fitted_b, _ = boot_model._fit_one_maturity(hist_b, T=float(T))
                    boot_model.models_by_T_[float(T)] = fitted_b
                except Exception:
                    continue

            boot_model.is_fitted_ = True

            try:
                out_b = boot_model._transform_info_no_bootstrap(info)
                f_p_draws[b] = out_b["physical_lr_surface"]
                F_p_draws[b] = out_b["physical_cdf_lr_surface"]
                kernel_draws[b] = out_b["pricing_kernel_surface"]
                rra_draws[b] = out_b["relative_risk_aversion_surface"]
                boot_success += 1
            except Exception:
                boot_fail += 1

        def ci(A):
            return {
                "lower": np.nanquantile(A, q_lo, axis=0),
                "upper": np.nanquantile(A, q_hi, axis=0),
            }

        boot_out = {
            "enabled": True,
            "B": B,
            "block_length": int(bootstrap_spec.block_length),
            "ci_level": float(bootstrap_spec.ci_level),
            "successes": int(boot_success),
            "failures": int(boot_fail),
            "physical_lr_surface_ci": ci(f_p_draws),
            "physical_cdf_lr_surface_ci": ci(F_p_draws),
            "pricing_kernel_surface_ci": ci(kernel_draws),
            "relative_risk_aversion_surface_ci": ci(rra_draws),
        }

        if bootstrap_spec.keep_draws:
            boot_out["draws"] = {
                "physical_lr_surface": f_p_draws,
                "physical_cdf_lr_surface": F_p_draws,
                "pricing_kernel_surface": kernel_draws,
                "relative_risk_aversion_surface": rra_draws,
            }

        return boot_out