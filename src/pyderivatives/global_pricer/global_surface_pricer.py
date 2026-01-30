from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .data import CallSurfaceDay
from .registry import get_model

from .postprocess.checks import (
    reprice_observed_points,
    error_summary,
    error_by_maturity,
)

from .postprocess.rnd import (
    SafetyClipConfig,
    breeden_litzenberger_pdf,
    apply_safety_clip_surface, strike_rnd_to_return_density
)

from .postprocess.iv import (
    IVConfig,
    iv_surface_from_calls,
    atm_summary_from_iv_surface,
    iv_surface_to_delta_surfaces,
)

from .postprocess.cdf import (
    CDFConfig,
    cdf_from_pdf_surface,
)

from .postprocess.moments import logreturn_moments_table, MomentsConfig


BoundsDict = Tuple[Dict[str, float], Dict[str, float]]


@dataclass
class FitState:
    model_name: str
    params: Dict[str, float]
    success: bool
    fit_meta: Dict[str, Any]


class GlobalSurfacePricer:
    """
    Two-step workflow:
      1) fit(day, ...)  -> stores fitted params in self.state_
      2) price(day, ...) -> evaluates C_fit on requested grids, plus optional post-processing
    """

    def __init__(self, model_name: str, *, Umax: float = 500.0, n_quad: int = 500):
        self.model_name = str(model_name)
        self.Umax = float(Umax)
        self.n_quad = int(n_quad)

        self.model_ = None          # instantiated ModelCls(S0,r,q,...) after fit
        self.state_: Optional[FitState] = None
        self.day_meta_: Optional[Dict[str, float]] = None  # S0, r, q used in fit

    # -----------------------
    # internal helpers
    # -----------------------
    @staticmethod
    def _normalize_params(p_hat) -> Dict[str, float]:
        if hasattr(p_hat, "to_dict"):
            d = p_hat.to_dict()
        elif hasattr(p_hat, "__dict__"):
            d = dict(p_hat.__dict__)
        else:
            d = dict(p_hat)
        return {str(k): float(v) for k, v in d.items()}

    @staticmethod
    def _build_T_grid(day: CallSurfaceDay, T_grid: Optional[np.ndarray]) -> np.ndarray:
        if T_grid is None:
            T_grid = np.unique(day.T_obs)
        T_grid = np.asarray(T_grid, float).ravel()
        T_grid = T_grid[np.isfinite(T_grid) & (T_grid > 0)]
        T_grid.sort()
        return T_grid

    @staticmethod
    def _build_K_grid(
        day: CallSurfaceDay,
        *,
        K_grid: Optional[np.ndarray],
        grid_mode: str,
        K_grid_n: int,
        m_grid: Optional[np.ndarray],
        m_bounds: tuple[float, float],
        m_grid_n: int,
    ) -> np.ndarray:
        if K_grid is None:
            if grid_mode == "strike":
                kmin, kmax = float(np.min(day.K_obs)), float(np.max(day.K_obs))
                K_grid = np.linspace(max(1e-12, kmin), kmax, int(K_grid_n))
            elif grid_mode == "moneyness":
                S0 = float(day.S0)
                if m_grid is None:
                    m_lo, m_hi = m_bounds
                    m_grid = np.linspace(float(m_lo), float(m_hi), int(m_grid_n))
                else:
                    m_grid = np.asarray(m_grid, float).ravel()
                m_grid = m_grid[np.isfinite(m_grid) & (m_grid > 0)]
                m_grid.sort()
                K_grid = S0 * m_grid
            else:
                raise ValueError("grid_mode must be 'strike' or 'moneyness'.")

        K_grid = np.asarray(K_grid, float).ravel()
        K_grid = K_grid[np.isfinite(K_grid) & (K_grid > 0)]
        K_grid.sort()
        return K_grid

    # -----------------------
    # public API
    # -----------------------
    def fit(
        self,
        day: CallSurfaceDay,
        *,
        x0=None,
        bounds=None,
        max_nfev: int = 250,
        q_override: Optional[float] = None,
    ) -> FitState:
        """
        Fit model parameters to observed (K_obs, T_obs, C_obs).
        Stores fitted params and model instance on self.
        """
        q_use = float(day.q) if q_override is None else float(q_override)

        ModelCls = get_model(self.model_name)
        model = ModelCls(S0=float(day.S0), r=float(day.r), q=q_use, Umax=self.Umax, n_quad=self.n_quad)

        fitres = model.fit(
            K_obs=day.K_obs,
            T_obs=day.T_obs,
            C_obs=day.C_obs,
            x0=x0,
            bounds=bounds,
            max_nfev=int(max_nfev),
        )

        params_dict = self._normalize_params(fitres.params)
        success = bool(getattr(fitres, "success", True))

        state = FitState(
            model_name=self.model_name,
            params=params_dict,
            success=success,
            fit_meta={
                "Umax": self.Umax,
                "n_quad": self.n_quad,
                "max_nfev": int(max_nfev),
                "n_obs": int(np.asarray(day.C_obs).size),
            },
        )

        self.model_ = model
        self.state_ = state
        self.day_meta_ = {"S0": float(day.S0), "r": float(day.r), "q": float(q_use)}
        return state

    def price(
        self,
        day: CallSurfaceDay,
        *,
        # grid controls
        K_grid: Optional[np.ndarray] = None,
        T_grid: Optional[np.ndarray] = None,
        K_grid_n: int = 200,
        grid_mode: str = "strike",      # "strike" | "moneyness"
        m_grid: Optional[np.ndarray] = None,
        m_bounds: tuple[float, float] = (0.5, 1.5),
        m_grid_n: int = 200,
        # outputs
        compute_rnd: bool = False,
        safety_clip: Optional[SafetyClipConfig] = None,
        compute_iv: bool = False,
        compute_cdf: bool = False,
        compute_moments: bool = False,
        compute_obs_reprice: bool = True,
        moments_cfg: Optional[MomentsConfig] = None,
        compute_delta: bool = False,
        iv_cfg: Optional[IVConfig] = None,
        cdf_cfg: Optional[CDFConfig] = None,
        # overrides
        params_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Price on a rectangular (K_grid x T_grid) using stored fitted params by default.
        """
        if self.model_ is None or self.state_ is None:
            raise RuntimeError("Call fit(...) before price(...).")

        # grids
        T_grid = self._build_T_grid(day, T_grid)
        K_grid = self._build_K_grid(
            day,
            K_grid=K_grid,
            grid_mode=grid_mode,
            K_grid_n=K_grid_n,
            m_grid=m_grid,
            m_bounds=m_bounds,
            m_grid_n=m_grid_n,
        )

        # parameters
        params = self.state_.params if params_override is None else {str(k): float(v) for k, v in params_override.items()}

        # compute fitted call surface
        C_fit = self.model_.price_surface(K_grid, T_grid, params)

        # assemble output (mirrors your old dict)
        q_use = float(self.day_meta_["q"]) if self.day_meta_ else float(day.q)

        out: Dict[str, Any] = {
            "model": self.model_name,
            "success": bool(self.state_.success),
            "params": dict(params),
            "S0": float(day.S0),
            "r": float(day.r),
            "q": float(q_use),
            "K_grid": K_grid,
            "T_grid": T_grid,
            "C_fit": np.asarray(C_fit, float),
            "meta": dict(self.state_.fit_meta),
        }

        # ========= RND via BL (+ safety clip) =========
        if compute_rnd:
            rnd_raw = breeden_litzenberger_pdf(
                out["C_fit"],
                K_grid=K_grid,
                T_grid=T_grid,
                r=float(day.r),
                floor=1e-12,
            )

            cfg = safety_clip if safety_clip is not None else SafetyClipConfig(enabled=False)
            rnd_clip, clip_info = apply_safety_clip_surface(
                rnd_raw,
                K_grid=K_grid,
                S0=float(day.S0),
                cfg=cfg,
            )

            out["rnd_surface"] = np.asarray(rnd_clip, float)  # PRIMARY (clipped)
            out["safety_clip"] = {
                "enabled": bool(cfg.enabled),
                "any_used": bool(any(d.get("used", False) for d in clip_info)),
                "per_row": clip_info,
            }
            
            # --- log-return RND surface (same rows as T_grid) ---
            r_grid, rnd_lr = strike_rnd_to_return_density(
                out["rnd_surface"],      # use clipped surface
                K_grid=K_grid,
                S0=float(day.S0),
                return_type="log",
                normalize=True,
            )

            out["rnd_lr_grid"] = np.asarray(r_grid, float)           # x-axis for lr density
            out["rnd_lr_surface"] = np.asarray(rnd_lr, float)        # q_r(r) surface

            if compute_moments:
                cfg_m = moments_cfg if moments_cfg is not None else MomentsConfig(
                    renormalize=True,
                    clip_negative=True,
                )
                out["rnd_moments_table"] = logreturn_moments_table(
                    out["rnd_surface"],
                    K_grid=out["K_grid"],
                    T_grid=out["T_grid"],
                    S0=float(day.S0),
                    cfg=cfg_m,
                )

        # ========= IV surface =========
        if compute_iv:
            cfg_iv = iv_cfg if iv_cfg is not None else IVConfig()
            iv_surf = iv_surface_from_calls(
                out["C_fit"],
                K_grid=K_grid,
                T_grid=T_grid,
                S0=float(day.S0),
                r=float(day.r),
                q=float(q_use),
                cfg=cfg_iv,
            )
            out["iv_surface"] = np.asarray(iv_surf, float)
            out.update(atm_summary_from_iv_surface(iv_surf, K_grid=K_grid, T_grid=T_grid, S0=float(day.S0)))

            if compute_delta:
                out["delta_dict"] = iv_surface_to_delta_surfaces(
                    iv_surface=iv_surf,
                    K_grid=K_grid,
                    T_grid=T_grid,
                    S0=float(day.S0),
                    r=float(day.r),
                )

        # ========= CDF surface (use CLIPPED RND) =========
        if compute_cdf:
            if "rnd_surface" not in out:
                raise ValueError("compute_cdf=True requires compute_rnd=True (CDF needs an RND surface).")
            cfg_cdf = cdf_cfg if cdf_cfg is not None else CDFConfig()
            out["cdf_surface"] = cdf_from_pdf_surface(
                out["rnd_surface"],
                K_grid=K_grid,
                cfg=cfg_cdf,
            )

        # ========= repricing diagnostics on observed points =========
        if compute_obs_reprice:
            C_hat_obs = reprice_observed_points(self.model_, day.K_obs, day.T_obs, params)
            out["C_hat_obs"] = np.asarray(C_hat_obs, float)
            out["errors"] = error_summary(day.C_obs, C_hat_obs)
            out["errors_by_T"] = error_by_maturity(day.K_obs, day.T_obs, day.C_obs, C_hat_obs)

        return out

    def fit_and_price(self, day: CallSurfaceDay, **kwargs) -> Dict[str, Any]:
        """
        Convenience wrapper to preserve your old API.
        Anything fit-related still needs to be passed under the same names.
        """
        # split kwargs into fit vs price buckets (simple, explicit)
        fit_keys = {"x0", "bounds", "max_nfev", "q_override"}
        fit_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in fit_keys}

        self.fit(day, **fit_kwargs)
        return self.price(day, **kwargs)
