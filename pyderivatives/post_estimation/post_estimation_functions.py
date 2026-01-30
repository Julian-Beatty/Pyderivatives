import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Sequence, Optional, Union, Tuple, Any
import pandas as pd
# def plot_result_dict(
#     plot_dict: Dict[str, dict],
#     maturity_list: Sequence[Union[int, float]],
#     date: Union[str, "np.datetime64", "object"],
#     *,
#     title: Optional[str] = None,
#     panel_shape: Optional[Tuple[int, int]] = None,   # e.g. (2,3); default auto-ish
#     figsize_per_panel: Tuple[float, float] = (4.0, 2.8),
#     sharex: bool = True,
#     sharey: bool = True,
#     xlim: Optional[Tuple[float, float]] = None,
#     ylim: Optional[Tuple[float, float]] = None,
#     legend: bool = True,
#     legend_loc: str = "best",
#     lw: float = 1.8,
#     alpha: float = 0.95,
#     grid: bool = True,
#     save: Optional[Union[str, Path]] = None,
#     dpi: int = 200,
# ) -> Tuple[plt.Figure, np.ndarray]:
#     """
#     Plot a panel of log-return risk-neutral densities (RND) for a given date,
#     overlaying the curves from each result_dict in plot_dict.

#     Assumed per-model structure:
#         day = result_dict[date]
#         day["rnd_lr_surface"] : array shape (nT, nX)
#         day["T_grid"]         : array shape (nT,)
#         day["rnd_lr_grid"]    : array shape (nX,)

#     maturity_list:
#         - If values are ints (or int-like), treated as maturity indices into T_grid.
#         - Otherwise treated as maturities (in same units as T_grid) and matched to nearest.

#     Returns
#     -------
#     fig, axes
#     """

#     # ---- helpers ----
#     def _date_key(res_dict: dict, date_in: Any):
#         # allow passing date as str like "2021-06-01"
#         if isinstance(date_in, str) and date_in in res_dict:
#             return date_in
#         # try pandas-ish Timestamp stringification without importing pandas
#         s = str(date_in)
#         if s in res_dict:
#             return s
#         # common case: "YYYY-MM-DD 00:00:00" -> "YYYY-MM-DD"
#         s2 = s[:10]
#         if s2 in res_dict:
#             return s2
#         raise KeyError(f"Date '{date_in}' not found in result_dict keys (tried '{s}' and '{s2}').")

#     def _pick_T_index(T_grid: np.ndarray, m):
#         # treat ints as indices if in range
#         if isinstance(m, (int, np.integer)):
#             j = int(m)
#             if j < 0 or j >= len(T_grid):
#                 raise IndexError(f"maturity index {j} out of range for T_grid length {len(T_grid)}.")
#             return j, float(T_grid[j])
#         # if float is very close to int and within range, still interpret as index
#         if isinstance(m, (float, np.floating)) and float(m).is_integer():
#             j = int(m)
#             if 0 <= j < len(T_grid):
#                 return j, float(T_grid[j])
#         # otherwise nearest maturity value
#         mval = float(m)
#         j = int(np.argmin(np.abs(T_grid - mval)))
#         return j, float(T_grid[j])

#     # ---- extract one model to size panels / choose panel layout ----
#     if len(plot_dict) == 0:
#         raise ValueError("plot_dict is empty.")

#     first_label = next(iter(plot_dict))
#     first_res = plot_dict[first_label]
#     date_key = _date_key(first_res, date)

#     first_day = first_res[date_key]
#     for k in ("rnd_lr_surface", "T_grid", "rnd_lr_grid"):
#         if k not in first_day:
#             raise KeyError(f"Missing '{k}' in plot_dict['{first_label}'][{date_key}].")

#     T_grid0 = np.asarray(first_day["T_grid"], float)
#     x_grid0 = np.asarray(first_day["rnd_lr_grid"], float)

#     # ---- resolve maturity indices (using first model's T_grid as reference) ----
#     mats = []
#     for m in maturity_list:
#         j, Tj = _pick_T_index(T_grid0, m)
#         mats.append((j, Tj))

#     n_panels = len(mats)

#     # ---- panel shape ----
#     if panel_shape is None:
#         # simple auto: up to 3 columns
#         ncols = min(3, n_panels)
#         nrows = int(np.ceil(n_panels / ncols))
#         panel_shape = (nrows, ncols)

#     nrows, ncols = panel_shape
#     if nrows * ncols < n_panels:
#         raise ValueError(f"panel_shape {panel_shape} too small for {n_panels} panels.")

#     figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
#     axes = np.asarray(axes).reshape(nrows, ncols)

#     # ---- plotting ----
#     for p, (j, Tj) in enumerate(mats):
#         ax = axes.flat[p]

#         for model_label, res_dict in plot_dict.items():
#             dkey = _date_key(res_dict, date)
#             day = res_dict[dkey]

#             # pull arrays
#             T_grid = np.asarray(day["T_grid"], float)
#             x_grid = np.asarray(day["rnd_lr_grid"], float)
#             surf = np.asarray(day["rnd_lr_surface"], float)

#             # if T grids differ slightly, remap by nearest T (safer than assuming same index)
#             jj = int(np.argmin(np.abs(T_grid - Tj)))

#             y = surf[jj, :]

#             # handle mismatch in x grids by requiring same length; (better interpolation optional)
#             if x_grid.shape != x_grid0.shape or not np.allclose(x_grid, x_grid0, atol=1e-12, rtol=1e-9):
#                 # fallback: interpolate to reference x_grid0
#                 y = np.interp(x_grid0, x_grid, y)
#                 x = x_grid0
#             else:
#                 x = x_grid

#             ax.plot(x, y, lw=lw, alpha=alpha, label=model_label)

#         ax.set_title(f"T ≈ {Tj:.6g} (panel idx {p})")
#         if grid:
#             ax.grid(True, alpha=0.25)

#         if xlim is not None:
#             ax.set_xlim(*xlim)
#         if ylim is not None:
#             ax.set_ylim(*ylim)

#     # turn off unused axes
#     for p in range(n_panels, nrows * ncols):
#         axes.flat[p].axis("off")

#     # labels
#     for ax in axes[:, 0]:
#         if ax.has_data():
#             ax.set_ylabel("RND (log-return)")
#     for ax in axes[-1, :]:
#         if ax.has_data():
#             ax.set_xlabel("log-return")

#     if title is None:
#         title = f"Log-return RND overlays on {str(date)[:10]}"
#     fig.suptitle(title, y=0.995)
#     fig.tight_layout(rect=[0, 0, 1, 0.97])

#     # one legend (global)
#     if legend:
#         # grab handles/labels from first active axis
#         handles, labels = None, None
#         for ax in axes.flat:
#             if ax.has_data():
#                 handles, labels = ax.get_legend_handles_labels()
#                 break
#         if handles:
#             fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)

#             # make room for legend
#             fig.subplots_adjust(bottom=0.08 + 0.03 * (len(labels) > 4))

#     # save
#     if save is not None:
#         save = Path(save)
#         save.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(save, dpi=dpi, bbox_inches="tight")

#     return fig, axes


# def call_fit_error_timeseries(
#     result_dict: Dict[Any, dict],
#     *,
#     date_col: str = "date",
#     metrics: Tuple[str, ...] = ("rmse", "mape"),
#     require_success: bool = True,
#     plot: bool = False,
#     title: Optional[str] = None,
#     save_csv: Optional[Union[str, Path]] = None,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Build a time series of fitted call price errors from a result_dict.

#     Expected structure per day:
#         day = result_dict[date]
#         day["errors"] is a dict containing keys like "rmse", "mape"
#         optionally day["success"] is bool

#     Parameters
#     ----------
#     result_dict : dict
#         date_key -> day dict
#     metrics : tuple[str]
#         which error metrics to extract from day["errors"]
#     require_success : bool
#         if True, skip days where day.get("success") is False
#     plot : bool
#         if True, plot the metric time series
#     save_csv : str|Path|None
#         if provided, save the time series DataFrame to CSV

#     Returns
#     -------
#     ts_df : pd.DataFrame
#         columns: [date_col] + metrics + optional 'success'
#     summary_df : pd.DataFrame
#         summary statistics for each metric (count, mean, std, min, p05, p25, median, p75, p95, max)
#     """

#     rows = []
#     for k, day in result_dict.items():
#         if not isinstance(day, dict):
#             continue

#         success = bool(day.get("success", True))
#         if require_success and not success:
#             continue

#         errs = day.get("errors", None)
#         if not isinstance(errs, dict):
#             continue

#         row = {date_col: k, "success": success}
#         ok = True
#         for m in metrics:
#             v = errs.get(m, np.nan)
#             try:
#                 row[m] = float(v)
#             except Exception:
#                 row[m] = np.nan
#             if np.isnan(row[m]):
#                 ok = False  # still keep row; you can choose to drop later
#         rows.append(row)

#     if len(rows) == 0:
#         ts_df = pd.DataFrame(columns=[date_col, "success", *metrics])
#         summary_df = pd.DataFrame(index=list(metrics))
#         return ts_df, summary_df

#     ts_df = pd.DataFrame(rows)

#     # normalize/parse date column
#     ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
#     ts_df = ts_df.sort_values(date_col).reset_index(drop=True)

#     # summary stats
#     def _summary(s: pd.Series) -> pd.Series:
#         s = pd.to_numeric(s, errors="coerce").dropna()
#         if s.empty:
#             return pd.Series(
#                 {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
#                  "p05": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan}
#             )
#         q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
#         return pd.Series(
#             {
#                 "count": int(s.count()),
#                 "mean": float(s.mean()),
#                 "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
#                 "min": float(s.min()),
#                 "p05": float(q.loc[0.05]),
#                 "p25": float(q.loc[0.25]),
#                 "median": float(q.loc[0.50]),
#                 "p75": float(q.loc[0.75]),
#                 "p95": float(q.loc[0.95]),
#                 "max": float(s.max()),
#             }
#         )

#     summary_df = pd.DataFrame({m: _summary(ts_df[m]) for m in metrics}).T

#     # save
#     if save_csv is not None:
#         save_csv = Path(save_csv)
#         save_csv.parent.mkdir(parents=True, exist_ok=True)
#         ts_df.to_csv(save_csv, index=False)

#     # plot
#     if plot:
#         ttl = title or "Call fit error time series"
#         for m in metrics:
#             plt.figure(figsize=(10, 3.2))
#             plt.plot(ts_df[date_col], ts_df[m])
#             plt.title(f"{ttl}: {m}")
#             plt.xlabel("Date")
#             plt.ylabel(m)
#             plt.grid(True, alpha=0.25)
#             plt.tight_layout()
#             plt.show()

#     return ts_df, summary_df


# def P_Q_K_multipanel_multi(
#     out_dict: Dict[str, dict],
#     *,
#     title: Optional[str] = None,
#     n_panels: Optional[int] = None,                 # if None uses panel_shape product
#     panel_shape: Tuple[int, int] = (2, 4),
#     save: Optional[Union[str, Path]] = None,
#     dpi: int = 200,
#     # ----- truncation controls -----
#     truncate: bool = True,
#     ptail_alpha: Tuple[float, float] = (0.10, 0.0), # (alpha_left, alpha_right) for p-CDF tails
#     trunc_mode: str = "cdf",                        # {"cdf","rbounds","none","cdf+rbounds"}
#     r_bounds: Optional[Tuple[float, float]] = None,
#     clip_trunc_to_support: bool = True,
#     # ----- kernel axis controls -----
#     kernel_linestyle: str = "--",
#     kernel_yscale: str = "linear",                  # {"linear","log"}
#     kernel_log_eps: float = 1e-300,
#     # ----- display controls -----
#     legend_loc: str = "upper center",
#     lw_density: float = 1.6,
#     lw_kernel: float = 1.4,
#     alpha_density: float = 0.95,
#     alpha_kernel: float = 0.90,
#     # labeling style
#     show_model_in_label: bool = True,               # labels like "q_R (ModelA)"
# ):
#     """
#     Multi-panel plot: q_R(R), p_R(R) and pricing kernel M(R) with dual y-axis,
#     OVERLAYED across multiple out dictionaries.

#     out_dict: {"label": out, ...}

#     Assumed per-out structure:
#       out["anchor_surfaces"]["qR_surface"], ["pR_surface"], ["M_surface"] with shape (nT, nR)
#       out["T_anchor"] : (nT,)
#       out["R_common"] : (nR,)
#     """

#     if not isinstance(out_dict, dict) or len(out_dict) == 0:
#         raise ValueError("out_dict must be a non-empty dict like {'Model A': outA, ...}.")

#     # -------------------------
#     # parse truncation settings
#     # -------------------------
#     mode = str(trunc_mode).lower().strip()
#     if not truncate:
#         mode = "none"

#     valid = {"none", "cdf", "rbounds", "cdf+rbounds"}
#     if mode not in valid:
#         raise ValueError(f"trunc_mode must be one of {valid}.")

#     use_cdf = mode in {"cdf", "cdf+rbounds"}
#     use_rbounds = mode in {"rbounds", "cdf+rbounds"}

#     aL, aR = float(ptail_alpha[0]), float(ptail_alpha[1])
#     if use_cdf:
#         if not (0.0 <= aL < 1.0 and 0.0 <= aR < 1.0):
#             raise ValueError("ptail_alpha must be in [0,1) for both tails.")
#         if not (aL + aR < 1.0):
#             raise ValueError("Require ptail_alpha[0] + ptail_alpha[1] < 1.")

#     kernel_yscale = str(kernel_yscale).lower().strip()
#     if kernel_yscale not in {"linear", "log"}:
#         raise ValueError("kernel_yscale must be one of {'linear','log'}.")
#     kernel_log_eps = float(kernel_log_eps)
#     if kernel_yscale == "log" and not (kernel_log_eps > 0):
#         raise ValueError("kernel_log_eps must be > 0 for log scale.")

#     # -------------------------
#     # reference grids from first model
#     # -------------------------
#     first_label = next(iter(out_dict))
#     out0 = out_dict[first_label]
#     if out0 is None or "anchor_surfaces" not in out0:
#         raise KeyError(f"out_dict['{first_label}'] must contain out['anchor_surfaces'].")

#     anchor0 = out0["anchor_surfaces"]
#     T_ref = np.asarray(out0.get("T_anchor", []), float).ravel()
#     R_ref = np.asarray(out0.get("R_common", []), float).ravel()

#     if T_ref.size == 0 or R_ref.size == 0:
#         raise ValueError("Missing T_anchor or R_common in the first out dict.")
#     if R_ref.size >= 2 and np.any(np.diff(R_ref) <= 0):
#         raise ValueError("R_common must be strictly increasing (reference model).")

#     # rbounds range (if enabled) in reference coordinates
#     R_min = R_max = None
#     if use_rbounds:
#         if r_bounds is None or len(r_bounds) != 2:
#             raise ValueError("For rbounds truncation, provide r_bounds=(R_min, R_max).")
#         R_min, R_max = float(r_bounds[0]), float(r_bounds[1])
#         if clip_trunc_to_support:
#             R_min = max(R_min, float(R_ref[0]))
#             R_max = min(R_max, float(R_ref[-1]))
#         if not (np.isfinite(R_min) and np.isfinite(R_max) and R_max > R_min):
#             raise ValueError("Invalid r_bounds after clipping.")

#     # -------------------------
#     # choose maturities for panels
#     # -------------------------
#     nrows, ncols = panel_shape
#     nT = T_ref.size
#     idx_pool = np.arange(nT)

#     n_pan = (nrows * ncols) if n_panels is None else int(n_panels)
#     n_pan = max(1, min(n_pan, idx_pool.size))
#     idxs = idx_pool[np.linspace(0, idx_pool.size - 1, n_pan, dtype=int)]

#     fig, axes = plt.subplots(
#         nrows, ncols,
#         figsize=(5.4 * ncols, 3.8 * nrows),
#         sharex=True,
#         constrained_layout=False
#     )
#     axes = np.array(axes).reshape(-1)
#     ax2_list = []

#     # -------------------------
#     # helpers
#     # -------------------------
#     def _interp_to_ref(R_src, y_src):
#         """Interpolate y(R_src) onto R_ref. Assumes R_src increasing."""
#         return np.interp(R_ref, R_src, y_src)

#     def _pcdf_cutoffs(R_src, p_src):
#         """
#         Compute p-CDF cutoffs on the model's full support (R_src),
#         return (R_left, R_right) corresponding to alpha_left / 1-alpha_right.
#         """
#         pj = np.maximum(np.asarray(p_src, float), 0.0)
#         dR = np.diff(R_src)
#         inc = 0.5 * (pj[1:] + pj[:-1]) * dR
#         cdf = np.empty_like(R_src)
#         cdf[0] = 0.0
#         cdf[1:] = np.cumsum(inc)
#         total = float(cdf[-1])
#         if not (total > 0 and np.isfinite(total)):
#             return None

#         cdf /= total

#         # left cutoff
#         if aL > 0:
#             idxL = np.where(cdf >= aL)[0]
#             iL = int(idxL[0]) if idxL.size else 0
#         else:
#             iL = 0

#         # right cutoff
#         if aR > 0:
#             idxR = np.where(cdf <= (1.0 - aR))[0]
#             iR = int(idxR[-1]) if idxR.size else (R_src.size - 1)
#         else:
#             iR = R_src.size - 1

#         if iR <= iL:
#             return None
#         return float(R_src[iL]), float(R_src[iR])

#     # -------------------------
#     # main plot loop
#     # -------------------------
#     for k, j in enumerate(idxs):
#         ax = axes[k]
#         ax2 = ax.twinx()
#         ax2_list.append(ax2)

#         # base mask on reference grid (rbounds applies to everything)
#         mask_all = np.isfinite(R_ref)
#         if use_rbounds:
#             mask_all &= (R_ref >= R_min) & (R_ref <= R_max)

#         R_all = R_ref[mask_all]

#         for model_label, out in out_dict.items():
#             if out is None or "anchor_surfaces" not in out:
#                 continue

#             anchor = out["anchor_surfaces"]
#             T = np.asarray(out.get("T_anchor", []), float).ravel()
#             R = np.asarray(out.get("R_common", []), float).ravel()
#             qR = np.asarray(anchor.get("qR_surface", []), float)
#             pR = np.asarray(anchor.get("pR_surface", []), float)
#             M  = np.asarray(anchor.get("M_surface", []), float)

#             if T.size == 0 or R.size == 0:
#                 continue
#             if qR.shape != (T.size, R.size) or pR.shape != (T.size, R.size) or M.shape != (T.size, R.size):
#                 continue

#             # match maturity by nearest T to reference T_ref[j]
#             jj = int(np.argmin(np.abs(T - T_ref[j])))

#             qj = qR[jj, :]
#             pj = pR[jj, :]
#             Mj = M[jj, :]

#             # interpolate to reference R grid if needed
#             if R.shape != R_ref.shape or (R.size > 1 and not np.allclose(R, R_ref, atol=1e-12, rtol=1e-9)):
#                 qj_ref = _interp_to_ref(R, qj)
#                 pj_ref = _interp_to_ref(R, pj)
#                 Mj_ref = _interp_to_ref(R, Mj)
#             else:
#                 qj_ref, pj_ref, Mj_ref = qj, pj, Mj

#             # apply rbounds mask to q/p/M for plotting on left axis and for kernel base
#             q_plot = np.asarray(qj_ref, float)[mask_all]
#             p_plot = np.asarray(pj_ref, float)[mask_all]
#             M_plot = np.asarray(Mj_ref, float)[mask_all].copy()

#             qlab = "q_R(R)"
#             plab = "p_R(R)"
#             mlab = "M(R)"
#             if show_model_in_label:
#                 qlab += f" ({model_label})"
#                 plab += f" ({model_label})"
#                 mlab += f" ({model_label})"

#             # --- plot densities and capture the model color (use q_R color as the model color) ---
#             (q_line,) = ax.plot(
#                 R_all, q_plot,
#                 linewidth=lw_density + 0.3,
#                 alpha=alpha_density,
#                 label=qlab,
#             )
#             model_color = q_line.get_color()
            
#             ax.plot(
#                 R_all, p_plot,
#                 linewidth=lw_density,
#                 alpha=0.60,                  # <-- lighter
#                 linestyle="-",               # same line, but subdued
#                 label=plab,
#                 color=model_color,
#             )
                        
#             # ---- kernel truncation by p-CDF tails on FULL model support ----
#             R_k = R_all
#             M_k = M_plot
            
#             if use_cdf:
#                 cut = _pcdf_cutoffs(R, pR[jj, :])  # model-native grid for tail cutoffs
#                 if cut is None:
#                     R_k = np.array([], float)
#                     M_k = np.array([], float)
#                 else:
#                     R_left, R_right = cut
#                     keep = (R_k >= R_left) & (R_k <= R_right)
#                     R_k = R_k[keep]
#                     M_k = M_k[keep]
#                     if show_model_in_label:
#                         mlab = f"M(R) ({model_label}, p-tails ≥ {aL:.0%}/{aR:.0%})"
#                     else:
#                         mlab = f"M(R) (p-tails ≥ {aL:.0%}/{aR:.0%})"
            
#             # log scale cleanup
#             if kernel_yscale == "log" and M_k.size > 0:
#                 pos = np.isfinite(M_k) & (M_k > 0) & np.isfinite(R_k)
#                 R_k = np.asarray(R_k, float)[pos]
#                 M_k = np.asarray(M_k, float)[pos]
#                 M_k = np.maximum(M_k, kernel_log_eps)
            
#             if R_k.size > 0:
#                 ax2.plot(
#                     R_k, M_k,
#                     label=mlab + ("" if kernel_yscale == "linear" else " (log y)"),
#                     linestyle=kernel_linestyle,     # dotted/dashed as you like
#                     linewidth=lw_kernel,
#                     alpha=alpha_kernel,
#                     color=model_color,              # <-- matches the model density color
#                 )


#         # panel title from reference maturity
#         T_days = float(T_ref[j] * 365.0)
#         ax.set_title(f"T≈{T_days:.1f}d", fontsize=11)

#         if (k % ncols) == 0:
#             ax.set_ylabel("Density (R-space)")
#         if k >= (n_pan - ncols):
#             ax.set_xlabel("Gross return R")
#         if (k % ncols) == (ncols - 1):
#             ax2.set_ylabel("Pricing kernel M(R)")

#         ax.grid(True, alpha=0.25)
#         ax2.set_yscale(kernel_yscale)

#         if use_rbounds:
#             ax.set_xlim(R_min, R_max)

#     # turn off unused axes
#     for k in range(n_pan, axes.size):
#         axes[k].axis("off")

#     # legend from first active axes
#     handles, labels = [], []
#     for ax in [axes[0], ax2_list[0]]:
#         h, l = ax.get_legend_handles_labels()
#         handles.extend(h)
#         labels.extend(l)

#     if title is None:
#         title = "q_R vs p_R with Pricing Kernel (multi-model overlay)"

#     fig.suptitle(title, y=0.995, fontsize=14)
#     fig.legend(
#         handles, labels,
#         loc=legend_loc,
#         bbox_to_anchor=(0.5, 0.965),
#         ncol=4,
#         frameon=False,
#         handlelength=2.8,
#         columnspacing=1.4
#     )
#     fig.subplots_adjust(top=0.88, wspace=0.28, hspace=0.30)

#     if save is not None:
#         save = Path(save)
#         save.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(save, dpi=dpi, bbox_inches="tight")
#         print(f"[saved] {save}")

#     plt.show()
#     return fig
# ##############################################Multiday surface
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from pathlib import Path
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# def _cdf_from_pdf_trapz(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
#     """
#     NaN-safe-ish CDF from a PDF on grid x using cumulative trapezoid integration.
#     Normalizes to end at 1 when total area > 0.
#     """
#     x = np.asarray(x, float)
#     f = np.asarray(pdf, float)

#     n = x.size
#     if n < 2:
#         return np.zeros_like(x)

#     dx = np.diff(x)
#     trap = 0.5 * (f[:-1] + f[1:]) * dx

#     cdf = np.empty(n, float)
#     cdf[0] = 0.0
#     cdf[1:] = np.cumsum(trap)

#     total = cdf[-1]
#     if np.isfinite(total) and total > 0:
#         cdf /= total
#     else:
#         cdf[:] = 0.0

#     return np.clip(cdf, 0.0, 1.0)


# def plot_pricing_kernel_3d_surface_by_T(
#     result_dict: dict,
#     T_target: float,
#     *,
#     anchor_key: str = "anchor_surfaces",
#     kernel_key: str = "mK_surface",
#     fP_key: str = "fP_surface",        # physical density used for truncation
#     r_key: str = "r_common",
#     T_key: str = "T_anchor",

#     # x-axis
#     x_mode: str = "log",               # {"log","gross"}
#     x_bounds: tuple[float, float] | None = None,

#     # thinning
#     max_dates: int = 250,
#     stride: int | None = None,

#     # z scaling
#     z_mode: str = "log",               # {"level","log"}
#     z_eps: float = 1e-300,

#     # appearance
#     cmap: str = "viridis",

#     # NEW: truncate plotting window by physical tail mass (per date)
#     truncate_by_ptails: bool = False,
#     ptail_alphas: tuple[float, float] = (0.01, 0.01),  # (alpha_left, alpha_right)

#     # output
#     title: str | None = None,
#     save: str | Path | None = None,
#     dpi: int = 200,

#     # interactive
#     interactive: bool = False,
#     interactive_engine: str = "plotly",
#     save_html: str | Path | None = None,
# ):
#     """
#     3D surface: x=(log return r) OR x=(gross return exp(r)),
#                 y=time (date),
#                 z=pricing kernel at nearest maturity to T_target.

#     If truncate_by_ptails=True:
#       For EACH date-row, compute physical CDF from fP_surface at that T,
#       then keep only x where CDF in [alpha_left, 1-alpha_right].
#       Outside that interval, set Z to NaN (ragged edges in the surface).
#     """

#     if x_mode not in {"log", "gross"}:
#         raise ValueError("x_mode must be 'log' or 'gross'")
#     if z_mode not in {"level", "log"}:
#         raise ValueError("z_mode must be 'level' or 'log'")
#     if truncate_by_ptails:
#         aL, aR = float(ptail_alphas[0]), float(ptail_alphas[1])
#         if not (0.0 <= aL < 1.0 and 0.0 <= aR < 1.0 and (aL + aR) < 1.0):
#             raise ValueError("ptail_alphas must satisfy 0<=aL<1, 0<=aR<1, and aL+aR<1.")

#     # --- sort dates ---
#     keys = list(result_dict.keys())
#     date_ts = pd.to_datetime(keys)
#     order = np.argsort(date_ts.values)
#     date_ts = date_ts[order]
#     keys = [keys[i] for i in order]

#     # --- thin dates ---
#     n_all = len(keys)
#     if stride is not None and stride > 1:
#         keep = np.arange(0, n_all, stride, dtype=int)
#     else:
#         if n_all <= max_dates:
#             keep = np.arange(n_all, dtype=int)
#         else:
#             keep = np.linspace(0, n_all - 1, max_dates).round().astype(int)

#     keys = [keys[i] for i in keep]
#     date_ts = date_ts[keep]

#     # --- build Z(time, r) (and physical density rows if needed) ---
#     r_ref = None
#     T_used = None
#     rows_Z = []
#     rows_fP = []
#     kept_dates = []

#     for dk, dt in zip(keys, date_ts):
#         day = result_dict[dk]
#         if anchor_key not in day:
#             continue
#         if (T_key not in day) or (r_key not in day) or (kernel_key not in day[anchor_key]):
#             continue
#         if truncate_by_ptails and (fP_key not in day[anchor_key]):
#             raise KeyError(f"truncate_by_ptails=True but '{fP_key}' missing in day[{anchor_key}].")

#         T_grid = np.asarray(day[T_key], float).ravel()
#         r_grid = np.asarray(day[r_key], float).ravel()

#         Ksurf = np.asarray(day[anchor_key][kernel_key], float)
#         if Ksurf.ndim != 2:
#             continue

#         j = int(np.nanargmin(np.abs(T_grid - float(T_target))))
#         z_row = Ksurf[j, :].astype(float, copy=True)

#         if r_ref is None:
#             r_ref = r_grid
#             T_used = float(T_grid[j])
#             # CDF needs a monotone grid
#             if not np.all(np.diff(r_ref) > 0):
#                 raise ValueError("r_common must be strictly increasing for CDF-based truncation.")
#         else:
#             if r_grid.shape != r_ref.shape or np.nanmax(np.abs(r_grid - r_ref)) > 1e-12:
#                 continue

#         rows_Z.append(z_row)
#         kept_dates.append(dt)

#         if truncate_by_ptails:
#             fPsurf = np.asarray(day[anchor_key][fP_key], float)
#             if fPsurf.ndim != 2:
#                 raise ValueError(f"{fP_key} must be 2D (nT, nr).")
#             rows_fP.append(fPsurf[j, :].astype(float, copy=True))

#     if not rows_Z:
#         raise ValueError(
#             "No rows extracted. Check kernel_key/paths. "
#             f"anchor_key='{anchor_key}', kernel_key='{kernel_key}', r_key='{r_key}', T_key='{T_key}'."
#         )

#     Z = np.vstack(rows_Z)  # (nt, nr)
#     kept_dates = pd.to_datetime(kept_dates)

#     # --- truncate each date-row by physical tail mass ---
#     if truncate_by_ptails:
#         FP = np.vstack(rows_fP)  # (nt, nr)
#         aL, aR = float(ptail_alphas[0]), float(ptail_alphas[1])

#         for i in range(Z.shape[0]):
#             fp = FP[i, :]
#             good = np.isfinite(fp) & np.isfinite(r_ref)
#             if np.sum(good) < 5:
#                 Z[i, :] = np.nan
#                 continue

#             rr = r_ref[good]
#             ff = np.maximum(fp[good], 0.0)  # clip small negative noise

#             cdf = _cdf_from_pdf_trapz(rr, ff)

#             # keep rr where cdf in [aL, 1-aR]
#             left_pos = int(np.searchsorted(cdf, aL, side="left"))
#             right_pos = int(np.searchsorted(cdf, 1.0 - aR, side="right")) - 1

#             if right_pos <= left_pos:
#                 Z[i, :] = np.nan
#                 continue

#             keep_good = np.zeros(rr.size, dtype=bool)
#             keep_good[left_pos : right_pos + 1] = True

#             keep_full = np.zeros(r_ref.size, dtype=bool)
#             keep_full[np.where(good)[0][keep_good]] = True

#             Z[i, ~keep_full] = np.nan

#     # --- choose x axis ---
#     if x_mode == "log":
#         x = r_ref.copy()
#         xlab = "log return r"
#     else:
#         x = np.exp(r_ref)
#         xlab = "gross return R = exp(r)"

#     # --- x_bounds in chosen units ---
#     if x_bounds is not None:
#         lo, hi = float(x_bounds[0]), float(x_bounds[1])
#         m = (x >= lo) & (x <= hi)
#         x = x[m]
#         Z = Z[:, m]

#     # --- z transform ---
#     if z_mode == "log":
#         Z_plot = np.log(np.maximum(Z, z_eps))
#         zlab = f"log({kernel_key})"
#     else:
#         Z_plot = Z
#         zlab = kernel_key

#     # --- title ---
#     ttl = title if title is not None else (
#         f"Pricing Kernel 3D Surface (T_target={T_target:.6f}y, used={T_used:.6f}y, x_mode={x_mode})"
#         + (f", ptails={ptail_alphas}" if truncate_by_ptails else "")
#     )

#     # --- mesh ---
#     y_num = mdates.date2num(kept_dates.to_pydatetime())
#     X, Y = np.meshgrid(x, y_num)

#     # ----------------------------
#     # Interactive (Plotly)
#     # ----------------------------
#     if interactive:
#         if interactive_engine != "plotly":
#             raise ValueError("interactive_engine currently only supports 'plotly'.")
#         try:
#             import plotly.graph_objects as go
#         except Exception as e:
#             raise ImportError("Plotly is required for interactive=True. Install with: pip install plotly") from e

#         y_labels = [d.strftime("%Y-%m-%d") for d in kept_dates]

#         fig = go.Figure(
#             data=go.Surface(
#                 x=x, y=y_labels, z=Z_plot,
#                 colorscale=cmap,  # e.g. "Viridis", "Inferno" (Plotly names)
#                 showscale=True
#             )
#         )
#         fig.update_layout(
#             title=ttl,
#             scene=dict(xaxis_title=xlab, yaxis_title="date", zaxis_title=zlab),
#             margin=dict(l=0, r=0, t=40, b=0),
#         )

#         html_path = None
#         if save_html is not None:
#             html_path = Path(save_html)
#         elif save is not None and str(save).lower().endswith(".html"):
#             html_path = Path(save)

#         if html_path is not None:
#             html_path.parent.mkdir(parents=True, exist_ok=True)
#             fig.write_html(str(html_path), include_plotlyjs="cdn")

#         return fig

#     # ----------------------------
#     # Static (Matplotlib)
#     # ----------------------------
#     fig = plt.figure(figsize=(12, 7))
#     ax = fig.add_subplot(111, projection="3d")
#     surf = ax.plot_surface(X, Y, Z_plot, linewidth=0, antialiased=True, cmap=cmap)

#     ax.set_xlabel(xlab)
#     ax.set_ylabel("date")
#     ax.set_zlabel(zlab)
#     ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
#     ax.set_title(ttl)

#     fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
#     fig.tight_layout()

#     # IO-friendly save (auto-create folder; auto-add .png)
#     if save is not None:
#         save_path = Path(save)
#         if save_path.suffix == "":
#             save_path = save_path.with_suffix(".png")
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

#     return fig, ax

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from pathlib import Path
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# def _cdf_from_pdf_trapz(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
#     x = np.asarray(x, float)
#     f = np.asarray(pdf, float)
#     n = x.size
#     if n < 2:
#         return np.zeros_like(x)

#     dx = np.diff(x)
#     trap = 0.5 * (f[:-1] + f[1:]) * dx

#     cdf = np.empty(n, float)
#     cdf[0] = 0.0
#     cdf[1:] = np.cumsum(trap)

#     total = cdf[-1]
#     if np.isfinite(total) and total > 0:
#         cdf /= total
#     else:
#         cdf[:] = 0.0

#     return np.clip(cdf, 0.0, 1.0)


# def plot_rnd_3d_surface_by_T(
#     result_dict: dict,
#     T_target: float,
#     *,
#     # --- paths/keys (matches your screenshots) ---
#     T_key: str = "T_grid",              # maturity grid (nT,)
#     r_key: str = "rnd_lr_grid",         # log-return grid for RND (nr,)
#     rnd_key: str = "rnd_lr_surface",    # risk-neutral density in log-return space (nT,nr)

#     # --- x-axis choice ---
#     x_mode: str = "log",                # {"log","gross"} where gross = exp(r)
#     x_bounds: tuple[float, float] | None = None,  # bounds in x-units matching x_mode

#     # --- thinning across dates ---
#     max_dates: int = 250,
#     stride: int | None = None,

#     # --- z scaling ---
#     z_mode: str = "log",                # {"level","log"}
#     z_eps: float = 1e-300,

#     # --- optional truncation by RND tail mass (per date) ---
#     truncate_by_tails: bool = False,
#     tail_alphas: tuple[float, float] = (0.01, 0.01),  # keep CDF in [aL, 1-aR]

#     # --- appearance/output ---
#     cmap: str = "viridis",
#     title: str | None = None,
#     save: str | Path | None = None,
#     dpi: int = 200,

#     # --- interactive ---
#     interactive: bool = False,
#     interactive_engine: str = "plotly",
#     save_html: str | Path | None = None,
# ):
#     """
#     Multi-day 3D surface of the RISK-NEUTRAL DENSITY at a fixed maturity.

#     Inputs expected (per date):
#       day[T_key]       -> (nT,)
#       day[r_key]       -> (nr,)
#       day[rnd_key]     -> (nT, nr)

#     Plots:
#       x = r (log return) OR exp(r) (gross return)
#       y = date
#       z = q(r | T) (level or log)

#     If truncate_by_tails=True:
#       Per-date, compute CDF from q and keep only where CDF in [aL, 1-aR],
#       setting the rest to NaN (ragged edges).
#     """
#     if x_mode not in {"log", "gross"}:
#         raise ValueError("x_mode must be 'log' or 'gross'")
#     if z_mode not in {"level", "log"}:
#         raise ValueError("z_mode must be 'level' or 'log'")
#     if truncate_by_tails:
#         aL, aR = float(tail_alphas[0]), float(tail_alphas[1])
#         if not (0.0 <= aL < 1.0 and 0.0 <= aR < 1.0 and (aL + aR) < 1.0):
#             raise ValueError("tail_alphas must satisfy 0<=aL<1, 0<=aR<1, and aL+aR<1.")

#     # --- sort dates ---
#     keys = list(result_dict.keys())
#     date_ts = pd.to_datetime(keys)
#     order = np.argsort(date_ts.values)
#     date_ts = date_ts[order]
#     keys = [keys[i] for i in order]

#     # --- thin dates ---
#     n_all = len(keys)
#     if stride is not None and stride > 1:
#         keep = np.arange(0, n_all, stride, dtype=int)
#     else:
#         if n_all <= max_dates:
#             keep = np.arange(n_all, dtype=int)
#         else:
#             keep = np.linspace(0, n_all - 1, max_dates).round().astype(int)

#     keys = [keys[i] for i in keep]
#     date_ts = date_ts[keep]

#     # --- collect rows ---
#     r_ref = None
#     T_used = None
#     rows = []
#     kept_dates = []

#     for dk, dt in zip(keys, date_ts):
#         day = result_dict[dk]

#         if (T_key not in day) or (r_key not in day) or (rnd_key not in day):
#             continue

#         T_grid = np.asarray(day[T_key], float).ravel()
#         r_grid = np.asarray(day[r_key], float).ravel()
#         Qsurf = np.asarray(day[rnd_key], float)

#         if Qsurf.ndim != 2:
#             continue

#         j = int(np.nanargmin(np.abs(T_grid - float(T_target))))
#         q_row = Qsurf[j, :].astype(float, copy=True)

#         if r_ref is None:
#             r_ref = r_grid
#             T_used = float(T_grid[j])
#             if truncate_by_tails and not np.all(np.diff(r_ref) > 0):
#                 raise ValueError("rnd_lr_grid must be strictly increasing for CDF-based truncation.")
#         else:
#             # require consistent r-grid across dates
#             if r_grid.shape != r_ref.shape or np.nanmax(np.abs(r_grid - r_ref)) > 1e-12:
#                 continue

#         rows.append(q_row)
#         kept_dates.append(dt)

#     if not rows:
#         raise ValueError(
#             "No rows extracted. Check keys or data availability. "
#             f"Expected day['{T_key}'], day['{r_key}'], day['{rnd_key}']."
#         )

#     Z = np.vstack(rows)  # (nt, nr)
#     kept_dates = pd.to_datetime(kept_dates)

#     # --- optional per-row truncation by tail mass of q ---
#     if truncate_by_tails:
#         aL, aR = float(tail_alphas[0]), float(tail_alphas[1])
#         for i in range(Z.shape[0]):
#             q = Z[i, :]
#             good = np.isfinite(q) & np.isfinite(r_ref)
#             if np.sum(good) < 5:
#                 Z[i, :] = np.nan
#                 continue

#             rr = r_ref[good]
#             qq = np.maximum(q[good], 0.0)  # clip tiny negatives

#             cdf = _cdf_from_pdf_trapz(rr, qq)
#             left_pos = int(np.searchsorted(cdf, aL, side="left"))
#             right_pos = int(np.searchsorted(cdf, 1.0 - aR, side="right")) - 1

#             if right_pos <= left_pos:
#                 Z[i, :] = np.nan
#                 continue

#             keep_good = np.zeros(rr.size, dtype=bool)
#             keep_good[left_pos : right_pos + 1] = True

#             keep_full = np.zeros(r_ref.size, dtype=bool)
#             keep_full[np.where(good)[0][keep_good]] = True

#             Z[i, ~keep_full] = np.nan

#     # --- choose x axis ---
#     if x_mode == "log":
#         x = r_ref.copy()
#         xlab = "log return r"
#     else:
#         x = np.exp(r_ref)
#         xlab = "gross return R = exp(r)"

#     # --- x bounds in chosen units ---
#     if x_bounds is not None:
#         lo, hi = float(x_bounds[0]), float(x_bounds[1])
#         m = (x >= lo) & (x <= hi)
#         x = x[m]
#         Z = Z[:, m]

#     # --- z transform ---
#     if z_mode == "log":
#         Z_plot = np.log(np.maximum(Z, z_eps))
#         zlab = f"log({rnd_key})"
#     else:
#         Z_plot = Z
#         zlab = rnd_key

#     # --- title ---
#     ttl = title if title is not None else (
#         f"Risk-Neutral Density 3D Surface (T_target={T_target:.6f}y, used={T_used:.6f}y, x_mode={x_mode})"
#         + (f", tails={tail_alphas}" if truncate_by_tails else "")
#     )

#     # --- mesh ---
#     y_num = mdates.date2num(kept_dates.to_pydatetime())
#     X, Y = np.meshgrid(x, y_num)

#     # ----------------------------
#     # Interactive (Plotly)
#     # ----------------------------
#     if interactive:
#         if interactive_engine != "plotly":
#             raise ValueError("interactive_engine currently only supports 'plotly'.")
#         try:
#             import plotly.graph_objects as go
#         except Exception as e:
#             raise ImportError("Plotly is required for interactive=True. Install with: pip install plotly") from e

#         y_labels = [d.strftime("%Y-%m-%d") for d in kept_dates]

#         fig = go.Figure(
#             data=go.Surface(
#                 x=x,
#                 y=y_labels,
#                 z=Z_plot,
#                 colorscale=cmap,  # e.g. "Viridis", "Inferno"
#                 showscale=True,
#             )
#         )
#         fig.update_layout(
#             title=ttl,
#             scene=dict(
#                 xaxis_title=xlab,
#                 yaxis_title="date",
#                 zaxis_title=zlab,
#             ),
#             margin=dict(l=0, r=0, t=40, b=0),
#         )

#         html_path = None
#         if save_html is not None:
#             html_path = Path(save_html)
#         elif save is not None and str(save).lower().endswith(".html"):
#             html_path = Path(save)

#         if html_path is not None:
#             html_path.parent.mkdir(parents=True, exist_ok=True)
#             fig.write_html(str(html_path), include_plotlyjs="cdn")

#         return fig, None  # (fig, ax) style, ax=None for interactive

#     # ----------------------------
#     # Static (Matplotlib)
#     # ----------------------------
#     fig = plt.figure(figsize=(12, 7))
#     ax = fig.add_subplot(111, projection="3d")
#     surf = ax.plot_surface(X, Y, Z_plot, linewidth=0, antialiased=True, cmap=cmap)

#     ax.set_xlabel(xlab)
#     ax.set_ylabel("date")
#     ax.set_zlabel(zlab)
#     ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
#     ax.set_title(ttl)

#     fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
#     fig.tight_layout()

#     if save is not None:
#         save_path = Path(save)
#         if save_path.suffix == "":
#             save_path = save_path.with_suffix(".png")
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

#     return fig, ax

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from pathlib import Path
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# def _cdf_from_pdf_trapz(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
#     x = np.asarray(x, float)
#     f = np.asarray(pdf, float)
#     n = x.size
#     if n < 2:
#         return np.zeros_like(x)

#     dx = np.diff(x)
#     trap = 0.5 * (f[:-1] + f[1:]) * dx

#     cdf = np.empty(n, float)
#     cdf[0] = 0.0
#     cdf[1:] = np.cumsum(trap)

#     total = cdf[-1]
#     if np.isfinite(total) and total > 0:
#         cdf /= total
#     else:
#         cdf[:] = 0.0

#     return np.clip(cdf, 0.0, 1.0)


# def plot_physical_density_3d_surface_by_T(
#     result_dict: dict,
#     T_target: float,
#     *,
#     anchor_key: str = "anchor_surfaces",
#     fP_key: str = "fP_surface",        # physical density surface (nT, nr)
#     r_key: str = "r_common",
#     T_key: str = "T_anchor",

#     # x-axis
#     x_mode: str = "log",               # {"log","gross"}
#     x_bounds: tuple[float, float] | None = None,

#     # thinning
#     max_dates: int = 250,
#     stride: int | None = None,

#     # z scaling
#     z_mode: str = "log",               # {"level","log"}
#     z_eps: float = 1e-300,

#     # optional truncation by physical tails (uses the density itself)
#     truncate_by_tails: bool = False,
#     tail_alphas: tuple[float, float] = (0.01, 0.01),  # keep CDF in [aL, 1-aR]

#     # appearance
#     cmap: str = "viridis",

#     # output
#     title: str | None = None,
#     save: str | Path | None = None,
#     dpi: int = 200,

#     # interactive
#     interactive: bool = False,
#     interactive_engine: str = "plotly",
#     save_html: str | Path | None = None,
# ):
#     """
#     3D surface for PHYSICAL density at nearest maturity to T_target:

#       x = log return r  OR  gross return exp(r)
#       y = time (date)
#       z = fP(r | T)  (level or log)

#     If truncate_by_tails=True:
#       Per-date, compute CDF from fP and keep only where CDF in [aL, 1-aR],
#       setting the rest to NaN (ragged edges).
#     """
#     if x_mode not in {"log", "gross"}:
#         raise ValueError("x_mode must be 'log' or 'gross'")
#     if z_mode not in {"level", "log"}:
#         raise ValueError("z_mode must be 'level' or 'log'")
#     if truncate_by_tails:
#         aL, aR = float(tail_alphas[0]), float(tail_alphas[1])
#         if not (0.0 <= aL < 1.0 and 0.0 <= aR < 1.0 and (aL + aR) < 1.0):
#             raise ValueError("tail_alphas must satisfy 0<=aL<1, 0<=aR<1, and aL+aR<1.")

#     # --- sort dates ---
#     keys = list(result_dict.keys())
#     date_ts = pd.to_datetime(keys)
#     order = np.argsort(date_ts.values)
#     date_ts = date_ts[order]
#     keys = [keys[i] for i in order]

#     # --- thin dates ---
#     n_all = len(keys)
#     if stride is not None and stride > 1:
#         keep = np.arange(0, n_all, stride, dtype=int)
#     else:
#         if n_all <= max_dates:
#             keep = np.arange(n_all, dtype=int)
#         else:
#             keep = np.linspace(0, n_all - 1, max_dates).round().astype(int)

#     keys = [keys[i] for i in keep]
#     date_ts = date_ts[keep]

#     # --- collect rows ---
#     r_ref = None
#     T_used = None
#     rows = []
#     kept_dates = []

#     for dk, dt in zip(keys, date_ts):
#         day = result_dict[dk]
#         if anchor_key not in day:
#             continue
#         if (T_key not in day) or (r_key not in day) or (fP_key not in day[anchor_key]):
#             continue

#         T_grid = np.asarray(day[T_key], float).ravel()
#         r_grid = np.asarray(day[r_key], float).ravel()
#         fPsurf = np.asarray(day[anchor_key][fP_key], float)
#         if fPsurf.ndim != 2:
#             continue

#         j = int(np.nanargmin(np.abs(T_grid - float(T_target))))
#         fp = fPsurf[j, :].astype(float, copy=True)

#         if r_ref is None:
#             r_ref = r_grid
#             T_used = float(T_grid[j])
#             if truncate_by_tails and not np.all(np.diff(r_ref) > 0):
#                 raise ValueError("r_common must be strictly increasing for CDF-based truncation.")
#         else:
#             if r_grid.shape != r_ref.shape or np.nanmax(np.abs(r_grid - r_ref)) > 1e-12:
#                 continue

#         rows.append(fp)
#         kept_dates.append(dt)

#     if not rows:
#         raise ValueError(
#             "No rows extracted. Check fP_key/paths. "
#             f"anchor_key='{anchor_key}', fP_key='{fP_key}', r_key='{r_key}', T_key='{T_key}'."
#         )

#     Z = np.vstack(rows)  # (nt, nr)
#     kept_dates = pd.to_datetime(kept_dates)

#     # --- optional truncation by tails (per row) ---
#     if truncate_by_tails:
#         aL, aR = float(tail_alphas[0]), float(tail_alphas[1])
#         for i in range(Z.shape[0]):
#             fp = Z[i, :]
#             good = np.isfinite(fp) & np.isfinite(r_ref)
#             if np.sum(good) < 5:
#                 Z[i, :] = np.nan
#                 continue

#             rr = r_ref[good]
#             ff = np.maximum(fp[good], 0.0)
#             cdf = _cdf_from_pdf_trapz(rr, ff)

#             left_pos = int(np.searchsorted(cdf, aL, side="left"))
#             right_pos = int(np.searchsorted(cdf, 1.0 - aR, side="right")) - 1
#             if right_pos <= left_pos:
#                 Z[i, :] = np.nan
#                 continue

#             keep_good = np.zeros(rr.size, dtype=bool)
#             keep_good[left_pos : right_pos + 1] = True

#             keep_full = np.zeros(r_ref.size, dtype=bool)
#             keep_full[np.where(good)[0][keep_good]] = True
#             Z[i, ~keep_full] = np.nan

#     # --- choose x axis ---
#     if x_mode == "log":
#         x = r_ref.copy()
#         xlab = "log return r"
#     else:
#         x = np.exp(r_ref)
#         xlab = "gross return R = exp(r)"

#     # --- apply x_bounds ---
#     if x_bounds is not None:
#         lo, hi = float(x_bounds[0]), float(x_bounds[1])
#         m = (x >= lo) & (x <= hi)
#         x = x[m]
#         Z = Z[:, m]

#     # --- z transform ---
#     if z_mode == "log":
#         Z_plot = np.log(np.maximum(Z, z_eps))
#         zlab = f"log({fP_key})"
#     else:
#         Z_plot = Z
#         zlab = fP_key

#     # --- title ---
#     ttl = title if title is not None else (
#         f"Physical Density 3D Surface (T_target={T_target:.6f}y, used={T_used:.6f}y, x_mode={x_mode})"
#         + (f", tails={tail_alphas}" if truncate_by_tails else "")
#     )

#     # --- mesh ---
#     y_num = mdates.date2num(kept_dates.to_pydatetime())
#     X, Y = np.meshgrid(x, y_num)

#     # ----------------------------
#     # Interactive (Plotly)
#     # ----------------------------
#     if interactive:
#         if interactive_engine != "plotly":
#             raise ValueError("interactive_engine currently only supports 'plotly'.")
#         try:
#             import plotly.graph_objects as go
#         except Exception as e:
#             raise ImportError("Plotly is required for interactive=True. Install with: pip install plotly") from e

#         y_labels = [d.strftime("%Y-%m-%d") for d in kept_dates]

#         fig = go.Figure(
#             data=go.Surface(
#                 x=x, y=y_labels, z=Z_plot,
#                 colorscale=cmap,
#                 showscale=True
#             )
#         )
#         fig.update_layout(
#             title=ttl,
#             scene=dict(xaxis_title=xlab, yaxis_title="date", zaxis_title=zlab),
#             margin=dict(l=0, r=0, t=40, b=0),
#         )

#         html_path = None
#         if save_html is not None:
#             html_path = Path(save_html)
#         elif save is not None and str(save).lower().endswith(".html"):
#             html_path = Path(save)

#         if html_path is not None:
#             html_path.parent.mkdir(parents=True, exist_ok=True)
#             fig.write_html(str(html_path), include_plotlyjs="cdn")

#         return fig

#     # ----------------------------
#     # Static (Matplotlib)
#     # ----------------------------
#     fig = plt.figure(figsize=(12, 7))
#     ax = fig.add_subplot(111, projection="3d")
#     surf = ax.plot_surface(X, Y, Z_plot, linewidth=0, antialiased=True, cmap=cmap)

#     ax.set_xlabel(xlab)
#     ax.set_ylabel("date")
#     ax.set_zlabel(zlab)
#     ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
#     ax.set_title(ttl)

#     fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
#     fig.tight_layout()

#     if save is not None:
#         save_path = Path(save)
#         if save_path.suffix == "":
#             save_path = save_path.with_suffix(".png")
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

#     return fig, ax


from typing import Optional, Union, Literal
import numpy as np
import pandas as pd




from typing import Optional, Literal
import numpy as np
import pandas as pd


from typing import Optional, Literal
import numpy as np
import pandas as pd

HorizonMethod = Literal["nearest", "interp"]

def extract_physical_moment_timeseries(
    result_dict: dict,
    moment_col: str,
    *,
    # choose ONE
    T_years: Optional[float] = None,
    T_days: Optional[float] = None,

    # ✅ NEW
    series_name: Optional[str] = None,

    method: HorizonMethod = "nearest",
    tol_years: Optional[float] = None,     # e.g. 3/365 for +/- 3 days
    date_format: Optional[str] = None,     # if your keys need parsing; else None
    drop_missing: bool = True,
    table_key: str = "physical_moments_table",
) -> pd.Series:
    """
    Extract a time series of a physical moment from each day's physical_moments_table
    at a chosen horizon.

    Expects per-day dict contains `table_key` (default 'physical_moments_table'),
    a DataFrame with:
      - column 'T' in YEARS
      - column `moment_col`

    Returns a pd.Series indexed by date.
    """
    if (T_years is None) == (T_days is None):
        raise ValueError("Provide exactly one of T_years or T_days.")

    T_target = float(T_years) if T_years is not None else float(T_days) / 365.0
    out = {}

    for k, day in result_dict.items():
        # --- parse date key ---
        try:
            dt = pd.to_datetime(k, format=date_format) if date_format else pd.to_datetime(k)
        except Exception:
            dt = k  # fallback: keep original key

        # --- fetch table ---
        if not isinstance(day, dict) or table_key not in day or day[table_key] is None:
            out[dt] = np.nan
            continue

        tbl = day[table_key]
        if not isinstance(tbl, pd.DataFrame):
            out[dt] = np.nan
            continue

        if "T" not in tbl.columns:
            raise KeyError(f"{table_key} must contain a 'T' column (years).")
        if moment_col not in tbl.columns:
            raise KeyError(
                f"'{moment_col}' not found in {table_key}. "
                f"Available columns: {list(tbl.columns)}"
            )

        T = pd.to_numeric(tbl["T"], errors="coerce").to_numpy()
        y = pd.to_numeric(tbl[moment_col], errors="coerce").to_numpy()

        mask = np.isfinite(T) & np.isfinite(y)
        T = T[mask]
        y = y[mask]

        if T.size == 0:
            out[dt] = np.nan
            continue

        # sort by maturity
        order = np.argsort(T)
        T = T[order]
        y = y[order]

        if method == "nearest":
            j = int(np.argmin(np.abs(T - T_target)))
            val = float(y[j])
            if tol_years is not None and abs(T[j] - T_target) > float(tol_years):
                val = np.nan
            out[dt] = val

        elif method == "interp":
            if T_target <= T[0]:
                val = float(y[0])
                if tol_years is not None and abs(T[0] - T_target) > float(tol_years):
                    val = np.nan
                out[dt] = val
            elif T_target >= T[-1]:
                val = float(y[-1])
                if tol_years is not None and abs(T[-1] - T_target) > float(tol_years):
                    val = np.nan
                out[dt] = val
            else:
                j_hi = int(np.searchsorted(T, T_target, side="right"))
                j_lo = j_hi - 1
                T0, T1 = float(T[j_lo]), float(T[j_hi])
                y0, y1 = float(y[j_lo]), float(y[j_hi])
                w = (T_target - T0) / (T1 - T0) if T1 != T0 else 0.0
                out[dt] = float((1 - w) * y0 + w * y1)
        else:
            raise ValueError("method must be 'nearest' or 'interp'.")

    s = pd.Series(out).sort_index()
    if drop_missing:
        s = s.dropna()

    # ✅ naming logic
    if series_name is not None:
        s.name = series_name
    else:
        s.name = f"physical:{moment_col}@{T_target:.6f}y"

    return s



# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.regression.quantile_regression import QuantReg
# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.regression.quantile_regression import QuantReg


# def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
#     """Circular moving-block bootstrap indices of length n."""
#     if block_len < 1:
#         raise ValueError("block_len must be >= 1")
#     if block_len == 1:
#         return rng.integers(0, n, size=n)

#     n_blocks = int(np.ceil(n / block_len))
#     starts = rng.integers(0, n, size=n_blocks)
#     idx = []
#     for s in starts:
#         idx.extend([(s + k) % n for k in range(block_len)])
#     return np.asarray(idx[:n], dtype=int)


# def _quantreg_fit_bootstrap(
#     y: np.ndarray,
#     X: np.ndarray,
#     *,
#     taus: tuple[float, ...],
#     B: int,
#     bootstrap: str,
#     block_len: int,
#     seed: int | None,
#     fit_kwargs: dict | None,
# ):
#     """Internal: fit QuantReg for each tau + bootstrap SEs."""
#     fit_kwargs = fit_kwargs or {}
#     rng = np.random.default_rng(seed)
#     n, p = X.shape

#     colnames = getattr(X, "columns", None)

#     params = {}
#     se_boot = {}
#     results = {}
#     boot_params = {}

#     for tau in taus:
#         res = QuantReg(y, X).fit(q=tau, **fit_kwargs)
#         results[tau] = res
#         params[tau] = res.params

#         b = np.zeros((B, p), float)
#         for j in range(B):
#             if bootstrap == "iid":
#                 idx = rng.integers(0, n, size=n)
#             elif bootstrap == "block":
#                 idx = _block_bootstrap_indices(n, block_len=block_len, rng=rng)
#             else:
#                 raise ValueError("bootstrap must be one of {'iid','block'}")

#             yb = y[idx]
#             Xb = X[idx, :]

#             ok = False
#             for _ in range(3):
#                 try:
#                     rb = QuantReg(yb, Xb).fit(q=tau, **fit_kwargs)
#                     b[j, :] = rb.params
#                     ok = True
#                     break
#                 except Exception:
#                     if bootstrap == "iid":
#                         idx = rng.integers(0, n, size=n)
#                     else:
#                         idx = _block_bootstrap_indices(n, block_len=block_len, rng=rng)
#                     yb = y[idx]
#                     Xb = X[idx, :]
#             if not ok:
#                 b[j, :] = np.nan

#         boot_params[tau] = b
#         se_boot[tau] = np.nanstd(b, axis=0, ddof=1)

#     return params, se_boot, results, boot_params


# def run_asym_moment_quantile_regressions(
#     *,
#     r_df: pd.DataFrame,          # must have ['date','ret_30'] (or whatever ret_col is)
#     var_s: pd.Series,            # Series named 'var' indexed by date OR with DatetimeIndex
#     skew_s: pd.Series,           # Series named 'skew'
#     kurt_s: pd.Series,           # Series named 'kurt'
#     date_col: str = "date",
#     ret_col: str = "ret_30",
#     horizon_label: str = "30d",  # just for naming outputs
#     n_ret_lags: int = 2,
#     n_mom_lags: int = 2,
#     taus=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
#     add_const: bool = True,
#     # bootstrap controls
#     bootstrap: str = "block",    # {"iid","block"}
#     B: int = 1000,
#     block_len: int = 10,
#     seed: int | None = 123,
#     fit_kwargs: dict | None = None,
#     dropna: bool = True,
# ) -> dict:
#     """
#     Implements paper-style regressions (A) Var, (B) Skew, (C) Kurt using:
#       - Koenker–Bassett linear quantile regression (statsmodels QuantReg)
#       - Bootstrap standard errors
#       - Separate regressions per tau

#     Data requirements
#     -----------------
#     r_df: DataFrame with columns [date_col, ret_col] (backward-looking horizon returns)
#     var_s, skew_s, kurt_s: Series indexed by date (DatetimeIndex) with names 'var','skew','kurt'
#                            OR pass series with any name; they'll be renamed internally.

#     Returns
#     -------
#     dict with keys:
#       - "data": modeling DataFrame (merged + features)
#       - "A_var": regression output dict
#       - "B_skew": regression output dict
#       - "C_kurt": regression output dict

#     Each regression output dict contains:
#       - params: DataFrame index=taus cols=regressors
#       - se_boot: DataFrame
#       - t_boot: DataFrame
#       - p_boot_norm: DataFrame (normal approx)
#       - results: dict[tau] -> statsmodels fit
#       - boot_params: dict[tau] -> (B,p) bootstrap draws
#       - X_cols: list of regressor names used
#     """
#     # --- normalize inputs ---
#     r = r_df[[date_col, ret_col]].copy()
#     r[date_col] = pd.to_datetime(r[date_col])

#     def _series_to_df(s: pd.Series, name: str) -> pd.DataFrame:
#         ss = s.copy()
#         ss.name = name
#         if not isinstance(ss.index, pd.DatetimeIndex):
#             # allow series with date index stored as strings
#             ss.index = pd.to_datetime(ss.index)
#         return ss.to_frame().reset_index().rename(columns={"index": date_col})

#     var_df = _series_to_df(var_s, "var")
#     skew_df = _series_to_df(skew_s, "skew")
#     kurt_df = _series_to_df(kurt_s, "kurt")

#     # --- merge on date ---
#     df = r.merge(var_df, on=date_col, how="inner") \
#           .merge(skew_df, on=date_col, how="inner") \
#           .merge(kurt_df, on=date_col, how="inner") \
#           .sort_values(date_col) \
#           .reset_index(drop=True)

#     # --- first differences of moments ---
#     df["d_var"] = df["var"].diff()
#     df["d_skew"] = df["skew"].diff()
#     df["d_kurt"] = df["kurt"].diff()

#     # --- split returns into + / - parts (paper asymmetry) ---
#     df["ret_pos"] = df[ret_col].clip(lower=0.0)
#     df["ret_neg"] = df[ret_col].clip(upper=0.0)

#     # --- return lags (include L=0 and 1..n_ret_lags) ---
#     for L in range(1, n_ret_lags + 1):
#         df[f"ret_pos_L{L}"] = df["ret_pos"].shift(L)
#         df[f"ret_neg_L{L}"] = df["ret_neg"].shift(L)

#     # --- moment-diff lags (usually 1..n_mom_lags) ---
#     for L in range(1, n_mom_lags + 1):
#         df[f"d_var_L{L}"] = df["d_var"].shift(L)
#         df[f"d_skew_L{L}"] = df["d_skew"].shift(L)
#         df[f"d_kurt_L{L}"] = df["d_kurt"].shift(L)

#     if dropna:
#         df_model = df.dropna().copy()
#     else:
#         df_model = df.copy()

#     # --- helper to run one equation ---
#     def _run_eq(dep: str, X_cols: list[str]) -> dict:
#         y = df_model[dep].to_numpy(float)
#         X = df_model[X_cols].astype(float)
#         if add_const:
#             X = sm.add_constant(X, has_constant="add")
#         Xmat = X.to_numpy(float)
#         colnames = list(X.columns)

#         taus_t = tuple(float(t) for t in taus)
#         params_dict, se_dict, results, boot_params = _quantreg_fit_bootstrap(
#             y=y,
#             X=Xmat,
#             taus=taus_t,
#             B=B,
#             bootstrap=bootstrap,
#             block_len=block_len,
#             seed=seed,
#             fit_kwargs=fit_kwargs,
#         )

#         params = pd.DataFrame([params_dict[t] for t in taus_t], index=taus_t, columns=colnames)
#         se_boot = pd.DataFrame([se_dict[t] for t in taus_t], index=taus_t, columns=colnames)
#         t_boot = params / se_boot

#         # normal-approx p-values (two-sided); matches common applied reporting
#         from math import erf, sqrt
#         def pval_from_t(tt):
#             Phi = 0.5 * (1.0 + erf(abs(float(tt)) / sqrt(2.0)))
#             return 2.0 * (1.0 - Phi)

#         p_boot_norm = t_boot.applymap(pval_from_t)

#         return {
#             "dep": dep,
#             "X_cols": colnames,
#             "params": params,
#             "se_boot": se_boot,
#             "t_boot": t_boot,
#             "p_boot_norm": p_boot_norm,
#             "results": results,
#             "boot_params": boot_params,
#             "meta": {
#                 "taus": taus_t,
#                 "bootstrap": bootstrap,
#                 "B": int(B),
#                 "block_len": int(block_len),
#                 "n_used": int(len(df_model)),
#                 "horizon_label": horizon_label,
#                 "n_ret_lags": int(n_ret_lags),
#                 "n_mom_lags": int(n_mom_lags),
#                 "ret_col": ret_col,
#             },
#         }

#     # --- build regressor lists for (A), (B), (C) ---
#     ret_terms = ["ret_pos", "ret_neg"] + \
#                 [f"ret_pos_L{L}" for L in range(1, n_ret_lags + 1)] + \
#                 [f"ret_neg_L{L}" for L in range(1, n_ret_lags + 1)]

#     # (A) ΔVar_t
#     X_A = ret_terms + \
#           [f"d_var_L{L}" for L in range(1, n_mom_lags + 1)] + \
#           [f"d_skew_L{L}" for L in range(1, n_mom_lags + 1)] + \
#           [f"d_kurt_L{L}" for L in range(1, n_mom_lags + 1)]

#     # (B) ΔSkew_t
#     X_B = ret_terms + \
#           [f"d_skew_L{L}" for L in range(1, n_mom_lags + 1)] + \
#           [f"d_var_L{L}" for L in range(1, n_mom_lags + 1)] + \
#           [f"d_kurt_L{L}" for L in range(1, n_mom_lags + 1)]

#     # (C) ΔKurt_t
#     X_C = ret_terms + \
#           [f"d_kurt_L{L}" for L in range(1, n_mom_lags + 1)] + \
#           [f"d_var_L{L}" for L in range(1, n_mom_lags + 1)] + \
#           [f"d_skew_L{L}" for L in range(1, n_mom_lags + 1)]

#     out = {
#         "data": df_model,
#         "A_var": _run_eq("d_var", X_A),
#         "B_skew": _run_eq("d_skew", X_B),
#         "C_kurt": _run_eq("d_kurt", X_C),
#     }
#     return out

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import statsmodels.api as sm


# def plot_asym_moment_quantile_coeffs(
#     out: dict,
#     *,
#     # which coefficient(s) to plot
#     coef: str | list[str] = "ret_pos",
#     # which equations to plot
#     eq_keys: tuple[str, ...] = ("A_var", "B_skew", "C_kurt"),
#     # plotting options
#     ci: float | None = 0.95,          # None -> no CI; else e.g. 0.95
#     use_boot_se: bool = True,         # use out[eq]["se_boot"] for CI
#     show_ols: bool = True,            # add mean-regression horizontal line
#     ols_hac_lags: int | None = None,  # if not None, use HAC(Newey-West) SEs for OLS line fit (slope unchanged)
#     title_prefix: str = "",
#     figsize: tuple[float, float] = (9.5, 5.0),
#     panel: bool = False,              # True -> 3 stacked panels in one figure
#     legend: bool = True,
# ) -> dict:
#     """
#     Plot quantile regression coefficient curves β(τ) for each moment equation from
#     run_asym_moment_quantile_regressions output.

#     Parameters
#     ----------
#     out : dict
#         Output dict from run_asym_moment_quantile_regressions().
#     coef : str or list[str]
#         Regressor name(s) to plot (e.g. "ret_pos", "ret_neg", ["ret_pos","ret_neg"]).
#         Must match column names in out[eq]["params"].
#     eq_keys : tuple[str,...]
#         Which equations to plot. Defaults to all three: ("A_var","B_skew","C_kurt").
#     ci : float or None
#         Confidence level for bands (e.g., 0.95). If None, no CI.
#     use_boot_se : bool
#         Use bootstrap SEs stored in out[eq]["se_boot"]. (This matches your estimator setup.)
#     show_ols : bool
#         Fit an OLS mean regression with same regressors and plot its slope as a horizontal line.
#     ols_hac_lags : int or None
#         If provided, uses HAC covariance for OLS fit (slope line same; mostly for reporting).
#     panel : bool
#         If True, returns a single figure with stacked panels (one per equation).
#         If False, returns separate figures per equation.

#     Returns
#     -------
#     dict with keys:
#         - "fig" if panel=True else "figs" dict[eq_key] -> fig
#         - "axes" similarly
#     """
#     coefs = [coef] if isinstance(coef, str) else list(coef)

#     # mapping for nicer titles
#     pretty = {
#         "A_var":  "ΔVar",
#         "B_skew": "ΔSkew",
#         "C_kurt": "ΔKurt",
#     }

#     # z-value for normal approx CI (you used normal approx p-values already)
#     if ci is not None:
#         from scipy.stats import norm
#         z = float(norm.ppf(0.5 + ci / 2.0))
#     else:
#         z = None
#     def _ols_line(eq_key: str, coef_name: str) -> float:
#         """OLS slope for the same dependent and regressor set used in quantile reg."""
#         df = out["data"]
#         dep = out[eq_key]["dep"]
#         X_cols = list(out[eq_key]["X_cols"])  # includes 'const' if you used add_const=True
    
#         # remove const if it's in X_cols (it won't be a real column in df)
#         has_const = ("const" in X_cols)
#         if has_const:
#             X_cols_wo = [c for c in X_cols if c != "const"]
#         else:
#             X_cols_wo = X_cols
    
#         # build design matrix from df, then add constant the same way as in _run_eq
#         X = df[X_cols_wo].astype(float).copy()
#         if has_const:
#             X = sm.add_constant(X, has_constant="add")
    
#         y = df[dep].astype(float)
    
#         model = sm.OLS(y.to_numpy(float), X.to_numpy(float))
#         if ols_hac_lags is None:
#             res = model.fit()
#         else:
#             res = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(ols_hac_lags)})
    
#         # coef_name might be in columns as-is
#         cols = list(X.columns)
#         if coef_name not in cols:
#             raise KeyError(f"coef '{coef_name}' not found in OLS X columns for {eq_key}. Columns: {cols}")
#         j = cols.index(coef_name)
#         return float(res.params[j])

#     # ---- plotting ----
#     if panel:
#         fig, axes = plt.subplots(len(eq_keys), 1, figsize=(figsize[0], figsize[1] * len(eq_keys)), sharex=True)
#         if len(eq_keys) == 1:
#             axes = [axes]
#         figs = {"fig": fig}
#         axes_out = {"axes": dict(zip(eq_keys, axes))}
#     else:
#         figs = {}
#         axes_out = {}

#     for i, eq_key in enumerate(eq_keys):
#         eq = out[eq_key]
#         taus = np.array(eq["params"].index, dtype=float)

#         if panel:
#             ax = axes[i]
#         else:
#             fig, ax = plt.subplots(1, 1, figsize=figsize)
#             figs[eq_key] = fig
#             axes_out[eq_key] = ax

#         for coef_name in coefs:
#             if coef_name not in eq["params"].columns:
#                 raise KeyError(
#                     f"coef '{coef_name}' not in out['{eq_key}']['params'].columns. "
#                     f"Available: {list(eq['params'].columns)}"
#                 )

#             beta = eq["params"][coef_name].to_numpy(float)

#             ax.plot(taus, beta, marker="o", linewidth=2, label=f"QRM: {coef_name}")

#             # CI band
#             if (ci is not None) and use_boot_se:
#                 se = eq["se_boot"][coef_name].to_numpy(float)
#                 lo = beta - z * se
#                 hi = beta + z * se
#                 ax.fill_between(taus, lo, hi, alpha=0.15)

#             # OLS horizontal line
#             if show_ols:
#                 b_ols = _ols_line(eq_key, coef_name)
#                 ax.axhline(b_ols, linestyle="--", linewidth=1.5, label=f"MRM (OLS): {coef_name}")

#         ax.set_ylabel("Coefficient (β)")
#         title = f"{title_prefix}{pretty.get(eq_key, eq_key)}: quantile slopes β(τ)"
#         ax.set_title(title)
#         ax.grid(True, alpha=0.25)

#         if legend:
#             ax.legend(loc="best")

#     if panel:
#         axes[-1].set_xlabel("Quantile (τ)")
#         fig.tight_layout()
#         return {"fig": fig, "axes": axes_out["axes"]}

#     else:
#         # add x-label to each
#         for eq_key in eq_keys:
#             axes_out[eq_key].set_xlabel("Quantile (τ)")
#             figs[eq_key].tight_layout()
#         return {"figs": figs, "axes": axes_out}

# import numpy as np
# import matplotlib.pyplot as plt


# def plot_qrm_across_quantiles(
#     res_by_key,
#     *,
#     eq_key: str = "A_var",                 # "A_var", "B_skew", "C_kurt"
#     coef_pos: str = "ret_pos",
#     coef_neg: str = "ret_neg",
#     keys_order=None,                       # optional explicit order for legend/lines
#     ci: float | None = 0.95,               # None => no CI shading
#     scale: float = 1.0,                    # use 100.0 if you want "Responses (%)"
#     title: str | None = None,
#     figsize=(10, 8),
#     legend: bool = True,
# ):
#     """
#     Fig-3 style: For each moment equation, plot β(τ) across quantiles with one line per
#     frequency/horizon. Two panels: positive-return coefficient and negative-return coefficient.

#     Parameters
#     ----------
#     res_by_key : dict or list
#         dict[label -> res_dict] OR list of (label, res_dict).
#         Each res_dict is the output from run_asym_moment_quantile_regressions.
#     eq_key : str
#         "A_var", "B_skew", or "C_kurt".
#     coef_pos / coef_neg : str
#         Column names in res[eq_key]["params"] to plot.
#     ci : float or None
#         If not None, uses bootstrap SEs res[eq_key]["se_boot"] to shade +/- z*SE.
#     scale : float
#         Multiply coefficients/CI by this (100.0 => percent units).
#     """
#     # normalize input to list[(label,res)]
#     if isinstance(res_by_key, dict):
#         items = list(res_by_key.items())
#     else:
#         items = list(res_by_key)

#     if keys_order is not None:
#         order_map = {k: i for i, k in enumerate(keys_order)}
#         items.sort(key=lambda kv: order_map.get(kv[0], 10**9))

#     # find common taus across all results
#     tau_sets = []
#     for _, res in items:
#         taus = np.array(res[eq_key]["params"].index, dtype=float)
#         tau_sets.append(set(np.round(taus, 10)))
#     common = sorted(set.intersection(*tau_sets))
#     taus_common = np.array(common, dtype=float)

#     # z for CI
#     z = None
#     if ci is not None:
#         from math import erf, sqrt
#         # approximate z via inverse error function is annoying; simplest: hardcode common 95%
#         # but keep generic with scipy if available; else fallback
#         try:
#             from scipy.stats import norm
#             z = float(norm.ppf(0.5 + ci / 2.0))
#         except Exception:
#             # fallback: 95% approx
#             z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96

#     fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
#     ax_pos, ax_neg = axes

#     for label, res in items:
#         eq = res[eq_key]
#         params = eq["params"].copy()
#         se = eq.get("se_boot", None)

#         # align to common taus
#         params = params.loc[taus_common]
#         bpos = params[coef_pos].to_numpy(float) * scale
#         bneg = params[coef_neg].to_numpy(float) * scale

#         ax_pos.plot(taus_common, bpos, marker="o", linewidth=2, label=str(label))
#         ax_neg.plot(taus_common, bneg, marker="o", linewidth=2, label=str(label))

#         if (ci is not None) and (se is not None) and (coef_pos in se.columns) and (coef_neg in se.columns):
#             se_al = se.loc[taus_common]
#             sepos = se_al[coef_pos].to_numpy(float) * scale
#             seneg = se_al[coef_neg].to_numpy(float) * scale

#             ax_pos.fill_between(taus_common, bpos - z * sepos, bpos + z * sepos, alpha=0.15)
#             ax_neg.fill_between(taus_common, bneg - z * seneg, bneg + z * seneg, alpha=0.15)

#     ax_pos.set_title(title or f"{eq_key}: coefficient curves across quantiles")
#     ax_pos.set_ylabel(f"{coef_pos} β(τ)" + (" (scaled)" if scale != 1.0 else ""))
#     ax_neg.set_ylabel(f"{coef_neg} β(τ)" + (" (scaled)" if scale != 1.0 else ""))
#     ax_neg.set_xlabel("Quantile (τ)")

#     ax_pos.grid(True, alpha=0.25)
#     ax_neg.grid(True, alpha=0.25)

#     if legend:
#         ax_pos.legend(loc="best", title="Frequency / Horizon")
#         ax_neg.legend(loc="best", title="Frequency / Horizon")

#     fig.tight_layout()
#     return fig, axes


# def plot_qrm_by_quantile_across_frequencies(
#     res_by_key,
#     *,
#     eq_key: str = "A_var",
#     coef_pos: str = "ret_pos",
#     coef_neg: str = "ret_neg",
#     taus_to_plot=(0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95),
#     keys_order=None,                       # x-axis order
#     ncols: int = 2,
#     ci: float | None = None,               # optional CI bars (uses +/- z*SE)
#     scale: float = 1.0,                    # use 100.0 for percent units
#     title: str | None = None,
#     figsize_per_panel=(5.4, 3.2),
#     legend: bool = True,
# ):
#     """
#     Fig-4 style: small multiples. Each subplot corresponds to a quantile τ.
#     Within each subplot, x-axis is frequency/horizon, and you plot two lines:
#     β_pos(τ) and β_neg(τ).

#     Parameters
#     ----------
#     res_by_key : dict or list
#         dict[label -> res_dict] OR list of (label, res_dict).
#     taus_to_plot : iterable
#         Quantiles to show as small-multiple panels.
#     ci : float or None
#         If provided, draws vertical error bars +/- z*SE for each point (bootstrap SE).
#     """
#     # normalize input
#     if isinstance(res_by_key, dict):
#         items = list(res_by_key.items())
#     else:
#         items = list(res_by_key)

#     if keys_order is not None:
#         order_map = {k: i for i, k in enumerate(keys_order)}
#         items.sort(key=lambda kv: order_map.get(kv[0], 10**9))

#     labels = [str(k) for k, _ in items]

#     # choose available taus closest to requested (per result); enforce common set by rounding
#     # We'll map each requested tau -> nearest tau in each result (within tolerance)
#     def _nearest_tau_index(taus_arr, t):
#         taus_arr = np.array(taus_arr, dtype=float)
#         j = int(np.argmin(np.abs(taus_arr - float(t))))
#         return float(taus_arr[j])

#     # z for CI
#     z = None
#     if ci is not None:
#         try:
#             from scipy.stats import norm
#             z = float(norm.ppf(0.5 + ci / 2.0))
#         except Exception:
#             z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96

#     taus_panels = list(taus_to_plot)
#     n_panels = len(taus_panels)
#     nrows = int(np.ceil(n_panels / ncols))

#     fig_w = figsize_per_panel[0] * ncols
#     fig_h = figsize_per_panel[1] * nrows
#     fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

#     # pre-extract for speed
#     extracted = []
#     for label, res in items:
#         eq = res[eq_key]
#         taus_av = np.array(eq["params"].index, dtype=float)
#         params = eq["params"]
#         se = eq.get("se_boot", None)
#         extracted.append((str(label), taus_av, params, se))

#     for i, t_req in enumerate(taus_panels):
#         r = i // ncols
#         c = i % ncols
#         ax = axes[r, c]

#         # x positions are 0..K-1, labeled by frequency/horizon label
#         x = np.arange(len(labels), dtype=float)

#         y_pos = np.zeros(len(labels), float)
#         y_neg = np.zeros(len(labels), float)
#         e_pos = np.zeros(len(labels), float) if (ci is not None) else None
#         e_neg = np.zeros(len(labels), float) if (ci is not None) else None

#         for j, (lab, taus_av, params, se) in enumerate(extracted):
#             t_use = _nearest_tau_index(taus_av, t_req)

#             y_pos[j] = float(params.loc[t_use, coef_pos]) * scale
#             y_neg[j] = float(params.loc[t_use, coef_neg]) * scale

#             if ci is not None and se is not None:
#                 e_pos[j] = float(se.loc[t_use, coef_pos]) * scale
#                 e_neg[j] = float(se.loc[t_use, coef_neg]) * scale

#         # lines
#         ax.plot(x, y_pos, marker="o", linewidth=2, label=f"{coef_pos}")
#         ax.plot(x, y_neg, marker="s", linewidth=2, label=f"{coef_neg}")

#         # optional error bars
#         if ci is not None:
#             ax.errorbar(x, y_pos, yerr=z * e_pos, fmt="none", capsize=3, alpha=0.8)
#             ax.errorbar(x, y_neg, yerr=z * e_neg, fmt="none", capsize=3, alpha=0.8)

#         ax.set_title(f"{eq_key}: τ≈{t_req:.2f}")
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, rotation=0)
#         ax.set_ylabel("Response" + (" (%)" if scale == 100.0 else ""))
#         ax.grid(True, alpha=0.25)

#         if legend:
#             ax.legend(loc="best")

#     # hide unused axes
#     for k in range(n_panels, nrows * ncols):
#         r = k // ncols
#         c = k % ncols
#         axes[r, c].axis("off")

#     if title is not None:
#         fig.suptitle(title, y=0.995)

#     fig.tight_layout()
#     return fig, axes

# def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
#     """
#     Circular moving-block bootstrap indices of length n.
#     """
#     if block_len < 1:
#         raise ValueError("block_len must be >= 1")
#     if block_len == 1:
#         return rng.integers(0, n, size=n)

#     n_blocks = int(np.ceil(n / block_len))
#     starts = rng.integers(0, n, size=n_blocks)
#     idx = []
#     for s in starts:
#         idx.extend([(s + k) % n for k in range(block_len)])
#     return np.asarray(idx[:n], dtype=int)


# def quantreg_koenker_bassett_bootstrap(
#     y,
#     X,
#     *,
#     taus=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
#     add_const: bool = True,
#     bootstrap: str = "block",     # {"iid","block"}
#     B: int = 500,
#     block_len: int = 10,
#     seed: int | None = 123,
#     fit_kwargs: dict | None = None,
# ) -> dict:
#     """
#     Koenker–Bassett linear quantile regression + bootstrap SEs, estimated separately for each tau.

#     Parameters
#     ----------
#     y : array-like or pd.Series
#         Dependent variable (n,).
#     X : array-like or pd.DataFrame
#         Regressors (n, p). You provide whatever variables you want.
#     taus : iterable
#         Quantiles to estimate (each tau estimated in its own regression).
#     add_const : bool
#         If True, adds an intercept column.
#     bootstrap : {"iid","block"}
#         IID bootstrap or (circular) moving-block bootstrap (recommended for time series).
#     B : int
#         Number of bootstrap replications for SEs.
#     block_len : int
#         Block length for block bootstrap.
#     seed : int or None
#         RNG seed.
#     fit_kwargs : dict or None
#         Extra kwargs passed to QuantReg.fit(q=tau, ...). Example:
#         {"max_iter": 5000}

#     Returns
#     -------
#     out : dict with keys
#         - "params": DataFrame (index=taus, columns=regressors)
#         - "se_boot": DataFrame bootstrap SEs
#         - "t_boot": DataFrame t-stats (params / se_boot)
#         - "p_boot_norm": DataFrame p-values using normal approx (two-sided)
#         - "results": dict[tau] -> fitted statsmodels result (original sample)
#         - "boot_params": dict[tau] -> (B, p) array of bootstrap estimates
#     """
#     fit_kwargs = fit_kwargs or {}
#     rng = np.random.default_rng(seed)

#     # --- coerce to arrays + preserve names ---
#     y_ser = pd.Series(y).astype(float).reset_index(drop=True)
#     if isinstance(X, pd.DataFrame):
#         X_df = X.copy()
#     else:
#         X_df = pd.DataFrame(X)

#     X_df = X_df.reset_index(drop=True)

#     # drop rows with any NaNs
#     df = pd.concat([y_ser.rename("y"), X_df], axis=1).dropna()
#     yv = df["y"].to_numpy(float)
#     Xv = df.drop(columns=["y"])

#     if add_const:
#         Xv = sm.add_constant(Xv, has_constant="add")

#     Xmat = np.asarray(Xv, float)
#     colnames = list(Xv.columns)
#     n, p = Xmat.shape

#     if bootstrap not in {"iid", "block"}:
#         raise ValueError("bootstrap must be one of {'iid','block'}")

#     taus = tuple(float(t) for t in taus)

#     params = pd.DataFrame(index=taus, columns=colnames, dtype=float)
#     se_boot = pd.DataFrame(index=taus, columns=colnames, dtype=float)

#     results = {}
#     boot_params = {}

#     # helper: normal approx p-values from t-stats
#     def _pval_from_t(t):
#         # 2-sided normal approximation (good with B large; matches common practice)
#         from math import erf, sqrt
#         # Phi(|t|)
#         Phi = 0.5 * (1.0 + erf(np.abs(t) / sqrt(2.0)))
#         return 2.0 * (1.0 - Phi)

#     for tau in taus:
#         # --- original sample fit ---
#         mod = QuantReg(yv, Xmat)
#         res = mod.fit(q=tau, **fit_kwargs)
#         results[tau] = res
#         params.loc[tau, :] = res.params

#         # --- bootstrap ---
#         b = np.zeros((B, p), float)
#         for j in range(B):
#             if bootstrap == "iid":
#                 idx = rng.integers(0, n, size=n)
#             else:
#                 idx = _block_bootstrap_indices(n, block_len=block_len, rng=rng)

#             yb = yv[idx]
#             Xb = Xmat[idx, :]

#             # QuantReg can sometimes fail to converge on weird resamples;
#             # if it does, we retry a couple times then keep NaNs.
#             ok = False
#             for _ in range(3):
#                 try:
#                     rb = QuantReg(yb, Xb).fit(q=tau, **fit_kwargs)
#                     b[j, :] = rb.params
#                     ok = True
#                     break
#                 except Exception:
#                     # new resample
#                     if bootstrap == "iid":
#                         idx = rng.integers(0, n, size=n)
#                     else:
#                         idx = _block_bootstrap_indices(n, block_len=block_len, rng=rng)
#                     yb = yv[idx]
#                     Xb = Xmat[idx, :]
#             if not ok:
#                 b[j, :] = np.nan

#         boot_params[tau] = b

#         # bootstrap SE = std of bootstrap estimates (ignore failed draws)
#         se = np.nanstd(b, axis=0, ddof=1)
#         se_boot.loc[tau, :] = se

#     t_boot = params / se_boot
#     p_boot_norm = t_boot.applymap(_pval_from_t)

#     out = {
#         "params": params,
#         "se_boot": se_boot,
#         "t_boot": t_boot,
#         "p_boot_norm": p_boot_norm,
#         "results": results,
#         "boot_params": boot_params,
#         "meta": {
#             "n_used": int(n),
#             "p": int(p),
#             "colnames": colnames,
#             "taus": taus,
#             "bootstrap": bootstrap,
#             "B": int(B),
#             "block_len": int(block_len),
#             "add_const": bool(add_const),
#         },
#     }
#     return out


# def quantreg_table(out: dict, *, stars: bool = True, digits: int = 4) -> dict:
#     """
#     Convenience: format a 'paper-style' coefficient table per tau.
#     Returns dict[tau] -> DataFrame with coef, (se), t, p.
#     """
#     params = out["params"]
#     se = out["se_boot"]
#     t = out["t_boot"]
#     p = out["p_boot_norm"]

#     tables = {}
#     for tau in params.index:
#         df = pd.DataFrame(
#             {
#                 "coef": params.loc[tau],
#                 "se": se.loc[tau],
#                 "t": t.loc[tau],
#                 "p": p.loc[tau],
#             }
#         )

#         if stars:
#             def _star(pp):
#                 if not np.isfinite(pp):
#                     return ""
#                 if pp < 0.01: return "***"
#                 if pp < 0.05: return "**"
#                 if pp < 0.10: return "*"
#                 return ""
#             df["coef_star"] = df.apply(lambda r: f"{r['coef']:.{digits}f}{_star(r['p'])}", axis=1)
#             df["(se)"] = df["se"].map(lambda x: f"({x:.{digits}f})" if np.isfinite(x) else "(nan)")
#             tables[tau] = df[["coef_star", "(se)", "t", "p"]]
#         else:
#             tables[tau] = df.round(digits)

#     return tables


import numpy as np
import pandas as pd


import numpy as np
import pandas as pd


import numpy as np
import pandas as pd


# def compute_horizon_returns_backward(
#     stock_df: pd.DataFrame,
#     *,
#     horizon: int,
#     price_col: str = "price",
#     date_col: str = "date",
#     split_col: str = "ajexdi",
#     return_type: str = "log",     # {"log","simple"}
#     group_col: str | None = "tic",
#     sort: bool = True,
#     return_dataframe: bool = True,
# ) -> pd.DataFrame | pd.Series:
#     """
#     Backward-looking (trailing) horizon returns:
#         r_t(h) = log(P_t) - log(P_{t-h})   [log]
#         r_t(h) = P_t / P_{t-h} - 1         [simple]

#     Uses split-adjusted prices if `split_col` exists; otherwise uses raw prices.

#     Returns
#     -------
#     pd.DataFrame (default)
#         Columns: [date_col, f"ret_{horizon}"]
#     or
#     pd.Series
#         If return_dataframe=False.
#     """
#     df = stock_df.copy()
#     df[date_col] = pd.to_datetime(df[date_col])

#     if group_col is not None:
#         groups = df.groupby(group_col, sort=False)
#     else:
#         groups = [(None, df)]

#     ret = pd.Series(index=df.index, dtype=float)

#     for _, g in groups:
#         g = g.copy()
#         if sort:
#             g = g.sort_values(date_col)

#         # split-adjusted price if available
#         if split_col in g.columns:
#             adj_price = g[price_col].astype(float) / g[split_col].astype(float)
#         else:
#             adj_price = g[price_col].astype(float)

#         p = adj_price.to_numpy(float)

#         # backward-looking: compare to t-horizon
#         p_lag = np.roll(p, horizon)  # p_{t-h} aligned with p_t

#         if return_type == "log":
#             r = np.log(p) - np.log(p_lag)
#         elif return_type == "simple":
#             r = (p / p_lag) - 1.0
#         else:
#             raise ValueError("return_type must be 'log' or 'simple'")

#         # first `horizon` obs have no lagged price
#         r[:horizon] = np.nan

#         ret.loc[g.index] = r

#     if not return_dataframe:
#         return ret

#     out = pd.DataFrame(
#         {
#             date_col: df[date_col],
#             f"ret_{horizon}": ret,
#         }
#     )
#     return out

