import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .base import ForecastDensity
from .windows import get_fit_dates


class RNDDensityEvaluator:
    def __init__(
        self,
        rnd_dict,
        logreturns,
        target_maturity=30 / 365,
        maturity_match_tol=None,
        return_col="ret_1",
        date_col="date",
        reserve_obs=0,
        model_name="RND",
    ):
        self.rnd_dict = rnd_dict
        self.logreturns = logreturns.copy()
        self.target_maturity = float(target_maturity)
        self.maturity_match_tol = maturity_match_tol
        self.horizon_days = int(round(365 * self.target_maturity))
        self.return_col = return_col
        self.date_col = date_col
        self.reserve_obs = int(reserve_obs)
        self.model_name = model_name

        self.ret_series = self._standardize_logreturns(self.logreturns)

    def _standardize_logreturns(self, logreturns):
        if isinstance(logreturns, pd.Series):
            s = logreturns.copy()
            s.index = pd.to_datetime(s.index)
            return s.sort_index().astype(float)

        df = logreturns.copy()

        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.sort_values(self.date_col).set_index(self.date_col)
        else:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

        if self.return_col in df.columns:
            s = df[self.return_col]
        elif df.shape[1] == 1:
            s = df.iloc[:, 0]
        else:
            raise ValueError(f"Could not find return_col='{self.return_col}'.")

        return s.astype(float).sort_index()

    def _realized_horizon_return(self, date):
        date = pd.Timestamp(date).tz_localize(None)
        future = self.ret_series[self.ret_series.index > date].dropna()

        if len(future) < self.horizon_days:
            return None, None

        realized = float(future.iloc[:self.horizon_days].sum())
        end_date = future.index[self.horizon_days - 1]
        return realized, end_date

    def _select_maturity_index(self, T_grid):
        T_grid = np.asarray(T_grid, dtype=float)
        j = int(np.argmin(np.abs(T_grid - self.target_maturity)))
        T_actual = float(T_grid[j])

        if self.maturity_match_tol is not None:
            if abs(T_actual - self.target_maturity) > self.maturity_match_tol:
                return None, None

        return j, T_actual

    def _get_rnd_cdf_surface(self, info):
        if "cdf_lr_surface" in info:
            return np.asarray(info["cdf_lr_surface"], dtype=float)

        if "rnd_cdf_surface" in info:
            return np.asarray(info["rnd_cdf_surface"], dtype=float)

        if "rnd_lr_surface" not in info:
            raise KeyError("Need either cdf_lr_surface, rnd_cdf_surface, or rnd_lr_surface.")

        x_grid = np.asarray(info["grid_lr"], dtype=float)
        density = np.asarray(info["rnd_lr_surface"], dtype=float)

        out = []

        for row in density:
            row = np.asarray(row, dtype=float)
            row = np.where(np.isfinite(row) & (row >= 0), row, 0.0)

            dx = np.diff(x_grid)
            inc = 0.5 * (row[1:] + row[:-1]) * dx

            cdf = np.empty_like(x_grid)
            cdf[0] = 0.0
            cdf[1:] = np.cumsum(inc)

            if cdf[-1] > 0:
                cdf = cdf / cdf[-1]

            out.append(cdf)

        return np.vstack(out)

    def run(
        self,
        start_i=None,
        end_i=None,
        save_path=None,
        eval_step=None,
        progress_every=100,
        verbose=True,
    ):
        rnd_by_ts = {
            pd.Timestamp(k).tz_localize(None): v
            for k, v in self.rnd_dict.items()
        }
    
        dates = sorted(rnd_by_ts.keys())
    
        if start_i is None:
            start_i = 0
        if end_i is None:
            end_i = len(dates)
    
        date_items = list(enumerate(dates))[start_i:end_i]
    
        if eval_step is None:
            eval_step = 1
    
        eval_step = int(eval_step)
        if eval_step < 1:
            raise ValueError("eval_step must be >= 1.")
    
        date_items = date_items[::eval_step]
    
        forecasts = []
    
        if verbose:
            print(
                f"[{self.model_name}] starting evaluation "
                f"| dates={len(date_items)} "
                f"| start_i={start_i} | end_i={end_i} "
                f"| eval_step={eval_step} "
                f"| window=none "
                f"| reserve_obs={self.reserve_obs} "
                f"| target_maturity={self.target_maturity:.6f}y "
                f"({self.horizon_days}d)",
                flush=True,
            )
    
        for count, (i, date) in enumerate(date_items, start=1):
            if verbose and (
                count == 1
                or count % progress_every == 0
                or count == len(date_items)
            ):
                print(
                    f"[{self.model_name}] {count}/{len(date_items)} "
                    f"| date={date.date()} | index={i} "
                    f"| stored={len(forecasts)}",
                    flush=True,
                )
    
            if i < self.reserve_obs:
                continue
    
            realized, end_date = self._realized_horizon_return(date)
            if realized is None:
                continue
    
            info = rnd_by_ts[date]
    
            if not info.get("success", True):
                continue
    
            T_grid = np.asarray(info["T_grid"], dtype=float)
            j, T_actual = self._select_maturity_index(T_grid)
    
            if j is None:
                continue
    
            x_grid = np.asarray(info["grid_lr"], dtype=float)
            pdf = np.asarray(info["rnd_lr_surface"], dtype=float)[j, :]
            cdf = self._get_rnd_cdf_surface(info)[j, :]
    
            forecasts.append(
                ForecastDensity(
                    date=date,
                    horizon=self.horizon_days,
                    x_grid=x_grid,
                    pdf=pdf,
                    cdf=cdf,
                    realized=realized,
                    model_name=self.model_name,
                )
            )
    
        if save_path is not None:
            payload = {
                "forecasts": forecasts,
                "start_i": start_i,
                "end_i": end_i,
                "eval_step": eval_step,
                "target_maturity": self.target_maturity,
                "horizon_days": self.horizon_days,
                "model_name": self.model_name,
            }
    
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
    
            with open(save_path, "wb") as f:
                pickle.dump(payload, f)
    
            if verbose:
                print(f"[{self.model_name}] saved -> {save_path}", flush=True)
    
        if verbose:
            print(
                f"[{self.model_name}] finished | forecasts={len(forecasts)}",
                flush=True,
            )
    
        return forecasts