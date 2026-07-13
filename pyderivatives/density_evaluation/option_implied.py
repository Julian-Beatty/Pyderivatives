# pyderivatives/density_evaluation/option_implied.py

import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .base import ForecastDensity
from .windows import get_fit_dates


class OptionImpliedDensityEvaluator:
    def __init__(
        self,
        rnd_dict,
        logreturns,
        transform,
        target_maturity=30 / 365,
        maturity_match_tol=None,
        return_col=None,
        price_col=None,
        date_col="date",
        horizon_days=None,
        reserve_obs=30,
        window_type="expanding",
        window_size=None,
        model_name="Option-Implied Physical",
        clone_transform=True,
        fit_stock_df=None,
        fit_kwargs=None,
    ):
        self.rnd_dict = rnd_dict
        self.logreturns = logreturns.copy()
        self.fit_stock_df = (
            fit_stock_df.copy()
            if fit_stock_df is not None
            else logreturns.copy()
        )

        self.transform = transform
        self.target_maturity = float(target_maturity)
        self.maturity_match_tol = maturity_match_tol

        self.return_col = return_col
        self.price_col = price_col
        self.date_col = date_col

        self.horizon_days = (
            int(round(365 * self.target_maturity))
            if horizon_days is None
            else int(horizon_days)
        )

        self.reserve_obs = int(reserve_obs)
        self.window_type = window_type
        self.window_size = window_size
        self.model_name = model_name
        self.clone_transform = bool(clone_transform)
        self.fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)

        self.ret_series = self._standardize_logreturns(self.logreturns)

    def _standardize_logreturns(self, logreturns):
        if isinstance(logreturns, pd.Series):
            s = logreturns.copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s.sort_index().astype(float)

        df = logreturns.copy()

        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.tz_localize(None)
            df = df.sort_values(self.date_col).set_index(self.date_col)
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()

        if self.return_col is not None and self.return_col in df.columns:
            s = df[self.return_col].astype(float)

        elif self.price_col is not None and self.price_col in df.columns:
            price = df[self.price_col].astype(float)
            s = np.log(price).diff()

        elif df.shape[1] == 1:
            s = df.iloc[:, 0].astype(float)

        else:
            raise ValueError(
                "Could not infer log returns. Supply return_col='ret_1' "
                "or price_col='price'."
            )

        return s.dropna().sort_index()

    def _standardize_rnd_dict(self):
        return {
            pd.Timestamp(k).tz_localize(None): v
            for k, v in self.rnd_dict.items()
        }

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

        if T_grid.size == 0:
            return None, None

        j = int(np.argmin(np.abs(T_grid - self.target_maturity)))
        T_actual = float(T_grid[j])

        if self.maturity_match_tol is not None:
            if abs(T_actual - self.target_maturity) > self.maturity_match_tol:
                return None, None

        return j, T_actual

    def _fit_model(self, model, rnd_fit_dict):
        kwargs = dict(self.fit_kwargs)

        if self.return_col is not None:
            kwargs.setdefault("return_col", self.return_col)

        if self.price_col is not None:
            kwargs.setdefault("price_col", self.price_col)

        return model.fit(
            rnd_fit_dict,
            self.fit_stock_df,
            **kwargs,
        )

    def _get_physical_density(self, physical, j):
        x_grid = np.asarray(physical["grid_lr"], dtype=float)

        pdf = np.asarray(
            physical["physical_lr_surface"],
            dtype=float,
        )[j, :]

        if "physical_cdf_lr_surface" in physical:
            cdf = np.asarray(
                physical["physical_cdf_lr_surface"],
                dtype=float,
            )[j, :]
        else:
            dx = np.diff(x_grid)
            inc = 0.5 * (pdf[1:] + pdf[:-1]) * dx

            cdf = np.empty_like(x_grid)
            cdf[0] = 0.0
            cdf[1:] = np.cumsum(inc)

            if np.isfinite(cdf[-1]) and cdf[-1] > 0:
                cdf = cdf / cdf[-1]

        return x_grid, pdf, cdf

    def run(
        self,
        start_i=None,
        end_i=None,
        save_path=None,
        progress_every=10,
        verbose=True,
        eval_step=None,
    ):
        rnd_by_ts = self._standardize_rnd_dict()
        dates = sorted(rnd_by_ts.keys())

        if start_i is None:
            start_i = 0
        if end_i is None:
            end_i = len(dates)

        if eval_step is None:
            eval_step = 1

        eval_step = int(eval_step)
        if eval_step < 1:
            raise ValueError("eval_step must be >= 1.")

        date_items = list(enumerate(dates))[start_i:end_i]
        date_items = date_items[::eval_step]

        forecasts = []

        if self.window_type == "expanding":
            window_msg = "expanding | window_size=ignored"
        else:
            window_msg = f"rolling | window_size={self.window_size}"

        if verbose:
            print(
                f"[{self.model_name}] starting evaluation "
                f"| dates={len(date_items)} "
                f"| start_i={start_i} | end_i={end_i} "
                f"| eval_step={eval_step} "
                f"| window={window_msg} "
                f"| reserve_obs={self.reserve_obs} "
                f"| target_maturity={self.target_maturity:.6f}y "
                f"({self.horizon_days}d)",
                flush=True,
            )

        for count, (i, date) in enumerate(date_items, start=1):

            fit_dates = get_fit_dates(
                dates=dates,
                i=i,
                window_type=self.window_type,
                window_size=self.window_size,
                reserve_obs=self.reserve_obs,
            )

            fit_n = 0 if fit_dates is None else len(fit_dates)

            if verbose and (
                count == 1
                or count % progress_every == 0
                or count == len(date_items)
            ):
                print(
                    f"[{self.model_name}] {count}/{len(date_items)} "
                    f"| date={date.date()} | index={i} "
                    f"| fit_obs={fit_n} "
                    f"| stored={len(forecasts)}",
                    flush=True,
                )

            if fit_dates is None:
                continue

            realized, end_date = self._realized_horizon_return(date)
            if realized is None:
                continue

            rnd_fit_dict = {
                d: rnd_by_ts[d]
                for d in fit_dates
                if d in rnd_by_ts and rnd_by_ts[d].get("success", True)
            }

            if len(rnd_fit_dict) < self.reserve_obs:
                continue

            model = (
                copy.deepcopy(self.transform)
                if self.clone_transform
                else self.transform
            )

            model.verbose = False

            try:
                self._fit_model(model, rnd_fit_dict)
                physical = model.transform_info(rnd_by_ts[date])

            except Exception as e:
                if verbose:
                    print(
                        f"[{self.model_name}] skipped date={date.date()} "
                        f"because fit/transform failed: {e}",
                        flush=True,
                    )
                continue

            T_grid = np.asarray(physical["T_grid"], dtype=float)
            j, T_actual = self._select_maturity_index(T_grid)

            if j is None:
                continue

            x_grid, pdf, cdf = self._get_physical_density(physical, j)

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
                "window_type": self.window_type,
                "window_size": self.window_size,
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