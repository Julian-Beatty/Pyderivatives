import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from arch import arch_model

from .base import ForecastDensity
from .windows import get_fit_dates


class GARCHDensityEvaluator:
    def __init__(
        self,
        logreturns,
        target_maturity=30 / 365,
        return_col="ret_1",
        date_col="date",
        reserve_obs=252,
        window_type="rolling",
        window_size=252 * 3,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="t",
        simulations=10000,
        grid_size=500,
        grid_pad=0.25,
        scale_returns=100.0,
        model_name="GARCH",
    ):
        self.logreturns = logreturns.copy()
        self.target_maturity = float(target_maturity)
        self.horizon_days = int(round(365 * self.target_maturity))
        self.return_col = return_col
        self.date_col = date_col
        self.reserve_obs = int(reserve_obs)
        self.window_type = window_type
        self.window_size = window_size

        self.mean = mean
        self.vol = vol
        self.p = int(p)
        self.o = int(o)
        self.q = int(q)
        self.dist = dist
        self.simulations = int(simulations)

        self.grid_size = int(grid_size)
        self.grid_pad = float(grid_pad)
        self.scale_returns = float(scale_returns)
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

    def _fit_and_simulate(self, sample):
        sample = pd.Series(sample).dropna().astype(float)

        if len(sample) < self.reserve_obs:
            return None

        y = sample * self.scale_returns

        try:
            am = arch_model(
                y,
                mean=self.mean,
                vol=self.vol,
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                rescale=False,
            )

            res = am.fit(disp="off", show_warning=False)

            fcst = res.forecast(
                horizon=self.horizon_days,
                method="simulation",
                simulations=self.simulations,
                reindex=False,
            )

            sim = fcst.simulations.values

            # Usually shape: (1, simulations, horizon)
            sim = np.asarray(sim, dtype=float)

            if sim.ndim == 3:
                terminal = sim[0, :, :].sum(axis=1)
            elif sim.ndim == 2:
                terminal = sim.sum(axis=1)
            else:
                return None

            terminal = terminal / self.scale_returns
            terminal = terminal[np.isfinite(terminal)]

            if len(terminal) < 50:
                return None

            return terminal

        except Exception:
            return None

    def _make_density(self, simulated_horizon_returns):
        x = np.asarray(simulated_horizon_returns, dtype=float)
        x = x[np.isfinite(x)]

        if len(x) < 50:
            return None

        kde = gaussian_kde(x)

        lo = float(np.min(x))
        hi = float(np.max(x))
        width = hi - lo

        if not np.isfinite(width) or width <= 0:
            return None

        x_grid = np.linspace(
            lo - self.grid_pad * width,
            hi + self.grid_pad * width,
            self.grid_size,
        )

        pdf = kde(x_grid)
        pdf = np.where(np.isfinite(pdf) & (pdf >= 0), pdf, 0.0)

        area = np.trapezoid(pdf, x_grid)
        if not np.isfinite(area) or area <= 0:
            return None

        pdf = pdf / area

        dx = np.diff(x_grid)
        inc = 0.5 * (pdf[1:] + pdf[:-1]) * dx

        cdf = np.empty_like(x_grid)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(inc)
        cdf = cdf / cdf[-1]

        return x_grid, pdf, cdf

    def run(
        self,
        start_i=None,
        end_i=None,
        save_path=None,
        eval_dates=None,
        progress_every=25,
        verbose=True,
        eval_step=None
    ):
        if self.window_type == "expanding":
            window_msg = "expanding | window_size=ignored"
        else:
            window_msg = f"rolling | window_size={self.window_size}"
    
        if eval_dates is None:
            dates = list(self.ret_series.index)
        else:
            dates = [
                pd.Timestamp(d).tz_localize(None)
                for d in pd.to_datetime(eval_dates)
            ]
            dates = sorted(dates)
    
        if start_i is None:
            start_i = 0
        if end_i is None:
            end_i = len(dates)
    
        date_items = list(enumerate(dates))[start_i:end_i]
        forecasts = []
        if eval_step is None:
            eval_step = 1
    
        date_items = date_items[::int(eval_step)]
        if verbose:
            print(
                f"[{self.model_name}] starting evaluation "
                f"| dates={len(date_items)} "
                f"| start_i={start_i} | end_i={end_i} "
                f"| window={window_msg} "
                f"| reserve_obs={self.reserve_obs} "
                f"| target_maturity={self.target_maturity:.6f}y "
                f"({self.horizon_days}d)",
                flush=True,
            )
    
        for count, (i, date) in enumerate(date_items, start=1):
    
            # Use all daily returns strictly before the forecast date.
            sample = self.ret_series[self.ret_series.index < date].dropna()
    
            if self.window_type == "rolling":
                if self.window_size is None:
                    raise ValueError("window_size must be provided for rolling windows.")
                sample = sample.iloc[-self.window_size:]
    
            elif self.window_type != "expanding":
                raise ValueError("window_type must be 'expanding' or 'rolling'.")
    
            fit_n = len(sample)
    
            if verbose and (
                count == 1
                or count % progress_every == 0
                or count == len(date_items)
            ):
                print(
                    f"[{self.model_name}] {count}/{len(date_items)} "
                    f"| date={pd.Timestamp(date).date()} "
                    f"| fit_obs={fit_n} "
                    f"| stored={len(forecasts)}",
                    flush=True,
                )
    
            if fit_n < self.reserve_obs:
                continue
    
            simulated = self._fit_and_simulate(sample.values)
            if simulated is None:
                continue
    
            density = self._make_density(simulated)
            if density is None:
                continue
    
            realized, end_date = self._realized_horizon_return(date)
            if realized is None:
                continue
    
            x_grid, pdf, cdf = density
    
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
                "window_type": self.window_type,
                "window_size": self.window_size,
                "target_maturity": self.target_maturity,
                "horizon_days": self.horizon_days,
                "model_name": self.model_name,
                "garch_spec": {
                    "mean": self.mean,
                    "vol": self.vol,
                    "p": self.p,
                    "o": self.o,
                    "q": self.q,
                    "dist": self.dist,
                    "simulations": self.simulations,
                },
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