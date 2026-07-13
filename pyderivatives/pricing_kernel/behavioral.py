from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from .utils import (
    _as_1d,
    _cdf_from_density,
    _trapz_normalize_density,
    _find_sigma,
)


class BehavioralOverlayMixin:
    """
    Mixin for optional Crisostomo-style behavioral adjustments.

    This is separated from MeasureTransform so the base class can focus on:
        - fitting
        - maturity matching
        - transforming RNDs into physical densities

    The behavioral overlay adjusts the already-transformed physical density
    using three sentiment channels:

        theta1:
            Mean shift from IV-change sentiment.

        theta2:
            Dispersion adjustment from volume/confidence sentiment.

        theta3:
            Tail adjustment from risk-neutral skewness.
    """

    def _kde_cdf_value(self, x_now: float, history: np.ndarray) -> float:
        """
        Estimate F(x_now) from historical observations using Gaussian KDE.

        Used to convert IV changes and volume ratios into empirical quantiles.
        Falls back to the empirical CDF if KDE is unstable.
        """
        history = np.asarray(history, dtype=float)
        history = history[np.isfinite(history)]

        if history.size < 5 or not np.isfinite(x_now):
            return np.nan

        sd = float(np.std(history, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return float(np.mean(history <= x_now))

        lo = min(float(np.min(history)), float(x_now)) - 4.0 * sd
        hi = max(float(np.max(history)), float(x_now)) + 4.0 * sd
        grid = np.linspace(lo, hi, 2000)

        try:
            kde = gaussian_kde(history)
            pdf = kde(grid)
        except Exception:
            return float(np.mean(history <= x_now))

        pdf = np.where(np.isfinite(pdf) & (pdf >= 0), pdf, 0.0)

        cdf = np.empty_like(grid)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(
            0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid)
        )

        total = cdf[-1]
        if not np.isfinite(total) or total <= self.eps:
            return float(np.mean(history <= x_now))

        cdf = cdf / total
        return float(np.clip(np.interp(x_now, grid, cdf), 0.0, 1.0))

    def _date_from_info(self, info: dict) -> Optional[pd.Timestamp]:
        """
        Extract the valuation date from a result dictionary.

        Checks top-level keys first, then result['meta'].
        """
        date_keys = ("date", "day", "anchor_date", "valuation_date")

        for key in date_keys:
            if key in info and info[key] is not None:
                try:
                    return pd.Timestamp(info[key]).tz_localize(None)
                except Exception:
                    return None

        meta = info.get("meta", {})
        if not isinstance(meta, dict):
            return None

        for key in date_keys:
            if key in meta and meta[key] is not None:
                try:
                    return pd.Timestamp(meta[key]).tz_localize(None)
                except Exception:
                    return None

        return None

    def _extract_rn_skew(self, info: dict, T: float) -> float:
        """
        Extract risk-neutral skewness from an already-computed moments table.

        This avoids recomputing BKM/moment quantities inside the behavioral
        overlay.
        """
        candidates = [
            info.get("rnd_moments_table"),
            info.get("risk_neutral_moments"),
            info.get("moments"),
        ]

        for tbl in candidates:
            if not isinstance(tbl, pd.DataFrame):
                continue

            tmp = tbl.copy()

            if "skew" in tmp.columns:
                skew_col = "skew"
            elif "skew_r" in tmp.columns:
                skew_col = "skew_r"
            else:
                continue

            if "T" in tmp.columns:
                idx = (tmp["T"].astype(float) - float(T)).abs().idxmin()
            else:
                idx = tmp.index[0]

            try:
                val = float(tmp.loc[idx, skew_col])
                if np.isfinite(val):
                    return val
            except Exception:
                pass

        return np.nan

    def _standardize_stock_df_for_sentiment(self) -> Optional[pd.DataFrame]:
        """
        Standardize stock data to a DatetimeIndex for sentiment calculations.
        """
        if self.stock_df is None:
            return None

        df = self.stock_df.copy()

        if self.stock_date_col in df.columns:
            df["_date"] = pd.to_datetime(df[self.stock_date_col]).dt.tz_localize(None)
            df = df.set_index("_date")
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)

        return df.sort_index()

    def _build_behavioral_sentiment_by_date(
        self,
        rnd_history_dict: Dict[Any, dict],
    ) -> Dict[pd.Timestamp, dict]:
        """
        Build date-level sentiment inputs.

        theta1:
            IV-change sentiment. Extreme low IV changes imply optimism;
            extreme high IV changes imply pessimism.

        theta2:
            Volume/confidence sentiment. Extreme volume ratios modify
            dispersion.

        theta3:
            Tail sentiment. Computed later, maturity by maturity, from
            risk-neutral skewness.
        """
        keys = sorted(rnd_history_dict.keys(), key=lambda x: pd.Timestamp(x))
        stock = self._standardize_stock_df_for_sentiment()

        records = []

        for raw_date in keys:
            date = pd.Timestamp(raw_date).tz_localize(None)
            info = rnd_history_dict[raw_date]

            sigma = _find_sigma(info, self.key_spec.sigma_keys, default=np.nan)

            volume_ratio = np.nan
            if stock is not None and self.volume_col in stock.columns:
                v = pd.to_numeric(stock[self.volume_col], errors="coerce").sort_index()

                # Current one-month proxy: last 20 trading days up to date.
                cur = v.loc[v.index <= date].tail(20).sum()

                # Prior three-month proxy: previous 60 trading days before current block.
                hist = v.loc[v.index < date].tail(80)
                prev = hist.iloc[:-20] if len(hist) > 20 else pd.Series(dtype=float)

                prev_20d_equiv = prev.mean() * 20.0 if len(prev) > 0 else np.nan

                if (
                    np.isfinite(cur)
                    and np.isfinite(prev_20d_equiv)
                    and prev_20d_equiv > 0
                ):
                    volume_ratio = float(cur / prev_20d_equiv)

            records.append(
                {
                    "date": date,
                    "sigma": sigma,
                    "volume_ratio": volume_ratio,
                }
            )

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

        if df.empty:
            return {}

        # IV-change proxy: current IV minus average IV over previous 3 observations.
        df["iv_lag_avg"] = df["sigma"].rolling(3, min_periods=1).mean().shift(1)
        df["iv_change"] = df["sigma"] - df["iv_lag_avg"]

        out: Dict[pd.Timestamp, dict] = {}
        a_iv = float(getattr(self, "iv_sentiment_alpha", getattr(self, "sentiment_alpha", 0.05)))
        a_vol = float(getattr(self, "volume_sentiment_alpha", getattr(self, "sentiment_alpha", 0.05)))

        for i, row in df.iterrows():
            date = pd.Timestamp(row["date"])

            past_iv = df.loc[: i - 1, "iv_change"].dropna().to_numpy(dtype=float)
            past_vol = df.loc[: i - 1, "volume_ratio"].dropna().to_numpy(dtype=float)

            iv_q = np.nan
            vol_q = np.nan

            theta1 = 0.0
            theta2 = 1.0

            iv_now = float(row["iv_change"]) if np.isfinite(row["iv_change"]) else np.nan

            if past_iv.size >= 20 and np.isfinite(iv_now):
                iv_q = self._kde_cdf_value(iv_now, past_iv)

                info_t = (
                    rnd_history_dict.get(date)
                    or rnd_history_dict.get(str(date.date()))
                    or rnd_history_dict.get(pd.Timestamp(date))
                    or rnd_history_dict.get(row["date"], {})
                    or {}
                )

                try:
                    rate = float(info_t.get("r", 0.0))
                except Exception:
                    rate = 0.0

                # One-month risk-free return scale.
                r_month = np.exp(rate / 12.0) - 1.0 if np.isfinite(rate) else 0.0
                
                if iv_q < a_iv:
                    theta1 = -float(self.k1) * r_month * ((a_iv - iv_q) / a_iv)
                elif iv_q > 1.0 - a_iv:
                    theta1 = float(self.k1) * r_month * ((iv_q - (1.0 - a_iv)) / a_iv)



            vol_now = (
                float(row["volume_ratio"])
                if np.isfinite(row["volume_ratio"])
                else np.nan
            )

            if past_vol.size >= 20 and np.isfinite(vol_now):
                vol_q = self._kde_cdf_value(vol_now, past_vol)

                if vol_q < a_vol:
                    theta2 = float(self.k2 ** ((vol_q - a_vol) / a_vol))
                elif vol_q > 1.0 - a_vol:
                    theta2 = float(self.k2 ** ((vol_q - (1.0 - a_vol)) / a_vol))
                else:
                    theta2 = 1.0

            out[date] = {
                "iv_quantile": iv_q,
                "volume_quantile": vol_q,
                "theta1": theta1,
                "theta2": theta2,
                "iv_change": iv_now,
                "volume_ratio": vol_now,
            }

        return out

    def plot_empirical_vs_kde(
        self,
        data,
        *,
        x_now=None,
        title="Empirical distribution vs Gaussian KDE",
        xlabel="Value",
        bins=30,
        grid_n=1000,
    ):
        """
        Diagnostic plot comparing the empirical histogram to a Gaussian KDE.
        """
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]

        if data.size < 5:
            raise ValueError("Need at least 5 finite observations.")

        sd = np.std(data, ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError("Data has zero or invalid standard deviation.")

        lo = min(np.min(data), x_now) if x_now is not None and np.isfinite(x_now) else np.min(data)
        hi = max(np.max(data), x_now) if x_now is not None and np.isfinite(x_now) else np.max(data)

        grid = np.linspace(lo - 3 * sd, hi + 3 * sd, grid_n)

        kde = gaussian_kde(data)
        kde_pdf = kde(grid)

        plt.figure(figsize=(9, 4.5))
        plt.hist(data, bins=bins, density=True, alpha=0.35, label="Empirical histogram")
        plt.plot(grid, kde_pdf, linewidth=2.2, label="Gaussian KDE")

        q05 = np.quantile(data, 0.05)
        q95 = np.quantile(data, 0.95)

        plt.axvline(q05, linestyle="--", linewidth=1.4, label="Empirical 5% / 95%")
        plt.axvline(q95, linestyle="--", linewidth=1.4)

        if x_now is not None and np.isfinite(x_now):
            plt.axvline(x_now, linestyle="-", linewidth=2.0, label="Current value")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _tail_theta_from_skew(self, skew: float) -> float:
        if not np.isfinite(skew):
            return 0.0

        pos_threshold = float(getattr(self, "positive_skew_threshold", getattr(self, "skew_threshold", 1.5)))
        neg_threshold = float(getattr(self, "negative_skew_threshold", -abs(getattr(self, "skew_threshold", 1.5))))

        if skew < neg_threshold:
            return float(self.k3 * (neg_threshold - skew))

        if skew > pos_threshold:
            return float(-self.k3 * (skew - pos_threshold))

        return 0.0

    def _apply_behavioral_overlay_if_needed(
        self,
        *,
        x_grid,
        f_q,
        f_p,
        info: dict,
        T: float,
    ):
        """
        Apply the behavioral overlay to the baseline physical density.

        If behavioral=False, this simply returns the baseline density and
        internally consistent kernel/measure weights.
        """
        x_grid = _as_1d(x_grid)
        f_q = _trapz_normalize_density(x_grid, f_q, eps=self.eps)
        f_p = _trapz_normalize_density(x_grid, f_p, eps=self.eps)

        if not self.behavioral:
            F_p = _cdf_from_density(x_grid, f_p, eps=self.eps)

            kernel = f_q / np.maximum(f_p, self.eps)
            Em = float(np.trapezoid(kernel * f_p, x_grid))
            if np.isfinite(Em) and Em > self.eps:
                kernel = kernel / Em

            weight = f_p / np.maximum(f_q, self.eps)

            return f_p, F_p, kernel, weight, {"enabled": False}

        date = self._date_from_info(info)
        s = self.sentiment_by_date_.get(date, {}) if date is not None else {}

        theta1 = float(s.get("theta1", 0.0))
        theta2 = float(s.get("theta2", 1.0))

        if not np.isfinite(theta1):
            theta1 = 0.0

        if not np.isfinite(theta2) or theta2 <= self.eps:
            theta2 = 1.0

        theta2 = float(np.clip(
            theta2,
            float(getattr(self, "theta_min", 0.25)),
            float(getattr(self, "theta_max", 4.0)),
        ))

        skew = self._extract_rn_skew(info, T=float(T))
        theta3 = self._tail_theta_from_skew(skew)

        # Mean/variance correction.
        mu = float(np.trapezoid(x_grid * f_p, x_grid))

        x_preimage = (x_grid - mu - theta1) / theta2 + mu

        f_mv = np.interp(
            x_preimage,
            x_grid,
            f_p,
            left=0.0,
            right=0.0,
        ) / theta2

        f_mv = _trapz_normalize_density(x_grid, f_mv, eps=self.eps)

        # Tail sentiment correction.
        F_mv = _cdf_from_density(x_grid, f_mv, eps=self.eps)
        a = float(getattr(self, "tail_sentiment_alpha", getattr(self, "sentiment_alpha", 0.05)))
        if np.all(np.isfinite(F_mv)):
            q_left = float(np.interp(a, F_mv, x_grid))
            q_right = float(np.interp(1.0 - a, F_mv, x_grid))
        else:
            q_left = np.nan
            q_right = np.nan

        m_ts = np.ones_like(x_grid, dtype=float)

        if (
            np.isfinite(theta3)
            and abs(theta3) > 0
            and np.isfinite(q_left)
            and np.isfinite(q_right)
        ):
            left = x_grid < q_left
            right = x_grid > q_right

            m_ts[left] = np.exp(
                np.clip(theta3 * (q_left - x_grid[left]), -700, 700)
            )

            m_ts[right] = np.exp(
                np.clip(-theta3 * (x_grid[right] - q_right), -700, 700)
            )

        f_final = f_mv / np.maximum(m_ts, self.eps)
        f_final = _trapz_normalize_density(x_grid, f_final, eps=self.eps)
        F_final = _cdf_from_density(x_grid, f_final, eps=self.eps)

        kernel = f_q / np.maximum(f_final, self.eps)
        Em = float(np.trapezoid(kernel * f_final, x_grid))
        if np.isfinite(Em) and Em > self.eps:
            kernel = kernel / Em

        weight = f_final / np.maximum(f_q, self.eps)

        return f_final, F_final, kernel, weight, {
            "enabled": True,
            "date": None if date is None else str(date.date()),
            "theta1": theta1,
            "theta2": theta2,
            "theta3": theta3,
            "rnd_skew_used": skew,
            "iv_quantile": s.get("iv_quantile", np.nan),
            "volume_quantile": s.get("volume_quantile", np.nan),
            "q_left": q_left,
            "q_right": q_right,
        }