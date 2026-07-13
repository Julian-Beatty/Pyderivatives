# pyderivatives/dealer_positioning.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

def get_nearest_stock_price(underlying_path, snapshot_et):
    u = pd.read_parquet(underlying_path).copy()
    u = u.rename(columns={"Time": "stock_time", "Price": "stock_price"})

    # underlying Time is Unix seconds
    u["stock_time_ns"] = u["stock_time"].astype("int64") * 1_000_000_000

    snapshot_ns = snapshot_et.tz_convert("UTC").value
    idx = (u["stock_time_ns"] - snapshot_ns).abs().idxmin()

    return float(u.loc[idx, "stock_price"])


def option_surface_snapshot_pyderivatives_style(
    quotes_path,
    underlying_path,
    risk_free_rate=0.0,
    minutes_after_open=14,
    market_open="09:30:00",
    market_close="16:00:00",
    timezone="America/New_York",
    verbose=True,
):
    q_pf = pq.ParquetFile(quotes_path)

    keep_cols = [
        "ticker",
        "sip_timestamp",
        "bid_price",
        "ask_price",
        "quote_date",
        "expiration",
        "option_root",
        "option_type",
        "strike",
    ]

    optional_cols = ["bid_size", "ask_size"]
    available_cols = set(q_pf.schema.names)
    read_cols = [c for c in keep_cols + optional_cols if c in available_cols]

    # Get trade/quote date from first row group only
    first = q_pf.read_row_group(
        0,
        columns=["sip_timestamp"],
    ).to_pandas()

    first_ts = int(first["sip_timestamp"].iloc[0])
    first_dt_et = pd.to_datetime(first_ts, unit="ns", utc=True).tz_convert(timezone)
    trade_date = first_dt_et.date()

    snapshot_et = (
        pd.Timestamp(f"{trade_date} {market_open}", tz=timezone)
        + pd.Timedelta(minutes=minutes_after_open)
    )

    snapshot_ns = snapshot_et.tz_convert("UTC").value

    latest_chunks = []

    for i in range(q_pf.num_row_groups):
        q = q_pf.read_row_group(i, columns=read_cols).to_pandas()
        q["sip_timestamp"] = q["sip_timestamp"].astype("int64")

        # Only quotes before snapshot
        q = q[q["sip_timestamp"] <= snapshot_ns]

        if q.empty:
            if verbose:
                print(f"quote row group {i+1}/{q_pf.num_row_groups}, kept 0")
            continue

        # Keep latest quote per ticker inside this row group
        q = (
            q.sort_values(["ticker", "sip_timestamp"])
             .groupby("ticker", as_index=False)
             .tail(1)
        )

        latest_chunks.append(q)

        if verbose:
            print(f"quote row group {i+1}/{q_pf.num_row_groups}, kept {len(q):,}")

    if not latest_chunks:
        raise ValueError("No quotes found before the snapshot time.")

    # Now combine only row-group-level latest quotes, much smaller
    q_latest = pd.concat(latest_chunks, ignore_index=True)

    q_snap = (
        q_latest
        .sort_values(["ticker", "sip_timestamp"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .copy()
    )

    if q_snap.empty:
        raise ValueError("No quotes found before the snapshot time.")

    # Force date and exdate to midnight
    q_snap["date"] = pd.to_datetime(q_snap["quote_date"]).dt.floor("D")
    q_snap["exdate"] = pd.to_datetime(q_snap["expiration"]).dt.floor("D")

    # Stock price from nearest 1-second underlying observation
    stock_price = get_nearest_stock_price(underlying_path, snapshot_et)
    q_snap["stock_price"] = stock_price

    # Minutes until expiration
    expiration_close_et = (
        q_snap["exdate"]
        .dt.tz_localize(timezone)
        + pd.to_timedelta(market_close)
    )

    q_snap["minutes_until_expiration"] = (
        expiration_close_et - snapshot_et
    ).dt.total_seconds() / 60

    q_snap["minutes_until_expiration"] = q_snap["minutes_until_expiration"].clip(lower=0)

    # 0DTE definition
    is_0dte = q_snap["date"].dt.date == q_snap["exdate"].dt.date
    calendar_days_to_exp = (q_snap["exdate"] - q_snap["date"]).dt.days.round()

    q_snap["rounded_maturity"] = np.where(
        is_0dte,
        q_snap["minutes_until_expiration"] / (365 * 24 * 60),
        calendar_days_to_exp / 365,
    )

    q_snap["moneyness"] = q_snap["strike"].astype(float) / q_snap["stock_price"]
    q_snap["option_right"] = q_snap["option_type"].astype(str).str.lower()

    q_snap["best_bid"] = q_snap["bid_price"]
    q_snap["best_offer"] = q_snap["ask_price"]
    q_snap["mid_price"] = (q_snap["best_bid"] + q_snap["best_offer"]) / 2

    q_snap["risk_free_rate"] = risk_free_rate

    if "bid_size" in q_snap.columns and "ask_size" in q_snap.columns:
        q_snap["volume"] = q_snap["bid_size"].fillna(0) + q_snap["ask_size"].fillna(0)
    else:
        q_snap["volume"] = np.nan

    q_snap["ticker"] = q_snap["option_root"]

    df = q_snap[
        [
            "date",
            "exdate",
            "rounded_maturity",
            "minutes_until_expiration",
            "moneyness",
            "option_right",
            "strike",
            "best_bid",
            "mid_price",
            "best_offer",
            "stock_price",
            "risk_free_rate",
            "volume",
            "ticker",
        ]
    ].sort_values(["exdate", "option_right", "strike"])

    return df.reset_index(drop=True)

def aggregate_dealer_inventory_until_time(
    quotes_path,
    trades_path,
    minutes_after_open=14,
    market_open="09:30:00",
    timezone="America/New_York",
):
    q_pf = pq.ParquetFile(quotes_path)
    t_pf = pq.ParquetFile(trades_path)

    # -----------------------------
    # 1. Load all trades
    # -----------------------------
    t_chunks = []

    for i in range(t_pf.num_row_groups):
        t = t_pf.read_row_group(i).to_pandas()
        t["sip_timestamp"] = t["sip_timestamp"].astype("int64")
        t_chunks.append(t)

    t_df = pd.concat(t_chunks, ignore_index=True)

    t_df["trade_dt_utc"] = pd.to_datetime(t_df["sip_timestamp"], unit="ns", utc=True)
    t_df["trade_dt_et"] = t_df["trade_dt_utc"].dt.tz_convert(timezone)

    trade_date = t_df["trade_dt_et"].dt.date.min()

    cutoff_et = (
        pd.Timestamp(f"{trade_date} {market_open}", tz=timezone)
        + pd.Timedelta(minutes=minutes_after_open)
    )

    cutoff_ns = cutoff_et.tz_convert("UTC").value

    # cumulative inventory up to cutoff
    t_df = t_df[t_df["sip_timestamp"] <= cutoff_ns].copy()

    if t_df.empty:
        raise ValueError("No trades found before cutoff time.")

    trade_tickers = set(t_df["ticker"].unique())

    # -----------------------------
    # 2. Load quotes only for traded tickers
    # -----------------------------
    q_cols = [
        "ticker",
        "sip_timestamp",
        "bid_price",
        "ask_price",
        "quote_date",
        "expiration",
        "dte",
        "option_root",
        "option_type",
        "strike",
    ]

    available_q_cols = set(q_pf.schema.names)
    q_cols = [c for c in q_cols if c in available_q_cols]

    quote_chunks = []

    for i in range(q_pf.num_row_groups):
        q = q_pf.read_row_group(i, columns=q_cols).to_pandas()
        q["sip_timestamp"] = q["sip_timestamp"].astype("int64")

        q = q[
            q["ticker"].isin(trade_tickers)
            & (q["sip_timestamp"] <= cutoff_ns)
        ]

        if not q.empty:
            quote_chunks.append(q)

        print(f"quote row group {i+1}/{q_pf.num_row_groups}, kept {len(q):,}")

    if not quote_chunks:
        raise ValueError("No matching quotes found before cutoff time.")

    q_df = pd.concat(quote_chunks, ignore_index=True)

    # -----------------------------
    # 3. Match each trade to latest prior quote
    # -----------------------------
    q_small = q_df.rename(
        columns={
            "sip_timestamp": "quote_timestamp",
            "bid_price": "best_bid_at_trade",
            "ask_price": "best_ask_at_trade",
        }
    )

    q_small = q_small.sort_values("quote_timestamp").reset_index(drop=True)
    t_df = t_df.sort_values("sip_timestamp").reset_index(drop=True)

    matched = pd.merge_asof(
        t_df,
        q_small,
        left_on="sip_timestamp",
        right_on="quote_timestamp",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )

    mid = (matched["best_bid_at_trade"] + matched["best_ask_at_trade"]) / 2

    matched["trade_initiator"] = np.select(
        [
            matched["price"] >= matched["best_ask_at_trade"],
            matched["price"] <= matched["best_bid_at_trade"],
            matched["price"] > mid,
            matched["price"] < mid,
        ],
        [
            "buyer_initiated_dealer_sold",
            "seller_initiated_dealer_bought",
            "buyer_initiated_inside_spread",
            "seller_initiated_inside_spread",
        ],
        default="unknown",
    )

    # -----------------------------
    # 4. Signed dealer inventory
    # -----------------------------
    size_col = "size" if "size" in matched.columns else "trade_size"

    matched["dealer_signed_contracts"] = np.select(
        [
            matched["trade_initiator"].isin([
                "buyer_initiated_dealer_sold",
                "buyer_initiated_inside_spread",
            ]),
            matched["trade_initiator"].isin([
                "seller_initiated_dealer_bought",
                "seller_initiated_inside_spread",
            ]),
        ],
        [
            -matched[size_col],
             matched[size_col],
        ],
        default=0,
    )

        # -----------------------------
    # 5. PyDerivatives-style columns
    # -----------------------------
    def pick_col(df, names):
        for name in names:
            if name in df.columns:
                return name
        raise KeyError(f"None of these columns found: {names}\nAvailable columns:\n{df.columns.tolist()}")

    quote_date_col = pick_col(matched, ["quote_date", "quote_date_y", "quote_date_x"])
    expiration_col = pick_col(matched, ["expiration", "expiration_y", "expiration_x"])
    option_type_col = pick_col(matched, ["option_type", "option_type_y", "option_type_x"])
    strike_col = pick_col(matched, ["strike", "strike_y", "strike_x"])
    option_root_col = pick_col(matched, ["option_root", "option_root_y", "option_root_x"])

    matched["date"] = pd.to_datetime(matched[quote_date_col]).dt.floor("D")
    matched["exdate"] = pd.to_datetime(matched[expiration_col]).dt.floor("D")

    matched["option_right"] = (
        matched[option_type_col]
        .astype(str)
        .str.lower()
        .str[0]
    )

    matched["strike_clean"] = matched[strike_col].astype(float)
    matched["ticker_root"] = matched[option_root_col].astype(str)

    # -----------------------------
    # 6. Aggregate inventory
    # -----------------------------
    inventory = (
        matched
        .groupby(
            ["date", "exdate", "ticker_root", "option_right", "strike_clean"],
            as_index=False
        )
        .agg(
            dealer_net_contracts=("dealer_signed_contracts", "sum"),
            total_contract_volume=(size_col, "sum"),
            n_trades=(size_col, "count"),
            first_trade_time=("trade_dt_et", "min"),
            last_trade_time=("trade_dt_et", "max"),
        )
    )

    inventory = inventory.rename(
        columns={
            "ticker_root": "ticker",
            "strike_clean": "strike",
        }
    )

    inventory["snapshot_time"] = cutoff_et
    inventory["minutes_after_open"] = minutes_after_open

    inventory = inventory[
        [
            "date",
            "exdate",
            "ticker",
            "option_right",
            "strike",
            "dealer_net_contracts",
            "total_contract_volume",
            "n_trades",
            "first_trade_time",
            "last_trade_time",
            "snapshot_time",
            "minutes_after_open",
        ]
    ].sort_values(["exdate", "option_right", "strike"])

    return inventory.reset_index(drop=True), matched
def plot_gamma_exposure_curve(
    gamma_exposure_df,
    *,
    exposure_col="dealer_gamma_exposure",
    gamma_col="gamma",
    x_bounds=None,          # moneyness/gross-return bounds, e.g. (0.98, 1.02)
    y_bounds=None,
    gamma_bounds=None,
    overlay_gamma=True,
    aggregate=True,
    title="Dealer Gamma Exposure by Strike",
    figsize=(10, 5),
    spot=None,
    save=None,
    dpi=200,
    show=True,
):
    df = gamma_exposure_df.copy()

    if spot is None and "S0" in df.columns:
        spot = float(df["S0"].dropna().iloc[0])

    if spot is None:
        raise ValueError("spot must be supplied or gamma_exposure_df must contain S0.")

    # User supplies x_bounds in moneyness/gross-return units.
    # Plot remains in strike units.
    strike_bounds = None
    if x_bounds is not None:
        lo_m, hi_m = x_bounds
        strike_bounds = (float(lo_m) * spot, float(hi_m) * spot)

    if aggregate:
        plot_df = (
            df.dropna(subset=["strike", exposure_col, gamma_col])
              .groupby("strike", as_index=False)
              .agg(
                  exposure=(exposure_col, "sum"),
                  gamma=(gamma_col, "mean"),
              )
              .sort_values("strike")
        )
    else:
        plot_df = (
            df[["strike", exposure_col, gamma_col]]
            .dropna()
            .rename(
                columns={
                    exposure_col: "exposure",
                    gamma_col: "gamma",
                }
            )
            .sort_values("strike")
        )

    if strike_bounds is not None:
        lo_k, hi_k = strike_bounds
        plot_df = plot_df[
            (plot_df["strike"] >= lo_k) &
            (plot_df["strike"] <= hi_k)
        ]

    if plot_df.empty:
        raise ValueError("No rows left after applying x_bounds.")

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(
        plot_df["strike"],
        plot_df["exposure"],
        marker="o",
        linestyle="-",
        label="Dealer Gamma Exposure",
    )

    ax1.axvline(
        spot,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Spot = {spot:.2f}",
    )

    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Dealer Gamma Exposure")
    ax1.grid(True)

    if y_bounds is not None:
        ax1.set_ylim(*y_bounds)

    if overlay_gamma:
        ax2 = ax1.twinx()

        ax2.plot(
            plot_df["strike"],
            plot_df["gamma"],
            color="red",
            linewidth=2,
            label="Black-Scholes Gamma",
        )

        ax2.set_ylabel("Black-Scholes Gamma")

        if gamma_bounds is not None:
            ax2.set_ylim(*gamma_bounds)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    else:
        ax1.legend(loc="best")

    plt.title(title)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return plot_df
def compute_gamma_exposure_from_inventory_single_res(
    inventory_df,
    res,
    *,
    inventory_col="dealer_net_contracts",
    multiplier=100,
    scale_by_underlying=True,
    combine_put_call=True,
):
    inv = inventory_df.copy()

    inv["date"] = pd.to_datetime(inv["date"]).dt.floor("D")
    inv["exdate"] = pd.to_datetime(inv["exdate"]).dt.floor("D")
    inv["strike"] = inv["strike"].astype(float)

    if combine_put_call:
        group_cols = ["date", "exdate", "ticker", "strike"]
    else:
        group_cols = ["date", "exdate", "ticker", "option_right", "strike"]

    inv_agg = (
        inv.groupby(group_cols, as_index=False)
        .agg(
            inventory=(inventory_col, "sum"),
            total_contract_volume=("total_contract_volume", "sum"),
            n_trades=("n_trades", "sum"),
        )
    )

    K_grid = np.asarray(res["grid_k"], float).ravel()
    T_grid = np.asarray(res["T_grid"], float).ravel()
    gamma_surface = np.asarray(res["gamma_surface"], float)

    if gamma_surface.shape != (T_grid.size, K_grid.size):
        raise ValueError(
            f"gamma_surface shape {gamma_surface.shape} does not match "
            f"(len(T_grid), len(K_grid)) = {(T_grid.size, K_grid.size)}"
        )

    S0 = float(res["S0"])

    res_date = pd.to_datetime(res.get("date", inv_agg["date"].iloc[0])).floor("D")
    inv_agg = inv_agg[inv_agg["date"] == res_date].copy()

    if inv_agg.empty:
        raise ValueError("No inventory rows match the result date.")

    inv_agg["rounded_maturity"] = (
        inv_agg["exdate"] - inv_agg["date"]
    ).dt.days.astype(float) / 365.0

    same_day = inv_agg["exdate"].dt.date == inv_agg["date"].dt.date

    if np.any(same_day) and T_grid.size == 1:
        inv_agg.loc[same_day, "rounded_maturity"] = float(T_grid[0])

    gamma_vals = []
    matched_T_vals = []

    for _, row in inv_agg.iterrows():
        K = float(row["strike"])
        T = float(row["rounded_maturity"])

        t_idx = int(np.argmin(np.abs(T_grid - T)))
        matched_T = float(T_grid[t_idx])

        gamma_row = gamma_surface[t_idx, :]

        valid = np.isfinite(K_grid) & np.isfinite(gamma_row)

        if np.sum(valid) < 2:
            gamma_vals.append(np.nan)
            matched_T_vals.append(matched_T)
            continue

        if K < np.nanmin(K_grid[valid]) or K > np.nanmax(K_grid[valid]):
            gamma_vals.append(np.nan)
            matched_T_vals.append(matched_T)
            continue

        gamma = float(np.interp(K, K_grid[valid], gamma_row[valid]))

        gamma_vals.append(gamma)
        matched_T_vals.append(matched_T)

    inv_agg["gamma"] = gamma_vals
    inv_agg["matched_T"] = matched_T_vals

    inv_agg["gamma_exposure"] = inv_agg["inventory"] * inv_agg["gamma"]

    if scale_by_underlying:
        inv_agg["dealer_gamma_exposure"] = (
            inv_agg["inventory"]
            * inv_agg["gamma"]
            * (multiplier ** 2)
            * S0
        )
    else:
        inv_agg["dealer_gamma_exposure"] = (
            inv_agg["inventory"]
            * inv_agg["gamma"]
            * (multiplier ** 2)
        )

    inv_agg["S0"] = S0
    inv_agg = inv_agg[inv_agg["gamma"].notna()].reset_index(drop=True)

    summary = {
        "total_inventory": float(inv_agg["inventory"].sum(skipna=True)),
        "total_gamma_exposure": float(inv_agg["gamma_exposure"].sum(skipna=True)),
        "total_dealer_gamma_exposure": float(
            inv_agg["dealer_gamma_exposure"].sum(skipna=True)
        ),
    }

    return inv_agg.reset_index(drop=True), summary