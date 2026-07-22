import pickle
from pathlib import Path
import pandas as pd
import pickle
from pathlib import Path
import pandas as pd
from pyderivatives.global_pricer.data import CallSurfaceDay
from pyderivatives import *
def merge_calibration_pickles(
    folder,
    pattern="*.pkl",
    output_file=None,
    on_conflict="overwrite",
    check_stock_df=True,
):
    folder = Path(folder)
    files = sorted(folder.glob(pattern))

    merged_result_dict = {}
    stock_df_master = None

    for f in files:
        print(f"\nLoading: {f.name}")

        with f.open("rb") as fh:
            obj = pickle.load(fh)

        if not isinstance(obj, dict):
            raise TypeError(f"{f.name} does not contain a dictionary")

        if "result_dict" not in obj:
            raise KeyError(f"{f.name} is missing 'result_dict'")

        if "stock_df" not in obj:
            raise KeyError(f"{f.name} is missing 'stock_df'")

        result_dict = obj["result_dict"]
        stock_df = obj["stock_df"]

        if not isinstance(result_dict, dict):
            raise TypeError(f"{f.name}['result_dict'] is not a dictionary")

        print(f"Found {len(result_dict)} dates")

        if stock_df_master is None:
            stock_df_master = stock_df
        else:
            if check_stock_df:
                try:
                    pd.testing.assert_frame_equal(
                        stock_df_master.reset_index(drop=True),
                        stock_df.reset_index(drop=True),
                        check_dtype=False,
                    )
                except AssertionError:
                    raise ValueError(
                        f"stock_df in {f.name} is not the same as the first file"
                    )

        for date_key, value in result_dict.items():
            if date_key in merged_result_dict:
                if on_conflict == "overwrite":
                    merged_result_dict[date_key] = value
                elif on_conflict == "skip":
                    continue
                elif on_conflict == "raise":
                    raise ValueError(f"Duplicate date key found: {date_key}")
                else:
                    raise ValueError(
                        "on_conflict must be 'overwrite', 'skip', or 'raise'"
                    )
            else:
                merged_result_dict[date_key] = value

    merged_obj = {
        "result_dict": merged_result_dict,
        "stock_df": stock_df_master,
    }

    print(f"\nFinal merged result_dict has {len(merged_result_dict)} dates")

    if output_file is not None:
        output_path = Path(output_file)

        with output_path.open("wb") as fh:
            pickle.dump(merged_obj, fh)

        print(f"Saved merged object to: {output_path}")

    return merged_obj



import time
import requests
import pandas as pd

# ============================================================
# MAIN EXECUTION INTERFACE
# ============================================================
def get_snapshot(
    api_key="iR3EPxer9Pc6bFkHBmZgxOZyJ0Hw4smF",
    underlying="COST",
    limit=250,
    request_sleep=0.01,
    market_date_override=None,
    output_file=None
):
    """
    Downloads, flattens, cleans, and saves a real-time options chain snapshot.
    Set market_date_override="YYYY-MM-DD" to manually set the session date.
    """
    if output_file is None:
        output_file = f"{underlying}_latest_option_chain_snapshot.csv"

    print("\n" + "=" * 60)
    print(f"Starting Option Chain Snapshot Pipeline for: {underlying}")
    print("=" * 60)

    # 1. Download data
    results, extracted_at_utc = _download_raw_snapshot(api_key, underlying, limit, request_sleep)
    
    # 2. Flatten and process data
    snapshot_df = _flatten_snapshot(results)

    if not snapshot_df.empty:
        # Handle date overrides conditionally
        if market_date_override is not None:
            snapshot_df["market_date"] = pd.to_datetime(market_date_override).normalize()
            print(f"Applying manual market date override: {market_date_override}")
        else:
            print("Using automatic market timestamps from API response.")

        # Post-process extraction timestamps safely
        snapshot_df["extracted_at_utc"] = pd.to_datetime(snapshot_df["extracted_at_utc"], utc=True, errors="coerce")
        snapshot_df["extracted_at_et"] = snapshot_df["extracted_at_utc"].dt.tz_convert("America/New_York")

        # 3. Save Master Output
        snapshot_df.to_csv(output_file, index=False)
        
        print(f"\nSUCCESS Summary:")
        print(f"  Total Shape Compiled : {snapshot_df.shape}")
        print(f"  Assigned Market Date : {snapshot_df['market_date'].iloc[0]}")
        print(f"  Extraction Time (ET) : {snapshot_df['extracted_at_et'].iloc[0]}")
        print(f"  Output Saved to      : {output_file}\n")
        return snapshot_df
    else:
        print("\nNo snapshot data downloaded.")
        return pd.DataFrame()


# ============================================================
# PRIVATE BACKEND HELPERS
# ============================================================
def _download_raw_snapshot(api_key, underlying, limit, request_sleep):
    base_url = "https://api.massive.com"
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{base_url}/v3/snapshot/options/{underlying}"
    params = {"limit": limit}
    
    all_results = []
    page = 1
    extracted_at_utc = pd.Timestamp.utcnow()

    while url is not None:
        print(f"  Downloading snapshot page {page}")
        r = requests.get(url, headers=headers, params=params)

        if r.status_code == 429:
            print("  Rate limited. Waiting 30 seconds...")
            time.sleep(30)
            continue
        if r.status_code == 401:
            raise RuntimeError("401 Unauthorized. Check API key.")
        if r.status_code != 200:
            print("  URL:", r.url, " | Status:", r.status_code)
            return [], extracted_at_utc

        data = r.json()
        results = data.get("results", [])

        for row in results:
            row["extracted_at_utc"] = extracted_at_utc.isoformat()

        all_results.extend(results)
        url = data.get("next_url")
        params = None
        page += 1
        time.sleep(request_sleep)

    return all_results, extracted_at_utc


def _flatten_snapshot(results):
    rows = []
    for x in results:
        details = x.get("details", {}) or {}
        greeks = x.get("greeks", {}) or {}
        day = x.get("day", {}) or {}
        last_quote = x.get("last_quote", {}) or {}
        last_trade = x.get("last_trade", {}) or {}
        underlying_asset = x.get("underlying_asset", {}) or {}

        rows.append({
            "extracted_at_utc": x.get("extracted_at_utc"),
            "ticker": details.get("ticker"),
            "underlying_ticker": details.get("underlying_ticker"),
            "contract_type": details.get("contract_type"),
            "exercise_style": details.get("exercise_style"),
            "expiration_date": details.get("expiration_date"),
            "strike_price": details.get("strike_price"),
            "shares_per_contract": details.get("shares_per_contract"),
            "implied_volatility": x.get("implied_volatility"),
            "open_interest": x.get("open_interest"),
            "break_even_price": x.get("break_even_price"),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
            "day_open": day.get("open"),
            "day_high": day.get("high"),
            "day_low": day.get("low"),
            "day_close": day.get("close"),
            "day_volume": day.get("volume"),
            "day_vwap": day.get("vwap"),
            "bid": last_quote.get("bid"),
            "ask": last_quote.get("ask"),
            "bid_size": last_quote.get("bid_size"),
            "ask_size": last_quote.get("ask_size"),
            "quote_timestamp": last_quote.get("last_updated"),
            "last_price": last_trade.get("price"),
            "last_size": last_trade.get("size"),
            "trade_timestamp": last_trade.get("sip_timestamp"),
            "underlying_price": underlying_asset.get("price"),
            "underlying_timestamp": underlying_asset.get("last_updated"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Numeric formatting rules
    numeric_cols = [
        "strike_price", "shares_per_contract", "implied_volatility", "open_interest",
        "break_even_price", "delta", "gamma", "theta", "vega", "day_open", "day_high",
        "day_low", "day_close", "day_volume", "day_vwap", "bid", "ask", "bid_size",
        "ask_size", "last_price", "last_size", "underlying_price"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Native parsed metadata dates
    df["extracted_at_utc"] = pd.to_datetime(df["extracted_at_utc"], utc=True, errors="coerce")
    df["extracted_at_et"] = df["extracted_at_utc"].dt.tz_convert("America/New_York")
    df["extracted_date_et"] = df["extracted_at_et"].dt.tz_localize(None).dt.normalize()

    # Back-populate timestamp checks
    for prefix in ["quote", "trade", "underlying"]:
        col = f"{prefix}_timestamp"
        df[f"{prefix}_dt_utc"] = pd.to_datetime(df[col], unit="ns", utc=True, errors="coerce") if col in df.columns else pd.NaT

    df["market_dt_utc"] = df["quote_dt_utc"].fillna(df["trade_dt_utc"]).fillna(df["underlying_dt_utc"])
    df["market_dt_et"] = df["market_dt_utc"].dt.tz_convert("America/New_York")
    df["market_date"] = df["market_dt_et"].dt.tz_localize(None).dt.normalize()

    # Derived processing targets
    if {"bid", "ask"}.issubset(df.columns):
        df["mid_price"] = 0.5 * (df["bid"] + df["ask"])
    if {"strike_price", "underlying_price"}.issubset(df.columns):
        df["moneyness"] = df["strike_price"] / df["underlying_price"]
        
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
    return df


def get_data(
    api_key="iR3EPxer9Pc6bFkHBmZgxOZyJ0Hw4smF",
    underlying="USO",
    start_date="2026-01-01",
    end_date="2026-05-24",
    max_contracts_to_price=None,
    min_dte=0,
    max_dte=21,
    min_strike=1,
    drop_blank_price_rows=True,
    require_positive_close=True,
    output_filename="USO_contracts_with_prices_ALL_DAYS.csv"
):
    """
    Downloads, processes, and combines options market data across a range of dates 
    from the Massive API into a single consolidated CSV file without saving intermediate pieces.
    Includes network timeout and retry handlers.
    """
    base_url = "https://api.massive.com"
    headers = {"Authorization": f"Bearer {api_key}"}
    limit = 1000
    request_sleep = 0.00
    TIMEOUT_SEC = 60  # Max seconds to wait for a connection before failing

    as_of_dates = pd.date_range(start_date, end_date, freq="B").strftime("%Y-%m-%d").tolist()
    
    # --------------------------------------------------------
    # NESTED HELPERS
    # --------------------------------------------------------
    def _get_contracts_for_day(as_of):
        url = f"{base_url}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "as_of": as_of,
            "expiration_date.gte": as_of,
            "limit": limit,
        }
        all_results = []
        page = 1
        
        while url is not None:
            print(f"  Contracts page {page} for {as_of}")
            try:
                r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT_SEC)
                
                if r.status_code == 429:
                    print("  Rate limited. Waiting 30 seconds...")
                    time.sleep(30)
                    continue
                if r.status_code == 401:
                    raise RuntimeError("401 Unauthorized. Check API key.")
                r.raise_for_status()
                
                data = r.json()
                all_results.extend(data.get("results", []))
                url = data.get("next_url")
                params = None
                page += 1
                time.sleep(request_sleep)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                print(f"  [Network Warning] Page {page} failed connection. Retrying in 5s... Error: {e}")
                time.sleep(5)
                continue
            
        df = pd.DataFrame(all_results)
        if not df.empty:
            df["as_of_date"] = as_of
        return df

    def _get_ticker_summary(option_ticker, as_of):
        url = f"{base_url}/v1/open-close/{option_ticker}/{as_of}"
        
        # Max 3 attempts per individual ticker contract to keep pipeline moving
        for attempt in range(1, 4):
            try:
                r = requests.get(url, headers=headers, params={"adjusted": "true"}, timeout=TIMEOUT_SEC)
                
                if r.status_code == 429:
                    print(f"  Rate limited on {option_ticker}. Waiting 30 seconds...")
                    time.sleep(30)
                    continue
                    
                if r.status_code != 200:
                    print(f"  Failed {option_ticker}: {r.status_code}")
                    return None
                    
                data = r.json()
                return {
                    "ticker": option_ticker,
                    "date": data.get("from"),
                    "open": data.get("open"),
                    "high": data.get("high"),
                    "low": data.get("low"),
                    "close": data.get("close"),
                    "volume": data.get("volume"),
                    "premarket": data.get("preMarket"),
                    "afterhours": data.get("afterHours"),
                }
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                print(f"  [Network Timeout] {option_ticker} attempt {attempt}/3 timed out. Retrying...")
                time.sleep(2)
                
        print(f"  [Skipped] Unstable network connection for contract: {option_ticker}")
        return None

    def _clean_prices(prices_df):
        if prices_df.empty:
            return prices_df
        numeric_cols = ["open", "high", "low", "close", "volume", "premarket", "afterhours"]
        for c in numeric_cols:
            if c in prices_df.columns:
                prices_df[c] = pd.to_numeric(prices_df[c], errors="coerce")
        if drop_blank_price_rows:
            prices_df = prices_df.dropna(subset=["open", "high", "low", "close"])
        if require_positive_close:
            prices_df = prices_df[prices_df["close"] > 0].copy()
        return prices_df.reset_index(drop=True)

    # --------------------------------------------------------
    # MAIN PIPELINE CORE LOOP
    # --------------------------------------------------------
    all_days_collected = []

    for as_of in as_of_dates:
        print("\n" + "=" * 60)
        print(f"Processing {underlying} Data for: {as_of}")
        print("=" * 60)

        # Step 1: Contracts
        contracts_df = _get_contracts_for_day(as_of)
        if contracts_df.empty:
            print(f"No contracts found for {as_of}")
            continue

        # Step 2: Filtering
        df = contracts_df.copy()
        df["expiration_date"] = pd.to_datetime(df["expiration_date"])
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        df["days_to_expiration"] = (df["expiration_date"] - df["as_of_date"]).dt.days

        filtered = df[
            (df["days_to_expiration"] >= min_dte) & 
            (df["days_to_expiration"] <= max_dte) & 
            (df["strike_price"] >= min_strike)
        ].copy()

        if max_contracts_to_price is not None:
            filtered = filtered.head(max_contracts_to_price)
            
        print(f"Contracts after filtering: {filtered.shape}")
        if filtered.empty:
            continue

        # Step 3: Fetch OHLCV Pricing Rows
        price_rows = []
        tickers = filtered["ticker"].tolist()
        
        for i, ticker in enumerate(tickers, start=1):
            print(f"  Pricing {i}/{len(tickers)}: {ticker}")
            row = _get_ticker_summary(ticker, as_of)
            if row is not None:
                price_rows.append(row)
            time.sleep(request_sleep)

        prices_df = pd.DataFrame(price_rows)
        prices_df = _clean_prices(prices_df)
        print(f"Clean price rows kept: {prices_df.shape}")
        if prices_df.empty:
            continue

        # Step 4: Merge Contracts and Prices in Memory
        day_final_df = filtered.merge(prices_df, on="ticker", how="inner")
        print(f"Day consolidated dataset shape: {day_final_df.shape}")
        
        all_days_collected.append(day_final_df)

    # --------------------------------------------------------
    # MASTER CONSOLIDATION AND SAVE
    # --------------------------------------------------------
    if all_days_collected:
        combined_df = pd.concat(all_days_collected, ignore_index=True)
        combined_df.to_csv(output_filename, index=False)
        print("\n" + "!" * 40)
        print(f"SUCCESS: Master dataset saved to: {output_filename}")
        print(f"Total shape compiled: {combined_df.shape}")
        print("!" * 40)
        return combined_df
    else:
        print("\nNo usable options data compiled across the target window.")
        return pd.DataFrame()

from pathlib import Path
import pickle
import pandas as pd


def quick_option_market(
    vendor_name,
    stock_filename,
    option_filename,
    maturity_filter,
    yield_curve_files,
    *,
    data_directory_path=None,
    ticker="USO",
    pickle_path=None,
):
    """
    Build and standardize an option market, retain selected OTM options,
    convert puts to call-equivalent prices using put-call parity, and return
    the combined option DataFrame and OptionMarketStandardizer object.

    Parameters
    ----------
    vendor_name : str
        Name of the option-data vendor.

    stock_filename : str or Path
        Stock-data filename.

    option_filename : str or Path
        Option-data filename or filename prefix.

    maturity_filter : sequence
        Maturity filter passed to ``OptionMarketStandardizer.keep_options``.

    yield_curve_files : str, Path, or sequence of str or Path
        Treasury yield-curve file or files used to construct the interest-rate
        surface.

    data_directory_path : str or Path, optional
        Directory containing the stock, option, and related input files.
        Defaults to the original Optiondata directory.

    ticker : str, default "USO"
        Underlying ticker.

    pickle_path : str or Path, optional
        Destination for saving the ``OptionMarketStandardizer`` object.
        No pickle is created when this is None.

    Returns
    -------
    otm_options_only_df : pandas.DataFrame
        Combined OTM call and put-call-parity-adjusted put observations.

    option_market_class : OptionMarketStandardizer
        Initialized option-market standardizer.
    """
    # Permit either one Treasury file or multiple Treasury files.
    if isinstance(yield_curve_files, (str, Path)):
        yield_curve_files = [yield_curve_files]

    yield_curve_files = [Path(file) for file in yield_curve_files]

    missing_files = [file for file in yield_curve_files if not file.exists()]
    if missing_files:
        missing_text = "\n".join(str(file) for file in missing_files)
        raise FileNotFoundError(
            f"The following yield-curve files were not found:\n{missing_text}"
        )


    data_directory_path = Path(data_directory_path)

    # 1. Build the yield curve and fit the Svensson surface.
    df_yield = build_yield_dataframe(yield_curve_files)

    rc_object = create_yield_curve(df_yield)

    sve_nsurface = rc_object.fit(
        "svensson",
        grid_days=[1, 365 * 3],
        fit_days_window=[1, 365 * 5],
    )

    # 2. Initialize the option-market standardizer.
    option_market_class = OptionMarketStandardizer(
        option_data_filename_prefix=option_filename,
        stock_data_filename=stock_filename,
        rate_curve_df=sve_nsurface,
        vendor_name=vendor_name,
        stock_date_col="date",
        stock_price_col="price",
        data_directory_path=data_directory_path,
        rate_date_col="Date",
        ticker=ticker,
    )

    # 3. Optionally save the standardized option-market object.
    if pickle_path is not None:
        save_path = Path(pickle_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Pickling option_market_class to: {save_path}")

        with save_path.open("wb") as file:
            pickle.dump(
                {"option_market": option_market_class},
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    # 4. Retain OTM calls.
    otm_calls = option_market_class.keep_options(
        maturity_filter=maturity_filter,
        moneyness_filter=[1.0, 1.3],
        min_volume_filter=-1,
        min_price_filter=0.01,
        option_right_filter="C",
    )

    # 5. Retain OTM puts.
    otm_puts = option_market_class.keep_options(
        maturity_filter=maturity_filter,
        moneyness_filter=[0.7, 0.99],
        min_volume_filter=-1,
        min_price_filter=0.01,
        option_right_filter="P",
    )

    # 6. Convert puts to call-equivalent prices and combine.
    otm_puts_to_calls = put_call_parity(otm_puts)

    otm_options_only_df = (
        pd.concat(
            [otm_calls, otm_puts_to_calls],
            ignore_index=True,
        )
        .sort_values(["date", "exdate", "strike"])
        .reset_index(drop=True)
    )

    return otm_options_only_df, option_market_class


from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd


def quick_calibrate(
    otm_options_df,
    T_grid,
    m_bounds=(0.02, 3.5),
    m_grid_n=300,
    plot=True,
    plot_surfaces=True,
    save_plots=False,
    save_dir: Optional[Union[str, Path]] = None,
    save_tag: Optional[str] = None,
    date_range=None,
    ticker=None,
    interactive_surfaces=False,
    show_plots=True,
    surface_x_axis="lr",
    surface_x_bounds=None,
):
    """
    Calibrate the Heston-Kou model for each selected date and optionally
    produce and save diagnostic plots.

    Parameters
    ----------
    otm_options_df : pandas.DataFrame
        Option-market DataFrame.

    T_grid : array-like
        Maturity grid used by the global surface pricer.

    m_bounds : tuple, default (0.02, 3.5)
        Moneyness bounds used when constructing the pricing grid.

    m_grid_n : int, default 300
        Number of moneyness-grid points.

    plot : bool, default True
        Whether to create call-curve and RND panel plots.

    plot_surfaces : bool, default True
        Whether to create IV and RND surface plots.

    save_plots : bool, default False
        Whether to save generated plots.

    save_dir : str or Path, optional
        Parent directory in which plot subfolders are created. When omitted,
        defaults to:

            results_tables_and_figures/quick_calibrate

    save_tag : str, optional
        Prefix used in saved filenames. If omitted, ticker is used. If ticker
        is also omitted, "calibration" is used.

        Example filenames:

            USO_IVsurf_2025_01_03.png
            USO_RNDsurf_2025_01_03.png

    date_range : None, tuple, or list
        Date-selection specification.

        None
            Use all dates.

        ("2025-01-01", "2025-03-31")
            Use dates between start and end, inclusive.

        ["2025-01-03", "2025-01-10"]
            Use only the listed dates.

    ticker : str, optional
        Ticker used in plot titles and default filenames.

    interactive_surfaces : bool, default False
        Whether IV and RND surfaces should use Plotly.

        For reliable PNG saving, use False unless the required Plotly static
        image dependencies are installed.

    show_plots : bool, default True
        Whether generated figures should be displayed.

    surface_x_axis : {"k", "lr", "r"}, default "lr"
        Horizontal axis for IV and RND surface plots.

    surface_x_bounds : tuple, optional
        Bounds applied to the selected surface x-axis. For example,
        (-0.5, 0.5) for log returns.

    Returns
    -------
    dict
        Dictionary mapping each calibration date to its pricing results.
    """
    x0 = dict(
                # fast variance factor (front-end skew / short maturities)
                v0=0.1,    # initial variance v1(0)
                theta=0.5,  # long-run mean of v1
                kappa=6.0, # mean reversion speed of v1 (large => fast)
                sigma_v=0.2,  # vol-of-vol of v1 (controls smile strength)
                rho=-0.6,   # corr(return shock, v1 shock): negative => left skew
            
                # # slow variance factor (term structure / persistence)
                # v02=0.06,    # initial variance v2(0)
                # theta2=0.5,  # long-run mean of v2
                # kappa2=15.5, # mean reversion speed of v2 (smaller => more persistent)
                # sigma2=0.2,  # vol-of-vol of v2
                # rho2=-0.20,  # corr(return shock, v2 shock)
            
                # Kou jumps (double exponential jump sizes)
                lam=0.6,     # jump intensity: expected jumps per year
                p_up=0.50,   # P(jump is upward)
                eta1=15.0,   # upward jump rate: E[J | J>0] = 1/eta1
                eta2=15.0,   # downward jump rate: E[-J | J<0] = 1/eta2
            )
        
            ###Setting up boundaries
    lb = dict(
            v0=0.005, theta=0.4, kappa=0.5,  sigma_v=0.1, rho=-0.85,
            #v02=0.005, theta2=0.05, kappa2=15.5,  sigma2=0.1, rho2=-0.95,
            lam=0.02, p_up=0.05, eta1=10.0, eta2=10.0,
            )
            
    ub = dict(
            v0=5.0, theta=1.0, kappa=5000.0, sigma_v=5.0, rho=0.85,
            #v02=10.9, theta2=1.0, kappa2=5000.0,  sigma2=15.9, rho2=0.95,  
            lam=30.0, p_up=0.95, eta1=20.0, eta2=20.0,
            )
    RND_today = {}

    safety_clip = SafetyClipConfig(
        enabled=True,
        clip_left=True,
        clip_right=True,
        center="mode",
    )

    iv_cfg = IVConfig(
        sigma_init=0.3,
        sigma_lo=1e-8,
        sigma_hi=5.0,
        newton_max_iter=150,
        newton_tol=1e-11,
        vega_floor=1e-11,
        brent_maxiter=150,
        time_value_floor=1e-11,
        reject_low_vega=1e-11,
    )

    # ==================================================
    # Plot directory setup
    # ==================================================
    plot_paths = {}

    if save_plots:
        base_save_dir = (
            Path(save_dir)
            if save_dir is not None
            else Path("results_tables_and_figures") / "quick_calibrate"
        )

        filename_tag = save_tag or ticker or "calibration"

        plot_paths = {
            "call_panels": base_save_dir / "call_panels",
            "rnd_panels": base_save_dir / "rnd_panels",
            "iv_surfaces": base_save_dir / "iv_surfaces",
            "rnd_surfaces": base_save_dir / "rnd_surfaces",
            "arbitrage_heatmaps": base_save_dir / "arbitrage_heatmaps",
        }

        for folder in plot_paths.values():
            folder.mkdir(parents=True, exist_ok=True)

    else:
        filename_tag = save_tag or ticker or "calibration"

    # ==================================================
    # Date selection
    # ==================================================
    df = otm_options_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    all_dates = pd.DatetimeIndex(
        df["date"].dropna().unique()
    ).sort_values()

    if date_range is None:
        date_list = all_dates

    elif isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0]).normalize()
        end_date = pd.to_datetime(date_range[1]).normalize()

        if start_date > end_date:
            raise ValueError(
                "The starting date cannot occur after the ending date."
            )

        date_list = all_dates[
            (all_dates >= start_date) &
            (all_dates <= end_date)
        ]

    elif isinstance(date_range, list):
        requested_dates = pd.DatetimeIndex(
            pd.to_datetime(date_range)
        ).normalize()

        date_list = all_dates[all_dates.isin(requested_dates)]

    else:
        raise ValueError(
            "date_range must be None, a tuple such as "
            "('2025-01-01', '2025-03-31'), or a list of dates."
        )

    if len(date_list) == 0:
        raise ValueError(
            "No option dates matched the requested date_range."
        )

    # ==================================================
    # Main calibration loop
    # ==================================================
    for date in date_list:
        option_day_df = df.loc[df["date"] == date].copy()

        readable_date = date.strftime("%Y_%m_%d")
        formatted_date = date.strftime("%Y-%m-%d")

        print(f"Calibrating {formatted_date}...")

        # ----------------------------------------------
        # Arbitrage repair
        # ----------------------------------------------
        repair_output = CallSurfaceArbRepair(
            RepairConfig()
        ).repair_one_date(option_day_df)

        repaired_df = repair_output["df_rep"]
        plot_data = repair_output["plot_data"]

        heatmap_save_path = None
        if save_plots:
            heatmap_save_path = (
                plot_paths["arbitrage_heatmaps"]
                / f"{filename_tag}_arbitrage_heatmap_{readable_date}.png"
            )

        if plot or save_plots:
            plot_perturb(
                plot_data,
                save=heatmap_save_path,
            )

        # ----------------------------------------------
        # Construct daily surface data
        # ----------------------------------------------
        original_day = make_day_from_df(
            repaired_df,
            price_col="C_rep",
        )

        day = CallSurfaceDay(
            S0=original_day.S0,
            r=original_day.r,
            q=original_day.q,
            K_obs=original_day.K_obs,
            T_obs=original_day.T_obs,
            C_obs=original_day.C_obs,
            ticker=original_day.ticker,
            date=date,
        )

        # ----------------------------------------------
        # Fit Heston-Kou surface
        # ----------------------------------------------
        pr = GlobalSurfacePricer(
            "heston_kou",
            Umax=1500.0,
            n_quad=1500,
        )

        pr.fit(
            day,
            x0=x0,
            bounds=(lb, ub),
        )

        results = pr.price(
            day,
            safety_clip=safety_clip,
            iv_cfg=iv_cfg,
            grid_mode="moneyness",
            m_bounds=m_bounds,
            m_grid_n=m_grid_n,
            T_grid=T_grid,
            compute_moments=True,
            compute_iv=True,
            compute_rnd=True,
            compute_delta=True,
            compute_cdf=True,
            compute_gamma=True,
        )

        print(
            f"Params for {formatted_date}: "
            f"{results['params']}"
        )

        RND_today[date] = results

        # ----------------------------------------------
        # Output paths for this date
        # ----------------------------------------------
        if save_plots:
            call_panel_path = (
                plot_paths["call_panels"]
                / f"{filename_tag}_call_panels_{readable_date}.png"
            )

            rnd_panel_path = (
                plot_paths["rnd_panels"]
                / f"{filename_tag}_RNDpanels_{readable_date}.png"
            )

            iv_surface_path = (
                plot_paths["iv_surfaces"]
                / f"{filename_tag}_IVsurf_{readable_date}.png"
            )

            rnd_surface_path = (
                plot_paths["rnd_surfaces"]
                / f"{filename_tag}_RNDsurf_{readable_date}.png"
            )

        else:
            call_panel_path = None
            rnd_panel_path = None
            iv_surface_path = None
            rnd_surface_path = None

        # ----------------------------------------------
        # Panel plots
        # ----------------------------------------------
        if plot or save_plots:
            panels.call_panels(
                results,
                day=day,
                n_panels=6,
                title=f"Call Curves on {formatted_date} for {ticker}",
                T_cluster_tol=1 / 365,
                x_axis="r",
                x_bounds=(0.5,1.5),
                save=call_panel_path,
                show=show_plots,
            )

            panels.rnd_panels(
                results,
                title=f"RND on {formatted_date} for {ticker}",
                show_spot=True,
                x_axis="lr",
                pct_lower=None,
                pct_upper=None,
                snap_percentiles_to_traded_strikes=False,
                only_plot_traded_maturities=False,
                rnd_dict={"day": day},
                x_bounds=(-0.85, 1.75),
                save=rnd_panel_path,
                show=show_plots,
            )

        # ----------------------------------------------
        # IV and RND surface plots
        # ----------------------------------------------
        if plot_surfaces or save_plots:
            surfaces.iv_surface_plot(
                results,
                title=(
                    f"Implied Volatility Surface on "
                    f"{formatted_date} for {ticker}"
                ),
                interactive=interactive_surfaces,
                show=show_plots,
                x_axis=surface_x_axis,
                x_bounds=surface_x_bounds,
                save=iv_surface_path,
            )

            surfaces.rnd_surface_plot(
                results,
                title=(
                    f"Risk-Neutral Density Surface on "
                    f"{formatted_date} for {ticker}"
                ),
                interactive=interactive_surfaces,
                show=show_plots,
                x_axis=surface_x_axis,
                x_bounds=surface_x_bounds,
                save=rnd_surface_path,
            )

    return RND_today

import pandas as pd

def slice_dict_by_date(rnd_dict, start_date=None, end_date=None):
    """
    Slice a dictionary whose keys are dates.

    Parameters
    ----------
    rnd_dict : dict
        Dictionary keyed by dates (str, datetime, or Timestamp).
    start_date : str or datetime, optional
        Inclusive start date.
    end_date : str or datetime, optional
        Inclusive end date.

    Returns
    -------
    dict
        Filtered dictionary.
    """

    start = pd.Timestamp.min if start_date is None else pd.to_datetime(start_date)
    end = pd.Timestamp.max if end_date is None else pd.to_datetime(end_date)

    return {
        k: v
        for k, v in rnd_dict.items()
        if start <= pd.to_datetime(k) <= end
    }
# test_data = get_data(
#         api_key="iR3EPxer9o;iPc6bFkHBmZgxOZyJ0Hw4smF",
#         underlying="USO",
#         start_date="2025-08-30",      # Just a 2-day testing window
#         end_date="2026-05-22",
#         max_contracts_to_price=1000,     # Cap it at 5 contracts so it finishes in 2 seconds
#         output_filename="TEST_USO_output.csv"
#     )