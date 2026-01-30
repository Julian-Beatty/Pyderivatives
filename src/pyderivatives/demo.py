from pyderivatives.arbitrage_repair import RepairConfig, CallSurfaceArbitrageRepair
from pyderivatives.yieldcurve.core import create_yield_curve
from pyderivatives.yieldcurve.build_yield_curve import build_yield_dataframe
from pyderivatives.yieldcurve.plotting_functions import plot_yield_surface, plot_yield_curve
import matplotlib.pyplot as plt




#############Build yield curve#####
files=["C:/Users/beatt/Spyder directory/State Price Density/Simulation_masterfile/par-yield-curve-rates-1990-2022.csv",
"C:/Users/beatt/Spyder directory/State Price Density/Simulation_masterfile/daily-treasury-rates (3).csv",
"C:/Users/beatt/Spyder directory/State Price Density/Simulation_masterfile/daily-treasury-rates (2).csv",
"C:/Users/beatt/Spyder directory/State Price Density/Simulation_masterfile/daily-treasury-rates (1).csv"]


yield_df=build_yield_dataframe(files)
yc_object=create_yield_curve(yield_df)
rate_surface=yc_object.fit("nelson_siegel",grid_days=[1,365],fit_days_window=[1,365*5])

yield_dict={"nelson":rate_surface}

fig=plot_yield_curve(yield_dict,yield_df,maturities_days_window=[1,375*5],single_date="2022-02-04")



help(plot_yield_curve)
############

from pyderivatives.option_market_standardizer.core import OptionMarketStandardizer
option_data_filename="Brazil_ETF_Options.csv"
stock_data_filename="Brazil_stock.csv"
rate_curve_df=rate_surface
data_directory_paths="C:/Users/beatt/Spyder directory/State Price Density/Simulation_masterfile"
vendor_name="optionmetrics"
stock_date_col="date"
stock_price_col="price"
rate_date_col="Date"




brazil_market=OptionMarketStandardizer(option_data_filename_prefix=option_data_filename,stock_data_filename=stock_data_filename,
                                       rate_curve_df=rate_surface, data_directory_path=data_directory_paths,
                           vendor_name=vendor_name,stock_date_col=stock_date_col,stock_price_col=stock_price_col,rate_date_col="Date")






brazil_market.load()
opt_std=brazil_market.opt_std
opt_raw=brazil_market.options_raw
opt_std = brazil_market.keep_options(
    date_filter=["2021-08-01", "2025-08-02"],
    maturity_filter=[7, 90],
    moneyness_filter=[0.5, 1.5],
    min_volume_filter=0.1,
    min_price_filter=0.05,
    option_right_filter="C",
)
from pyderivatives.arbitrage_repair.arb_repair import CallSurfaceArbitrageRepair, RepairConfig
from pyderivatives.option_market_standardizer.utils import slice_call_surfaces_by_date,extract_call_surface_from_df



call_slice_dict=slice_call_surfaces_by_date(opt_std)
call_slice_dict2=slice_call_surfaces_by_date(option_df)

cfg=RepairConfig()
classif=CallSurfaceArbitrageRepair(cfg)


unique_dates=list(call_slice_dict.keys())
t=classif.repair(call_slice_dict[unique_dates[0]])

v=call_slice_dict[unique_dates[0]]

date_strings = [d.strftime("%Y_%m_%d") for d in unique_dates]


tag = str(date_strings[0]).replace("-", "_")
classif.save_all_plots(t, tag=tag, title_prefix=str(tag))


help(CallSurfaceArbitrageRepair)

strikes, maturities, C_mkt, S0, r=extract_call_surface_from_df(t,price_col="C_rep")

help(extract_call_surface_from_df)