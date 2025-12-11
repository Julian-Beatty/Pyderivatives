#from main_option_market import*
#from create_yield_curve import*
import pickle
from call_surface_precleaner import*
###Starter code using main_option market. Commented out. I will just import Gold data to save time. But if you can edit this code to to easily generate new surfaces.

# with open("yields_dict.pkl", "rb") as file:
#     loaded_dict = pickle.load(file)
    
    
# spot=loaded_dict["spot"]  
# forward=loaded_dict["forward"] 
# ###Datafiles

# option_data_prefix="options_gld.csv"
# stock_price_data_prefix='GLD price.csv'


# option_market=OptionMarket(option_data_prefix,stock_price_data_prefix,spot,forward) #Initialization
# option_df=option_market.original_market_df #Raw dataframe
# option_market.set_master_dataframe(volume=-1, maturity=[0,900],moneyness=[-0.95,0.95],rolling=False,date=["2021-06-01","2021-06-01"]) #Filter options
# option_df=option_market.master_dataframe 

# data_result={"GLD Options": option_df}
# with open("GLD_options.pkl", "wb") as f:
#     pickle.dump(data_result, f)

with open("GLD_options.pkl", "rb") as f:
    pickle_results = pickle.load(f)
    

option_df=pickle_results["GLD Options"] ##Call surface from GLD on 2021-06-01

####################Step 1: Unravel neccessary attributes from option_df
strikes, maturities, C_mkt, S0, r = extract_call_surface_from_df(option_df)


####################Step 2: Configure mixture settings. This calls the mixture from previous packages
# Mixture config: Configure the parameters for the mixtures. I set to mixture of 3 lognormals
config = MixtureSurfaceConfig(
        n_lognormal=3, ##Be wary of overfitting
        n_weibull=0,
        var_c=0.1,
        var_penalty=1e4,
        random_starts=1,
        seed=123,
        full_estimation=True,       # replaces nan and observed qoutes with mixture estimations. Set to true if you want clean surface.
        use_maturity_interp=True,   # Interpolates across time using cubic spline
        day_step=7,                 # weekly maturity grid can set to 1 if you want daily. Calculation time is very long
        fine_strike_factor=2,    #  An integer. evaluates mixtures on a grid that is twice as dense.
    )
# initialize mixture cleaner class
cleaner = MixtureCallSurfaceCleaner(
        strikes=strikes,
        maturities=maturities,
        S0=S0,
        r=r,
        config=config,
    )

##fits the entire call surface. Handles extrapolation, interpolation. Will be free of butterfly arbitrage
result = cleaner.fit_surface(C_mkt)

# Plot cleaned call + IV surfaces
cleaner.plot_call_and_iv_surfaces()
cleaner.plot_rnd_surface()

# Plot some RND curves from C_clean
cleaner.plot_some_rnds(n_curves=6)

plot_original_vs_final_surface(
    cleaner,
    strikes_orig=strikes,
    maturities_orig=maturities,
    C_orig=C_mkt,title="GLD Call Surface on 2021-06-01"
)