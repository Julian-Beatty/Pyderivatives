# from main_option_market import*
# from create_yield_curve import*
import pickle
from call_surface_precleaner import*
import numpy as np
import matplotlib.pyplot as plt
######Starter code using main_option market. Commented out. I will just import data to save time. But if you can edit this code to to easily generate new surfaces.

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

# data_result={"Option_df": option_df}
# with open("Chevron_options.pkl", "wb") as f:
#     pickle.dump(data_result, f)

#######################Program Begins here ##Directly load data for convenience###################################
with open("Chevron_options.pkl", "rb") as f:
    pickle_results = pickle.load(f)
    

option_df=pickle_results["Option_df"] ##Call Option Surface

####################Step 1: Unravel neccessary attributes from option_df
strikes, maturities, C_mkt, S0, r = extract_call_surface_from_df(option_df)


####################Step 2: Configure mixture settings. This calls the mixture from previous packages
# Mixture config: Configure the parameters for the mixtures. I set to mixture of 2 or 3 lognormals, or you can use the evolutionary algo
config = MixtureSurfaceConfig(
        n_lognormal=2, ##Be wary of overfitting and computational time.
        n_weibull=0, ##Not recommended to have weibuls.
        var_c=0.1, ##recommended 0.1, but can increase if densities are unstable.
        var_penalty=1e4,
        random_starts=1, ##keep at one
        seed=123,
        full_estimation=True,       # replaces nan and observed qoutes with mixture estimations. Set to true if you want clean surface.
        use_maturity_interp=True,   # Interpolates across time using cubic spline
        day_step=14,                 # bi-weekly maturity grid can set to 1 if you want daily. Calculation time is very long
        fine_strike_factor=5,    #  An integer. evaluates mixtures on a grid that is twice as dense. If densities look blocky, increase this number.
        strike_extension=0.6##I extend the observed strike grid by up to 40% just to make sure the RND is complete.
    )
##Alternatively you can use the evolutionary algorithmn which matches exactly Yifan 2024
# config = MixtureSurfaceConfig(
#         use_evolutionary=True, #uses evolutionary algorithmn
#         M_max=2, #Can go higher for more precision but much slower
#         var_c=0.1, ##recommended 0.1, but can increase if densities are unstable.
#         var_penalty=1e4,
#         random_starts=1, ##keep at one
#         seed=123,
#         full_estimation=True,       # replaces nan and observed qoutes with mixture estimations. Set to true if you want clean surface.
#         use_maturity_interp=False,   # Interpolates across time using cubic spline, much slower
#         day_step=7,                 # weekly maturity grid can set to 1 if you want daily. Calculation time is very long
#         fine_strike_factor=15,    #  An integer. evaluates mixtures on a grid that is twice as dense. If densities look blocky, increase this number.
#         strike_extension=0.6 ##I extend the observed strike grid by up to 40% just to make sure the RND is complete.
#     )




# initialize mixture cleaner class
cleaner = MixtureCallSurfaceCleaner(
        strikes=strikes,
        maturities=maturities,
        S0=S0,
        r=r,
        config=config,
    )

#fits the entire call surface. Handles extrapolation, interpolation. Will be free of butterfly arbitrage. Result has the call/iv/rnd surface.
result = cleaner.fit_surface(C_mkt)
#################################################Plotting Diagnostics########################################
#Compare fitted vs original incomplete call surface. The original surface is highly illiquid, but we can still extract something reasonable.
plot_original_vs_final_surface(
    cleaner,
    strikes_orig=strikes,
    maturities_orig=maturities,
    C_orig=C_mkt,title="Chevron Call Surface on 2020-04-01: Covid Shock"
)

# Plot cleaned call + IV surfaces
cleaner.plot_call_and_iv_surfaces()
#Plot RND surface (multi-curve)
cleaner.plot_rnd_surface()
#Plot each individual RND curve just to make sure
cleaner.plot_some_rnds_panel(n_curves=25)





#######Plotting individual call curves just to make sure mixtures fitted well.
# 1) Plot some random call curves just to make sure everything is aligned
plot_random_observed_vs_mixture_curve(
    cleaner,
    strikes_orig=strikes,
    maturities_orig=maturities,
    C_orig=C_mkt,
    n_curves=25,
    random_state=0,
)

# 2) Plot some original slices AND some interpolated+mixture slices
plot_random_observed_vs_mixture_curve(
    cleaner,
    strikes_orig=strikes,
    maturities_orig=maturities,
    C_orig=C_mkt,
    n_curves=3,
    random_state=123,
    plot_interpolated=True,
)



# 3) Plot *all* original call slices vs stage-1 mixtures, but no interpolated ones
plot_random_observed_vs_mixture_curve(
    cleaner,
    strikes_orig=strikes,
    maturities_orig=maturities,
    C_orig=C_mkt,
    plot_all_original=True,
)



# 4) Plot all originals and all interpolated curves
plot_random_observed_vs_mixture_curve(
    cleaner,
    strikes_orig=strikes,
    maturities_orig=maturities,
    C_orig=C_mkt,
    plot_all_original=True,
    plot_interpolated=True,
)
