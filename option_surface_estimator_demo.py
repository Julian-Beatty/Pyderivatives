# from main_option_market import*
# from create_yield_curve import*
import pickle
from option_surface_estimator import*
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

    # --- choose row model here --- ##Eg. Using Fixed Normals here
cfg = SurfaceConfig(
        row_model=RowModelConfig(
            model="mixture_fixed",  # <-- swap: "mixture_fixed","mixture_evolutionary or "gen_gamma", generalized Lambda Distribution "gld" Make sure to put the right input arguments or it falls back to default values.
            n_lognormal=2, #<-- You can fix the number of lognormals
            n_weibull=0, #<-- You can fix the number of Weibulls
            var_c=0.1, #<-- This constraint basically prevents very tiny spikes.
            var_penalty=1e7, #<-- Basically hard constraint as in Yifan, but can soften this if you want.
            random_starts=1,
            seed=3,
        ),
        full_estimation=True,
        fine_strike_factor=6, #<-- Controls how dense your strike grid will be
        strike_extension=0.85, # <-- Controls how much extrapolation on the strike. Set to 70%.
        use_maturity_interp=True, # <-- If you want to interpolate across maturity.
        day_step=14, #<-- Controls how much interpolation on the maturity. Set to 14 day intervals, can be set to shorter but more time%.
    )
cfg = SurfaceConfig(
        row_model=RowModelConfig(
            model="mixture_evolutionary",  
            M_max=2, #<-- Controls the maximum evolutions. Generally need only 2 or 3
            var_c=0.1, 
            var_penalty=1e7,
            use_wald=True, # <-- Uses Wald test for diagnostic evolution
            wald_alpha=0.05, # <-- Rejection threshold for the Wald test
            wald_p=1, #<-- Fourier basis setting. He recommends p=q=1, possibly more maybe up to 2 or 3.
            wald_q=1,
            random_starts=1,
            seed=855,
        ),
        full_estimation=True,
        fine_strike_factor=6, #<-- Controls how dense your strike grid will be
        strike_extension=0.85, # <-- Controls how much extrapolation on the strike. Set to 70%.
        use_maturity_interp=True, # <-- If you want to interpolate across maturity.
        day_step=14, #<-- IControls how much interpolation on the maturity. Set to 14 day intervals, can be set to shorter but more time%.
    )

cfg = SurfaceConfig(#Generalized Lambda Distribution
        row_model=RowModelConfig( #Some ultra low maturities at extreme strikes are nan.
            model="gld",  # <-- Experimental setting. Check to cross pannel graphs to ensure fit. Check Option_estimator "row_config" for more optional parameters if extra tweaking needed.
            seed=20,
            gld_sigma0= 0.45 # <-- Initial value for sigma, very sensitive might have to tweak, but keep around 0.3-0.45

        ),
        full_estimation=True,
        fine_strike_factor=6, #<-- Controls how dense your strike grid will be
        strike_extension=0.85, # <-- Controls how much extrapolation on the strike. Set to 70%.
        use_maturity_interp=True, # <-- If you want to interpolate across maturity.
        day_step=14, #<-- IControls how much interpolation on the maturity. Set to 14 day intervals, can be set to shorter but more time%.
    )

cfg = SurfaceConfig(
        row_model=RowModelConfig( #Generalized Gamma Distribution
            model="gen_gamma",  # <-- Experimental setting. Check to cross pannel graphs to ensure fit. 
            seed=20,
        ),
        full_estimation=True,
        fine_strike_factor=6, #<-- Controls how dense your strike grid will be
        strike_extension=0.85, # <-- Controls how much extrapolation on the strike. Set to 70%.
        use_maturity_interp=True, # <-- If you want to interpolate across maturity.
        day_step=14, #<-- IControls how much interpolation on the maturity. Set to 14 day intervals, can be set to shorter but more time%.
    )

####################Step 2: Call Surface Estimator

est = CallSurfaceEstimator(
        strikes=strikes,
        maturities=maturities,
        S0=S0,
        r=r,
        config=cfg,
    )

obj = est.fit_surface(C_mkt) #<-- Returns Call/IV/RND surface, also returns moments (mean/var/ect.) of the log return distribution.
#################Step 3: Sanity check. Check All RND slices, and Call slices
    # --- surfaces ---
est.plot_call_and_iv_surfaces() ##Don't worry too much about IV surface. IV not well defined for far OTM strikes.
est.plot_some_rnds(layout="panels",n_curves=10) #plots RND pannels
est.plot_some_rnds(layout="overlay",n_curves=10) #plots RND pannels

est.plot_rnd_surface(title="Risk Neutral Density Surface (Multi-Curve)") #plots RND multi-curve surface

plot_original_vs_final_surface(
    est,
    strikes_orig=strikes,
    maturities_orig=maturities,
    C_orig=C_mkt,
    title="Chevron Surface: Observed vs Final Estimated Call Surface",
)

plot_random_observed_vs_model_curve(
    est=est,
    strikes_orig=strikes,
    maturities_orig=maturities,
    C_orig=C_mkt,
    plot_all_original=True,
    title_prefix="Observed vs Stage-1 Model Call Curve",
    plot_interpolated=False,   # True if you also want interpolated curves
)

