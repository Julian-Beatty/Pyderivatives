import numpy as np
import pandas as pd
from option_surface_estimator import*
from PricingKernel_estimator_schreindorfer_2025 import*


with open("btc_options_demo.pkl", "rb") as f:
        demo_data = pickle.load(f)
        

###Load demo data. BTC options from deribit after I preprocess. I truncated the data, it only goes for 40 days, but in practice you should use as long a time span as possible.
option_df=demo_data["option_df"]
stock_df=demo_data["stock_df"]

##We slice our option market into a dated dictionary, with key->qouted day :value->Option chain for that day (keeping only absolutely neccessary information)
cols = ["date", "exdate", "rounded_maturity", "stock_price",
        "risk_free_rate", "strike", "mid_price"]

call_surface_slice_dict = {
    d: g[cols].sort_values(["rounded_maturity", "strike"]).reset_index(drop=True)
    for d, g in option_df.groupby("date", sort=False)
}



########################################Using Call_surface estimator to get well defined Call and RND surfaces##################################################
###Calibrate Heston Kou on each slide store. For now, I recommend only the configuring either Kou,Heston,Bates,Heston_kou
#I'm going to calculate RND/call surfaces each day, and store them in this dictionary. I use a simple for loop.
information_dict={} 
unique_dates = option_df["date"].unique() #This may take a while, feel free to reduce the dates even further.
print(unique_dates)
for i in unique_dates:
    call_surface=call_surface_slice_dict[i]
    strikes, maturities, C_mkt, S0, r = extract_call_surface_from_df(call_surface)
    
    cfg = SurfaceConfig(
        row_model=RowModelConfig(
            model="kou_heston", ##You may use other models
            hkde_quad_N=96,
            hkde_quad_u_max=200.0,
            hkde_q=0.0,
            hkde_x0=HKDEParams(
                v0=0.03, theta=0.05, kappa=1.5, sigma_v=0.50, rho=-0.3,
                lam=0.8, p_up=0.5, eta1=6.0, eta2=10.0
            ),
            hkde_bounds=None,
            hkde_use_vega_weights=False,
            hkde_vega_floor=1e-3,
            hkde_w_cap=2e3,
            hkde_iv_obs=None,
            hkde_verbose=1,
            hkde_max_nfev=400,
        ),
        use_maturity_interp=True,
        day_step=1,
        strike_extension=0.8,
        fine_strike_factor=3,
        apply_safety_clip=True, ###Clips densities. If issue set to false. 
        safety_clip_center="spot", #Treats the center as roughly spot
        safety_clip_jump_factor=np.e, #Clipping threshold
    )
    est = CallSurfaceEstimator(
            strikes=strikes,
            maturities=maturities,
            S0=S0,
            r=r,
            config=cfg,
        )
    obj = est.fit_surface(C_mkt)
    
    ###Now I plot. These will save as folders in your directory.
    readable_date=i.replace("-", "_")
    
    est.plot_rnd_surface(title=f"Risk Neutral Density Surface (Multi-Curve) {i}",save=f"RND_surface_folder/RND_Surface_{readable_date}.png") #plots RND multi-curve surface
    plot_random_observed_vs_model_curve(
        est=est,
        strikes_orig=strikes,
        maturities_orig=maturities,
        C_orig=C_mkt,
        plot_all_original=True,
        title_prefix=f"Observed vs Fitted Curves Bitcoin {i}",
        save=f"Call_pannels/Call_pannels_{readable_date}.png"
    )
    
    plot_original_vs_final_surface(
        est,
        strikes_orig=strikes,
        maturities_orig=maturities,
        C_orig=C_mkt,
        title=f"Bitcoin Surface: Observed vs Final Estimated Call Surface {i}",
        save=f"Call_surfaces/Call_surface_{readable_date}.png"

    )
    est.plot_some_cdfs(n_curves=10, layout="panels", save=f"cdf_pannels/CDF_pannels{readable_date}.png")
    est.plot_cdf_surface(save=f"CDF_Surface/CDF_surface{readable_date}.png")
    est.plot_some_rnds(layout="panels",n_curves=10,save=f"RND_pannels/RND_pannels{readable_date}.png") #plots RND pannels

    information_dict[i]=obj

#####################################################Pricing kernel and Physical densitiees#################################
##Pricingkernel downward sloping. For very small sample sizes like in this demo it may be upward sloping in tiny maturities.
pricing_kernel_information={}
for date in unique_dates:
    est = PricingKernelSurfaceEstimator(information_dict, stock_df)
    results = est.fit_pricing_kernel(N=1, anchor_date=date, min_obs_per_maturity=8, verbose=True) #N is the polynomial order, for now recommend only 1.
    readable_date = date
    est.plot_pricing_kernel_surface(title=f"Kernel Surface {readable_date}", save=f"Kernel_surfaces/kernel_{readable_date}.png")
    est.plot_physical_density_surface(title=f"Physical Surface {readable_date}", save=f"Physical_surfaces/p_{readable_date}.png")
    est.plot_panel(title=f"Panels {readable_date}", save=f"Panels/panel_{readable_date}.png", panel_shape=(2,4))
    pricing_kernel_information[date]=results

    