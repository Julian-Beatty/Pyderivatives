import pickle
from option_surface_estimator import*
from pricing_kernel_schreindorfer_2025 import*
from Option_market_helper import*

############################ Demo Description #################################
# PyDerivatives 3.0 Demo — Brazilian ETF (Options → Densities → Pricing Kernel → RRA)
#
# ----
# Demonstrate an end-to-end “state-of-the-art” pipeline that takes call option data on a
# Brazilian ETF and produces:
#   (1) Smooth Call Price Surfaces
#   (2) Risk-Neutral Density (RND) Surfaces
#   (3) Physical (Real-World) Density Surfaces
#   (4) Pricing Kernel Surfaces
#   (5) Relative Risk Aversion (RRA) Surfaces
#   (6) Averaged RRA (summary measure across horizons / dates)
#
# Required files in the working directory
# --------------------------------------
# 1) brazil_etf_demo_mini.pkl
#    - Contains: option data, stock/spot data, and VIX (or volatility proxy) data.
#
# 2) brazil_information_dict_demo_part_1.pkl  (and parts 2,3,4,5)
#    - Contains: outputs from calibration of the Heston–Kou (HK) model
#      (e.g., per-date fitted parameters, fitted/cleaned call surfaces, diagnostics).
#
# Deliverables produced by this demo
# ----------------------------------
# - Call Surfaces:
#     Observed vs fitted call prices across strikes and maturities for each date.
#
# - RND Surfaces:
#     Risk-neutral density surface q(·) implied from the fitted call surface.
#
# - Physical Density Surfaces:
#     Real-world density surface p(·) obtained by adjusting q(·) using the estimated pricing kernel.
#
# - Pricing Kernel Surfaces:
#     M(·) estimated across return states and maturities, by date (and possibly averaged).
#
# - Relative Risk Aversion (RRA) Surfaces:
#     Local risk aversion extracted from the slope/shape of ln M with respect to ln R.
#
# - Averaged Relative Risk Aversion:
#     A maturity-averaged (and/or state-averaged) RRA metric for each date, suitable for
#     time-series plotting and comparison.
############################################################################### Program starts here########################################



###Helper function

######################################################  Skipping this part, just load from pickle
# with open("yields_dict.pkl", "rb") as file:
#     loaded_dict = pickle.load(file)
    
    
# spot=loaded_dict["spot"]  
# forward=loaded_dict["forward"] 

# option_data_prefix="Brazil_ETF_Options.csv" ##I use ishares MSCI Brazil Fund, which is a brazilan ETF 
# stock_price_data_prefix='Brazil_stock.csv'


# option_market=OptionMarket(option_data_prefix,stock_price_data_prefix,spot,forward) #Initialization
# option_df=option_market.original_market_df #Raw dataframe
# option_market.set_master_dataframe(volume=-1, maturity=[1,90],moneyness=[-0.5,0.5],rolling=False,date=["2021-01-02","2022-01-03"]) #Filter options Only going to fit to first 90 days of options
# option_df=option_market.master_dataframe 
# stock_df=option_market.stock_df


##################Contains Option Chain for Brazi ETF EWZ, historical stock prices for EWZ, and historical VIX
name="brazil_etf_demo_mini.pkl" #contains only few surfaces
#name="brazil_etf_demo_large.pkl"
with open(name, "rb") as file:
    brazil_etf_EWZ_demo = pickle.load(file)

option_df=brazil_etf_EWZ_demo["option_df"]
stock_df=brazil_etf_EWZ_demo["stock_df"]
vix=brazil_etf_EWZ_demo["vix_df"]

######################################Preparing for Model Calibration


information_dict={} 
unique_dates = option_df["date"].unique() #This may take a while, feel free to reduce the dates even further.
print(unique_dates)


cols = ["date", "exdate", "rounded_maturity", "stock_price",
        "risk_free_rate", "strike", "mid_price"] ##Keep only absolutely neccerssary columns.

##I store each option surface in its own dictionary, indexed by date. 
call_surface_slice_dict = {
    d: g[cols].sort_values(["rounded_maturity", "strike"]).reset_index(drop=True)
    for d, g in option_df.groupby("date", sort=False)
}

#####################################Fit Heston_kou Model to each option cross section using a for loop###################
##May take a while to fit many years worth.
information_dict={} 
print(unique_dates)
for i in unique_dates: ##Loop through call_surface dictionary, and fit Kou_heston
    call_surface=call_surface_slice_dict[i]
    strikes, maturities, C_mkt, S0, r = extract_call_surface_from_df(call_surface) ###This extracts strikes/maturities ect. in N, form.
    
    cfg = SurfaceConfig(
        row_model=RowModelConfig(
            model="kou_heston", ##You may use other models
            hkde_quad_N=96,
            hkde_quad_u_max=200.0,


        ),
        fixed_moneyness=np.linspace(0.25, 1.75, 250), ###I evaluate the model on a grid from 0.15 moneyness to 1.75 which should be more than enough
        fixed_maturity=np.arange(7, 61)/365, ##I evaluate the model from a maturity of 7 days to 90 at 1 day interval.
        apply_safety_clip=True, ###Clips densities. If issue set to false
        safety_clip_center="spot",
        safety_clip_jump_factor=0.5*np.e, #these will clip the density to zero towards extreme ends for stability. If the densities still osscilate increase this number to np.e or higher. If the densities are collapsing to zero everywhere, raise this number or turn off
        right_safety_clip_jump_factor=0.5*np.e,
        apply_right_safety_clip=True, 

    )
    est = CallSurfaceEstimator(
            strikes=strikes,
            maturities=maturities,
            S0=S0,
            r=r,
            config=cfg,
        )
    obj = est.fit_surface(C_mkt)
    
    ##############################Now I plot. These will save as folders in your directory. Check to pannels of RND to make sure everything looks clean
    readable_date=i.replace("-", "_")
    
    est.plot_rnd_surface(title=f"Risk Neutral Density Surface (Multi-Curve) {i}",save=f"RND_surface_folder/RND_Surface_{readable_date}.png") #plots RND multi-curve surface
    plot_random_observed_vs_model_curve(est=est,
        strikes_orig=strikes,
        maturities_orig=maturities,
        C_orig=C_mkt,
        plot_all_original=True,
        title_prefix=f"Observed vs Fitted Curves {i}",
        save=f"Call_pannels/Call_pannels_{readable_date}.png"
    )
    ##plot observed vs estimated call surface
    plot_original_vs_final_surface(
        est,
        strikes_orig=strikes,
        maturities_orig=maturities,
        C_orig=C_mkt,
        title=f"Surface: Observed vs Final Estimated Call Surface {i}",
        save=f"Call_surfaces/Call_surface_{readable_date}.png"

    )
    
    est.plot_some_cdfs(n_curves=10, layout="panels", save=f"cdf_pannels/CDF_pannels{readable_date}.png") #plots select CDFS check integration to 1
    est.plot_cdf_surface(save=f"CDF_Surface/CDF_surface{readable_date}.png") #plots CDF check full integration to 1
    est.plot_some_rnds(layout="panels",n_curves=10,save=f"RND_pannels/RND_pannels{readable_date}.png") #plots RND pannels check to make sure no weird behavior

    information_dict[i]=obj ##contains all calculated information indexed by day



###########################Pricing Kernel Required data##################
#You need large amount of data for this part, than what I can store in github. Preloaded condensed.
information_dict = load_and_combine_pickles_cwd(
filename_prefix="brazil_information_dict_demo_part_")
########################Pricing Kernels
####Implements Pricing Kernel according to: Conditional risk and the pricing kernel (2025) Journal of Financial Econometrics
#Jointly estimates the pricing kernel and the physical density, by assuming M(R) follows a flexible parametric form.
# vix_df=pd.read_csv("VIXCLS.csv")
est = PricingKernelSurfaceEstimator(information_dict, stock_df)
pricing_kernel_information={} ##Will store information here

#######################Estimation of the parameter vector Theta(T), I estimate the parameter vector for each day between your largest Maturity option and smallest maturity option
##Interpolated on one day. For each Maturity T
##1) Search each dictionary for the Risk neutral density of maturity T Q(T). If that exact maturity isn't found, it will fallback and use for next closest density, up to a tolerance level.
##If no suitable density within that tolerance exists, it skips that dictionary and moves to the next day.
##(2) After collecting densities, it runs basic numerical checks to densities, and throws out them out. If the Density qouted at day t, of Maturity T does not having a matching realized return
## Rt+T that can be calculated in stock_df, it is thrown out.
##(2.1) If not enough observations are available for LL it just uses initial parameters. Bad estimation, may change this later to fall back to later Theta(T)
##(3)For the estimate of conditional volatility, I use either the VIX, or the median ATM-IV.
##(4) Estimates the likelihood according to paper 

theta_cache = est.fit_theta_master_grid(
    N=2, #2nd order polynomial as recommended by paper Produces U shaped P-kernels
    Ksig=1, #allows P-kernel to vary with vol
    tol_days=5,
    day_step=1,
    min_obs_per_T=12,
    r_grid_size=800,
    use_vix=False,
    multistart=False,
    n_random_starts=0,
    theta_bound=None,  # no bounds on c
    verbose=True,
)

for date in unique_dates: #Loops through dates and calculates P(T) ect..
    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")  # key format

    test_date=date
    out = est.evaluate_surfaces_for_date_master(date_str, theta_cache, tol_days_eval=4, warn=True)

    est.plot_pricing_kernel_surface(title=f"Pricing Kernel Surface Brazilian ETF {date_str}",R_bounds=(0.9,1.1),T_bounds=(10/365,40/365))
    est.plot_physical_density_surface(title=f"Physical Density Surface Brazilian ETF {date_str}",save=f"Physical_surfaces/p_{date_str}.png")
    est.plot_panel(title=f"P Density Panels {date_str}", save=f"physical_Panels/physical_panel_{date_str}.png", panel_shape=(2,4),truncate=True, trunc_mode="rbounds", r_bounds=(0.8, 1.15)) #can adjust alpha to see more plot
    
    pricing_kernel_information[date_str]=out
 


###Estimation of pricing kernels/Physical Density using Theta(T). Plugs Theta(T) to calculate the Pricing Kernel M(R) at maturity T. Then Uses M(R) to find P(T) Eq(2) in their paper.
#If for whatever reason you try to estimate M(R,T) and you don't have Theta(T) it falls back to the nearest Theta(T) in theta cache.
#Stores Pricing Kernel Surface, and physical Densities and other diagnostics in dictionary


############################Helper funcitons: Pricing kernels->>Relative Risk Aversion curve, Coefficient of Risk, at each Maturity. Continue at bottom###

# ============================================================
# Deliverables
##RRA Dict contains three versions of "risk aversion"

##(1) Bliss 2003 assumes either the exponential or power utility function (Table 2 in their paper). However in our derivation of the pricing kernel we did not makee any
#asusmption on the utility function. I can solve for utility parameter estimates, but its somewhat contradictory. Remember they assume basically a monotonically decreasing pricing kernel
#but, but we end up with a U shaped pricing kernel which is generally seen in other markets. See U shaped pricing kernels
##(2) If we want to stay in a non-parametric world, and not assume a utility function, another way is to just "average" the the RRA curve for a particular
#tenure over some relevant states. I set a reasonable window to be from R=0.95 to 1.05. You can play around with this window, or add weights.
#Hard to say what this "average" means, but you can say that if the Kernel is symmetric about R=1, and the window is also centered around 1, then the average measures the *tilt* of the pricing kernel basically
# ============================================================
###Stores the RRA(T,R) surface for each day. Computes the three versions of gamma for each Maturity T. Gamma_exponential and gamma_power are probably closest in spirit to bliss 2003.
#Plots the RRA surface each day. Tends to change very very little.
#Plots the weighted Gamma average each day across Maturity. Tends to change very very little.
rra_dict = compute_rra_for_all_dates(
    pricing_kernel_information,
    out_dir="rra_outputs",
    make_plots=True, #can set to true/false just RRA curves
    window=(0.85, 1.15),
    use_p_weights_if_available=True, #
    winsor_quantiles=None, #optional, can cap extreme outliers
)


# Plot pannel curves of the RRA, which are positive and decreasing which is consistent with the U shape pricing Kernel that many research papers find. See Jacobs K.
fig, axes = plot_rra_panel_from_dict(
    rra_dict,
    date="2021-01-14",
    n_curves=6,
    which_T="even",
    R_bounds=(0.75, 1.25),   # optional
    ncols=2,
    save=None
)
plt.show()

# # Existing gamma(T) (p-weighted RRA avg)
##We can plot the time series of weighted average Gamma(T), across each day for select maturities
plot_param_timeseries(
    rra_dict,
    param_key="gamma_by_T",
    T_targets_days=[7,20,30,60],
    tol_days=4,
    out_dir="rra_outputs",
    fname_prefix="ts",
    ylabel="gamma (avg RRA)",
)



