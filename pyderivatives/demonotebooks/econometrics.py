import pickle
from pyderivatives import*

with open("moments_returns.pkl", "rb") as f:
    obj = pickle.load(f)
    
moments_premia_dict=obj["moments_by_asset"]["BTC"]
logreturns=obj["btc_logreturns"]

import pickle
from importlib.resources import files

def load_moments_by_asset():
    """
    Load the packaged demo pickle: moments_by_asset.pkl
    """
    p = files("pyderivatives.demodata") / "moments_by_asset.pkl"
    with p.open("rb") as f:
        return pickle.load(f)

mom = load_moments_by_asset()

assymetric_moments_dict={}

investment_horizon=[7,14,21,28,35,60]

for horizon in investment_horizon:
    assymetric_moments_dict[horizon]= run_asym_quantreg_with_controls(
        r_df=logreturns,            
        var_s=moments_premia_dict[horizon]["phys_vol_ann"],
        skew_s=moments_premia_dict[horizon]["phys_skew"],
        kurt_s=moments_premia_dict[horizon]["phys_kurt"],
        ret_col="ret_1",
        n_controls_lags=2,
        n_ret_lags=2,
        n_mom_lags=2,
        B=2,
        block_len=10)
    
    
result_dict={"7d":assymetric_moments_dict[7],"14d":assymetric_moments_dict[14],"21d":assymetric_moments_dict[21],
     "28d":assymetric_moments_dict[28],"35d":assymetric_moments_dict[35],"60d":assymetric_moments_dict[60]}


fig, axes = plot_qrm_across_quantiles_selectcoef(
    res_by_key=result_dict,
    eq_key="A_var",
    coefs=("ret_pos","ret_neg"),          
    ci=0.95,
    show_ols=False,
    ols_hac_lags=10,
)





fig, axes = plot_qrm_by_quantile_across_frequencies_selectcoef(
     res_by_key=result_dict,
    eq_key="A_var",
    coefs=("ret_pos","ret_neg"),  # any controls you added
    ci=0.95,
    show_ols=False,
)


moments_by_asset=moments_premia_dict=obj["moments_by_asset"]
taus = [0.10, 0.25, 0.50, 0.75, 0.90]
roll_dict = {}
static_dict = {}

feature_list = ["phys_vol_ann", "phys_skew", "phys_kurt",]
investment_horizon=[14]
for horizon in investment_horizon:
    roll_dict[horizon] = {}
    static_dict[horizon] = {}

    for feature in feature_list:
        # ---- panel: only the feature across assets ----
        df = build_feature_panel(moments_by_asset, horizon=horizon, feature=feature)
        dfz = zscore(df).dropna()

        if len(dfz) < 260:
            print(f"Skipping {feature} H={horizon}: only {len(dfz)} rows.")
            continue

        # ---- rolling ----
        roll = rolling_quantile_connectedness(
            dfz, taus=taus, p=1, H=10, window=250, step=30
        )
        roll_dict[horizon][feature] = roll

        # ---- static stats for table + networks ----
        static = {tau: quantile_connectedness(dfz, tau=tau, p=1, H=10) for tau in taus}
        static_dict[horizon][feature] = static




        # =========================
        # (B) TCI figure
        # =========================
        fig_tci, ax = plt.subplots(figsize=(10, 4))
        for tau in roll.tci.columns:
            ax.plot(roll.tci.index, roll.tci[tau], label=f"$\\tau={tau}$")

        ax.legend(ncol=3, fontsize=9)
        ax.set_title("Time-Varying Total Connectedness Index")
        fig_tci.suptitle(f"TCI for {feature} (Horizon = {horizon} days)", fontsize=14)
        fig_tci.tight_layout(rect=[0, 0, 1, 0.92])


        # =========================
        # (C) Network panels
        # =========================
        fig_net, axes = plt.subplots(1, len(taus), figsize=(22, 8))
        if len(taus) == 1:
            axes = [axes]

        for ax_i, tau in zip(axes, taus):
            st = static[tau]
            Theta = pd.DataFrame(st.Theta, index=dfz.columns, columns=dfz.columns)
            net   = pd.Series(st.NET, index=dfz.columns)

            plot_spillover_network_arc_on_ax(
                ax_i,
                Theta,
                net=net,
                title=f"$\\tau={tau}$",
                top_k=12,
                curvature=0.25
            )

        fig_net.suptitle(f"Connectedness: {feature} (Horizon = {horizon} days)", fontsize=16)
        fig_net.tight_layout(rect=[0, 0, 1, 0.92])


specs = [
    ("Vol (14d)",  moments_premia_dict["BTC"][14]["phys_vol_ann"],  "wtc_var.png"),
    ("Skew (14d)", moments_premia_dict["BTC"][14]["phys_skew"], "wtc_skew.png"),
    ("Kurtosis (14d)", moments_premia_dict["BTC"][14]["phys_kurt"], "wtc_kurt.png"),
]

# --- 1) Generate each WTC plot and save as image ---
for title, ydata, fname in specs:
    fig_i, ax_i = plot_wtc(
        logreturns, ydata,
        x_name="ret_1",
        y_name=title,
        value_col_x="ret_1",
        transform="none",
        detrend=True,
        sig=True,
        sig_method="mc-phase",
        mc_B=100,
        mc_alpha=0.95,
        period_max=128,
    )



