from pyderivatives import*

#################################Implementing Quantile Regressions of BTC moments/premiumns onto returns+controls and plots ##########################


##########import data
import pickle
from pyderivatives import *

with open("BTC_kernel_dict.pkl", "rb") as f:
    BTC = pickle.load(f)
with open("BTC_information_dict_merged.pkl", "rb") as f:
    BTCinfo = pickle.load(f)
    
with open("spy_information_dict_merged.pkl", "rb") as f:
    SPYinfo = pickle.load(f)
    
    
BTCresult=BTCinfo["result_dict"] ##containts RND moments
stock_df=BTCinfo["stock_df"]
spy_stock=SPYinfo["stock_df"]

####Extract Time series
btc_res = extract_moment_premia_timeseries(
    physical_dict=BTC,
    rnd_dict=BTCresult,
    moments=("var", "skew", "kurt"),
    T_days=[7,14,21,30,60],
)

logreturns_btc=compute_horizon_returns_backward(
    stock_df,
    horizon=1,
    group_col=None
    
)

logreturns_spy = compute_horizon_returns_backward(
    spy_stock,
    horizon=1,
    group_col=None
    
)
######Plot formula for Wave coherence



fig, ax = plot_wtc(
    logreturns_btc, btc_res[7]["phys_var"],
    x_name="Ret",
    y_name="var~7days",
    value_col_x="ret_1",   # <-- THIS
    transform="none",
    detrend=True,
    sig=True,
    sig_method="mc-phase",
    mc_B=250,
    mc_alpha=0.95,
    period_max=128,
)





spyr = compute_horizon_returns_backward(
    spy_stock,
    horizon=1,
    group_col=None
    
)
spyr_ser = (
    spyr.assign(date=pd.to_datetime(r30["date"]))
       .set_index("date")["ret_1"]   # <-- change "ret" to your returns column name
       .sort_index()
)
spyr_df = spyr_ser.reset_index()   # makes 'date' a column
spyr_df.columns = ["date", "ret_1"]  # ensure correct names





############Hypothesis 1 BTC returns vs BTC Momments QR

qr_btc_mom_7d = run_asym_moment_quantile_regressions(
    r_df=logreturns_btc,
    var_s=btc_res[7]["phys_var"] if isinstance(btc_res[7]["phys_var"], pd.DataFrame) else btc_res[7]["phys_var"],
    skew_s=btc_res[7]["phys_skew"] if isinstance(btc_res[7]["phys_skew"], pd.DataFrame) else btc_res[7]["phys_skew"],
    kurt_s=btc_res[7]["phys_kurt"] if isinstance(btc_res[7]["phys_kurt"], pd.DataFrame) else btc_res[7]["phys_kurt"],
    ret_col="ret_1",
    bootstrap="block",
    block_len=10,
    B=100,
)


qr_btc_mom_14d = run_asym_moment_quantile_regressions(
    r_df=logreturns_btc,
    var_s=btc_res[14]["phys_var"] if isinstance(btc_res[14]["phys_var"], pd.DataFrame) else btc_res[14]["phys_var"],
    skew_s=btc_res[14]["phys_skew"] if isinstance(btc_res[14]["phys_skew"], pd.DataFrame) else btc_res[14]["phys_skew"],
    kurt_s=btc_res[14]["phys_kurt"] if isinstance(btc_res[14]["phys_kurt"], pd.DataFrame) else btc_res[14]["phys_kurt"],
    ret_col="ret_1",
    bootstrap="block",
    block_len=10,
    B=100,
)

qr_btc_mom_21d = run_asym_moment_quantile_regressions(
    r_df=logreturns_btc,
    var_s=btc_res[21]["phys_var"] if isinstance(btc_res[21]["phys_var"], pd.DataFrame) else btc_res[21]["phys_var"],
    skew_s=btc_res[21]["phys_skew"] if isinstance(btc_res[21]["phys_skew"], pd.DataFrame) else btc_res[21]["phys_skew"],
    kurt_s=btc_res[21]["phys_kurt"] if isinstance(btc_res[21]["phys_kurt"], pd.DataFrame) else btc_res[21]["phys_kurt"],
    ret_col="ret_1",
    bootstrap="block",
    block_len=10,
    B=100,
)




btc_qr_mom_dict={"7day":qr_btc_mom_7d,"14day":qr_btc_mom_14d,"21day":qr_btc_mom_21d}

fig, axes = plot_qrm_across_quantiles(
    btc_qr_mom_dict,
    eq_key="A_var",                 # or "B_skew", "C_kurt"
    coef_pos="ret_pos",
    coef_neg="ret_neg",
    ci=0.95,
    scale=100.0,                    # if you want “Responses (%)”
    title="ΔVar: responses across quantiles (positive vs negative returns)",
)


plot_qrm_by_quantile_across_frequencies(btc_qr_mom_dict,eq_key="A_var")
plot_qrm_by_quantile_across_frequencies(btc_qr_mom_dict,eq_key="B_skew")
plot_qrm_by_quantile_across_frequencies(btc_qr_mom_dict,eq_key="C_kurt")


plots = plot_asym_moment_quantile_coeffs(
    btc_qr_mom_dict["7day"],
    coef=["ret_pos", "ret_neg"],
    ci=0.95,
    show_ols=True,
)



####################TVP QSVAR

import pandas as pd

gld = pd.read_csv(
    r"C:\Users\beatt\Spyder directory\State Price Density\Optiondata\GLD_stock.csv"
)
spy = pd.read_csv(
    r"C:\Users\beatt\Spyder directory\State Price Density\Optiondata\SPY_stock.csv"
)
btc = pd.read_csv(
    r"C:\Users\beatt\Spyder directory\State Price Density\BTC_data_new.csv"
)
xle = pd.read_csv(
    r"C:\Users\beatt\Spyder directory\State Price Density\Optiondata\XLE_stock.csv"
)

gld_c = clean_price_df(gld, "GLD")
spy_c = clean_price_df(spy, "SPY")
xle_c = clean_price_df(xle, "XLE")
btc_c = clean_price_df(btc, "BTC")

prices = (
    gld_c
    .merge(spy_c, on="date", how="inner")
    .merge(xle_c, on="date", how="inner")
    .sort_values("date")
    .set_index("date")
)




returns = np.log(prices).diff().dropna()
# make sure both are indexed by datetime


# inner join on index (keeps only common dates)
df = returns.join(var_s, how="inner")

# print(df.head())
# def connectedness_table(stats, asset_names):
#     Theta = pd.DataFrame(
#         stats.Theta * 100,
#         index=asset_names,
#         columns=asset_names
#     )

#     from_others = Theta.sum(axis=1) - np.diag(Theta)
#     to_others   = Theta.sum(axis=0) - np.diag(Theta)
#     net         = to_others - from_others

#     table = Theta.copy()
#     table["From others"] = from_others

#     summary = pd.DataFrame(
#         [to_others, to_others + np.diag(Theta), net],
#         index=["TO", "Inc.Own", "Net"],
#         columns=asset_names
#     )

#     return pd.concat([table, summary])


stats_05 = quantile_connectedness(df, tau=0.05, p=1, H=10)
stats_50 = quantile_connectedness(df, tau=0.50, p=1, H=10)
stats_95 = quantile_connectedness(df, tau=0.95, p=1, H=10)
tab_50 = connectedness_table(stats_50, df.columns)
latex_str = connectedness_table_latex(
    stats_by_tau={0.50: stats_50, 0.05: stats_05, 0.95: stats_95},
    names=df.columns,   # IMPORTANT: must match estimation input
    caption="Static return connectedness in full sample.",
    label="tab:static_connectedness",
    decimals=2,
)

print("TCI (0.05):", stats_05.TCI)
print("NET spillovers (0.05):")
print(pd.Series(stats_05.NET, index=df.columns))
roll = rolling_quantile_connectedness(
    df,
    taus=[0.05, 0.50, 0.95],
    p=1,
    H=10,
    window=250,
    step=1
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
for tau in roll.tci.columns:
    plt.plot(roll.tci.index, roll.tci[tau], label=f"τ={tau}")
plt.legend()
plt.title("Time-Varying Total Connectedness Index (TCI)")
plt.tight_layout()
plt.show()



tau = 0.05

plt.figure(figsize=(10,4))
for asset in df.columns:
    plt.plot(roll.net[tau].index, roll.net[tau][asset], label=asset)

plt.axhline(0, color="black", linewidth=1)
plt.legend()
plt.title(f"NET Spillovers (τ = {tau})")
plt.tight_layout()
plt.show()

tau = 0.95

plt.figure(figsize=(10,4))
for asset in df.columns:
    plt.plot(roll.net[tau].index, roll.net[tau][asset], label=asset)

plt.axhline(0, color="black", linewidth=1)
plt.legend()
plt.title(f"NET Spillovers (τ = {tau})")
plt.tight_layout()
plt.show()

import pandas as pd

taus = [0.05, 0.5, 0.95]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, tau in zip(axes, taus):
    st = quantile_connectedness(df, tau=tau, p=1, H=10)   # <-- df not returns
    Theta = pd.DataFrame(st.Theta, index=df.columns, columns=df.columns)
    net   = pd.Series(st.NET, index=df.columns)

    plot_spillover_network_arc_on_ax(
        ax,
        Theta,
        net=net,
        title=f"τ = {tau}",
        top_k=12,        # with VAR added, more edges exist
        curvature=0.30   # arcs look nicer with more nodes
    )

plt.tight_layout()
plt.show()





