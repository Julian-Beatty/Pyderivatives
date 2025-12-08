from New_Simulation_density import*


###### Step 1: Pick your Heston model. The monte carlo simulation will randomly draw parameters from your interval. Initialize the class
heston_dict = {"kappa":(0.6,0.6),"theta":(0.3,0.3),"sigma":(0.2,0.20),"v0":(0.02,0.02),"rho":(-0.9,0.9)}
market_dict={"strikes":np.linspace(25,250,100),"underlying_price":120,"risk_free_rate":0.03,"maturity":0.5}
rng = random.Random(123)
hp = pick_model_param(heston_dict, rng)
print(hp)




###Step 2: Pick models. Dictionaries must contain the method, nickname, and neccessary parameters. 
##We demonstrate Evolutionary Mixtures of Lognormal and Wiebul (Yifan Li 2024) and piecewwise splines by Bliss and Panigirtzoglou 2004
##To see optimal theoretical performance, set oracle to True, otherwise leave false
##For exact arguments, check the program files
models_dict={}
models_dict["evolutionary_mixtures"]={"method":"mixture","nickname":"Evolutionary Mixture Yifan (2024)","n_starts":3,"oracle":True,"M_max":3,"variance_constraint":0.1}
models_dict["splines"]={"method":"splines","nickname":"Cubic Spline Bliss (2004)","oracle":True,"degree":3}


##step 3: Initialize class, and set monte carlo parameters
monte_example=monte_carlo(market_dict)

###### Step 4: Pick your Noise model. 
##Noise process based on maximum bid spread allowed by exchange following Bondarenko (2002). Increasing noise as you go OTM.   
noise_dict_1={'mode': 'bondarenko', 'scale': 0.1}



##Noise process based on maximum bid spread allowed by exchange following Bondarenko (2002). Increasing noise as you go OTM.   
monte_carlo_runs=5

##Pick your wave perturbation setting which will add further skewness/kurtosis.
wave_dict_str="uni" ## or "none", "multi", "both"
##step 5: Run simulation. Returns dataframe of Kolmogorov P values, MSE values, and lists of each these values.
df_koml_values,list_kolm_pvalues,df_mse_values,list_mse_values=monte_example.simulate_mc(market_dict,models_dict,monte_carlo_runs,noise_dict_1,wave_dict_str,heston_dict)


