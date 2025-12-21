# **Option Toolbox for Python**

## **Overview**
Pyderivatives is an easy-to-use toolbox for option pricing, with a strong emphasis towards implementation of state-of-the-art models from leading journals in finance, applied mathematics and econometrics. 

## **Risk Neutral Density Surface Estimation**
<img width="720" alt="Kernel Ridge Regression" src="Images/RND_Surface_2021_01_04.png" />

---
## **Physical (True) Density Surface Estimation**
<img width="720" alt="Kernel Ridge Regression" src="Images/Physical_surface_2021_01_04.png" />

---
## **Pricing Kernel Surface Estimation**
<img width="720" alt="Kernel Ridge Regression" src="Images/Pricing_kernel_surface_2021_01_05.png" />

---
## **Physical vs Risk Neutral Density vs Pricing Kernel**
<img width="720" alt="Kernel Ridge Regression" src="Images/Physical_pannels_2021_01_04.png" />

---
## **Full Call Surface Estimation**
<img width="720" alt="Kernel Ridge Regression" src="Images/Call_surface_2021_01_04.png" />

---
## **Implemented Pricing Kernels/Physical Density Estimation**

### **1. Conditional risk and the pricing kernel**  
**Source:** *Journal of Financial Economics, 2025.David Schreindorfer et al*  
- **(a)** Flexible parametric pricing kernel  
 
## **Implemented Parametric models Pricers**

### **1. CALIBRATION AND OPTION PRICING WITH STOCHASTIC VOLATILITY AND DOUBLE EXPONENTIAL JUMPS**  
**Source:** *Journal of Computational and Applied Mathematics, 2025.GAETANO AGAZZOTTI et al*  
- **(a)** Stochastic Volatility with Double Exponential Jumps  (Heston-Kou)
- **(b)** Complemented by implementations of Heston, Kou, and Bates models

## **Implemented Option Semi-parametric/parametric Pricers**

### **1. Parametric Risk-Neutral Density Estimation**  
**Source:** *Journal of Econometrics, Yifan Li (2024)*  
- **(a)** Mixture of lognormal and Weibull distributions  
- **(b)** Evolutionary-algorithm–based model selection  

---

### **2. Nonparametric Option Pricing Under Shape Restrictions**  
**Source:** *Journal of Econometrics, Aït-Sahalia (2003)*  
- **(a)** Locally linear kernel regression of the call price function  
- **(b)** Expanded bandwidth selection: Scott’s rule and cross-validation using a 2-lognormal mixture (new)  
- **(c)** Kernel density estimator–based post-processing for RND extraction  

---

### **3. Quartic Polynomial Implied Volatility Fitting**  
**Source:** *Federal Reserve Conference, Figlewski (2010)*  
- **(a)** Polynomial fitting of the IV curve with a knot at-the-money  
- **(b)** GEV-based extrapolation (coming soon)

---

### **4. Option-Implied Risk Aversion Estimation**  
**Source:** *Journal of Finance, Bliss & Panigirtzoglou (2004)*  
- **(a)** Cubic spline fitting of the implied volatility curve  

---

### **5. Positive Convolution Approximation for RND Estimation**  
**Source:** *Journal of Econometrics, Bondarenko (2003)*  
- **(a)** Positive convolutions of Gaussian kernels to fit the call price function  

---

### **6. Sieve Estimation of the Option-Implied State Price Density**  
**Source:** *Journal of Econometrics, Lu & Qu (2021)*  
- **(a)** Hermite-basis sieve estimator applied to call prices  

---

### **7. Approximate Option Valuation Using Hermite Expansions**  
**Source:** *Journal of Financial Economics, Jarrow & Rudd (1982)*  
- **(a)** Fourth-order Hermite series approximation to option prices  

---

### **8. Risk-Neutral Density Recovery via Spectral Analysis**  
**Source:** *SIAM Journal on Financial Mathematics, Monnier (2013)*  
- **(a)** Inverse-problem spectral method for computing the RND  

---

### **9. Spline and Hypergeometric Methods for RND Estimation**  
**Source:** *Econometrics Journal, Ruijun Bu (2007)*  
- **(a)** Hypergeometric-function–based fitting of option prices  

---


# Sample Plots

## Evolutionary Mixtures vs Natural Cubic Splines
Evolutionary Mixtures are compared to natural cubic splines
<img width="720" alt="Kernel Ridge Regression" src="Images/Splines_Evolutions.png" />
