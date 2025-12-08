# Option Toolbox for Python

## Overview
This repository provides an easy to use toolbox for option pricing with an emphasis on the easy implementation of state of the art models from journals in finance and mathematics.

## Option Pricers
This release implements a range of option price estimators.
(1) Parametric risk-neutral density estimation via finite lognormal-Weibull mixtures (Journal of Econometrics: Yifan Li 2024)
(a) Fitting of a mixture of lognormal and Weibull distributions
(b) Model selection via an evolutionary algorithmn

(2) Nonparametric option pricing under shape restrictions (Journal of Econometrics: Yacine Aı̈t-Sahalia 2003)
(a) Locally Linear Kernel Regression of the call function
(b) With expanded bandwidth selectors; Scotts Rule, Cross Validation via mixture of 2 Lognormal (New)
(c) Kernel Density Estimators for post processing

(3) Quartic Polynomial fitting of the IV, (Federal Reserve Federal Reserve Conference Conference: Figlewski 2010)
(a) Fits a polynomial through the IV with a knot at the money
(b) GEV extrapolation coming soon

(4) Option-implied risk aversion estimates (Journal of Finance: Bliss R.R., Panigirtzoglou N 2004)
(a) Fits a cubic spline through the IV curve

(5) Estimation of risk-neutral densities using positive convolution approximation (Journal of Econometrics: Bondarenko 2003)
(a) Positive Convolutions of the Gaussian kernel to fit the call price function.

(6) Sieve estimation of option-implied state price density (Journal of Econometrics: Lu and Qu 2021)
(a) fits to call options using sieve estimation with Hermite basis

(7) Approximate option valuation for arbitrary stochastic processes (Journal of Financial Economics: Jarrow R., Rudd A 1982)
(a) Fits to call prices via 4th order Hermite expansion

(8) Risk-Neutral Density Recovery via Spectral Analysis (SIAM Journal on Financial Mathematic: Jean-Baptiste Monnier 2013)
(a) Solves for the RND using an inverse program approach

(9) Estimating option implied risk-neutral densities using spline and hypergeometric functions (Econometrics Journal: RUIJUN BU 2007)
(a) Fits option prices using hypergeometric functions


# Sample Plots

## Evolutionary Mixtures vs Natural Cubic Splines
Kernel Ridge Regression is used to estimate the implied volatility (IV) surface.
<img width="720" alt="Kernel Ridge Regression" src="Images/GLD KRR.png" />

## Local Linear Regression
Locally linear regression provides a nonparametric fit to the IV surface.
<img width="720" alt="Local Linear Regression" src="Images/Local Linear GLD.png" />

## Quadratic Polynomial Fit
A quadratic polynomial is used to approximate the shape of the IV surface.
<img width="720" alt="Quadratic Polynomial Fit" src="Images/Quadratic GLD.png" />


