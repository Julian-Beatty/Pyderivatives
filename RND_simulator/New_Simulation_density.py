import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Mapping, Sequence
import random
from math import erf
from scipy.stats import ks_2samp
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error
from KDEpy import*
from statsmodels.nonparametric.kernel_regression import KernelReg
import pandas as pd

from Heston_model import*
#from LocalReg import*
from Local_Regression import*
from Figlewski import*
from Pca import*
from Hermite import*
from bliss_splines import*
from sieve import*
from spectral import*
from Mixture_LWD import*
from generalized_lambda_pricer import*
from Generalized_Gamma import*

def _uniform_draw(bounds: Sequence[float], rng: random.Random) -> float:
    """
    Draw uniformly from `bounds`.
      - (lo, hi) or (hi, lo)
      - (v, v) returns v
      - (v,)   returns v
    """
    if len(bounds) == 1:
        return float(bounds[0])
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi < lo:
        lo, hi = hi, lo
    return lo if lo == hi else rng.uniform(lo, hi)

def pick_model_param(model_dict: Mapping[str, Sequence[float]],
                     rng: random.Random = random) -> HestonParams:
    """
    Sample Heston params uniformly from the provided bounds.
    NO Feller check, NO rho clipping.
    """
    return HestonParams(
        kappa=_uniform_draw(model_dict["kappa"], rng),
        theta=_uniform_draw(model_dict["theta"], rng),
        sigma=_uniform_draw(model_dict["sigma"], rng),
        v0=_uniform_draw(model_dict["v0"], rng),
        rho=_uniform_draw(model_dict["rho"], rng),
    )




class monte_carlo:
    def __init__(self,market_dict):
        self.market_dict=market_dict    

    def simulate_mc(self,market_dict,models_dict,runs,noise_dict,wave_dict_str,heston_dict):
        """
        

        Returns
        -------
        None.

        """
        # ---------- 0)
        
        # ---------- 1) simulate svi curve from svi dictionary, obtain iv,calls and rnd ----------
        market_dict=self.market_dict
        k=market_dict["strikes"]
        s=market_dict["underlying_price"]
        r=market_dict["risk_free_rate"]
        t=market_dict["maturity"]
        
        
        
        
        
        ##Iterative monte carlo loop begins here M=
        # ---------- 2) perturb rnd by multiplying it by a moving sin wave. We call this the true rnd. Returns the new underlying price from the mean of this rnd. ----------
        #runs=1000
        p_value_collector=[]
        mse_collector=[]

        for i in range(0,runs):
            #wave_dict=self.wave_dict
            #wave_dict={"a":0.015,"w":np.random.uniform(0.05, 0.10),"c":0.02} #unimodal

            #wave_dict={"a":0.015,"w":np.random.uniform(0.10, 0.20),"c":0.02} #bit of both multimodal
            #wave_dict={"a":0.015,"w":np.random.uniform(0.15, 0.20),"c":0.02} multi modal
            
            
            heston_parameters=pick_model_param(heston_dict)
            heston_calls = heston_call_prices_fast(s, k, r, 0, t, heston_parameters)

            heston_rnd=rnd_from_calls(k,heston_calls,r,t)
            
            if wave_dict_str=="none":
                wave_dict={"a":0.015,"w":np.random.uniform(0.05, 0.20),"c":0.02,"cancel":0} #no wave
                
            if wave_dict_str=="uni":
                wave_dict={"a":0.015,"w":np.random.uniform(0.05, 0.10),"c":0.02} #unimodal
            if wave_dict_str=="multi":
                wave_dict={"a":0.015,"w":np.random.uniform(0.10, 0.15),"c":0.02} #bit of  multimodal
            if wave_dict_str=="both":
                initial_roll=np.random.randint(0, 3)   # returns 0, 1, or 2
                if initial_roll==0:
                    wave_dict={"a":0.015,"w":np.random.uniform(0.05, 0.20),"c":0.02,"cancel":0} #no wave
                if initial_roll==1:
                    wave_dict={"a":0.015,"w":np.random.uniform(0.05, 0.10),"c":0.02} #unimodal
                if initial_roll==2:
                    wave_dict={"a":0.015,"w":np.random.uniform(0.10, 0.15),"c":0.02} #bit of multimodal


                
                


            
            
            true_rnd,true_s=truernd_from_sinwave(k,heston_rnd,r,t,wave_dict)
            true_call=calls_from_rnd(true_rnd, k, s, r, t, normalize=False)    
            

            #plt.plot(k/true_s,true_rnd)
            
            
        # ---------- 3) Add noise to each call option price to mimic microstructure noise ----------
            #noise_dict={'mode': 'rel_atm', 'sigma': 0.03, 'decay': 0.02, 'p': 3}
            
            
            noisy_call = noisy_data_function(k, true_call, true_s, noise_dict)
            noisy_rnd=rnd_from_calls(k,noisy_call,r,t)
            noisy_iv=implied_vol_from_calls(noisy_call, true_s, k, r, t, q=0.0, tol=1e-8, max_iter=100, vol_low=1e-8, vol_high=5.0)
            
            
            # plt.figure(figsize=(6, 5))

            # plt.plot(k, true_rnd, label="Benchmark RND", linewidth=2.5)
            # plt.plot(k, heston_rnd,  label="Heston RND", linewidth=2)
            # plt.plot(k, noisy_rnd, label="Noisy RND", linewidth=1.8, linestyle="--")  # <- dotted line
            # plt.grid(alpha=0.25)
            # plt.legend()


            # plt.ylabel("Density $q(K)$")
            # plt.xlabel("Strike")
            # plt.title("Simulation of RNDs")
            # plt.tight_layout()

            # save_filename=f"simulated"
            # save_dir = os.path.join(os.getcwd(), "comparison_rnd_folder.png")
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, save_filename)
            # plt.savefig(save_path, dpi=300)
            # plt.close()
            #plt.plot(k,noisy_iv)
            # plt.plot(k,noisy_rnd)
            # plt.plot(k,true_rnd)
            #plt.plot(k,noisy_calls)
        

            # kde = NaiveKDE(bw="scott").fit(k,weights=noisy_rnd)
            # kde_pdf=kde.evaluate(k)
            # bw=kde.bw/8
            # kde = NaiveKDE(bw=bw).fit(k,weights=noisy_rnd)
            # kde_pdf=kde.evaluate(k)

            # area=np.trapz(kde_pdf,k)
            # kde_pdf=kde_pdf/area
            # plt.plot(k,noisy_rnd)
            # plt.plot(k,kde_pdf,label="kde")
            # plt.plot(k,true_rnd)
            # plt.legend()

        # ---------- 2) Apply models to each function ----------
        #for M in range (0,10):
            result_dict,mse_dict=compare_rnd_models(true_call, true_rnd, noisy_call,noisy_iv, k, true_s, r, t,i, models_dict,noisy_rnd) #returns the KS statistic for each model in form of {model_nick:ks..}
            p_value_collector.append(result_dict) ##appends into list
            mse_collector.append(mse_dict)

        
        # ---------- 3)Aggregate into dataframe and summarize ----------
        pvalue_df = pd.DataFrame(p_value_collector)
        pvalue_df = pvalue_df.median().to_frame().T
        
        mse_df = pd.DataFrame(mse_collector)
        mse_df = mse_df.median().to_frame().T

        return pvalue_df,p_value_collector,mse_df,mse_collector


#####################Functions


def simulate_svi(k, s, r, t, a, b, rho, m, sigma):
    """
    Compute implied vols from raw SVI given strikes, spot S, and rate R.

    Parameters
    ----------
    K : array-like
        Strikes.
    S : float
        Spot price.
    R : float
        Continuously compounded risk-free rate (annualized).
    T : float
        Time to maturity in years.
    a, b, rho, m, sigma : floats
        Raw SVI parameters with b > 0, sigma > 0, |rho| < 1.

    Returns
    -------
    iv : np.ndarray
        Implied volatility for each strike K.
    """
    
    
    
    
    
    k = np.asarray(k, dtype=float)
    f = s * np.exp(r * t)                         # forward from s and r
    k = np.log(np.maximum(k, 1e-300) / f)         # log-moneyness vs f

    w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))  # total variance
    w = np.maximum(w, 1e-12)                      # guard
    iv = np.sqrt(w / t)                           # implied vol
    return iv

def implied_vol_from_calls(C, S, K, R, T, q=0.0, tol=1e-8, max_iter=100, vol_low=1e-8, vol_high=5.0):
    """
    Invert Black–Scholes to get implied vol(s) from call price(s) via bisection.

    Parameters
    ----------
    C : array-like
        Call prices (same shape as K).
    S : float
        Spot.
    K : array-like
        Strikes.
    R : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity (years).
    q : float
        Continuous dividend yield.
    tol : float
        Absolute tolerance on price error.
    max_iter : int
        Max bisection iterations.
    vol_low, vol_high : float
        Initial volatility bracket (in annualized vol units).

    Returns
    -------
    iv : np.ndarray
        Implied vols; NaN where no solution in the bracket (e.g., price outside BS bounds).
    """
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)
    iv = np.full_like(C, np.nan, dtype=float)

    # No-arbitrage price bounds for a European call (with carry q)
    disc_r = np.exp(-R*T); disc_q = np.exp(-q*T)
    lower = np.maximum(disc_q*S - disc_r*K, 0.0)
    upper = disc_q*S  # BS call is increasing in vol, bounded above by discounted spot

    # Precompute price at bracket ends
    Cl = bs_call_price(S, K, R, T, vol_low, q=q)
    Ch = bs_call_price(S, K, R, T, vol_high, q=q)

    # Valid where C within [Cl, Ch] and [lower, upper]
    valid = (C >= np.maximum(lower, Cl) - 1e-12) & (C <= np.minimum(upper, Ch) + 1e-12)

    # Bisection per point
    a = np.full_like(C, vol_low); fa = Cl - C
    b = np.full_like(C, vol_high); fb = Ch - C

    # Ensure signs differ; otherwise mark invalid
    ok = valid & (fa * fb <= 0)

    # Work arrays (only where ok)
    a = a[ok]; b = b[ok]; K_ok = K[ok]; C_ok = C[ok]
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = bs_call_price(S, K_ok, R, T, m, q=q) - C_ok
        left = fm > 0
        # Update brackets
        b = np.where(left, m, b)
        a = np.where(left, a, m)
        if np.max(np.abs(fm)) < tol:
            break

    iv_ok = 0.5 * (a + b)
    iv[ok] = iv_ok
    return iv
def bs_call_price(S, K, R, T, vol, q=0.0):
    S = float(S); K = np.asarray(K, dtype=float); vol = np.asarray(vol, dtype=float)
    eps = 1e-12
    disc_r = np.exp(-R*T)
    disc_q = np.exp(-q*T)
    volT = np.maximum(vol*np.sqrt(max(T, eps)), eps)
    d1 = (np.log(np.maximum(S, eps)/np.maximum(K, eps)) + (R - q + 0.5*vol**2)*T) / volT
    d2 = d1 - volT
    return disc_q*S*_norm_cdf(d1) - disc_r*K*_norm_cdf(d2)

def _norm_cdf(x):
    """Standard normal CDF without SciPy, vectorized via math.erf."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))

def rnd_from_calls(K, C, R, T, clip_zero=True):
    """
    Risk-neutral density f_Q(K) via Breeden–Litzenberger:
        f_Q(K) = exp(R*T) * d^2 C / dK^2

    Parameters
    ----------
    K : array-like
        Strikes (monotone increasing recommended).
    C : array-like
        Call prices corresponding to K.
    R : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    clip_zero : bool
        If True, clip tiny negative densities to 0 (from numerical noise).

    Returns
    -------
    f : np.ndarray
        Risk-neutral density evaluated on K.
    """
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)

    dC_dK   = np.gradient(C, K, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, K, edge_order=2)

    f = np.exp(R * T) * d2C_dK2
    if clip_zero:
        f = np.maximum(f, 0.0)
    return f

def truernd_from_sinwave(k,svi_rnd,r,t,wave_dict):
    """
    

    Parameters
    ----------
    k : TYPE
        DESCRIPTION.
    svi_rnd : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    wave_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    
    ## f(k)=a*sin(w*k)**2 + c
    a=wave_dict["a"]
    w=wave_dict["w"]
    c=wave_dict["c"]
    p = wave_dict.get('cancel',1)
    
    ## Multiply by moving sin wave, then normalize. Extract the "true" stock price as the mean of this density, discounted to preseent
    moving_sin=a*(np.sin(w*k))**2+c
    perturbed_pdf=svi_rnd*(moving_sin)**p
    area=np.trapz(perturbed_pdf,k)
    true_rnd=perturbed_pdf/area
    true_s=np.trapz(true_rnd*k,k)*np.exp(-r*t)
    
    
    #
    # plt.plot(k,true_rnd)
    # plt.plot(k,svi_rnd)
    
    
    
    
    return true_rnd,true_s

def calls_from_rnd(rnd, K, S, R, T, normalize=False):
    """
    Recover Black–Scholes/Merton-style call prices from a risk-neutral PDF on S_T.

    Formula:
        C(K) = exp(-R*T) * ∫_{K}^{∞} (x - K) f_Q(x) dx
             = exp(-R*T) * [  ∫_{K}^{∞} x f_Q(x) dx  -  K ∫_{K}^{∞} f_Q(x) dx ]

    Parameters
    ----------
    rnd : array-like
        Risk-neutral density f_Q evaluated on the same grid as K (i.e., x-grid of S_T).
    K : array-like
        Monotonically increasing grid (interpreted as terminal price grid); prices are returned at these K's.
    S : float
        Spot price today (unused in formula, but kept for interface consistency/sanity).
    R : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity (years).
    normalize : bool (default False)
        If True, renormalizes `rnd` to integrate to 1 on the provided grid.

    Returns
    -------
    C : np.ndarray
        Call price curve evaluated at strikes equal to the grid K.
        (Tail beyond max(K) is assumed negligible.)
    """
    K = np.asarray(K, dtype=float)
    f = np.asarray(rnd, dtype=float)
    n = K.size
    if f.size != n:
        raise ValueError("rnd and K must have the same length.")
    if np.any(np.diff(K) <= 0):
        raise ValueError("K must be strictly increasing.")

    # Optional normalization of the PDF over the finite grid
    dx = np.diff(K)
    area = np.sum(0.5 * (f[:-1] + f[1:]) * dx)
    if normalize and area > 0:
        f = f / area

    # Right-tail cumulative integrals using trapezoids:
    # Tail probability A_i = ∫_{K_i}^{∞} f(x) dx
    seg_area_f = 0.5 * (f[:-1] + f[1:]) * dx
    A = np.zeros(n)
    if n > 1:
        A[:-1] = np.cumsum(seg_area_f[::-1])[::-1]  # right cumulative
        A[-1] = 0.0

    # Tail first moment B_i = ∫_{K_i}^{∞} x f(x) dx
    g = K * f
    seg_area_g = 0.5 * (g[:-1] + g[1:]) * dx
    B = np.zeros(n)
    if n > 1:
        B[:-1] = np.cumsum(seg_area_g[::-1])[::-1]
        B[-1] = 0.0

    # Call prices: C_i = e^{-R T} [ B_i - K_i * A_i ]
    disc = np.exp(-R * T)
    C = disc * (B - K * A)

    # Numerical guard: tiny negatives to zero
    return np.maximum(C, 0.0)

def implied_vol_from_calls(C, S, K, R, T, q=0.0,
                           tol=1e-8, max_iter=100,
                           vol_low=1e-8, vol_high=5.0,
                           max_expand=12):
    """
    Invert Black–Scholes implied volatility from DISCOUNTED call prices.

    Parameters
    ----------
    C : float or array
        Discounted call price(s) (same convention as your bs_call_price).
    S : float
        Spot price (scalar).
    K : float or array
        Strike(s). May be scalar or broadcastable to C.
    R : float
        Risk-free rate (scalar).
    T : float
        Time to maturity in years (scalar).
    q : float, default 0.0
        Dividend/foreign yield (scalar).
    tol : float
        Root-finder tolerance.
    max_iter : int
        Max iterations for Brent.
    vol_low, vol_high : float
        Initial bracketing interval for vol.
    max_expand : int
        How many times to expand the upper bracket if needed.

    Returns
    -------
    iv : float or ndarray
        Implied vol(s). Scalar if both C and K are scalar; otherwise an array
        with broadcasted shape. np.nan where no bracket/solution was found.
    """
    # Broadcast C and K to a common 1-D working shape
    C_arr, K_arr = np.broadcast_arrays(np.asarray(C, float), np.asarray(K, float))
    out_shape = C_arr.shape
    C_flat = C_arr.ravel()
    K_flat = K_arr.ravel()

    disc_r = np.exp(-R*T)
    disc_q = np.exp(-q*T)

    # No-arbitrage bounds for *discounted* call price
    lower = np.maximum(disc_q*S - disc_r*K_flat, 0.0)
    upper = np.full_like(lower, disc_q*S)

    iv = np.full_like(C_flat, np.nan, dtype=float)

    # Root function for a single (K, C) pair
    def _f(sig, k, c):
        return bs_call_price(S, k, R, T, sig, q=q) - c

    for i in range(C_flat.size):
        k_i = K_flat[i]
        # Clip price into [lower, upper] to avoid tiny numerical violations
        c_i = float(np.clip(C_flat[i], lower[i], upper[i]))

        # Evaluate endpoints
        f_lo = _f(vol_low,  k_i, c_i)
        f_hi = _f(vol_high, k_i, c_i)

        # Ensure the bracket straddles the root; expand if necessary
        hi_eff = vol_high
        if f_lo * f_hi > 0:
            hi = vol_high
            ok = False
            for _ in range(max_expand):
                hi *= 2.0
                f_hi = _f(hi, k_i, c_i)
                if f_lo * f_hi <= 0:
                    hi_eff = hi
                    ok = True
                    break
            if not ok:
                # Could not bracket → leave NaN
                continue

        # Brent root find
        try:
            iv[i] = brentq(_f, vol_low, hi_eff, args=(k_i, c_i),
                           xtol=tol, maxiter=max_iter)
        except Exception:
            iv[i] = np.nan

    iv = iv.reshape(out_shape)
    # Return scalar when inputs were scalar
    return float(iv) if iv.shape == () else iv


def implied_vol_from_calls_grid(C, S, K, R, T, q=0.0, **kwargs):
    """
    Convenience wrapper for 2D arrays (e.g., maturities x strikes). Calls
    implied_vol_from_calls elementwise but keeps the 2D shape.

    Parameters
    ----------
    C : array-like (M x N)
        Discounted call prices.
    K : array-like (M x N) or (N,) or scalar
        Strikes broadcastable to C.
    (S, R, T, q) : scalars
        Model inputs (your bs_call_price convention).
    kwargs : passed through to implied_vol_from_calls

    Returns
    -------
    iv : ndarray with same shape as C
    """
    C_arr = np.asarray(C, float)
    iv = implied_vol_from_calls(C_arr, S, K, R, T, q=q, **kwargs)
    return np.asarray(iv).reshape(C_arr.shape)


import numpy as np

def bs_call_price(S, K, R, T, vol, q=0.0):
    S = float(S)
    K = np.asarray(K, dtype=float)
    vol = np.asarray(vol, dtype=float)

    eps    = 1e-12
    disc_r = np.exp(-R*T)
    disc_q = np.exp(-q*T)
    sqrtT  = np.sqrt(max(T, eps))
    volT   = np.maximum(vol*sqrtT, eps)

    d1 = (np.log(np.maximum(S, eps)/np.maximum(K, eps)) + (R - q + 0.5*vol**2)*T) / volT
    d2 = d1 - volT

    return disc_q*S*norm.cdf(d1) - disc_r*K*norm.cdf(d2)
def max_spread_M(q):
    """
    Compute M(q) as defined in the text.

    Knots (q in dollars):
        M(0)  = 1/8
        M(2)  = 1/4
        M(5)  = 3/8
        M(10) = 1/2
        M(20) = 3/4
        M(q)  = 1 for q >= 50
    Linearly interpolated for 0 < q < 50.
    """
    q = np.asarray(q, dtype=float)

    # q grid and corresponding M(q) values
    q_knots = np.array([0.0,  2.0,  5.0, 10.0, 20.0, 50.0])
    M_knots = np.array([1/8, 1/4, 3/8, 1/2, 3/4,  1.0])

    # linear interpolation, with left/right behavior as in the text
    Mq = np.interp(q, q_knots, M_knots,
                   left=M_knots[0],    # q <= 0  -> M(0) = 1/8
                   right=M_knots[-1])  # q >= 50 -> 1
    return Mq
def relevant_spread_si(C, x, S_t, c=1.0):
    """
    Compute s_i = c * min(M(P_i), M(S_t + P_i - x_i)).

    Parameters
    ----------
    C : array_like
        Call prices C_i.
    x : array_like
        Strikes x_i (same shape as P).
    S_t : float
        Underlying spot price at time t.
    c : float, optional
        Scaling constant (e.g. 0.5 or 1.0). Default is 1.0.

    Returns
    -------
    s : ndarray
        The relevant spreads s_i.
    """
    C = np.asarray(C, dtype=float)
    x = np.asarray(x, dtype=float)
    S_t = float(S_t)

    # Synthetic call quote via (simplified) put–call parity
    #C = S_t + P - x
    P=C+x-S_t
    P_clipped = np.maximum(P, 0.001)

    # Maximum spreads for put and (synthetic) call
    M_put  = max_spread_M(P_clipped)
    M_call = max_spread_M(C)

    # Relevant spread: min of the two, scaled by c
    s = c * np.minimum(M_put, M_call)
    return s
def noisy_data_function(k, true_calls, true_s, noise_dict):
    """
    Add noise to theoretical call prices.

    Parameters
    ----------
    k : array_like
        Strike grid K_n.
    true_calls : array_like
        Theoretical call prices C*_n.
    true_s : float
        Underlying spot S.
    noise_dict : dict
        Controls the noise design. Important keys:

        Common:
          - mode : str
              'abs', 'rel_atm', 'static', 'sait', 'bondarenko',
              'yifan_li' / 'yifan_li_basic',
              'yifan_li_I', 'yifan_li_II', 'yifan_li_III'
          - seed : int or None

        For 'abs' / 'rel_atm':
          - sigma : float, base scale (default 0.02)
          - decay : float, moneyness scale (default 0.07)
          - p     : float, decay exponent (default 2.0)

        For 'static':
          - upper : float, upper bound of U[0, upper] (default 0.01)

        For 'sait':
          - scale : float, fraction of option price used to build spread (default 0.05)

        For 'bondarenko':
          - scale : float, constant c in relevant_spread_si (default 0.5)

        Yifan Li designs (price domain or IV domain):
          - gamma_u  : float, scale for price errors (default 0.1715; "medium" case)
          - gamma_iv : float, scale for IV errors (default 0.0985; medium-ish)
          - r, q, T  : floats, needed for IV-based designs
              (risk-free rate, dividend yield, maturity)
    """
    mode  = noise_dict.get('mode', 'abs').lower()
    sigma = float(noise_dict.get('sigma', 0.02))
    decay = float(noise_dict.get('decay', 0.07))
    p     = float(noise_dict.get('p', 2.0))
    seed  = noise_dict.get('seed', None)

    k = np.asarray(k, dtype=float)
    C = np.asarray(true_calls, dtype=float)
    S = float(true_s)

    rng = np.random.default_rng(seed)

    # ---------- 1. Our original "ATM-heavier" Gaussian noise ----------
    if mode in ("abs", "rel_atm"):
        # distance from ATM in moneyness
        d = np.abs(k / S - 1.0)
        # weight: 1 at ATM, decays as distance grows
        w = np.exp(- (d / decay) ** p)

        if mode == "rel_atm":
            atm_idx = np.argmin(np.abs(k / S - 1.0))
            atm_level = max(C[atm_idx], 1e-12)
            scale = sigma * atm_level
        else:
            scale = sigma

        eps = rng.standard_normal(C.shape)
        noisy = C + (scale * w) * eps
        return noisy

    # ---------- 2. Static multiplicative bump ----------
    if mode == "static":
        upper = float(noise_dict.get("upper", 0.01))
        eps = rng.uniform(0.0, upper, size=C.shape)
        noisy = C * (1.0 + eps)
        return noisy

    # ---------- 3. SAIT-style spread noise ----------
    if mode == "sait":
        scale = float(noise_dict.get('scale', 0.05))  # fraction of option value

        min_spread = 0.00036 * S     # 0.036% of underlying
        max_spread = 0.0014  * S     # 0.14% of underlying

        moneyness = k / S
        # more illiquid away from ATM
        liquidity_factor = 1.0 + (2.0 / 0.20) * np.abs(moneyness - 1.0)

        raw_spread = scale * C
        spread = np.clip(raw_spread, min_spread, max_spread)

        high = 0.5 * spread * liquidity_factor
        eps = rng.uniform(low=0.0, high=high)
        noisy = C + eps
        return noisy

    # ---------- 4. Bondarenko-style bid–ask perturbation ----------
    if mode == "bondarenko":
        scale = float(noise_dict.get('scale', 0.5))
        s_half = relevant_spread_si(true_calls, k, true_s, c=scale)
        s = np.asarray(s_half, dtype=float)

        eps = rng.uniform(low=-s, high=s)
        noisy = C + eps
        return noisy

    # ============================================================
    # 5. Yifan Li (2024) – basic homoscedastic price errors
    #    C_n = max(C*_n + u_n, 0),  u_n = gamma_u * Z_n,  Z_n ~ U(-1,1)
    # ============================================================
    if mode in ("yifan_li", "yifan_li_basic"):
        gamma_u = float(noise_dict.get("gamma_u", 0.1715))  # "medium" variance
        Z = rng.uniform(-1.0, 1.0, size=C.shape)
        u = gamma_u * Z
        noisy = np.maximum(C + u, 0.0)
        return noisy

    # Helper to get BS IV and price again (for IV-based designs)
    def _true_iv_from_price(C_star):
        # Needs bs_iv_from_price_disc to be defined elsewhere in your file
        r = float(noise_dict.get("r", 0.0))
        q = float(noise_dict.get("q", 0.0))
        T = float(noise_dict.get("T", 1.0))

        ivs = []
        for Ki, Ci in zip(k, C_star):
            iv = bs_iv_from_price_disc(S, Ki, r, q, T, Ci)
            ivs.append(iv)
        return np.array(ivs), r, q, T

    def _price_from_iv(ivs, r, q, T):
        # Needs bs_call_disc to be defined elsewhere in your file
        prices = []
        for Ki, sig in zip(k, ivs):
            prices.append(bs_call_disc(S, Ki, r, q, T, sig))
        return np.array(prices)

    # ============================================================
    # 6. Alternative error design I (IV i.i.d. errors)
    #    IV*_n -> IV_n = max(IV*_n + u_IV,n, 0),
    #    u_IV,n = gamma_IV * Z_IV,n,  Z_IV,n ~ U(-1,1)
    #    Then C_n = Black(S, K_n, IV_n, ...)
    # ============================================================
    if mode == "yifan_li_i":
        gamma_iv = float(noise_dict.get("gamma_iv", 0.0985))
        iv_star, r, q, T = _true_iv_from_price(C)
        Z_iv = rng.uniform(-1.0, 1.0, size=iv_star.shape)
        u_iv = gamma_iv * Z_iv
        iv_obs = np.maximum(iv_star + u_iv, 0.0)
        noisy = _price_from_iv(iv_obs, r, q, T)
        return noisy

    # ============================================================
    # 7. Alternative error design II (AR(1) price errors)
    #    u_n = gamma_u * Z_n,
    #    Z_n = 2 Φ(η_n) - 1,
    #    η_n = 0.8 η_{n-1} + e_n,  e_n ~ N(0, 0.36)
    # ============================================================
    if mode == "yifan_li_ii":
        gamma_u = float(noise_dict.get("gamma_u", 0.1715))

        N = C.shape[0]
        eta = np.empty(N)
        # start from stationary distribution: var = 0.36 / (1 - 0.8^2) ≈ 1
        eta[0] = rng.normal(0.0, 1.0)
        for n in range(1, N):
            e_n = rng.normal(0.0, np.sqrt(0.36))
            eta[n] = 0.8 * eta[n - 1] + e_n

        Z = 2.0 * norm.cdf(eta) - 1.0
        u = gamma_u * Z
        noisy = np.maximum(C + u, 0.0)
        return noisy

    # ============================================================
    # 8. Alternative error design III (AR(1) IV errors)
    #    u_IV,n = gamma_IV * Z_IV,n,
    #    Z_IV,n defined as AR(1) above, applied in IV space,
    #    then mapped back to prices via Black.
    # ============================================================
    if mode == "yifan_li_iii":
        gamma_iv = float(noise_dict.get("gamma_iv", 0.0985))
        iv_star, r, q, T = _true_iv_from_price(C)

        N = iv_star.shape[0]
        eta = np.empty(N)
        eta[0] = rng.normal(0.0, 1.0)
        for n in range(1, N):
            e_n = rng.normal(0.0, np.sqrt(0.36))
            eta[n] = 0.8 * eta[n - 1] + e_n

        Z_iv = 2.0 * norm.cdf(eta) - 1.0
        u_iv = gamma_iv * Z_iv
        iv_obs = np.maximum(iv_star + u_iv, 0.0)
        noisy = _price_from_iv(iv_obs, r, q, T)
        return noisy

    # ---------- fallback (if unknown mode) ----------
    raise ValueError(f"Unknown noise mode: {mode!r}")




def compute_samples(x, fx, n_samples=50_000, *, clip_negative=True):
    x = np.asarray(x)
    fx = np.asarray(fx)

    if x.ndim != 1 or fx.ndim != 1 or x.size != fx.size:
        raise ValueError("x and fx must be 1-D arrays of the same length.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

    if clip_negative:
        fx = np.clip(fx, 0, None)
    if np.any(~np.isfinite(fx)):
        raise ValueError("fx contains non-finite values.")

    # Normalize PDF
    area = np.trapz(fx, x)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("PDF area must be positive and finite.")
    fx = fx / area

    # Build CDF (length == len(x))
    dx = np.diff(x)
    cdf_inner = np.cumsum((fx[:-1] + fx[1:]) * 0.5 * dx)
    cdf = np.concatenate(([0.0], cdf_inner))
    # Numerical safety: force last point to 1 exactly
    cdf[-1] = 1.0

    # Draw uniforms and invert CDF -> samples
    u = np.random.random(n_samples)  # (0,1)
    samples = np.interp(u, cdf, x)   # piecewise-linear inverse

    return samples

def compare_rnd_models(true_call,true_rnd,noisy_call,noisy_iv,k,true_s,r,t,run_number,models_dict,noisy_rnd):
    """
    

    Parameters
    ----------
    true_call : TYPE
        DESCRIPTION.
    true_rnd : TYPE
        DESCRIPTION.
    noisy_call : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.
    true_s : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    models_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    result_dict={}
    rnd_iv_dict={}
    mse_dict={}

    true_rnd_samples=compute_samples(k,true_rnd)
    
    
    # models_dict={}
    # models_dict["mixture"]={"method":"mixture","nickname":"2W1L","n_lognormal":2,"n_weibull":1,"n_starts":1}
    # models_dict["local_regression"]={"method":"local_regression","nickname":"loc_reg"}
    # models_dict["figlewski"]={"method":"figlewski","nickname":"quartic"}
    # models_dict["pca"]={"method":"pca","nickname":"pca_v1","pad":1.2,"m":23*4}
    # models_dict["local_regression_kde"]={"method":"local_regression_kde","nickname":"loc_regkde","bw_setting":"recoemmended","cv_method":'loo',"kde_method":6}
    # models_dict["hermite"]={"method":"hermite","nickname":"hermite_def"}


    # bw_setting=argument_dict.get("bandwidth_setting","recommended")
    # cv_method = argument_dict.get('cv_method','loo')
    # kde_method = argument_dict.get('kde_method',8)
    # alpha = argument_dict.get('alpha',0.5)

    # nickname=argument_dict.get('nickname',"generic local_polynomial_kde")
    print(models_dict.keys())
    for model_key in models_dict:
        
        if models_dict[model_key]["method"]=="mixture":
            print("Attempting mixtures")
            
            nickname=models_dict[model_key]["nickname"]
            
            #general settings
            n_starts=models_dict[model_key].get("n_starts",1)
            metric=models_dict[model_key].get("oracle","loss")
            oracle=models_dict[model_key].get("oracle",False)

            c=models_dict[model_key].get("variance_constraint",0.1)
            lambda_penalty=models_dict[model_key].get("lambda",0) #penalty for martingale constraint

            #check if fixing the distributions
            N_mixtures=models_dict[model_key].get("N_mixtures",None)
            N_lognormals=models_dict[model_key].get("N_lognormals",None)
            
            #evolutionary settings
            use_wald=models_dict[model_key].get("use_wald",True)
            N_max=models_dict[model_key].get("N_max",3)
            wald_alpha=models_dict[model_key].get("alpha",0.05)
            q=models_dict[model_key].get("q",1)
            p=models_dict[model_key].get("p",1)


            if oracle==True:
                    
                    fit, chosen_spec = evolutionary_lwm_fit(
                        K=k,
                        C_mkt=noisy_call,
                        S0=true_s,
                        r=r,
                        T=t,
                        M_max=N_max,
                        penalty_lambda=lambda_penalty,
                        random_starts=n_starts,
                        seed=42,
                        var_c=c,
                        var_penalty=1e4,
                        improvement_tol=1e-4,
                        metric="density",   # or "loss"
                        rnd_true=true_rnd,
                        k_true=k,
                        use_wald=use_wald,
                        wald_alpha=wald_alpha,
                        wald_p=p,
                        wald_q=q,
                        weights=None,
                        fixed_M=N_mixtures,
                        fixed_M1=N_lognormals,
                        # or an open-interest vector
                    )
                    print("Chosen spec: M1 (lognormals) =", chosen_spec.n_lognormal,
                          ", total M =", chosen_spec.n_lognormal + chosen_spec.n_weibull)
                    
            if oracle!=True:
                    fit, chosen_spec = evolutionary_lwm_fit(
                        K=k,
                        C_mkt=noisy_call,
                        S0=true_s,
                        r=r,
                        T=t,
                        M_max=N_max,
                        penalty_lambda=lambda_penalty,
                        random_starts=n_starts,
                        seed=42,
                        var_c=c,
                        var_penalty=1e4,
                        improvement_tol=1e-4,
                        metric=metric,   # or "loss"
                        use_wald=use_wald,
                        wald_alpha=wald_alpha,
                        wald_p=p,
                        wald_q=q,
                        weights=None,
                        fixed_M =N_mixtures,
                        fixed_M1=N_lognormals,
                        # or an open-interest vector
                    )
                

            
            rnd, cdf_mix = evaluate_rnd(fit, k)
            
            #plt.plot(k,rnd)
            #plt.plot(k,rnd1)

            #plt.scatter(k,true_rnd)
            
            C_fit = np.exp(-r*t) * mixture_call_undisc(k, fit.weights, fit.types, fit.params)
            rnd=rnd_from_calls(k,C_fit,r,t)
            # plt.scatter(k,noisy_call)
            # plt.plot(k,C_fit,color="orange")
            
            pvalue_list=[]
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse

            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_dict={"rnd":rnd}
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit,"strikes":k}
            
        if models_dict[model_key]["method"]=="local_regression":
            
            print("Attempting Local Regression")
            nickname=models_dict[model_key]["nickname"]
            
            ##General settings
            oracle=models_dict[model_key].get("oracle",False)
            bandwidth_method=models_dict[model_key].get("bandwidth_method","ait_sahalia")
            ##--Extract the rnd, take samples and perform the KS test (extract pvalue).
            
            use_kde   = models_dict[model_key].get("use_kde", False)
            kde_scale = models_dict[model_key].get("kde_scale", 1)
            
            
            
            if oracle==True:
                print("oracle used")
                res=fit_ll_rnd_on_calls(k, noisy_call, r, t, bandwidth_method="oracle", q_true=true_rnd, K_true=k,use_kde=use_kde,kde_scale=kde_scale,S0_for_mixture=true_s)
            else:
                print("no oracle")

                res = fit_ll_rnd_on_calls(k, noisy_call, r, t, bandwidth_method=bandwidth_method,use_kde=use_kde,kde_scale=kde_scale,S0_for_mixture=true_s)


            C_fit=res.C_hat_at_data
            rnd=res.q_fit
            if use_kde==True:
                rnd=res.q_kde
                C_fit=res.C_from_kde
                
            pvalue_list=[]
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse
            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit,"strikes":k}
            
            # plt.scatter(k,true_rnd)
            # #plt.plot(k,noisy_rnd)
            # plt.plot(k,true_rnd,color="orange")
            
            
        # if models_dict[model_key]["method"]=="local_regression":
            
        #     print("Attempting Local Regression")
        #     nickname=models_dict[model_key]["nickname"]
            
        #     ##--Extract the rnd, take samples and perform the KS test (extract pvalue).
        #     res = fit_ll_rnd_cv_on_calls(k, noisy_call, r, t, h_grid=None, eval_points=len(k), kfold=2, rng=2)
        #     C_fit=res.C_hat_at_data

            
        #     rnd=res.q_fit
            
        #     pvalue_list=[]
        #     for i in range(0,100):
        #         specific_samples=compute_samples(k,rnd,1000)
        #         D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
        #         pvalue_list.append(pvalue)  # no assignment here
        #     average_pvalue=np.max(pvalue_list)
            
        #     mse=mean_square_error(true_rnd, rnd)
        #     mse_dict[nickname]=mse
            
        #     ##--Store the KS Pvalue into a dictionary, indexed by the nickname
        #     result_dict[nickname]=average_pvalue
        #     ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
        #     rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit,"strikes":k}
            
        #     plt.scatter(k,true_rnd)
        #     #plt.plot(k,noisy_rnd)
        #     plt.plot(k,true_rnd,color="orange")
        if models_dict[model_key]["method"]=="figlewski":
            
            print("Attempting Figlewski")
            nickname=models_dict[model_key]["nickname"]
            
            F = true_s * np.exp((r - 0) * t)
            
            beta, iv_fit, meta = fit_iv_quartic_knot_safe(k, noisy_iv, F)
            # clip tiny negatives if any numerical wiggles
            iv_fit = np.clip(iv_fit, 1e-6, 5.0)
            # plt.plot(k,iv_fit,color="orange")
            # plt.scatter(k,noisy_iv)
            
            # Convert fitted IVs back to discounted call prices
            C_fit = np.array([bs_call_disc(true_s, K, r, 0, t, sig)
                              for K, sig in zip(k, iv_fit)])

            ##--Extract the rnd, take samples and perform the KS test (extract pvalue).
            rnd = rnd_from_calls(k, C_fit, r, t)
            # plt.plot(k,rnd)
            # plt.scatter(k,true_rnd)
            # plt.plot(k,C_fit,color="orange")
            # plt.scatter(k,noisy_call)


            
            
            pvalue_list=[]
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse
            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit,"strikes":k}
            
        if models_dict[model_key]["method"]=="GGD":
                
            print("Attempting Generalized Gamma Density")
            nickname=models_dict[model_key]["nickname"]
            a_bounds=models_dict[model_key].get("a_bounds",None)
            d_bounds=models_dict[model_key].get("d_bounds",(0.2, 25.0))
            p_bounds=models_dict[model_key].get("p_bounds",(0.2, 50.0))
            forward_penalty_lambda=models_dict[model_key].get("forward_penalty_lambda",1e-4)


            
            
            fit = GenGammaRNDModel(
                S0=true_s,
                r=r,
                T=t,
                K=k,
                C_mkt=noisy_call,
                a0=true_s,
                d0=1.5,
                p0=2.0,
                forward_penalty_lambda=1e-4,
                a_bounds=a_bounds,
                d_bounds=d_bounds,
                p_bounds=p_bounds,
                # a_bounds=None -> defaults to (0.2*S0, 5.0*S0)
                # d_bounds and p_bounds already have defaults
            )                # clip tiny negatives if any numerical wiggles
            

            rnd=fit.qhat(k)
            C_fit=fit.chat(k)
            
            pvalue_list=[]
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse
            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit,"strikes":k}
                
            
        if models_dict[model_key]["method"]=="GLD":
            
            print("Attempting Generalized Lambda Density")
            nickname=models_dict[model_key]["nickname"]
            sigma_bounds=models_dict[model_key].get("sigma_bounds",(1e-4, 3.0))
            k3_bounds=models_dict[model_key].get("k3_bounds",(-0.49, 5.0))
            k4_bounds=models_dict[model_key].get("k4_bounds", (-0.49, 5.0))
            forward_penalty_lambda=models_dict[model_key].get("forward_penalty_lambda",1e-4)


            
            fit = GLDRNDModel(
                S0=true_s, r=r, T=t,
                K=k, C_mkt=noisy_call,
                sigma0=0.30, k30=0.05, k40=0.02,
                forward_penalty_lambda=forward_penalty_lambda,
                sigma_bounds=sigma_bounds,
                k3_bounds=k3_bounds,
                k4_bounds=k4_bounds,
                eps_p=1e-10,
                bisect_max_iter=60,
                bisect_tol=1e-12,
                pdf_pgrid_size=20000,
            )

            rnd=fit.qhat(k)
            C_fit=fit.chat(k)
            
            pvalue_list=[]
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse
            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit,"strikes":k}
        
        if models_dict[model_key]["method"]=="pca":
            
            print("Attempting PCA")
            nickname=models_dict[model_key]["nickname"]
            m_candidates=models_dict[model_key].get("m",5)
            pad=models_dict[model_key].get("pad",1.2)
            oracle=models_dict[model_key].get("oracle",False)

            ##Initialize PCA model
            

            # h0_log = h0_log_from_coverage(k, m=m_candidates, pad=pad)
            # centers = geometric_centers(true_s*np.exp((r-0)*t), h0_log, m=m_candidates, S0=true_s)

            # fit = lpca_fit_fixed_h0(
            #     k, noisy_call, S0=true_s, r=r, q=0, T=t,
            #     h0_log=h0_log, m=m, centers=centers,
            #     lambda_eq=0, l2_reg=0
            # )
            #K_fine = np.linspace(strikes.min(), strikes.max(), 400)
            ##Fitting PCA model for calls and RND
            # C_fit = lpca_price_calls(fit, k)
            # rnd = lpca_density(fit, k)
            # plt.plot(k,rnd)
            # plt.scatter(k,true_rnd)
            

            if oracle==True:
                print("oracle")
                fit, m_best = lpca_fit_select_m(
                    k, noisy_call,
                    S0=true_s, r=r, q=0, T=t,
                    m_values=m_candidates,
                    rnd_true=true_rnd,
                    k_true=k,
                    pad=pad,
                    lambda_eq=0.0,    # you can turn these back on if you like
                    l2_reg=0.0  # grid for density comparison
                )
            else:
                print("no oracle")
                fit, m_best = lpca_fit_select_m(
                    k, noisy_call,
                    S0=true_s, r=r, q=0, T=t,
                    m_values=m_candidates,
                    pad=pad,
                    lambda_eq=0.0,    # you can turn these back on if you like
                    l2_reg=0.0  # grid for density comparison
                )
                
            
            
            K_fine = np.linspace(k.min(), k.max(), len(k))
            C_fit = lpca_price_calls(fit, K_fine)
            
            # --- PCA RND on the same grid as BL for apples-to-apples ---
            rnd = lpca_density(fit, k)
            pvalue_list=[]
            
            
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse
            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit,"strikes":k}
            
        if models_dict[model_key]["method"]=="hermite":
            
            print("Attempting Hermite")
            nickname=models_dict[model_key]["nickname"]
    
            F = true_s*np.exp((r - 0)*t)
            eta1_0 = F                         # forward-ish location (not enforced)
            sigma0 = 0.25                      # rough vol guess
            eta2_0 = sigma0*np.sqrt(t)         # σ√T
            # (eta3_0, eta4_0) default to 0; feel free to seed from BL if you like

            res = fit_calls_hermite_no_forward(
                k, noisy_call, r, t,
                eta1_0=eta1_0, eta2_0=eta2_0, eta3_0=0.0, eta4_0=0.0
            )
            
            eta1, eta2, eta3, eta4 = res.x
            
            C_fit = call_curve_hermite(k, (eta1, eta2, eta3, eta4), r, t)
            rnd = rnd_hermite_jr(k, eta1, eta2, eta3, eta4)
            
            
            # plt.plot(k,rnd)
            # plt.scatter(k,true_rnd)
            
            # plt.scatter(k,noisy_call)
            # plt.plot(k,C_fit,color="orange")
            

            
            
            pvalue_list=[]
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse
            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit}
            
        if models_dict[model_key]["method"]=="local_regression_kde":
            nickname=models_dict[model_key]["nickname"]
            print(f"Attempting local Regression with KDE. Saving as: {nickname}")
            
            ##---Extract the dictionary of model parameters for local regression
            local_polynomial_kde_param_dict=models_dict[model_key]

            ##---Run the local regression with kde model
            specific_model_dict=local_regression_kde(k,noisy_call,noisy_iv,r,true_s,t,local_polynomial_kde_param_dict) #should return at the very least iv,call,rnd
            
            ##--Extract the rnd, take samples and perform the KS test (extract pvalue).
            rnd=specific_model_dict["rnd"]
            C_fit=specific_model_dict["calls"]
            ##-- The power of the KS test increases with higher sample. To avoid the trivial case of instant-rejection from high N, I take 100 random draws and average the pvalue.
            pvalue_list=[]
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse
            
            plt.plot(k,rnd)
            plt.scatter(k,true_rnd)
            # # Plot the PDF
            # plt.hist(specific_samples, bins=100, density=True, alpha=0.5, edgecolor='black', label='Sample Histogram')
            
            # # Overlay the original PDF
            # plt.plot(k, rnd, 'r-', linewidth=2, label='Risk-Neutral PDF')
            # plt.plot(k,true_rnd,label='true_rnd')
            # # Labels and legend
            # plt.xlabel('Value')
            # plt.ylabel('Density')
            # plt.title('Histogram vs Risk-Neutral PDF')
            # plt.legend()
            # plt.show()
            
            # plt.plot(k,rnd)
            # plt.plot(k,true_rnd)
            
            result_dict[nickname]=average_pvalue
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit}
        
        if models_dict[model_key]["method"]=="sieve":
            nickname=models_dict[model_key]["nickname"]
            oracle=models_dict[model_key].get("oracle",False)
            degree_mode=models_dict[model_key].get("degree_mode","auto")
            degree_grid=models_dict[model_key].get("degree_grid",range(1, 10))


            
            
            
            if oracle==True:
                
                res = hermite_sieve_calls_only(
                    Kcall=k,
                    Pcall=noisy_call,
                    st=true_s,
                    rf=r,
                    tau=t,
                    Sigma=None,
                    degree_mode="oracle",
                    degree_grid=degree_grid,
                    m_grid=1000,
                    nFold=10,
                    random_state=0,
                    alphaR_grid=[0.000],
                    S_true=k,
                    q_true=true_rnd,
                )
                
            if oracle!=True:
                
                res = hermite_sieve_calls_only(
                    Kcall=k,
                    Pcall=noisy_call,
                    st=true_s,
                    rf=r,
                    tau=t,
                    Sigma=None,
                    degree_mode=degree_mode,
                    degree_grid=degree_grid,
                    m_grid=1000,
                    nFold=10,
                    random_state=0,
                    alphaR_grid=[0.000],
                )
            
            
            print("Attempting Sieve")
            rnd = res.rnd_ST(k)
            C_fit = res.call_prices(k)




            
            
            
            # --- PCA RND on the same grid as BL for apples-to-apples ---
            #rnd = lpca_density(fit, k)
            pvalue_list=[]
            
            
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse
            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit,"strikes":k}
            
            
        if models_dict[model_key]["method"]=="splines":
            nickname=models_dict[model_key]["nickname"]
            oracle=models_dict[model_key].get("oracle",False)
            degree=models_dict[model_key].get("degree",3)
            alpha_grid=models_dict[model_key].get("alpha_grid",None)


            
            
            
            if oracle==True:
                
                
                res= fit_iv_smoothing_spline(
                    strikes=k,
                    iv=noisy_iv,
                    degree=degree,           # or 4
                    oracle=True,
                    true_strikes=k,  # where true_rnd is given
                    true_rnd=true_rnd,
                    S0=true_s,
                    r=r,
                    q=0,
                    T=t,
                    alpha_grid=None
                )
                
            if oracle!=True:
                
                res= fit_iv_smoothing_spline(
                    strikes=k,
                    iv=noisy_iv,
                    degree=degree,           # or 4
                    oracle=False,
                    S0=true_s,
                    r=r,
                    q=0,
                    T=t,
                    alpha_grid=None
                )
            
            print("Attempting splines")
            rnd = res.rnd_est
            C_fit = res.C_smooth
            
            # --- PCA RND on the same grid as BL for apples-to-apples ---
            pvalue_list=[]
            
            
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse
            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit,"strikes":k}
        
        if models_dict[model_key]["method"]=="spectral":
            
            nickname=models_dict[model_key]["nickname"]
            oracle=models_dict[model_key].get("oracle",False)
            select_reg=models_dict[model_key].get("select_reg","fixed")


            
            
            
            if oracle==True:
                res= spectral_rnd_from_call_mids(
                    strikes=k,
                    call_mid=noisy_call,
                    S0=true_s,
                    r=r,
                    q_div=0,
                    T=t,
                    select_reg="oracle",
                    reg_grid=[1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                    oracle_s=k,
                    oracle_q=true_rnd,
                )
            
                
            if oracle!=True:
                
                res= spectral_rnd_from_call_mids(
                    strikes=k,
                    call_mid=noisy_call,
                    S0=true_s,
                    r=r,
                    q_div=0,
                    T=t,
                    select_reg=select_reg,
                    reg_grid=[1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                )
            
            
            
            print("Attempting spectral")
            rnd = res.rnd(k)
            C_fit = res.calls(k)

            
            # --- PCA RND on the same grid as BL for apples-to-apples ---
            pvalue_list=[]
            
            
            for i in range(0,100):
                specific_samples=compute_samples(k,rnd,1000)
                D,pvalue= ks_2samp(true_rnd_samples, specific_samples, alternative='two-sided', method='auto')
                pvalue_list.append(pvalue)  # no assignment here
            average_pvalue=np.max(pvalue_list)
            
            mse=mean_square_error(true_rnd, rnd)
            mse_dict[nickname]=mse
            
            ##--Store the KS Pvalue into a dictionary, indexed by the nickname
            result_dict[nickname]=average_pvalue
            ##--Store the IV and RND curves into a seperate dictionary, indexed by nickname (for plotting purposes later)
            rnd_iv_dict[nickname]={"rnd":rnd,"nickname":nickname,"calls":C_fit,"strikes":k}

            
    plot_overlaid_rnd_calls(rnd_iv_dict,run_number,true_call,true_rnd,k,noisy_rnd)
    #plot function. Loop has ended. We can extract RND/calls/IVs from result_dict to plot if need be. We implement a function later
    #plot_overlaid_rnds(true_rnd,rnd_iv_dict,k,run_number,noisy_rnd,save_filename=foldername, normalize=True)
    return result_dict,mse_dict
        
def local_regression(k,call,iv,r,s,t, argument_dict):
    """
    Performs cross validation to select the best bandwidth for KernelReg.

    Parameters:
        x (np.array): 1D array of the independent variable (e.g. strikes).
        y (np.array): 1D array of the dependent variable (e.g. iv).
        candidate_bw (array): Array of candidate bandwidth values.
        cv_type (str): 'kfold' (default) for KFold CV or 'loo' for Leave-One-Out CV.
        n_splits (int): Number of folds for KFold CV (ignored for LOO).

    Returns:
        best_bw (float): The candidate bandwidth with the lowest average MSE.
        cv_errors (dict): Dictionary mapping candidate bandwidths to their average MSE.
    """
    ######### Extracting relavant information from the option Dataframe

    ############################################################## Program Begins here
    
    bw_setting=argument_dict.get("bandwidth_setting","recommended")
    cv_method = argument_dict.get('cv_method','loo')
    kde_method = argument_dict.get('kde_method',8)
    nickname=argument_dict.get('nickname',"generic local_polynomial")
    
    original_k=k
    original_iv=iv
    
    
    best_bw = None
    best_error = np.inf
    cv_errors = {}

    mask = ~np.isnan(iv)  # True where iv is not NaN

    iv = iv[mask]
    k = k[mask]
    #### Step 1: Preprocess the IV curve.

    if cv_method == 'loo':
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=cv_method, shuffle=True, random_state=42)
    
    
    #Select bandwidth settings 
    if bw_setting=="recommended": #lower bound is the average strike difference
        candidate_bw=np.linspace(np.mean(np.diff(k)),4*max(np.diff(k)),5)
    if bw_setting=="other": #lower bound is the average strike difference
        candidate_bw=np.linspace(np.mean(np.diff(k)),20*max(np.diff(k)),5)
        print("Other")
    if isinstance(bw_setting, (int, float)):        #check if bw_setting is a number
        fixed_bw=True
        best_bw=bw_setting
    else:
        fixed_bw=False
        
        
    if fixed_bw==False:
        for bw in candidate_bw:
            errors = []
            for train_idx, test_idx in cv.split(k):
                # Extract the training and test subsets
                k_train, k_test = k[train_idx], k[test_idx]
                iv_train, iv_test = iv[train_idx], iv[test_idx]
            
                # Reshape so each is (n_samples, 1) rather than 1D or scalar
                k_train = k_train.reshape(-1, 1)
                k_test = k_test.reshape(-1, 1)
            
                kr = KernelReg(endog=iv_train, exog=k_train, reg_type="ll", var_type='c', bw=[bw])
            
                # Now strike_test is guaranteed to be 2D
                y_pred, _ = kr.fit(k_test)
            
                errors.append(mean_squared_error(iv_test, y_pred))
                mean_error = np.mean(errors)
            
                # Checking for overfitting or zero estimates
                strike_grid = np.linspace(k[0], k[-1], 25).reshape(-1, 1)
                iv_est, _ = kr.fit(strike_grid)
                if np.any(iv_est == 0):
                    print("At least one value in iv_est is zero. Discarding this bandwidth.")
                    mean_error = np.inf
    
            cv_errors[bw] = mean_error
            
            if mean_error < best_error:
                best_error = mean_error
                best_bw = bw
        kr_final = KernelReg(endog=iv, exog=k, reg_type="ll", var_type='c', bw=[best_bw])
    
    kr_final = KernelReg(endog=iv, exog=k, reg_type="ll", var_type='c', bw=[best_bw])

    interpolated_iv, _ = kr_final.fit(original_k[:, None])
    interpolated_calls=black_scholes_call_from_iv(original_k, s, r, t, interpolated_iv, q=0.0)
    rnd=rnd_from_calls(original_k, interpolated_calls, r, t)
    
    
    
    
    # ### Step 3 Apply Weighted Kernel Density Estimation to Risk Neutral density
    # if kde_method=="ISJ":
    #     kde = NaiveKDE(bw=kde_method).fit(original_k,weights=rnd)
    #     kde_pdf=kde.evaluate(k)
    # else:
    #     kde = NaiveKDE(bw="scott").fit(original_k,weights=rnd)
    #     bw=kde.bw/kde_method
    #     kde = NaiveKDE(bw=bw).fit(original_k,weights=rnd)
    #     kde_pdf=kde.evaluate(original_k)
        

    #     plt.plot(original_k,rnd,label="ll rnd")
    #     plt.plot(original_k,true_rnd,label='true rnd')
    #     plt.plot(original_k,kde_pdf,label="kde rnd")
    #     plt.legend()

        
    return_dict={"iv":interpolated_iv,"calls":interpolated_calls,"rnd":rnd,"nickname":nickname}
   

    return return_dict  


def local_regression_kde(k,call,iv,r,s,t, argument_dict):
    """
    Performs cross validation to select the best bandwidth for KernelReg.

    Parameters:
        x (np.array): 1D array of the independent variable (e.g. strikes).
        y (np.array): 1D array of the dependent variable (e.g. iv).
        candidate_bw (array): Array of candidate bandwidth values.
        cv_type (str): 'kfold' (default) for KFold CV or 'loo' for Leave-One-Out CV.
        n_splits (int): Number of folds for KFold CV (ignored for LOO).

    Returns:
        best_bw (float): The candidate bandwidth with the lowest average MSE.
        cv_errors (dict): Dictionary mapping candidate bandwidths to their average MSE.
    """
    ######### Extracting relavant information from the option Dataframe

    ############################################################## Program Begins here
    
    bw_setting=argument_dict.get("bandwidth_setting","recommended")
    cv_method = argument_dict.get('cv_method','loo')
    kde_method = argument_dict.get('kde_method',8)
    alpha = argument_dict.get('alpha',0.5)

    nickname=argument_dict.get('nickname',"generic local_polynomial_kde")
    
    original_k=k
    original_iv=iv
    
    
    best_bw = None
    best_error = np.inf
    cv_errors = {}

    mask = ~np.isnan(iv)  # True where iv is not NaN

    iv = iv[mask]
    k = k[mask]
    #### Step 1: Preprocess the IV curve.

    if cv_method == 'loo':
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=cv_method, shuffle=True, random_state=42)
    
    
    #Select bandwidth settings 
    if bw_setting=="recommended": #lower bound is the average strike difference
        candidate_bw=np.linspace(np.mean(1*np.diff(k)),4*max(np.diff(k)),5)
    if isinstance(bw_setting, (int, float)):        #check if bw_setting is a number
        fixed_bw=True
        best_bw=bw_setting
    else:
        fixed_bw=False
        
        
    if fixed_bw==False:
        for bw in candidate_bw:
            errors = []
            for train_idx, test_idx in cv.split(k):
                # Extract the training and test subsets
                k_train, k_test = k[train_idx], k[test_idx]
                iv_train, iv_test = iv[train_idx], iv[test_idx]
            
                # Reshape so each is (n_samples, 1) rather than 1D or scalar
                k_train = k_train.reshape(-1, 1)
                k_test = k_test.reshape(-1, 1)
            
                kr = KernelReg(endog=iv_train, exog=k_train, reg_type="ll", var_type='c', bw=[bw])
            
                # Now strike_test is guaranteed to be 2D
                y_pred, _ = kr.fit(k_test)
            
                errors.append(mean_squared_error(iv_test, y_pred))
                mean_error = np.mean(errors)
            
                # Checking for overfitting or zero estimates
                strike_grid = np.linspace(k[0], k[-1], 25).reshape(-1, 1)
                iv_est, _ = kr.fit(strike_grid)
                if np.any(iv_est == 0):
                    print("At least one value in iv_est is zero. Discarding this bandwidth.")
                    mean_error = np.inf
    
            cv_errors[bw] = mean_error
            
            if mean_error < best_error:
                best_error = mean_error
                best_bw = bw
        kr_final = KernelReg(endog=iv, exog=k, reg_type="ll", var_type='c', bw=[best_bw])
    
    kr_final = KernelReg(endog=iv, exog=k, reg_type="ll", var_type='c', bw=[best_bw])

    interpolated_iv, _ = kr_final.fit(original_k[:, None])
    interpolated_calls=black_scholes_call_from_iv(original_k, s, r, t, interpolated_iv, q=0.0)
    rnd=rnd_from_calls(original_k, interpolated_calls, r, t)
    
    
    
    
    ### Step 3 Apply Weighted Kernel Density Estimation to Risk Neutral density
    if kde_method=="ISJ":
        kde = NaiveKDE(bw=kde_method).fit(original_k,weights=rnd)
        kde_pdf=kde.evaluate(k)
        
    if kde_method=="adaptive":
        model = adaptive_kde_fit(original_k, weights=rnd, alpha=alpha, lambda_clip=(0.25, 4.0))
        kde_pdf = adaptive_kde_eval(model, original_k)
        
    if isinstance(kde_method,(int,float))==True:
        kde = NaiveKDE(bw="scott").fit(original_k,weights=rnd)
        bw=kde.bw/kde_method
        kde = NaiveKDE(bw=bw).fit(original_k,weights=rnd)
        kde_pdf=kde.evaluate(original_k)
    
        
        
        # plt.plot(original_k,rnd,label="ll rnd")
        # plt.plot(original_k,true_rnd,label='true rnd')
        # plt.plot(original_k,kde_pdf,label="kde rnd")
        # plt.legend()

    
    repriced_calls=calls_from_rnd(kde_pdf, original_k, s, r, t, normalize=False)
    
    repriced_iv=implied_vol_from_calls(repriced_calls, s, original_k, r, t, 0, tol=1e-8, max_iter=100, vol_low=1e-8, vol_high=5.0)
    
    # plt.plot(original_k,repriced_calls)
    # plt.plot(original_k,interpolated_calls)
    
    return_dict={"iv":repriced_iv,"calls":repriced_calls,"rnd":kde_pdf,"nickname":nickname}
   

    return return_dict   
        
def plot_overlaid_rnd_calls(
    rnd_iv_dict: dict,
    run_number: int,
    true_call: np.ndarray,
    true_rnd: np.ndarray,
    true_k: np.ndarray,
    noisy_rnd: np.ndarray | None = None,
    normalize: bool = True,
    save_filename: str = "comparison_rnds",
):
    """
    Make a 1x2 figure:
      (1) True calls vs fitted calls from rnd_iv_dict (over strikes)
      (2) True RND (and optional noisy true RND) vs all estimated RNDs

    Parameters
    ----------
    rnd_iv_dict : dict
        Mapping method_key -> {
            'rnd': np.ndarray,             # density on strike grid
            'calls': np.ndarray,           # fitted call prices on strike grid (optional)
            'strikes': np.ndarray,         # strike grid used for this method (optional)
            'nickname': str                # pretty label (optional)
        }
    true_call : np.ndarray
        True call prices on true_k.
    true_rnd : np.ndarray
        True risk-neutral density on true_k.
    true_k : np.ndarray
        Strike grid for the true series.
    noisy_rnd : np.ndarray | None
        Optional noisy version of the true density on true_k.
    normalize : bool
        If True, each density is clipped at 0 and normalized to integrate to 1 by trapz.
    save_filename : str
        Basename for the saved PNG inside ./comparison_rnd_folder/.

    Returns
    -------
    fig, (ax_calls, ax_rnd) : matplotlib Figure and Axes
    """
    # --- helpers ---
    def _normalize_pdf(pdf, k):
        pdf = np.asarray(pdf, float)
        pdf = np.clip(pdf, 0, None)
        area = np.trapz(pdf, k)
        if area > 0:
            pdf = pdf / area
        return pdf

    # ensure folder
    out_dir = "comparison_rnd_folder"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{save_filename}_{run_number}.png")

    # figure
    fig, (ax_calls, ax_rnd) = plt.subplots(1, 2, figsize=(12, 5), dpi=140)

    # --- LEFT: Calls ---
    # plot true call
    ax_calls.plot(true_k, true_call, label="True calls", linewidth=2.5, color="black")

    # overlay fitted calls for each method (if available)
    for key, payload in rnd_iv_dict.items():
        label = payload.get("nickname", str(key))
        k_i = payload.get("strikes", true_k)
        calls_i = payload.get("calls", None)
        if calls_i is None:
            continue  # skip if no calls present
        # Basic length guard (skip if inconsistent)
        if len(k_i) != len(calls_i):
            continue
        ax_calls.plot(k_i, calls_i, label=label, alpha=0.9)

    ax_calls.set_xlabel("Strike (K)")
    ax_calls.set_ylabel("Call price")
    ax_calls.set_title("True vs Fitted Call Prices")
    ax_calls.grid(True, alpha=0.25)
    ax_calls.legend(ncol=1, fontsize=9)

    # --- RIGHT: Densities ---
    # true RND
    tr_pdf = _normalize_pdf(true_rnd, true_k) if normalize else true_rnd
    ax_rnd.plot(true_k, tr_pdf, label="True RND", linewidth=2.5, color="black")

    # optional noisy true RND
    if noisy_rnd is not None:
        nr = _normalize_pdf(noisy_rnd, true_k) if normalize else noisy_rnd
        ax_rnd.plot(true_k, nr, label="Noisy true RND", linestyle="--", linewidth=2.0)

    # overlay method RNDs
    for key, payload in rnd_iv_dict.items():
        label = payload.get("nickname", str(key))
        k_i = payload.get("strikes", true_k)
        rnd_i = payload.get("rnd", None)
        if rnd_i is None or len(k_i) != len(rnd_i):
            continue
        rnd_plot = _normalize_pdf(rnd_i, k_i) if normalize else rnd_i
        ax_rnd.plot(k_i, rnd_plot, label=label, alpha=0.95)

    ax_rnd.set_xlabel("Strike (K)")
    ax_rnd.set_ylabel("Density")
    ax_rnd.set_title("Risk-Neutral Densities (overlaid)")
    ax_rnd.grid(True, alpha=0.25)
    ax_rnd.legend(ncol=1, fontsize=9)

    fig.suptitle("Call Fits and Risk-Neutral Densities", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    return fig, (ax_calls, ax_rnd)





        
# def plot_overlaid_rnds(true_rnd, rnd_iv_dict, k,run_number,noisy_rnd,normalize=True,
#                        save_filename="comparison_rnds.png"):
#     """
#     Plot the true risk-neutral density (true_rnd) and all RNDs in rnd_iv_dict overlaid vs strikes k,
#     and save the figure to 'comparison_rnd_folder' in the current directory.

#     Parameters
#     ----------
#     true_rnd : array_like
#         The true RND values aligned with k.
#     rnd_iv_dict : dict
#         Dict like {
#             'model_a': {'rnd': np.array, 'iv': ..., 'calls': ..., 'nickname': '...'},
#             'model_b': {...},
#             ...
#         }
#     k : array_like
#         1-D array of strikes aligned with each rnd.
#     normalize : bool, optional
#         If True, scale each RND to integrate to 1 on k.
#     save_filename : str, optional
#         The name of the saved file (default 'comparison_rnds.png').
#     """
#     k = np.asarray(k)
#     randomnumbers=np.random.uniform(1, 3)
#     plt.figure(figsize=(6, 5))

#     # Plot the true RND
#     mask = np.isfinite(k) & np.isfinite(true_rnd)
#     kk_true, rr_true = k[mask], np.asarray(true_rnd)[mask]
#     order = np.argsort(kk_true)
#     kk_true, rr_true = kk_true[order], rr_true[order]

#     if normalize:
#         area = np.trapz(rr_true, kk_true)
#         if np.isfinite(area) and area > 0:
#             rr_true = rr_true / area

#     plt.plot(kk_true, rr_true, 'k--', linewidth=2.5, label='True RND')
#     plt.plot(k,noisy_rnd,label="Noisy RND")
#     # Plot model RNDs
#     for key, d in (rnd_iv_dict.items() if isinstance(rnd_iv_dict, dict)
#                    else enumerate(rnd_iv_dict)):
#         rnd = np.asarray(d.get('rnd'))
#         if rnd is None:
#             continue

#         # Align lengths and drop non-finite values
#         n = min(len(k), len(rnd))
#         kk = k[:n]
#         rr = rnd[:n]
#         mask = np.isfinite(kk) & np.isfinite(rr)
#         kk, rr = kk[mask], rr[mask]

#         # Sort by strike
#         order = np.argsort(kk)
#         kk, rr = kk[order], rr[order]

#         # Optional normalization
#         if normalize:
#             area = np.trapz(rr, kk)
#             if np.isfinite(area) and area > 0:
#                 rr = rr / area

#         label = d.get('nickname') or (key if isinstance(key, str) else f"model_{key}")
#         plt.plot(kk, rr, linewidth=2, label=label)

#     plt.xlabel('Strike')
#     plt.ylabel('Density $q(K)$')
#     plt.title('Overlaid RNDs')
#     plt.grid(alpha=0.25)
#     plt.legend()
#     plt.tight_layout()

#     # Create save folder if not exists
#     save_filename1=f"compared_rnd_{run_number}"
#     save_dir = os.path.join(os.getcwd(), save_filename+"folder")
#     os.makedirs(save_dir, exist_ok=True)

#     save_path = os.path.join(save_dir, save_filename1)
#     plt.savefig(save_path, dpi=300)
#     plt.close()

#     print(f"Figure saved to: {save_path}")
#     return None


def mean_square_error(true_rnd,estimated_rnd):
    """
    Computes the Mean square error between the true RND and the estiamted RND. Assume they are evaluated on the same axis, and already normalized to one.

    Parameters
    ----------
    true_rnd : array
        values of the true RND.
    estimated_rnd : array
        values of the estiamted rnd.

    Returns
    -------
    mse : TYPE
        DESCRIPTION.

    """
    mse=np.sqrt(np.mean((true_rnd-estimated_rnd)**2))
    return mse
def black_scholes_call_from_iv(K, S, R, T, iv, q=0.0):
    """
    Convert IVs to Black–Scholes call prices C(K).

    Parameters
    ----------
    K : array-like
        Strikes.
    S : float
        Spot price.
    R : float
        Continuously compounded risk-free rate.
    T : float
        Maturity in years.
    iv : array-like
        Implied vols corresponding to K.
    q : float
        Continuous dividend yield (default 0).

    Returns
    -------
    C : np.ndarray
        Call prices for each strike K.
    """
    K = np.asarray(K, dtype=float)
    iv = np.asarray(iv, dtype=float)

    eps = 1e-12
    volT = np.maximum(iv * np.sqrt(max(T, eps)), eps)

    d1 = (np.log(np.maximum(S, eps) / np.maximum(K, eps)) + (R - q + 0.5 * iv**2) * T) / volT
    d2 = d1 - volT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)

    disc_r = np.exp(-R * T)
    disc_q = np.exp(-q * T)

    C = disc_q * S * Nd1 - disc_r * K * Nd2

    # Handle near-degenerate cases
    intrinsic = np.maximum(disc_q * S - disc_r * K, 0.0)
    near_degenerate = (iv < 1e-8) | (T < 1e-8)
    C = np.where(near_degenerate, intrinsic, C)

    return C
def load_and_merge(prefix, folder="."):
    """
    Loads all pickle files starting with prefix and merges:
      - df_kolm  : concatenated DataFrame
      - df_mse   : concatenated DataFrame
      - list_kolm: merged list
      - list_mse : merged list
    """
    
    merged_df_kolm = []
    merged_df_mse = []
    merged_list_kolm = []
    merged_list_mse = []
    count = 0

    for fname in os.listdir(folder):
        if fname.startswith(prefix) and fname.endswith(".pkl"):
            with open(os.path.join(folder, fname), "rb") as f:
                data = pickle.load(f)

            merged_df_kolm.append(data["df_kolm"])
            merged_df_mse.append(data["df_mse"])
            merged_list_kolm.extend(data["list_kolm"])
            merged_list_mse.extend(data["list_mse"])

            count += 1

    # Final concatenation
    # df_kolm_all = pd.concat(merged_df_kolm, ignore_index=True)
    # df_mse_all  = pd.concat(merged_df_mse, ignore_index=True)
    
    df_kolm = pd.DataFrame(merged_list_kolm)
    df_mse_all = pd.DataFrame(merged_list_mse)


    return df_kolm_all, df_mse_all, merged_list_kolm, merged_list_mse, count
def bs_call_disc(S0, K, r, q, T, sigma):
    if sigma <= 0:
        return max(S0*np.exp(-q*T) - K*np.exp(-r*T), 0.0)
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_iv_from_price_disc(S0, K, r, q, T, price, lo=1e-6, hi=5.0):
    lower = max(S0*np.exp(-q*T) - K*np.exp(-r*T), 0.0)
    upper = S0*np.exp(-q*T)
    target = float(np.clip(price, lower, upper))
    f = lambda vol: bs_call_disc(S0, K, r, q, T, vol) - target
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        hi_try = hi
        for _ in range(10):
            hi_try *= 2.0
            if f(lo) * f(hi_try) <= 0:
                hi = hi_try
                break
        if f(lo) * f(hi) > 0:
            return np.nan
    try:
        return brentq(f, lo, hi, maxiter=200, xtol=1e-10)
    except ValueError:
        return np.nan
    
# #####Sait noisee
# heston_dict = {"kappa":(0.6,0.6),"theta":(0.3,0.3),"sigma":(0.2,0.20),"v0":(0.02,0.02),"rho":(-0.9,0.9)}
# market_dict={"strikes":np.linspace(25,250,100),"underlying_price":120,"risk_free_rate":0.03,"maturity":0.5}
# rng = random.Random(123)
# hp = pick_model_param(heston_dict, rng)
# print(hp)

# models_dict={}
# models_dict["mixture"]={"method":"mixture","nickname":"2W1L","n_lognormal":2,"n_weibull":1,"n_starts":1}
# models_dict["local_regression"]={"method":"local_regression","nickname":"loc_regcv","bw_method":"cv"}
# models_dict["local_regression2"]={"method":"local_regression","nickname":"loc_regSait","bw_method":"ait_sahalia"}

# models_dict["figlewski"]={"method":"figlewski","nickname":"quartic"}
# models_dict["pca"]={"method":"pca","nickname":"pca_v1","pad":1.2,"m":23*4}
# models_dict["local_regression_kde1"]={"method":"local_regression_kde","nickname":"loc_regkde8","bw_setting":"recoemmended","cv_method":'loo',"kde_method":8}
# #models_dict["local_regression_kde2"]={"method":"local_regression_kde","nickname":"loc_regkde7","bw_setting":"recoemmended","cv_method":'loo',"kde_method":7}
# #models_dict["local_regression_kde3"]={"method":"local_regression_kde","nickname":"loc_regkde6","bw_setting":"recoemmended","cv_method":'loo',"kde_method":6}

# models_dict["hermite"]={"method":"hermite","nickname":"hermite_def"}

# noise_dict_1={'mode': 'boradenko', 'scale': 0.5}
# noise_dict_2={'mode': 'boradenko', 'scale': 0.25}


# monte_example=monte_carlo(market_dict)
# # dfuni175,tt_multi,mse_multidf175,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_1,"uni","unimodal",heston_dict)
# # dfuni275,tt_multi,mse_multidf275,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_2,"uni","unimodal",heston_dict)
# # dfuni375,tt_multi,mse_multidf375,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_3,"uni","unimodal",heston_dict)
# # dfuni475,tt_multi,mse_multidf475,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_4,"uni","unimodal",heston_dict)

# dfsimp1bo,tt_multi,mse_simpledf1bo,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_1,"none","unimodal",heston_dict)
# dfsimp2bo,tt_multi,mse_simpledf2bo,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_2,"none","unimodal",heston_dict)

# dfunibo1,tt_multi,mse_unidbodf1,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_1,"uni","unimodal",heston_dict)
# dfunibo2,tt_multi,mse_unidbodf2,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_2,"uni","unimodal",heston_dict)


# ##################################my noise
# heston_dict = {"kappa":(0.6,0.6),"theta":(0.3,0.3),"sigma":(0.2,0.20),"v0":(0.02,0.02),"rho":(-0.9,0.9)}
# market_dict={"strikes":np.linspace(25,250,100),"underlying_price":120,"risk_free_rate":0.03,"maturity":0.5}
# rng = random.Random(123)
# hp = pick_model_param(heston_dict, rng)
# print(hp)

# models_dict={}
# models_dict["mixture"]={"method":"mixture","nickname":"2W1L","n_lognormal":2,"n_weibull":1,"n_starts":1}
# models_dict["local_regression"]={"method":"local_regression","nickname":"loc_regcv","bw_method":"cv"}
# models_dict["local_regression2"]={"method":"local_regression","nickname":"loc_regSait","bw_method":"ait_sahalia"}

# models_dict["figlewski"]={"method":"figlewski","nickname":"quartic"}
# models_dict["pca"]={"method":"pca","nickname":"pca_v1","pad":1.2,"m":23*4}
# models_dict["local_regression_kde1"]={"method":"local_regression_kde","nickname":"loc_regkde8","bw_setting":"recoemmended","cv_method":'loo',"kde_method":8}
# #models_dict["local_regression_kde2"]={"method":"local_regression_kde","nickname":"loc_regkde7","bw_setting":"recoemmended","cv_method":'loo',"kde_method":7}
# #models_dict["local_regression_kde3"]={"method":"local_regression_kde","nickname":"loc_regkde6","bw_setting":"recoemmended","cv_method":'loo',"kde_method":6}

# models_dict["hermite"]={"method":"hermite","nickname":"hermite_def"}
# noise_dict_0={'mode': 'rel_atm', 'sigma': 0.01, 'decay': 0.2, 'p': 1}
# noise_dict_1={'mode': 'rel_atm', 'sigma': 0.02, 'decay': 0.2, 'p': 1}
# noise_dict_2={'mode': 'rel_atm', 'sigma': 0.03, 'decay': 0.2, 'p': 1}
# noise_dict_3={'mode': 'rel_atm', 'sigma': 0.04, 'decay': 0.2, 'p': 1}
# noise_dict_4={'mode': 'rel_atm', 'sigma': 0.05, 'decay': 0.2, 'p': 1}

# #################


# monte_example=monte_carlo(market_dict)
# dfuni075,tt_multi,mse_multidf0175,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_0,"uni","unimodal",heston_dict)
# dfuni175,tt_multi,mse_multidf175,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_1,"uni","unimodal",heston_dict)
# dfuni275,tt_multi,mse_multidf275,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_2,"uni","unimodal",heston_dict)
# dfuni375,tt_multi,mse_multidf375,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_3,"uni","unimodal",heston_dict)
# dfuni475,tt_multi,mse_multidf475,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_4,"uni","unimodal",heston_dict)

# dfsimp0,tt_multi,mse_simpledf0,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_0,"none","unimodal",heston_dict)
# dfsimp1,tt_multi,mse_simpledf1,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_1,"none","unimodal",heston_dict)
# dfsimp2,tt_multi,mse_simpledf2,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_2,"none","unimodal",heston_dict)
# dfsimp3,tt_multi,mse_simpledf3,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_3,"none","unimodal",heston_dict)
# dfsimp4,tt_multi,mse_simpledf4,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_4,"none","unimodal",heston_dict)


# df_list=[mse_multidf1,mse_multidf2,mse_multidf3,mse_multidf4]
# v = pd.concat(df_list, ignore_index=True, join="inner")

# heston_dict = {"kappa":(0.6,0.6),"theta":(0.3,0.3),"sigma":(0.2,0.20),"v0":(0.02,0.02),"rho":(-0.9,0.9)}
# market_dict={"strikes":np.linspace(25,250,100),"underlying_price":120,"risk_free_rate":0.03,"maturity":0.5}
# rng = random.Random(123)
# hp = pick_model_param(heston_dict, rng)
# print(hp)
# noise_dict_low={'mode': 'rel_atm', 'sigma': 0.04, 'decay': 0.2, 'p': 1}

# models_dict={}
# models_dict["mixture"]={"method":"mixture","nickname":"2W1L","n_lognormal":2,"n_weibull":1,"n_starts":1}
# models_dict["local_regression"]={"method":"local_regression","nickname":"loc_regcv","bw_method":"cv"}
# models_dict["local_regression2"]={"method":"local_regression","nickname":"loc_regSait","bw_method":"ait_sahalia"}

# models_dict["figlewski"]={"method":"figlewski","nickname":"quartic"}
# models_dict["pca"]={"method":"pca","nickname":"pca_v1","pad":1.2,"m":23*4}
# models_dict["local_regression_kde1"]={"method":"local_regression_kde","nickname":"loc_regkde8","bw_setting":"recoemmended","cv_method":'loo',"kde_method":8}
# models_dict["local_regression_kde2"]={"method":"local_regression_kde","nickname":"loc_regkde7","bw_setting":"recoemmended","cv_method":'loo',"kde_method":7}
# models_dict["local_regression_kde3"]={"method":"local_regression_kde","nickname":"loc_regkde6","bw_setting":"recoemmended","cv_method":'loo',"kde_method":6}

# models_dict["hermite"]={"method":"hermite","nickname":"hermite_def"}

# monte_example=monte_carlo(market_dict)
# dfsimple,tt_simple,mse_simpledf,mse_simplelis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low,"none","unimodal",heston_dict)
# dfuni,tt_multi,mse_multidf,mse_multilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low,"uni","unimodal",heston_dict)
# dfbi,tt_bi,mse_bidf,mse_bilis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low,"multi","unimodal",heston_dict)
# dfboth,tt_both,mse_bothdf,mse_bothlis=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low,"both","unimodal",heston_dict)

# heston_dict = {"kappa":(0.6,0.6),"theta":(0.3,0.3),"sigma":(0.2,0.20),"v0":(0.02,0.02),"rho":(-0.6,-0.6)}

# #dffixed1,tt_fixed,mse_fixed,mse_fixed=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low,"none","unimodal",heston_dict)


# ########################Lower noise
# noise_dict_low2={'mode': 'rel_atm', 'sigma': 0.05, 'decay': 0.2, 'p': 1}

# heston_dict = {"kappa":(0.6,0.6),"theta":(0.3,0.3),"sigma":(0.2,0.20),"v0":(0.02,0.02),"rho":(-0.9,0.9)}
# market_dict={"strikes":np.linspace(25,250,100),"underlying_price":120,"risk_free_rate":0.03,"maturity":0.5}
# rng = random.Random(123)
# hp = pick_model_param(heston_dict, rng)
# print(hp)
    
# models_dict={}
# models_dict["mixture"]={"method":"mixture","nickname":"2W1L","n_lognormal":2,"n_weibull":1,"n_starts":1}
# models_dict["local_regression"]={"method":"local_regression","nickname":"loc_regcv","bw_method":"cv"}
# models_dict["local_regression2"]={"method":"local_regression","nickname":"loc_regSait","bw_method":"ait_sahalia"}

# models_dict["figlewski"]={"method":"figlewski","nickname":"quartic"}
# models_dict["pca"]={"method":"pca","nickname":"pca_v1","pad":1.2,"m":23*4}
# models_dict["local_regression_kde1"]={"method":"local_regression_kde","nickname":"loc_regkde8","bw_setting":"recoemmended","cv_method":'loo',"kde_method":8}
# models_dict["local_regression_kde2"]={"method":"local_regression_kde","nickname":"loc_regkde7","bw_setting":"recoemmended","cv_method":'loo',"kde_method":7}
# models_dict["local_regression_kde3"]={"method":"local_regression_kde","nickname":"loc_regkde6","bw_setting":"recoemmended","cv_method":'loo',"kde_method":6}

# models_dict["hermite"]={"method":"hermite","nickname":"hermite_def"}

# monte_example=monte_carlo(market_dict)
# dfsimple2,tt_simple2,mse_simpledf2,mse_simpleli2s=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low2,"none","unimodal",heston_dict)
# dfuni2,tt_multi2,mse_multid2f,mse_multilis2=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low2,"uni","unimodal",heston_dict)
# dfbi2,tt_bi2,mse_bidf2,mse_bilis2=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low2,"multi","unimodal",heston_dict)
# dfboth2,tt_both2,mse_both2,mse_both2=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low2,"both","unimodal",heston_dict)

# heston_dict = {"kappa":(0.6,0.6),"theta":(0.3,0.3),"sigma":(0.2,0.20),"v0":(0.02,0.02),"rho":(-0.6,-0.6)}

# #dffixed2,tt_fixed2,mse_fixed2,mse_fixed2=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low2,"none","unimodal",heston_dict)


# # noise_dict_low2={'mode': 'rel_atm', 'sigma': 0.01, 'decay': 0.2, 'p': 1}
# # heston_dict = {"kappa":(0.6,0.6),"theta":(0.3,0.3),"sigma":(0.2,0.20),"v0":(0.02,0.02),"rho":(-0.9,0.9)}
# # market_dict={"strikes":np.linspace(25,250,100),"underlying_price":120,"risk_free_rate":0.03,"maturity":0.5}
# # rng = random.Random(123)
# # hp = pick_model_param(heston_dict, rng)
# # print(hp)
# # monte_example=monte_carlo(market_dict)
# # #dffixed2,tt_fixed2,mse_fixed2,mse_fixed22=monte_example.simulate_mc(market_dict,models_dict,300,noise_dict_low2,"none","unimodal")
# # dfsimple2,tt_simple2,mse_simpledf2,mse_simplelis2=monte_example.simulate_mc(market_dict,models_dict,30,noise_dict_low2,"none","unimodal")
# # dfuni2,tt_multi2,mse_multidf2,mse_multilis2=monte_example.simulate_mc(market_dict,models_dict,30,noise_dict_low2,"uni","unimodal")
# # dfbi2,tt_bi2,mse_bidf2,mse_bilis2=monte_example.simulate_mc(market_dict,models_dict,30,noise_dict_low2,"multi","unimodal")