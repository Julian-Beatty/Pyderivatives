import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from Heston_model import*

# ==========================
# Hermite (Jarrow–Rudd / Gram–Charlier A) parameterization
# ==========================
# z = ( ln(x/η1) + η2^2/2 ) / η2   where η2 = σ√T
def _z(x, eta1, eta2):
    x = np.asarray(x)
    return (np.log(np.maximum(x, 1e-300) / eta1) + 0.5*eta2**2) / np.maximum(eta2, 1e-12)

def H3(z): return z**3 - 3.0*z
def H4(z): return z**4 - 6.0*z**2 + 3.0

def rnd_hermite_jr(x, eta1, eta2, eta3, eta4):
    """
    RND from the paper's parameterization (Eq. D.13 in your screenshot):
        f(x;η) = 1/(√(2π) η2 x) * exp(-z^2/2) * [ 1 + (η3/3!)H3(z) + (η4/4!)H4(z) ]
    NOTE: Truncated GC can go slightly negative in tails; we clip to 0 for plotting.
    """
    x = np.asarray(x)
    z = _z(x, eta1, eta2)
    base = np.exp(-0.5*z**2) / (np.sqrt(2*np.pi) * np.maximum(eta2, 1e-12) * np.maximum(x, 1e-300))
    corr = 1.0 + (eta3/6.0)*H3(z) + (eta4/24.0)*H4(z)
    f = base * corr
    return np.clip(f, 0.0, np.inf)

# ==========================
# Analytic call price under Hermite expansion (Eq. D.14)
# ==========================
def _d1_d2(K, eta1, eta2):
    # Black–Scholes d1,d2 but with "forward" proxy eta1 and total vol eta2
    # d1 = [ ln(η1/K) + 0.5 η2^2 ] / η2,  d2 = d1 - η2
    K = np.asarray(K)
    d1 = (np.log(np.maximum(eta1,1e-300)/np.maximum(K,1e-300)) + 0.5*eta2**2) / np.maximum(eta2,1e-12)
    d2 = d1 - eta2
    return d1, d2

def _F3_F4(d1, d2, eta2):
    """
    Correction terms in D.14 (the standard JR forms):
        F3 = η2^3 Φ(d1) - φ(d1) η2 (d2 - η2)
        F4 = η2^4 Φ(d1) + φ(d1) ( η2*(d1**2 - 1) - 3*η2**2*d2 )
    """
    Phi_d1 = norm.cdf(d1)
    phi_d1 = norm.pdf(d1)
    F3 = (eta2**3)*Phi_d1 - phi_d1*eta2*(d2 - eta2)
    F4 = (eta2**4)*Phi_d1 + phi_d1*( eta2*(d1**2 - 1.0) - 3.0*(eta2**2)*d2 )
    return F3, F4

def call_price_hermite(K, eta1, eta2, eta3, eta4, r, T):
    """
    O(K;η) = e^{-rT}[ η1 Φ(d1) - K Φ(d2) + η1*η3*F3/3! + η1*η4*F4/4! ]
    with d1,d2 as above and F3,F4 from _F3_F4.
    """
    d1, d2 = _d1_d2(K, eta1, eta2)
    F3, F4 = _F3_F4(d1, d2, eta2)
    price_term = eta1*norm.cdf(d1) - K*norm.cdf(d2) + eta1*(eta3*F3/6.0 + eta4*F4/24.0)
    return np.exp(-r*T) * price_term

# Vectorized wrapper to get a price curve
def call_curve_hermite(Ks, eta, r, T):
    eta1, eta2, eta3, eta4 = eta
    return call_price_hermite(np.asarray(Ks), eta1, eta2, eta3, eta4, r, T)

# ==========================
# Fit (least squares in call space), NO forward constraint
# ==========================
def fit_calls_hermite_no_forward(strikes, C_mkt, r, T, 
                                 eta1_0, eta2_0, eta3_0=0.0, eta4_0=0.0,
                                 bounds=((1e-8,None),(1e-8,5.0),(-5.0,5.0),(-5.0,10.0)),
                                 weights=None):
    """
    Minimize mean( w_i * (O(K_i;η) - C_mkt_i)^2 ) over η = (η1,η2,η3,η4),
    subject to simple bounds η1>0, η2>0. NO constraint on E[S_T].

    Initialization: (eta1_0, eta2_0, 0, 0)
    """
    K = np.asarray(strikes)
    C_mkt = np.asarray(C_mkt)
    if weights is None:
        w = np.ones_like(K)
    else:
        w = np.asarray(weights)

    def obj(theta):
        C_fit = call_curve_hermite(K, theta, r, T)
        return np.mean(w * (C_fit - C_mkt)**2)

    theta0 = np.array([eta1_0, eta2_0, eta3_0, eta4_0], dtype=float)
    res = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=1000))
    return res
def trapz(y, x): return np.trapz(y, x)

# ==========================
# === USE WITH YOUR DATA ===
# ==========================
# Known inputs & market calls:
if __name__ == "__main__":

    S0, r, q, T = 120.0, 0.02, 0.00, 0.5
    strikes = np.linspace(25, 250, 80)
    
        # TRUE market: Heston (closed-form)
    hparams = HestonParams(kappa=0.5, theta=0.2, sigma=0.15, v0=0.02, rho=0.95)
    C_mkt = heston_call_prices_fast(S0, strikes, r, q, T, hparams)  # discounted prices
    
    
    
    dC_dK   = np.gradient(C_mkt, strikes, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK,  strikes, edge_order=2)
    q_heston = np.exp(r*T) * d2C_dK2
    
    # --- Initialization (simple & robust) ---
    F = S0*np.exp((r - q)*T)
    eta1_0 = F                         # forward-ish location (not enforced)
    sigma0 = 0.25                      # rough vol guess
    eta2_0 = sigma0*np.sqrt(T)         # σ√T
    # (eta3_0, eta4_0) default to 0; feel free to seed from BL if you like
    
    res = fit_calls_hermite_no_forward(
        strikes, C_mkt, r, T,
        eta1_0=eta1_0, eta2_0=eta2_0, eta3_0=0.0, eta4_0=0.0
    )
    
    eta1, eta2, eta3, eta4 = res.x
    print("Success:", res.success, "|", res.message)
    print(f"eta: eta1={eta1:.6f}, eta2={eta2:.6f}, eta3={eta3:.6f}, eta4={eta4:.6f}")
    
    # === Outputs: fitted calls and fitted RND (paper's closed forms) ===
    C_fit = call_curve_hermite(strikes, (eta1, eta2, eta3, eta4), r, T)
    q_fit = rnd_hermite_jr(strikes, eta1, eta2, eta3, eta4)
    
    # Diagnostics
    rmse_calls = np.sqrt(np.mean((C_fit - C_mkt)**2))
    print(f"Call RMSE: {rmse_calls:.6e}")
    
    # ==========================
    # Plots
    # ==========================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    
    # ---------- (a) Risk-Neutral Density ----------
    axes[0].plot(strikes, q_heston, label="Heston RND (from BL)", lw=2, alpha=0.9)
    axes[0].plot(strikes, q_fit, label="Hermite RND (analytic JR fit)", lw=2, ls="--")
    axes[0].set_title("Risk-Neutral Density: Heston vs Hermite (JR, no forward constraint)")
    axes[0].set_xlabel("Strike $K$")
    axes[0].set_ylabel("$q(K)$")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # ---------- (b) Call Prices ----------
    axes[1].plot(strikes, C_mkt, label="Heston Calls (discounted)", lw=2, alpha=0.9)
    axes[1].plot(strikes, C_fit, label="Hermite-fitted Calls ", lw=2, ls="--")
    axes[1].set_title("Call Prices: Heston vs Hermite-fitted")
    axes[1].set_xlabel("Strike $K$")
    axes[1].set_ylabel("Call Price")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()