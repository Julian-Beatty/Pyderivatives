from Mixture_LWD import *
from New_Simulation_density import *
from Heston_model import *
import numpy as np
import matplotlib.pyplot as plt

#####################Replicating Figure 2.1 
###Depending on the shape of the RND its possible that mixtures of Wiebulls can fit better than mixtures of lognormals. As is the case of right skewed.
S0, r, q, T = 120.0, 0.02, 0.00, 0.5
strikes = np.linspace(50, 250, 200)

def fit_lognormal_mixtures(strikes, C_mkt, S0, r, T, max_M=3, Ms=(3, 2, 1, 0)):
    """
    Fit LWM models with fixed numbers of lognormal components M1 in Ms.
    Returns:
        fits: dict {M1: FittedMixtureObject}
    """
    fits = {}
    for M1 in Ms:
        fit, spec = evolutionary_lwm_fit(
            K=strikes,
            C_mkt=C_mkt,
            S0=S0,
            r=r,
            T=T,
            M_max=max_M,
            penalty_lambda=0.0,
            random_starts=1,
            seed=42,
            var_c=0.1,
            var_penalty=1e4,
            improvement_tol=1e-4,
            metric="loss",  # using plain loss here
            use_wald=False,
            wald_alpha=0.05,
            wald_p=1,
            wald_q=1,
            weights=None,
            fixed_M=max_M,
            fixed_M1=M1,
        )
        fits[M1] = fit
    return fits

def make_case(rho):
    """
    Build true Heston RND and LWM fits for a given rho (controls skew).
    """
    true_p = HestonParams(kappa=0.5, theta=0.05, sigma=0.25, v0=0.02, rho=rho)
    C = heston_call_prices_fast(S0, strikes, r, q, T, true_p)
    rnd_true = rnd_from_calls(strikes, C, r, T)
    fits = fit_lognormal_mixtures(strikes, C, S0, r, T)
    return C, rnd_true, fits

# ----- Left-skewed (rho = -0.9) and right-skewed (rho = +0.9) -----

C_left, rnd_left, fits_left = make_case(rho=-0.9)
C_right, rnd_right, fits_right = make_case(rho=+0.9)

# ----- Plotting -----

fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=True)

# common colors / labels mapping
M_list = [3,0]
labels = {
    3: "3 lognormals",
    2: "2 lognormals",
    1: "1 lognormal",
    0: "0 lognormals"
}

# Style settings for clarity
true_style = dict(color="black", lw=3.5, ls="-", alpha=1.0)
mix_style  = dict(lw=1.6, alpha=0.7)

# ---------------- LEFT SKEW ---------------- #
ax = axes[0]

# TRUE RND — make it pop
ax.plot(strikes, rnd_left, **true_style, label="True RND (Heston)")

# Mixture fits
for M in M_list:
    ax.plot(strikes, fits_left[M].qhat(strikes),
            label=labels[M], **mix_style)

ax.set_title("Left-skewed RND (ρ = -0.9)", fontsize=12)
ax.set_xlabel("Terminal price $S_T$")
ax.set_ylabel("Density $q(S_T)$")
ax.legend(loc="best")
ax.grid(True, alpha=0.25)

# ---------------- RIGHT SKEW ---------------- #
ax = axes[1]

# TRUE RND — same emphasis
ax.plot(strikes, rnd_right, **true_style, label="True RND (Heston)")

# Mixture fits
for M in M_list:
    ax.plot(strikes, fits_right[M].qhat(strikes),
            label=labels[M], **mix_style)

ax.set_title("Right-skewed RND (ρ = 0.9)", fontsize=12)
ax.set_xlabel("Terminal price $S_T$")
ax.legend(loc="best")
ax.grid(True, alpha=0.25)

fig.suptitle("Comparison of LWM Fits: True RND vs 3, 2, 1, and 0 Lognormals",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

from numpy.linalg import norm

def best_model(rnd_true, fits, strikes):
    """
    Returns the M1 that minimizes L2 distance between
    the true RND and fitted RND.
    """
    errors = {}
    for M1, fit in fits.items():
        qhat = fit.qhat(strikes)
        err = norm(rnd_true - qhat)  # L2 error
        errors[M1] = err

    # Pick the M with minimum error
    best_M1 = min(errors, key=errors.get)
    return best_M1, errors

# ----- Evaluate best models -----

best_left_M, errors_left = best_model(rnd_left, fits_left, strikes)
best_right_M, errors_right = best_model(rnd_right, fits_right, strikes)

total_M = 3  # since fixed_M = 3 for all fits

print("\n======================")
print(" Best Models Summary")
print("======================")

print("\nLeft-skewed RND (rho = -0.9): ")
print(" Errors by M1 (lognormals):")
for M1, e in errors_left.items():
    M_weib = total_M - M1
    print(f"   M1 = {M1}: ({M1} LN, {M_weib} WB)  L2 error = {e:.6f}")
best_left_weib = total_M - best_left_M
print(f" --> Best model: {best_left_M} lognormals, {best_left_weib} Weibulls")

print("\nRight-skewed RND (rho = 0.9): ")
print(" Errors by M1 (lognormals):")
for M1, e in errors_right.items():
    M_weib = total_M - M1
    print(f"   M1 = {M1}: ({M1} LN, {M_weib} WB)  L2 error = {e:.6f}")
best_right_weib = total_M - best_right_M
print(f" --> Best model: {best_right_M} lognormals, {best_right_weib} Weibulls\n")


# ============================================================
#   Setup: Same as before
# ============================================================

true_p = HestonParams(kappa=0.5, theta=0.05, sigma=0.25, v0=0.02, rho=-0.9)
S0, r, q, T = 120.0, 0.02, 0.00, 0.5
strikes = np.linspace(50, 250, 100)

# true "clean" market data
C_clean = heston_call_prices_fast(S0, strikes, r, q, T, true_p)

# noise
###The "low noise" setting he mentions in the paper seems to be quite a lot. Lowering it would give a better fit to the RND.
noise_dict_small_error = {"mode": "yifan_li", "scale": 0.1212}
C_noisy = noisy_data_function(strikes, C_clean, S0, noise_dict_small_error)

# true RND (for plotting only)
true_rnd = rnd_from_call_curve_BL(strikes, C_clean, r, T)

# ============================================================
# Fit LWM
# ============================================================

fit, chosen_spec = evolutionary_lwm_fit(
    K=strikes,
    C_mkt=C_noisy,
    S0=S0,
    r=r,
    T=T,
    M_max=3,
    penalty_lambda=0.0,
    random_starts=1,
    seed=42,
    var_c=0.1,
    var_penalty=1e4,
    improvement_tol=1e-4,
    metric="loss",
    use_wald=True,
    wald_alpha=0.05,
    wald_p=1,
    wald_q=1,
    weights=None
)

M1 = chosen_spec.n_lognormal
M  = chosen_spec.n_weibull + M1

wald_val = getattr(fit, "wald_stat", np.nan)
SSE = float(np.sum(fit.eps**2))

print(f"Chosen spec: M1={M1}, total M={M}")
print(f"Wald={wald_val:.3f}, SSE={SSE:.3f}")

# ============================================================
#             Create 2×1 Diagnostic Figure 2.3
# ============================================================
##The residuals should look approximately like white noise. If so, the null hypothesis is not rejected. If the null hypotheesis is rejected than there exists
##unexplained structure which means more mixtures are needed.
fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex=False)

# ------------------------------------------------------------
#  Top Panel: True RND vs Fitted RND
# ------------------------------------------------------------
ax = axes[0]

ax.plot(strikes, true_rnd,
        lw=3, color="black", label="True RND")

ax.plot(strikes, fit.qhat(strikes),
        lw=2, color="red", alpha=0.85, label="Fitted LWM RND")

ax.set_title("True vs Fitted RND", fontsize=14)
ax.set_xlabel("Strike", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.grid(alpha=0.25)
ax.legend()

# ------------------------------------------------------------
#  Bottom Panel: Residuals + Fourier Regression Fit
# ------------------------------------------------------------
ax = axes[1]

# residual scatter
ax.scatter(strikes, fit.eps,
           facecolors="none",
           edgecolors="k",
           s=30,
           label="Residuals")

# Fourier regression fit
ax.plot(strikes, fit.eps_fit,
        lw=2.0,
        color="red",
        label="Fourier Regression Fit")

# zero line
ax.axhline(0.0, linestyle="--", color="gray", lw=1)

# Title (LaTeX style)
title = (
    rf"$M={M},\ M_1={M1},\ "
    rf"\mathrm{{Wald}}_{{1,1}}={wald_val:.3f},\ "
    rf"\mathrm{{SSE}}={SSE:.3f}$"
)
ax.set_title(title, fontsize=13)

ax.set_xlabel("Strike", fontsize=12)
ax.set_ylabel("Residuals", fontsize=12)
ax.grid(alpha=0.25)
ax.legend()

plt.tight_layout()
plt.show()