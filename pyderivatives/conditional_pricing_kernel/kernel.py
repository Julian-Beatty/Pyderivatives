import numpy as np

def unpack_theta(theta: np.ndarray, N: int, Ksig: int) -> np.ndarray:
    theta = np.asarray(theta, float).ravel()
    return theta.reshape((Ksig + 1, N), order="F")

def c_it(sigma: float, theta_mat: np.ndarray) -> np.ndarray:
    Ksig = theta_mat.shape[0] - 1
    sig_pow = float(sigma) ** np.arange(Ksig + 1, dtype=float)
    return sig_pow @ theta_mat  # (N,)

def g_r_sigma(r: np.ndarray, sigma: float, theta: np.ndarray, *, N: int, Ksig: int) -> np.ndarray:
    theta_mat = unpack_theta(theta, N, Ksig)
    c = c_it(float(sigma), theta_mat)  # (N,)
    powers = np.vstack([r**i for i in range(1, N + 1)]).T
    return powers @ c
