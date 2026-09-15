"""
Simulation of the Gaussian Volterra integral increments G^i_{k,l}.

Custom — Context7 found no library equivalent for this kernel-weighted
Brownian integral discretization.

Two approaches are implemented:
1. Approximation (primary): G^i_{k,l} ≈ K_i(t_k - t_{l-1/2}) * Delta_W_l
   where t_{l-1/2} = (t_{l-1} + t_l) / 2. This is an O(Delta^alpha) approximation
   used as the main engine because the exact covariance Cholesky approach is
   O(n^3) and impractical for n=200 paths.

2. Left-point approximation (simpler): G^i_{k,l} ≈ K_i(t_k - t_{l-1}) * Delta_W_l
   Used when t_k = t_l (i.e., l=k), kernel evaluated at right endpoint.

The approximation label is preserved in all calling code comments.

Reference: Semi-integrated Euler-Maruyama scheme, paper equation (63).
"""

import numpy as np
from scipy.special import gamma as gamma_func
from src.kernels import kernel, volterra_weights


def simulate_brownian_increments(
    n: int,
    T: float,
    d: int = 2,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simulate standard Brownian increments Delta_W_l for l=1,...,n.

    Parameters
    ----------
    n : int
        Number of time steps.
    T : float
        Horizon.
    d : int
        Dimension of the Brownian motion.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Shape (n, d); each entry ~ N(0, Delta).
    """
    if rng is None:
        rng = np.random.default_rng()
    Delta = T / n
    return rng.standard_normal(size=(n, d)) * np.sqrt(Delta)


def build_gaussian_volterra_matrix(
    dW: np.ndarray,
    T: float,
    alpha: float,
    midpoint: bool = True,
) -> np.ndarray:
    """
    Build the matrix G[k, l] ≈ K_alpha(t_k - t_l^*) * dW_l for l=1,...,k.

    This is an approximation to the exact stochastic Volterra integral
    G^i_{k,l} = int_{t_{l-1}}^{t_l} K_alpha(t_k - s) dW_s.

    Approximation label: interval-kernel approximation (not exact covariance).

    Parameters
    ----------
    dW : np.ndarray
        Shape (n,) Brownian increments for a single dimension.
    T : float
        Horizon.
    alpha : float
        Kernel exponent.
    midpoint : bool
        If True, evaluate kernel at interval midpoint; else at left endpoint.

    Returns
    -------
    np.ndarray
        Lower-triangular matrix G of shape (n, n); G[k, l] = 0 for l > k.
    """
    n = len(dW)
    Delta = T / n
    G = np.zeros((n, n))

    for k in range(n):
        t_k = (k + 1) * Delta        # t_{k+1} in 1-indexed; grid point for step k
        for l in range(k + 1):       # l = 0, ..., k (0-indexed)
            t_l_left  = l * Delta    # t_{l-1} in 1-indexed = t_l in 0-indexed
            t_l_mid   = (l + 0.5) * Delta
            t_eval = t_l_mid if midpoint else t_l_left
            tau = t_k - t_eval
            if tau > 0:
                G[k, l] = kernel(np.array([tau]), alpha)[0] * dW[l]
            elif tau == 0:
                # Diagonal: t_k = t_l_mid => kernel is singular; use weight C_{k,l} * dW_l / Delta
                # Fall back to: use integral weight * dW_l
                w = volterra_weights(k + 1, n, T, alpha)
                G[k, l] = w[l] * dW[l] / Delta  # kernel * Delta / Delta = kernel
                G[k, l] = kernel(np.array([t_k - t_l_left + 1e-10]), alpha)[0] * dW[l]
            else:
                G[k, l] = 0.0
    return G


def build_gaussian_volterra_fast(
    dW: np.ndarray,
    T: float,
    alpha: float,
) -> np.ndarray:
    """
    Vectorized construction of the lower-triangular G matrix.

    Uses midpoint kernel evaluation as the approximation.

    Parameters
    ----------
    dW : np.ndarray
        Shape (n,) Brownian increments.
    T : float
        Horizon.
    alpha : float
        Kernel exponent.

    Returns
    -------
    np.ndarray
        Lower-triangular G matrix of shape (n, n).
    """
    n = len(dW)
    Delta = T / n

    # t_k for k = 1, ..., n (1-indexed, shape (n,))
    t_k_vals = (np.arange(1, n + 1)) * Delta          # shape (n,)
    # midpoint of interval l: l*Delta + Delta/2, l=0,...,n-1
    t_l_mid  = (np.arange(n) + 0.5) * Delta           # shape (n,)

    # tau[k, l] = t_{k+1} - t_l_mid; shape (n, n)
    tau = t_k_vals[:, None] - t_l_mid[None, :]        # broadcast

    # Mask causal structure
    causal = tau > 0
    tau_safe = np.where(causal, tau, 1.0)              # avoid log/pow errors

    K_vals = np.where(causal, tau_safe ** (alpha - 1) / gamma_func(alpha), 0.0)

    G = K_vals * dW[None, :]                           # broadcast dW across rows
    # Zero out upper triangle (l > k)
    mask = np.tril(np.ones((n, n), dtype=bool))
    G = G * mask
    return G
