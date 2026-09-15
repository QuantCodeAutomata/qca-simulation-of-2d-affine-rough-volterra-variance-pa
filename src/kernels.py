"""
Fractional power kernels and deterministic Volterra integration weights.

Custom — Context7 found no library equivalent (paper Section 4, eq. 63 scheme).

For the power kernel K_alpha(t) = t^{alpha-1} / Gamma(alpha), the integral
weights are available analytically:

    C_{k,l}^i = int_{t_{l-1}}^{t_l} K_{alpha_i}(t_k - s) ds
              = [(t_k - t_{l-1})^{alpha_i} - (t_k - t_l)^{alpha_i}] / Gamma(alpha_i + 1)

valid for k >= l (causal Volterra structure). C_{k,l} = 0 for k < l.

The Adams-Bashforth-Moulton (ABM) predictor-corrector weights for the
fractional Riccati ODE system are also computed here following the paper.
"""

import numpy as np
from scipy.special import gamma as gamma_func
from typing import Tuple


def kernel(t: np.ndarray, alpha: float) -> np.ndarray:
    """
    Evaluate the power kernel K_alpha(t) = t^{alpha-1} / Gamma(alpha).

    Parameters
    ----------
    t : np.ndarray
        Time values (must be positive).
    alpha : float
        Kernel exponent in (0, 1).

    Returns
    -------
    np.ndarray
        Kernel values; singular at t=0 when alpha < 1.
    """
    t = np.asarray(t, dtype=float)
    assert np.all(t >= 0), "Kernel requires non-negative time arguments."
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(t > 0, t ** (alpha - 1) / gamma_func(alpha), np.inf)
    return result


def volterra_weights(k: int, n: int, T: float, alpha: float) -> np.ndarray:
    """
    Compute the deterministic Volterra integration weights C_{k,l} for l=1,...,k.

    C_{k,l} = int_{t_{l-1}}^{t_l} K_alpha(t_k - s) ds
            = [(t_k - t_{l-1})^alpha - (t_k - t_l)^alpha] / Gamma(alpha + 1)

    Parameters
    ----------
    k : int
        Current time index (1-indexed for the sum).
    n : int
        Total number of grid steps.
    T : float
        Horizon.
    alpha : float
        Kernel exponent.

    Returns
    -------
    np.ndarray
        Array of shape (k,) giving C_{k,l} for l = 1, ..., k.
    """
    Delta = T / n
    t_k = k * Delta
    l_vals = np.arange(1, k + 1)              # l = 1, ..., k
    t_lm1 = (l_vals - 1) * Delta              # t_{l-1}
    t_l   = l_vals * Delta                    # t_l

    gap_left  = np.maximum(t_k - t_lm1, 0.0)  # t_k - t_{l-1} >= 0
    gap_right = np.maximum(t_k - t_l,   0.0)  # t_k - t_l >= 0 (0 when k==l)

    weights = (gap_left ** alpha - gap_right ** alpha) / gamma_func(alpha + 1)
    return weights


def build_weight_matrix(n: int, T: float, alpha: float) -> np.ndarray:
    """
    Build the full lower-triangular weight matrix C[k, l] for k, l = 0, ..., n-1.

    C[k, l] corresponds to C_{k+1, l+1} using 0-indexed arrays.
    Entry C[k, l] = 0 for l > k (causal structure).

    Parameters
    ----------
    n : int
        Total number of steps.
    T : float
        Horizon.
    alpha : float
        Kernel exponent.

    Returns
    -------
    np.ndarray
        Lower-triangular matrix of shape (n, n).
    """
    Delta = T / n
    C = np.zeros((n, n))
    for k in range(n):          # k = 0, ..., n-1 corresponds to time step k+1
        if k >= 0:
            w = volterra_weights(k + 1, n, T, alpha)
            C[k, :k + 1] = w
    return C


def abm_weights_predictor(k: int, Delta: float, alpha: float) -> np.ndarray:
    """
    Compute Adams-Bashforth predictor weights b_{j,k+1} for j=0,...,k.

    b_{j,k+1} = Delta^alpha / Gamma(alpha+1) * [(k+1-j)^alpha - (k-j)^alpha]

    Parameters
    ----------
    k : int
        Current step index (0-indexed).
    Delta : float
        Step size.
    alpha : float
        Fractional order.

    Returns
    -------
    np.ndarray
        Weights of shape (k+1,) for j = 0, ..., k.
    """
    j_vals = np.arange(0, k + 1, dtype=float)
    b = Delta ** alpha / gamma_func(alpha + 1) * ((k + 1 - j_vals) ** alpha
                                                   - (k - j_vals) ** alpha)
    return b


def abm_weights_corrector(k: int, Delta: float, alpha: float) -> Tuple[np.ndarray, float]:
    """
    Compute Adams-Moulton corrector weights a_{j,k+1} for j=0,...,k+1.

    The piecewise coefficients are:
        j = 0:       a_{0,k+1} = Delta^alpha/Gamma(alpha+2) * [k^{alpha+1} - (k-alpha)*(k+1)^alpha]
        1 <= j <= k: a_{j,k+1} = Delta^alpha/Gamma(alpha+2) * [(k-j+2)^{alpha+1}
                                  + (k-j)^{alpha+1} - 2*(k-j+1)^{alpha+1}]
        j = k+1:     a_{k+1,k+1} = Delta^alpha/Gamma(alpha+2) * 1

    Returns
    -------
    a_prev : np.ndarray
        Weights a_{j,k+1} for j=0,...,k, shape (k+1,).
    a_pred : float
        Weight a_{k+1,k+1} for the predictor step.
    """
    coeff = Delta ** alpha / gamma_func(alpha + 2)

    # j = 0
    if k == 0:
        a0 = coeff * 1.0  # special case: k^{alpha+1}=0, (k-alpha)*(k+1)^alpha with k=0 => -alpha*1
        a0 = coeff * (0.0 - (-alpha) * 1.0)
    else:
        a0 = coeff * (k ** (alpha + 1) - (k - alpha) * (k + 1) ** alpha)

    # j = 1, ..., k
    if k >= 1:
        j_vals = np.arange(1, k + 1, dtype=float)
        a_mid = coeff * ((k - j_vals + 2) ** (alpha + 1)
                         + (k - j_vals) ** (alpha + 1)
                         - 2.0 * (k - j_vals + 1) ** (alpha + 1))
    else:
        a_mid = np.array([])

    a_prev = np.concatenate([[a0], a_mid])  # shape (k+1,)

    # j = k+1 (predictor coefficient)
    a_pred = coeff * 1.0

    return a_prev, a_pred


def compensator_drift(beta: float, kappa: float) -> float:
    """
    Compute the compensator drift correction int eta(e) nu_0(de).

    With eta(e) = kappa * e * 1_{e>0}, e ~ N(0,1), and nu_0(de) = beta * phi(e) de:
        int_{-inf}^{inf} kappa * e * 1_{e>0} * beta * phi(e) de
      = beta * kappa * int_0^inf e * phi(e) de
      = beta * kappa / sqrt(2*pi)

    Parameters
    ----------
    beta : float
        Jump intensity.
    kappa : float
        Jump amplitude scaling.

    Returns
    -------
    float
        Scalar compensator correction applied to each variance component.
    """
    return beta * kappa / np.sqrt(2.0 * np.pi)
