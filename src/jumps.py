"""
Jump simulation for the affine rough-Volterra variance model.

Custom — Context7 found no library equivalent for this specific jump structure.

The jump measure is:
    nu_0(de) = beta * N(0,1)(de)   (background driving intensity)
    nu_i = 0 for i=1,2             (no state-dependent jumps)

Jump size function: eta^i(e) = kappa * e * 1_{e > 0} for each component i.

Per-interval jump contribution:
    J_{k,l} = sum_{m: arrival in (t_{l-1}, t_l]} eta(e_m)

where arrival counts are Poisson(beta * Delta) and marks e_m ~ N(0,1).
"""

import numpy as np
from typing import Tuple


def simulate_jump_contributions(
    n: int,
    T: float,
    beta: float,
    kappa: float,
    d: int = 2,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simulate jump contributions J[l, i] for each interval l=0,...,n-1 and
    dimension i=0,...,d-1.

    J[l, i] = sum_{m: arrival in interval l} kappa * e_m * 1_{e_m > 0}

    Parameters
    ----------
    n : int
        Number of time steps.
    T : float
        Horizon.
    beta : float
        Poisson arrival intensity.
    kappa : float
        Jump amplitude scaling.
    d : int
        Number of variance components (both share the same jump).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n, d) array of jump contributions per interval per component.
        All components receive identical jump realizations (shared background).
    """
    if rng is None:
        rng = np.random.default_rng()

    Delta = T / n
    # Per-interval Poisson counts
    counts = rng.poisson(lam=beta * Delta, size=n)   # shape (n,)

    J = np.zeros((n, d))
    for l in range(n):
        nc = counts[l]
        if nc > 0:
            marks = rng.standard_normal(size=nc)      # e_m ~ N(0,1)
            # eta(e) = kappa * e * 1_{e>0}
            eta_vals = kappa * marks * (marks > 0)
            jump_sum = np.sum(eta_vals)               # scalar (shared across dims)
            J[l, :] = jump_sum                        # identical for each component
    return J


def simulate_jumps_and_marks(
    n: int,
    T: float,
    beta: float,
    kappa: float,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return full arrival and mark arrays for diagnostic purposes.

    Returns
    -------
    arrivals : np.ndarray
        Shape (n,) integer count of jump arrivals per interval.
    eta_sums : np.ndarray
        Shape (n,) of sum_{m in interval} eta(e_m).
    """
    if rng is None:
        rng = np.random.default_rng()

    Delta = T / n
    arrivals = rng.poisson(lam=beta * Delta, size=n)
    eta_sums = np.zeros(n)
    for l in range(n):
        nc = arrivals[l]
        if nc > 0:
            marks = rng.standard_normal(nc)
            eta_sums[l] = np.sum(kappa * marks * (marks > 0))
    return arrivals, eta_sums


def h_gamma(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Scaled cumulant generating function: h_gamma(x) = (exp(gamma*x) - gamma*x - 1) / gamma.

    Parameters
    ----------
    x : np.ndarray
        Argument.
    gamma : float
        Risk aversion parameter.

    Returns
    -------
    np.ndarray
        h_gamma values.
    """
    x = np.asarray(x, dtype=float)
    return (np.exp(gamma * x) - gamma * x - 1.0) / gamma


def h(x: np.ndarray) -> np.ndarray:
    """
    Cumulant function: h(x) = exp(x) - x - 1.

    Parameters
    ----------
    x : np.ndarray
        Argument.

    Returns
    -------
    np.ndarray
        h values.
    """
    x = np.asarray(x, dtype=float)
    return np.exp(x) - x - 1.0


def jump_integral_exp_utility(
    psi: np.ndarray,
    beta: float,
    kappa: float,
    gamma: float,
    n_quad: int = 200,
) -> float:
    """
    Compute int_E h_gamma(psi^T eta(e)) nu_0(de) for exponential utility.

    With eta(e) = kappa * e * 1_{e>0} shared across both components,
    psi^T eta(e) = (psi_1 + psi_2) * kappa * e * 1_{e>0}.

    Uses Gauss-Hermite quadrature adapted to standard normal density on [0, inf).

    Parameters
    ----------
    psi : np.ndarray
        Shape (d,) current Riccati solution.
    beta : float
        Jump intensity.
    kappa : float
        Jump amplitude.
    gamma : float
        Risk aversion.
    n_quad : int
        Quadrature points.

    Returns
    -------
    float
        Scalar integral value.
    """
    from scipy import integrate as sci_int

    psi_sum = float(np.sum(psi))

    def integrand(e: float) -> float:
        eta_e = kappa * e  # e > 0 enforced by integration limits
        arg = psi_sum * eta_e
        return h_gamma(np.array([arg]), gamma)[0] * beta * np.exp(-0.5 * e ** 2) / np.sqrt(2.0 * np.pi)

    result, _ = sci_int.quad(integrand, 0.0, np.inf, limit=100)
    return result


def jump_integral_power_utility(
    psi: np.ndarray,
    beta: float,
    kappa: float,
    n_quad: int = 200,
) -> float:
    """
    Compute int_E h(psi^T eta(e)) nu_0(de) for power utility.

    Parameters
    ----------
    psi : np.ndarray
        Shape (d,) current Riccati solution.
    beta : float
        Jump intensity.
    kappa : float
        Jump amplitude.
    n_quad : int
        Quadrature points (unused, kept for API consistency).

    Returns
    -------
    float
        Scalar integral value.
    """
    from scipy import integrate as sci_int

    psi_sum = float(np.sum(psi))

    def integrand(e: float) -> float:
        eta_e = kappa * e
        arg = psi_sum * eta_e
        return h(np.array([arg]))[0] * beta * np.exp(-0.5 * e ** 2) / np.sqrt(2.0 * np.pi)

    result, _ = sci_int.quad(integrand, 0.0, np.inf, limit=100)
    return result
