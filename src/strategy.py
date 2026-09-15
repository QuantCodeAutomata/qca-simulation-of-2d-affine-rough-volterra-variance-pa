"""
Deterministic optimal strategy curves for the 2D affine rough-Volterra model.

Custom — Context7 found no library equivalent for these Riccati-based strategy
formulas (paper Section 3 and Figure 3).

Optimal strategy formulas (variance-normalized, deterministic):
    pi_{i,t}^* = alpha_{i,t}^* / sqrt(V_t^i)

Exponential utility:
    pi_{i,t}^{exp,*} = exp(-r*(T-t)) / gamma * [theta_i + gamma*rho_i*sigma_v_i*psi^i(T-t)]

Power utility:
    pi_{i,t}^{pow,*} = [theta_i + rho_i*sigma_v_i*psi^i(T-t)] / (1 - gamma)

These are deterministic functions of time given psi from the Riccati solution.
"""

import numpy as np
from typing import Literal
from src.riccati import solve_riccati_volterra


def compute_strategy_curves(
    n: int,
    T: float,
    alpha: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    sigma_v: np.ndarray,
    lam: np.ndarray,
    gamma: float,
    r: float = 0.02,
    utility: Literal['exponential', 'power'] = 'power',
    zeta: np.ndarray = None,
) -> tuple:
    """
    Compute deterministic optimal strategy curves pi_{i,t}^* for t in [0, T].

    Parameters
    ----------
    n : int
        Number of time steps.
    T : float
        Horizon.
    alpha : np.ndarray
        Shape (d,) kernel exponents.
    theta : np.ndarray
        Shape (d,) market price of risk.
    rho : np.ndarray
        Shape (d,) leverage/correlation.
    sigma_v : np.ndarray
        Shape (d,) volatility-of-volatility.
    lam : np.ndarray
        Shape (d,) mean-reversion magnitudes.
    gamma : float
        Risk aversion.
    r : float
        Risk-free rate.
    utility : str
        'exponential' or 'power'.
    zeta : np.ndarray, optional
        Liability parameter for exponential utility.

    Returns
    -------
    time_grid : np.ndarray
        Shape (n+1,) time grid [0, T].
    pi : np.ndarray
        Shape (n+1, d) strategy curves.
    psi : np.ndarray
        Shape (n+1, d) Riccati solution.
    """
    d = len(alpha)
    if zeta is None:
        zeta = np.zeros(d)

    # Solve Riccati-Volterra: psi[k] = psi(tau_k) with tau = T - t
    psi = solve_riccati_volterra(
        n=n, T=T, alpha=alpha, theta=theta, rho=rho,
        sigma_v=sigma_v, lam=lam, gamma=gamma,
        utility=utility, zeta=zeta,
    )
    # psi[k] corresponds to tau_k = k*Delta => t_k = T - tau_k
    # So psi as function of t: psi(T-t_k) = psi[k]
    # Strategy at time t_k: need psi(T - t_k) = psi[k], but grid is in tau order
    # Time grid in t: t_k = k * Delta, k=0,...,n
    # tau grid: tau_k = T - t_k = (n-k)*Delta... but psi is computed forward in tau
    # so psi[k] = psi at tau = k*Delta = T - t_{n-k}
    # To get pi(t), we need psi(T-t) = psi at tau=T-t
    # psi array: psi[k] at tau_k=k*Delta; map to t_k=T-tau_k=(n-k)*Delta
    # => pi[j] for t_j=j*Delta: psi(T-t_j)=psi(j*(T/n)*...wait
    # psi[k] at tau_k = k*Delta; t = T - tau => at t = t_j = j*Delta:
    #   T - t_j = (n-j)*Delta => tau_{n-j}=(n-j)*Delta => psi[n-j]
    # So pi at index j uses psi[n-j] (time reversal)

    Delta = T / n
    time_grid = np.arange(n + 1) * Delta   # t_j = j*Delta

    pi = np.zeros((n + 1, d))
    for j in range(n + 1):
        tau_idx = n - j   # index into psi array for time T - t_j
        psi_val = psi[tau_idx]   # shape (d,)

        if utility == 'exponential':
            disc = np.exp(-r * (T - time_grid[j]))
            pi[j] = disc / gamma * (theta + gamma * rho * sigma_v * psi_val)
        else:  # power
            pi[j] = (theta + rho * sigma_v * psi_val) / (1.0 - gamma)

    # Validate finiteness
    assert np.all(np.isfinite(pi)), "Non-finite strategy values detected."

    return time_grid, pi, psi


def compute_full_alpha_control(
    V_path: np.ndarray,
    time_grid: np.ndarray,
    psi: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    sigma_v: np.ndarray,
    gamma: float,
    r: float,
    T: float,
    utility: Literal['exponential', 'power'],
) -> np.ndarray:
    """
    Compute the full stochastic optimal control alpha_t^* from a simulated V path.

    alpha_{i,t}^* = pi_{i,t}^* * sqrt(V_t^i)

    Parameters
    ----------
    V_path : np.ndarray
        Shape (n+1, d) variance path.
    time_grid : np.ndarray
        Shape (n+1,) time grid.
    psi : np.ndarray
        Shape (n+1, d) Riccati solution (indexed forward in tau).
    theta, rho, sigma_v : np.ndarray
        Shape (d,) parameter arrays.
    gamma : float
        Risk aversion.
    r : float
        Risk-free rate.
    T : float
        Horizon.
    utility : str
        'exponential' or 'power'.

    Returns
    -------
    np.ndarray
        Shape (n+1, d) stochastic optimal control alpha_t^*.
    """
    n = len(time_grid) - 1
    d = V_path.shape[1]
    alpha_ctrl = np.zeros((n + 1, d))

    for j in range(n + 1):
        tau_idx = n - j
        psi_val = psi[tau_idx]
        V_j = np.maximum(V_path[j], 0.0)
        sqrt_V = np.sqrt(V_j)
        Lambda = sigma_v * psi_val * sqrt_V   # shape (d,)
        lambda_t = theta * sqrt_V             # shape (d,)

        if utility == 'exponential':
            disc = np.exp(-r * (T - time_grid[j]))
            alpha_ctrl[j] = disc * (lambda_t / gamma + rho * Lambda)
        else:  # power
            alpha_ctrl[j] = (lambda_t + rho * Lambda) / (1.0 - gamma)

    return alpha_ctrl
