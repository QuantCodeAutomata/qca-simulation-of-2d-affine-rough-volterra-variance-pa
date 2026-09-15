"""
Semi-closed-form value function evaluation for the 2D affine rough-Volterra model.

Custom — Context7 found no library equivalent for these specific value function
formulas (paper equations 41, 56, 62).

Three utility classes:
  - Exponential: V(x, V, t) = Theta_0 * exp(integral in exponent)
  - Power:       V(x, V, t) = Theta_0 * exp(integral in exponent)
  - Logarithmic: V(x, V, t) = log(x_0) + r*T + theta^2/2 * integral

Key ingredients:
  - g_0^i(t) = V_0^i + mu_0^i * t^{alpha_i} / Gamma(alpha_i + 1)  [under varphi=I, mu=const]
  - psi(t) from Riccati solution
  - Jump integrals against nu_0 (nonzero when beta > 0)
  - Compensator integrals

Notes on missing parameters (documented):
  - zeta: defaulted to zeros (no-liability interpretation)
  - x_0:  defaulted to 1.0 (scale normalization)
  - beta, kappa: must be provided by caller
"""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy import integrate as sci_int
from typing import Optional

from src.riccati import solve_riccati_volterra
from src.jumps import h_gamma, h, jump_integral_exp_utility, jump_integral_power_utility


def compute_g0(
    t: np.ndarray,
    V0: np.ndarray,
    mu0: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Compute deterministic adjusted forward input g_0(t).

    g_0^i(t) = V_0^i + mu_0^i * t^{alpha_i} / Gamma(alpha_i + 1)

    This follows from the convolution of the power kernel with the constant drift
    under varphi = I.

    Parameters
    ----------
    t : np.ndarray
        Shape (n_t,) time points.
    V0 : np.ndarray
        Shape (d,) initial variance.
    mu0 : np.ndarray
        Shape (d,) constant drift.
    alpha : np.ndarray
        Shape (d,) kernel exponents.

    Returns
    -------
    np.ndarray
        Shape (n_t, d) adjusted forward input.
    """
    d = len(V0)
    t = np.asarray(t, dtype=float)
    g = np.zeros((len(t), d))
    for i in range(d):
        g[:, i] = V0[i] + mu0[i] * t ** alpha[i] / gamma_func(alpha[i] + 1)
    return g


def evaluate_exponential_value(
    x0: float,
    T: float,
    n: int,
    V0: np.ndarray,
    mu0: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    sigma_v: np.ndarray,
    lam: np.ndarray,
    gamma: float,
    r: float,
    beta: float,
    kappa: float,
    zeta: Optional[np.ndarray] = None,
) -> dict:
    """
    Evaluate the exponential utility value function (paper eq. 41).

    V^{exp}(x_0, V_0, 0) = Theta_0 * exp(exponent_integral)

    where Theta_0 = -(1/gamma) * exp(-gamma * exp(r*T) * x_0)
    and the exponent contains:
      gamma * int_0^T [jump_integral + sum_i (a_i^{exp} + F_i(s, psi)) * g_0^i(s)] ds

    Parameters
    ----------
    (see module docstring and config.py for parameter descriptions)

    Returns
    -------
    dict
        Value estimate and diagnostic quantities.
    """
    d = len(V0)
    if zeta is None:
        zeta = np.zeros(d)

    Delta = T / n
    time_grid = np.arange(n + 1) * Delta   # t in [0, T]

    # Solve Riccati system (psi indexed forward in tau = T - t)
    psi = solve_riccati_volterra(
        n=n, T=T, alpha=alpha, theta=theta, rho=rho,
        sigma_v=sigma_v, lam=lam, gamma=gamma,
        utility='exponential', zeta=zeta,
    )

    # g_0(t) at each grid point
    g0 = compute_g0(time_grid, V0, mu0, alpha)   # shape (n+1, d)

    # Precompute rhs values at each t in [0, T]
    from src.riccati import _rhs_exponential
    integrand_vals = np.zeros(n + 1)
    for j in range(n + 1):
        tau_idx = n - j
        psi_j = psi[tau_idx]

        # Jump integral: int_E h_gamma(psi^T eta(e)) nu_0(de)
        ji = jump_integral_exp_utility(psi_j, beta, kappa, gamma) if beta > 0 else 0.0

        # sum_i (a_i + F_i) * g0_i
        rhs_j = _rhs_exponential(psi_j, j * Delta, theta, rho, sigma_v, lam, gamma, zeta)
        state_sum = float(np.dot(rhs_j, g0[j]))

        integrand_vals[j] = ji + state_sum

    # Integrate over [0, T] using trapezoid rule
    exponent = gamma * np.trapz(integrand_vals, time_grid)

    # Base factor Theta_0
    Theta_0 = -(1.0 / gamma) * np.exp(-gamma * np.exp(r * T) * x0)

    value = Theta_0 * np.exp(exponent)

    return {
        'value': float(value),
        'Theta_0': float(Theta_0),
        'exponent': float(exponent),
        'integrand_max': float(np.max(np.abs(integrand_vals))),
        'psi_range': [float(psi.min()), float(psi.max())],
    }


def evaluate_power_value(
    x0: float,
    T: float,
    n: int,
    V0: np.ndarray,
    mu0: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    sigma_v: np.ndarray,
    lam: np.ndarray,
    gamma: float,
    r: float,
    beta: float,
    kappa: float,
) -> dict:
    """
    Evaluate the power utility value function (paper eq. 56).

    V^{pow}(x_0, V_0, 0) = Theta_0 * exp(exponent_integral)

    where Theta_0 = x_0^gamma / gamma * exp(gamma * r * T)
    and the exponent contains:
      int_0^T [jump_integral + sum_i (a_i^{pow} + F_i) * g_0^i(s)] ds

    Returns
    -------
    dict
        Value estimate and diagnostics.
    """
    d = len(V0)
    Delta = T / n
    time_grid = np.arange(n + 1) * Delta

    psi = solve_riccati_volterra(
        n=n, T=T, alpha=alpha, theta=theta, rho=rho,
        sigma_v=sigma_v, lam=lam, gamma=gamma,
        utility='power',
    )

    g0 = compute_g0(time_grid, V0, mu0, alpha)

    from src.riccati import _rhs_power
    integrand_vals = np.zeros(n + 1)
    for j in range(n + 1):
        tau_idx = n - j
        psi_j = psi[tau_idx]

        ji = jump_integral_power_utility(psi_j, beta, kappa) if beta > 0 else 0.0

        rhs_j = _rhs_power(psi_j, j * Delta, theta, rho, sigma_v, lam, gamma)
        state_sum = float(np.dot(rhs_j, g0[j]))

        integrand_vals[j] = ji + state_sum

    exponent = np.trapz(integrand_vals, time_grid)

    Theta_0 = x0 ** gamma / gamma * np.exp(gamma * r * T)
    value = Theta_0 * np.exp(exponent)

    return {
        'value': float(value),
        'Theta_0': float(Theta_0),
        'exponent': float(exponent),
        'integrand_max': float(np.max(np.abs(integrand_vals))),
        'psi_range': [float(psi.min()), float(psi.max())],
    }


def evaluate_log_value(
    x0: float,
    T: float,
    n: int,
    V0: np.ndarray,
    mu0: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    r: float,
    lam: Optional[np.ndarray] = None,
) -> dict:
    """
    Evaluate the log utility value function (paper eq. 62).

    V^{log}(x_0, V_0, 0) = log(x_0) + r*T + (theta o theta)^T / 2
                             * int_0^T (I - int_0^{T-s} R_D(u) du) g_0(s) ds

    For diagonal D = diag(-lambda_i), the resolvent R_D is computed componentwise.
    R_D^i(u) = exp(-lambda_i * u) for the standard Volterra convolution resolvent
    of K_i * D^i. However, for fractional kernels this is more complex; we use
    a numerical discretization of the Volterra resolvent equation.

    For simplicity and to avoid the resolvent complexity, we use the simplified
    approximation: int_0^{T-s} R_D^i(u) du ≈ int_0^{T-s} exp(-lam_i * u) du
    = (1 - exp(-lam_i * (T-s))) / lam_i. This is exact for standard Ornstein-
    Uhlenbeck (alpha=1) and serves as an approximation for fractional kernels.

    Implementation note: The exact resolvent for fractional kernels requires
    solving a Volterra integral equation of the second kind; we use the
    exponential decay approximation and flag this as a documented approximation.

    Returns
    -------
    dict
        Value estimate and diagnostics.
    """
    d = len(V0)
    Delta = T / n
    time_grid = np.arange(n + 1) * Delta

    if lam is None:
        lam = np.zeros(d)   # resolvent integral = 0 → factor = 1

    g0 = compute_g0(time_grid, V0, mu0, alpha)   # shape (n+1, d)

    # Compute (I - int_0^{T-s} R_D(u) du) for each s in the grid
    # Result shape: (n+1, d)
    integrand_vals = np.zeros((n + 1, d))
    for j in range(n + 1):
        tau = T - time_grid[j]   # T - s
        for i in range(d):
            if lam[i] > 0 and tau > 0:
                # Approximate resolvent integral: int_0^tau exp(-lam_i * u) du
                resolvent_int = (1.0 - np.exp(-lam[i] * tau)) / lam[i]
            else:
                resolvent_int = 0.0
            # Factor: (1 - resolvent_int) * g0[j, i]
            integrand_vals[j, i] = (1.0 - resolvent_int) * g0[j, i]

    # Integrate: int_0^T (I - R_D integral) g_0(s) ds per component
    integrals = np.trapz(integrand_vals, time_grid, axis=0)   # shape (d,)

    # theta o theta = theta^2 componentwise
    theta_sq = theta ** 2
    value_addition = 0.5 * float(np.dot(theta_sq, integrals))

    log_value = np.log(x0) + r * T + value_addition

    return {
        'value': float(log_value),
        'log_x0': float(np.log(x0)),
        'rT': float(r * T),
        'value_addition': float(value_addition),
        'integrals_per_component': integrals.tolist(),
    }


def sensitivity_analysis(
    x0: float,
    T: float,
    n: int,
    V0: np.ndarray,
    mu0: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    sigma_v: np.ndarray,
    lam: np.ndarray,
    gamma: float,
    r: float,
    beta_grid: list,
    kappa_grid: list,
    utility: str = 'power',
) -> dict:
    """
    Sensitivity analysis of value function over beta and kappa grids.

    Returns
    -------
    dict
        Grid of value estimates indexed by (beta, kappa).
    """
    results = {}
    for beta in beta_grid:
        for kappa in kappa_grid:
            key = (round(beta, 4), round(kappa, 4))
            try:
                if utility == 'power':
                    res = evaluate_power_value(
                        x0, T, n, V0, mu0, alpha, theta, rho,
                        sigma_v, lam, gamma, r, beta, kappa,
                    )
                else:
                    res = evaluate_exponential_value(
                        x0, T, n, V0, mu0, alpha, theta, rho,
                        sigma_v, lam, gamma, r, beta, kappa,
                    )
                results[key] = res['value']
            except Exception as e:
                results[key] = f"Error: {e}"
    return results
