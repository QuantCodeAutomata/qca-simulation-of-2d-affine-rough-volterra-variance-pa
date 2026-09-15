"""
Monte Carlo simulation of wealth dynamics under optimal and perturbed controls.

Custom — Context7 found no library equivalent for this specific stochastic
control problem on the rough-jump variance model.

Wealth dynamics (self-financing, transformed representation):
    dX_t = r*X_t*dt + alpha_t^T * [lambda_t*dt + dB_t]

where:
  - lambda_t = (theta_1*sqrt(V_t^1), theta_2*sqrt(V_t^2))^T
  - dB_t ~ N(0, I*dt) independent of variance-driving W

Discrete Euler scheme (left-point predictable control):
    X_{k+1} = X_k + r*X_k*Delta + alpha_k^T * lambda_k * Delta + alpha_k^T * dB_k
"""

import numpy as np
from typing import Literal, Optional, Tuple, List
from src.volterra_sim import simulate_variance_paths
from src.strategy import compute_full_alpha_control


def simulate_wealth_paths(
    M: int,
    n: int,
    T: float,
    x0: float,
    V0: np.ndarray,
    mu0: np.ndarray,
    lam: np.ndarray,
    sigma_v: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    gamma: float,
    r: float,
    beta: float,
    kappa: float,
    utility: Literal['exponential', 'power', 'log'],
    control_scale: float = 1.0,
    zeta: Optional[np.ndarray] = None,
    base_seed: int = 42,
    V_paths_precomputed: Optional[np.ndarray] = None,
    psi_precomputed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate M paths of terminal wealth under a scaled version of the
    candidate optimal control.

    control_scale = 1.0 gives the candidate optimal control.
    control_scale != 1.0 gives a perturbed control alpha_t^{(c)} = c * alpha_t^*.

    Parameters
    ----------
    M : int
        Number of Monte Carlo paths.
    n : int
        Time steps.
    T : float
        Horizon.
    x0 : float
        Initial wealth.
    V0, mu0, lam, sigma_v, alpha, theta, rho : np.ndarray
        Model parameters.
    gamma : float
        Risk aversion.
    r : float
        Risk-free rate.
    beta, kappa : float
        Jump parameters.
    utility : str
        Utility class for Riccati solution.
    control_scale : float
        Multiplier c for alpha_t^{(c)} = c * alpha_t^*.
    zeta : np.ndarray, optional
        Liability vector for exponential utility.
    base_seed : int
        Base random seed.
    V_paths_precomputed : np.ndarray, optional
        Pre-simulated variance paths of shape (M, n+1, d). If supplied,
        variance simulation is skipped (major speedup for perturbation sweeps).
    psi_precomputed : np.ndarray, optional
        Pre-solved Riccati array of shape (n+1, d). If supplied, Riccati
        solve is skipped.

    Returns
    -------
    X_T : np.ndarray
        Shape (M,) terminal wealth values.
    V_paths : np.ndarray
        Shape (M, n+1, d) variance paths.
    """
    d = len(V0)
    if zeta is None:
        zeta = np.zeros(d)

    # Solve Riccati once (deterministic) — or reuse pre-computed solution
    from src.riccati import solve_riccati_volterra
    if psi_precomputed is not None:
        psi = psi_precomputed
    elif utility in ('exponential', 'power'):
        psi = solve_riccati_volterra(
            n=n, T=T, alpha=alpha, theta=theta, rho=rho,
            sigma_v=sigma_v, lam=lam, gamma=gamma,
            utility=utility, zeta=zeta,
        )
    else:
        psi = np.zeros((n + 1, d))  # log utility: psi not needed for lambda_t term

    Delta = T / n
    time_grid = np.arange(n + 1) * Delta

    # Simulate variance paths — or reuse pre-computed paths
    if V_paths_precomputed is not None:
        V_paths = V_paths_precomputed
    else:
        V_paths, _ = simulate_variance_paths(
            n=n, T=T, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v, alpha=alpha,
            beta=beta, kappa=kappa, n_paths=M, enable_jumps=(beta > 0),
            rng_seed=base_seed,
        )

    X_T = np.zeros(M)

    for m in range(M):
        V_path = V_paths[m]   # shape (n+1, d)
        rng = np.random.default_rng(base_seed + 10000 + m)

        # Simulate asset-return Brownian dB (independent of variance W)
        dB = rng.standard_normal(size=(n, d)) * np.sqrt(Delta)  # shape (n, d)

        # Compute optimal control along this path
        alpha_ctrl = compute_full_alpha_control(
            V_path=V_path, time_grid=time_grid,
            psi=psi, theta=theta, rho=rho, sigma_v=sigma_v,
            gamma=gamma, r=r, T=T, utility=utility if utility != 'log' else 'power',
        )

        if utility == 'log':
            # Log optimal: alpha_t^* = lambda_t = theta * sqrt(V_t)
            alpha_ctrl = np.zeros_like(alpha_ctrl)
            for j in range(n + 1):
                alpha_ctrl[j] = theta * np.sqrt(np.maximum(V_path[j], 0.0))

        # Scale control
        alpha_ctrl = control_scale * alpha_ctrl

        # Simulate wealth
        X = x0
        for k in range(n):
            V_k = np.maximum(V_path[k], 0.0)
            sqrt_V_k = np.sqrt(V_k)
            lambda_k = theta * sqrt_V_k              # shape (d,)
            alpha_k = alpha_ctrl[k]                  # shape (d,)

            dX = (r * X * Delta
                  + np.dot(alpha_k, lambda_k) * Delta
                  + np.dot(alpha_k, dB[k]))
            X = X + dX

        X_T[m] = X

    return X_T, V_paths


def compute_expected_utility(
    X_T: np.ndarray,
    gamma: float,
    utility: Literal['exponential', 'power', 'log'],
    V_paths: Optional[np.ndarray] = None,
    zeta: Optional[np.ndarray] = None,
    time_grid: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """
    Estimate expected utility and standard error from terminal wealth.

    Parameters
    ----------
    X_T : np.ndarray
        Shape (M,) terminal wealth.
    gamma : float
        Risk aversion.
    utility : str
        Utility class.
    V_paths : np.ndarray, optional
        Shape (M, n+1, d) variance paths for computing liability Xi.
    zeta : np.ndarray, optional
        Liability vector for exponential utility.
    time_grid : np.ndarray, optional
        Shape (n+1,) for integrating Xi.

    Returns
    -------
    mean_util : float
        Sample mean expected utility.
    std_err : float
        Standard error of the mean.
    ci_low : float
        95% CI lower bound.
    ci_high : float
        95% CI upper bound.
    """
    M = len(X_T)

    if utility == 'exponential':
        if zeta is not None and V_paths is not None and time_grid is not None:
            # Compute liability Xi = int_0^T zeta^T V_s ds per path
            Xi = np.trapz(
                np.einsum('mki,i->mk', V_paths, zeta), time_grid, axis=1
            )  # shape (M,)
        else:
            Xi = np.zeros(M)

        with np.errstate(over='ignore'):
            utils = -(1.0 / gamma) * np.exp(-gamma * (X_T - Xi))
    elif utility == 'power':
        pos_mask = X_T > 0
        utils = np.where(pos_mask, X_T ** gamma / gamma, -1e10)
    elif utility == 'log':
        pos_mask = X_T > 0
        utils = np.where(pos_mask, np.log(X_T), -1e10)
    else:
        raise ValueError(f"Unknown utility: {utility}")

    mean_util = float(np.mean(utils))
    std_err = float(np.std(utils, ddof=1) / np.sqrt(M))
    ci_low = mean_util - 1.96 * std_err
    ci_high = mean_util + 1.96 * std_err

    return mean_util, std_err, ci_low, ci_high


def dominance_test(
    M: int,
    n: int,
    T: float,
    x0: float,
    V0: np.ndarray,
    mu0: np.ndarray,
    lam: np.ndarray,
    sigma_v: np.ndarray,
    alpha: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    gamma: float,
    r: float,
    beta: float,
    kappa: float,
    utility: Literal['exponential', 'power', 'log'],
    perturbations: List[float] = None,
    base_seed: int = 42,
) -> dict:
    """
    Test local dominance: compare expected utility of optimal vs perturbed controls.

    Variance paths and the Riccati solution are computed once and shared
    across all perturbation scales, which gives a ~len(perturbations)x speedup.

    Parameters
    ----------
    perturbations : list
        List of control scales c to test (e.g., [0.8, 0.9, 1.0, 1.1, 1.2]).

    Returns
    -------
    dict
        Dominance results per perturbation scale.
    """
    if perturbations is None:
        perturbations = [0.8, 0.9, 1.0, 1.1, 1.2]

    d = len(V0)
    zeta = np.zeros(d)

    # --- Shared pre-computation (done once, not per perturbation) ---
    # 1. Solve Riccati
    from src.riccati import solve_riccati_volterra
    if utility in ('exponential', 'power'):
        psi = solve_riccati_volterra(
            n=n, T=T, alpha=alpha, theta=theta, rho=rho,
            sigma_v=sigma_v, lam=lam, gamma=gamma,
            utility=utility, zeta=zeta,
        )
    else:
        psi = np.zeros((n + 1, d))

    # 2. Simulate variance paths once
    V_paths, _ = simulate_variance_paths(
        n=n, T=T, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v, alpha=alpha,
        beta=beta, kappa=kappa, n_paths=M, enable_jumps=(beta > 0),
        rng_seed=base_seed,
    )

    results = {}
    for c in perturbations:
        X_T, _ = simulate_wealth_paths(
            M=M, n=n, T=T, x0=x0, V0=V0, mu0=mu0, lam=lam,
            sigma_v=sigma_v, alpha=alpha, theta=theta, rho=rho,
            gamma=gamma, r=r, beta=beta, kappa=kappa,
            utility=utility, control_scale=c, base_seed=base_seed,
            V_paths_precomputed=V_paths,
            psi_precomputed=psi,
        )
        mean_u, se, ci_l, ci_h = compute_expected_utility(X_T, gamma, utility)
        results[c] = {
            'mean_utility': mean_u,
            'std_error': se,
            'ci_low': ci_l,
            'ci_high': ci_h,
            'n_negative_wealth': int(np.sum(X_T <= 0)),
        }

    # Determine dominance
    opt_util = results[1.0]['mean_utility'] if 1.0 in results else None
    if opt_util is not None:
        for c, res in results.items():
            res['dominates_optimal'] = (res['mean_utility'] > opt_util) if c != 1.0 else None

    return results
