"""
Semi-integrated Euler-Maruyama discretization for the 2D affine rough-Volterra
variance process with exogenous upward jumps.

Custom — Context7 found no library equivalent (paper equation 63).

Scheme (componentwise, i=1,2):

    Vbar_{t_k}^{i,n} = V_0^i
        + sum_{l=1}^k (mu_0^i - lambda_i * Vbar_{t_{l-1}}^{i,n} - comp^i) * C^i_{k,l}
        + sum_{l=1}^k sigma_i^v * sqrt(max(Vbar_{t_{l-1}}^{i,n}, 0)) * G^i_{k,l}
        + sum_{l=1}^k J^i_{k,l}

where:
  - C^i_{k,l} = Volterra integration weights (from kernels.py)
  - G^i_{k,l} = approximated Gaussian Volterra integral (from gaussian_volterra.py)
  - J^i_{k,l} = jump contribution in interval [t_{l-1}, t_l] mapped into [0, t_k]
               via the kernel: J^i_{k,l} = J_l^i * C^i_{k,l}
  - comp^i    = beta * kappa / sqrt(2*pi) (compensator drift)

Positivity note: Full truncation max(V, 0) is applied before the sqrt. This is
a documented deviation from the paper, which does not specify a correction.
"""

import numpy as np
from typing import Optional, Tuple
from src.kernels import build_weight_matrix, compensator_drift
from src.jumps import simulate_jump_contributions
from src.gaussian_volterra import build_gaussian_volterra_fast


def simulate_variance_paths(
    n: int,
    T: float,
    V0: np.ndarray,
    mu0: np.ndarray,
    lam: np.ndarray,
    sigma_v: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    kappa: float,
    n_paths: int = 5,
    enable_jumps: bool = True,
    rng_seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate sample paths of the 2D affine rough-Volterra variance process.

    Parameters
    ----------
    n : int
        Number of time steps.
    T : float
        Horizon.
    V0 : np.ndarray
        Shape (d,) initial variance vector.
    mu0 : np.ndarray
        Shape (d,) drift vector.
    lam : np.ndarray
        Shape (d,) mean-reversion magnitudes (positive).
    sigma_v : np.ndarray
        Shape (d,) volatility-of-volatility vector.
    alpha : np.ndarray
        Shape (d,) kernel exponents.
    beta : float
        Jump intensity.
    kappa : float
        Jump amplitude scaling.
    n_paths : int
        Number of independent paths to simulate.
    enable_jumps : bool
        If False, sets beta=0 for the no-jump benchmark.
    rng_seed : int, optional
        Base random seed.

    Returns
    -------
    V : np.ndarray
        Shape (n_paths, n+1, d) variance paths. V[:, 0, :] = V0.
    time_grid : np.ndarray
        Shape (n+1,) uniform time grid.
    """
    d = len(V0)
    Delta = T / n
    time_grid = np.linspace(0.0, T, n + 1)

    if not enable_jumps:
        beta_eff = 0.0
    else:
        beta_eff = beta

    # Precompute weight matrices for each component (shape (n, n))
    C = [build_weight_matrix(n, T, float(alpha[i])) for i in range(d)]

    # Compensator drift per component
    comp = compensator_drift(beta_eff, kappa)  # scalar, same for both components

    # Allocate output
    V = np.zeros((n_paths, n + 1, d))
    V[:, 0, :] = V0[None, :]    # broadcast initial condition

    for path_idx in range(n_paths):
        rng = np.random.default_rng(rng_seed + path_idx if rng_seed is not None else None)

        # Simulate Brownian increments: shape (n, d)
        dW = rng.standard_normal(size=(n, d)) * np.sqrt(Delta)

        # Simulate jump contributions: shape (n, d)
        J = simulate_jump_contributions(n, T, beta_eff, kappa, d=d, rng=rng)

        # Build Gaussian Volterra matrices for each component
        G = [build_gaussian_volterra_fast(dW[:, i], T, float(alpha[i])) for i in range(d)]

        # Iteratively propagate variance
        Vpath = np.zeros((n + 1, d))
        Vpath[0] = V0

        for k in range(n):  # k = 0, ..., n-1; produce Vpath[k+1]
            for i in range(d):
                # Sum over l = 0, ..., k (0-indexed) = 1, ..., k+1 (1-indexed)
                V_prev = Vpath[:k + 1, i]   # Vbar_{t_{l-1}} for l=0,...,k; shape (k+1,)

                # Drift term: (mu_0^i - lambda_i * V_{l-1} - comp) * C[k, l]
                drift_term = np.sum(
                    (mu0[i] - lam[i] * V_prev - comp) * C[i][k, :k + 1]
                )

                # Diffusion term: sigma_v^i * sqrt(max(V_{l-1}, 0)) * G[k, l]
                sqrt_V = np.sqrt(np.maximum(V_prev, 0.0))
                diffusion_term = np.sum(
                    sigma_v[i] * sqrt_V * G[i][k, :k + 1]
                )

                # Jump term: J_l^i convolved with kernel weight C[k, l]
                jump_term = np.sum(J[:k + 1, i] * C[i][k, :k + 1])

                Vpath[k + 1, i] = V0[i] + drift_term + diffusion_term + jump_term

        V[path_idx] = Vpath

    return V, time_grid


def compute_roughness_diagnostics(
    V: np.ndarray,
    time_grid: np.ndarray,
    scales: Optional[list] = None,
) -> dict:
    """
    Compute roughness diagnostics for simulated variance paths.

    Metrics:
      - Mean absolute increment (MAI) at scale Delta and 2*Delta.
      - Realized quadratic variation estimate.
      - Log-log slope of MAI vs lag (Hurst surrogate).

    Parameters
    ----------
    V : np.ndarray
        Shape (n_paths, n+1, d) variance paths.
    time_grid : np.ndarray
        Shape (n+1,) time grid.
    scales : list, optional
        List of lag integers for increment analysis.

    Returns
    -------
    dict
        Roughness metrics per component.
    """
    if scales is None:
        scales = [1, 2, 4, 8]

    n_paths, n_plus_1, d = V.shape
    n = n_plus_1 - 1
    Delta = time_grid[1] - time_grid[0]

    diagnostics = {}
    for i in range(d):
        comp_diag = {}
        for lag in scales:
            if lag >= n:
                continue
            increments = V[:, lag:, i] - V[:, :-lag, i]   # shape (n_paths, n+1-lag)
            mai = np.mean(np.abs(increments))
            comp_diag[f'MAI_lag{lag}'] = float(mai)

        # Realized QV (lag=1)
        inc1 = V[:, 1:, i] - V[:, :-1, i]
        rqv = np.mean(np.sum(inc1 ** 2, axis=1))
        comp_diag['realized_QV'] = float(rqv)

        # Hurst surrogate from log-log slope
        lag_vals = [s for s in scales if s < n]
        if len(lag_vals) >= 2:
            mai_vals = [comp_diag.get(f'MAI_lag{s}', np.nan) for s in lag_vals]
            valid = [(np.log(s * Delta), np.log(m)) for s, m in zip(lag_vals, mai_vals)
                     if m > 0 and not np.isnan(m)]
            if len(valid) >= 2:
                xs, ys = zip(*valid)
                slope = np.polyfit(xs, ys, 1)[0]
                comp_diag['hurst_surrogate'] = float(slope)

        diagnostics[f'component_{i+1}'] = comp_diag

    return diagnostics


def check_positivity(V: np.ndarray) -> dict:
    """
    Check for negative values in simulated variance paths.

    Parameters
    ----------
    V : np.ndarray
        Shape (n_paths, n+1, d) variance paths.

    Returns
    -------
    dict
        Positivity report per component.
    """
    n_paths, _, d = V.shape
    report = {}
    for i in range(d):
        neg_count = int(np.sum(V[:, :, i] < 0))
        min_val = float(np.min(V[:, :, i]))
        report[f'component_{i+1}'] = {
            'n_negative': neg_count,
            'min_value': min_val,
            'positivity_fix': 'full truncation sqrt(max(V, 0)) applied',
        }
    return report
