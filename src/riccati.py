"""
Fractional Adams-Bashforth-Moulton predictor-corrector solver for the
coupled Riccati-Volterra equations arising in exponential and power utility
optimization under the 2D affine rough-Volterra model.

Custom — Context7 found no library equivalent for this Riccati-Volterra system
(paper equations 32-33 and 50-51).

The equations take the form:
    psi^i(T) = int_0^T K_{alpha_i}(T-s) [a_i + F_i(s, psi(s))] ds

Solved forward in tau = T - s via the Adams predictor-corrector scheme.

Under the stated numerical parameters:
  - nu_i = 0 => state-dependent jump terms vanish from F_i
  - D is diagonal => (D^T psi)_i = D_{ii} * psi_i = -lambda_i * psi_i
  - varsigma = I => no additional scaling

Exponential utility (eq. 32-33):
    a_i^{exp} = zeta_i - theta_i^2 / (2 * gamma)
    F_i^{exp}(s, psi) = -theta_i * rho_i * sigma_v_i * psi_i
                        + (D^T psi)_i
                        + gamma * (sigma_v_i)^2 / 2 * (1 - rho_i^2) * psi_i^2

Power utility (eq. 50-51):
    a_i^{pow} = gamma * theta_i^2 / (2 * (1 - gamma))
    F_i^{pow}(s, psi) = gamma/(1-gamma) * theta_i * rho_i * sigma_v_i * psi_i
                        + (D^T psi)_i
                        + (sigma_v_i)^2 / 2 * psi_i^2 * (1 + gamma/(1-gamma) * rho_i^2)
"""

import numpy as np
from scipy.special import gamma as gamma_func
from typing import Literal, Tuple, Optional


def build_abm_predictor_weights(
    n: int, Delta: float, alpha: float
) -> np.ndarray:
    """
    Build lower-triangular predictor weight matrix B[k, j] = b_{j,k+1} for j=0..k.

    b_{j,k+1} = Delta^alpha / Gamma(alpha+1) * [(k+1-j)^alpha - (k-j)^alpha]

    Returns
    -------
    np.ndarray
        Shape (n, n) lower-triangular matrix.
    """
    B = np.zeros((n, n))
    coeff = Delta ** alpha / gamma_func(alpha + 1)
    for k in range(n):
        j_vals = np.arange(0, k + 1, dtype=float)
        B[k, :k + 1] = coeff * ((k + 1 - j_vals) ** alpha - (k - j_vals) ** alpha)
    return B


def build_abm_corrector_weights(
    n: int, Delta: float, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build corrector weight matrices A_prev[k, j] (j=0..k) and A_pred (length n).

    Returns
    -------
    A_prev : np.ndarray
        Shape (n, n) lower-triangular matrix for A_{j,k+1}, j=0..k.
    A_pred : np.ndarray
        Shape (n,) vector a_{k+1, k+1} for the predictor step.
    """
    coeff = Delta ** alpha / gamma_func(alpha + 2)
    A_prev = np.zeros((n, n))
    A_pred = np.full(n, coeff)  # a_{k+1,k+1} = coeff * 1 for all k

    for k in range(n):
        # j = 0
        if k == 0:
            a0 = coeff * (0.0 - (-alpha) * 1.0)  # k^{alpha+1}=0, (k-alpha)*(k+1)^alpha
        else:
            a0 = coeff * (k ** (alpha + 1) - (k - alpha) * (k + 1) ** alpha)
        A_prev[k, 0] = a0

        # j = 1, ..., k
        if k >= 1:
            j_vals = np.arange(1, k + 1, dtype=float)
            A_prev[k, 1:k + 1] = coeff * (
                (k - j_vals + 2) ** (alpha + 1)
                + (k - j_vals) ** (alpha + 1)
                - 2.0 * (k - j_vals + 1) ** (alpha + 1)
            )

    return A_prev, A_pred


def _rhs_exponential(
    psi: np.ndarray,
    tau: float,
    theta: np.ndarray,
    rho: np.ndarray,
    sigma_v: np.ndarray,
    lam: np.ndarray,
    gamma: float,
    zeta: np.ndarray,
) -> np.ndarray:
    """
    RHS vector f(tau, psi) for exponential utility Riccati equation.

    f_i = a_i + F_i(tau, psi)

    where:
        a_i = zeta_i - theta_i^2 / (2*gamma)
        F_i = -theta_i * rho_i * sigma_v_i * psi_i
              - lambda_i * psi_i
              + gamma * (sigma_v_i)^2 / 2 * (1 - rho_i^2) * psi_i^2

    Notes: (D^T psi)_i = -lambda_i * psi_i (diagonal D)
           nu_i = 0 => jump term = 0
    """
    a = zeta - theta ** 2 / (2.0 * gamma)
    F = (- theta * rho * sigma_v * psi
         - lam * psi
         + gamma * sigma_v ** 2 / 2.0 * (1.0 - rho ** 2) * psi ** 2)
    return a + F


def _rhs_power(
    psi: np.ndarray,
    tau: float,
    theta: np.ndarray,
    rho: np.ndarray,
    sigma_v: np.ndarray,
    lam: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    RHS vector f(tau, psi) for power utility Riccati equation.

    f_i = a_i + F_i(tau, psi)

    where:
        a_i = gamma * theta_i^2 / (2 * (1 - gamma))
        F_i = gamma/(1-gamma) * theta_i * rho_i * sigma_v_i * psi_i
              - lambda_i * psi_i
              + (sigma_v_i)^2 / 2 * psi_i^2 * (1 + gamma/(1-gamma) * rho_i^2)
    """
    r_frac = gamma / (1.0 - gamma)
    a = r_frac * theta ** 2 / 2.0
    F = (r_frac * theta * rho * sigma_v * psi
         - lam * psi
         + sigma_v ** 2 / 2.0 * psi ** 2 * (1.0 + r_frac * rho ** 2))
    return a + F


def solve_riccati_volterra(
    n: int,
    T: float,
    alpha: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    sigma_v: np.ndarray,
    lam: np.ndarray,
    gamma: float,
    utility: Literal['exponential', 'power'],
    zeta: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve the coupled fractional Riccati-Volterra system using the ABM scheme.

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
        Shape (d,) leverage/correlation diagonal entries.
    sigma_v : np.ndarray
        Shape (d,) volatility-of-volatility.
    lam : np.ndarray
        Shape (d,) mean-reversion magnitudes.
    gamma : float
        Risk aversion (0 < gamma < 1 for power; gamma > 0 for exponential).
    utility : str
        'exponential' or 'power'.
    zeta : np.ndarray, optional
        Shape (d,) liability parameter for exponential utility. Defaults to zeros.

    Returns
    -------
    psi : np.ndarray
        Shape (n+1, d) Riccati solution, with psi[0] = 0 (initial condition).
    """
    if utility not in ('exponential', 'power'):
        raise ValueError(
            f"Unknown utility '{utility}'. Must be 'exponential' or 'power'."
        )

    d = len(alpha)
    Delta = T / n
    if zeta is None:
        zeta = np.zeros(d)

    # psi[k] corresponds to psi(tau_k) where tau_k = k * Delta = T - t_k
    psi = np.zeros((n + 1, d))  # psi[0] = psi(0) = 0

    # Precompute ABM weights for each component
    B_mats = []   # predictor
    A_mats = []   # corrector prev
    A_preds = []  # corrector pred
    for i in range(d):
        B = build_abm_predictor_weights(n, Delta, float(alpha[i]))
        A_prev, A_pred = build_abm_corrector_weights(n, Delta, float(alpha[i]))
        B_mats.append(B)
        A_mats.append(A_prev)
        A_preds.append(A_pred)

    def rhs(psi_val: np.ndarray, k: int) -> np.ndarray:
        """Evaluate f(tau_k, psi_k) for given k."""
        tau = k * Delta
        if utility == 'exponential':
            return _rhs_exponential(psi_val, tau, theta, rho, sigma_v, lam, gamma, zeta)
        else:
            return _rhs_power(psi_val, tau, theta, rho, sigma_v, lam, gamma)

    # Precompute rhs at step 0
    f_vals = np.zeros((n + 1, d))
    f_vals[0] = rhs(psi[0], 0)

    for k in range(n):
        # Predictor: for each component i
        psi_pred = np.zeros(d)
        for i in range(d):
            psi_pred[i] = np.dot(B_mats[i][k, :k + 1], f_vals[:k + 1, i])

        # Evaluate rhs at predictor
        f_pred = rhs(psi_pred, k + 1)

        # Corrector: for each component i
        psi_new = np.zeros(d)
        for i in range(d):
            psi_new[i] = (
                np.dot(A_mats[i][k, :k + 1], f_vals[:k + 1, i])
                + A_preds[i][k] * f_pred[i]
            )

        psi[k + 1] = psi_new
        f_vals[k + 1] = rhs(psi_new, k + 1)

    return psi


# ── Per-step weight helpers (used by tests and external callers) ──────────────

def abm_predictor_weights(k: int, Delta: float, alpha: float) -> np.ndarray:
    """
    Return the Adams-Bashforth-Moulton predictor weights b_{j,k+1} for j=0,...,k.

    b_{j,k+1} = Delta^alpha / Gamma(alpha+1) * [(k+1-j)^alpha - (k-j)^alpha]

    Parameters
    ----------
    k : int
        Current step index (0-based).
    Delta : float
        Time step size.
    alpha : float
        Fractional order.

    Returns
    -------
    np.ndarray
        Shape (k+1,) array of predictor weights.
    """
    coeff = Delta ** alpha / gamma_func(alpha + 1)
    j_vals = np.arange(0, k + 1, dtype=float)
    return coeff * ((k + 1 - j_vals) ** alpha - (k - j_vals) ** alpha)


def abm_corrector_weights(k: int, Delta: float, alpha: float) -> np.ndarray:
    """
    Return the Adams-Bashforth-Moulton corrector weights a_{j,k+1} for j=0,...,k+1.

    Piecewise formula from the paper:
      j=0    : coeff * [k^{alpha+1} - (k-alpha)*(k+1)^alpha]
      1<=j<=k: coeff * [(k-j+2)^{alpha+1} + (k-j)^{alpha+1} - 2*(k-j+1)^{alpha+1}]
      j=k+1  : coeff  (the predictor coefficient)

    Parameters
    ----------
    k : int
        Current step index (0-based).
    Delta : float
        Time step size.
    alpha : float
        Fractional order.

    Returns
    -------
    np.ndarray
        Shape (k+2,) array of corrector weights.
    """
    coeff = Delta ** alpha / gamma_func(alpha + 2)
    result = np.zeros(k + 2)

    # j = 0
    if k == 0:
        result[0] = coeff * alpha  # special case: 0^{alpha+1}-(0-alpha)*1^alpha = alpha
    else:
        result[0] = coeff * (k ** (alpha + 1) - (k - alpha) * (k + 1) ** alpha)

    # j = 1,...,k
    if k >= 1:
        j_vals = np.arange(1, k + 1, dtype=float)
        result[1:k + 1] = coeff * (
            (k - j_vals + 2) ** (alpha + 1)
            + (k - j_vals) ** (alpha + 1)
            - 2.0 * (k - j_vals + 1) ** (alpha + 1)
        )

    # j = k+1 (predictor contribution)
    result[k + 1] = coeff

    return result
