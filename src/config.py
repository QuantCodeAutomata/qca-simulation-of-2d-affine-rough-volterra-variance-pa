"""
Model parameters for 2D affine rough-Volterra variance process with jumps.

All parameters follow Section 4 of the paper. Missing parameters (beta, kappa,
zeta) are documented explicitly in the manifest below and chosen to enable
reproducible numerical experiments.

Parameter manifest:
    beta  = 0.5  (jump intensity; NOT given in paper)
    kappa = 0.3  (jump amplitude scaling; NOT given in paper)
    zeta  = [0.0, 0.0]  (exponential utility liability; NOT given in paper,
                          default = no-liability interpretation)
    x_0   = 1.0  (initial wealth; NOT given in paper, scale normalization)
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class ModelParams:
    """Core model parameters for the 2D affine rough-Volterra model."""

    # --- Dimension ---
    d: int = 2

    # --- Kernel / roughness parameters ---
    # alpha_i = H_i + 0.5; H_1=0.1 => alpha_1=0.6, H_2=0.4 => alpha_2=0.9
    alpha: List[float] = field(default_factory=lambda: [0.6, 0.9])

    # --- Initial variance ---
    V0: List[float] = field(default_factory=lambda: [0.01, 0.03])

    # --- Drift vector mu_0 ---
    mu0: List[float] = field(default_factory=lambda: [2.0, 2.5])

    # --- Mean-reversion magnitudes (positive; D = diag(-lambda)) ---
    lam: List[float] = field(default_factory=lambda: [0.2, 0.6])  # lambda_1, lambda_2

    # --- Leverage / correlation diagonal entries (rho_i from Sigma = diag(rho)) ---
    rho: List[float] = field(default_factory=lambda: [-0.7, -0.55])

    # --- Volatility-of-volatility ---
    sigma_v: List[float] = field(default_factory=lambda: [0.4, 0.2])

    # --- Market price of risk ---
    theta: List[float] = field(default_factory=lambda: [0.1, 0.1])

    # --- Risk-free rate ---
    r: float = 0.02

    # --- Jump parameters (NOT provided by paper; documented as user-supplied) ---
    beta: float = 0.5    # Poisson intensity of the background driving process nu_0
    kappa: float = 0.3   # Jump amplitude scaling in eta^i(e) = kappa*e*1_{e>0}

    # --- Exponential utility liability (NOT provided by paper) ---
    zeta: List[float] = field(default_factory=lambda: [0.0, 0.0])

    # --- Initial wealth (NOT provided by paper) ---
    x0: float = 1.0

    # Derived property helpers
    @property
    def alpha_arr(self) -> np.ndarray:
        return np.array(self.alpha)

    @property
    def V0_arr(self) -> np.ndarray:
        return np.array(self.V0)

    @property
    def mu0_arr(self) -> np.ndarray:
        return np.array(self.mu0)

    @property
    def lam_arr(self) -> np.ndarray:
        return np.array(self.lam)

    @property
    def rho_arr(self) -> np.ndarray:
        return np.array(self.rho)

    @property
    def sigma_v_arr(self) -> np.ndarray:
        return np.array(self.sigma_v)

    @property
    def theta_arr(self) -> np.ndarray:
        return np.array(self.theta)

    @property
    def zeta_arr(self) -> np.ndarray:
        return np.array(self.zeta)

    @property
    def D(self) -> np.ndarray:
        """Mean-reversion matrix D = diag(-lambda)."""
        return np.diag([-l for l in self.lam])

    @property
    def Sigma_mat(self) -> np.ndarray:
        """Leverage matrix Sigma = diag(rho)."""
        return np.diag(self.rho)

    @property
    def hurst(self) -> List[float]:
        """Hurst exponents H_i = alpha_i - 0.5."""
        return [a - 0.5 for a in self.alpha]


# Default parameter set
DEFAULT_PARAMS = ModelParams()

# Grid parameters
T_DEFAULT = 1.0
N_DEFAULT = 200
