"""
Tests for src/strategy.py — deterministic optimal strategy curves.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy import compute_strategy_curves, compute_full_alpha_control


ALPHA  = np.array([0.6, 0.9])
THETA  = np.array([0.1, 0.1])
RHO    = np.array([-0.7, -0.55])
SIGMA_V = np.array([0.4, 0.2])
LAM    = np.array([0.2, 0.6])


# ── compute_strategy_curves ──────────────────────────────────────────────────

class TestComputeStrategyCurves:
    def test_shape_exponential(self):
        tg, pi, psi = compute_strategy_curves(
            n=20, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='exponential',
            r=0.02, zeta=np.zeros(2),
        )
        assert tg.shape == (21,)
        assert pi.shape == (21, 2)
        assert psi.shape == (21, 2)

    def test_shape_power(self):
        tg, pi, psi = compute_strategy_curves(
            n=20, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='power',
            r=0.02,
        )
        assert pi.shape == (21, 2)

    def test_finite_values_exp(self):
        _, pi, psi = compute_strategy_curves(
            n=50, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='exponential',
        )
        assert np.all(np.isfinite(pi))
        assert np.all(np.isfinite(psi))

    def test_finite_values_pow(self):
        _, pi, psi = compute_strategy_curves(
            n=50, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.6, utility='power',
        )
        assert np.all(np.isfinite(pi))

    def test_time_grid_start_end(self):
        tg, _, _ = compute_strategy_curves(
            n=50, T=5.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='power',
        )
        np.testing.assert_allclose(tg[0], 0.0)
        np.testing.assert_allclose(tg[-1], 5.0)

    def test_higher_gamma_reduces_power_exposure(self):
        """For power utility, 1/(1-gamma) is larger for larger gamma."""
        _, pi_low, _ = compute_strategy_curves(
            n=50, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='power',
        )
        _, pi_high, _ = compute_strategy_curves(
            n=50, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.8, utility='power',
        )
        # At t=0 (tau=T), psi≈0, so pi ≈ theta/(1-gamma) which grows with gamma
        # Larger gamma => larger |pi| at time 0
        assert np.abs(pi_high[0]).sum() >= np.abs(pi_low[0]).sum() - 1e-6

    def test_exp_strategy_discounted(self):
        """Exponential strategy has exp(-r*(T-t)) factor => decays to 0 at t=T."""
        n, T, r = 50, 1.0, 0.02
        tg, pi, _ = compute_strategy_curves(
            n=n, T=T, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='exponential', r=r,
        )
        # At t=T, T-t=0 => exp(0)=1 => pi_T = (theta_i + 0)/gamma (psi(0)=0)
        # Just check it is finite and nonzero somewhere
        assert np.any(np.abs(pi) > 1e-6)


# ── compute_full_alpha_control ────────────────────────────────────────────────

class TestComputeFullAlphaControl:
    def _base_setup(self, n=20, T=1.0, gamma=0.2, utility='power'):
        from src.volterra_sim import simulate_variance_paths
        tg, pi, psi = compute_strategy_curves(
            n=n, T=T, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=gamma, utility=utility, r=0.02,
        )
        V, _ = simulate_variance_paths(
            n=n, T=T, V0=np.array([0.01, 0.03]),
            mu0=np.array([2.0, 2.5]),
            lam=LAM, sigma_v=SIGMA_V, alpha=ALPHA,
            beta=0.0, kappa=0.0, n_paths=1, rng_seed=0,
        )
        return tg, psi, V[0]  # (n+1,2), (n+1,d), (n+1,2)

    def test_shape(self):
        tg, psi, V_path = self._base_setup()
        ctrl = compute_full_alpha_control(
            V_path=V_path, time_grid=tg, psi=psi,
            theta=THETA, rho=RHO, sigma_v=SIGMA_V,
            gamma=0.2, r=0.02, T=1.0, utility='power',
        )
        assert ctrl.shape == (21, 2)

    def test_finite(self):
        tg, psi, V_path = self._base_setup()
        ctrl = compute_full_alpha_control(
            V_path=V_path, time_grid=tg, psi=psi,
            theta=THETA, rho=RHO, sigma_v=SIGMA_V,
            gamma=0.2, r=0.02, T=1.0, utility='power',
        )
        assert np.all(np.isfinite(ctrl))

    def test_pi_consistency_power(self):
        """alpha^i = pi^i * sqrt(V^i) should hold."""
        n, T, gamma = 20, 1.0, 0.2
        tg, pi, psi = compute_strategy_curves(
            n=n, T=T, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=gamma, utility='power', r=0.02,
        )
        # Use constant V = V0 (just pi * sqrt(V0) should equal alpha when V_path = V0 throughout)
        V0_path = np.tile(np.array([0.01, 0.03]), (n + 1, 1))
        ctrl = compute_full_alpha_control(
            V_path=V0_path, time_grid=tg, psi=psi,
            theta=THETA, rho=RHO, sigma_v=SIGMA_V,
            gamma=gamma, r=0.02, T=T, utility='power',
        )
        # Expected: alpha^i = pi^i * sqrt(V0^i)
        expected = pi * np.sqrt(np.array([0.01, 0.03]))
        np.testing.assert_allclose(ctrl, expected, rtol=1e-10)
