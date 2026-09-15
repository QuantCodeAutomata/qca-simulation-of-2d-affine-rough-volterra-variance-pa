"""
Tests for src/riccati.py — fractional Adams-Bashforth-Moulton solver for
Riccati-Volterra equations (paper eqs. 32-33, 50-51).
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.riccati import (
    solve_riccati_volterra,
    abm_predictor_weights,
    abm_corrector_weights,
)


ALPHA = np.array([0.6, 0.9])
THETA = np.array([0.1, 0.1])
RHO   = np.array([-0.7, -0.55])
SIGMA_V = np.array([0.4, 0.2])
LAM   = np.array([0.2, 0.6])


# ── ABM weight tests ─────────────────────────────────────────────────────────

class TestABMWeights:
    def test_predictor_shape(self):
        b = abm_predictor_weights(k=4, Delta=0.1, alpha=0.6)
        assert len(b) == 5  # j=0..k

    def test_predictor_positive(self):
        b = abm_predictor_weights(k=5, Delta=0.05, alpha=0.7)
        assert np.all(b >= 0)

    def test_corrector_shape(self):
        a = abm_corrector_weights(k=4, Delta=0.1, alpha=0.6)
        assert len(a) == 6  # j=0..k+1

    def test_corrector_positive(self):
        a = abm_corrector_weights(k=5, Delta=0.05, alpha=0.8)
        assert np.all(a >= 0)

    def test_predictor_weight_formula(self):
        """b_{j,k+1} = Delta^alpha/Gamma(alpha+1) * [(k+1-j)^alpha - (k-j)^alpha]."""
        from scipy.special import gamma as gf
        k, Delta, alpha = 3, 0.1, 0.6
        b = abm_predictor_weights(k, Delta, alpha)
        g1 = gf(alpha + 1)
        for j in range(k + 1):
            expected = Delta ** alpha / g1 * ((k + 1 - j) ** alpha - (k - j) ** alpha)
            np.testing.assert_allclose(b[j], expected, rtol=1e-12)


# ── solve_riccati_volterra ────────────────────────────────────────────────────

class TestSolveRiccatiVolterra:
    def test_shape_output(self):
        """Output shape is (n+1, d)."""
        psi = solve_riccati_volterra(
            n=10, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='exponential',
        )
        assert psi.shape == (11, 2)

    def test_initial_condition_zero(self):
        """psi(0) = 0 by construction (Adams scheme with psi_0=0)."""
        psi = solve_riccati_volterra(
            n=10, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='power',
        )
        np.testing.assert_array_equal(psi[0], np.zeros(2))

    def test_exp_nonpositive(self):
        """Under baseline parameters, exponential psi should be non-positive."""
        psi = solve_riccati_volterra(
            n=200, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='exponential',
            zeta=np.zeros(2),
        )
        assert np.all(psi <= 1e-8), f"Non-positive sign violated: max={psi.max()}"

    def test_pow_nonnegative(self):
        """Under baseline parameters, power utility psi should be non-negative."""
        psi = solve_riccati_volterra(
            n=200, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='power',
        )
        assert np.all(psi >= -1e-8), f"Non-negative sign violated: min={psi.min()}"

    def test_finite_values(self):
        """All psi values must be finite."""
        for utility in ['exponential', 'power']:
            psi = solve_riccati_volterra(
                n=50, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
                sigma_v=SIGMA_V, lam=LAM, gamma=0.5, utility=utility,
            )
            assert np.all(np.isfinite(psi)), f"Non-finite in {utility}"

    def test_zero_input_gives_zero(self):
        """When theta=0, sigma_v=0, lam=0, zeta=0 => psi should be near zero."""
        psi = solve_riccati_volterra(
            n=20, T=1.0,
            alpha=ALPHA,
            theta=np.zeros(2),
            rho=np.zeros(2),
            sigma_v=np.zeros(2),
            lam=np.zeros(2),
            gamma=0.2,
            utility='exponential',
            zeta=np.zeros(2),
        )
        np.testing.assert_allclose(psi, 0.0, atol=1e-12)

    def test_grid_refinement_stability(self):
        """Increasing n should not produce diverging max |psi|."""
        maxvals = []
        for n_ref in [50, 100, 200]:
            psi = solve_riccati_volterra(
                n=n_ref, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
                sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='power',
            )
            maxvals.append(float(np.abs(psi).max()))
        # max value should not grow by more than 2x across refinements
        assert maxvals[-1] < 2 * maxvals[0] + 1e-3

    def test_different_gamma_changes_psi(self):
        """Different risk aversion should change psi solutions."""
        psi_a = solve_riccati_volterra(
            n=50, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='power',
        )
        psi_b = solve_riccati_volterra(
            n=50, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
            sigma_v=SIGMA_V, lam=LAM, gamma=0.8, utility='power',
        )
        assert not np.allclose(psi_a, psi_b)

    def test_invalid_utility_raises(self):
        with pytest.raises((ValueError, KeyError)):
            solve_riccati_volterra(
                n=10, T=1.0, alpha=ALPHA, theta=THETA, rho=RHO,
                sigma_v=SIGMA_V, lam=LAM, gamma=0.2, utility='logarithmic',
            )
