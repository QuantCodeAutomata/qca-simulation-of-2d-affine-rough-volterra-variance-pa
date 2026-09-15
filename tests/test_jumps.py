"""
Tests for src/jumps.py — Poisson jumps, mark simulation, and utility functions.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jumps import (
    simulate_jump_contributions,
    simulate_jumps_and_marks,
    h_gamma,
    h,
    jump_integral_exp_utility,
    jump_integral_power_utility,
)


# ── simulate_jump_contributions ──────────────────────────────────────────────

class TestSimulateJumpContributions:
    def test_shape(self):
        J = simulate_jump_contributions(n=100, T=1.0, beta=0.5, kappa=0.3, d=2,
                                        rng=np.random.default_rng(0))
        assert J.shape == (100, 2)

    def test_nonnegative(self):
        """All jump contributions are nonneg because eta(e)=kappa*e*1_{e>0} >= 0."""
        J = simulate_jump_contributions(n=200, T=1.0, beta=2.0, kappa=1.0, d=2,
                                        rng=np.random.default_rng(7))
        assert np.all(J >= 0)

    def test_zero_beta(self):
        """No jumps when beta=0."""
        J = simulate_jump_contributions(n=100, T=1.0, beta=0.0, kappa=1.0, d=2,
                                        rng=np.random.default_rng(1))
        np.testing.assert_array_equal(J, 0.0)

    def test_zero_kappa(self):
        """Zero jump size when kappa=0 (arrives but eta=0)."""
        J = simulate_jump_contributions(n=100, T=1.0, beta=5.0, kappa=0.0, d=2,
                                        rng=np.random.default_rng(2))
        np.testing.assert_array_equal(J, 0.0)

    def test_identical_across_dims(self):
        """Both variance components receive the same jump realization."""
        J = simulate_jump_contributions(n=50, T=1.0, beta=1.0, kappa=0.5, d=2,
                                        rng=np.random.default_rng(3))
        np.testing.assert_array_equal(J[:, 0], J[:, 1])

    def test_reproducible_seed(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        J1 = simulate_jump_contributions(50, 1.0, 0.5, 0.3, d=2, rng=rng1)
        J2 = simulate_jump_contributions(50, 1.0, 0.5, 0.3, d=2, rng=rng2)
        np.testing.assert_array_equal(J1, J2)


# ── simulate_jumps_and_marks ─────────────────────────────────────────────────

class TestSimulateJumpsAndMarks:
    def test_output_shapes(self):
        arr, eta = simulate_jumps_and_marks(100, 1.0, 0.5, 0.3,
                                            rng=np.random.default_rng(10))
        assert arr.shape == (100,)
        assert eta.shape == (100,)

    def test_arrivals_nonneg(self):
        arr, _ = simulate_jumps_and_marks(100, 1.0, 0.5, 0.3,
                                          rng=np.random.default_rng(11))
        assert np.all(arr >= 0)

    def test_eta_nonneg(self):
        _, eta = simulate_jumps_and_marks(200, 1.0, 2.0, 1.0,
                                          rng=np.random.default_rng(12))
        assert np.all(eta >= 0)


# ── h_gamma and h ────────────────────────────────────────────────────────────

class TestHFunctions:
    def test_h_zero(self):
        """h(0) = exp(0) - 0 - 1 = 0."""
        assert h(np.array([0.0]))[0] == 0.0

    def test_h_positive_input(self):
        """h(x) > 0 for x != 0."""
        assert h(np.array([1.0]))[0] > 0
        assert h(np.array([-1.0]))[0] > 0

    def test_h_gamma_zero(self):
        """h_gamma(0, gamma) = 0."""
        result = h_gamma(np.array([0.0]), gamma=0.5)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-15)

    def test_h_gamma_positive(self):
        """h_gamma(x, gamma) > 0 for x != 0."""
        assert h_gamma(np.array([1.0]), gamma=0.2)[0] > 0
        assert h_gamma(np.array([-1.0]), gamma=0.2)[0] > 0

    def test_h_convex(self):
        """h is convex: h((x+y)/2) <= (h(x)+h(y))/2."""
        x, y = np.array([1.0]), np.array([3.0])
        assert h((x + y) / 2)[0] <= (h(x)[0] + h(y)[0]) / 2 + 1e-12

    def test_h_gamma_formula(self):
        """h_gamma(x, g) = (exp(g*x) - g*x - 1)/g."""
        x = np.array([0.5])
        gamma = 0.3
        expected = (np.exp(gamma * x) - gamma * x - 1.0) / gamma
        np.testing.assert_allclose(h_gamma(x, gamma), expected, rtol=1e-12)


# ── jump_integral functions ──────────────────────────────────────────────────

class TestJumpIntegrals:
    def test_zero_beta_exp(self):
        """Integral vanishes when beta=0."""
        psi = np.array([0.1, 0.2])
        result = jump_integral_exp_utility(psi, beta=0.0, kappa=0.3, gamma=0.2)
        assert result == 0.0

    def test_zero_beta_pow(self):
        psi = np.array([0.1, 0.2])
        result = jump_integral_power_utility(psi, beta=0.0, kappa=0.3)
        assert result == 0.0

    def test_nonneg_exp_utility(self):
        """h_gamma >= 0 => integral >= 0 for positive psi."""
        psi = np.array([0.05, 0.1])
        result = jump_integral_exp_utility(psi, beta=0.5, kappa=0.3, gamma=0.2)
        assert result >= 0

    def test_nonneg_pow_utility(self):
        psi = np.array([0.05, 0.1])
        result = jump_integral_power_utility(psi, beta=0.5, kappa=0.3)
        assert result >= 0

    def test_finite_values(self):
        psi = np.array([0.1, 0.2])
        r1 = jump_integral_exp_utility(psi, beta=0.5, kappa=0.3, gamma=0.2)
        r2 = jump_integral_power_utility(psi, beta=0.5, kappa=0.3)
        assert np.isfinite(r1)
        assert np.isfinite(r2)
