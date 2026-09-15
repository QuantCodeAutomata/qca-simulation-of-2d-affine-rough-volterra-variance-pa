"""
Tests for src/kernels.py — fractional power kernels and Volterra weights.
"""

import numpy as np
import pytest
from scipy.special import gamma as gamma_func

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernels import kernel, volterra_weights, build_weight_matrix, compensator_drift


# ── kernel ──────────────────────────────────────────────────────────────────

class TestKernel:
    def test_positive_values_alpha06(self):
        """K_alpha(t) > 0 for t > 0."""
        t = np.array([0.1, 0.5, 1.0, 2.0])
        vals = kernel(t, alpha=0.6)
        assert np.all(vals > 0)

    def test_positive_values_alpha09(self):
        t = np.array([0.1, 0.5, 1.0])
        vals = kernel(t, alpha=0.9)
        assert np.all(vals > 0)

    def test_known_value_alpha1(self):
        """K_1(t) = t^0 / Gamma(1) = 1 for all t > 0."""
        t = np.array([0.5, 1.0, 2.0])
        vals = kernel(t, alpha=1.0)
        np.testing.assert_allclose(vals, 1.0, rtol=1e-12)

    def test_singular_at_zero(self):
        """K_alpha(0) = inf when alpha < 1."""
        vals = kernel(np.array([0.0]), alpha=0.6)
        assert np.isinf(vals[0])

    def test_formula(self):
        """Verify K_alpha(t) = t^{alpha-1}/Gamma(alpha) analytically."""
        alpha = 0.7
        t = np.array([0.3, 0.8, 1.5])
        expected = t ** (alpha - 1) / gamma_func(alpha)
        np.testing.assert_allclose(kernel(t, alpha), expected, rtol=1e-12)

    def test_decreasing_for_alpha_lt1(self):
        """For alpha < 1, K_alpha is decreasing in t."""
        t = np.linspace(0.01, 2.0, 50)
        vals = kernel(t, 0.5)
        assert np.all(np.diff(vals) < 0)

    def test_nonneg_input_required(self):
        with pytest.raises(AssertionError):
            kernel(np.array([-0.1]), 0.6)


# ── volterra_weights ─────────────────────────────────────────────────────────

class TestVolterraWeights:
    def test_shape_and_positivity(self):
        """Weights have correct length and are positive."""
        w = volterra_weights(k=5, n=10, T=1.0, alpha=0.6)
        assert len(w) == 5
        assert np.all(w > 0)

    def test_k_equals_1(self):
        """Single weight at k=1 covers interval [0, Delta]."""
        n, T, alpha = 10, 1.0, 0.6
        Delta = T / n
        w = volterra_weights(k=1, n=n, T=T, alpha=alpha)
        assert len(w) == 1
        # C_{1,1} = [(t_1-t_0)^alpha - 0] / Gamma(alpha+1) = Delta^alpha / Gamma(alpha+1)
        expected = Delta ** alpha / gamma_func(alpha + 1)
        np.testing.assert_allclose(w[0], expected, rtol=1e-12)

    def test_sum_equals_total_integral(self):
        """Sum of weights C_{n,l} for l=1..n equals T^alpha/Gamma(alpha+1)."""
        n, T, alpha = 20, 1.0, 0.7
        w = volterra_weights(k=n, n=n, T=T, alpha=alpha)
        total = np.sum(w)
        expected = T ** alpha / gamma_func(alpha + 1)
        np.testing.assert_allclose(total, expected, rtol=1e-8)

    def test_sum_partial(self):
        """Sum of weights at intermediate k also satisfies integral bound."""
        n, T, alpha = 20, 1.0, 0.8
        k = 10
        Delta = T / n
        t_k = k * Delta
        w = volterra_weights(k=k, n=n, T=T, alpha=alpha)
        total = np.sum(w)
        expected = t_k ** alpha / gamma_func(alpha + 1)
        np.testing.assert_allclose(total, expected, rtol=1e-8)


# ── build_weight_matrix ──────────────────────────────────────────────────────

class TestBuildWeightMatrix:
    def test_shape(self):
        C = build_weight_matrix(n=10, T=1.0, alpha=0.6)
        assert C.shape == (10, 10)

    def test_lower_triangular(self):
        C = build_weight_matrix(n=8, T=1.0, alpha=0.7)
        assert np.all(C == np.tril(C))

    def test_positive_entries(self):
        C = build_weight_matrix(n=5, T=1.0, alpha=0.8)
        assert np.all(C[C != 0] > 0)

    def test_consistency_with_volterra_weights(self):
        """C[k, :k+1] must equal volterra_weights(k+1, n, T, alpha)."""
        n, T, alpha = 10, 1.0, 0.65
        C = build_weight_matrix(n=n, T=T, alpha=alpha)
        for k in range(n):
            w = volterra_weights(k + 1, n, T, alpha)
            np.testing.assert_allclose(C[k, :k + 1], w, rtol=1e-12)


# ── compensator_drift ─────────────────────────────────────────────────────────

class TestCompensatorDrift:
    def test_formula(self):
        """comp = beta*kappa/sqrt(2*pi)."""
        beta, kappa = 0.5, 0.3
        expected = beta * kappa / np.sqrt(2.0 * np.pi)
        np.testing.assert_allclose(compensator_drift(beta, kappa), expected, rtol=1e-12)

    def test_zero_beta(self):
        assert compensator_drift(0.0, 0.3) == 0.0

    def test_zero_kappa(self):
        assert compensator_drift(0.5, 0.0) == 0.0

    def test_positive(self):
        assert compensator_drift(1.0, 1.0) > 0
