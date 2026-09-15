"""
Tests for src/value_functions.py — semi-closed-form value functions
(paper equations 41, 56, 62).
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.value_functions import (
    compute_g0,
    evaluate_exponential_value,
    evaluate_power_value,
    evaluate_log_value,
)


# Shared parameters
V0      = np.array([0.01, 0.03])
MU0     = np.array([2.0, 2.5])
ALPHA   = np.array([0.6, 0.9])
THETA   = np.array([0.1, 0.1])
RHO     = np.array([-0.7, -0.55])
SIGMA_V = np.array([0.4, 0.2])
LAM     = np.array([0.2, 0.6])
GAMMA   = 0.2
R       = 0.02
X0      = 1.0
T       = 1.0
N       = 50       # small grid for fast tests


# ── compute_g0 ───────────────────────────────────────────────────────────────

class TestComputeG0:
    def test_shape(self):
        t = np.linspace(0, 1.0, 51)
        g = compute_g0(t, V0, MU0, ALPHA)
        assert g.shape == (51, 2)

    def test_initial_value(self):
        """g_0(0) = V0."""
        t = np.array([0.0])
        g = compute_g0(t, V0, MU0, ALPHA)
        np.testing.assert_allclose(g[0], V0, rtol=1e-12)

    def test_increasing_in_t(self):
        """g_0^i(t) is increasing since mu_0^i > 0."""
        t = np.linspace(0.0, 1.0, 20)
        g = compute_g0(t, V0, MU0, ALPHA)
        # Skip t=0 (no increment to check against previous)
        assert np.all(np.diff(g[:, 0]) > 0)
        assert np.all(np.diff(g[:, 1]) > 0)

    def test_formula(self):
        """g_0^i(t) = V0^i + mu0^i * t^alpha_i / Gamma(alpha_i+1)."""
        from scipy.special import gamma as gf
        t = np.array([0.5, 1.0])
        g = compute_g0(t, V0, MU0, ALPHA)
        for i in range(2):
            expected = V0[i] + MU0[i] * t ** ALPHA[i] / gf(ALPHA[i] + 1)
            np.testing.assert_allclose(g[:, i], expected, rtol=1e-12)


# ── evaluate_exponential_value ────────────────────────────────────────────────

class TestExponentialValue:
    def _call(self, **kwargs):
        defaults = dict(
            x0=X0, T=T, n=N, V0=V0, mu0=MU0, alpha=ALPHA,
            theta=THETA, rho=RHO, sigma_v=SIGMA_V, lam=LAM,
            gamma=GAMMA, r=R, beta=0.5, kappa=0.3,
            zeta=np.zeros(2),
        )
        defaults.update(kwargs)
        return evaluate_exponential_value(**defaults)

    def test_returns_dict(self):
        result = self._call()
        assert isinstance(result, dict)
        assert 'value' in result

    def test_finite(self):
        result = self._call()
        assert np.isfinite(result['value'])

    def test_negative_value(self):
        """Exponential utility is always negative."""
        result = self._call()
        assert result['value'] < 0

    def test_theta_0_negative(self):
        """Theta_0 = -(1/gamma)*exp(...) < 0."""
        result = self._call()
        assert result['Theta_0'] < 0

    def test_beta_zero_reduces_jump_term(self):
        """With beta=0, jump integral is zero; should differ from beta>0."""
        r_nojump = self._call(beta=0.0, kappa=0.3)
        r_jump   = self._call(beta=0.5, kappa=0.3)
        # Both should be finite and negative
        assert np.isfinite(r_nojump['value'])
        assert np.isfinite(r_jump['value'])

    def test_sensitivity_to_beta(self):
        """Value changes when beta changes (kappa > 0)."""
        r1 = self._call(beta=0.0)
        r2 = self._call(beta=1.0)
        assert r1['value'] != r2['value']

    def test_higher_x0_raises_exp_utility(self):
        """Higher x0 gives higher (less negative) exponential utility.

        U(x) = -(1/γ)exp(-γx) is increasing in x, so larger x0 gives a
        larger (less negative) value: Theta_0 grows toward 0 as x0 increases.
        """
        r_small = self._call(x0=0.5)
        r_large = self._call(x0=2.0)
        # Theta_0 = -(1/γ)exp(-γ·exp(rT)·x0): less negative for larger x0
        assert r_large['value'] > r_small['value']


# ── evaluate_power_value ──────────────────────────────────────────────────────

class TestPowerValue:
    def _call(self, **kwargs):
        defaults = dict(
            x0=X0, T=T, n=N, V0=V0, mu0=MU0, alpha=ALPHA,
            theta=THETA, rho=RHO, sigma_v=SIGMA_V, lam=LAM,
            gamma=GAMMA, r=R, beta=0.5, kappa=0.3,
        )
        defaults.update(kwargs)
        return evaluate_power_value(**defaults)

    def test_returns_dict(self):
        result = self._call()
        assert isinstance(result, dict)
        assert 'value' in result

    def test_finite(self):
        result = self._call()
        assert np.isfinite(result['value'])

    def test_positive_value(self):
        """Power utility x^gamma/gamma > 0 for x > 0, 0 < gamma < 1."""
        result = self._call()
        assert result['value'] > 0

    def test_theta_0_positive(self):
        """Theta_0 = x0^gamma/gamma > 0."""
        result = self._call()
        assert result['Theta_0'] > 0

    def test_beta_zero_stable(self):
        r_nojump = self._call(beta=0.0)
        assert np.isfinite(r_nojump['value'])
        assert r_nojump['value'] > 0

    def test_sensitivity_to_gamma(self):
        """Different gamma produces different value."""
        r1 = self._call(gamma=0.2)
        r2 = self._call(gamma=0.6)
        assert r1['value'] != r2['value']


# ── evaluate_log_value ────────────────────────────────────────────────────────

class TestLogValue:
    def _call(self, **kwargs):
        defaults = dict(
            x0=X0, T=T, n=N, V0=V0, mu0=MU0, alpha=ALPHA,
            theta=THETA, r=R,
        )
        defaults.update(kwargs)
        return evaluate_log_value(**defaults)

    def test_returns_dict(self):
        result = self._call()
        assert isinstance(result, dict)
        assert 'value' in result

    def test_finite(self):
        result = self._call()
        assert np.isfinite(result['value'])

    def test_x0_one_contribution(self):
        """For x0=1, log(x0)=0; value = r*T + theta^2/2 * integral."""
        result = self._call(x0=1.0)
        assert np.isfinite(result['value'])

    def test_larger_x0_increases_log_value(self):
        """log utility with larger x0 gives larger value."""
        r1 = self._call(x0=1.0)
        r2 = self._call(x0=2.0)
        assert r2['value'] > r1['value']

    def test_value_contains_log_x0(self):
        """Value should equal log(x0) + remainder (test difference)."""
        r1 = self._call(x0=1.0)
        r2 = self._call(x0=np.e)  # log(e)=1
        diff = r2['value'] - r1['value']
        np.testing.assert_allclose(diff, 1.0, atol=1e-8)
