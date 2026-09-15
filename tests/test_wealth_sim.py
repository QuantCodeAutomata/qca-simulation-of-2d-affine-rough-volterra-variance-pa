"""
Tests for src/wealth_sim.py — Monte Carlo wealth simulation and expected
utility estimation for exp_5 dominance checks.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wealth_sim import (
    simulate_wealth_paths,
    compute_expected_utility,
    dominance_test,
)


# Shared parameters (small for speed)
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
N       = 30
M       = 50        # small M for fast tests


def _common_wealth(utility='power', beta=0.0, kappa=0.0, c=1.0):
    return simulate_wealth_paths(
        M=M, n=N, T=T, x0=X0, V0=V0, mu0=MU0, lam=LAM,
        sigma_v=SIGMA_V, alpha=ALPHA, theta=THETA, rho=RHO,
        gamma=GAMMA, r=R, beta=beta, kappa=kappa,
        utility=utility, control_scale=c, base_seed=0,
    )


# ── simulate_wealth_paths ────────────────────────────────────────────────────

class TestSimulateWealthPaths:
    def test_output_shapes(self):
        X_T, V_paths = _common_wealth()
        assert X_T.shape == (M,)
        assert V_paths.shape == (M, N + 1, 2)

    def test_wealth_mostly_positive_power(self):
        """Power utility — most wealth paths should stay positive."""
        X_T, _ = _common_wealth(utility='power')
        frac_positive = np.mean(X_T > 0)
        assert frac_positive > 0.5

    def test_wealth_finite(self):
        X_T, _ = _common_wealth()
        assert np.all(np.isfinite(X_T))

    def test_reproducible_with_same_seed(self):
        X1, _ = _common_wealth()
        X2, _ = _common_wealth()
        np.testing.assert_array_equal(X1, X2)

    def test_different_seed_gives_different_paths(self):
        X1, _ = simulate_wealth_paths(
            M=M, n=N, T=T, x0=X0, V0=V0, mu0=MU0, lam=LAM,
            sigma_v=SIGMA_V, alpha=ALPHA, theta=THETA, rho=RHO,
            gamma=GAMMA, r=R, beta=0.0, kappa=0.0,
            utility='power', control_scale=1.0, base_seed=0,
        )
        X2, _ = simulate_wealth_paths(
            M=M, n=N, T=T, x0=X0, V0=V0, mu0=MU0, lam=LAM,
            sigma_v=SIGMA_V, alpha=ALPHA, theta=THETA, rho=RHO,
            gamma=GAMMA, r=R, beta=0.0, kappa=0.0,
            utility='power', control_scale=1.0, base_seed=99,
        )
        assert not np.allclose(X1, X2)

    def test_scale_zero_is_risk_free(self):
        """control_scale=0 means no risky investment; wealth grows only at r."""
        X_T, _ = simulate_wealth_paths(
            M=M, n=N, T=T, x0=X0, V0=V0, mu0=MU0, lam=LAM,
            sigma_v=SIGMA_V, alpha=ALPHA, theta=THETA, rho=RHO,
            gamma=GAMMA, r=R, beta=0.0, kappa=0.0,
            utility='power', control_scale=0.0, base_seed=0,
        )
        # Without risky investment, exact solution: X_T = X0 * exp(r*T)
        expected = X0 * np.exp(R * T)
        # Euler approximation may differ slightly
        np.testing.assert_allclose(X_T, expected, rtol=0.05)

    def test_log_utility_wealth(self):
        X_T, _ = _common_wealth(utility='log')
        assert np.all(np.isfinite(X_T))

    def test_exponential_utility_wealth(self):
        X_T, _ = _common_wealth(utility='exponential')
        assert np.all(np.isfinite(X_T))


# ── compute_expected_utility ──────────────────────────────────────────────────

class TestComputeExpectedUtility:
    def test_power_positive_mean(self):
        X_T = np.ones(50) * 1.5  # all positive wealth
        mean_u, se, ci_l, ci_h = compute_expected_utility(X_T, GAMMA, 'power')
        expected = 1.5 ** GAMMA / GAMMA
        np.testing.assert_allclose(mean_u, expected, rtol=1e-10)
        assert se == 0.0  # all identical values → zero variance

    def test_log_formula(self):
        X_T = np.ones(20) * np.e
        mean_u, se, ci_l, ci_h = compute_expected_utility(X_T, GAMMA, 'log')
        np.testing.assert_allclose(mean_u, 1.0, atol=1e-10)

    def test_exponential_negative(self):
        X_T = np.ones(20) * 1.0
        mean_u, se, ci_l, ci_h = compute_expected_utility(X_T, GAMMA, 'exponential')
        assert mean_u < 0

    def test_confidence_interval_contains_mean(self):
        X_T, _ = _common_wealth(utility='power')
        mean_u, se, ci_l, ci_h = compute_expected_utility(X_T, GAMMA, 'power')
        assert ci_l <= mean_u <= ci_h

    def test_non_positive_wealth_handled(self):
        """Non-positive wealth should not raise; returns finite mean."""
        X_T = np.array([0.5, -0.1, 0.8, 1.2])
        mean_u, se, ci_l, ci_h = compute_expected_utility(X_T, GAMMA, 'power')
        assert np.isfinite(mean_u)

    def test_ci_width_inversely_proportional_to_sqrt_M(self):
        """Halving M should roughly double standard error."""
        rng = np.random.default_rng(42)
        X1 = rng.lognormal(0, 0.2, 400)
        X2 = X1[:100]
        _, se1, _, _ = compute_expected_utility(X1, GAMMA, 'power')
        _, se2, _, _ = compute_expected_utility(X2, GAMMA, 'power')
        # se2/se1 ≈ sqrt(400/100) = 2
        ratio = se2 / se1
        assert 1.5 < ratio < 3.0

    def test_returns_four_values(self):
        X_T = np.ones(10) * 1.0
        result = compute_expected_utility(X_T, GAMMA, 'power')
        assert len(result) == 4


# ── dominance_test ────────────────────────────────────────────────────────────

class TestDominanceTest:
    def _run(self, utility='power', M_small=30):
        return dominance_test(
            M=M_small, n=N, T=T, x0=X0, V0=V0, mu0=MU0, lam=LAM,
            sigma_v=SIGMA_V, alpha=ALPHA, theta=THETA, rho=RHO,
            gamma=GAMMA, r=R, beta=0.0, kappa=0.0,
            utility=utility, perturbations=[0.8, 1.0, 1.2],
            base_seed=0,
        )

    def test_returns_dict_with_all_scales(self):
        results = self._run()
        assert set(results.keys()) == {0.8, 1.0, 1.2}

    def test_each_entry_has_required_keys(self):
        results = self._run()
        required = {'mean_utility', 'std_error', 'ci_low', 'ci_high',
                    'n_negative_wealth', 'dominates_optimal'}
        for c, res in results.items():
            assert required.issubset(res.keys()), f"Missing keys for c={c}"

    def test_optimal_dominates_optimal_is_none(self):
        """dominates_optimal for c=1.0 should be None."""
        results = self._run()
        assert results[1.0]['dominates_optimal'] is None

    def test_finite_utility_estimates(self):
        results = self._run()
        for c, res in results.items():
            assert np.isfinite(res['mean_utility']), f"Non-finite utility for c={c}"

    def test_power_utility_positive_wealth(self):
        """No negative-wealth issues with conservative no-jump setup."""
        results = self._run(utility='power')
        assert results[1.0]['n_negative_wealth'] == 0

    def test_dominance_flag_not_all_true(self):
        """Not every perturbed control dominates; we expect local optimality."""
        results = self._run()
        non_opt = {c: r for c, r in results.items() if c != 1.0}
        dominates = [r['dominates_optimal'] for r in non_opt.values()]
        # The local optimum should cause at least one perturbation to be inferior
        # (This may fail with tiny M; relax to: not all perturbations dominate)
        assert not all(d is True for d in dominates)
