"""
Tests for src/volterra_sim.py — variance path simulation, roughness diagnostics,
and positivity checks.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.volterra_sim import simulate_variance_paths, compute_roughness_diagnostics, check_positivity


# ── Shared fixture parameters ────────────────────────────────────────────────
V0     = np.array([0.01, 0.03])
mu0    = np.array([2.0, 2.5])
lam    = np.array([0.2, 0.6])
sigma_v = np.array([0.4, 0.2])
alpha  = np.array([0.6, 0.9])


# ── simulate_variance_paths ──────────────────────────────────────────────────

class TestSimulateVariancePaths:
    def test_output_shape(self):
        V, tg = simulate_variance_paths(
            n=20, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.5, kappa=0.3, n_paths=3, rng_seed=0,
        )
        assert V.shape == (3, 21, 2)
        assert tg.shape == (21,)

    def test_initial_condition(self):
        """V[:, 0, :] == V0 for all paths."""
        V, _ = simulate_variance_paths(
            n=20, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.5, kappa=0.3, n_paths=5, rng_seed=1,
        )
        np.testing.assert_array_equal(V[:, 0, :], np.broadcast_to(V0, (5, 2)))

    def test_time_grid_bounds(self):
        _, tg = simulate_variance_paths(
            n=50, T=2.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.0, kappa=0.0, n_paths=1, rng_seed=2,
        )
        np.testing.assert_allclose(tg[0], 0.0)
        np.testing.assert_allclose(tg[-1], 2.0)

    def test_no_jumps_gives_finite_paths(self):
        V, _ = simulate_variance_paths(
            n=50, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.0, kappa=0.0, n_paths=5, enable_jumps=False,
            rng_seed=3,
        )
        assert np.all(np.isfinite(V))

    def test_with_jumps_finite(self):
        V, _ = simulate_variance_paths(
            n=50, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.5, kappa=0.3, n_paths=5, enable_jumps=True,
            rng_seed=4,
        )
        assert np.all(np.isfinite(V))

    def test_reproducibility(self):
        """Same seed => identical paths."""
        V1, _ = simulate_variance_paths(
            n=20, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.5, kappa=0.3, n_paths=3, rng_seed=42,
        )
        V2, _ = simulate_variance_paths(
            n=20, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.5, kappa=0.3, n_paths=3, rng_seed=42,
        )
        np.testing.assert_array_equal(V1, V2)

    def test_jump_increases_increment_kurtosis(self):
        """Jumps create heavy-tailed increments (higher kurtosis) vs no-jump paths.

        The compensator removes the mean drift contribution of jumps, so mean
        levels are comparable. The correct jump signature is that sudden upward
        spikes produce fatter-tailed step distributions: higher excess kurtosis.
        """
        from scipy.stats import kurtosis as excess_kurtosis

        V_jump, _ = simulate_variance_paths(
            n=100, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=2.0, kappa=1.0, n_paths=20, enable_jumps=True,
            rng_seed=10,
        )
        V_nojump, _ = simulate_variance_paths(
            n=100, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.0, kappa=0.0, n_paths=20, enable_jumps=False,
            rng_seed=10,
        )
        kurt_jump   = excess_kurtosis(np.diff(V_jump,  axis=1).ravel())
        kurt_nojump = excess_kurtosis(np.diff(V_nojump, axis=1).ravel())
        assert kurt_jump > kurt_nojump, (
            f"Expected higher kurtosis with jumps ({kurt_jump:.3f}) "
            f"than without ({kurt_nojump:.3f})"
        )

    def test_single_path(self):
        V, tg = simulate_variance_paths(
            n=10, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.0, kappa=0.0, n_paths=1, rng_seed=5,
        )
        assert V.shape == (1, 11, 2)


# ── compute_roughness_diagnostics ────────────────────────────────────────────

class TestRoughnessDiagnostics:
    def _get_paths(self, seed=99):
        V, tg = simulate_variance_paths(
            n=100, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.5, kappa=0.3, n_paths=10, rng_seed=seed,
        )
        return V, tg

    def test_returns_dict_for_each_component(self):
        V, tg = self._get_paths()
        diag = compute_roughness_diagnostics(V, tg)
        assert 'component_1' in diag
        assert 'component_2' in diag

    def test_mai_positive(self):
        V, tg = self._get_paths()
        diag = compute_roughness_diagnostics(V, tg)
        for comp_key in diag:
            for key, val in diag[comp_key].items():
                if 'MAI' in key:
                    assert val >= 0

    def test_realized_qv_positive(self):
        V, tg = self._get_paths()
        diag = compute_roughness_diagnostics(V, tg)
        for comp_key in diag:
            assert diag[comp_key]['realized_QV'] >= 0

    def test_hurst_surrogate_range(self):
        """Hurst surrogate should be between 0 and 2 in typical regimes."""
        V, tg = self._get_paths()
        diag = compute_roughness_diagnostics(V, tg)
        for comp_key in diag:
            hs = diag[comp_key].get('hurst_surrogate')
            if hs is not None:
                assert 0.0 < hs < 2.0, f"Hurst surrogate {hs} out of expected range"

    def test_rougher_component_lower_hurst(self):
        """Component 1 (alpha=0.6, H=0.1) should be rougher than component 2 (alpha=0.9)."""
        # Use many paths for robustness
        V, tg = simulate_variance_paths(
            n=200, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.0, kappa=0.0, n_paths=30, rng_seed=77,
        )
        diag = compute_roughness_diagnostics(V, tg)
        hs1 = diag['component_1'].get('hurst_surrogate')
        hs2 = diag['component_2'].get('hurst_surrogate')
        if hs1 is not None and hs2 is not None:
            assert hs1 < hs2, f"Expected H1={hs1:.4f} < H2={hs2:.4f}"


# ── check_positivity ─────────────────────────────────────────────────────────

class TestCheckPositivity:
    def test_returns_report(self):
        V, _ = simulate_variance_paths(
            n=20, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.0, kappa=0.0, n_paths=5, rng_seed=6,
        )
        report = check_positivity(V)
        assert 'component_1' in report
        assert 'component_2' in report

    def test_min_value_key(self):
        V, _ = simulate_variance_paths(
            n=20, T=1.0, V0=V0, mu0=mu0, lam=lam, sigma_v=sigma_v,
            alpha=alpha, beta=0.0, kappa=0.0, n_paths=5, rng_seed=7,
        )
        report = check_positivity(V)
        for comp in report.values():
            assert 'min_value' in comp
            assert 'n_negative' in comp
