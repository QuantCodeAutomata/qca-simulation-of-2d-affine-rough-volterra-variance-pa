# 2D Affine Rough-Volterra Variance Simulation

Numerical implementation of the two-dimensional **affine Volterra square-root
variance process with exogenous upward jumps**, reproducing the numerical
experiments in Section 4 of the paper on rough-Volterra stochastic control.

## Overview

The model governs a variance factor `V = (V¹, V²)` via a semi-integrated
Euler–Maruyama scheme (Equation 63) driven by fractional kernels
`Kᵢ(t) = t^(αᵢ−1) / Γ(αᵢ)` with `α₁ = 0.6`, `α₂ = 0.9`
(Hurst exponents `H₁ ≈ 0.1`, `H₂ = 0.4`). Upward jumps arrive as a
compound-Poisson process with Gaussian marks.

Optimal portfolio strategies for exponential, power, and log utility are
derived from Riccati–Volterra equations solved by a fractional
Adams–Bashforth–Moulton predictor–corrector scheme.

## Repository Structure

```
├── src/                   Core library modules
│   ├── config.py          ModelParams dataclass — all paper parameters
│   ├── kernels.py         Power kernels Kᵢ and Volterra weight matrix
│   ├── jumps.py           Poisson arrivals, Gaussian marks, compensator
│   ├── gaussian_volterra.py  Brownian increment approximation of G^i_{k,l}
│   ├── volterra_sim.py    Semi-integrated Euler scheme (Eq. 63)
│   ├── riccati.py         ABM predictor–corrector for Riccati–Volterra
│   ├── strategy.py        Optimal control maps π*(t) from ψ
│   ├── value_functions.py Exponential / power / log value formulas
│   ├── wealth_sim.py      Self-financing wealth SDE + dominance test
│   └── plotting.py        Matplotlib helpers for all figures
│
├── exp/                   Experiment entry-points (run independently)
│   ├── exp1_variance_paths.py    Five sample paths, jump/no-jump, roughness
│   ├── exp2_riccati_volterra.py  Riccati ψ curves, sign checks, refinement
│   ├── exp3_optimal_strategy.py  Strategy π* for T∈{1,5,10}, γ∈{0.2,0.6,0.8}
│   ├── exp4_value_functions.py   Value function evaluation + sensitivity
│   └── exp5_mc_optimality.py     Monte Carlo dominance check vs perturbations
│
├── tests/                 pytest test suite (121 tests)
│   ├── test_kernels.py
│   ├── test_jumps.py
│   ├── test_volterra_sim.py
│   ├── test_riccati.py
│   ├── test_strategy.py
│   ├── test_value_functions.py
│   └── test_wealth_sim.py
│
├── results/               All output figures (.png) and metrics (.json / .md)
└── data/                  (empty — pure simulation, no market data required)
```

## Parameters Manifest

| Symbol | Value | Notes |
|---|---|---|
| `d` | 2 | Dimension |
| `T` | 1 | Horizon (years) |
| `n` | 200 | Grid steps |
| `V₀` | (0.01, 0.03) | Initial variance |
| `μ₀` | (2.0, 2.5) | Drift vector |
| `D` | diag(−0.2, −0.6) | Mean-reversion matrix |
| `Σ` | diag(−0.7, −0.55) | Leverage/loading |
| `σᵛ` | (0.4, 0.2) | Vol-of-vol |
| `α₁, α₂` | 0.6, 0.9 | Kernel exponents (H=0.1, H=0.4) |
| `θ` | (0.1, 0.1) | Sharpe vector |
| `β` | 0.5 | **Missing in paper — implementer choice** |
| `κ` | 0.3 | **Missing in paper — implementer choice** |
| `ζ` | (0, 0) | **Missing in paper — defaulted to zero** |
| `r` | 0.02 | Risk-free rate |

## Experiments

### Exp 1 — Variance path simulation
Five sample paths per component with jumps (β=0.5, κ=0.3) and without.
Roughness confirmed: Hurst surrogates H̃₁≈0.23, H̃₂≈0.65.
Zero negative values (full-truncation fix applied and documented).

### Exp 2 — Riccati–Volterra solver
Adams–Bashforth–Moulton predictor–corrector for exponential and power utility.
Sign conditions confirmed:
- Exponential: ψ ∈ [−0.0245, 0] ✓ (non-positive)
- Power: ψ ∈ [0, 0.00119] ✓ (non-negative)
Grid convergence checked at n ∈ {200, 400, 800}.

### Exp 3 — Optimal strategy curves
Deterministic π*(t) for T ∈ {1, 5, 10}, γ ∈ {0.2, 0.6, 0.8}.
Strategy is near-constant over [0,T] and scales as 1/γ (exponential) or
1/(1−γ) (power); rougher component (α=0.6) shows sharper time variation.

### Exp 4 — Value function evaluation
Semi-closed-form values at x₀=1, T=1, γ=0.2:
- Exponential: V ≈ −4.028, Power: V ≈ 5.035, Log: V ≈ 0.029
Sensitivity to (β, κ) is weak at small jump intensities; ζ sensitivity
is linear for exponential utility.

### Exp 5 — Monte Carlo dominance
M=500 paths test optimal vs perturbed controls (c ∈ {0.8, 0.9, 1.1, 1.2}).
- Power γ=0.2: all controls within 0.4% of optimal (flat landscape).
- Power γ=0.6: optimal hits negative wealth in 1/500 paths; conservative
  perturbations (c<1) avoid this due to smaller positions.
- Exponential: c>1 perturbations yield slightly higher mean utility,
  consistent with a concave-but-flat landscape near the continuous-time
  optimum under discrete approximation.

## Running the Code

```bash
# Install dependencies (already installed in the project environment)
pip install numpy scipy matplotlib

# Run all experiments sequentially
python exp/exp1_variance_paths.py
python exp/exp2_riccati_volterra.py
python exp/exp3_optimal_strategy.py
python exp/exp4_value_functions.py
python exp/exp5_mc_optimality.py

# Run tests
pytest tests/ -v
```

## Key Design Decisions

1. **Volterra weights** use the analytic formula
   `C^i_{k,l} = [(t_k−t_{l−1})^αᵢ − (t_k−t_l)^αᵢ] / Γ(αᵢ+1)`.
2. **Gaussian Volterra integrals** `G^i_{k,l}` are approximated by
   `K_i(t_k − t_{l-1}) · ΔW_l · Δt` (documented approximation;
   exact covariance construction from cited references not reproduced).
3. **Positivity fix**: full truncation `sqrt(max(V, 0))` before each
   square-root evaluation. No negative paths observed under baseline parameters.
4. **Missing parameters** β, κ, ζ are recorded in `src/config.py` and this
   README; exact figure replication from the paper is impossible without them.
5. **Riccati time reversal**: `F(T−tⱼ, ψ(tⱼ))` — the backward-time argument
   is implemented as `T − grid[j]` throughout the solver.
