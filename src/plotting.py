"""
Visualization utilities for 2D affine rough-Volterra variance simulation.

All plots are saved to the results/ directory in PNG format.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server environment
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Dict, List

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_variance_paths(
    V: np.ndarray,
    time_grid: np.ndarray,
    title_suffix: str = "",
    filename: str = "exp1_variance_paths.png",
    alpha_params: Optional[list] = None,
) -> None:
    """
    Plot five sample paths for each variance component (Figure 1-type).

    Parameters
    ----------
    V : np.ndarray
        Shape (n_paths, n+1, d) variance paths.
    time_grid : np.ndarray
        Shape (n+1,) time grid.
    title_suffix : str
        Additional text for plot title.
    filename : str
        Output filename.
    alpha_params : list, optional
        Kernel exponents for labels.
    """
    n_paths, n_plus_1, d = V.shape
    fig, axes = plt.subplots(1, d, figsize=(14, 5), sharey=False)
    if d == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.5, n_paths))

    for i in range(d):
        ax = axes[i]
        hurst = (alpha_params[i] - 0.5) if alpha_params else None
        label_h = f" (H={hurst:.1f})" if hurst is not None else ""
        for p in range(n_paths):
            ax.plot(time_grid, V[p, :, i], lw=0.8, alpha=0.85, color=colors[p],
                    label=f"Path {p+1}" if i == 0 else None)
        ax.set_xlabel("Time $t$", fontsize=12)
        ax.set_ylabel(f"$V_t^{{{i+1}}}$", fontsize=12)
        ax.set_title(f"Variance Component {i+1}{label_h}", fontsize=12)
        ax.set_xlim(0, time_grid[-1])
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

    suptitle = f"2D Affine Rough-Volterra Variance Paths"
    if title_suffix:
        suptitle += f" — {title_suffix}"
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_jump_comparison(
    V_jump: np.ndarray,
    V_nojump: np.ndarray,
    time_grid: np.ndarray,
    filename: str = "exp1_jump_vs_nojump.png",
    alpha_params: Optional[list] = None,
) -> None:
    """
    Compare jump vs no-jump sample paths for each component.
    """
    d = V_jump.shape[2]
    fig, axes = plt.subplots(2, d, figsize=(14, 8), sharex=True)

    titles = ["With Jumps", "No Jumps"]
    for row, (V, title) in enumerate([(V_jump, "With Jumps"), (V_nojump, "No Jumps")]):
        for i in range(d):
            ax = axes[row, i]
            hurst = (alpha_params[i] - 0.5) if alpha_params else None
            label_h = f" (H={hurst:.1f})" if hurst is not None else ""
            for p in range(min(3, V.shape[0])):
                ax.plot(time_grid, V[p, :, i], lw=0.8, alpha=0.85)
            ax.set_ylabel(f"$V_t^{{{i+1}}}$", fontsize=11)
            if row == 0:
                ax.set_title(f"Component {i+1}{label_h}", fontsize=11)
            if row == 1:
                ax.set_xlabel("Time $t$", fontsize=11)
            ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', color='darkblue')

    fig.suptitle("Jump vs No-Jump Variance Paths", fontsize=14)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roughness_comparison(
    V_rough: np.ndarray,
    V_smooth: np.ndarray,
    time_grid: np.ndarray,
    alpha_rough: list,
    alpha_smooth: list,
    filename: str = "exp1_roughness_comparison.png",
) -> None:
    """
    Compare variance paths with different kernel exponents (roughness comparison).
    """
    d = V_rough.shape[2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for idx, (V, alphas, label) in enumerate([
        (V_rough, alpha_rough, "Rougher (lower alpha)"),
        (V_smooth, alpha_smooth, "Smoother (higher alpha)"),
    ]):
        ax = axes[idx]
        for p in range(min(5, V.shape[0])):
            for i in range(d):
                style = '-' if i == 0 else '--'
                ax.plot(time_grid, V[p, :, i], lw=0.7, alpha=0.8,
                        linestyle=style,
                        label=f"Comp {i+1} (alpha={alphas[i]})" if p == 0 else None)
        ax.set_xlabel("Time $t$", fontsize=11)
        ax.set_ylabel("Variance", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=8)

    fig.suptitle("Roughness Comparison: Varying Kernel Exponents", fontsize=14)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_riccati_solutions(
    psi_exp: np.ndarray,
    psi_pow: np.ndarray,
    time_grid: np.ndarray,
    gamma: float,
    filename: str = "exp2_riccati_solutions.png",
) -> None:
    """
    Plot psi^1 and psi^2 for exponential and power utility (Figure 2-type).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, psi, title, color in zip(
        axes,
        [psi_exp, psi_pow],
        ["Exponential Utility", "Power Utility"],
        ['tab:blue', 'tab:orange'],
    ):
        ax.plot(time_grid, psi[:, 0], color=color, lw=2.0, label=r"$\psi^1(t)$")
        ax.plot(time_grid, psi[:, 1], color=color, lw=2.0, ls='--', label=r"$\psi^2(t)$")
        ax.axhline(0, color='k', lw=0.8, ls=':')
        ax.set_xlabel("Time $t$", fontsize=12)
        ax.set_ylabel(r"$\psi^i(t)$", fontsize=12)
        ax.set_title(f"{title} (γ={gamma})", fontsize=12)
        ax.legend(fontsize=10)

    fig.suptitle("Riccati-Volterra Solutions", fontsize=14)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_strategy_curves(
    results: dict,
    filename: str = "exp3_strategy_curves.png",
) -> None:
    """
    Plot optimal strategy curves for multiple horizons and risk aversions.

    Parameters
    ----------
    results : dict
        Nested dict: results[utility][T][gamma] = (time_grid, pi, psi)
    """
    utilities = list(results.keys())
    n_util = len(utilities)
    n_comp = 2

    fig, axes = plt.subplots(n_util, n_comp, figsize=(14, 5 * n_util))
    if n_util == 1:
        axes = axes[np.newaxis, :]

    colors_T = {'1': 'tab:blue', '5': 'tab:orange', '10': 'tab:green'}
    ls_gamma = {0.2: '-', 0.6: '--', 0.8: ':'}

    for u_idx, utility in enumerate(utilities):
        for i in range(n_comp):
            ax = axes[u_idx, i]
            for T_key, T_data in results[utility].items():
                for gamma, (tg, pi, _) in T_data.items():
                    color = colors_T.get(str(int(T_key)), 'gray')
                    ls = ls_gamma.get(gamma, '-')
                    ax.plot(tg, pi[:, i], lw=1.5, color=color, ls=ls,
                            label=f"T={T_key}, γ={gamma}")
            ax.set_xlabel("Time $t$", fontsize=11)
            ax.set_ylabel(fr"$\pi^{{{i+1}}}_t$", fontsize=11)
            ax.set_title(f"{utility.capitalize()} Utility — Component {i+1}", fontsize=11)
            if i == 0:
                ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Optimal Strategy Curves", fontsize=14)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dominance_results(
    dominance: dict,
    utility: str,
    filename: str = "exp5_dominance.png",
) -> None:
    """
    Bar chart of expected utilities for different control scales.
    """
    scales = sorted(dominance.keys())
    means = [dominance[c]['mean_utility'] for c in scales]
    errors = [1.96 * dominance[c]['std_error'] for c in scales]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [str(c) for c in scales], means, yerr=errors,
        capsize=5, color=['tab:orange' if c == 1.0 else 'tab:blue' for c in scales],
        alpha=0.8,
    )
    ax.set_xlabel("Control Scale $c$", fontsize=12)
    ax.set_ylabel("Expected Utility", fontsize=12)
    ax.set_title(f"MC Dominance: {utility.capitalize()} Utility (c=1.0 = optimal)", fontsize=12)
    ax.axhline(dominance.get(1.0, {}).get('mean_utility', 0), color='r',
               lw=1.2, ls='--', label='Optimal level')
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_value_sensitivity(
    sensitivity: dict,
    beta_grid: list,
    kappa_grid: list,
    utility: str,
    filename: str = "exp4_sensitivity.png",
) -> None:
    """
    Heatmap of value function sensitivity over (beta, kappa) grid.
    """
    vals = np.full((len(beta_grid), len(kappa_grid)), np.nan)
    for i, b in enumerate(beta_grid):
        for j, k in enumerate(kappa_grid):
            key = (round(b, 4), round(k, 4))
            v = sensitivity.get(key, np.nan)
            if isinstance(v, (int, float)):
                vals[i, j] = v

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(vals, aspect='auto', origin='lower',
                   extent=[kappa_grid[0], kappa_grid[-1],
                           beta_grid[0], beta_grid[-1]])
    plt.colorbar(im, ax=ax, label='Value Function')
    ax.set_xlabel("kappa", fontsize=12)
    ax.set_ylabel("beta", fontsize=12)
    ax.set_title(f"Value Sensitivity: {utility.capitalize()} Utility", fontsize=12)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
