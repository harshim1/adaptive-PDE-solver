"""
Basic example of using the adaptive PDE solver for option pricing.

This example demonstrates:
1. Setting up option specifications
2. Configuring market parameters
3. Running the adaptive solver
4. Analyzing results and Greeks
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import solve_option, OptionSpec, MarketParams, SolverConfig, AdaptivitySpec


def main():
    """Run a basic example of adaptive PDE option pricing."""
    
    print("=== Adaptive PDE Solver for Option Pricing ===\n")
    
    # 1. Define the option
    option_spec = OptionSpec(
        option_type="call",
        strike=100.0,
        maturity=1.0,  # 1 year
        spot=100.0     # At-the-money
    )
    
    print(f"Option: {option_spec.option_type.upper()} option")
    print(f"Strike: ${option_spec.strike}")
    print(f"Maturity: {option_spec.maturity} years")
    print(f"Spot: ${option_spec.spot}")
    
    # 2. Set market parameters
    market_params = MarketParams(
        risk_free_rate=0.05,    # 5% risk-free rate
        volatility=0.2,         # 20% volatility
        dividend_yield=0.0      # No dividends
    )
    
    print(f"\nMarket Parameters:")
    print(f"Risk-free rate: {market_params.risk_free_rate:.1%}")
    print(f"Volatility: {market_params.volatility:.1%}")
    print(f"Dividend yield: {market_params.dividend_yield:.1%}")
    
    # 3. Configure solver
    solver_config = SolverConfig(
        initial_cells=100,
        domain_min=0.0,
        domain_max=300.0,       # 3x the strike
        grading_type="log",     # Log-spacing for better resolution near strike
        time_tolerance=1e-4,
        max_time_steps=1000
    )
    
    # 4. Configure adaptivity
    adaptivity_spec = AdaptivitySpec(
        estimator_type="combined",  # Use combined error estimator
        refine_threshold=0.1,       # Refine when error > 0.1
        coarsen_threshold=0.01,     # Coarsen when error < 0.01
        max_refinement_levels=5,
        min_cell_size=1e-6,
        max_cell_size=10.0
    )
    
    print(f"\nSolver Configuration:")
    print(f"Initial cells: {solver_config.initial_cells}")
    print(f"Domain: [{solver_config.domain_min}, {solver_config.domain_max}]")
    print(f"Grading: {solver_config.grading_type}")
    print(f"Adaptivity: {adaptivity_spec.estimator_type} estimator")
    
    # 5. Solve the option
    print(f"\nSolving option...")
    solution = solve_option(
        option_spec=option_spec,
        market_params=market_params,
        solver_config=solver_config,
        adaptivity_spec=adaptivity_spec,
        target_tolerance=1e-4,
        max_time=10.0
    )
    
    # 6. Display results
    print(f"\n=== RESULTS ===")
    print(f"Option Price: ${solution.option_price:.4f}")
    print(f"Runtime: {solution.runtime_stats['total_time']:.2f} seconds")
    print(f"Time steps: {solution.runtime_stats['time_steps']}")
    print(f"Final cells: {solution.runtime_stats['final_cells']}")
    print(f"Mesh adaptations: {solution.runtime_stats['mesh_adaptations']}")
    
    # 7. Analyze Greeks at spot
    spot_idx = np.argmin(np.abs(solution.mesh_points - option_spec.spot))
    delta = solution.greeks['Delta'][spot_idx]
    gamma = solution.greeks['Gamma'][spot_idx]
    theta = solution.greeks['Theta'][spot_idx]
    
    print(f"\nGreeks at spot (S=${option_spec.spot}):")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Theta: {theta:.4f}")
    
    # 8. Create visualizations
    create_plots(solution, option_spec)
    
    return solution


def create_plots(solution, option_spec):
    """Create visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Option value surface
    axes[0, 0].plot(solution.mesh_points, solution.solution_surface, 'b-', linewidth=2)
    axes[0, 0].axvline(x=option_spec.spot, color='r', linestyle='--', alpha=0.7, label=f'Spot=${option_spec.spot}')
    axes[0, 0].axvline(x=option_spec.strike, color='g', linestyle='--', alpha=0.7, label=f'Strike=${option_spec.strike}')
    axes[0, 0].set_xlabel('Underlying Price (S)')
    axes[0, 0].set_ylabel('Option Value')
    axes[0, 0].set_title('Option Value Surface')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Delta
    axes[0, 1].plot(solution.mesh_points, solution.greeks['Delta'], 'g-', linewidth=2)
    axes[0, 1].axvline(x=option_spec.spot, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Underlying Price (S)')
    axes[0, 1].set_ylabel('Delta')
    axes[0, 1].set_title('Delta (∂V/∂S)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Gamma
    axes[1, 0].plot(solution.mesh_points, solution.greeks['Gamma'], 'm-', linewidth=2)
    axes[1, 0].axvline(x=option_spec.spot, color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Underlying Price (S)')
    axes[1, 0].set_ylabel('Gamma')
    axes[1, 0].set_title('Gamma (∂²V/∂S²)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Mesh spacing
    dx = np.diff(solution.mesh_points)
    axes[1, 1].semilogy(solution.mesh_points[:-1], dx, 'k-', linewidth=1)
    axes[1, 1].axvline(x=option_spec.spot, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Underlying Price (S)')
    axes[1, 1].set_ylabel('Cell Size (log scale)')
    axes[1, 1].set_title('Adaptive Mesh Spacing')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('option_pricing_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlots saved as 'option_pricing_results.png'")


def compare_adaptive_vs_uniform():
    """Compare adaptive vs uniform mesh results."""
    
    print("\n=== ADAPTIVE vs UNIFORM MESH COMPARISON ===")
    
    # Common parameters
    option_spec = OptionSpec(option_type="call", strike=100.0, maturity=1.0, spot=100.0)
    market_params = MarketParams(risk_free_rate=0.05, volatility=0.2)
    
    # Uniform mesh (no adaptivity)
    uniform_config = SolverConfig(initial_cells=200, grading_type="uniform")
    uniform_adaptivity = AdaptivitySpec(refine_threshold=1e10)  # Effectively no refinement
    
    uniform_solution = solve_option(
        option_spec=option_spec,
        market_params=market_params,
        solver_config=uniform_config,
        adaptivity_spec=uniform_adaptivity,
        target_tolerance=1e-4,
        max_time=10.0
    )
    
    # Adaptive mesh
    adaptive_config = SolverConfig(initial_cells=100, grading_type="log")
    adaptive_adaptivity = AdaptivitySpec(estimator_type="combined", refine_threshold=0.1)
    
    adaptive_solution = solve_option(
        option_spec=option_spec,
        market_params=market_params,
        solver_config=adaptive_config,
        adaptivity_spec=adaptive_adaptivity,
        target_tolerance=1e-4,
        max_time=10.0
    )
    
    print(f"Uniform mesh:")
    print(f"  Price: ${uniform_solution.option_price:.4f}")
    print(f"  Runtime: {uniform_solution.runtime_stats['total_time']:.2f}s")
    print(f"  Cells: {uniform_solution.runtime_stats['final_cells']}")
    
    print(f"Adaptive mesh:")
    print(f"  Price: ${adaptive_solution.option_price:.4f}")
    print(f"  Runtime: {adaptive_solution.runtime_stats['total_time']:.2f}s")
    print(f"  Cells: {adaptive_solution.runtime_stats['final_cells']}")
    
    price_diff = abs(adaptive_solution.option_price - uniform_solution.option_price)
    print(f"Price difference: ${price_diff:.6f}")
    
    return uniform_solution, adaptive_solution


if __name__ == "__main__":
    # Run basic example
    solution = main()
    
    # Run comparison
    uniform_sol, adaptive_sol = compare_adaptive_vs_uniform()
