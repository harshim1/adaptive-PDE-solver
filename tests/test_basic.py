"""
Basic tests for the adaptive PDE solver.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import solve_option, OptionSpec, MarketParams, SolverConfig, AdaptivitySpec
from adaptivity.mesh import Mesh, MeshConfig
from pde.discretization import BlackScholesDiscretizer


def test_mesh_creation():
    """Test basic mesh creation and properties."""
    config = MeshConfig(domain_min=0.0, domain_max=100.0, initial_cells=50)
    mesh = Mesh(config)
    
    assert len(mesh.x) == 51  # n_cells + 1
    assert mesh.n_cells == 50
    assert mesh.x[0] == 0.0
    assert mesh.x[-1] == 100.0
    assert len(mesh.cell_sizes) == 50


def test_mesh_refinement():
    """Test mesh refinement functionality."""
    config = MeshConfig(domain_min=0.0, domain_max=100.0, initial_cells=10)
    mesh = Mesh(config)
    
    # Refine first cell
    refined_mesh = mesh.refine_cells([0])
    
    assert refined_mesh.n_cells > mesh.n_cells
    assert refined_mesh.n_cells == mesh.n_cells + 1


def test_black_scholes_discretizer():
    """Test Black-Scholes discretizer."""
    discretizer = BlackScholesDiscretizer(r=0.05, sigma=0.2)
    
    config = MeshConfig(domain_min=0.0, domain_max=200.0, initial_cells=20)
    mesh = Mesh(config)
    
    # Test matrix assembly
    A = discretizer.assemble_matrix(mesh, dt=0.01)
    B = discretizer.assemble_rhs_matrix(mesh, dt=0.01)
    
    assert A.shape == (21, 21)  # n_nodes x n_nodes
    assert B.shape == (21, 21)


def test_european_call_option():
    """Test solving a European call option."""
    option_spec = OptionSpec(
        option_type="call",
        strike=100.0,
        maturity=1.0,
        spot=100.0
    )
    
    market_params = MarketParams(
        risk_free_rate=0.05,
        volatility=0.2
    )
    
    solver_config = SolverConfig(
        initial_cells=50,
        domain_max=300.0
    )
    
    adaptivity_spec = AdaptivitySpec(
        estimator_type="gradient",
        refine_threshold=0.2
    )
    
    # Solve the option
    solution = solve_option(
        option_spec=option_spec,
        market_params=market_params,
        solver_config=solver_config,
        adaptivity_spec=adaptivity_spec,
        target_tolerance=1e-3,
        max_time=5.0
    )
    
    # Basic checks
    assert solution.option_price > 0
    assert solution.option_price < option_spec.spot  # Call price < spot
    assert len(solution.solution_surface) == len(solution.mesh_points)
    assert "Delta" in solution.greeks
    assert "Gamma" in solution.greeks
    assert "Theta" in solution.greeks


def test_european_put_option():
    """Test solving a European put option."""
    option_spec = OptionSpec(
        option_type="put",
        strike=100.0,
        maturity=1.0,
        spot=100.0
    )
    
    market_params = MarketParams(
        risk_free_rate=0.05,
        volatility=0.2
    )
    
    solver_config = SolverConfig(
        initial_cells=50,
        domain_max=300.0
    )
    
    adaptivity_spec = AdaptivitySpec(
        estimator_type="gradient",
        refine_threshold=0.2
    )
    
    # Solve the option
    solution = solve_option(
        option_spec=option_spec,
        market_params=market_params,
        solver_config=solver_config,
        adaptivity_spec=adaptivity_spec,
        target_tolerance=1e-3,
        max_time=5.0
    )
    
    # Basic checks
    assert solution.option_price > 0
    assert solution.option_price < option_spec.strike  # Put price < strike
    assert len(solution.solution_surface) == len(solution.mesh_points)


def test_put_call_parity():
    """Test put-call parity relationship."""
    strike = 100.0
    spot = 100.0
    maturity = 1.0
    r = 0.05
    
    # Call option
    call_spec = OptionSpec(option_type="call", strike=strike, maturity=maturity, spot=spot)
    call_market = MarketParams(risk_free_rate=r, volatility=0.2)
    
    call_solution = solve_option(
        option_spec=call_spec,
        market_params=call_market,
        target_tolerance=1e-3,
        max_time=5.0
    )
    
    # Put option
    put_spec = OptionSpec(option_type="put", strike=strike, maturity=maturity, spot=spot)
    put_market = MarketParams(risk_free_rate=r, volatility=0.2)
    
    put_solution = solve_option(
        option_spec=put_spec,
        market_params=put_market,
        target_tolerance=1e-3,
        max_time=5.0
    )
    
    # Put-call parity: C - P = S - K*exp(-r*T)
    call_price = call_solution.option_price
    put_price = put_solution.option_price
    parity_value = spot - strike * np.exp(-r * maturity)
    
    # Check parity (with some tolerance for numerical errors)
    parity_error = abs((call_price - put_price) - parity_value)
    assert parity_error < 0.1, f"Put-call parity error: {parity_error}"


if __name__ == "__main__":
    pytest.main([__file__])
