"""
Main API for the adaptive PDE solver.

This module provides the high-level interface for solving option pricing PDEs
with adaptive mesh refinement.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from pde.discretization import BlackScholesDiscretizer
from pde.time_integrators import CrankNicolsonIntegrator, AdaptiveTimeStepper
from adaptivity.mesh import Mesh, MeshConfig
from adaptivity.estimators import create_error_estimator

logger = logging.getLogger(__name__)


@dataclass
class OptionSpec:
    """Specification for an option contract."""
    option_type: str  # "call" or "put"
    strike: float
    maturity: float
    spot: float = 100.0  # Current underlying price


@dataclass
class MarketParams:
    """Market parameters for Black-Scholes model."""
    risk_free_rate: float = 0.05
    volatility: float = 0.2
    dividend_yield: float = 0.0


@dataclass
class AdaptivitySpec:
    """Specification for adaptive mesh refinement."""
    estimator_type: str = "combined"
    refine_threshold: float = 0.1
    coarsen_threshold: float = 0.01
    max_refinement_levels: int = 5
    min_cell_size: float = 1e-6
    max_cell_size: float = 10.0


@dataclass
class SolverConfig:
    """Configuration for the PDE solver."""
    initial_cells: int = 100
    domain_min: float = 0.0
    domain_max: float = 300.0
    grading_type: str = "log"  # "uniform", "log", "geometric"
    time_tolerance: float = 1e-4
    max_time_steps: int = 1000
    use_adaptive_time: bool = True


@dataclass
class Solution:
    """Solution object containing results and metadata."""
    option_price: float
    solution_surface: np.ndarray
    mesh_points: np.ndarray
    greeks: Dict[str, np.ndarray]
    runtime_stats: Dict[str, Any]
    mesh_history: Optional[list] = None


class AdaptivePDESolver:
    """
    Main solver class for adaptive PDE option pricing.
    """
    
    def __init__(self, market_params: MarketParams, 
                 solver_config: SolverConfig,
                 adaptivity_spec: AdaptivitySpec):
        """
        Initialize the adaptive PDE solver.
        
        Args:
            market_params: Market parameters (r, sigma, q)
            solver_config: Solver configuration
            adaptivity_spec: Adaptivity configuration
        """
        self.market_params = market_params
        self.solver_config = solver_config
        self.adaptivity_spec = adaptivity_spec
        
        # Initialize components
        self.discretizer = BlackScholesDiscretizer(
            r=market_params.risk_free_rate,
            sigma=market_params.volatility,
            q=market_params.dividend_yield
        )
        
        self.integrator = CrankNicolsonIntegrator(self.discretizer)
        
        self.error_estimator = create_error_estimator(
            adaptivity_spec.estimator_type
        )
        
        # Initialize mesh
        mesh_config = MeshConfig(
            domain_min=solver_config.domain_min,
            domain_max=solver_config.domain_max,
            initial_cells=solver_config.initial_cells,
            min_cell_size=adaptivity_spec.min_cell_size,
            max_cell_size=adaptivity_spec.max_cell_size,
            grading_type=solver_config.grading_type
        )
        
        self.mesh = Mesh(mesh_config)
        
        # Initialize time stepper
        self.time_stepper = AdaptiveTimeStepper(
            self.integrator,
            initial_dt=0.01,
            target_error=solver_config.time_tolerance
        )
    
    def solve(self, option_spec: OptionSpec, 
              target_tolerance: float = 1e-4,
              max_time: float = 10.0) -> Solution:
        """
        Solve the option pricing PDE with adaptive mesh refinement.
        
        Args:
            option_spec: Option specification
            target_tolerance: Target accuracy tolerance
            max_time: Maximum runtime in seconds
            
        Returns:
            Solution object with results
        """
        import time
        start_time = time.time()
        
        logger.info(f"Solving {option_spec.option_type} option: K={option_spec.strike}, T={option_spec.maturity}")
        
        # Initialize solution
        V = self._initial_condition(option_spec)
        t = 0.0
        T = option_spec.maturity
        
        # Store mesh history for visualization
        mesh_history = [self.mesh]
        
        # Time stepping loop
        step_count = 0
        while t < T and step_count < self.solver_config.max_time_steps:
            # Check runtime limit
            if time.time() - start_time > max_time:
                logger.warning("Maximum runtime exceeded")
                break
            
            # Choose time step
            dt = self.time_stepper.choose_time_step(V, self.mesh, t, T)
            
            # Perform time step
            V_new, t_new, dt_used = self.time_stepper.step(
                V, self.mesh, t, T, option_spec.option_type, option_spec.strike
            )
            
            # Compute error indicators
            error_indicators = self.error_estimator.compute_indicators(
                self.mesh, V_new, self.discretizer, dt_used
            )
            
            # Check if mesh adaptation is needed
            max_error = np.max(error_indicators)
            if max_error > self.adaptivity_spec.refine_threshold:
                # Adapt mesh
                old_mesh = self.mesh
                self.mesh = self.mesh.adapt(
                    error_indicators,
                    self.adaptivity_spec.refine_threshold,
                    self.adaptivity_spec.coarsen_threshold
                )
                
                # Project solution to new mesh
                V_new = self.mesh.project_solution(old_mesh, V_new)
                
                # Store mesh history
                mesh_history.append(self.mesh)
                
                logger.info(f"Mesh adapted: {old_mesh.n_cells} -> {self.mesh.n_cells} cells")
            
            # Update solution and time
            V = V_new
            t = t_new
            step_count += 1
            
            # Log progress
            if step_count % 100 == 0:
                logger.info(f"Step {step_count}: t={t:.4f}, cells={self.mesh.n_cells}, max_error={max_error:.2e}")
        
        # Compute final option price at spot
        option_price = self._interpolate_at_spot(V, option_spec.spot)
        
        # Compute Greeks
        greeks = self._compute_greeks(V)
        
        # Runtime statistics
        runtime = time.time() - start_time
        runtime_stats = {
            "total_time": runtime,
            "time_steps": step_count,
            "final_cells": self.mesh.n_cells,
            "mesh_adaptations": len(mesh_history) - 1,
            "final_time": t
        }
        
        logger.info(f"Solution completed in {runtime:.2f}s, {step_count} steps, {self.mesh.n_cells} cells")
        
        return Solution(
            option_price=option_price,
            solution_surface=V,
            mesh_points=self.mesh.x,
            greeks=greeks,
            runtime_stats=runtime_stats,
            mesh_history=mesh_history
        )
    
    def _initial_condition(self, option_spec: OptionSpec) -> np.ndarray:
        """Set initial condition (payoff at maturity)."""
        V = np.zeros(len(self.mesh.x))
        
        if option_spec.option_type.lower() == "call":
            V = np.maximum(self.mesh.x - option_spec.strike, 0.0)
        elif option_spec.option_type.lower() == "put":
            V = np.maximum(option_spec.strike - self.mesh.x, 0.0)
        else:
            raise ValueError(f"Unknown option type: {option_spec.option_type}")
        
        return V
    
    def _interpolate_at_spot(self, V: np.ndarray, spot: float) -> float:
        """Interpolate solution at the current spot price."""
        from scipy.interpolate import interp1d
        
        interp_func = interp1d(
            self.mesh.x, V, 
            kind='linear', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        
        return float(interp_func(spot))
    
    def _compute_greeks(self, V: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute Greeks for the solution."""
        Delta, Gamma, Theta = self.discretizer.compute_greeks(V, self.mesh)
        
        return {
            "Delta": Delta,
            "Gamma": Gamma,
            "Theta": Theta
        }


def solve_option(option_spec: OptionSpec,
                market_params: MarketParams,
                solver_config: Optional[SolverConfig] = None,
                adaptivity_spec: Optional[AdaptivitySpec] = None,
                target_tolerance: float = 1e-4,
                max_time: float = 10.0) -> Solution:
    """
    High-level function to solve an option pricing problem.
    
    Args:
        option_spec: Option specification
        market_params: Market parameters
        solver_config: Solver configuration (optional)
        adaptivity_spec: Adaptivity configuration (optional)
        target_tolerance: Target accuracy tolerance
        max_time: Maximum runtime in seconds
        
    Returns:
        Solution object
    """
    if solver_config is None:
        solver_config = SolverConfig()
    
    if adaptivity_spec is None:
        adaptivity_spec = AdaptivitySpec()
    
    solver = AdaptivePDESolver(market_params, solver_config, adaptivity_spec)
    return solver.solve(option_spec, target_tolerance, max_time)
