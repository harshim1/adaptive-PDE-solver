"""
Time integration schemes for PDEs.

This module provides:
- Crank-Nicolson implicit time stepping
- Adaptive time step control
- Linear system solvers
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class CrankNicolsonIntegrator:
    """
    Crank-Nicolson time integrator for Black-Scholes PDE.
    
    The Crank-Nicolson scheme is:
    (I - dt/2 * L) * V^{n+1} = (I + dt/2 * L) * V^n + boundary_terms
    
    where L is the spatial discretization operator.
    """
    
    def __init__(self, discretizer, solver_type: str = "direct"):
        """
        Initialize Crank-Nicolson integrator.
        
        Args:
            discretizer: BlackScholesDiscretizer instance
            solver_type: Type of linear solver ("direct" or "iterative")
        """
        self.discretizer = discretizer
        self.solver_type = solver_type
        
    def time_step(self, V_old: np.ndarray, mesh, dt: float, 
                  option_type: str, K: float, T: float, t: float) -> np.ndarray:
        """
        Perform one time step using Crank-Nicolson scheme.
        
        Args:
            V_old: Solution at time t
            mesh: Mesh object
            dt: Time step size
            option_type: "call" or "put"
            K: Strike price
            T: Maturity time
            t: Current time
            
        Returns:
            Solution at time t + dt
        """
        # Assemble matrices
        A = self.discretizer.assemble_matrix(mesh, dt)
        B = self.discretizer.assemble_rhs_matrix(mesh, dt)
        
        # Compute right-hand side
        rhs = B @ V_old
        
        # Apply boundary conditions to RHS
        V_bc = self.discretizer.apply_boundary_conditions(
            np.zeros_like(V_old), mesh, option_type, K, T, t + dt
        )
        
        # Modify RHS to account for boundary conditions
        # For Dirichlet BCs, we set the corresponding rows to the boundary values
        rhs[0] = V_bc[0]
        rhs[-1] = V_bc[-1]
        
        # Solve linear system
        if self.solver_type == "direct":
            V_new = spsolve(A, rhs)
        else:
            # For iterative solvers, we would use scipy.sparse.linalg methods
            # like gmres, bicgstab, etc. with preconditioning
            V_new = spsolve(A, rhs)  # Fallback to direct for now
        
        # Ensure non-negativity for option values
        V_new = np.maximum(V_new, 0.0)
        
        return V_new
    
    def adaptive_time_step(self, V_old: np.ndarray, mesh, dt_old: float,
                          option_type: str, K: float, T: float, t: float,
                          target_error: float = 1e-4, max_iterations: int = 5) -> Tuple[np.ndarray, float]:
        """
        Perform adaptive time step with error estimation.
        
        Args:
            V_old: Solution at time t
            mesh: Mesh object
            dt_old: Previous time step size
            option_type: "call" or "put"
            K: Strike price
            T: Maturity time
            t: Current time
            target_error: Target local truncation error
            max_iterations: Maximum number of adaptive iterations
            
        Returns:
            Tuple of (new_solution, new_time_step)
        """
        dt = dt_old
        iteration = 0
        
        while iteration < max_iterations:
            # Try full step
            V_full = self.time_step(V_old, mesh, dt, option_type, K, T, t)
            
            # Try two half steps
            dt_half = dt / 2
            V_half1 = self.time_step(V_old, mesh, dt_half, option_type, K, T, t)
            V_half2 = self.time_step(V_half1, mesh, dt_half, option_type, K, T, t + dt_half)
            
            # Estimate local truncation error
            error_estimate = np.linalg.norm(V_full - V_half2) / (np.linalg.norm(V_half2) + 1e-10)
            
            # Adjust time step based on error
            if error_estimate < target_error / 2:
                # Error is small, can increase time step
                dt_new = min(dt * 1.2, dt * (target_error / (error_estimate + 1e-10))**0.2)
                return V_full, dt_new
            elif error_estimate > target_error:
                # Error is too large, need to reduce time step
                dt = dt * (target_error / (error_estimate + 1e-10))**0.2
                iteration += 1
            else:
                # Error is acceptable
                return V_full, dt
        
        # If we've exceeded max iterations, return the best we have
        logger.warning(f"Adaptive time stepping exceeded max iterations ({max_iterations})")
        return V_full, dt


class AdaptiveTimeStepper:
    """
    Adaptive time stepping controller.
    
    Controls time step size based on:
    - Local truncation error estimates
    - Stability constraints
    - User-specified accuracy requirements
    """
    
    def __init__(self, integrator, initial_dt: float = 0.01, 
                 min_dt: float = 1e-6, max_dt: float = 0.1,
                 target_error: float = 1e-4):
        """
        Initialize adaptive time stepper.
        
        Args:
            integrator: Time integrator instance
            initial_dt: Initial time step size
            min_dt: Minimum allowed time step
            max_dt: Maximum allowed time step
            target_error: Target local truncation error
        """
        self.integrator = integrator
        self.initial_dt = initial_dt
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.target_error = target_error
        self.current_dt = initial_dt
        
    def choose_time_step(self, V: np.ndarray, mesh, t: float, T: float) -> float:
        """
        Choose appropriate time step size.
        
        Args:
            V: Current solution
            mesh: Mesh object
            t: Current time
            T: Final time
            
        Returns:
            Recommended time step size
        """
        # Time remaining
        time_remaining = T - t
        
        # Stability constraint (CFL-like condition for diffusion)
        # For Black-Scholes: dt < C * dx² / (σ² * S²)
        min_dx = mesh.cell_sizes.min()
        max_S = mesh.x.max()
        stability_dt = 0.5 * min_dx**2 / (self.integrator.discretizer.sigma**2 * max_S**2)
        
        # Accuracy constraint (based on time remaining)
        accuracy_dt = time_remaining / 10  # At least 10 steps remaining
        
        # Choose minimum of all constraints
        dt = min(self.current_dt, stability_dt, accuracy_dt, self.max_dt)
        dt = max(dt, self.min_dt)
        
        # Don't exceed final time
        dt = min(dt, time_remaining)
        
        return dt
    
    def step(self, V_old: np.ndarray, mesh, t: float, T: float,
             option_type: str, K: float) -> Tuple[np.ndarray, float, float]:
        """
        Perform one adaptive time step.
        
        Args:
            V_old: Solution at time t
            mesh: Mesh object
            t: Current time
            T: Final time
            option_type: "call" or "put"
            K: Strike price
            
        Returns:
            Tuple of (new_solution, new_time, new_time_step)
        """
        # Choose time step
        dt = self.choose_time_step(V_old, mesh, t, T)
        
        # Perform time step
        if hasattr(self.integrator, 'adaptive_time_step'):
            V_new, dt_used = self.integrator.adaptive_time_step(
                V_old, mesh, dt, option_type, K, T, t, self.target_error
            )
        else:
            V_new = self.integrator.time_step(V_old, mesh, dt, option_type, K, T, t)
            dt_used = dt
        
        # Update current time step for next iteration
        self.current_dt = dt_used
        
        return V_new, t + dt_used, dt_used
