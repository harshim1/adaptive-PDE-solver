"""
Error estimation for adaptive mesh refinement.

This module provides:
- Residual-based error estimators
- Gradient-based indicators
- Dual-weighted residual estimators
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ErrorEstimator:
    """
    Base class for error estimators.
    """
    
    def compute_indicators(self, mesh, solution: np.ndarray, 
                          discretizer, dt: float) -> np.ndarray:
        """
        Compute error indicators for each cell.
        
        Args:
            mesh: Mesh object
            solution: Current solution values
            discretizer: PDE discretizer
            dt: Current time step
            
        Returns:
            Array of error indicators for each cell
        """
        raise NotImplementedError


class ResidualErrorEstimator(ErrorEstimator):
    """
    Residual-based a posteriori error estimator.
    
    Computes local residuals of the PDE to estimate discretization error.
    """
    
    def __init__(self, norm_type: str = "L2"):
        """
        Initialize residual error estimator.
        
        Args:
            norm_type: Type of norm to use ("L2", "L1", "Linf")
        """
        self.norm_type = norm_type
    
    def compute_indicators(self, mesh, solution: np.ndarray, 
                          discretizer, dt: float) -> np.ndarray:
        """
        Compute residual-based error indicators.
        
        For Black-Scholes PDE: ∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
        
        The residual is computed as the difference between the continuous
        PDE operator applied to the discrete solution and zero.
        """
        n_cells = mesh.n_cells
        indicators = np.zeros(n_cells)
        
        x = mesh.x
        dx = mesh.cell_sizes
        
        # Compute residuals for each cell
        for i in range(n_cells):
            # Get solution values at cell boundaries
            V_left = solution[i]
            V_right = solution[i + 1]
            
            # Cell center
            S_center = 0.5 * (x[i] + x[i + 1])
            dx_cell = dx[i]
            
            # Approximate derivatives using finite differences
            if i == 0:
                # Left boundary cell
                dV_dS = (V_right - V_left) / dx_cell
                d2V_dS2 = (solution[i + 2] - 2*V_right + V_left) / (dx_cell**2)
            elif i == n_cells - 1:
                # Right boundary cell
                dV_dS = (V_right - V_left) / dx_cell
                d2V_dS2 = (V_right - 2*V_left + solution[i - 1]) / (dx_cell**2)
            else:
                # Interior cell
                dV_dS = (V_right - V_left) / dx_cell
                d2V_dS2 = ((V_right - V_left) / dx_cell - 
                          (V_left - solution[i - 1]) / dx[i - 1]) / (0.5 * (dx[i - 1] + dx_cell))
            
            # Compute PDE residual
            # For steady-state or implicit schemes, we approximate ∂V/∂t ≈ 0
            # In practice, you'd use the actual time derivative from the time integrator
            residual = (0.5 * discretizer.sigma**2 * S_center**2 * d2V_dS2 + 
                       discretizer.r * S_center * dV_dS - 
                       discretizer.r * V_left)
            
            # Scale by cell size for proper error estimation
            if self.norm_type == "L2":
                indicators[i] = abs(residual) * np.sqrt(dx_cell)
            elif self.norm_type == "L1":
                indicators[i] = abs(residual) * dx_cell
            else:  # Linf
                indicators[i] = abs(residual)
        
        return indicators


class GradientErrorEstimator(ErrorEstimator):
    """
    Gradient-based error indicator.
    
    Uses solution gradients as a proxy for discretization error.
    High gradients indicate regions where refinement may be needed.
    """
    
    def __init__(self, gradient_threshold: float = 0.1):
        """
        Initialize gradient error estimator.
        
        Args:
            gradient_threshold: Threshold for gradient-based refinement
        """
        self.gradient_threshold = gradient_threshold
    
    def compute_indicators(self, mesh, solution: np.ndarray, 
                          discretizer, dt: float) -> np.ndarray:
        """
        Compute gradient-based error indicators.
        """
        n_cells = mesh.n_cells
        indicators = np.zeros(n_cells)
        
        dx = mesh.cell_sizes
        
        # Compute gradients for each cell
        for i in range(n_cells):
            # Gradient approximation
            gradient = abs(solution[i + 1] - solution[i]) / dx[i]
            
            # Scale by cell size
            indicators[i] = gradient * dx[i]
        
        return indicators


class GammaErrorEstimator(ErrorEstimator):
    """
    Gamma (second derivative) based error indicator.
    
    High Gamma values indicate regions where the solution has high curvature,
    which typically require finer discretization.
    """
    
    def __init__(self, gamma_threshold: float = 0.01):
        """
        Initialize Gamma error estimator.
        
        Args:
            gamma_threshold: Threshold for Gamma-based refinement
        """
        self.gamma_threshold = gamma_threshold
    
    def compute_indicators(self, mesh, solution: np.ndarray, 
                          discretizer, dt: float) -> np.ndarray:
        """
        Compute Gamma-based error indicators.
        """
        n_cells = mesh.n_cells
        indicators = np.zeros(n_cells)
        
        dx = mesh.cell_sizes
        
        # Compute Gamma for each cell
        for i in range(n_cells):
            if i == 0:
                # Left boundary
                gamma = abs(solution[i + 2] - 2*solution[i + 1] + solution[i]) / (dx[i]**2)
            elif i == n_cells - 1:
                # Right boundary
                gamma = abs(solution[i + 1] - 2*solution[i] + solution[i - 1]) / (dx[i]**2)
            else:
                # Interior
                gamma = abs(((solution[i + 1] - solution[i]) / dx[i] - 
                           (solution[i] - solution[i - 1]) / dx[i - 1]) / 
                          (0.5 * (dx[i - 1] + dx[i])))
            
            # Scale by cell size
            indicators[i] = gamma * dx[i]
        
        return indicators


class CombinedErrorEstimator(ErrorEstimator):
    """
    Combined error estimator using multiple indicators.
    
    Combines residual, gradient, and Gamma indicators with weights.
    """
    
    def __init__(self, residual_weight: float = 1.0, 
                 gradient_weight: float = 0.5, 
                 gamma_weight: float = 0.3):
        """
        Initialize combined error estimator.
        
        Args:
            residual_weight: Weight for residual indicator
            gradient_weight: Weight for gradient indicator
            gamma_weight: Weight for Gamma indicator
        """
        self.residual_estimator = ResidualErrorEstimator()
        self.gradient_estimator = GradientErrorEstimator()
        self.gamma_estimator = GammaErrorEstimator()
        
        self.residual_weight = residual_weight
        self.gradient_weight = gradient_weight
        self.gamma_weight = gamma_weight
    
    def compute_indicators(self, mesh, solution: np.ndarray, 
                          discretizer, dt: float) -> np.ndarray:
        """
        Compute combined error indicators.
        """
        # Get individual indicators
        residual_indicators = self.residual_estimator.compute_indicators(
            mesh, solution, discretizer, dt
        )
        gradient_indicators = self.gradient_estimator.compute_indicators(
            mesh, solution, discretizer, dt
        )
        gamma_indicators = self.gamma_estimator.compute_indicators(
            mesh, solution, discretizer, dt
        )
        
        # Normalize indicators
        if np.max(residual_indicators) > 0:
            residual_indicators = residual_indicators / np.max(residual_indicators)
        if np.max(gradient_indicators) > 0:
            gradient_indicators = gradient_indicators / np.max(gradient_indicators)
        if np.max(gamma_indicators) > 0:
            gamma_indicators = gamma_indicators / np.max(gamma_indicators)
        
        # Combine with weights
        combined_indicators = (self.residual_weight * residual_indicators + 
                             self.gradient_weight * gradient_indicators + 
                             self.gamma_weight * gamma_indicators)
        
        return combined_indicators


def create_error_estimator(estimator_type: str, **kwargs) -> ErrorEstimator:
    """
    Factory function to create error estimators.
    
    Args:
        estimator_type: Type of estimator ("residual", "gradient", "gamma", "combined")
        **kwargs: Additional arguments for the estimator
        
    Returns:
        ErrorEstimator instance
    """
    if estimator_type == "residual":
        return ResidualErrorEstimator(**kwargs)
    elif estimator_type == "gradient":
        return GradientErrorEstimator(**kwargs)
    elif estimator_type == "gamma":
        return GammaErrorEstimator(**kwargs)
    elif estimator_type == "combined":
        return CombinedErrorEstimator(**kwargs)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")
