"""
Finite difference discretization for Black-Scholes PDE.

This module provides:
- Black-Scholes PDE discretization on non-uniform grids
- Boundary condition handling
- Matrix assembly for implicit time stepping
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BlackScholesDiscretizer:
    """
    Finite difference discretization of the Black-Scholes PDE:
    
    ∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
    
    For European options, this is a linear PDE.
    For American options, we add the early exercise constraint: V ≥ max(S-K, 0)
    """
    
    def __init__(self, r: float, sigma: float, q: float = 0.0):
        """
        Initialize Black-Scholes discretizer.
        
        Args:
            r: Risk-free interest rate
            sigma: Volatility
            q: Dividend yield (default: 0.0)
        """
        self.r = r
        self.sigma = sigma
        self.q = q
        
    def assemble_matrix(self, mesh, dt: float) -> sparse.csr_matrix:
        """
        Assemble the system matrix for implicit time stepping.
        
        For Crank-Nicolson: (I - dt/2 * L) * V^{n+1} = (I + dt/2 * L) * V^n
        
        where L is the spatial discretization operator.
        
        Args:
            mesh: Mesh object containing grid points
            dt: Time step size
            
        Returns:
            Sparse matrix representing (I - dt/2 * L)
        """
        n = len(mesh.x)
        x = mesh.x
        dx = mesh.cell_sizes
        
        # Pre-allocate matrix elements
        data = []
        row_indices = []
        col_indices = []
        
        # Interior points (i = 1, ..., n-2)
        for i in range(1, n-1):
            # Get cell sizes around point i
            dx_left = dx[i-1]   # Size of cell to the left
            dx_right = dx[i]    # Size of cell to the right
            
            # Coordinates
            S = x[i]
            
            # Finite difference coefficients for non-uniform grid
            # Second derivative: d²V/dS² ≈ (V_{i+1} - V_i)/dx_right - (V_i - V_{i-1})/dx_left
            #                    / ((dx_left + dx_right)/2)
            h_avg = 0.5 * (dx_left + dx_right)
            
            # Diffusion term: (1/2)σ²S² * d²V/dS²
            diff_coeff = 0.5 * self.sigma**2 * S**2 / h_avg
            
            # First derivative: dV/dS ≈ (V_{i+1} - V_{i-1}) / (dx_left + dx_right)
            # Convection term: rS * dV/dS
            conv_coeff = self.r * S / (dx_left + dx_right)
            
            # Reaction term: -rV
            react_coeff = -self.r
            
            # Assemble coefficients for V_{i-1}, V_i, V_{i+1}
            # For Crank-Nicolson: (I - dt/2 * L)
            
            # V_{i-1} coefficient
            coeff_left = -dt/2 * (diff_coeff / dx_left - conv_coeff)
            data.append(coeff_left)
            row_indices.append(i)
            col_indices.append(i-1)
            
            # V_i coefficient (diagonal)
            coeff_diag = 1.0 - dt/2 * (-2*diff_coeff/h_avg + react_coeff)
            data.append(coeff_diag)
            row_indices.append(i)
            col_indices.append(i)
            
            # V_{i+1} coefficient
            coeff_right = -dt/2 * (diff_coeff / dx_right + conv_coeff)
            data.append(coeff_right)
            row_indices.append(i)
            col_indices.append(i+1)
        
        # Boundary conditions
        # Left boundary (S = 0): V = 0 for call, V = K for put
        data.append(1.0)
        row_indices.append(0)
        col_indices.append(0)
        
        # Right boundary (S = S_max): V = S - K*exp(-r*T) for call, V = 0 for put
        # For now, use Dirichlet boundary condition
        data.append(1.0)
        row_indices.append(n-1)
        col_indices.append(n-1)
        
        # Create sparse matrix
        A = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        
        return A
    
    def assemble_rhs_matrix(self, mesh, dt: float) -> sparse.csr_matrix:
        """
        Assemble the right-hand side matrix for Crank-Nicolson.
        
        Returns (I + dt/2 * L) for the RHS of CN scheme.
        
        Args:
            mesh: Mesh object containing grid points
            dt: Time step size
            
        Returns:
            Sparse matrix representing (I + dt/2 * L)
        """
        n = len(mesh.x)
        x = mesh.x
        dx = mesh.cell_sizes
        
        # Pre-allocate matrix elements
        data = []
        row_indices = []
        col_indices = []
        
        # Interior points (i = 1, ..., n-2)
        for i in range(1, n-1):
            # Get cell sizes around point i
            dx_left = dx[i-1]
            dx_right = dx[i]
            
            # Coordinates
            S = x[i]
            
            # Finite difference coefficients (same as in assemble_matrix)
            h_avg = 0.5 * (dx_left + dx_right)
            diff_coeff = 0.5 * self.sigma**2 * S**2 / h_avg
            conv_coeff = self.r * S / (dx_left + dx_right)
            react_coeff = -self.r
            
            # For RHS: (I + dt/2 * L) - note the + sign
            
            # V_{i-1} coefficient
            coeff_left = dt/2 * (diff_coeff / dx_left - conv_coeff)
            data.append(coeff_left)
            row_indices.append(i)
            col_indices.append(i-1)
            
            # V_i coefficient (diagonal)
            coeff_diag = 1.0 + dt/2 * (-2*diff_coeff/h_avg + react_coeff)
            data.append(coeff_diag)
            row_indices.append(i)
            col_indices.append(i)
            
            # V_{i+1} coefficient
            coeff_right = dt/2 * (diff_coeff / dx_right + conv_coeff)
            data.append(coeff_right)
            row_indices.append(i)
            col_indices.append(i+1)
        
        # Boundary conditions (identity for Dirichlet)
        data.append(1.0)
        row_indices.append(0)
        col_indices.append(0)
        
        data.append(1.0)
        row_indices.append(n-1)
        col_indices.append(n-1)
        
        # Create sparse matrix
        B = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        
        return B
    
    def apply_boundary_conditions(self, V: np.ndarray, mesh, 
                                option_type: str, K: float, 
                                T: float, t: float) -> np.ndarray:
        """
        Apply boundary conditions to the solution vector.
        
        Args:
            V: Solution vector
            mesh: Mesh object
            option_type: "call" or "put"
            K: Strike price
            T: Maturity time
            t: Current time
            
        Returns:
            Solution vector with boundary conditions applied
        """
        V_bc = V.copy()
        
        if option_type.lower() == "call":
            # Call option boundary conditions
            V_bc[0] = 0.0  # V(0,t) = 0
            V_bc[-1] = mesh.x[-1] - K * np.exp(-self.r * (T - t))  # V(S_max, t) = S - K*exp(-r(T-t))
        elif option_type.lower() == "put":
            # Put option boundary conditions
            V_bc[0] = K * np.exp(-self.r * (T - t))  # V(0,t) = K*exp(-r(T-t))
            V_bc[-1] = 0.0  # V(S_max, t) = 0
        else:
            raise ValueError(f"Unknown option type: {option_type}")
        
        return V_bc
    
    def compute_greeks(self, V: np.ndarray, mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Greeks (Delta, Gamma, Theta) using finite differences.
        
        Args:
            V: Option values at grid points
            mesh: Mesh object
            
        Returns:
            Tuple of (Delta, Gamma, Theta) arrays
        """
        n = len(V)
        x = mesh.x
        dx = mesh.cell_sizes
        
        Delta = np.zeros(n)
        Gamma = np.zeros(n)
        
        # Interior points
        for i in range(1, n-1):
            dx_left = dx[i-1]
            dx_right = dx[i]
            h_avg = 0.5 * (dx_left + dx_right)
            
            # Delta: dV/dS
            Delta[i] = (V[i+1] - V[i-1]) / (dx_left + dx_right)
            
            # Gamma: d²V/dS²
            Gamma[i] = ((V[i+1] - V[i]) / dx_right - (V[i] - V[i-1]) / dx_left) / h_avg
        
        # Boundary points (use one-sided differences)
        Delta[0] = (V[1] - V[0]) / dx[0]
        Delta[-1] = (V[-1] - V[-2]) / dx[-1]
        
        Gamma[0] = (V[2] - 2*V[1] + V[0]) / (dx[0]**2)
        Gamma[-1] = (V[-1] - 2*V[-2] + V[-3]) / (dx[-1]**2)
        
        # Theta: -dV/dt (approximated using the PDE)
        Theta = np.zeros(n)
        for i in range(n):
            S = x[i]
            if i > 0 and i < n-1:
                # Use the PDE: ∂V/∂t = -(1/2)σ²S²∂²V/∂S² - rS∂V/∂S + rV
                Theta[i] = -(0.5 * self.sigma**2 * S**2 * Gamma[i] + 
                            self.r * S * Delta[i] - self.r * V[i])
        
        return Delta, Gamma, Theta
