"""
Mesh data structure for adaptive finite difference grids.

This module provides a flexible mesh representation that supports:
- Non-uniform grid spacing
- h-refinement (cell subdivision)
- Solution projection between meshes
- Conservative interpolation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MeshConfig:
    """Configuration for mesh generation."""
    domain_min: float = 0.0
    domain_max: float = 200.0
    initial_cells: int = 100
    min_cell_size: float = 1e-6
    max_cell_size: float = 10.0
    grading_type: str = "uniform"  # "uniform", "log", "geometric"
    grading_factor: float = 1.1


class Mesh:
    """
    Adaptive mesh for 1D finite difference discretization.
    
    The mesh stores:
    - Node coordinates (x)
    - Cell centers and sizes
    - Refinement markers
    - Solution values at nodes
    """
    
    def __init__(self, config: MeshConfig):
        self.config = config
        self.x = self._generate_initial_mesh()
        self.n_cells = len(self.x) - 1
        self.refinement_markers = np.zeros(self.n_cells, dtype=bool)
        self.solution = None
        self._cell_sizes = None
        self._cell_centers = None
        
    def _generate_initial_mesh(self) -> np.ndarray:
        """Generate initial mesh based on configuration."""
        if self.config.grading_type == "uniform":
            return np.linspace(
                self.config.domain_min, 
                self.config.domain_max, 
                self.config.initial_cells + 1
            )
        elif self.config.grading_type == "log":
            # Log-spacing for asset price (S) domain
            log_min = np.log(max(self.config.domain_min, 1e-10))
            log_max = np.log(self.config.domain_max)
            log_x = np.linspace(log_min, log_max, self.config.initial_cells + 1)
            return np.exp(log_x)
        elif self.config.grading_type == "geometric":
            # Geometric progression
            ratio = (self.config.domain_max / self.config.domain_min) ** (1.0 / self.config.initial_cells)
            x = np.zeros(self.config.initial_cells + 1)
            x[0] = self.config.domain_min
            for i in range(1, self.config.initial_cells + 1):
                x[i] = x[i-1] * ratio
            return x
        else:
            raise ValueError(f"Unknown grading type: {self.config.grading_type}")
    
    @property
    def cell_sizes(self) -> np.ndarray:
        """Get cell sizes (dx)."""
        if self._cell_sizes is None:
            self._cell_sizes = np.diff(self.x)
        return self._cell_sizes
    
    @property
    def cell_centers(self) -> np.ndarray:
        """Get cell center coordinates."""
        if self._cell_centers is None:
            self._cell_centers = 0.5 * (self.x[:-1] + self.x[1:])
        return self._cell_centers
    
    def refine_cells(self, cell_indices: List[int]) -> 'Mesh':
        """
        Create a new mesh with specified cells refined.
        
        Args:
            cell_indices: List of cell indices to refine
            
        Returns:
            New mesh with refined cells
        """
        if not cell_indices:
            return self
        
        # Create new node array
        new_x = []
        for i, x_val in enumerate(self.x[:-1]):  # Exclude last point
            new_x.append(x_val)
            if i in cell_indices:
                # Add midpoint for refinement
                midpoint = 0.5 * (self.x[i] + self.x[i+1])
                new_x.append(midpoint)
        
        # Add the last point
        new_x.append(self.x[-1])
        
        # Create new mesh
        new_config = MeshConfig(
            domain_min=self.config.domain_min,
            domain_max=self.config.domain_max,
            initial_cells=len(new_x) - 1,
            min_cell_size=self.config.min_cell_size,
            max_cell_size=self.config.max_cell_size,
            grading_type=self.config.grading_type,
            grading_factor=self.config.grading_factor
        )
        
        new_mesh = Mesh(new_config)
        new_mesh.x = np.array(new_x)
        new_mesh.n_cells = len(new_x) - 1
        new_mesh._cell_sizes = None  # Force recomputation
        new_mesh._cell_centers = None
        
        return new_mesh
    
    def coarsen_cells(self, cell_indices: List[int]) -> 'Mesh':
        """
        Create a new mesh with specified cells coarsened.
        
        Args:
            cell_indices: List of cell indices to coarsen
            
        Returns:
            New mesh with coarsened cells
        """
        if not cell_indices:
            return self
        
        # For coarsening, we remove every other node in refined regions
        # This is a simplified approach - more sophisticated coarsening
        # would require tracking refinement history
        keep_nodes = np.ones(len(self.x), dtype=bool)
        
        for i in cell_indices:
            if i < len(self.x) - 1:
                # Remove the right node of the cell (except for boundary)
                if i + 1 < len(self.x) - 1:
                    keep_nodes[i + 1] = False
        
        new_x = self.x[keep_nodes]
        
        new_config = MeshConfig(
            domain_min=self.config.domain_min,
            domain_max=self.config.domain_max,
            initial_cells=len(new_x) - 1,
            min_cell_size=self.config.min_cell_size,
            max_cell_size=self.config.max_cell_size,
            grading_type=self.config.grading_type,
            grading_factor=self.config.grading_factor
        )
        
        new_mesh = Mesh(new_config)
        new_mesh.x = new_x
        new_mesh.n_cells = len(new_x) - 1
        new_mesh._cell_sizes = None
        new_mesh._cell_centers = None
        
        return new_mesh
    
    def project_solution(self, old_mesh: 'Mesh', old_solution: np.ndarray) -> np.ndarray:
        """
        Project solution from old mesh to current mesh using linear interpolation.
        
        Args:
            old_mesh: Source mesh
            old_solution: Solution values on source mesh
            
        Returns:
            Interpolated solution on current mesh
        """
        from scipy.interpolate import interp1d
        
        # Create interpolation function
        interp_func = interp1d(
            old_mesh.x, 
            old_solution, 
            kind='linear', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        
        # Interpolate to new mesh
        new_solution = interp_func(self.x)
        
        # Ensure monotonicity for option values (no negative values)
        new_solution = np.maximum(new_solution, 0.0)
        
        return new_solution
    
    def get_refinement_candidates(self, error_indicators: np.ndarray, 
                                refine_threshold: float) -> List[int]:
        """
        Get cell indices that should be refined based on error indicators.
        
        Args:
            error_indicators: Error indicator for each cell
            refine_threshold: Threshold for refinement
            
        Returns:
            List of cell indices to refine
        """
        candidates = []
        for i, error in enumerate(error_indicators):
            if error > refine_threshold:
                # Check if cell is not too small
                if self.cell_sizes[i] > self.config.min_cell_size:
                    candidates.append(i)
        
        return candidates
    
    def get_coarsening_candidates(self, error_indicators: np.ndarray, 
                                coarsen_threshold: float) -> List[int]:
        """
        Get cell indices that should be coarsened based on error indicators.
        
        Args:
            error_indicators: Error indicator for each cell
            coarsen_threshold: Threshold for coarsening
            
        Returns:
            List of cell indices to coarsen
        """
        candidates = []
        for i, error in enumerate(error_indicators):
            if error < coarsen_threshold:
                # Check if cell is not too large
                if self.cell_sizes[i] < self.config.max_cell_size:
                    candidates.append(i)
        
        return candidates
    
    def adapt(self, error_indicators: np.ndarray, 
              refine_threshold: float, 
              coarsen_threshold: float) -> 'Mesh':
        """
        Create adapted mesh based on error indicators.
        
        Args:
            error_indicators: Error indicator for each cell
            refine_threshold: Threshold for refinement
            coarsen_threshold: Threshold for coarsening
            
        Returns:
            New adapted mesh
        """
        refine_cells = self.get_refinement_candidates(error_indicators, refine_threshold)
        coarsen_cells = self.get_coarsening_candidates(error_indicators, coarsen_threshold)
        
        logger.info(f"Refining {len(refine_cells)} cells, coarsening {len(coarsen_cells)} cells")
        
        # First refine
        if refine_cells:
            mesh = self.refine_cells(refine_cells)
        else:
            mesh = self
        
        # Then coarsen (on the refined mesh)
        if coarsen_cells:
            # Map coarsening candidates to new mesh indices
            # This is simplified - in practice, you'd need proper mapping
            new_coarsen_cells = []
            for old_idx in coarsen_cells:
                if old_idx < mesh.n_cells:
                    new_coarsen_cells.append(old_idx)
            
            if new_coarsen_cells:
                mesh = mesh.coarsen_cells(new_coarsen_cells)
        
        return mesh
    
    def __repr__(self) -> str:
        return (f"Mesh(n_cells={self.n_cells}, "
                f"domain=[{self.x[0]:.2f}, {self.x[-1]:.2f}], "
                f"min_dx={self.cell_sizes.min():.2e}, "
                f"max_dx={self.cell_sizes.max():.2e})")
