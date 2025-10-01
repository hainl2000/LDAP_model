"""
CAGrad (Conflict-Averse Gradient descent) Optimizer Implementation

This module implements the CAGrad algorithm for multi-objective optimization,
specifically designed to handle conflicting gradients in multi-task learning scenarios.

Reference: Du, Y., Czarnecki, W. M., Jayakumar, S. M., Farajtabar, M., Pascanu, R., & Lakshminarayanan, B. (2018).
Adapting auxiliary losses using gradient similarity. arXiv preprint arXiv:1812.02224.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import torch.nn.functional as F


class CAGradOptimizer:
    """
    Conflict-Averse Gradient descent (CAGrad) optimizer for multi-objective optimization.
    
    CAGrad addresses the problem of conflicting gradients in multi-task learning by
    projecting gradients onto a conflict-free direction when conflicts are detected.
    
    Args:
        optimizer: Base optimizer (e.g., Adam, SGD)
        alpha: Regularization parameter for gradient projection (default: 0.5)
        rescale: Rescaling factor for gradient norms (default: 1)
        conflict_threshold: Cosine similarity threshold for conflict detection (default: 0.0)
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 alpha: float = 0.5,
                 rescale: float = 1.0,
                 conflict_threshold: float = 0.0,
                 task_priorities: Optional[Dict[str, float]] = None):
        self.optimizer = optimizer
        self.alpha = alpha
        self.rescale = rescale
        self.conflict_threshold = conflict_threshold
        self.task_priorities = task_priorities or {}
        
        # Store gradients for each objective
        self.objective_gradients = {}
        self.gradient_history = []
        
    def zero_grad(self):
        """Zero gradients of the base optimizer."""
        self.optimizer.zero_grad()
        
    def step(self, 
             losses: Dict[str, torch.Tensor], 
             model: nn.Module,
             retain_graph: bool = False) -> Dict[str, float]:
        """
        Perform CAGrad optimization step.
        
        Args:
            losses: Dictionary of individual loss components
            model: The neural network model
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dictionary containing conflict statistics and gradient information
        """
        # Step 1: Compute gradients for each objective
        objective_gradients = self._compute_objective_gradients(losses, model, retain_graph)
        
        # Step 2: Detect conflicts between gradients
        conflict_info = self._detect_conflicts(objective_gradients)
        
        # Step 3: Apply CAGrad if conflicts are detected
        if conflict_info['has_conflicts']:
            final_gradient = self._apply_cagrad(objective_gradients, conflict_info)
        else:
            # No conflicts: use simple weighted sum
            final_gradient = self._compute_weighted_sum(objective_gradients)
        
        # Step 4: Apply the final gradient to model parameters
        self._apply_gradient_to_model(final_gradient, model)
        
        # Step 5: Update using base optimizer
        self.optimizer.step()
        
        # Store gradient history for analysis
        self.gradient_history.append({
            'conflicts': conflict_info,
            'gradient_norms': {k: torch.norm(v).item() for k, v in objective_gradients.items()}
        })
        
        return conflict_info
    
    def _compute_objective_gradients(self, 
                                   losses: Dict[str, torch.Tensor], 
                                   model: nn.Module,
                                   retain_graph: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute gradients for each individual objective.
        
        Args:
            losses: Dictionary of loss components
            model: Neural network model
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dictionary of gradients for each objective
        """
        objective_gradients = {}
        
        for i, (loss_name, loss_value) in enumerate(losses.items()):
            # Zero gradients before computing each objective's gradient
            self.optimizer.zero_grad()
            
            # Compute gradient for this objective
            is_last = (i == len(losses) - 1)
            loss_value.backward(retain_graph=retain_graph or not is_last)
            
            # Collect gradients
            grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.view(-1))
                else:
                    grad_vec.append(torch.zeros_like(param.view(-1)))
            
            objective_gradients[loss_name] = torch.cat(grad_vec)
            
        return objective_gradients
    
    def _detect_conflicts(self, objective_gradients: Dict[str, torch.Tensor]) -> Dict:
        """
        Detect conflicts between objective gradients using cosine similarity.
        
        Args:
            objective_gradients: Dictionary of gradients for each objective
            
        Returns:
            Dictionary containing conflict detection results
        """
        gradient_names = list(objective_gradients.keys())
        n_objectives = len(gradient_names)
        
        # Compute pairwise cosine similarities
        similarities = {}
        conflicts = {}
        
        for i in range(n_objectives):
            for j in range(i + 1, n_objectives):
                name_i, name_j = gradient_names[i], gradient_names[j]
                grad_i, grad_j = objective_gradients[name_i], objective_gradients[name_j]
                
                # Compute cosine similarity
                cos_sim = F.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0)).item()
                similarities[f"{name_i}_{name_j}"] = cos_sim
                
                # Check for conflict (negative cosine similarity below threshold)
                conflicts[f"{name_i}_{name_j}"] = cos_sim < self.conflict_threshold
        
        has_conflicts = any(conflicts.values())
        
        return {
            'has_conflicts': has_conflicts,
            'similarities': similarities,
            'conflicts': conflicts,
            'n_conflicts': sum(conflicts.values())
        }
    
    def _apply_cagrad(self, 
                     objective_gradients: Dict[str, torch.Tensor], 
                     conflict_info: Dict) -> torch.Tensor:
        """
        Apply CAGrad algorithm to resolve gradient conflicts.
        
        The CAGrad algorithm projects conflicting gradients onto a conflict-free direction
        using a quadratic programming formulation.
        
        Args:
            objective_gradients: Dictionary of gradients for each objective
            conflict_info: Information about detected conflicts
            
        Returns:
            Final gradient after CAGrad projection
        """
        gradients = list(objective_gradients.values())
        n_objectives = len(gradients)
        
        # Stack gradients into matrix G where each column is a gradient
        G = torch.stack(gradients, dim=1)  # [grad_dim, n_objectives]
        
        # Compute Gram matrix (G^T G)
        GtG = torch.mm(G.t(), G)  # [n_objectives, n_objectives]
        
        # Solve for optimal weights using CAGrad formulation with task priorities
        # min ||Gw||^2 + alpha * ||w - p||^2
        # where p is the priority weight vector (defaults to uniform if no priorities)
        
        n = n_objectives
        
        # Create priority weights based on task_priorities
        if self.task_priorities:
            task_names = list(objective_gradients.keys())
            priority_weights = torch.tensor([
                self.task_priorities.get(task, 1.0) for task in task_names
            ], device=G.device, dtype=G.dtype)
            # Normalize priority weights to sum to 1
            priority_weights = priority_weights / priority_weights.sum()
        else:
            # Default to uniform weights
            priority_weights = torch.ones(n, device=G.device) / n
        
        # Add regularization term: GtG + alpha * I
        regularized_GtG = GtG + self.alpha * torch.eye(n, device=G.device)
        
        # Solve: (GtG + alpha * I) * w = alpha * priority_weights
        try:
            weights = torch.linalg.solve(regularized_GtG, self.alpha * priority_weights)
        except RuntimeError:
            # Fallback to pseudo-inverse if matrix is singular
            weights = torch.linalg.pinv(regularized_GtG) @ (self.alpha * priority_weights)
        
        # Ensure weights are non-negative and sum to 1
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum() + 1e-8)
        
        # Compute final gradient as weighted combination
        final_gradient = torch.mm(G, weights.unsqueeze(1)).squeeze(1)
        
        # Apply rescaling
        if self.rescale != 1.0:
            final_gradient = final_gradient * self.rescale
            
        return final_gradient
    
    def _compute_weighted_sum(self, objective_gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute simple weighted sum of gradients when no conflicts are detected.
        
        Args:
            objective_gradients: Dictionary of gradients for each objective
            
        Returns:
            Weighted sum of gradients
        """
        gradients = list(objective_gradients.values())
        n_objectives = len(gradients)
        
        # Use uniform weights when no conflicts
        weight = 1.0 / n_objectives
        final_gradient = sum(grad * weight for grad in gradients)
        
        return final_gradient
    
    def _apply_gradient_to_model(self, final_gradient: torch.Tensor, model: nn.Module):
        """
        Apply the computed final gradient to model parameters.
        
        Args:
            final_gradient: The final gradient vector to apply
            model: Neural network model
        """
        # Zero existing gradients
        self.optimizer.zero_grad()
        
        # Apply final gradient to parameters
        start_idx = 0
        for param in model.parameters():
            if param.grad is not None:
                param_size = param.numel()
                param.grad = final_gradient[start_idx:start_idx + param_size].view(param.shape)
                start_idx += param_size
    
    def get_conflict_statistics(self) -> Dict:
        """
        Get statistics about gradient conflicts over training history.
        
        Returns:
            Dictionary containing conflict statistics
        """
        if not self.gradient_history:
            return {}
        
        total_steps = len(self.gradient_history)
        conflict_steps = sum(1 for step in self.gradient_history if step['conflicts']['has_conflicts'])
        
        avg_conflicts_per_step = np.mean([step['conflicts']['n_conflicts'] for step in self.gradient_history])
        
        return {
            'total_steps': total_steps,
            'conflict_steps': conflict_steps,
            'conflict_ratio': conflict_steps / total_steps if total_steps > 0 else 0.0,
            'avg_conflicts_per_step': avg_conflicts_per_step
        }
    
    def state_dict(self) -> Dict:
        """Return state dictionary for saving."""
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'alpha': self.alpha,
            'rescale': self.rescale,
            'conflict_threshold': self.conflict_threshold,
            'gradient_history': self.gradient_history
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dictionary."""
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.alpha = state_dict['alpha']
        self.rescale = state_dict['rescale']
        self.conflict_threshold = state_dict['conflict_threshold']
        self.gradient_history = state_dict.get('gradient_history', [])


def create_cagrad_optimizer(model: nn.Module, 
                          lr: float = 1e-3,
                          weight_decay: float = 0.0,
                          alpha: float = 0.5,
                          rescale: float = 1.0,
                          conflict_threshold: float = 0.0,
                          base_optimizer: str = 'adam',
                          task_priorities: Optional[Dict[str, float]] = None) -> CAGradOptimizer:
    """
    Factory function to create CAGrad optimizer with specified base optimizer.
    
    Args:
        model: Neural network model
        lr: Learning rate
        weight_decay: Weight decay
        alpha: CAGrad regularization parameter
        rescale: Gradient rescaling factor
        conflict_threshold: Conflict detection threshold
        base_optimizer: Base optimizer type ('adam', 'sgd', 'adamw')
        
    Returns:
        CAGradOptimizer instance
    """
    if base_optimizer.lower() == 'adam':
        base_opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif base_optimizer.lower() == 'sgd':
        base_opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif base_optimizer.lower() == 'adamw':
        base_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported base optimizer: {base_optimizer}")
    
    return CAGradOptimizer(
        optimizer=base_opt,
        alpha=alpha,
        rescale=rescale,
        conflict_threshold=conflict_threshold,
        task_priorities=task_priorities
    )