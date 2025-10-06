"""
Focal Loss Implementation for lncRNA-Disease Association Prediction

This module implements Focal Loss to address class imbalance in bioinformatics
problems where positive samples (disease-associated lncRNAs) are much fewer
than negative samples.

Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
where:
- α_t: weighting factor for class t (addresses class imbalance)
- γ: focusing parameter (reduces loss for well-classified examples)
- p_t: predicted probability for the true class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    
    Particularly effective for bioinformatics problems where positive samples
    (e.g., disease-associated lncRNAs) are significantly fewer than negative samples.
    
    Args:
        alpha (float): Weighting factor for positive class (0 < alpha < 1)
                      Higher alpha gives more weight to positive class
                      For lncRNA-disease: typically 0.7-0.9 due to severe imbalance
        gamma (float): Focusing parameter (gamma >= 0)
                      Higher gamma focuses more on hard examples
                      Typical values: 1.0-3.0
        reduction (str): Specifies the reduction to apply to the output
                        'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss computation.
        
        Args:
            inputs (torch.Tensor): Predicted logits or probabilities [N, 1] or [N]
            targets (torch.Tensor): Ground truth binary labels [N, 1] or [N]
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Ensure inputs and targets have the same shape
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
            
        # Convert targets to float for computation
        targets = targets.float()
        
        # Apply sigmoid to get probabilities if inputs are logits
        # Check if inputs are already probabilities (between 0 and 1)
        if torch.any(inputs < 0) or torch.any(inputs > 1):
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs
            
        # Compute p_t: probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t: class weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        # Add small epsilon to prevent log(0)
        eps = 1e-8
        focal_loss = -alpha_t * focal_weight * torch.log(p_t + eps)
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that automatically adjusts alpha based on class distribution.
    
    This variant automatically computes alpha from the actual class distribution
    in each batch, making it more robust for varying imbalance ratios.
    """
    
    def __init__(self, gamma=2.0, reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Forward pass with adaptive alpha computation.
        
        Args:
            inputs (torch.Tensor): Predicted logits or probabilities
            targets (torch.Tensor): Ground truth binary labels
            
        Returns:
            torch.Tensor: Computed adaptive focal loss
        """
        # Ensure proper shapes
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
            
        targets = targets.float()
        
        # Compute adaptive alpha based on class distribution
        pos_count = targets.sum()
        neg_count = len(targets) - pos_count
        total_count = len(targets)
        
        if pos_count == 0 or neg_count == 0:
            # Fallback to standard BCE if only one class present
            return F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
        
        # Alpha should be higher for the minority class
        alpha = neg_count / total_count  # Weight for positive class
        
        # Apply sigmoid to get probabilities
        if torch.any(inputs < 0) or torch.any(inputs > 1):
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs
            
        # Compute focal loss components
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        eps = 1e-8
        focal_loss = -alpha_t * focal_weight * torch.log(p_t + eps)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_focal_loss(alpha=0.8, gamma=2.0, adaptive=False, reduction='mean'):
    """
    Factory function to create appropriate Focal Loss instance.
    
    Args:
        alpha (float): Weighting factor for positive class (ignored if adaptive=True)
        gamma (float): Focusing parameter
        adaptive (bool): Whether to use adaptive alpha computation
        reduction (str): Reduction method
        
    Returns:
        nn.Module: Focal Loss instance
    """
    if adaptive:
        return AdaptiveFocalLoss(gamma=gamma, reduction=reduction)
    else:
        return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)


# Utility function for bioinformatics researchers
def get_recommended_focal_params(pos_ratio):
    """
    Get recommended Focal Loss parameters based on positive class ratio.
    
    Args:
        pos_ratio (float): Ratio of positive samples (0 < pos_ratio < 1)
        
    Returns:
        dict: Recommended parameters
    """
    if pos_ratio > 0.3:
        # Mild imbalance
        return {'alpha': 0.6, 'gamma': 1.0}
    elif pos_ratio > 0.1:
        # Moderate imbalance (common in bioinformatics)
        return {'alpha': 0.75, 'gamma': 2.0}
    elif pos_ratio > 0.05:
        # Severe imbalance (typical for lncRNA-disease)
        return {'alpha': 0.85, 'gamma': 2.5}
    else:
        # Extreme imbalance
        return {'alpha': 0.9, 'gamma': 3.0}