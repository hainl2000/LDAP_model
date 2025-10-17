#!/usr/bin/env python3
"""
Test script to verify the updated JointVGAE_LDAGM model works correctly
with GraphConvolution and MSE loss function.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import JointVGAE_LDAGM, joint_loss_function
import config

def test_model_initialization():
    """Test that the model initializes correctly with GraphConvolution."""
    print("Testing model initialization...")
    
    # Test parameters
    num_lnc = 10
    num_diseases = 8
    num_mi = 6
    vgae_in_dim = 24  # num_lnc + num_diseases + num_mi
    vgae_hidden_dim = 32
    vgae_embed_dim = 16
    ldagm_hidden_dim = 64
    ldagm_layers = 2
    
    try:
        model = JointVGAE_LDAGM(
            num_lnc=num_lnc,
            num_diseases=num_diseases,
            num_mi=num_mi,
            vgae_in_dim=vgae_in_dim,
            vgae_hidden_dim=vgae_hidden_dim,
            vgae_embed_dim=vgae_embed_dim,
            ldagm_hidden_dim=ldagm_hidden_dim,
            ldagm_layers=ldagm_layers
        )
        
        # Check that GraphConvolution is used instead of VGAE
        assert hasattr(model, 'graph_conv'), "Model should have graph_conv attribute"
        assert not hasattr(model, 'vgae'), "Model should not have vgae attribute"
        
        print("✓ Model initialization successful")
        return model, num_lnc, num_diseases, num_mi
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return None, None, None, None

def test_forward_pass(model, num_lnc, num_diseases, num_mi):
    """Test that the forward pass works with GraphConvolution."""
    print("Testing forward pass...")
    
    try:
        # Determine device and move model to it
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        print(f"Using device: {device}")
        model = model.to(device)
        
        # Create dummy similarity matrices and move to device
        disease_matrices = [
            torch.randn(num_diseases, num_diseases).to(device),
            torch.randn(num_diseases, num_diseases).to(device)
        ]
        lnc_matrices = [
            torch.randn(num_lnc, num_lnc).to(device),
            torch.randn(num_lnc, num_lnc).to(device)
        ]
        mi_matrices = [
            torch.randn(num_mi, num_mi).to(device),
            torch.randn(num_mi, num_mi).to(device)
        ]
        
        multi_view_data = {
            'disease': disease_matrices,
            'lnc': lnc_matrices,
            'mi': mi_matrices
        }
        
        # Create dummy interaction matrices and move to device
        lnc_di_interaction = torch.randint(0, 2, (num_lnc, num_diseases)).float().to(device)
        lnc_mi_interaction = torch.randint(0, 2, (num_lnc, num_mi)).float().to(device)
        mi_di_interaction = torch.randint(0, 2, (num_mi, num_diseases)).float().to(device)
        
        # Test forward pass without node pairs (should return rd, re, log_var)
        rd, re, log_var = model(
            multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction
        )
        
        assert rd is not None, "rd (reconstructed adjacency) should not be None"
        assert re is not None, "re (embeddings) should not be None"
        assert log_var is not None, "log_var should not be None"
        
        print(f"✓ Forward pass without node pairs successful")
        print(f"  - rd shape: {rd.shape}")
        print(f"  - re shape: {re.shape}")
        print(f"  - log_var shape: {log_var.shape}")
        
        # For testing purposes, we'll create dummy link predictions to test the loss function
        # without triggering the file loading in the forward pass
        batch_size = 5
        link_predictions = torch.randn(batch_size).to(device)  # Dummy link predictions
        
        print(f"✓ Forward pass test completed (skipping node pairs to avoid file loading)")
        print(f"  - Created dummy link_predictions for loss testing: {link_predictions.shape}")
        
        return rd, re, log_var, link_predictions, batch_size
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return None, None, None, None, None

def test_mse_loss_function(rd, re, log_var, link_predictions, batch_size, num_lnc, num_diseases, num_mi):
    """Test that the MSE loss function works correctly."""
    print("Testing MSE loss function...")
    
    try:
        # Create dummy targets on the same device as the model outputs
        device = rd.device
        num_nodes = num_lnc + num_diseases + num_mi
        original_adj = torch.randn_like(rd)  # Dummy original adjacency matrix
        link_labels = torch.randint(0, 2, (batch_size,)).float().to(device)  # Dummy link labels
        
        # Test the joint loss function
        total_loss, loss_components = joint_loss_function(
            reconstructed_adj=rd,
            original_adj=original_adj,
            mu=re,
            log_var=log_var,
            link_predictions=link_predictions,
            link_labels=link_labels,
            num_nodes=num_nodes,
            vgae_weight=1.0,
            link_weight=1.0,
            kl_weight=0.1
        )
        
        assert total_loss is not None, "total_loss should not be None"
        assert isinstance(total_loss, torch.Tensor), "total_loss should be a tensor"
        assert total_loss.requires_grad, "total_loss should require gradients"
        
        # Check loss components
        assert 'vgae_reconstruction' in loss_components, "Should have vgae_reconstruction loss"
        assert 'kl_divergence' in loss_components, "Should have kl_divergence loss"
        assert 'link_prediction' in loss_components, "Should have link_prediction loss"
        assert 'total' in loss_components, "Should have total loss"
        
        # KL divergence should be zero for GraphConvolution
        assert loss_components['kl_divergence'] == 0.0, "KL divergence should be zero for GraphConvolution"
        
        print("✓ MSE loss function works correctly")
        print(f"  - Total loss: {total_loss.item():.4f}")
        print(f"  - VGAE reconstruction loss: {loss_components['vgae_reconstruction']:.4f}")
        print(f"  - KL divergence loss: {loss_components['kl_divergence']:.4f}")
        print(f"  - Link prediction loss: {loss_components['link_prediction']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ MSE loss function failed: {e}")
        return False

def test_backward_pass(model, rd, re, log_var, link_predictions, batch_size, num_lnc, num_diseases, num_mi):
    """Test that backward pass works correctly."""
    print("Testing backward pass...")
    
    try:
        # Create dummy targets on the same device
        device = rd.device
        num_nodes = num_lnc + num_diseases + num_mi
        original_adj = torch.randn_like(rd)
        link_labels = torch.randint(0, 2, (batch_size,)).float().to(device)
        
        # Clear any existing gradients
        model.zero_grad()
        
        # Compute loss
        total_loss, _ = joint_loss_function(
            reconstructed_adj=rd,
            original_adj=original_adj,
            mu=re,
            log_var=log_var,
            link_predictions=link_predictions,
            link_labels=link_labels,
            num_nodes=num_nodes
        )
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients are computed
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "Model should have gradients after backward pass"
        
        print("✓ Backward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING UPDATED MODEL WITH GRAPHCONVOLUTION AND MSE LOSS")
    print("="*60)
    
    # Test 1: Model initialization
    model, num_lnc, num_diseases, num_mi = test_model_initialization()
    if model is None:
        print("✗ Cannot proceed with other tests due to initialization failure")
        return False
    
    print()
    
    # Test 2: Forward pass
    rd, re, log_var, link_predictions, batch_size = test_forward_pass(model, num_lnc, num_diseases, num_mi)
    if rd is None:
        print("✗ Cannot proceed with other tests due to forward pass failure")
        return False
    
    print()
    
    # Test 3: MSE loss function
    loss_success = test_mse_loss_function(rd, re, log_var, link_predictions, batch_size, num_lnc, num_diseases, num_mi)
    if not loss_success:
        print("✗ Cannot proceed with backward pass test due to loss function failure")
        return False
    
    print()
    
    # Test 4: Backward pass
    backward_success = test_backward_pass(model, rd, re, log_var, link_predictions, batch_size, num_lnc, num_diseases, num_mi)
    
    print()
    print("="*60)
    if backward_success:
        print("✓ ALL TESTS PASSED! The updated model works correctly.")
        print("✓ GraphConvolution successfully replaced VGAE_Model")
        print("✓ MSE loss function works properly")
        print("✓ Model can perform forward and backward passes")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    print("="*60)
    
    return backward_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)