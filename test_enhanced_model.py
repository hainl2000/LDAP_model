#!/usr/bin/env python3
"""
Test script for the enhanced LDAGM model with:
- Focal Loss for Imbalanced Data
- Residual Connections in Hidden Layers
- Enhanced Aggregation Mechanism with Attention
"""

import torch
import numpy as np
import config
from joint_end_to_end_main import JointVGAE_LDAGM, focal_loss
from LDAGM import EnhancedLDAGM, ResidualHiddenLayer, EnhancedAttentionAggregateLayer

def test_enhanced_components():
    """
    Test the enhanced components individually and together.
    """
    print("Testing Enhanced LDAGM Components...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test dimensions
    batch_size = 32
    input_dim = 64
    hidden_dim = 64
    num_nodes = 100
    
    # Create sample data
    sample_input = torch.randn(batch_size, input_dim).to(device)
    sample_adj = torch.randn(num_nodes, num_nodes).to(device)
    
    print("\n1. Testing ResidualHiddenLayer...")
    try:
        residual_layer = ResidualHiddenLayer(input_dim, config.DROP_RATE).to(device)
        residual_output = residual_layer(sample_input)
        print(f"   ✓ ResidualHiddenLayer output shape: {residual_output.shape}")
    except Exception as e:
        print(f"   ✗ ResidualHiddenLayer failed: {e}")
    
    print("\n2. Testing EnhancedAttentionAggregateLayer...")
    try:
        attention_layer = EnhancedAttentionAggregateLayer(
            input_dim, config.ATTENTION_HEADS, config.DROP_RATE
        ).to(device)
        # Need 3D input for attention layer
        sample_input_3d = sample_input.unsqueeze(0)  # Add batch dimension
        attention_output = attention_layer(sample_input_3d, sample_input_3d)
        print(f"   ✓ EnhancedAttentionAggregateLayer output shape: {attention_output.shape}")
    except Exception as e:
        print(f"   ✗ EnhancedAttentionAggregateLayer failed: {e}")
    
    print("\n3. Testing EnhancedLDAGM (Individual Component)...")
    try:
        enhanced_ldagm = EnhancedLDAGM(
            input_dimension=input_dim,
            hidden_dimension=hidden_dim,
            feature_num=1,
            hiddenLayer_num=config.LDAGM_LAYERS,
            drop_rate=config.DROP_RATE,
            use_aggregate=config.USE_AGGREGATE,
            use_residual=config.USE_RESIDUAL_CONNECTIONS,
            use_attention_aggregate=config.USE_ATTENTION_AGGREGATION,
            num_attention_heads=config.ATTENTION_HEADS
        ).to(device)
        
        ldagm_output = enhanced_ldagm(sample_input)
        print(f"   ✓ EnhancedLDAGM output shape: {ldagm_output.shape}")
    except Exception as e:
        print(f"   ✗ EnhancedLDAGM failed: {e}")
    
    print("\n4. Testing Focal Loss...")
    try:
        # Create sample predictions and labels
        predictions = torch.sigmoid(torch.randn(batch_size, 1)).to(device)
        labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)
        
        # Test focal loss
        focal_loss_value = focal_loss(
            predictions, labels, 
            alpha=config.FOCAL_ALPHA, 
            gamma=config.FOCAL_GAMMA
        )
        print(f"   ✓ Focal Loss computed: {focal_loss_value.item():.4f}")
        
        # Compare with standard BCE
        bce_loss = torch.nn.functional.binary_cross_entropy(predictions, labels)
        print(f"   ✓ Standard BCE Loss: {bce_loss.item():.4f}")
        print(f"   ✓ Focal vs BCE ratio: {focal_loss_value.item() / bce_loss.item():.4f}")
    except Exception as e:
        print(f"   ✗ Focal Loss failed: {e}")
    
    print("\n5. Testing Enhanced Components Integration...")
    try:
        # Test EnhancedLDAGM with correct parameters
        enhanced_ldagm = EnhancedLDAGM(
            input_dimension=input_dim,
            hidden_dimension=hidden_dim,
            feature_num=1,  # Single feature vector per sample
            hiddenLayer_num=config.LDAGM_LAYERS,
            drop_rate=config.DROP_RATE,
            use_aggregate=config.USE_AGGREGATE,
            use_residual=config.USE_RESIDUAL_CONNECTIONS,
            use_attention_aggregate=config.USE_ATTENTION_AGGREGATION,
            num_attention_heads=config.ATTENTION_HEADS
        ).to(device)
        
        # Test with proper input shape for LDAGM
        ldagm_input = torch.randn(batch_size, input_dim).to(device)
        ldagm_output = enhanced_ldagm(ldagm_input)
        print(f"   ✓ Enhanced LDAGM output shape: {ldagm_output.shape}")
        
        # Test JointVGAE_LDAGM with correct parameters
        num_lnc = 50
        num_diseases = 30
        num_mi = 40
        vgae_in_dim = 128
        
        joint_model = JointVGAE_LDAGM(
            num_lnc=num_lnc,
            num_diseases=num_diseases,
            num_mi=num_mi,
            vgae_in_dim=vgae_in_dim,
            vgae_hidden_dim=config.VGAE_HIDDEN_DIM,
            vgae_embed_dim=config.VGAE_EMBED_DIM,
            ldagm_hidden_dim=config.LDAGM_HIDDEN_DIM,
            ldagm_layers=config.LDAGM_LAYERS,
            drop_rate=config.DROP_RATE,
            use_aggregate=config.USE_AGGREGATE
        ).to(device)
        
        print(f"   ✓ JointVGAE_LDAGM model created successfully")
        print(f"   ✓ Model uses residual connections: {config.USE_RESIDUAL_CONNECTIONS}")
        print(f"   ✓ Model uses attention aggregation: {config.USE_ATTENTION_AGGREGATION}")
        print(f"   ✓ Number of attention heads: {config.ATTENTION_HEADS}")
        
    except Exception as e:
        print(f"   ✗ Enhanced Components Integration failed: {e}")
    
    print("\n" + "="*50)
    print("Enhanced LDAGM Model Test Summary:")
    print(f"- Focal Loss: Implemented for imbalanced data handling")
    print(f"- Residual Connections: Enabled for better gradient flow")
    print(f"- Attention Aggregation: {config.ATTENTION_HEADS}-head attention mechanism")
    print(f"- Model Depth: {config.LDAGM_LAYERS} layers")
    print(f"- Hidden Dimension: {config.LDAGM_HIDDEN_DIM}")
    print(f"- Dropout Rate: {config.DROP_RATE}")
    print("="*50)

if __name__ == "__main__":
    test_enhanced_components()