"""
Example configurations for running hyperparameter tuning.
This file provides ready-to-use examples for different search scenarios.
"""

import torch
import numpy as np
from hyperparameter_tuning import generate_hyperparameter_combinations, hyperparameter_search
import config

# ============================================================================
# Example 1: Quick Architecture Search
# Only vary model architecture, fix training params
# ============================================================================

ARCHITECTURE_SEARCH_GRID = {
    'GCN_HIDDEN_DIM': [32, 64, 128],
    'FUSION_OUTPUT_DIM': [32, 64, 128],
    'VGAE_HIDDEN_DIM': [32, 64, 128],
    'VGAE_EMBED_DIM': [32, 64, 128],
    'LDAGM_HIDDEN_DIM': [30, 40, 50],
    'LDAGM_LAYERS': [3, 4, 5],
    'DROP_RATE': [0.2],  # Fixed
    'LEARNING_RATE': [5e-4],  # Fixed
    'WEIGHT_DECAY': [1e-4],  # Fixed
    'BATCH_SIZE': [32],  # Fixed
    'EPOCHS': [50],  # Fixed - use fewer for quick search
    'VGAE_WEIGHT': [1.0],  # Fixed
    'LINK_WEIGHT': [3.0],  # Fixed
    'KL_WEIGHT': [0.1],  # Fixed
}

# ============================================================================
# Example 2: Learning Rate & Regularization Search
# Focus on training hyperparameters
# ============================================================================

TRAINING_SEARCH_GRID = {
    'GCN_HIDDEN_DIM': [64],  # Fixed at best from architecture search
    'FUSION_OUTPUT_DIM': [64],  # Fixed
    'VGAE_HIDDEN_DIM': [64],  # Fixed
    'VGAE_EMBED_DIM': [64],  # Fixed
    'LDAGM_HIDDEN_DIM': [40],  # Fixed
    'LDAGM_LAYERS': [4],  # Fixed
    'DROP_RATE': [0.1, 0.2, 0.3, 0.4],
    'LEARNING_RATE': [1e-4, 5e-4, 1e-3, 5e-3],
    'WEIGHT_DECAY': [1e-5, 1e-4, 1e-3],
    'BATCH_SIZE': [16, 32, 64],
    'EPOCHS': [100],
    'VGAE_WEIGHT': [1.0],  # Fixed
    'LINK_WEIGHT': [3.0],  # Fixed
    'KL_WEIGHT': [0.1],  # Fixed
}

# ============================================================================
# Example 3: Loss Weight Search
# Focus on balancing different loss components
# ============================================================================

LOSS_WEIGHT_SEARCH_GRID = {
    'GCN_HIDDEN_DIM': [64],  # Fixed at best values
    'FUSION_OUTPUT_DIM': [64],  # Fixed
    'VGAE_HIDDEN_DIM': [64],  # Fixed
    'VGAE_EMBED_DIM': [64],  # Fixed
    'LDAGM_HIDDEN_DIM': [40],  # Fixed
    'LDAGM_LAYERS': [4],  # Fixed
    'DROP_RATE': [0.2],  # Fixed
    'LEARNING_RATE': [5e-4],  # Fixed
    'WEIGHT_DECAY': [1e-4],  # Fixed
    'BATCH_SIZE': [32],  # Fixed
    'EPOCHS': [100],
    'VGAE_WEIGHT': [0.1, 0.5, 1.0, 2.0, 5.0],
    'LINK_WEIGHT': [0.5, 1.0, 2.0, 3.0, 5.0],
    'KL_WEIGHT': [0.01, 0.05, 0.1, 0.2, 0.5],
}

# ============================================================================
# Example 4: Fine-tuning Around Best Configuration
# Small variations around a known good configuration
# ============================================================================

FINE_TUNE_GRID = {
    # Assume best config was: 64, 64, 64, 64, 40, 4, 0.2, 5e-4, 1e-4
    'GCN_HIDDEN_DIM': [56, 64, 72],  # ±12.5% variation
    'FUSION_OUTPUT_DIM': [56, 64, 72],
    'VGAE_HIDDEN_DIM': [56, 64, 72],
    'VGAE_EMBED_DIM': [56, 64, 72],
    'LDAGM_HIDDEN_DIM': [36, 40, 44],
    'LDAGM_LAYERS': [3, 4, 5],
    'DROP_RATE': [0.15, 0.2, 0.25],
    'LEARNING_RATE': [3e-4, 5e-4, 7e-4],
    'WEIGHT_DECAY': [5e-5, 1e-4, 2e-4],
    'BATCH_SIZE': [32],
    'EPOCHS': [100],
    'VGAE_WEIGHT': [0.8, 1.0, 1.2],
    'LINK_WEIGHT': [2.5, 3.0, 3.5],
    'KL_WEIGHT': [0.08, 0.1, 0.12],
}

# ============================================================================
# Example 5: Ultra-Quick Test (for debugging)
# Minimal combinations to verify everything works
# ============================================================================

DEBUG_GRID = {
    'GCN_HIDDEN_DIM': [64],
    'FUSION_OUTPUT_DIM': [64],
    'VGAE_HIDDEN_DIM': [64],
    'VGAE_EMBED_DIM': [64],
    'LDAGM_HIDDEN_DIM': [40],
    'LDAGM_LAYERS': [4],
    'DROP_RATE': [0.2],
    'LEARNING_RATE': [5e-4, 1e-3],  # Just 2 values
    'WEIGHT_DECAY': [1e-4],
    'BATCH_SIZE': [32],
    'EPOCHS': [10],  # Very few epochs for quick test
    'VGAE_WEIGHT': [1.0],
    'LINK_WEIGHT': [3.0],
    'KL_WEIGHT': [0.1],
}


def print_grid_info(grid, name):
    """Print information about a hyperparameter grid."""
    combinations = generate_hyperparameter_combinations(grid)
    total = len(combinations)
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Total combinations: {total}")
    print(f"\nVariable parameters:")
    for key, values in grid.items():
        if len(values) > 1:
            print(f"  {key}: {values} ({len(values)} options)")
    print(f"{'='*60}\n")


def run_example(example_number):
    """
    Run a specific example search.
    
    Args:
        example_number: 1-5, corresponding to the examples above
    """
    
    examples = {
        1: ("Architecture Search", ARCHITECTURE_SEARCH_GRID),
        2: ("Training Hyperparameter Search", TRAINING_SEARCH_GRID),
        3: ("Loss Weight Search", LOSS_WEIGHT_SEARCH_GRID),
        4: ("Fine-tuning Search", FINE_TUNE_GRID),
        5: ("Debug/Quick Test", DEBUG_GRID),
    }
    
    if example_number not in examples:
        print(f"Error: Example {example_number} not found. Choose 1-5.")
        return
    
    name, grid = examples[example_number]
    
    print(f"\n{'='*60}")
    print(f"Running Example {example_number}: {name}")
    print(f"{'='*60}\n")
    
    print_grid_info(grid, name)
    
    # Generate combinations
    combinations = generate_hyperparameter_combinations(grid)
    
    # Estimate time (very rough estimate)
    # Assume ~5 minutes per configuration with 5 folds
    estimated_minutes = len(combinations) * 5
    hours = estimated_minutes // 60
    minutes = estimated_minutes % 60
    
    print(f"Estimated runtime: ~{hours} hours and {minutes} minutes")
    print(f"(This is a rough estimate, actual time may vary significantly)")
    
    response = input("\nProceed with this search? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("\nStarting hyperparameter search...\n")
        
        # Import here to avoid loading heavy dependencies if just printing info
        from hyperparameter_tuning import hyperparameter_search
        import hyperparameter_tuning
        
        # Temporarily replace the grid in the module
        original_grid = hyperparameter_tuning.HYPERPARAMETER_GRID
        hyperparameter_tuning.HYPERPARAMETER_GRID = grid
        
        # Run the search
        all_results, best_hyperparams = hyperparameter_search()
        
        # Restore original
        hyperparameter_tuning.HYPERPARAMETER_GRID = original_grid
        
        print(f"\n{'='*60}")
        print(f"Example {example_number} Complete!")
        print(f"{'='*60}\n")
    else:
        print("\nSearch cancelled.")


def compare_all_examples():
    """Print information about all example grids for comparison."""
    
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH EXAMPLES OVERVIEW")
    print("="*80)
    
    examples = [
        ("Example 1: Architecture Search", ARCHITECTURE_SEARCH_GRID),
        ("Example 2: Training Hyperparameter Search", TRAINING_SEARCH_GRID),
        ("Example 3: Loss Weight Search", LOSS_WEIGHT_SEARCH_GRID),
        ("Example 4: Fine-tuning Search", FINE_TUNE_GRID),
        ("Example 5: Debug/Quick Test", DEBUG_GRID),
    ]
    
    for name, grid in examples:
        print_grid_info(grid, name)
    
    print("\nRecommended workflow:")
    print("  1. Run Example 5 (Debug) first to verify everything works")
    print("  2. Run Example 1 (Architecture) to find best model size")
    print("  3. Run Example 2 (Training) with best architecture from step 2")
    print("  4. Run Example 3 (Loss Weights) with best config from step 3")
    print("  5. Run Example 4 (Fine-tuning) to optimize around best overall config")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_hyperparameter_examples.py [example_number|compare]")
        print("\nExamples:")
        print("  python run_hyperparameter_examples.py 1      # Run architecture search")
        print("  python run_hyperparameter_examples.py 2      # Run training param search")
        print("  python run_hyperparameter_examples.py 3      # Run loss weight search")
        print("  python run_hyperparameter_examples.py 4      # Run fine-tuning search")
        print("  python run_hyperparameter_examples.py 5      # Run debug/quick test")
        print("  python run_hyperparameter_examples.py compare # Compare all examples")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg == 'compare':
        compare_all_examples()
    else:
        try:
            example_num = int(arg)
            run_example(example_num)
        except ValueError:
            print(f"Error: '{arg}' is not a valid example number or 'compare'")
            sys.exit(1)

