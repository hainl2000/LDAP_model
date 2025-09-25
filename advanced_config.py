"""
Hyperparameter configuration for joint_end_to_end_main.py
"""

# Dataset and Fold
DATASET = "dataset2"
TOTAL_FOLDS = 5
NETWORK_NUM = 4  # Number of networks for A_encoder loading
A_ENCODER_DIM = 128

# Device
DEVICE = "mps"  # "cuda" or "cpu"

# Model Hyperparameters
GCN_HIDDEN_DIM = 64
FUSION_OUTPUT_DIM =64
VGAE_HIDDEN_DIM = 64
VGAE_EMBED_DIM = 64
LDAGM_HIDDEN_DIM = 64  # Increased from 40 for better performance
LDAGM_LAYERS = 6       # Increased from 4 for deeper learning
DROP_RATE = 0.3        # Increased for better regularization
USE_AGGREGATE = True

# Enhanced LDAGM Features
USE_RESIDUAL_CONNECTIONS = True
USE_ATTENTION_AGGREGATION = True
ATTENTION_HEADS = 4

# Focal Loss Parameters
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

# Loss Function Weights
VGAE_WEIGHT = 1.0
LINK_WEIGHT = 3.0
KL_WEIGHT = 0.1

# Logging
LOG_FILE = "logs/joint_training_log.txt"
CSV_LOG_FILE = "logs/hhm_experiment_results.csv"