"""
Hyperparameter configuration for joint_end_to_end_main.py
"""

# Dataset and Fold
DATASET = "dataset2"
FOLD = 1

# Device
DEVICE = "mps"  # "cuda" or "cpu"

# Model Hyperparameters
GCN_HIDDEN_DIM = 128
FUSION_OUTPUT_DIM = 64
VGAE_HIDDEN_DIM = 32
VGAE_EMBED_DIM = 32
LDAGM_HIDDEN_DIM = 40
LDAGM_LAYERS = 5
DROP_RATE = 0.2
USE_AGGREGATE = True

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

# Loss Function Weights
VGAE_WEIGHT = 1.0
LINK_WEIGHT = 3.0
KL_WEIGHT = 0.1

# Logging
LOG_FILE = "logs/joint_training_log.txt"
CSV_LOG_FILE = "logs/experiment_results.csv"