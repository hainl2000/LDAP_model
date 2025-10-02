"""
Hyperparameter configuration for joint_end_to_end_main.py

This file contains the configuration for single runs of the model.
For automated hyperparameter tuning, see hyperparameter_tuning.py
"""

# Dataset and Fold
DATASET = "dataset2"  # Options: "dataset1", "dataset2", "dataset3"
TOTAL_FOLDS = 5  # Number of cross-validation folds
NETWORK_NUM = 4  # Number of networks for A_encoder loading
A_ENCODER_DIM = 128  # Dimension of A_encoder features

# Device Configuration
DEVICE = "mps"  # Options: "cuda", "mps", "cpu"

# ============================================================================
# Model Architecture Hyperparameters
# These control the size and complexity of the neural network components
# ============================================================================

# GCN (Graph Convolutional Network) parameters
GCN_HIDDEN_DIM = 64  # Hidden dimension for GCN layers (try: 32, 64, 128)
FUSION_OUTPUT_DIM = 64  # Output dimension after multi-view fusion (try: 32, 64, 128)

# VGAE (Variational Graph Auto-Encoder) parameters
VGAE_HIDDEN_DIM = 64  # Hidden dimension for VGAE encoder (try: 32, 64, 128)
VGAE_EMBED_DIM = 64  # Embedding dimension for latent space (try: 32, 64, 128)

# LDAGM (Link Prediction) parameters
LDAGM_HIDDEN_DIM = 40  # Hidden dimension for LDAGM (try: 30, 40, 50)
LDAGM_LAYERS = 4  # Number of hidden layers in LDAGM (try: 3, 4, 5)

# Regularization
DROP_RATE = 0.2  # Dropout rate for regularization (try: 0.1, 0.2, 0.3, 0.4)
USE_AGGREGATE = True  # Use aggregation layers in LDAGM (True/False)

# ============================================================================
# Training Hyperparameters
# These control the optimization process
# ============================================================================

BATCH_SIZE = 32  # Mini-batch size (try: 16, 32, 64)
EPOCHS = 100  # Number of training epochs (try: 50, 100, 150)
LEARNING_RATE = 5e-4  # Learning rate for optimizer (try: 1e-4, 5e-4, 1e-3)
WEIGHT_DECAY = 1e-4  # L2 regularization weight (try: 1e-5, 1e-4, 1e-3)

# ============================================================================
# Loss Function Weights
# These balance different components of the loss function
# ============================================================================

VGAE_WEIGHT = 1.0  # Weight for VGAE reconstruction loss (try: 0.5, 1.0, 2.0)
LINK_WEIGHT = 3.0  # Weight for link prediction loss (try: 1.0, 2.0, 3.0)
KL_WEIGHT = 0.1  # Weight for KL divergence loss (try: 0.05, 0.1, 0.2)

# Training Configuration
# GRAD_CLIP = 1.0
# EVAL_EVERY = 1
# LR_PATIENCE = 5

# Logging
LOG_FILE = "logs/joint_training_log.txt"
CSV_LOG_FILE = "logs/hhm_experiment_results.csv"