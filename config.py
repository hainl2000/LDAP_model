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
LDAGM_HIDDEN_DIM = 40
LDAGM_LAYERS = 4
DROP_RATE = 0.2
USE_AGGREGATE = True

#GCN_HIDDEN_DIM = [32,64,128]
# FUSION_OUTPUT_DIM =[32,64,128]
# VGAE_HIDDEN_DIM = [32,64,128]
# VGAE_EMBED_DIM = [32,64,128]
# LDAGM_HIDDEN_DIM = [30,40,50]
# LDAGM_LAYERS = [3,4,5]
# DROP_RATE = [0.1,0.2,0.3,0.4]
# USE_AGGREGATE = True 

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
#EPOCHS = [1,2,3,4,5,6,7,8,9,10]
# Loss Function Weights
VGAE_WEIGHT = 1.0
LINK_WEIGHT = 3.0
KL_WEIGHT = 0.1

# Training Configuration
# GRAD_CLIP = 1.0
# EVAL_EVERY = 1
# LR_PATIENCE = 5

# Logging
LOG_FILE = "logs/joint_training_log.txt"
CSV_LOG_FILE = "logs/hhm_experiment_results.csv"