"""
Hyperparameter configuration for joint_end_to_end_main.py
"""

# Dataset and Fold
DATASET = "dataset2"
TOTAL_FOLDS = 5
NETWORK_NUM = 4  # Number of networks for A_encoder loading
A_ENCODER_DIM = 128

# Device
DEVICE = "mps"  # "cuda", "mps", or "cpu"

# Model Hyperparameters - Best Hyperparameters Applied
GCN_HIDDEN_DIM = 128
FUSION_OUTPUT_DIM = 128
VGAE_HIDDEN_DIM = 32
VGAE_EMBED_DIM = 256
LDAGM_HIDDEN_DIM = 192
LDAGM_LAYERS = 8
DROP_RATE = 0.11235069657748148
USE_AGGREGATE = True

#GCN_HIDDEN_DIM = [32,64,128]
# FUSION_OUTPUT_DIM =[32,64,128]
# VGAE_HIDDEN_DIM = [32,64,128]
# VGAE_EMBED_DIM = [32,64,128]
# LDAGM_HIDDEN_DIM = [30,40,50]
# LDAGM_LAYERS = [3,4,5]
# DROP_RATE = [0.1,0.2,0.3,0.4]
# USE_AGGREGATE = True 

# Training Hyperparameters - Best Hyperparameters Applied
BATCH_SIZE = 48
EPOCHS = 1
LEARNING_RATE = 0.0071144760093434225
WEIGHT_DECAY = 0.00015702970884055374
#EPOCHS = [1,2,3,4,5,6,7,8,9,10]
# Loss Function Weights - Best Hyperparameters Applied
VGAE_WEIGHT = 1.397987726295555
LINK_WEIGHT = 0.8900466011060912
KL_WEIGHT = 0.039638958863878505

# Training Configuration
# GRAD_CLIP = 1.0
# EVAL_EVERY = 1
# LR_PATIENCE = 5

# Logging
LOG_FILE = "logs/joint_training_log.txt"
CSV_LOG_FILE = "logs/hhm_experiment_results.csv"