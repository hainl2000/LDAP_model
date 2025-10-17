"""
Hyperparameter configuration for joint_end_to_end_main.py
"""

# Dataset and Fold
DATASET = "dataset5"
TOTAL_FOLDS = 5
NETWORK_NUM = 4  # Number of networks for A_encoder loading
A_ENCODER_DIM = 128

# Device
DEVICE = "mps"  # "cuda" or "cpu"

# Model Hyperparameters
GCN_HIDDEN_DIM = 128
FUSION_OUTPUT_DIM = 128
VGAE_HIDDEN_DIM = 32
VGAE_EMBED_DIM = 32
LDAGM_HIDDEN_DIM = 50
LDAGM_LAYERS = 2
DROP_RATE = 0.24381064755129345
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
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.00028957072334927773
WEIGHT_DECAY = 0.00021750107956406713
#EPOCHS = [1,2,3,4,5,6,7,8,9,10]
# Loss Function Weights
VGAE_WEIGHT = 0.8841637241431217
LINK_WEIGHT = 1.2566840656725193
KL_WEIGHT = 0.2766611462898132

# Training Configuration
# GRAD_CLIP = 1.0
# EVAL_EVERY = 1
# LR_PATIENCE = 5

# Logging
LOG_FILE = "logs/joint_training_log.txt"
CSV_LOG_FILE = "logs/hhm_experiment_results.csv"