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
EPOCHS = 5
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
#EPOCHS = [1,2,3,4,5,6,7,8,9,10]
# Loss Function Weights
VGAE_WEIGHT = 1.0
LINK_WEIGHT = 3.0
KL_WEIGHT = 0.1

# CAGrad Configuration
USE_CAGRAD = True
CAGRAD_ALPHA = 0.5
CAGRAD_RESCALE = 1
CAGRAD_CONFLICT_THRESHOLD = 0.1

# CAGrad Task Priorities (for multi-task learning)
CAGRAD_TASK_PRIORITIES = {
    "vgae_recon": 1.0,    # VGAE reconstruction loss priority
    "kl_div": 1.0,        # KL divergence loss priority  
    "link_pred": 2.0      # Link prediction loss priority (higher for prediction task)
}

# Focal Loss Configuration (for class imbalance in lncRNA-disease prediction)
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.85        # Weight for positive class (higher for severe imbalance)
FOCAL_GAMMA = 2.5         # Focusing parameter (higher focuses on hard examples)
FOCAL_ADAPTIVE = False    # Whether to use adaptive alpha based on batch distribution
FOCAL_REDUCTION = 'mean'  # Reduction method: 'mean', 'sum', or 'none'

# Training Configuration
# GRAD_CLIP = 1.0
# EVAL_EVERY = 1
# LR_PATIENCE = 5

# Logging
LOG_FILE = "logs/joint_training_log.txt"
CSV_LOG_FILE = "logs/hhm_experiment_results.csv"