# Hyperparameter Tuning Guide

This guide explains how to use the hyperparameter tuning script for the joint end-to-end model.

## Features

- **Optuna-based optimization**: Uses Bayesian optimization for efficient hyperparameter search
- **Early stopping**: Automatically stops training when validation AUC doesn't improve
- **Fixed parameters**: Always uses NETWORK_NUM = 4 and use_aggregate = True as requested
- **CSV logging**: Saves all trial results and hyperparameters to CSV file
- **Automatic config update**: Updates config.py with best hyperparameters after tuning

## Usage

### Basic Usage

```bash
python hyperparameter_tuning.py
```

### Configuration

The tuning script uses the following settings (can be modified in the script):

- `N_TRIALS = 150`: Number of Optuna trials to run
- `EARLY_STOPPING_PATIENCE = 30`: Stop training if no improvement for this many epochs
- `EARLY_STOPPING_MIN_DELTA = 0.001`: Minimum AUC improvement to reset patience counter
- `FIXED_NETWORK_NUM = 4`: Fixed number of networks (as requested)
- `use_aggregate = True`: Fixed aggregation setting (as requested)

### Hyperparameter Search Space

The script searches over the following hyperparameters:

- `batch_size`: [16, 128] (step=16)
- `learning_rate`: [1e-5, 1e-2] (log scale)
- `weight_decay`: [1e-6, 1e-3] (log scale)
- `vgae_weight`: [0.5, 2.0]
- `link_weight`: [0.5, 3.0]
- `kl_weight`: [0.01, 0.2]
- `vgae_hidden_dim`: [32, 128] (step=32)
- `vgae_embed_dim`: [64, 256] (step=64)
- `ldagm_hidden_dim`: [64, 256] (step=64)
- `ldagm_layers`: [3, 10]
- `drop_rate`: [0.1, 0.7]
- `gcn_hidden_dim`: [32, 128] (step=32)
- `fusion_output_dim`: [32, 128] (step=32)

### Output Files

1. **logs/hyperparameter_tuning_results.csv**: Contains all trial results
   - Trial number, timestamp, runtime
   - All hyperparameter values
   - Average metrics across 5 folds (AUC, AUPR, MCC, ACC, Precision, Recall, F1)
   - Standard deviation of AUC
   - Average epochs trained (with early stopping)

2. **logs/best_hyperparameters.txt**: Best hyperparameters found

3. **config.py**: Automatically updated with best hyperparameters

### Monitoring Progress

The script provides real-time feedback:
- Progress bar showing trial completion
- Per-fold AUC results during each trial
- Early stopping information
- Final summary statistics

### Early Stopping Behavior

- Training stops if validation AUC doesn't improve for `EARLY_STOPPING_PATIENCE` epochs
- Uses 20% of training data for validation
- Restores best model weights based on validation AUC

### Example Output

```
Starting hyperparameter tuning for joint_end_to_end_main...
Number of trials: 150
Early stopping patience: 30
Fixed NETWORK_NUM: 4
Results will be saved to: logs/hyperparameter_tuning_results.csv

Trial 0: Testing hyperparameters...
  Processing fold 1/5...
    Fold 1 - Test AUC: 0.8543, Val AUC: 0.8421, Epochs: 45
  ...

============================================================
HYPERPARAMETER TUNING COMPLETED
============================================================
Best trial number: 67
Best average AUC: 0.8765

Best hyperparameters:
  batch_size: 64
  learning_rate: 0.0003214
  ...
```

## Tips

1. **Reduce trials for quick testing**: Set `N_TRIALS = 10` for initial testing
2. **Adjust patience**: Increase `EARLY_STOPPING_PATIENCE` if models need more epochs to converge
3. **Monitor CSV file**: Check the CSV file periodically to see trial results
4. **Pruning**: Optuna automatically prunes unpromising trials to save time

## Troubleshooting

- **Out of memory**: Reduce `batch_size` search range or use smaller `vgae_embed_dim` values
- **Slow convergence**: Increase `max_epochs` in the hyperparams dictionary
- **Poor results**: Expand search ranges or increase `N_TRIALS`
