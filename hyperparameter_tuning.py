"""
Hyperparameter tuning script for main.py using Optuna
with early stopping and CSV logging for each trial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
import time
import os
import csv
from datetime import datetime
import optuna
from optuna.trial import TrialState
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
)
import warnings
warnings.filterwarnings("ignore")

# Import necessary components from the main project
from main import (
    JointDataset, 
    JointVGAE_LDAGM,
    joint_loss_function
)
import config

# Fixed NETWORK_NUM as requested
FIXED_NETWORK_NUM = 4

# Tuning configuration
TUNING_CSV_FILE = "logs/hyperparameter_tuning_results.csv"
BEST_PARAMS_FILE = "logs/best_hyperparameters.txt"
N_TRIALS = 150  # Number of Optuna trials
EARLY_STOPPING_PATIENCE = 30  # Stop training if no improvement for this many epochs
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum change to qualify as improvement


class EarlyStopping:
    """Early stopping to stop training when validation AUC doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            return True
            
        return False


def log_trial_to_csv(trial_params, metrics, trial_number, runtime):
    """Log each trial's hyperparameters and results to CSV"""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(TUNING_CSV_FILE), exist_ok=True)
    
    # Combine all data
    row_data = {
        'trial_number': trial_number,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_seconds': runtime,
        **trial_params,
        **metrics
    }
    
    # Write to CSV
    file_exists = os.path.isfile(TUNING_CSV_FILE)
    
    with open(TUNING_CSV_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def train_with_early_stopping(model, train_dataset, val_dataset, multi_view_data,
                            lnc_di_interaction, lnc_mi_interaction, mi_di_interaction,
                            num_nodes, pos_weight, fold, hyperparams, device):
    """
    Train model with early stopping based on validation AUC
    """
    # Extract hyperparameters
    batch_size = hyperparams['batch_size']
    lr = hyperparams['learning_rate']
    weight_decay = hyperparams['weight_decay']
    vgae_weight = hyperparams['vgae_weight']
    link_weight = hyperparams['link_weight']
    kl_weight = hyperparams['kl_weight']
    max_epochs = hyperparams['max_epochs']
    
    # Setup training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)
    
    best_val_auc = 0
    best_model_state = None
    actual_epochs = 0
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_node_pairs, batch_labels in train_loader:
            batch_node_pairs = batch_node_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed_adj, mu, log_var, link_predictions = model(
                multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction,
                batch_node_pairs, network_num=FIXED_NETWORK_NUM, fold=fold, dataset=config.DATASET
            )
            
            # Get original adjacency for loss calculation
            with torch.no_grad():
                original_adj_reconstructed, _, _ = model(
                    multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction,
                    None, network_num=FIXED_NETWORK_NUM, fold=fold, dataset=config.DATASET
                )
            
            # Compute loss
            total_loss, loss_components = joint_loss_function(
                reconstructed_adj, original_adj_reconstructed.detach(), mu, log_var,
                link_predictions, batch_labels, num_nodes, pos_weight,
                vgae_weight, link_weight, kl_weight
            )
            
            total_loss.backward()
            optimizer.step()
            
            train_losses.append(loss_components['total'])
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch_node_pairs, batch_labels in val_loader:
                batch_node_pairs = batch_node_pairs.to(device)
                
                _, _, _, link_predictions = model(
                    multi_view_data, lnc_di_interaction, lnc_mi_interaction, mi_di_interaction,
                    batch_node_pairs, network_num=FIXED_NETWORK_NUM, fold=fold, dataset=config.DATASET
                )
                
                predictions = torch.sigmoid(link_predictions)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(batch_labels.numpy())
        
        # Calculate validation AUC
        val_auc = roc_auc_score(val_labels, val_predictions)
        
        # Update best model if improved
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
        
        actual_epochs = epoch + 1
        
        # Check early stopping
        if early_stopping(val_auc):
            print(f"Early stopping triggered at epoch {epoch + 1}. Best validation AUC: {best_val_auc:.4f}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: Train Loss: {np.mean(train_losses):.4f}, Val AUC: {val_auc:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_auc, actual_epochs


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization
    """
    start_time = time.time()
    
    # Suggest hyperparameters
    hyperparams = {
        'batch_size': trial.suggest_int('batch_size', 16, 128, step=16),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'vgae_weight': trial.suggest_float('vgae_weight', 0.5, 2.0),
        'link_weight': trial.suggest_float('link_weight', 0.5, 3.0),
        'kl_weight': trial.suggest_float('kl_weight', 0.01, 0.2),
        'vgae_hidden_dim': trial.suggest_int('vgae_hidden_dim', 32, 128, step=32),
        'vgae_embed_dim': trial.suggest_int('vgae_embed_dim', 64, 256, step=64),
        'ldagm_hidden_dim': trial.suggest_int('ldagm_hidden_dim', 64, 256, step=64),
        'ldagm_layers': trial.suggest_int('ldagm_layers', 3, 10),
        'drop_rate': trial.suggest_float('drop_rate', 0.1, 0.7),
        'gcn_hidden_dim': trial.suggest_int('gcn_hidden_dim', 32, 128, step=32),
        'fusion_output_dim': trial.suggest_int('fusion_output_dim', 32, 128, step=32),
        'max_epochs': 150  # Maximum epochs before early stopping
    }
    
    # Configuration
    dataset = config.DATASET
    # Proper device selection handling both CUDA and MPS
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Store results for all folds
    fold_aucs = []
    fold_metrics = []
    
    print(f"\nTrial {trial.number}: Testing hyperparameters...")
    print(f"Config DEVICE: {config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {device}")
    print(f"Hyperparameters: {hyperparams}")
    
    try:
        # Run 5-fold cross-validation
        for fold in range(config.TOTAL_FOLDS):
            print(f"\n  Processing fold {fold + 1}/5...")
            
            # Load data (same as in main script)
            positive5foldsidx = np.load(
                f"./our_dataset/{dataset}/index/positive5foldsidx.npy",
                allow_pickle=True,
            )
            negative5foldsidx = np.load(
                f"./our_dataset/{dataset}/index/negative5foldsidx.npy",
                allow_pickle=True,
            )
            positive_ij = np.load(f"./our_dataset/{dataset}/index/positive_ij.npy")
            negative_ij = np.load(f"./our_dataset/{dataset}/index/negative_ij.npy")
            
            # Split train/validation/test
            train_positive_idx = positive5foldsidx[fold]["train"]
            train_negative_idx = negative5foldsidx[fold]["train"]
            
            # Use 20% of training data for validation
            val_split = int(0.8 * len(train_positive_idx))
            
            train_positive_ij = positive_ij[train_positive_idx[:val_split]]
            train_negative_ij = negative_ij[train_negative_idx[:val_split]]
            
            val_positive_ij = positive_ij[train_positive_idx[val_split:]]
            val_negative_ij = negative_ij[train_negative_idx[val_split:]]
            
            test_positive_ij = positive_ij[positive5foldsidx[fold]["test"]]
            test_negative_ij = negative_ij[negative5foldsidx[fold]["test"]]
            
            # Load similarity matrices
            di_semantic_similarity = np.load(f"./our_dataset/{dataset}/multi_similarities/di_semantic_similarity.npy")
            di_gip_similarity = np.load(f"./our_dataset/{dataset}/multi_similarities/di_gip_similarity_fold_{fold+1}.npy")
            lnc_gip_similarity = np.load(f"./our_dataset/{dataset}/multi_similarities/lnc_gip_similarity_fold_{fold+1}.npy")
            lnc_func_similarity = np.load(f"./our_dataset/{dataset}/multi_similarities/lnc_func_similarity_fold_{fold+1}.npy")
            mi_gip_similarity = np.load(f"./our_dataset/{dataset}/multi_similarities/mi_gip_similarity.npy")
            mi_func_similarity = np.load(f"./our_dataset/{dataset}/multi_similarities/mi_func_similarity.npy")
            
            # Load interaction matrices
            lnc_di = pd.read_csv(f'./our_dataset/{dataset}/interaction/lnc_di.csv')
            lnc_di.set_index('0', inplace=True)
            lnc_di = lnc_di.values
            lnc_di_copy = copy.copy(lnc_di)
            
            lnc_mi = pd.read_csv(f'./our_dataset/{dataset}/interaction/lnc_mi.csv', index_col='0').values
            mi_di = pd.read_csv(f'./our_dataset/{dataset}/interaction/mi_di.csv')
            mi_di.set_index('0', inplace=True)
            mi_di = mi_di.values
            
            # Get dimensions
            num_diseases = di_semantic_similarity.shape[0]
            num_lnc = lnc_gip_similarity.shape[0]
            num_mi = mi_gip_similarity.shape[0]
            lncRNALen = num_lnc
            
            # Remove test edges from training adjacency matrix
            for ij in positive_ij[positive5foldsidx[fold]['test']]:
                lnc_di_copy[ij[0], ij[1] - lncRNALen] = 0
            
            # Create adjacency matrices
            disease_adjacency_matrices = [
                torch.tensor(di_semantic_similarity, dtype=torch.float32).to(device),
                torch.tensor(di_gip_similarity, dtype=torch.float32).to(device)
            ]
            lnc_adjacency_matrices = [
                torch.tensor(lnc_gip_similarity, dtype=torch.float32).to(device),
                torch.tensor(lnc_func_similarity, dtype=torch.float32).to(device)
            ]
            mi_adjacency_matrices = [
                torch.tensor(mi_gip_similarity, dtype=torch.float32).to(device),
                torch.tensor(mi_func_similarity, dtype=torch.float32).to(device)
            ]
            
            # Create datasets
            train_dataset = JointDataset(train_positive_ij, train_negative_ij, "train", dataset)
            val_dataset = JointDataset(val_positive_ij, val_negative_ij, "val", dataset)
            test_dataset = JointDataset(test_positive_ij, test_negative_ij, "test", dataset)
            
            # Prepare multi-view data
            multi_view_data = {
                'disease': disease_adjacency_matrices,
                'lnc': lnc_adjacency_matrices,
                'mi': mi_adjacency_matrices
            }
            
            # Prepare interaction tensors
            lnc_di_tensor = torch.tensor(lnc_di_copy, dtype=torch.float32).to(device)
            lnc_mi_tensor = torch.tensor(lnc_mi, dtype=torch.float32).to(device)
            mi_di_tensor = torch.tensor(mi_di, dtype=torch.float32).to(device)
            
            # Model parameters
            num_nodes = num_lnc + num_diseases + num_mi
            in_dimension = num_nodes
            total_links = (lnc_di_copy.sum() + lnc_mi.sum() + mi_di.sum())*2 + num_nodes
            pos_weight = torch.tensor(float(num_nodes**2 - total_links) / total_links, dtype=torch.float32, device=device)
            
            # Initialize model
            model = JointVGAE_LDAGM(
                num_lnc=num_lnc,
                num_diseases=num_diseases,
                num_mi=num_mi,
                vgae_in_dim=in_dimension,
                vgae_hidden_dim=hyperparams['vgae_hidden_dim'],
                vgae_embed_dim=hyperparams['vgae_embed_dim'],
                ldagm_hidden_dim=hyperparams['ldagm_hidden_dim'],
                ldagm_layers=hyperparams['ldagm_layers'],
                drop_rate=hyperparams['drop_rate'],
                use_aggregate=True,  # Fixed to True as requested
                gcn_hidden_dim=hyperparams['gcn_hidden_dim'],
                fusion_output_dim=hyperparams['fusion_output_dim']
            ).to(device)
            
            # Train with early stopping
            trained_model, best_val_auc, actual_epochs = train_with_early_stopping(
                model, train_dataset, val_dataset, multi_view_data,
                lnc_di_tensor, lnc_mi_tensor, mi_di_tensor,
                num_nodes, pos_weight, fold, hyperparams, device
            )
            
            # Test the model
            test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
            
            trained_model.eval()
            test_predictions = []
            test_labels = []
            
            with torch.no_grad():
                for batch_node_pairs, batch_labels in test_loader:
                    batch_node_pairs = batch_node_pairs.to(device)
                    
                    _, _, _, link_predictions = trained_model(
                        multi_view_data, lnc_di_tensor, lnc_mi_tensor, mi_di_tensor,
                        batch_node_pairs, network_num=FIXED_NETWORK_NUM, fold=fold, dataset=dataset
                    )
                    
                    predictions = torch.sigmoid(link_predictions)
                    test_predictions.extend(predictions.cpu().numpy())
                    test_labels.extend(batch_labels.numpy())
            
            # Calculate metrics
            test_predictions = np.array(test_predictions)
            test_labels = np.array(test_labels)
            
            AUC = roc_auc_score(test_labels, test_predictions)
            precision, recall, _ = precision_recall_curve(test_labels, test_predictions)
            AUPR = auc(recall, precision)
            
            binary_preds = (test_predictions > 0.5).astype(int)
            MCC = matthews_corrcoef(test_labels, binary_preds)
            ACC = accuracy_score(test_labels, binary_preds)
            P = precision_score(test_labels, binary_preds)
            R = recall_score(test_labels, binary_preds)
            F1 = f1_score(test_labels, binary_preds)
            
            fold_aucs.append(AUC)
            fold_metrics.append({
                'fold': fold + 1,
                'AUC': AUC,
                'AUPR': AUPR,
                'MCC': MCC,
                'ACC': ACC,
                'Precision': P,
                'Recall': R,
                'F1': F1,
                'epochs_trained': actual_epochs,
                'best_val_auc': best_val_auc
            })
            
            print(f"    Fold {fold + 1} - Test AUC: {AUC:.4f}, Val AUC: {best_val_auc:.4f}, Epochs: {actual_epochs}")
            
            # Report intermediate value for pruning
            trial.report(np.mean(fold_aucs), fold)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Calculate average metrics
        avg_auc = np.mean(fold_aucs)
        runtime = time.time() - start_time
        
        # Prepare metrics for logging
        avg_metrics = {
            f'avg_{key}': np.mean([m[key] for m in fold_metrics if key in m])
            for key in ['AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1', 'epochs_trained']
        }
        avg_metrics['std_AUC'] = np.std(fold_aucs)
        
        # Log trial results
        log_trial_to_csv(hyperparams, avg_metrics, trial.number, runtime)
        
        print(f"\nTrial {trial.number} completed - Average AUC: {avg_auc:.4f}")
        
        return avg_auc
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        return 0.0


def update_config_with_best_params(best_params):
    """Update config.py with the best hyperparameters"""
    
    # Read current config file
    with open('config.py', 'r') as f:
        lines = f.readlines()
    
    # Update specific parameters
    param_mapping = {
        'batch_size': 'BATCH_SIZE',
        'learning_rate': 'LEARNING_RATE',
        'weight_decay': 'WEIGHT_DECAY',
        'vgae_weight': 'VGAE_WEIGHT',
        'link_weight': 'LINK_WEIGHT',
        'kl_weight': 'KL_WEIGHT',
        'vgae_hidden_dim': 'VGAE_HIDDEN_DIM',
        'vgae_embed_dim': 'VGAE_EMBED_DIM',
        'ldagm_hidden_dim': 'LDAGM_HIDDEN_DIM',
        'ldagm_layers': 'LDAGM_LAYERS',
        'drop_rate': 'DROP_RATE',
        'gcn_hidden_dim': 'GCN_HIDDEN_DIM',
        'fusion_output_dim': 'FUSION_OUTPUT_DIM'
    }
    
    # Update lines
    for i, line in enumerate(lines):
        for param_name, config_name in param_mapping.items():
            if line.strip().startswith(f'{config_name} ='):
                value = best_params[param_name]
                lines[i] = f'{config_name} = {value}\n'
                break
    
    # Write back to config file
    with open('config.py', 'w') as f:
        f.writelines(lines)
    
    print(f"\nConfig file updated with best hyperparameters!")


def main():
    """Main function to run hyperparameter tuning"""
    
    print("Starting hyperparameter tuning for main.py...")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Fixed NETWORK_NUM: {FIXED_NETWORK_NUM}")
    print(f"Results will be saved to: {TUNING_CSV_FILE}")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("="*60)
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best average AUC: {best_value:.4f}")
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save best parameters to file
    os.makedirs(os.path.dirname(BEST_PARAMS_FILE), exist_ok=True)
    with open(BEST_PARAMS_FILE, 'w') as f:
        f.write(f"Best trial number: {study.best_trial.number}\n")
        f.write(f"Best average AUC: {best_value:.4f}\n")
        f.write("\nBest hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    
    # Update config.py with best parameters
    update_config_with_best_params(best_params)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TUNING SUMMARY")
    print("="*60)
    
    # Read and analyze the CSV results
    if os.path.exists(TUNING_CSV_FILE):
        df = pd.read_csv(TUNING_CSV_FILE)
        print(f"Total trials completed: {len(df)}")
        print(f"Best AUC: {df['avg_AUC'].max():.4f}")
        print(f"Worst AUC: {df['avg_AUC'].min():.4f}")
        print(f"Average AUC: {df['avg_AUC'].mean():.4f}")
        print(f"Std AUC: {df['avg_AUC'].std():.4f}")
        
        # Find trial with best AUC
        best_trial_idx = df['avg_AUC'].idxmax()
        best_trial_data = df.iloc[best_trial_idx]
        
        print(f"\nBest trial details:")
        print(f"  Trial number: {best_trial_data['trial_number']}")
        print(f"  Average epochs trained: {best_trial_data['avg_epochs_trained']:.1f}")
        print(f"  Runtime: {best_trial_data['runtime_seconds']:.2f} seconds")
        
        print(f"\n[Results saved to {TUNING_CSV_FILE}]")
        print(f"[Best parameters saved to {BEST_PARAMS_FILE}]")
        print("[Config file updated with best hyperparameters]")


if __name__ == '__main__':
    main()