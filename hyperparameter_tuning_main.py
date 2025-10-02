"""
Hyperparameter Tuning Script for main.py using Optuna
This script uses Optuna for efficient hyperparameter optimization and trains/tests
exactly like main.py
"""

import torch
import numpy as np
import pandas as pd
import copy
import time
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from main import (
    JointDataset, 
    JointVGAE_LDAGM, 
    joint_train, 
    joint_test
)
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
import os
import csv
import config
warnings.filterwarnings("ignore")


# ============================================================================
# Hyperparameter Search Space Definition (for Optuna)
# ============================================================================

HYPERPARAMETER_SPACE = {
    # Model architecture - categorical choices
    'GCN_HIDDEN_DIM': [32, 64, 128],
    'FUSION_OUTPUT_DIM': [32, 64, 128],
    'VGAE_HIDDEN_DIM': [32, 64, 128],
    'VGAE_EMBED_DIM': [32, 64, 128],
    'LDAGM_HIDDEN_DIM': [30, 40, 50],
    'LDAGM_LAYERS': [3, 4, 5],
    'USE_AGGREGATE': [True],  # Fixed to True
    
    # Continuous parameters
    'DROP_RATE': (0.1, 0.4),  # min, max
    'LEARNING_RATE': (1e-5, 1e-2),  # min, max (log scale)
    'WEIGHT_DECAY': (1e-6, 1e-3),  # min, max (log scale)
    
    # Categorical parameters
    'BATCH_SIZE': [16, 32, 64],
    'EPOCHS': [100],  # Fixed to 100
    
        # Loss weights - continuous
        'VGAE_WEIGHT': (0.5, 1.0),
        'LINK_WEIGHT': (1.0, 4.0),
        'KL_WEIGHT': (0.1, 0.4),
}

# Optuna configuration
N_TRIALS = 150  # Number of trials to run
N_STARTUP_TRIALS = 10  # Number of random trials before TPE starts (patience)
PRUNING_ENABLED = True  # Enable pruning of unpromising trials


def load_fold_data(dataset, device, fold):
    """Load data for a specific fold (optimized to avoid reloading)"""
    # Load indices
    positive5foldsidx = np.load("./our_dataset/" + dataset + "/index/positive5foldsidx.npy", allow_pickle=True)
    negative5foldsidx = np.load("./our_dataset/" + dataset + "/index/negative5foldsidx.npy", allow_pickle=True)
    positive_ij = np.load("./our_dataset/" + dataset + "/index/positive_ij.npy")
    negative_ij = np.load("./our_dataset/" + dataset + "/index/negative_ij.npy")
    
    train_positive_ij = positive_ij[positive5foldsidx[fold]["train"]]
    train_negative_ij = negative_ij[negative5foldsidx[fold]["train"]]
    test_positive_ij = positive_ij[positive5foldsidx[fold]["test"]]
    test_negative_ij = negative_ij[negative5foldsidx[fold]["test"]]
    
    # Load similarity matrices
    di_semantic_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/di_semantic_similarity.npy")
    di_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/di_gip_similarity_fold_{fold+1}.npy")
    lnc_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/lnc_gip_similarity_fold_{fold+1}.npy")
    lnc_func_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/lnc_func_similarity_fold_{fold+1}.npy")
    mi_gip_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/mi_gip_similarity.npy")
    mi_func_similarity = np.load("./our_dataset/" + dataset + f"/multi_similarities/mi_func_similarity.npy")
    
    # Load interaction matrices
    lnc_di = pd.read_csv("./our_dataset/" + dataset + "/interaction/lnc_di.csv")
    lnc_di.set_index("0", inplace=True)
    lnc_di = lnc_di.values
    lnc_di_copy = copy.copy(lnc_di)
    
    lnc_mi = pd.read_csv("./our_dataset/" + dataset + "/interaction/lnc_mi.csv", index_col="0").values
    mi_di = pd.read_csv("./our_dataset/" + dataset + "/interaction/mi_di.csv")
    mi_di.set_index("0", inplace=True)
    mi_di = mi_di.values
    
    # Get dimensions
    num_diseases = di_semantic_similarity.shape[0]
    num_lnc = lnc_gip_similarity.shape[0]
    num_mi = mi_gip_similarity.shape[0]
    lncRNALen = num_lnc
    
    # Remove test edges
    for ij in positive_ij[positive5foldsidx[fold]["test"]]:
        lnc_di_copy[ij[0], ij[1] - lncRNALen] = 0
    
    # Create adjacency matrices
    disease_adjacency_matrices = [
        torch.tensor(di_semantic_similarity, dtype=torch.float32).to(device),
        torch.tensor(di_gip_similarity, dtype=torch.float32).to(device)
    ]
    lnaRNA_adjacency_matrices = [
        torch.tensor(lnc_gip_similarity, dtype=torch.float32).to(device),
        torch.tensor(lnc_func_similarity, dtype=torch.float32).to(device)
    ]
    miRNA_adjacency_matrices = [
        torch.tensor(mi_gip_similarity, dtype=torch.float32).to(device),
        torch.tensor(mi_func_similarity, dtype=torch.float32).to(device)
    ]
    
    # Create datasets
    train_dataset = JointDataset(train_positive_ij, train_negative_ij, "train", dataset)
    test_dataset = JointDataset(test_positive_ij, test_negative_ij, "test", dataset)
    
    # Prepare multi-view data
    multi_view_data = {
        "disease": disease_adjacency_matrices,
        "lnc": lnaRNA_adjacency_matrices,
        "mi": miRNA_adjacency_matrices
    }
    
    # Prepare interaction tensors
    lnc_di_tensor = torch.tensor(lnc_di_copy, dtype=torch.float32).to(device)
    lnc_mi_tensor = torch.tensor(lnc_mi, dtype=torch.float32).to(device)
    mi_di_tensor = torch.tensor(mi_di, dtype=torch.float32).to(device)
    
    return {
        "num_lnc": num_lnc,
        "num_diseases": num_diseases,
        "num_mi": num_mi,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "multi_view_data": multi_view_data,
        "lnc_di_tensor": lnc_di_tensor,
        "lnc_mi_tensor": lnc_mi_tensor,
        "mi_di_tensor": mi_di_tensor
    }


def train_and_evaluate_single_fold(
    fold, dataset, fold_data, hyperparams, device
):
    """
    Train and evaluate a single fold with given hyperparameters.
    """
    
    # Train the model
    trained_model, loss_history = joint_train(
        num_lnc=fold_data["num_lnc"],
        num_diseases=fold_data["num_diseases"],
        num_mi=fold_data["num_mi"],
        train_dataset=fold_data["train_dataset"],
        multi_view_data=fold_data["multi_view_data"],
        lnc_di_interaction=fold_data["lnc_di_tensor"],
        lnc_mi_interaction=fold_data["lnc_mi_tensor"],
        mi_di_interaction=fold_data["mi_di_tensor"],
        fold=fold,
        batch_size=hyperparams['BATCH_SIZE'],
        epochs=hyperparams['EPOCHS'],
        lr=hyperparams['LEARNING_RATE'],
        weight_decay=hyperparams['WEIGHT_DECAY'],
        device=device,
        vgae_weight=hyperparams['VGAE_WEIGHT'],
        link_weight=hyperparams['LINK_WEIGHT'],
        kl_weight=hyperparams['KL_WEIGHT'],
        vgae_hidden_dim=hyperparams['VGAE_HIDDEN_DIM'],
        vgae_embed_dim=hyperparams['VGAE_EMBED_DIM'],
        ldagm_hidden_dim=hyperparams['LDAGM_HIDDEN_DIM'],
        ldagm_layers=hyperparams['LDAGM_LAYERS']
    )
    
    # Test the model
    test_labels, test_predictions = joint_test(
        model=trained_model,
        test_dataset=fold_data["test_dataset"],
        multi_view_data=fold_data["multi_view_data"],
        lnc_di_interaction=fold_data["lnc_di_tensor"],
        lnc_mi_interaction=fold_data["lnc_mi_tensor"],
        mi_di_interaction=fold_data["mi_di_tensor"],
        fold=fold,
        batch_size=hyperparams['BATCH_SIZE'],
        device=device
    )
    
    # Calculate metrics
    AUC = roc_auc_score(test_labels, test_predictions)
    precision, recall, _ = precision_recall_curve(test_labels, test_predictions)
    AUPR = auc(recall, precision)
    
    # Binary predictions for other metrics
    binary_preds = (test_predictions > 0.5).astype(int)
    MCC = matthews_corrcoef(test_labels, binary_preds)
    ACC = accuracy_score(test_labels, binary_preds)
    P = precision_score(test_labels, binary_preds)
    R = recall_score(test_labels, binary_preds)
    F1 = f1_score(test_labels, binary_preds)
    
    metrics = {
        "AUC": AUC,
        "AUPR": AUPR,
        "MCC": MCC,
        "ACC": ACC,
        "Precision": P,
        "Recall": R,
        "F1-Score": F1
    }
    
    return metrics


def objective(trial, dataset, device, total_folds=5):
    """
    Optuna objective function for hyperparameter optimization.
    """
    
    # Sample hyperparameters from the search space
    hyperparams = {
        'GCN_HIDDEN_DIM': trial.suggest_categorical('GCN_HIDDEN_DIM', HYPERPARAMETER_SPACE['GCN_HIDDEN_DIM']),
        'FUSION_OUTPUT_DIM': trial.suggest_categorical('FUSION_OUTPUT_DIM', HYPERPARAMETER_SPACE['FUSION_OUTPUT_DIM']),
        'VGAE_HIDDEN_DIM': trial.suggest_categorical('VGAE_HIDDEN_DIM', HYPERPARAMETER_SPACE['VGAE_HIDDEN_DIM']),
        'VGAE_EMBED_DIM': trial.suggest_categorical('VGAE_EMBED_DIM', HYPERPARAMETER_SPACE['VGAE_EMBED_DIM']),
        'LDAGM_HIDDEN_DIM': trial.suggest_categorical('LDAGM_HIDDEN_DIM', HYPERPARAMETER_SPACE['LDAGM_HIDDEN_DIM']),
        'LDAGM_LAYERS': trial.suggest_categorical('LDAGM_LAYERS', HYPERPARAMETER_SPACE['LDAGM_LAYERS']),
        'USE_AGGREGATE': trial.suggest_categorical('USE_AGGREGATE', HYPERPARAMETER_SPACE['USE_AGGREGATE']),
        'DROP_RATE': trial.suggest_float('DROP_RATE', *HYPERPARAMETER_SPACE['DROP_RATE']),
        'LEARNING_RATE': trial.suggest_float('LEARNING_RATE', *HYPERPARAMETER_SPACE['LEARNING_RATE'], log=True),
        'WEIGHT_DECAY': trial.suggest_float('WEIGHT_DECAY', *HYPERPARAMETER_SPACE['WEIGHT_DECAY'], log=True),
        'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', HYPERPARAMETER_SPACE['BATCH_SIZE']),
        'EPOCHS': trial.suggest_categorical('EPOCHS', HYPERPARAMETER_SPACE['EPOCHS']),
        'VGAE_WEIGHT': trial.suggest_float('VGAE_WEIGHT', *HYPERPARAMETER_SPACE['VGAE_WEIGHT']),
        'LINK_WEIGHT': trial.suggest_float('LINK_WEIGHT', *HYPERPARAMETER_SPACE['LINK_WEIGHT']),
        'KL_WEIGHT': trial.suggest_float('KL_WEIGHT', *HYPERPARAMETER_SPACE['KL_WEIGHT']),
    }
    
    all_fold_results = []
    
    for fold in range(total_folds):
        print(f"\n  Trial {trial.number} - Fold {fold + 1}/{total_folds}...")
        
        # Load data for this fold (optimized - load once per fold)
        fold_data = load_fold_data(dataset, device, fold)
        
        # Train and evaluate for this fold
        fold_metrics = train_and_evaluate_single_fold(
            fold=fold,
            dataset=dataset,
            fold_data=fold_data,
            hyperparams=hyperparams,
            device=device
        )
        
        all_fold_results.append(fold_metrics)
        
        print(f"  Fold {fold + 1} - AUC: {fold_metrics['AUC']:.4f}, AUPR: {fold_metrics['AUPR']:.4f}")
        
        # Clean up memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Optuna pruning: check intermediate results
        if PRUNING_ENABLED and fold < total_folds - 1:
            intermediate_auc = np.mean([r['AUC'] for r in all_fold_results])
            trial.report(intermediate_auc, fold)
            
            if trial.should_prune():
                print(f"  Trial {trial.number} pruned at fold {fold + 1}")
                raise optuna.TrialPruned()
    
    # Calculate average metrics across all folds
    avg_auc = np.mean([r['AUC'] for r in all_fold_results])
    avg_aupr = np.mean([r['AUPR'] for r in all_fold_results])
    avg_mcc = np.mean([r['MCC'] for r in all_fold_results])
    avg_acc = np.mean([r['ACC'] for r in all_fold_results])
    avg_precision = np.mean([r['Precision'] for r in all_fold_results])
    avg_recall = np.mean([r['Recall'] for r in all_fold_results])
    avg_f1 = np.mean([r['F1-Score'] for r in all_fold_results])
    
    # Calculate standard deviations
    std_auc = np.std([r['AUC'] for r in all_fold_results])
    std_aupr = np.std([r['AUPR'] for r in all_fold_results])
    std_mcc = np.std([r['MCC'] for r in all_fold_results])
    std_acc = np.std([r['ACC'] for r in all_fold_results])
    std_precision = np.std([r['Precision'] for r in all_fold_results])
    std_recall = np.std([r['Recall'] for r in all_fold_results])
    std_f1 = np.std([r['F1-Score'] for r in all_fold_results])
    
    # Store all metrics as user attributes
    trial.set_user_attr('avg_aupr', avg_aupr)
    trial.set_user_attr('avg_mcc', avg_mcc)
    trial.set_user_attr('avg_acc', avg_acc)
    trial.set_user_attr('avg_precision', avg_precision)
    trial.set_user_attr('avg_recall', avg_recall)
    trial.set_user_attr('avg_f1', avg_f1)
    
    trial.set_user_attr('std_auc', std_auc)
    trial.set_user_attr('std_aupr', std_aupr)
    trial.set_user_attr('std_mcc', std_mcc)
    trial.set_user_attr('std_acc', std_acc)
    trial.set_user_attr('std_precision', std_precision)
    trial.set_user_attr('std_recall', std_recall)
    trial.set_user_attr('std_f1', std_f1)
    
    return avg_auc  # Optuna will maximize this


def hyperparameter_search(n_trials=N_TRIALS, study_name=None):
    """
    Main hyperparameter search function using Optuna.
    """
    
    # Configuration
    dataset = config.DATASET
    
    # Device selection
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        if config.DEVICE in ["cuda", "mps"]:
            print(f"Warning: {config.DEVICE} not available, falling back to CPU")
    
    print(f"Using device: {device}")
    print(f"Dataset: {dataset}")
    
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING WITH OPTUNA (for main.py)")
    print(f"{'='*60}")
    print(f"Number of trials: {n_trials}")
    print(f"Pruning enabled: {PRUNING_ENABLED}")
    print(f"Search space:")
    for key, value in HYPERPARAMETER_SPACE.items():
        if isinstance(value, tuple):
            print(f"  {key}: [{value[0]}, {value[1]}] (continuous)")
        else:
            print(f"  {key}: {value} (categorical)")
    print(f"{'='*60}\n")
    
    # Create or load study
    if study_name is None:
        study_name = f"ldap_main_tuning_{dataset}_{int(time.time())}"
    
    # Configure sampler and pruner
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=42)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS) if PRUNING_ENABLED else None
    
    # Create study (in-memory only)
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )
    
    # Log file for results
    log_file = f"logs/optuna_main_study_{dataset}.csv"
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    
    # Optimize
    start_time = time.time()
    
    def callback(study, trial):
        """Callback to save results after each trial"""
        result = {
            'trial_number': trial.number,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start,
            'datetime_complete': trial.datetime_complete,
            'duration_seconds': (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None,
        }
        
        result.update(trial.params)
        
        result['AUC'] = trial.value if trial.value is not None else None
        result['AUPR'] = trial.user_attrs.get('avg_aupr', None)
        result['MCC'] = trial.user_attrs.get('avg_mcc', None)
        result['ACC'] = trial.user_attrs.get('avg_acc', None)
        result['Precision'] = trial.user_attrs.get('avg_precision', None)
        result['Recall'] = trial.user_attrs.get('avg_recall', None)
        result['F1_Score'] = trial.user_attrs.get('avg_f1', None)
        
        result['AUC_std'] = trial.user_attrs.get('std_auc', None)
        result['AUPR_std'] = trial.user_attrs.get('std_aupr', None)
        result['MCC_std'] = trial.user_attrs.get('std_mcc', None)
        result['ACC_std'] = trial.user_attrs.get('std_acc', None)
        result['Precision_std'] = trial.user_attrs.get('std_precision', None)
        result['Recall_std'] = trial.user_attrs.get('std_recall', None)
        result['F1_Score_std'] = trial.user_attrs.get('std_f1', None)
        
        fieldnames = [
            'trial_number', 'state', 'datetime_start', 'datetime_complete', 'duration_seconds',
            'GCN_HIDDEN_DIM', 'FUSION_OUTPUT_DIM', 'VGAE_HIDDEN_DIM', 'VGAE_EMBED_DIM',
            'LDAGM_HIDDEN_DIM', 'LDAGM_LAYERS', 'USE_AGGREGATE', 'DROP_RATE', 'LEARNING_RATE', 'WEIGHT_DECAY',
            'BATCH_SIZE', 'EPOCHS', 'VGAE_WEIGHT', 'LINK_WEIGHT', 'KL_WEIGHT',
            'AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1_Score',
            'AUC_std', 'AUPR_std', 'MCC_std', 'ACC_std', 'Precision_std', 'Recall_std', 'F1_Score_std'
        ]
        
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)
        
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"\nTrial {trial.number} completed:")
            print(f"  AUC: {trial.value:.4f} ± {trial.user_attrs.get('std_auc', 0):.4f}")
            print(f"  AUPR: {trial.user_attrs.get('avg_aupr', 0):.4f} ± {trial.user_attrs.get('std_aupr', 0):.4f}")
            print(f"  MCC: {trial.user_attrs.get('avg_mcc', 0):.4f} ± {trial.user_attrs.get('std_mcc', 0):.4f}")
            print(f"  Best AUC so far: {study.best_value:.4f} (Trial {study.best_trial.number})")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"\nTrial {trial.number} pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            print(f"\nTrial {trial.number} failed: {trial.system_attrs.get('fail_reason', 'Unknown error')}")
    
    try:
        study.optimize(
            lambda trial: objective(trial, dataset, device, config.TOTAL_FOLDS),
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    # End timing
    end_time = time.time()
    total_seconds = end_time - start_time
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    
    # Print final results
    print(f"\n{'='*60}")
    print("OPTUNA OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"Total trials run: {len(study.trials)}")
    print(f"Total runtime: {int(h)} hours, {int(m)} minutes, and {s:.2f} seconds")
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {log_file}")
    print(f"{'='*60}\n")
    
    # Save summary
    summary_file = f"logs/optuna_main_summary_{dataset}.txt"
    with open(summary_file, 'w') as f:
        f.write("OPTUNA HYPERPARAMETER OPTIMIZATION SUMMARY (main.py)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"Total runtime: {int(h)} hours, {int(m)} minutes, and {s:.2f} seconds\n\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best AUC: {study.best_value:.4f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    
    return study


if __name__ == '__main__':
    print("Starting Hyperparameter Search with Optuna for main.py...")
    print("Edit N_TRIALS at the top of this file to adjust the number of trials.")
    print("Edit HYPERPARAMETER_SPACE to customize the search space.\n")
    
    study = hyperparameter_search(n_trials=N_TRIALS)
    
    print("\nHyperparameter search completed successfully!")
    print("Check the logs/ directory for detailed results.")
    print("\nTo visualize results:")
    print("  import optuna")
    print("  # Note: Study is in-memory only, cannot resume after script ends")
    print("  optuna.visualization.plot_optimization_history(study)")
    print("  optuna.visualization.plot_param_importances(study)")
