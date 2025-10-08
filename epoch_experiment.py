"""
Epoch Experiment Script
Tests different epoch values with multiple runs and logs results.

Epochs to test: 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100
Each epoch configuration runs 10 times for statistical significance.
"""

import torch
import numpy as np
import pandas as pd
import copy
import time
from datetime import datetime
import os
import sys
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

# Import from main.py
from main import JointVGAE_LDAGM, JointDataset, joint_train, joint_test
import config

# Experiment configuration
EPOCHS_TO_TEST = [200 500]
RUNS_PER_EPOCH = 3
LOG_DIR = "logs/epoch_experiments"
RESULTS_FILE = f"{LOG_DIR}/epoch_experiment_results.csv"
SUMMARY_FILE = f"{LOG_DIR}/epoch_experiment_summary.txt"
DETAILED_LOG = f"{LOG_DIR}/epoch_experiment_detailed_log.txt"

# Create log directory
os.makedirs(LOG_DIR, exist_ok=True)

def modify_config(epoch_value):
    """Modify config.py to set the EPOCHS value"""
    with open('config.py', 'r') as f:
        lines = f.readlines()
    
    with open('config.py', 'w') as f:
        for line in lines:
            if line.strip().startswith('EPOCHS ='):
                f.write(f'EPOCHS = {epoch_value}\n')
            else:
                f.write(line)
    
    print(f"✓ Config updated: EPOCHS = {epoch_value}")

def restore_config_backup():
    """Restore original config if needed"""
    # This assumes the original config has EPOCHS = 5
    # You might want to backup config.py first
    pass

def run_main_once(epoch_value, run_number, dataset='dataset2'):
    """Run training and testing using main.py functions directly"""
    print(f"\n{'='*60}")
    print(f"Epoch: {epoch_value} | Run: {run_number}/{RUNS_PER_EPOCH}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Get device
        if config.DEVICE == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif config.DEVICE == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        print(f"Using device: {device}")
        
        # Run 5-fold cross-validation
        fold_metrics = []
        
        for fold in range(config.TOTAL_FOLDS):
            print(f"  Fold {fold + 1}/5...", end=' ')
            fold_start = time.time()
            
            # Load data indices
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
            
            train_positive_ij = positive_ij[positive5foldsidx[fold]["train"]]
            train_negative_ij = negative_ij[negative5foldsidx[fold]["train"]]
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
            
            # Create adjacency matrices for each entity type
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
            
            # Prepare multi-view data structure
            multi_view_data = {
                'disease': disease_adjacency_matrices,
                'lnc': lnaRNA_adjacency_matrices,
                'mi': miRNA_adjacency_matrices
            }
            
            # Prepare interaction matrices as tensors
            lnc_di_tensor = torch.tensor(lnc_di_copy, dtype=torch.float32).to(device)
            lnc_mi_tensor = torch.tensor(lnc_mi, dtype=torch.float32).to(device)
            mi_di_tensor = torch.tensor(mi_di, dtype=torch.float32).to(device)
            
            # Train model
            trained_model, loss_history = joint_train(
                num_lnc=num_lnc,
                num_diseases=num_diseases,
                num_mi=num_mi,
                train_dataset=train_dataset,
                multi_view_data=multi_view_data,
                lnc_di_interaction=lnc_di_tensor,
                lnc_mi_interaction=lnc_mi_tensor,
                mi_di_interaction=mi_di_tensor,
                fold=fold,
                device=device
            )
            
            # Test model
            test_labels, test_predictions = joint_test(
                model=trained_model,
                test_dataset=test_dataset,
                multi_view_data=multi_view_data,
                lnc_di_interaction=lnc_di_tensor,
                lnc_mi_interaction=lnc_mi_tensor,
                mi_di_interaction=mi_di_tensor,
                fold=fold,
                batch_size=config.BATCH_SIZE,
                device=device
            )
            
            # Calculate metrics
            AUC = roc_auc_score(test_labels, test_predictions)
            precision, recall, _ = precision_recall_curve(test_labels, test_predictions)
            AUPR = auc(recall, precision)
            
            binary_preds = (test_predictions > 0.5).astype(int)
            MCC = matthews_corrcoef(test_labels, binary_preds)
            ACC = accuracy_score(test_labels, binary_preds)
            P = precision_score(test_labels, binary_preds)
            R = recall_score(test_labels, binary_preds)
            F1 = f1_score(test_labels, binary_preds)
            
            fold_metrics.append({
                'AUC': AUC, 'AUPR': AUPR, 'MCC': MCC, 'ACC': ACC,
                'Precision': P, 'Recall': R, 'F1-Score': F1
            })
            
            fold_time = (time.time() - fold_start) / 60
            print(f"Done ({fold_time:.1f}min, AUC={AUC:.4f})")
        
        # Calculate average metrics across folds
        avg_metrics = {
            'AUC': np.mean([f['AUC'] for f in fold_metrics]),
            'AUPR': np.mean([f['AUPR'] for f in fold_metrics]),
            'MCC': np.mean([f['MCC'] for f in fold_metrics]),
            'ACC': np.mean([f['ACC'] for f in fold_metrics]),
            'Precision': np.mean([f['Precision'] for f in fold_metrics]),
            'Recall': np.mean([f['Recall'] for f in fold_metrics]),
            'F1-Score': np.mean([f['F1-Score'] for f in fold_metrics]),
        }
        
        end_time = time.time()
        duration = end_time - start_time
        
        avg_metrics['duration_seconds'] = duration
        avg_metrics['duration_minutes'] = duration / 60
        avg_metrics['success'] = True
        
        # Log to detailed log
        with open(DETAILED_LOG, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Epochs: {epoch_value} | Run: {run_number}\n")
            f.write(f"Duration: {duration/60:.2f} minutes\n")
            f.write(f"Metrics: {avg_metrics}\n")
            f.write(f"Fold-wise metrics:\n")
            for i, fm in enumerate(fold_metrics):
                f.write(f"  Fold {i+1}: AUC={fm['AUC']:.4f}, AUPR={fm['AUPR']:.4f}\n")
            f.write(f"{'='*80}\n")
        
        print(f"✓ Completed in {duration/60:.2f} minutes")
        print(f"  AUC: {avg_metrics['AUC']:.4f}")
        print(f"  AUPR: {avg_metrics['AUPR']:.4f}")
        
        return avg_metrics
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Log error
        with open(DETAILED_LOG, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ERROR - Timestamp: {datetime.now()}\n")
            f.write(f"Epochs: {epoch_value} | Run: {run_number}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc())
            f.write(f"{'='*80}\n")
        
        return {
            'AUC': None, 'AUPR': None, 'MCC': None, 'ACC': None,
            'Precision': None, 'Recall': None, 'F1-Score': None,
            'duration_seconds': 0, 'duration_minutes': 0,
            'success': False, 'error': str(e)
        }

def run_epoch_experiment(epoch_value):
    """Run the experiment for a specific epoch value"""
    print(f"\n{'#'*80}")
    print(f"# STARTING EXPERIMENT FOR EPOCHS = {epoch_value}")
    print(f"{'#'*80}")
    
    # Modify config
    modify_config(epoch_value)
    
    # Reload config to get updated EPOCHS value
    import importlib
    importlib.reload(config)
    
    # Store all results for this epoch
    epoch_results = []
    
    # Run multiple times
    for run_num in range(1, RUNS_PER_EPOCH + 1):
        metrics = run_main_once(epoch_value, run_num, dataset=config.DATASET)
        metrics['epoch_value'] = epoch_value
        metrics['run_number'] = run_num
        metrics['timestamp'] = datetime.now().isoformat()
        epoch_results.append(metrics)
        
        # Save intermediate results
        df = pd.DataFrame([metrics])
        if os.path.exists(RESULTS_FILE):
            df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(RESULTS_FILE, mode='w', header=True, index=False)
    
    return epoch_results

def compute_statistics(results):
    """Compute statistics for each epoch configuration"""
    df = pd.DataFrame(results)
    
    # Filter successful runs
    successful = df[df['success'] == True]
    
    stats = {
        'epoch_value': results[0]['epoch_value'],
        'total_runs': len(results),
        'successful_runs': len(successful),
        'failed_runs': len(results) - len(successful),
    }
    
    # Compute mean and std for metrics
    metric_names = ['AUC', 'AUPR', 'MCC', 'ACC', 'Precision', 'Recall', 'F1-Score', 'duration_minutes']
    
    for metric in metric_names:
        if len(successful) > 0:
            values = successful[metric].dropna()
            if len(values) > 0:
                stats[f'{metric}_mean'] = values.mean()
                stats[f'{metric}_std'] = values.std()
                stats[f'{metric}_min'] = values.min()
                stats[f'{metric}_max'] = values.max()
            else:
                stats[f'{metric}_mean'] = None
                stats[f'{metric}_std'] = None
                stats[f'{metric}_min'] = None
                stats[f'{metric}_max'] = None
        else:
            stats[f'{metric}_mean'] = None
            stats[f'{metric}_std'] = None
            stats[f'{metric}_min'] = None
            stats[f'{metric}_max'] = None
    
    return stats

def print_summary(all_stats):
    """Print and save summary of all experiments"""
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("EPOCH EXPERIMENT RESULTS SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append(f"Total configurations tested: {len(all_stats)}")
    summary_lines.append(f"Runs per configuration: {RUNS_PER_EPOCH}")
    summary_lines.append(f"Timestamp: {datetime.now()}")
    summary_lines.append("="*80)
    summary_lines.append("")
    
    for stats in all_stats:
        epoch = stats['epoch_value']
        summary_lines.append(f"\nEPOCHS = {epoch}")
        summary_lines.append("-" * 40)
        summary_lines.append(f"Successful runs: {stats['successful_runs']}/{stats['total_runs']}")
        
        if stats['AUC_mean'] is not None:
            summary_lines.append(f"AUC:       {stats['AUC_mean']:.4f} ± {stats['AUC_std']:.4f}  (min: {stats['AUC_min']:.4f}, max: {stats['AUC_max']:.4f})")
            summary_lines.append(f"AUPR:      {stats['AUPR_mean']:.4f} ± {stats['AUPR_std']:.4f}  (min: {stats['AUPR_min']:.4f}, max: {stats['AUPR_max']:.4f})")
            summary_lines.append(f"ACC:       {stats['ACC_mean']:.4f} ± {stats['ACC_std']:.4f}  (min: {stats['ACC_min']:.4f}, max: {stats['ACC_max']:.4f})")
            summary_lines.append(f"F1-Score:  {stats['F1-Score_mean']:.4f} ± {stats['F1-Score_std']:.4f}  (min: {stats['F1-Score_min']:.4f}, max: {stats['F1-Score_max']:.4f})")
            summary_lines.append(f"Avg Time:  {stats['duration_minutes_mean']:.2f} ± {stats['duration_minutes_std']:.2f} minutes")
        else:
            summary_lines.append("No successful runs")
    
    # Print to console
    for line in summary_lines:
        print(line)
    
    # Save to file
    with open(SUMMARY_FILE, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ Summary saved to: {SUMMARY_FILE}")
    print(f"✓ Detailed results saved to: {RESULTS_FILE}")
    print(f"✓ Detailed logs saved to: {DETAILED_LOG}")

def main():
    """Main experiment runner"""
    print(f"\n{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + "EPOCH EXPERIMENT SCRIPT".center(78) + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}\n")
    
    print(f"Epochs to test: {EPOCHS_TO_TEST}")
    print(f"Runs per epoch: {RUNS_PER_EPOCH}")
    print(f"Total runs: {len(EPOCHS_TO_TEST) * RUNS_PER_EPOCH}")
    print(f"Log directory: {LOG_DIR}")
    
    # Confirm before starting
    print("\n⚠️  This experiment will take a long time!")
    response = input("Do you want to continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Experiment cancelled.")
        return
    
    experiment_start = time.time()
    
    # Initialize detailed log
    with open(DETAILED_LOG, 'w') as f:
        f.write(f"EPOCH EXPERIMENT - DETAILED LOG\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"Epochs to test: {EPOCHS_TO_TEST}\n")
        f.write(f"Runs per epoch: {RUNS_PER_EPOCH}\n")
        f.write(f"{'='*80}\n\n")
    
    # Run experiments for each epoch value
    all_results = []
    all_stats = []
    
    for epoch_value in EPOCHS_TO_TEST:
        epoch_results = run_epoch_experiment(epoch_value)
        all_results.extend(epoch_results)
        
        # Compute and display statistics for this epoch
        stats = compute_statistics(epoch_results)
        all_stats.append(stats)
        
        print(f"\n📊 Statistics for EPOCHS = {epoch_value}:")
        if stats['AUC_mean'] is not None:
            print(f"   AUC: {stats['AUC_mean']:.4f} ± {stats['AUC_std']:.4f}")
            print(f"   Time: {stats['duration_minutes_mean']:.2f} ± {stats['duration_minutes_std']:.2f} min")
        print(f"   Success rate: {stats['successful_runs']}/{stats['total_runs']}")
    
    experiment_end = time.time()
    total_duration = (experiment_end - experiment_start) / 3600  # hours
    
    # Print final summary
    print_summary(all_stats)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETED")
    print(f"Total time: {total_duration:.2f} hours")
    print(f"{'='*80}\n")
    
    # Save statistics summary as CSV
    stats_df = pd.DataFrame(all_stats)
    stats_file = f"{LOG_DIR}/epoch_experiment_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"✓ Statistics saved to: {stats_file}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        print("Partial results have been saved.")
    except Exception as e:
        print(f"\n\n✗ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()

