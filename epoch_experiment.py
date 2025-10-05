"""
Epoch Experiment Script
Tests different epoch values with multiple runs and logs results.

Epochs to test: 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100
Each epoch configuration runs 10 times for statistical significance.
"""

import subprocess
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import os
import sys

# Experiment configuration
EPOCHS_TO_TEST = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100]
RUNS_PER_EPOCH = 10
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

def parse_output_for_metrics(output):
    """
    Parse the output from main.py to extract metrics.
    Returns a dictionary with AUC, AUPR, ACC, etc.
    """
    metrics = {
        'AUC': None,
        'AUPR': None,
        'MCC': None,
        'ACC': None,
        'Precision': None,
        'Recall': None,
        'F1-Score': None
    }
    
    try:
        # Look for lines like "AUC: 0.xxxx ± 0.xxxx"
        lines = output.split('\n')
        for line in lines:
            for metric_name in metrics.keys():
                if metric_name in line and '±' in line:
                    # Extract the mean value (before ±)
                    parts = line.split(':')
                    if len(parts) >= 2:
                        value_part = parts[1].strip().split('±')[0].strip()
                        try:
                            metrics[metric_name] = float(value_part)
                        except ValueError:
                            pass
    except Exception as e:
        print(f"Error parsing output: {e}")
    
    return metrics

def run_main_once(epoch_value, run_number):
    """Run main.py once and return the results"""
    print(f"\n{'='*60}")
    print(f"Epoch: {epoch_value} | Run: {run_number}/{RUNS_PER_EPOCH}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run main.py and capture output
        result = subprocess.run(
            [sys.executable, 'main.py'],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse metrics from output
        metrics = parse_output_for_metrics(result.stdout)
        metrics['duration_seconds'] = duration
        metrics['duration_minutes'] = duration / 60
        metrics['success'] = True
        
        # Log to detailed log
        with open(DETAILED_LOG, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Epochs: {epoch_value} | Run: {run_number}\n")
            f.write(f"Duration: {duration/60:.2f} minutes\n")
            f.write(f"Metrics: {metrics}\n")
            f.write(f"{'='*80}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"STDERR:\n{result.stderr}\n")
        
        print(f"✓ Completed in {duration/60:.2f} minutes")
        if metrics['AUC'] is not None:
            print(f"  AUC: {metrics['AUC']:.4f}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 2 hours")
        return {
            'AUC': None, 'AUPR': None, 'MCC': None, 'ACC': None,
            'Precision': None, 'Recall': None, 'F1-Score': None,
            'duration_seconds': 7200, 'duration_minutes': 120,
            'success': False, 'error': 'Timeout'
        }
    except Exception as e:
        print(f"✗ Error: {e}")
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
    
    # Store all results for this epoch
    epoch_results = []
    
    # Run multiple times
    for run_num in range(1, RUNS_PER_EPOCH + 1):
        metrics = run_main_once(epoch_value, run_num)
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

