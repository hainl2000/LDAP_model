# 🧪 Complete Epoch Experiment Guide

## 📋 What You Have

I've created a complete experiment framework to test different epoch values:

### 🎯 Main Scripts

1. **`epoch_experiment.py`** - Main experiment script
   - Tests epochs: 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100
   - Runs each configuration 10 times
   - Total: 110 complete training runs
   - ⏱️ Time: 50-100+ hours

2. **`quick_test_experiment.py`** - Quick test version  
   - Tests epochs: 1, 2, 3
   - Runs each 2 times
   - Total: 6 runs
   - ⏱️ Time: ~30-60 minutes
   - **Use this first to verify everything works!**

3. **`visualize_epoch_results.py`** - Visualization script
   - Creates beautiful plots
   - Analyzes results
   - Finds optimal epoch value

4. **`run_epoch_experiment.sh`** - Easy launcher
   - Backs up config
   - Runs experiment
   - Restores config

## 🚀 Quick Start

### Step 1: Quick Test (RECOMMENDED)
```bash
python quick_test_experiment.py
```
This runs a small test (~30-60 min) to verify everything works.

### Step 2: Run Full Experiment
```bash
./run_epoch_experiment.sh
```
⚠️ This takes DAYS! Consider running in background:
```bash
nohup ./run_epoch_experiment.sh > experiment.out 2>&1 &
```

### Step 3: Visualize Results
```bash
python visualize_epoch_results.py
```
Creates plots and analysis of the results.

## 📊 What Gets Generated

### During Experiment:
```
logs/epoch_experiments/
├── epoch_experiment_results.csv          # All raw data
├── epoch_experiment_statistics.csv       # Statistical summary  
├── epoch_experiment_summary.txt          # Human-readable summary
└── epoch_experiment_detailed_log.txt     # Complete logs
```

### After Visualization:
```
logs/epoch_experiments/visualizations/
├── AUC_vs_epochs.png                     # AUC performance
├── AUPR_vs_epochs.png                    # AUPR performance
├── ACC_vs_epochs.png                     # Accuracy performance
├── F1-Score_vs_epochs.png                # F1 score
├── all_metrics_vs_epochs.png             # All metrics combined
├── variance_analysis.png                 # Box plots showing variance
├── performance_vs_time.png               # Efficiency analysis
└── visual_summary.txt                    # Best configurations
```

## 📈 Example Results

### Summary Output:
```
EPOCHS = 5
----------------------------------------
Successful runs: 10/10
AUC:       0.9234 ± 0.0123  (min: 0.9100, max: 0.9350)
AUPR:      0.8567 ± 0.0145  (min: 0.8400, max: 0.8700)
ACC:       0.8934 ± 0.0098  (min: 0.8800, max: 0.9050)
F1-Score:  0.8876 ± 0.0110  (min: 0.8750, max: 0.9000)
Avg Time:  45.23 ± 3.45 minutes
```

## 🔧 Customization

### Test Different Epochs
Edit `epoch_experiment.py`:
```python
EPOCHS_TO_TEST = [5, 10, 20, 50]  # Your custom list
```

### Change Number of Runs
Edit `epoch_experiment.py`:
```python
RUNS_PER_EPOCH = 5  # Fewer runs = faster but less reliable
```

### Modify Config Settings
The script automatically modifies `config.py` and restores it after.
To change other parameters, edit them in `config.py` before running.

## 📱 Monitoring Progress

### Watch Live Progress:
```bash
tail -f logs/epoch_experiments/epoch_experiment_detailed_log.txt
```

### Check Partial Results:
```bash
# View results so far
python -c "import pandas as pd; df = pd.read_csv('logs/epoch_experiments/epoch_experiment_results.csv'); print(df.groupby('epoch_value')['AUC'].describe())"
```

### Background Job Status:
```bash
# Check if running
ps aux | grep epoch_experiment

# Check output
tail experiment.out
```

## 🎨 Analyzing Results

### Load in Python:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('logs/epoch_experiments/epoch_experiment_results.csv')
stats = pd.read_csv('logs/epoch_experiments/epoch_experiment_statistics.csv')

# Find best epoch for AUC
best = stats.loc[stats['AUC_mean'].idxmax()]
print(f"Best: {best['epoch_value']} epochs")
print(f"AUC: {best['AUC_mean']:.4f} ± {best['AUC_std']:.4f}")

# Plot
plt.errorbar(stats['epoch_value'], stats['AUC_mean'], 
             yerr=stats['AUC_std'], marker='o')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('AUC vs Epochs')
plt.show()
```

## ⚠️ Important Notes

### Time Estimates
Based on 5 epochs taking ~45 minutes:
- 1 epoch: 9 min × 10 = **1.5 hours**
- 5 epochs: 45 min × 10 = **7.5 hours**
- 10 epochs: 90 min × 10 = **15 hours**
- 50 epochs: 450 min × 10 = **75 hours**
- 100 epochs: 900 min × 10 = **150 hours (6+ days)**

**Total for all configurations: 50-100+ hours**

### Resource Usage
- Uses GPU if available (faster)
- Memory: ~8-16GB RAM recommended
- Disk: ~5-10GB for logs and results

### Safety Features
- Auto-saves after each run
- Backs up config.py
- Results preserved even if interrupted
- Can resume by removing completed epochs from `EPOCHS_TO_TEST`

## 🆘 Troubleshooting

### Script Won't Run
```bash
# Make executable
chmod +x run_epoch_experiment.sh

# Run with python directly
python epoch_experiment.py
```

### Out of Memory
- Reduce `BATCH_SIZE` in config.py
- Close other applications
- Use a machine with more RAM

### Takes Too Long
- Run on GPU machine
- Reduce `RUNS_PER_EPOCH` to 3-5
- Test fewer epoch values
- Use quick test first!

### Config Not Restored
```bash
# Manually restore
cp config.py.backup config.py
```

### Check for Errors
```bash
# View detailed log
less logs/epoch_experiments/epoch_experiment_detailed_log.txt

# Search for errors
grep -i "error" logs/epoch_experiments/epoch_experiment_detailed_log.txt
```

## 📞 What Each File Does

| File | Purpose | When to Use |
|------|---------|-------------|
| `quick_test_experiment.py` | Fast test (6 runs) | **Use first!** Verify setup |
| `epoch_experiment.py` | Full experiment (110 runs) | After test passes |
| `run_epoch_experiment.sh` | Easy launcher with backup | Easiest way to run |
| `visualize_epoch_results.py` | Create plots | After experiment completes |
| `EPOCH_EXPERIMENT_README.md` | Detailed documentation | Reference guide |
| `EXPERIMENT_GUIDE.md` | This file | Quick start guide |

## ✅ Recommended Workflow

1. **Verify Setup** (5 min)
   ```bash
   python -c "import torch; print(torch.__version__)"
   python main.py  # Run once manually to verify
   ```

2. **Quick Test** (30-60 min)
   ```bash
   python quick_test_experiment.py
   ```

3. **Review Test Results**
   - Check `logs/epoch_experiments_test/test_results.csv`
   - Verify metrics look reasonable

4. **Run Full Experiment** (Days)
   ```bash
   nohup ./run_epoch_experiment.sh > experiment.out 2>&1 &
   ```

5. **Monitor Progress** (Throughout)
   ```bash
   tail -f logs/epoch_experiments/epoch_experiment_detailed_log.txt
   ```

6. **Visualize Results** (After completion)
   ```bash
   python visualize_epoch_results.py
   ```

7. **Update Config** (After analysis)
   - Set `EPOCHS` to optimal value found
   - Run final validation

## 🎯 Expected Outcomes

You'll learn:
- ✅ Optimal number of epochs for your model
- ✅ Trade-off between performance and training time
- ✅ Variance/stability at different epoch values
- ✅ Diminishing returns point
- ✅ Most efficient configuration

## 📚 Additional Resources

- Main training code: `main.py`
- Configuration: `config.py`
- Model architecture: See imports in `main.py`
- Training guide: `JOINT_END_TO_END_TRAINING_GUIDE.md`

---

**Good luck with your experiments! 🚀**

*Questions? Check the detailed log files or the main README.*

