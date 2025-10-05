#!/bin/bash
# Epoch Experiment Launcher
# This script runs the epoch experiment with proper environment setup

echo "=================================================="
echo "    EPOCH EXPERIMENT LAUNCHER"
echo "=================================================="
echo ""
echo "Testing epochs: 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100"
echo "Runs per epoch: 10"
echo ""
echo "⚠️  WARNING: This will take a VERY long time!"
echo "   Estimated total time: Several hours to days"
echo ""
echo "Results will be saved to: logs/epoch_experiments/"
echo ""

# Backup original config.py
echo "Creating backup of config.py..."
cp config.py config.py.backup
echo "✓ Backup created: config.py.backup"
echo ""

# Run the experiment
echo "Starting experiment..."
python epoch_experiment.py

# Restore original config
echo ""
echo "Restoring original config.py..."
cp config.py.backup config.py
echo "✓ Config restored"

echo ""
echo "=================================================="
echo "    EXPERIMENT COMPLETE"
echo "=================================================="
echo ""
echo "Check results in: logs/epoch_experiments/"

