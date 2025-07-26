#!/bin/bash

# =================================================================
#  Clean, Retrain, and Run Script for the Enhanced Trading Bot
# =================================================================
# This script ensures a completely fresh start by deleting all
# state and model files before launching the main application,
# thereby forcing a full model retraining session.

echo "--- Starting Clean, Retrain, and Run Process ---"

# --- Step 1: Activate the Conda Environment ---
# We need to source the conda initialization script first,
# then we can use 'conda activate'.
echo "[1/4] Activating 'tradingbot' conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tradingbot
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment. Aborting."
    exit 1
fi
echo "Environment 'tradingbot' activated."
echo

# --- Step 2: Define and Navigate to Project Directory ---
# This makes the script runnable from anywhere.
PROJECT_DIR="/mnt/c/Users/sache/binance-advanced"
SRC_DIR="$PROJECT_DIR/src"

echo "[2/4] Navigating to source directory: $SRC_DIR"
cd "$SRC_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to find source directory. Aborting."
    exit 1
fi
echo "Successfully changed directory."
echo

# --- Step 3: Perform the "Clean Slate" ---
# Deleting all old state files to force a fresh start and retrain.
echo "[3/4] Cleaning old state, model, and log files..."
rm -f enhanced_strategy_state.json position_state.json trading_model.pkl trading.log trading_performance.db
echo "Cleanup complete."
echo

# --- Step 4: Launch the Main Application ---
# The bot will now be forced to retrain before starting.
echo "[4/4] Starting a fresh run with forced model retraining..."
echo "------------------------------------------------------------"
python enhanced_main_execution.py --mode paper

# --- End of Script ---
echo "------------------------------------------------------------"
echo "--- Script finished. ---"
