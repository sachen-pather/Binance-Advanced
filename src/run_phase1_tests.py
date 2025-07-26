# run_phase1_tests.py (Parallel Version)

import logging
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Import your necessary components
from enhanced_strategy_integration import EnhancedTradingStrategy, StrategyConfiguration
from data_fetcher import DataFetcher
from utils import setup_logging
# Import the AdvancedBacktester class directly, not the sweep function
from advanced_backtesting_framework import AdvancedBacktester

# Configure logging
setup_logging('logs/phase1_parallel_backtest.log', logging.INFO)
logger = logging.getLogger("BinanceTrading.Phase1TestRunner")

def run_single_backtest_configuration(params):
    """
    This function is designed to be run in a separate, isolated process.
    It takes one set of parameters, runs one complete backtest, and returns the key results.
    """
    # Unpack the parameters for this specific run
    holding_period, threshold, strategy_config_dict, start_date, end_date, symbols, timeframe = params
    
    # Each process needs its own instances of the main objects to avoid conflicts.
    data_fetcher = DataFetcher()
    
    # Recreate the StrategyConfiguration object from the dictionary passed in
    # This ensures it's correctly initialized in the new process.
    strategy_config = StrategyConfiguration(**strategy_config_dict)
    strategy = EnhancedTradingStrategy(strategy_config, paper_trade=True)
    
    # Create the backtester for this specific run
    backtester = AdvancedBacktester(initial_capital=10000, phase1_mode=True)
    backtester.set_phase1_parameter('max_holding_hours', holding_period)
    backtester.set_phase1_parameter('signal_confidence_threshold', threshold)

    print(f"--- [PID:{os.getpid()}] STARTING Test: {holding_period}H holding, {threshold} confidence ---")
    
    try:
        # Run the backtest with the specific configuration
        result = backtester.run_backtest(strategy, data_fetcher, start_date, end_date, symbols, timeframe)
        
        # We only need to return the key metrics, not the huge list of trades, to save memory.
        metrics = {
            'holding_period': holding_period,
            'confidence_threshold': threshold,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'sharpe_ratio': result.sharpe_ratio,
            'total_return': result.total_return,
            'max_drawdown': result.max_drawdown,
            'total_trades': result.total_trades
        }
        print(f"--- [PID:{os.getpid()}] FINISHED Test: {holding_period}H holding, {threshold} confidence | Sharpe: {result.sharpe_ratio:.2f} ---")
        return f"{holding_period}H_{threshold}", metrics
    except Exception as e:
        print(f"--- [PID:{os.getpid()}] FAILED Test: {holding_period}H holding, {threshold} confidence | Error: {e} ---")
        return f"{holding_period}H_{threshold}", None

def main():
    logger.info("="*80)
    logger.info("STARTING PARALLEL PHASE 1: ALPHA CORE VALIDATION PARAMETER SWEEP")
    logger.info("="*80)

    # --- Configuration for the Test ---
    # We pass this dictionary to each process to recreate the config object.
    strategy_config_dict = {
        'strategy_weights': {
            'trend_following': 0.3, 'mean_reversion': 0.25, 'breakout': 0.2, 
            'momentum': 0.15, 'accumulation': 0.1
        },
        'regime_strategy_multipliers': {},
        # Add any other required fields by StrategyConfiguration with default values
        'max_concurrent_analysis': 10, 'analysis_timeout_seconds': 30,
        'min_opportunity_score': 0.6, 'max_daily_trades': 20,
        'portfolio_heat_limit': 0.8, 'correlation_limit': 0.7,
        'max_drawdown_limit': 0.15, 'ml_confidence_threshold': 0.6,
        'model_retrain_threshold': 0.55, 'ensemble_min_agreement': 0.7,
    }

    start_date = "2022-01-01"
    end_date = "2023-12-31"
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    timeframe = '1h'

    # --- Define the parameter grid for all 16 tests ---
    holding_periods = [24, 48, 72, 168]
    confidence_thresholds = [0.55, 0.60, 0.65, 0.70]
    
    tasks = []
    for holding_period in holding_periods:
        for threshold in confidence_thresholds:
            tasks.append((holding_period, threshold, strategy_config_dict, start_date, end_date, symbols, timeframe))

    # --- Run the Experiment in Parallel ---
    all_results = {}
    
    # Set max_workers to a specific number to control RAM. 4 or 6 is a safe start.
    # If your PC has 16+ GB of RAM, you can try 8.
    num_workers = 4 
    print(f"Starting parallel execution with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks to the pool and create a dictionary to track them.
        futures = {executor.submit(run_single_backtest_configuration, task): task for task in tasks}
        
        # as_completed waits for any future to finish, so we can process results as they come in.
        for future in as_completed(futures):
            try:
                config_key, metrics = future.result()
                if metrics:
                    all_results[config_key] = metrics
            except Exception as e:
                print(f"A task generated a critical exception: {e}")

    # --- Print the Final Summary Table ---
    print("\n" + "="*80)
    print("PARALLEL PHASE 1 PARAMETER SWEEP RESULTS")
    print("="*80)
    
    # Sort results by the most important metric: Sharpe Ratio
    sorted_results = sorted(all_results.items(), key=lambda item: item[1]['sharpe_ratio'], reverse=True)

    print(f"{'Config':<15} {'Win Rate':<10} {'PF':<8} {'Sharpe':<8} {'Return':<8} {'DD':<8} {'Trades':<8}")
    print("-"*80)
    
    for config_key, metrics in sorted_results:
        print(f"{config_key:<15} {metrics['win_rate']:<10.1%} {metrics['profit_factor']:<8.2f} "
              f"{metrics['sharpe_ratio']:<8.2f} {metrics['total_return']:<8.1%} "
              f"{metrics['max_drawdown']:<8.1%} {metrics['total_trades']:<8}")
    
    # Find and log the best configuration
    if sorted_results:
        best_config_key, best_result = sorted_results[0]
        logger.info("\n" + "="*80)
        logger.info("BEST PERFORMING PHASE 1 CONFIGURATION")
        logger.info("="*80)
        logger.info(f"Configuration Key: {best_config_key}")
        logger.info(f"Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
        logger.info(f"Profit Factor: {best_result['profit_factor']:.2f}")
        logger.info(f"Win Rate: {best_result['win_rate']:.1%}")
        logger.info(f"Total Return: {best_result['total_return']:.1%}")
        logger.info("="*80)

if __name__ == "__main__":
    # This ensures the parallel code only runs when the script is executed directly
    main()