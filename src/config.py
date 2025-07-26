"""
Configuration module for trading strategy parameters and constants.
--- MODIFIED FOR A HIGH-FREQUENCY / SCALPING STRATEGY ---
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# BINANCE API CONFIGURATION - Load from environment variables
BINANCE_CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY'),
    'api_secret': os.getenv('BINANCE_SECRET_KEY'),
    'testnet': False,  # Set to False for live trading
    'base_url': 'https://api.binance.com'
}

# Validate that API keys are loaded
if not BINANCE_CONFIG['api_key'] or not BINANCE_CONFIG['api_secret']:
    raise ValueError("API keys not found! Make sure your .env file contains BINANCE_API_KEY and BINANCE_SECRET_KEY")

# --- RISK MANAGEMENT PARAMETERS (Calibrated for High-Frequency Trading) ---
# HFT is a game of many small, precise trades. Risk per trade must be minimal.
RISK_PARAMS = {
    # Position Sizing: Small size per trade to allow for many concurrent positions.
    'base_position_size': 0.005,     # 0.5% of equity per trade as a starting point.
    'max_position_size': 0.02,       # Absolute maximum of 2% of equity in any single trade.
    'min_position_size': 0.001,      # 0.1% minimum, for very low-conviction signals.
    
    # Order Constraints: Based on Binance exchange rules.
    'min_order_value_usdt': 6,       # Minimum 6 USDT per order (Binance minimum is often ~$5, this adds a buffer).
    'force_minimum_orders': True,    # Essential to ensure orders are not rejected.
    
    # Trade-Level Risk: TIGHT stops and profit targets are the core of HFT/scalping.
    'stop_loss_pct': 0.005,          # 0.5% stop loss. Scalping cannot afford deep pullbacks.
    'take_profit_pct': 0.007,        # 0.7% take profit. Aim for a >1 Risk/Reward ratio.
    
    # Portfolio-Level Risk: Protects the entire portfolio from black swans or bad market days.
    'max_daily_loss': 0.03,          # Halt trading if equity drops 3% in a day. CRITICAL for HFT.
    'max_open_positions': 30,        # Allow more concurrent positions, as each is small.
    
    # Statistical & Environmental Risk
    'win_rate': 0.55,                # Target a win rate slightly better than a coin flip.
    'win_loss_ratio': 1.4,           # Target average win to be 1.4x the average loss.
    'correlation_threshold': 0.85,   # Avoid stacking risk on highly correlated assets.
    'extreme_volatility_threshold': 0.03, # Trigger circuit breaker on a 3% price move in a short time.

    # Execution Costs: Realistic estimates.
    'maker_fee': 0.001,
    'taker_fee': 0.001,
    'slippage': 0.0005,              # Slippage is critical in HFT.
}

# --- STRATEGY PARAMETERS (Tuned for Fast Signals on Short Timeframes) ---
# Indicators must be sensitive to small, rapid price changes.
STRATEGY_PARAMS = {
    # Timeframes: Focus on very recent data. The bot will primarily use 1m or 5m charts.
    'lookback_periods': {
        'short': '1 hour',
        'medium': '6 hours',
        'long': '24 hours'
    },
    # Indicator Periods: Faster settings to increase sensitivity.
    'ma_periods': {
        'fast': 5,
        'medium': 10,
        'slow': 20
    },
    'rsi_period': 5,                 # Very sensitive RSI.
    'rsi_overbought': 70,            # More standard overbought level.
    'rsi_oversold': 30,              # More standard oversold level for dip-buying.
    'bb_period': 10,                 # Faster Bollinger Bands.
    'bb_std_dev': 2.1,               # Standard deviation for bands.
    'volume_threshold': 1.5,         # Require volume to be 1.5x the recent average for confirmation.
    'atr_period': 10,                # Standard ATR period for volatility measurement.
    'adx_period': 10,                # Faster ADX.
    'adx_threshold': 20,             # Only trade when there is at least some directional trend.
    
    # ML & Trailing Stops
    'ml_features': [
        'rsi', 'macd', 'bb_position', 'atr', 'adx',
        'volume_change', 'price_change', 'ma_cross'
    ],
    'ml_retrain_hours': 4,           # Retrain model more frequently as market dynamics change faster.
    'trailing_stop_factor': {        # TIGHT trailing stops to lock in small profits quickly.
        'BULL_TREND': 0.003,         # 0.3% trailing stop.
        'BEAR_TREND': 0.003,
        'RANGE_CONTRACTION': 0.004,
        'RANGE_EXPANSION': 0.002,
        'UNKNOWN': 0.005
    }
}

# --- CIRCUIT BREAKER CONFIGURATION (ENABLED - CRITICAL FOR HFT) ---
CIRCUIT_BREAKER_CONFIG = {
    'active': True,                  # ENABLED. This is your primary safety net.
    'triggered_time': None,
    'cooldown_minutes': 5,           # Pause for 5 minutes after a circuit breaker event.
    'triggered_symbols': set()
}

# Performance metrics configuration
PERFORMANCE_METRICS_TEMPLATE = {
    'trades': [],
    'daily_pnl': {},
    'win_rate': 0,
    'profit_factor': 0,
    'max_drawdown': 0,
    'sharpe_ratio': 0
}

# Environment configuration
ENV_CONFIG = {
    'testnet_url': 'https://testnet.binance.vision/api',
    'log_file': 'trading_hft.log',       # Use a separate log file for HFT runs.
    'state_file': 'strategy_state_hft.json',
    'model_file': 'trading_model_hft.pkl'
}