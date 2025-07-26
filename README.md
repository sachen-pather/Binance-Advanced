# Quantum Edge: Professional Quantitative Trading Platform

A comprehensive, institutional-grade quantitative trading platform designed for cryptocurrency markets. Built from the ground up in Python with GPU acceleration, this system combines sophisticated data analysis, multi-model machine learning, dynamic risk management, and persistent performance analytics.

## Overview

Quantum Edge is not just a trading bot—it's a complete end-to-end solution for developing, backtesting, and deploying quantitative trading strategies. The platform autonomously analyzes market data, identifies high-probability trading opportunities using an ensemble of ML models, manages positions with dynamic exit strategies, and continuously tracks performance for strategy refinement.

## Core Features

### Modular Architecture
- Decoupled components for easy testing, maintenance, and upgrades
- Each module (data, ML, strategy, risk, execution) operates independently

### Multi-Strategy Alpha Generation
- Analyzes markets through multiple strategic lenses (Trend Following, Mean Reversion, Breakout)
- Confluence-based approach for robust trading opportunity identification
- No single-point-of-failure reliance on individual signals

### GPU-Accelerated Machine Learning
- Powerful StackingClassifier ensemble combining:
  - XGBoost
  - LightGBM
  - Keras/TensorFlow Neural Network
  - Random Forest
- CUDA-accelerated training and hyperparameter optimization via Optuna

### Sophisticated Risk & Position Management
- Dynamic position sizing based on market regime and volatility
- Enhanced position management with trailing stops and partial profit-taking
- Detailed excursion tracking and risk limit enforcement

### High-Fidelity Backtesting
- Dedicated backtesting engine for strategy validation
- Realistic simulation including fees, slippage, and latency
- Historical data analysis capabilities

### Persistent Performance Analytics
- SQLite database logging for all trades and performance snapshots
- In-depth offline analysis and performance attribution
- Long-term tracking and strategy refinement support

### Robust State Management
- Periodic state saving (positions, models, parameters)
- Graceful shutdown and recovery capabilities
- Maintains operational context across restarts

## System Architecture

The platform consists of several key modules working in concert:

- **TradingBotController** (`enhanced_main_execution.py`) - Central orchestrator managing the main trading loop
- **DataFetcher** - Binance API interface with retry logic
- **EnhancedTechnicalIndicators** - Comprehensive feature factory calculating 100+ technical indicators
- **AdvancedMarketAnalyzer** - Primary alpha engine for market structure analysis
- **AdvancedMLEngine** - Machine learning brain for feature engineering and predictions
- **AdvancedRiskManager** - Safety layer for position sizing and risk limits
- **EnhancedPositionManager** - Trade lifecycle management with dynamic exit logic
- **ComprehensiveAnalytics** - Persistent logging and performance tracking

## Technology Stack

- **Language**: Python 3.11
- **Core Libraries**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: XGBoost, LightGBM, TensorFlow 2.x, Optuna, Scikeras
- **API**: python-binance
- **Environment**: Conda
- **Hardware Acceleration**: NVIDIA CUDA

## Prerequisites

- Linux environment (tested on Ubuntu 22.04+ via WSL2)
- Git
- Miniconda or Anaconda
- NVIDIA GPU with CUDA Toolkit installed and configured

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd binance-advanced
```

### 2. Create Conda Environment

```bash
conda create --name tradingbot python=3.11
conda activate tradingbot
```

### 3. Install Dependencies

```bash
pip install pandas numpy scikit-learn optuna xgboost lightgbm tensorflow[and-cuda] scikeras matplotlib seaborn python-dotenv python-binance ta
```

*Note: If you have issues with LightGBM GPU support, you may need to compile it from source.*

### 4. Configure API Keys

Create a `.env` file in the project root:

```bash
touch .env
```

Add your Binance API credentials:

```
BINANCE_API_KEY="YOUR_API_KEY_HERE"
BINANCE_SECRET_KEY="YOUR_SECRET_KEY_HERE"
```

## Usage

All commands should be run from within the activated `tradingbot` environment.

### Initial Setup (Recommended)

Force a full model retrain for a clean start:

```bash
./retrain_and_run.sh
```

*This process takes 20-40 minutes depending on hardware as it performs full hyperparameter search and training.*

### Paper Trading Mode

Start the bot in paper trading mode (no real trades):

```bash
cd src/
python enhanced_main_execution.py --mode paper
```

### Live Trading Mode

**⚠️ WARNING: Use with extreme caution. This executes real trades with real capital.**

```bash
cd src/
python enhanced_main_execution.py --mode live
```

### Backtesting

Validate strategy against historical data:

```bash
cd src/
python enhanced_main_execution.py --mode backtest
```

## Project Structure

```
binance-advanced/
├── src/                                    # Main source code
│   ├── advanced_backtesting_framework.py
│   ├── advanced_market_analysis.py
│   ├── advanced_ml_engine.py
│   ├── advanced_risk_management.py
│   ├── comprehensive_analytics.py
│   ├── config.py
│   ├── data_fetcher.py
│   ├── enhanced_indicators.py
│   ├── enhanced_main_execution.py
│   ├── enhanced_position_manager.py
│   ├── enhanced_strategy_integration.py
│   └── utils.py
├── logs/                                   # Log files
├── data/                                   # Saved data/backtests
├── retrain_and_run.sh                     # Clean restart utility
├── gpu_test.py                            # GPU setup verification
└── README.md
```

## Roadmap

### Future Enhancements

- **C++ Acceleration**: Re-implement performance-critical paths for lower latency
- **FPGA Integration**: Hardware acceleration for high-frequency strategies
- **Sell-Side Logic**: Short-selling strategies for bidirectional trading
- **Web Dashboard**: FastAPI/Flask interface for remote monitoring and control

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

⚠️ **IMPORTANT**: This is a highly experimental project. Trading financial markets involves substantial risk of loss and is not suitable for every investor. The authors and contributors are not responsible for any financial losses incurred through the use of this software. Do not run this bot with real money unless you fully understand the code and the risks involved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
