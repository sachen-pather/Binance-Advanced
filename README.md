Quantum Edge: A Professional Quantitative Trading Platform
![alt text](https://img.shields.io/badge/python-3.11-blue.svg)

![alt text](https://img.shields.io/badge/license-MIT-green.svg)

![alt text](https://img.shields.io/badge/status-in%20development-orange.svg)
Overview
This is not just a trading bot; it is a comprehensive, institutional-grade quantitative trading platform designed for the cryptocurrency markets. Built from the ground up in Python, this system leverages a modular architecture to integrate sophisticated data analysis, a multi-model machine learning core, dynamic risk management, and persistent performance analytics.
The platform is designed to be a complete end-to-end solution for developing, backtesting, and deploying quantitative trading strategies. Its core function is to autonomously analyze market data, identify high-probability trading opportunities using an ensemble of ML models, manage positions with dynamic exit strategies, and continuously track its own performance to facilitate strategy refinement.
Core Features
Modular, Decoupled Architecture: Each component of the system (data fetching, feature engineering, ML, strategy, risk, execution) is a separate, high-functioning module, allowing for easy testing, maintenance, and upgrades.
Multi-Strategy Alpha Generation: The system doesn't rely on a single signal. It analyzes the market through multiple strategic lenses (Trend Following, Mean Reversion, Breakout, etc.) and uses a confluence-based approach to identify robust trading opportunities.
GPU-Accelerated Machine Learning Core: Utilizes a powerful StackingClassifier that ensembles XGBoost, LightGBM, a Keras/TensorFlow Neural Network, and a Random Forest. The entire training and hyperparameter optimization process (via Optuna) is accelerated on NVIDIA GPUs using CUDA.
Sophisticated Risk & Position Management: Goes beyond simple stop-losses. Features dynamic position sizing based on market regime and volatility, and an EnhancedPositionManager that handles trailing stops, partial profit-taking, and detailed excursion tracking.
High-Fidelity Backtesting Framework: Includes a dedicated backtesting engine to validate strategies against historical data, simulating realistic conditions like fees, slippage, and latency.
Persistent Performance Analytics: All trades and performance snapshots are logged to a local SQLite database, allowing for in-depth offline analysis, performance attribution by strategy, and long-term tracking.
Robust State Management: The bot saves its state (open positions, model files, strategy parameters) periodically, allowing for graceful shutdowns and recovery from restarts without losing its operational context.
System Architecture
The platform is composed of several key modules that work in concert:
TradingBotController (enhanced_main_execution.py): The central orchestrator. It manages the main trading loop, handles initialization and graceful shutdowns, and coordinates all other modules.
DataFetcher: The sole interface to the Binance API. Responsible for fetching historical klines, account information, and tradable symbols with built-in retry logic.
EnhancedTechnicalIndicators: A comprehensive feature factory that calculates over 100 technical indicators from raw price/volume data.
AdvancedMarketAnalyzer: The primary "alpha" engine. It analyzes market structure, detects the current market regime, and applies multiple trading strategies to generate structured TradingOpportunity objects.
AdvancedMLEngine: The machine learning brain. It handles feature engineering, creating labels, training the ensemble model on the GPU, and making live predictions to confirm or reject trading signals.
AdvancedRiskManager: A critical safety layer. It calculates optimal position sizes based on equity, volatility, and existing portfolio risk. It also enforces global risk limits like maximum drawdown.
EnhancedPositionManager: Manages the lifecycle of every trade, from entry to exit. It tracks P&L in real-time and executes dynamic exit logic (e.g., trailing stops).
ComprehensiveAnalytics: The system's memory. It records every trade and performance snapshot to a persistent SQLite database for analysis and reporting.
Technology Stack
Language: Python 3.11
Core Libraries: Pandas, NumPy, Scikit-learn
Machine Learning: XGBoost, LightGBM, TensorFlow 2.x (via Keras), Optuna, Scikeras
API / Connectivity: python-binance
Environment: Conda for environment management
Hardware Acceleration: NVIDIA CUDA
Setup and Installation
Prerequisites
A Linux environment (tested on Ubuntu 22.04+ via WSL2)
Git
Miniconda or Anaconda
NVIDIA GPU with appropriate drivers and CUDA Toolkit installed and configured for WSL2.
1. Clone the Repository
Generated bash
git clone <your-repo-url>
cd binance-advanced
Use code with caution.
Bash
2. Create the Conda Environment
A conda environment is required to manage the complex dependencies.
Generated bash
conda create --name tradingbot python=3.11
conda activate tradingbot
Use code with caution.
Bash
3. Install Dependencies
Install all required Python packages using the comprehensive pip command:
Generated bash
pip install pandas numpy scikit-learn optuna xgboost lightgbm tensorflow[and-cuda] scikeras matplotlib seaborn python-dotenv python-binance ta
Use code with caution.
Bash
(Note: If you have issues with LightGBM GPU support, you may need to compile it from source.)
4. Configure API Keys
The bot loads API keys from a .env file for security. Create this file in the main project root (binance-advanced/).
Generated bash
touch .env
Use code with caution.
Bash
Open the .env file and add your Binance API keys:
Generated code
BINANCE_API_KEY="YOUR_API_KEY_HERE"
BINANCE_SECRET_KEY="YOUR_SECRET_KEY_HERE"
Use code with caution.
Usage
All commands should be run from within the activated tradingbot environment.
Forcing a Full Model Retrain (Recommended First Step)
To ensure a clean start, delete all old state and retrain the model from scratch, use the provided shell script from the main project directory:
Generated bash
./retrain_and_run.sh
Use code with caution.
Bash
This process will take 20-40 minutes depending on your hardware as it performs a full hyperparameter search and training.
Running in Paper Trading Mode
After a model has been trained, you can start the bot in paper trading mode. It will load the existing model and begin its live analysis loop without executing real trades.
Generated bash
# Navigate to the source directory
cd src/

# Run the main script
python enhanced_main_execution.py --mode paper
Use code with caution.
Bash
Running in Live Trading Mode
WARNING: Use with extreme caution. This will execute real trades with real capital.
Generated bash
cd src/
python enhanced_main_execution.py --mode live
Use code with caution.
Bash
Running the Backtester
To validate the strategy against historical data:
Generated bash
cd src/
python enhanced_main_execution.py --mode backtest
Use code with caution.
Bash
Project Structure
Generated code
binance-advanced/
├── src/                          # Main source code
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
├── logs/                         # Log files will be created here
├── data/                         # For any saved data/backtests
├── retrain_and_run.sh            # Utility script for clean restarts
├── gpu_test.py                   # Script to verify GPU setup
└── README.md                     # This file
Use code with caution.
Roadmap & Future Enhancements
This platform is a strong foundation for further research and development.
C++ Acceleration: Re-implement performance-critical paths (e.g., the prediction engine, event loop) in C++ with Python bindings (pybind11) for lower latency.
FPGA Integration: For higher-frequency strategies, explore using an FPGA for market data filtering and parsing to further reduce latency.
Add Sell-Side Logic: Implement strategies for shorting to capitalize on both upward and downward market moves.
Web Dashboard/API: Build a simple web interface using FastAPI or Flask to monitor the bot's status, performance, and control its operation (pause, resume, stop) remotely.
Disclaimer
This is a highly experimental project. Trading financial markets involves substantial risk of loss and is not suitable for every investor. The author and contributors are not responsible for any financial losses incurred through the use of this software. Do not run this bot with real money unless you fully understand the code and the risks involved.
