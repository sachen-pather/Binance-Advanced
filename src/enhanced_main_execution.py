"""
Enhanced Main Execution Script for Advanced Cryptocurrency Trading Bot
Integrates all enhanced components with comprehensive monitoring and control.
"""

import asyncio
import logging
import os
import sys
import time
import json
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import enhanced components
from enhanced_strategy_integration import EnhancedTradingStrategy, DEFAULT_STRATEGY_CONFIG
from advanced_backtesting_framework import run_strategy_backtest
from data_fetcher import DataFetcher
from utils import setup_logging
from config import ENV_CONFIG, BINANCE_CONFIG

# Configure logging
setup_logging('logs/enhanced_trading.log', logging.INFO)
logger = logging.getLogger("BinanceTrading.Main")


class TradingBotController:
    """Main controller for the enhanced trading bot"""
    
    def __init__(self, paper_trade: bool = True):
        self.paper_trade = paper_trade
        self.running = False
        self.strategy = None
        self.data_fetcher = None
        self.supported_symbols = []
        
        # Control flags
        self.pause_trading = False
        self.emergency_stop = False
        self.shutdown_requested = False
        
        # Performance tracking
        self.start_time = None
        self.loop_count = 0
        self.last_equity_update = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Trading Bot Controller initialized (Paper Trade: {paper_trade})")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
        if self.strategy:
            self.strategy.emergency_stop("Signal received")
    
    def initialize_components(self):
        """Initialize all trading components"""
        
        try:
            logger.info("Initializing trading components...")
            
            # Initialize data fetcher
            self.data_fetcher = DataFetcher()
            
            # Get supported symbols
            logger.info("Fetching supported symbols...")
            self.supported_symbols = self.data_fetcher.get_tradable_symbols(volume_threshold=500000)
            logger.info(f"Found {len(self.supported_symbols)} supported symbols")
            
            # Initialize strategy with configuration
            logger.info("Initializing enhanced trading strategy...")
            self.strategy = EnhancedTradingStrategy(DEFAULT_STRATEGY_CONFIG, self.paper_trade)
            
            # Load previous state if available
            state_loaded = self.strategy.load_strategy_state()
            if state_loaded:
                logger.info("Previous strategy state loaded successfully")
            else:
                logger.info("Starting with fresh strategy state")
            
            # Validate configuration
            self._validate_configuration()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def _validate_configuration(self):
        """Validate configuration and API connectivity"""
        
        try:
            # Test API connectivity
            logger.info("Testing API connectivity...")
            
            # Test basic data fetch
            test_data = self.data_fetcher.get_historical_data('BTCUSDT', '1h', '1 day')
            if test_data is None or len(test_data) == 0:
                raise ValueError("Failed to fetch test data from API")
            
            # Test account access (if not paper trading)
            if not self.paper_trade:
                logger.info("Testing account access...")
                equity = self.data_fetcher.get_account_equity(paper_trade=False)
                if equity <= 0:
                    raise ValueError("Failed to get account equity")
                logger.info(f"Account equity: ${equity:.2f}")
            
            # Validate strategy configuration
            config = self.strategy.config
            if abs(sum(config.strategy_weights.values()) - 1.0) > 0.01:
                raise ValueError("Strategy weights must sum to 1.0")
            
            logger.info("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    async def run_trading_loop(self):
        """Main trading loop with enhanced error handling"""
        
        try:
            self.running = True
            self.start_time = datetime.now()
            
            logger.info("Starting enhanced trading loop...")
            
            # Initial ML model training
            await self._initial_model_training()
            
            # Main execution loop
            while self.running and not self.shutdown_requested:
                try:
                    loop_start = time.time()
                    self.loop_count += 1
                    
                    # Check for pause
                    if self.pause_trading:
                        logger.info("Trading paused, waiting...")
                        await asyncio.sleep(30)
                        continue
                    
                    # Check emergency stop
                    if self.emergency_stop or self.strategy.strategy_state.get('emergency_stop', False):
                        logger.warning("Emergency stop active, halting trading")
                        break
                    
                    # Log loop start
                    logger.info(f"=== Trading Loop {self.loop_count} Started ===")
                    
                    # Run comprehensive analysis
                    analysis_result = await self.strategy.run_comprehensive_analysis(
                        self.data_fetcher, self.supported_symbols
                    )
                    
                    # Process results
                    await self._process_analysis_results(analysis_result)
                    
                    # Update performance tracking
                    self._update_performance_tracking(analysis_result)
                    
                    # Save state periodically
                    if self.loop_count % 10 == 0:
                        self.strategy.save_strategy_state()
                        logger.info("Strategy state saved")
                    
                    # Calculate loop duration
                    loop_duration = time.time() - loop_start
                    logger.info(f"Loop {self.loop_count} completed in {loop_duration:.2f}s")
                    
                    # Dynamic sleep based on market conditions
                    sleep_time = self._calculate_sleep_time(analysis_result)
                    logger.info(f"Sleeping for {sleep_time}s before next iteration")
                    
                    await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in trading loop iteration {self.loop_count}: {e}")
                    
                    # Increment error count and handle gracefully
                    await self._handle_loop_error(e)
                    
                    # Continue after error handling
                    await asyncio.sleep(60)  # Wait longer after error
            
            logger.info("Trading loop stopped")
            
        except Exception as e:
            logger.critical(f"Critical error in trading loop: {e}")
            self.emergency_stop = True
            raise
        finally:
            self.running = False
            await self._cleanup()
    
    async def _initial_model_training(self):
        """Perform initial ML model training"""
        
        try:
            logger.info("Starting initial ML model training check...")

            # ### START OF FIX ###
            # More robust check: The model must exist AND be fitted (trained).
            # We check for the 'estimators_' attribute which is created after .fit()
            model_is_usable = (
                self.strategy.ml_engine.ensemble_model is not None and
                hasattr(self.strategy.ml_engine.ensemble_model, 'estimators_')
            )

            if model_is_usable:
                logger.info("Usable ML model already loaded, skipping initial training.")
                return
            
            # If we are here, the model needs to be trained.
            if self.strategy.ml_engine.ensemble_model is not None:
                logger.warning("Loaded ML model is not fitted. Forcing retrain.")
            else:
                logger.info("No ML model found. Starting initial training.")
            # ### END OF FIX ###
            
            # Train ensemble model
            # NOTE: Using a broader set of symbols for more robust training
            symbols_for_training = [s['symbol'] for s in self.supported_symbols[:10]]
            if not symbols_for_training:
                symbols_for_training = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT'] # Fallback

            success, accuracy = await asyncio.to_thread(
                self.strategy.ml_engine.train_ensemble_model,
                self.data_fetcher,
                self.strategy.indicator_calculator,
                symbols=symbols_for_training,
                optimize_hyperparams=True
            )
            
            if success:
                logger.info(f"Initial ML model training completed successfully (Accuracy: {accuracy:.3f})")
            else:
                logger.warning("Initial ML model training failed, using default signals")
            
        except Exception as e:
            logger.error(f"Error in initial model training: {e}")
    
    async def _process_analysis_results(self, analysis_result: Dict):
        """Process and log analysis results"""
        
        try:
            if 'error' in analysis_result:
                logger.error(f"Analysis error: {analysis_result['error']}")
                return
            
            # Extract key metrics
            market_regime = analysis_result.get('market_regime', 'UNKNOWN')
            opportunities = analysis_result.get('opportunities_found', 0)
            trades_executed = analysis_result.get('trades_executed', 0)
            current_equity = analysis_result.get('current_equity', 0)
            
            # Log summary
            logger.info(f"Market Regime: {market_regime}")
            logger.info(f"Opportunities Found: {opportunities}")
            logger.info(f"Trades Executed: {trades_executed}")
            logger.info(f"Current Equity: ${current_equity:.2f}")
            
            # Log recommendations
            recommendations = analysis_result.get('recommendations', [])
            if recommendations:
                logger.info("Strategy Recommendations:")
                for rec in recommendations:
                    logger.info(f"  - {rec}")
            
            # Log exit signals
            exit_signals = analysis_result.get('exit_signals', [])
            if exit_signals:
                logger.info("Exit Signals:")
                for signal in exit_signals:
                    logger.info(f"  - {signal}")
            
            # Check for warnings
            portfolio_risk = analysis_result.get('portfolio_risk', {})
            risk_warnings = portfolio_risk.get('risk_warnings', [])
            if risk_warnings:
                logger.warning("Risk Warnings:")
                for warning in risk_warnings:
                    logger.warning(f"  - {warning}")
            
        except Exception as e:
            logger.error(f"Error processing analysis results: {e}")
    
    def _update_performance_tracking(self, analysis_result: Dict):
        """Update performance tracking metrics"""
        
        try:
            current_equity = analysis_result.get('current_equity', 0)
            
            # Update equity tracking
            if self.last_equity_update is None:
                self.last_equity_update = current_equity
            
            # Calculate performance since start
            if current_equity > 0 and self.strategy.analytics.initial_equity > 0:
                total_return = (current_equity - self.strategy.analytics.initial_equity) / self.strategy.analytics.initial_equity
                
                # Log performance every 10 loops
                if self.loop_count % 10 == 0:
                    uptime = datetime.now() - self.start_time
                    logger.info(f"Performance Update (Loop {self.loop_count}):")
                    logger.info(f"  Total Return: {total_return:.2%}")
                    logger.info(f"  Current Equity: ${current_equity:.2f}")
                    logger.info(f"  Uptime: {uptime}")
            
            self.last_equity_update = current_equity
            
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def _calculate_sleep_time(self, analysis_result: Dict) -> int:
        """Calculate dynamic sleep time based on market conditions"""
        
        try:
            # Base sleep time
            base_sleep = 90  # 1.5 minutes
            
            # Adjust based on market regime
            market_regime = analysis_result.get('market_regime', 'UNKNOWN')
            
            if market_regime == 'HIGH_VOLATILITY':
                sleep_multiplier = 0.5  # More frequent updates
            elif market_regime == 'LOW_VOLATILITY':
                sleep_multiplier = 1.5  # Less frequent updates
            elif market_regime in ['BULL_TREND', 'BEAR_TREND']:
                sleep_multiplier = 0.8  # Slightly more frequent
            else:
                sleep_multiplier = 1.0
            
            # Adjust based on number of opportunities
            opportunities = analysis_result.get('opportunities_found', 0)
            if opportunities > 5:
                sleep_multiplier *= 0.8  # More active market
            elif opportunities == 0:
                sleep_multiplier *= 1.2  # Quiet market
            
            # Adjust based on time of day (simplified)
            hour = datetime.now().hour
            if 22 <= hour or hour <= 6:  # Night hours UTC
                sleep_multiplier *= 1.5  # Less active trading
            
            final_sleep = int(base_sleep * sleep_multiplier)
            return max(30, min(300, final_sleep))  # Clamp between 30s and 5 minutes
            
        except Exception as e:
            logger.error(f"Error calculating sleep time: {e}")
            return 90  # Default
    
    async def _handle_loop_error(self, error: Exception):
        """Handle errors in trading loop"""
        
        try:
            error_type = type(error).__name__
            logger.error(f"Loop error ({error_type}): {error}")
            
            # Increment error count
            if not hasattr(self, 'error_count'):
                self.error_count = 0
            self.error_count += 1
            
            # Check if we should stop due to too many errors
            if self.error_count > 10:
                logger.critical("Too many consecutive errors, stopping trading")
                self.emergency_stop = True
                return
            
            # Reset error count on successful loop
            if self.loop_count > 0 and self.loop_count % 5 == 0:
                self.error_count = max(0, self.error_count - 1)
            
            # Save state on error
            if self.strategy:
                self.strategy.save_strategy_state()
            
        except Exception as e:
            logger.critical(f"Error in error handler: {e}")
    
    async def _cleanup(self):
        """Cleanup resources and save final state"""
        
        try:
            logger.info("Performing cleanup...")
            
            if self.strategy:
                # Save final state
                self.strategy.save_strategy_state()
                
                # Get final status
                final_status = self.strategy.get_comprehensive_status()
                
                # Log final statistics
                logger.info("=== FINAL STATISTICS ===")
                portfolio_status = final_status.get('portfolio_status', {})
                logger.info(f"Final Equity: ${portfolio_status.get('current_equity', 0):.2f}")
                logger.info(f"Total Realized P&L: ${portfolio_status.get('total_realized_pnl', 0):.2f}")
                logger.info(f"Open Positions: {portfolio_status.get('open_positions', 0)}")
                
                execution_metrics = final_status.get('execution_metrics', {})
                logger.info(f"Total Trades Executed: {execution_metrics.get('trades_executed', 0)}")
                logger.info(f"Opportunities Analyzed: {execution_metrics.get('opportunities_analyzed', 0)}")
                logger.info(f"Average Analysis Time: {execution_metrics.get('analysis_time_avg', 0):.2f}s")
            
            # Calculate total uptime
            if self.start_time:
                total_uptime = datetime.now() - self.start_time
                logger.info(f"Total Uptime: {total_uptime}")
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def pause(self):
        """Pause trading (can be resumed)"""
        self.pause_trading = True
        logger.info("Trading paused")
    
    def resume(self):
        """Resume trading"""
        self.pause_trading = False
        logger.info("Trading resumed")
    
    def stop(self):
        """Stop trading gracefully"""
        self.running = False
        logger.info("Graceful stop requested")
    
    def emergency_stop_trading(self, reason: str = "Manual emergency stop"):
        """Emergency stop all trading"""
        self.emergency_stop = True
        if self.strategy:
            self.strategy.emergency_stop(reason)
        logger.critical(f"EMERGENCY STOP: {reason}")


async def run_backtest_mode():
    """Run backtesting mode"""
    
    try:
        logger.info("Starting backtest mode...")
        
        # Initialize data fetcher
        data_fetcher = DataFetcher()
        
        # Define backtest parameters
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']
        
        # Run backtest
        results = run_strategy_backtest(
            DEFAULT_STRATEGY_CONFIG,
            data_fetcher,
            start_date,
            end_date,
            symbols
        )
        
        logger.info(f"Backtest completed successfully")
        logger.info(f"Total Return: {results.total_return:.2%}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in backtest mode: {e}")
        raise


def main():
    """Main entry point"""
    
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Enhanced Cryptocurrency Trading Bot')
        parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                          default='paper', help='Trading mode')
        parser.add_argument('--config', type=str, help='Path to configuration file')
        parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                          default='INFO', help='Logging level')
        
        args = parser.parse_args()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Display startup banner
        print("=" * 60)
        print("ENHANCED CRYPTOCURRENCY TRADING BOT")
        print("=" * 60)
        print(f"Mode: {args.mode.upper()}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log Level: {args.log_level}")
        print("=" * 60)
        
        # Run appropriate mode
        if args.mode == 'backtest':
            # Run backtest
            asyncio.run(run_backtest_mode())
            
        else:
            # Run live/paper trading
            paper_trade = (args.mode == 'paper')
            
            # Create and initialize controller
            controller = TradingBotController(paper_trade=paper_trade)
            
            if not controller.initialize_components():
                logger.error("Failed to initialize components, exiting")
                return 1
            
            # Display final confirmation
            mode_str = "PAPER TRADING" if paper_trade else "LIVE TRADING"
            print(f"\n🚀 Starting {mode_str} mode...")
            print("Press Ctrl+C to stop gracefully")
            print("-" * 40)
            
            # Run trading loop
            asyncio.run(controller.run_trading_loop())
        
        logger.info("Program completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        return 0
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())