"""
Advanced Backtesting Framework for Strategy Validation - Phase 1 Modified
Comprehensive backtesting with realistic execution, slippage,
and detailed performance attribution analysis.
PHASE 1 MODIFICATIONS: Frictionless environment, fixed position sizing, signal-based exits only
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
import gc # <--- IMPORT GARBAGE COLLECTOR
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("BinanceTrading.Backtesting")


@dataclass
class BacktestTrade:
    """Individual backtest trade record"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    strategy: str
    confidence: float
    market_regime: str
    holding_period: timedelta
    exit_reason: str
    max_favorable_excursion: float
    max_adverse_excursion: float


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    # Basic metrics
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_95: float
    
    # Advanced metrics
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    strategy_attribution: Dict = field(default_factory=dict)
    regime_performance: Dict = field(default_factory=dict)
    monthly_returns: pd.Series = field(default_factory=pd.Series)


class AdvancedBacktester:
    """Advanced backtesting engine with realistic execution modeling - Phase 1 Modified"""
    
    def __init__(self, initial_capital: float = 10000, phase1_mode: bool = True):
        self.initial_capital = initial_capital
        self.phase1_mode = phase1_mode  # Phase 1: Alpha Core Validation mode
        
        # PHASE 1 PARAMETERS - Easily tunable
        self.phase1_params = {
            'max_holding_hours': 72,        # Maximum holding period in hours (24, 48, 72, 168)
            'position_size_pct': 0.01,      # Fixed 1% position size
            'signal_confidence_threshold': 0.55,  # Minimum signal confidence (0.55, 0.60, 0.65, 0.70)
            'enable_costs': False,          # Disable all costs in Phase 1
            'enable_risk_management': False, # Disable stops/targets in Phase 1
            'max_positions': 10             # Maximum concurrent positions
        }
        
        # Execution parameters (disabled in Phase 1)
        self.execution_params = {
            'commission_rate': 0.001 if not phase1_mode else 0.0,    # 0% commission in Phase 1
            'slippage_rate': 0.0005 if not phase1_mode else 0.0,     # 0% slippage in Phase 1
            'min_order_size': 10,        
            'max_position_size': 0.1,    
            'latency_ms': 100,           
            'partial_fill_probability': 0.05
        }
        
        # Market impact model (disabled in Phase 1)
        self.market_impact = {
            'impact_coefficient': 0.1 if not phase1_mode else 0.0,
            'liquidity_threshold': 1000,
            'volatility_multiplier': 2.0 if not phase1_mode else 1.0
        }
        
        if phase1_mode:
            logger.info("AdvancedBacktester initialized in PHASE 1 MODE - Alpha Core Validation")
            logger.info(f"Phase 1 Parameters: {self.phase1_params}")
        
    def set_phase1_parameter(self, param_name: str, value: Any):
        """Easily modify Phase 1 parameters for testing"""
        if param_name in self.phase1_params:
            old_value = self.phase1_params[param_name]
            self.phase1_params[param_name] = value
            logger.info(f"Phase 1 parameter '{param_name}' changed from {old_value} to {value}")
        else:
            logger.error(f"Unknown Phase 1 parameter: {param_name}")
            
    def run_backtest(self, strategy, data_fetcher, start_date: str, end_date: str,
                    symbols: List[str] = None, timeframe: str = '1h') -> BacktestResults:
        """Run comprehensive backtest - Phase 1 Modified"""
        
        try:
            if self.phase1_mode:
                logger.info("="*60)
                logger.info("PHASE 1: ALPHA CORE VALIDATION & SIGNAL PURITY ANALYSIS")
                logger.info("="*60)
                logger.info(f"Max Holding Period: {self.phase1_params['max_holding_hours']} hours")
                logger.info(f"Position Size: {self.phase1_params['position_size_pct']*100}% fixed")
                logger.info(f"Signal Confidence Threshold: {self.phase1_params['signal_confidence_threshold']}")
                logger.info(f"Costs Enabled: {self.phase1_params['enable_costs']}")
                logger.info(f"Risk Management Enabled: {self.phase1_params['enable_risk_management']}")
                logger.info("="*60)
            
            logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Convert date strings to datetime
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            # Default symbols if not provided
            if symbols is None:
                symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']
            
            # Prepare historical data
            historical_data = self._prepare_historical_data(
                data_fetcher, symbols, start_date, end_date, timeframe
            )
            
            if not historical_data:
                raise ValueError("No historical data available for backtesting")
            
            # Initialize backtest state
            backtest_state = {
                'current_capital': self.initial_capital,
                'positions': {},
                'trades': [],
                'equity_history': [],
                'current_time': start_dt,
                'phase1_stats': {  # Track Phase 1 specific metrics
                    'signals_generated': 0,
                    'signals_above_threshold': 0,
                    'time_exits': 0,
                    'signal_exits': 0
                }
            }
            
            # Get time series
            time_index = self._create_time_index(historical_data, timeframe)
            
            # Run backtest simulation
            logger.info(f"Running simulation over {len(time_index)} time steps")
            
            for i, current_time in enumerate(time_index):
                # Update current time
                backtest_state['current_time'] = current_time
                
                # Get current market data
                current_data = self._get_current_market_data(historical_data, current_time)
                
                if not current_data:
                    continue
                
                # Update existing positions
                self._update_positions(backtest_state, current_data)
                
                # Generate trading signals (every 4th step to reduce computation)
                if i % 4 == 0:
                    signals = self._generate_signals(strategy, current_data, backtest_state)
                    
                    # Execute trades
                    for signal in signals:
                        trade = self._execute_backtest_trade(signal, current_data, backtest_state)
                        # Trade will be recorded when position is closed
                
                # Record equity
                total_equity = self._calculate_total_equity(backtest_state, current_data)
                backtest_state['equity_history'].append({
                    'timestamp': current_time,
                    'equity': total_equity
                })
                
                # Progress reporting
                if i % 1000 == 0:
                    progress = (i / len(time_index)) * 100
                    logger.info(f"Backtest progress: {progress:.1f}%")
               
                if i % 100 == 0:
                    gc.collect()
                
            # Close any remaining positions at the end
            self._close_remaining_positions(backtest_state, historical_data)
            
            # Calculate final results
            results = self._calculate_backtest_results(backtest_state, start_dt, end_dt)
            
            if self.phase1_mode:
                self._print_phase1_summary(backtest_state, results)
            
            logger.info(f"Backtest completed. Final return: {results.total_return:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
 # In advanced_backtesting_framework.py -> class AdvancedBacktester

    def _prepare_historical_data(self, data_fetcher, symbols: List[str], 
                                start_date: str, end_date: str, timeframe: str) -> Dict:
        """Prepare historical data for backtesting from local files."""
        
        try:
            historical_data = {}
            
            for symbol in symbols:
                logger.info(f"Preparing historical data for {symbol} using local source...")
                
                # --- THIS IS THE KEY CHANGE ---
                # Call the new local data function instead of the API one.
                df = data_fetcher.get_historical_data_from_local(
                    symbol, timeframe, start_date, end_date
                )
                # --- END OF CHANGE ---
                
                if df is not None and not df.empty:
                    # In Phase 1, we assume the full file has enough data for indicator lookback.
                    # We just need to check if our selected date range slice is valid.
                    if len(df) > 200: 
                        historical_data[symbol] = df
                        logger.info(f"Loaded {len(df)} data points for {symbol} for the backtest period.")
                    else:
                        logger.warning(f"Insufficient data for {symbol} in the specified date range [{start_date} to {end_date}].")
                else:
                    logger.warning(f"No local data was loaded for {symbol}. It will be skipped in this backtest.")
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error preparing historical data from local files: {e}", exc_info=True)
            return {}
    
    def _calculate_lookback_days(self, start_date: str) -> int:
        """Calculate required lookback days for indicators"""
        return 200  # 200 days should be sufficient for most indicators
    
    def _create_time_index(self, historical_data: Dict, timeframe: str) -> List[datetime]:
        """Create unified time index for backtesting"""
        
        try:
            # Get all timestamps from all symbols
            all_timestamps = set()
            
            for symbol, df in historical_data.items():
                all_timestamps.update(df.index)
            
            # Sort timestamps
            time_index = sorted(list(all_timestamps))
            
            logger.info(f"Created time index with {len(time_index)} timestamps")
            
            return time_index
            
        except Exception as e:
            logger.error(f"Error creating time index: {e}")
            return []
    
    def _get_current_market_data(self, historical_data: Dict, 
                                current_time: datetime) -> Dict:
        """Get market data for current timestamp"""
        
        try:
            current_data = {}
            
            for symbol, df in historical_data.items():
                if current_time in df.index:
                    # Get current bar and recent history
                    current_idx = df.index.get_loc(current_time)
                    
                    if current_idx >= 200:  # Ensure we have enough history
                        recent_data = df.iloc[max(0, current_idx-200):current_idx+1]
                        current_data[symbol] = {
                            'current_bar': df.loc[current_time],
                            'recent_data': recent_data,
                            'price': df.loc[current_time, 'close']
                        }
            
            return current_data
            
        except Exception as e:
            logger.debug(f"Error getting current market data: {e}")
            return {}
    
    def _update_positions(self, backtest_state: Dict, current_data: Dict):
        """Update existing positions with current market data - Phase 1 Modified"""
        
        try:
            positions_to_close = []
            
            for position_id, position in backtest_state['positions'].items():
                symbol = position['symbol']
                
                if symbol not in current_data:
                    continue
                
                current_price = current_data[symbol]['price']
                
                # Update position value
                if position['side'] == 'BUY':
                    current_pnl = (current_price - position['entry_price']) * position['quantity']
                else:  # SELL
                    current_pnl = (position['entry_price'] - current_price) * position['quantity']
                
                position['current_pnl'] = current_pnl
                position['current_price'] = current_price
                
                # Update excursions
                pnl_pct = current_pnl / (position['entry_price'] * position['quantity'])
                if pnl_pct > 0:
                    position['max_favorable_excursion'] = max(
                        position.get('max_favorable_excursion', 0), pnl_pct
                    )
                else:
                    position['max_adverse_excursion'] = min(
                        position.get('max_adverse_excursion', 0), pnl_pct
                    )
                
                # Check exit conditions (Phase 1: only time-based exit)
                exit_reason = self._check_exit_conditions(position, current_data[symbol], backtest_state['current_time'])
                if exit_reason:
                    positions_to_close.append((position_id, exit_reason))
            
            # Close positions that meet exit criteria
            for position_id, exit_reason in positions_to_close:
                self._close_position(position_id, exit_reason, backtest_state, current_data)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _check_exit_conditions(self, position: Dict, market_data: Dict, current_time: datetime) -> Optional[str]:
        """Check if position should be closed - Phase 1 Modified (Time-based exit only)"""
        
        try:
            # PHASE 1: ONLY TIME-BASED EXIT
            if not self.phase1_params['enable_risk_management']:
                # Only check maximum holding period
                entry_time = position['entry_time']
                
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                
                holding_period = current_time - entry_time
                max_holding = timedelta(hours=self.phase1_params['max_holding_hours'])
                
                if holding_period >= max_holding:
                    return 'time_exit'
                
                return None
            
            # Original logic for non-Phase 1 mode
            current_price = market_data['price']
            
            # Stop loss
            if position['side'] == 'BUY' and current_price <= position.get('stop_loss', 0):
                return 'stop_loss'
            elif position['side'] == 'SELL' and current_price >= position.get('stop_loss', float('inf')):
                return 'stop_loss'
            
            # Take profit
            if position['side'] == 'BUY' and current_price >= position.get('take_profit', float('inf')):
                return 'take_profit'
            elif position['side'] == 'SELL' and current_price <= position.get('take_profit', 0):
                return 'take_profit'
            
            # Time-based exit
            entry_time = position['entry_time']
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            
            holding_period = current_time - entry_time
            if holding_period > timedelta(hours=72):
                return 'time_exit'
            
            return None
            
        except Exception as e:
            logger.debug(f"Error checking exit conditions: {e}")
            return None
    
    def _close_position(self, position_id: str, exit_reason: str, 
                       backtest_state: Dict, current_data: Dict):
        """Close a position and record the trade - Phase 1 Modified"""
        
        try:
            position = backtest_state['positions'][position_id]
            symbol = position['symbol']
            current_price = current_data[symbol]['price']
            
            # PHASE 1: NO SLIPPAGE OR FEES
            if self.phase1_params['enable_costs']:
                # Original slippage and fees calculation
                slippage = self._calculate_slippage(position, current_price)
                exit_price = current_price - slippage if position['side'] == 'BUY' else current_price + slippage
                
                # Calculate fees
                entry_fee = position['entry_price'] * position['quantity'] * self.execution_params['commission_rate']
                exit_fee = exit_price * position['quantity'] * self.execution_params['commission_rate']
                total_fees = entry_fee + exit_fee
            else:
                # Phase 1: Zero costs
                slippage = 0.0
                exit_price = current_price  # No slippage
                total_fees = 0.0  # No fees
            
            # Calculate P&L
            if position['side'] == 'BUY':
                gross_pnl = (exit_price - position['entry_price']) * position['quantity']
            else:  # SELL
                gross_pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            # Net P&L (same as gross in Phase 1)
            net_pnl = gross_pnl - total_fees
            
            # Create trade record
            trade = BacktestTrade(
                entry_time=position['entry_time'],
                exit_time=backtest_state['current_time'],
                symbol=symbol,
                side=position['side'],
                entry_price=position['entry_price'],
                exit_price=exit_price,
                quantity=position['quantity'],
                gross_pnl=gross_pnl,
                fees=total_fees,
                slippage=slippage,
                net_pnl=net_pnl,
                strategy=position.get('strategy', 'unknown'),
                confidence=position.get('confidence', 0.5),
                market_regime=position.get('market_regime', 'unknown'),
                holding_period=backtest_state['current_time'] - position['entry_time'],
                exit_reason=exit_reason,
                max_favorable_excursion=position.get('max_favorable_excursion', 0),
                max_adverse_excursion=position.get('max_adverse_excursion', 0)
            )
            
            # Update capital
            backtest_state['current_capital'] += net_pnl
            
            # Record trade
            backtest_state['trades'].append(trade)
            
            # Update Phase 1 stats
            if exit_reason == 'time_exit':
                backtest_state['phase1_stats']['time_exits'] += 1
            else:
                backtest_state['phase1_stats']['signal_exits'] += 1
            
            # Remove position
            del backtest_state['positions'][position_id]
            
            if self.phase1_mode:
                logger.debug(f"Phase 1 trade closed: {symbol} {position['side']} P&L: ${net_pnl:.2f} ({exit_reason})")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _generate_signals(self, strategy, current_data: Dict, backtest_state: Dict) -> List[Dict]:
        """Generate trading signals using the strategy - Phase 1 Modified"""
        
        try:
            signals = []
            
            # Limit number of positions
            if len(backtest_state['positions']) >= self.phase1_params['max_positions']:
                return signals
            
            for symbol, data in current_data.items():
                # Skip if already have position in this symbol
                if any(pos['symbol'] == symbol for pos in backtest_state['positions'].values()):
                    continue
                
                try:
                    # Calculate indicators
                    df_with_indicators = strategy.indicator_calculator.calculate_all_indicators(
                        data['recent_data']
                    )
                    
                    if df_with_indicators is None or len(df_with_indicators) < 50:
                        continue
                    
                    # Generate opportunities
                    opportunities = strategy.market_analyzer.generate_buy_signals(
                        symbol, df_with_indicators, strategy.ml_engine
                    )
                    
                    # Update Phase 1 stats
                    backtest_state['phase1_stats']['signals_generated'] += len(opportunities)
                    
                    # Convert opportunities to signals with Phase 1 confidence threshold
                    for opp in opportunities:
                        if opp.confidence >= self.phase1_params['signal_confidence_threshold']:
                            backtest_state['phase1_stats']['signals_above_threshold'] += 1
                            signals.append({
                                'symbol': symbol,
                                'side': 'BUY',
                                'price': data['price'],
                                'confidence': opp.confidence,
                                'strategy': opp.signal_type,
                                'expected_return': opp.expected_return,
                                'max_risk': opp.max_risk
                            })
                
                except Exception as e:
                    logger.debug(f"Error generating signals for {symbol}: {e}")
                    continue
            
            return signals[:3]  # Limit to 3 signals per iteration
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def _execute_backtest_trade(self, signal: Dict, current_data: Dict, 
                               backtest_state: Dict) -> Optional[BacktestTrade]:
        """Execute a trade in backtest - Phase 1 Modified"""
        
        try:
            symbol = signal['symbol']
            side = signal['side']
            price = signal['price']
            
            # PHASE 1: FIXED 1% POSITION SIZE
            position_size_pct = self.phase1_params['position_size_pct']
            position_value = backtest_state['current_capital'] * position_size_pct
            
            # Check minimum order size
            if position_value < self.execution_params['min_order_size']:
                return None
            
            # Calculate quantity
            quantity = position_value / price
            
            # PHASE 1: NO SLIPPAGE OR FEES
            if self.phase1_params['enable_costs']:
                # Original slippage and fees calculation
                slippage = self._calculate_slippage({'quantity': quantity, 'side': side}, price)
                entry_price = price + slippage if side == 'BUY' else price - slippage
                fees = entry_price * quantity * self.execution_params['commission_rate']
            else:
                # Phase 1: Zero costs
                slippage = 0.0
                entry_price = price  # No slippage
                fees = 0.0  # No fees
            
            # Check if we have enough capital
            total_cost = entry_price * quantity + fees
            if total_cost > backtest_state['current_capital']:
                return None
            
            # PHASE 1: NO STOP LOSS OR TAKE PROFIT
            stop_loss = None
            take_profit = None
            
            if self.phase1_params['enable_risk_management']:
                # Original stop/target logic
                if side == 'BUY':
                    stop_loss = entry_price * 0.97  # 3% stop loss
                    take_profit = entry_price * 1.06  # 6% take profit
                else:  # SELL
                    stop_loss = entry_price * 1.03
                    take_profit = entry_price * 0.94
            
            # Create position
            position_id = f"{symbol}_{side}_{int(backtest_state['current_time'].timestamp())}"
            position = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'quantity': quantity,
                'entry_time': backtest_state['current_time'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': signal.get('strategy', 'unknown'),
                'confidence': signal.get('confidence', 0.5),
                'max_favorable_excursion': 0,
                'max_adverse_excursion': 0
            }
            
            # Add position to backtest state
            backtest_state['positions'][position_id] = position
            
            # Update capital
            backtest_state['current_capital'] -= total_cost
            
            if self.phase1_mode:
                logger.debug(f"Phase 1 trade opened: {symbol} {side} @ ${entry_price:.4f} (Confidence: {signal.get('confidence', 0.5):.3f})")
            
            return None  # We'll create the trade record when position is closed
            
        except Exception as e:
            logger.error(f"Error executing backtest trade: {e}")
            return None
    
    def _calculate_slippage(self, position: Dict, price: float) -> float:
        """Calculate realistic slippage"""
        
        try:
            if not self.phase1_params['enable_costs']:
                return 0.0  # Phase 1: No slippage
                
            base_slippage = price * self.execution_params['slippage_rate']
            
            # Increase slippage for larger orders (simplified)
            quantity = position['quantity']
            order_value = quantity * price
            
            if order_value > 10000:  # Large order
                slippage_multiplier = 1.5
            elif order_value > 5000:  # Medium order
                slippage_multiplier = 1.2
            else:  # Small order
                slippage_multiplier = 1.0
            
            return base_slippage * slippage_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return 0.0 if self.phase1_params['enable_costs'] else 0.0
    
    def _calculate_total_equity(self, backtest_state: Dict, current_data: Dict) -> float:
        """Calculate total equity including open positions"""
        
        try:
            total_equity = backtest_state['current_capital']
            
            # Add unrealized P&L from open positions
            for position in backtest_state['positions'].values():
                symbol = position['symbol']
                if symbol in current_data:
                    current_price = current_data[symbol]['price']
                    
                    if position['side'] == 'BUY':
                        unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                    else:  # SELL
                        unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    total_equity += unrealized_pnl
            
            return total_equity
            
        except Exception as e:
            logger.error(f"Error calculating total equity: {e}")
            return backtest_state['current_capital']
    
    def _close_remaining_positions(self, backtest_state: Dict, historical_data: Dict):
        """Close any remaining open positions at the end of backtest"""
        
        try:
            if not backtest_state['positions']:
                return
                
            # Get final prices
            final_data = {}
            for symbol in backtest_state['positions'].keys():
                if symbol in historical_data:
                    df = historical_data[symbol]
                    if len(df) > 0:
                        final_data[symbol] = {
                            'price': df.iloc[-1]['close']
                        }
            
            # Close all remaining positions
            positions_to_close = list(backtest_state['positions'].keys())
            for position_id in positions_to_close:
                position = backtest_state['positions'][position_id]
                if position['symbol'] in final_data:
                    self._close_position(position_id, 'end_of_backtest', backtest_state, final_data)
            
        except Exception as e:
            logger.error(f"Error closing remaining positions: {e}")
    
    def _print_phase1_summary(self, backtest_state: Dict, results: BacktestResults):
        """Print Phase 1 specific summary"""
        
        try:
            stats = backtest_state['phase1_stats']
            
            print("\n" + "="*60)
            print("PHASE 1 ALPHA CORE VALIDATION - SUMMARY")
            print("="*60)
            print(f"Max Holding Period: {self.phase1_params['max_holding_hours']} hours")
            print(f"Signal Confidence Threshold: {self.phase1_params['signal_confidence_threshold']}")
            print(f"Fixed Position Size: {self.phase1_params['position_size_pct']*100}%")
            print("\nSIGNAL STATISTICS:")
            print(f"Total Signals Generated: {stats['signals_generated']}")
            print(f"Signals Above Threshold: {stats['signals_above_threshold']}")
            print(f"Signal Acceptance Rate: {stats['signals_above_threshold']/max(stats['signals_generated'], 1)*100:.1f}%")
            print("\nEXIT REASONS:")
            print(f"Time-based Exits: {stats['time_exits']}")
            print(f"Signal-based Exits: {stats['signal_exits']}")
            print("\nPERFORMANCE METRICS:")
            print(f"Win Rate: {results.win_rate:.1%}")
            print(f"Profit Factor: {results.profit_factor:.2f}")
            print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {results.max_drawdown:.1%}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error printing Phase 1 summary: {e}")
    
    def _calculate_backtest_results(self, backtest_state: Dict, 
                                   start_dt: datetime, end_dt: datetime) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        try:
            trades = backtest_state['trades']
            equity_history = backtest_state['equity_history']
            
            # Basic metrics
            final_capital = backtest_state['current_capital']
            # Add value of any remaining open positions at current prices
            # (simplified - would need final market data)
            
            total_return = (final_capital - self.initial_capital) / self.initial_capital
            days = (end_dt - start_dt).days
            annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            
            # Trade statistics
            if trades:
                winning_trades = [t for t in trades if t.net_pnl > 0]
                losing_trades = [t for t in trades if t.net_pnl <= 0]
                
                win_rate = len(winning_trades) / len(trades)
                avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0
                
                gross_profit = sum(t.net_pnl for t in winning_trades)
                gross_loss = abs(sum(t.net_pnl for t in losing_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            else:
                winning_trades = losing_trades = []
                win_rate = avg_win = avg_loss = profit_factor = 0
            
            # Risk metrics
            if len(equity_history) > 1:
                equity_series = pd.Series([eq['equity'] for eq in equity_history])
                returns = equity_series.pct_change().dropna()
                
                # Drawdown calculation
                peak = equity_series.expanding(min_periods=1).max()
                drawdown = (equity_series - peak) / peak
                max_drawdown = abs(drawdown.min())
                
                # Volatility and Sharpe ratio
                volatility = returns.std() * np.sqrt(252)  # Annualized
                sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0
                
                # Sortino ratio
                downside_returns = returns[returns < 0]
                downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
                sortino_ratio = (annualized_return / downside_std) if downside_std > 0 else 0
                
                # Calmar ratio
                calmar_ratio = (annualized_return / max_drawdown) if max_drawdown > 0 else 0
                
                # VaR
                var_95 = abs(np.percentile(returns, 5)) * final_capital if len(returns) > 0 else 0
                
            else:
                equity_series = pd.Series([self.initial_capital])
                drawdown = pd.Series([0])
                max_drawdown = volatility = sharpe_ratio = sortino_ratio = calmar_ratio = var_95 = 0
            
            # Strategy attribution
            strategy_attribution = {}
            for trade in trades:
                strategy = trade.strategy
                if strategy not in strategy_attribution:
                    strategy_attribution[strategy] = {
                        'trades': 0, 'wins': 0, 'total_pnl': 0, 'avg_pnl': 0
                    }
                
                stats = strategy_attribution[strategy]
                stats['trades'] += 1
                stats['total_pnl'] += trade.net_pnl
                if trade.net_pnl > 0:
                    stats['wins'] += 1
            
            # Calculate derived metrics for strategies
            for strategy, stats in strategy_attribution.items():
                stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            # Market regime performance
            regime_performance = {}
            for trade in trades:
                regime = trade.market_regime
                if regime not in regime_performance:
                    regime_performance[regime] = {
                        'trades': 0, 'wins': 0, 'total_pnl': 0, 'avg_pnl': 0
                    }
                
                stats = regime_performance[regime]
                stats['trades'] += 1
                stats['total_pnl'] += trade.net_pnl
                if trade.net_pnl > 0:
                    stats['wins'] += 1
            
            # Calculate derived metrics for regimes
            for regime, stats in regime_performance.items():
                stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            # Monthly returns
            if len(equity_history) > 30:
                equity_df = pd.DataFrame(equity_history)
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                equity_df.set_index('timestamp', inplace=True)
                monthly_returns = equity_df['equity'].resample('M').last().pct_change().dropna()
            else:
                monthly_returns = pd.Series()
            
            return BacktestResults(
                start_date=start_dt,
                end_date=end_dt,
                initial_capital=self.initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                annualized_return=annualized_return,
                total_trades=len(trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                var_95=var_95,
                trades=trades,
                equity_curve=equity_series,
                drawdown_series=drawdown,
                strategy_attribution=strategy_attribution,
                regime_performance=regime_performance,
                monthly_returns=monthly_returns
            )
            
        except Exception as e:
            logger.error(f"Error calculating backtest results: {e}")
            raise
    
    def generate_backtest_report(self, results: BacktestResults) -> str:
        """Generate comprehensive backtest report"""
        
        try:
            report = []
            report.append("=" * 60)
            if self.phase1_mode:
                report.append("PHASE 1: ALPHA CORE VALIDATION - BACKTEST RESULTS")
            else:
                report.append("BACKTEST RESULTS REPORT")
            report.append("=" * 60)
            
            # Period and basic metrics
            report.append(f"\nBACKTEST PERIOD:")
            report.append(f"Start Date: {results.start_date.strftime('%Y-%m-%d')}")
            report.append(f"End Date: {results.end_date.strftime('%Y-%m-%d')}")
            report.append(f"Duration: {(results.end_date - results.start_date).days} days")
            
            if self.phase1_mode:
                report.append(f"\nPHASE 1 CONFIGURATION:")
                report.append(f"Max Holding Period: {self.phase1_params['max_holding_hours']} hours")
                report.append(f"Signal Confidence Threshold: {self.phase1_params['signal_confidence_threshold']}")
                report.append(f"Fixed Position Size: {self.phase1_params['position_size_pct']*100}%")
                report.append(f"Costs Enabled: {self.phase1_params['enable_costs']}")
                report.append(f"Risk Management Enabled: {self.phase1_params['enable_risk_management']}")
            
            report.append(f"\nPERFORMANCE SUMMARY:")
            report.append(f"Initial Capital: ${results.initial_capital:,.2f}")
            report.append(f"Final Capital: ${results.final_capital:,.2f}")
            report.append(f"Total Return: {results.total_return:.2%}")
            report.append(f"Annualized Return: {results.annualized_return:.2%}")
            
            # Trade statistics
            report.append(f"\nTRADE STATISTICS:")
            report.append(f"Total Trades: {results.total_trades}")
            report.append(f"Winning Trades: {results.winning_trades}")
            report.append(f"Losing Trades: {results.losing_trades}")
            report.append(f"Win Rate: {results.win_rate:.2%}")
            report.append(f"Average Win: ${results.avg_win:.2f}")
            report.append(f"Average Loss: ${results.avg_loss:.2f}")
            report.append(f"Profit Factor: {results.profit_factor:.2f}")
            
            # Risk metrics
            report.append(f"\nRISK METRICS:")
            report.append(f"Maximum Drawdown: {results.max_drawdown:.2%}")
            report.append(f"Volatility (Annualized): {results.volatility:.2%}")
            report.append(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            report.append(f"Sortino Ratio: {results.sortino_ratio:.2f}")
            report.append(f"Calmar Ratio: {results.calmar_ratio:.2f}")
            report.append(f"VaR (95%): ${results.var_95:.2f}")
            
            # Strategy attribution
            if results.strategy_attribution:
                report.append(f"\nSTRATEGY ATTRIBUTION:")
                for strategy, stats in results.strategy_attribution.items():
                    report.append(f"{strategy}:")
                    report.append(f"  Trades: {stats['trades']}")
                    report.append(f"  Win Rate: {stats['win_rate']:.2%}")
                    report.append(f"  Total P&L: ${stats['total_pnl']:.2f}")
                    report.append(f"  Avg P&L: ${stats['avg_pnl']:.2f}")
            
            # Best and worst trades
            if results.trades:
                best_trade = max(results.trades, key=lambda t: t.net_pnl)
                worst_trade = min(results.trades, key=lambda t: t.net_pnl)
                
                report.append(f"\nBEST TRADE:")
                report.append(f"Symbol: {best_trade.symbol}")
                report.append(f"Strategy: {best_trade.strategy}")
                report.append(f"P&L: ${best_trade.net_pnl:.2f}")
                report.append(f"Holding Period: {best_trade.holding_period}")
                report.append(f"Exit Reason: {best_trade.exit_reason}")
                
                report.append(f"\nWORST TRADE:")
                report.append(f"Symbol: {worst_trade.symbol}")
                report.append(f"Strategy: {worst_trade.strategy}")
                report.append(f"P&L: ${worst_trade.net_pnl:.2f}")
                report.append(f"Holding Period: {worst_trade.holding_period}")
                report.append(f"Exit Reason: {worst_trade.exit_reason}")
            
            if self.phase1_mode:
                report.append(f"\nPHASE 1 SUCCESS CRITERIA CHECK:")
                report.append(f"Win Rate ≥54%: {'✓' if results.win_rate >= 0.54 else '✗'} ({results.win_rate:.1%})")
                report.append(f"Profit Factor ≥1.3: {'✓' if results.profit_factor >= 1.3 else '✗'} ({results.profit_factor:.2f})")
                report.append(f"Sharpe Ratio ≥1.0: {'✓' if results.sharpe_ratio >= 1.0 else '✗'} ({results.sharpe_ratio:.2f})")
                
                # Calculate win/loss ratio
                win_loss_ratio = abs(results.avg_win / results.avg_loss) if results.avg_loss != 0 else float('inf')
                report.append(f"Win/Loss Ratio ≥1.2: {'✓' if win_loss_ratio >= 1.2 else '✗'} ({win_loss_ratio:.2f})")
            
            report.append("=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating backtest report: {e}")
            return f"Error generating report: {str(e)}"
    
    def plot_backtest_results(self, results: BacktestResults, save_path: str = None):
        """Plot comprehensive backtest results"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            if self.phase1_mode:
                fig.suptitle('Phase 1: Alpha Core Validation - Backtest Results', fontsize=16)
            else:
                fig.suptitle('Backtest Results Analysis', fontsize=16)
            
            # Equity curve
            axes[0, 0].plot(results.equity_curve.index, results.equity_curve.values)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Equity ($)')
            axes[0, 0].grid(True)
            
            # Drawdown
            axes[0, 1].fill_between(results.drawdown_series.index, 
                                   results.drawdown_series.values * 100, 0, 
                                   alpha=0.3, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].grid(True)
            
            # Monthly returns
            if len(results.monthly_returns) > 0:
                axes[1, 0].bar(range(len(results.monthly_returns)), 
                              results.monthly_returns.values * 100)
                axes[1, 0].set_title('Monthly Returns')
                axes[1, 0].set_ylabel('Return (%)')
                axes[1, 0].grid(True)
            
            # Trade P&L distribution
            if results.trades:
                pnl_values = [trade.net_pnl for trade in results.trades]
                axes[1, 1].hist(pnl_values, bins=30, alpha=0.7)
                axes[1, 1].set_title('Trade P&L Distribution')
                axes[1, 1].set_xlabel('P&L ($)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Backtest plots saved to {save_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting backtest results: {e}")


def run_strategy_backtest(strategy_config, data_fetcher, start_date: str, 
                         end_date: str, symbols: List[str] = None) -> BacktestResults:
    """Convenience function to run a complete strategy backtest"""
    
    try:
        # Create backtester in Phase 1 mode
        backtester = AdvancedBacktester(initial_capital=10000, phase1_mode=True)
        
        # Create strategy instance (mock for backtesting)
        from enhanced_strategy_integration import EnhancedTradingStrategy
        strategy = EnhancedTradingStrategy(strategy_config, paper_trade=True)
        
        # Run backtest
        results = backtester.run_backtest(
            strategy, data_fetcher, start_date, end_date, symbols
        )
        
        # Generate report
        report = backtester.generate_backtest_report(results)
        print(report)
        
        # Plot results
        backtester.plot_backtest_results(results, 'phase1_backtest_results.png')
        
        return results
        
    except Exception as e:
        logger.error(f"Error running strategy backtest: {e}")
        raise


# Phase 1 Testing Helper Functions
def run_phase1_parameter_sweep(strategy_config, data_fetcher, start_date: str, end_date: str):
    """Run Phase 1 parameter sweep for different configurations"""
    
    # Test different holding periods
    holding_periods = [24, 48, 72, 168]  # 1 day, 2 days, 3 days, 1 week
    confidence_thresholds = [0.55, 0.60, 0.65, 0.70]
    
    results = {}
    
    for holding_period in holding_periods:
        for threshold in confidence_thresholds:
            print(f"\nTesting: {holding_period}H holding, {threshold} confidence threshold")
            
            # Create backtester
            backtester = AdvancedBacktester(initial_capital=10000, phase1_mode=True)
            backtester.set_phase1_parameter('max_holding_hours', holding_period)
            backtester.set_phase1_parameter('signal_confidence_threshold', threshold)
            
            # Create strategy
            from enhanced_strategy_integration import EnhancedTradingStrategy
            strategy = EnhancedTradingStrategy(strategy_config, paper_trade=True)
            
            # Run backtest
            result = backtester.run_backtest(strategy, data_fetcher, start_date, end_date)
            
            results[f"{holding_period}H_{threshold}"] = {
                'holding_period': holding_period,
                'confidence_threshold': threshold,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades
            }
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 1 PARAMETER SWEEP RESULTS")
    print("="*80)
    print(f"{'Config':<15} {'Win Rate':<10} {'PF':<8} {'Sharpe':<8} {'Return':<8} {'DD':<8} {'Trades':<8}")
    print("-"*80)
    
    for config, metrics in results.items():
        print(f"{config:<15} {metrics['win_rate']:<10.1%} {metrics['profit_factor']:<8.2f} "
              f"{metrics['sharpe_ratio']:<8.2f} {metrics['total_return']:<8.1%} "
              f"{metrics['max_drawdown']:<8.1%} {metrics['total_trades']:<8}")
    
    return results