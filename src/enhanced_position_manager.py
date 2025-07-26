"""
Enhanced Position Management with Dynamic Exit Strategies,
Position Aging, and Sophisticated Order Management.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger("BinanceTrading.EnhancedPositionManager")


class ExitReason(Enum):
    """Enumeration of exit reasons for detailed tracking"""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_BASED = "time_based"
    REGIME_CHANGE = "regime_change"
    RISK_MANAGEMENT = "risk_management"
    CORRELATION_LIMIT = "correlation_limit"
    MANUAL = "manual"
    REBALANCING = "rebalancing"
    TECHNICAL_EXIT = "technical_exit"
    MOMENTUM_LOSS = "momentum_loss"
    PROFIT_TAKING = "profit_taking"


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "open"
    PARTIAL_CLOSE = "partial_close"
    CLOSED = "closed"
    PENDING = "pending"
    ERROR = "error"


@dataclass
class DynamicExit:
    """Dynamic exit configuration for positions"""
    trailing_stop_distance: float
    profit_protection_level: float  # % profit to protect
    time_based_exit_hours: Optional[int]
    momentum_exit_threshold: float
    regime_change_sensitivity: float
    partial_exit_levels: List[Tuple[float, float]]  # (profit_level, exit_percentage)


@dataclass
class EnhancedPosition:
    """Enhanced position with comprehensive tracking"""
    # Basic position info
    trade_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime
    value: float
    
    # Risk management
    stop_loss: float
    take_profit: float
    original_stop_loss: float
    original_take_profit: float
    
    # Dynamic exit configuration
    dynamic_exit: DynamicExit
    
    # Position tracking
    status: PositionStatus = PositionStatus.OPEN
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    highest_value: float = field(init=False)
    lowest_value: float = field(init=False)
    
    # Exit tracking
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None
    partial_exits: List[Dict] = field(default_factory=list)
    
    # Performance tracking
    max_favorable_excursion: float = 0.0  # Best profit during trade
    max_adverse_excursion: float = 0.0    # Worst loss during trade
    holding_period: timedelta = field(init=False)
    
    # Market context
    entry_market_regime: str = "unknown"
    entry_volatility: float = 0.0
    entry_rsi: float = 50.0
    
    # Orders
    orders: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.highest_value = self.value
        self.lowest_value = self.value
        self.holding_period = timedelta(0)


class EnhancedPositionManager:
    """Enhanced position manager with dynamic exits and sophisticated tracking"""
    
    def __init__(self):
        # Position storage
        self.positions: Dict[str, EnhancedPosition] = {}
        self.closed_positions: List[EnhancedPosition] = []
        
        # Paper trading storage
        self.paper_positions: Dict[str, EnhancedPosition] = {}
        self.paper_closed_positions: List[EnhancedPosition] = []
        
        # Exit strategy configurations
        self.exit_strategies = {
            'trend_following': DynamicExit(
                trailing_stop_distance=0.05,  # 5% trailing stop
                profit_protection_level=0.02,  # Protect after 2% profit
                time_based_exit_hours=72,      # 3 days max hold
                momentum_exit_threshold=-0.3,  # Exit if momentum drops 30%
                regime_change_sensitivity=0.8,
                partial_exit_levels=[(0.1, 0.3), (0.2, 0.5)]  # Take profits at 10% and 20%
            ),
            'mean_reversion': DynamicExit(
                trailing_stop_distance=0.03,
                profit_protection_level=0.015,
                time_based_exit_hours=24,      # 1 day max hold
                momentum_exit_threshold=-0.5,
                regime_change_sensitivity=0.6,
                partial_exit_levels=[(0.05, 0.5), (0.08, 0.8)]
            ),
            'breakout': DynamicExit(
                trailing_stop_distance=0.08,
                profit_protection_level=0.03,
                time_based_exit_hours=48,      # 2 days max hold
                momentum_exit_threshold=-0.4,
                regime_change_sensitivity=0.7,
                partial_exit_levels=[(0.15, 0.25), (0.25, 0.5)]
            ),
            'momentum': DynamicExit(
                trailing_stop_distance=0.06,
                profit_protection_level=0.025,
                time_based_exit_hours=60,      # 2.5 days max hold
                momentum_exit_threshold=-0.35,
                regime_change_sensitivity=0.75,
                partial_exit_levels=[(0.12, 0.3), (0.20, 0.6)]
            ),
            'accumulation': DynamicExit(
                trailing_stop_distance=0.04,
                profit_protection_level=0.01,
                time_based_exit_hours=168,     # 1 week max hold
                momentum_exit_threshold=-0.25,
                regime_change_sensitivity=0.9,
                partial_exit_levels=[(0.08, 0.2), (0.15, 0.4)]
            )
        }
        
        # Performance tracking
        self.exit_performance = {reason.value: {'count': 0, 'total_pnl': 0, 'avg_pnl': 0} 
                                for reason in ExitReason}
        
        # Configuration
        self.config = {
            'update_frequency_minutes': 5,
            'trailing_stop_activation_profit': 0.02,  # Activate trailing stop after 2% profit
            'partial_exit_cooldown_minutes': 30,
            'max_adverse_excursion_limit': 0.15,  # Force exit if loss exceeds 15%
            'time_based_warning_hours': 12,  # Warn when approaching time-based exit
            'momentum_calculation_periods': 20
        }
    
    def add_position(self, trade_info: Dict, strategy_type: str = 'trend_following', 
                    paper_trade: bool = False) -> str:
        """Add a new enhanced position with dynamic exit configuration"""
        
        try:
            # Get appropriate exit strategy
            exit_strategy = self.exit_strategies.get(strategy_type, self.exit_strategies['trend_following'])
            
            # Create enhanced position
            position = EnhancedPosition(
                trade_id=trade_info.get('trade_id', f"{trade_info['symbol']}_{int(datetime.now().timestamp())}"),
                symbol=trade_info['symbol'],
                side=trade_info['side'],
                quantity=trade_info['quantity'],
                entry_price=trade_info['entry_price'],
                entry_time=trade_info.get('entry_time', datetime.now()),
                value=trade_info.get('value', trade_info['quantity'] * trade_info['entry_price']),
                stop_loss=trade_info.get('stop_loss', 0),
                take_profit=trade_info.get('take_profit', 0),
                original_stop_loss=trade_info.get('stop_loss', 0),
                original_take_profit=trade_info.get('take_profit', 0),
                dynamic_exit=exit_strategy,
                entry_market_regime=trade_info.get('market_regime', 'unknown'),
                entry_volatility=trade_info.get('volatility', 0),
                entry_rsi=trade_info.get('rsi', 50),
                orders=trade_info.get('orders', {})
            )
            
            # Store position
            if paper_trade:
                self.paper_positions[position.trade_id] = position
                logger.info(f"Paper position added: {position.trade_id} ({strategy_type})")
            else:
                self.positions[position.trade_id] = position
                logger.info(f"Live position added: {position.trade_id} ({strategy_type})")
            
            return position.trade_id
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return None
    
    def update_positions(self, current_prices: Dict[str, float], 
                        market_data: Dict[str, pd.DataFrame] = None,
                        current_market_regime: str = 'unknown',
                        paper_trade: bool = False) -> Tuple[List[EnhancedPosition], List[str]]:
        """Update all positions with dynamic exit logic"""
        
        try:
            positions_to_use = self.paper_positions if paper_trade else self.positions
            closed_positions = []
            exit_signals = []
            
            for trade_id, position in list(positions_to_use.items()):
                if position.status != PositionStatus.OPEN:
                    continue
                
                symbol = position.symbol
                current_price = current_prices.get(symbol)
                
                if current_price is None:
                    logger.warning(f"No current price for {symbol}")
                    continue
                
                # Update position metrics
                self._update_position_metrics(position, current_price)
                
                # Check for exit conditions
                exit_reason = self._check_exit_conditions(
                    position, current_price, market_data, current_market_regime
                )
                
                if exit_reason:
                    # Execute exit
                    closed_position = self._execute_exit(
                        position, current_price, exit_reason, paper_trade
                    )
                    if closed_position:
                        closed_positions.append(closed_position)
                        exit_signals.append(f"{symbol}: {exit_reason.value}")
                        
                        # Remove from active positions
                        del positions_to_use[trade_id]
                
                # Check for partial exit conditions
                elif self._should_partial_exit(position, current_price):
                    partial_exit_info = self._execute_partial_exit(position, current_price)
                    if partial_exit_info:
                        exit_signals.append(f"{symbol}: Partial exit - {partial_exit_info}")
            
            # Move closed positions to history
            if paper_trade:
                self.paper_closed_positions.extend(closed_positions)
            else:
                self.closed_positions.extend(closed_positions)
            
            return closed_positions, exit_signals
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            return [], []
    
    def _update_position_metrics(self, position: EnhancedPosition, current_price: float):
        """Update position performance metrics"""
        
        try:
            # Calculate current P&L
            if position.side == 'BUY':
                current_pnl = (current_price - position.entry_price) * position.quantity
                pnl_pct = (current_price - position.entry_price) / position.entry_price
            else:  # SELL
                current_pnl = (position.entry_price - current_price) * position.quantity
                pnl_pct = (position.entry_price - current_price) / position.entry_price
            
            position.unrealized_pnl = current_pnl
            
            # Update excursion tracking
            if pnl_pct > 0:
                position.max_favorable_excursion = max(position.max_favorable_excursion, pnl_pct)
            else:
                position.max_adverse_excursion = min(position.max_adverse_excursion, pnl_pct)
            
            # Update value tracking
            current_value = position.quantity * current_price
            position.highest_value = max(position.highest_value, current_value)
            position.lowest_value = min(position.lowest_value, current_value)
            
            # Update holding period
            position.holding_period = datetime.now() - position.entry_time
            
            # Update trailing stop if profitable
            if pnl_pct > position.dynamic_exit.profit_protection_level:
                self._update_trailing_stop(position, current_price, pnl_pct)
            
        except Exception as e:
            logger.error(f"Error updating position metrics: {e}")
    
    def _update_trailing_stop(self, position: EnhancedPosition, current_price: float, pnl_pct: float):
        """Update trailing stop loss"""
        
        try:
            trailing_distance = position.dynamic_exit.trailing_stop_distance
            
            if position.side == 'BUY':
                # For long positions, trail stop up
                new_stop = current_price * (1 - trailing_distance)
                if new_stop > position.stop_loss:
                    old_stop = position.stop_loss
                    position.stop_loss = new_stop
                    logger.debug(f"Trailing stop updated for {position.symbol}: {old_stop:.6f} -> {new_stop:.6f}")
            else:  # SELL
                # For short positions, trail stop down
                new_stop = current_price * (1 + trailing_distance)
                if new_stop < position.stop_loss or position.stop_loss == 0:
                    old_stop = position.stop_loss
                    position.stop_loss = new_stop
                    logger.debug(f"Trailing stop updated for {position.symbol}: {old_stop:.6f} -> {new_stop:.6f}")
                    
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
    
    def _check_exit_conditions(self, position: EnhancedPosition, current_price: float,
                              market_data: Dict[str, pd.DataFrame], 
                              current_market_regime: str) -> Optional[ExitReason]:
        """Check all exit conditions for a position"""
        
        try:
            # 1. Stop Loss
            if position.side == 'BUY' and current_price <= position.stop_loss:
                return ExitReason.STOP_LOSS
            elif position.side == 'SELL' and current_price >= position.stop_loss:
                return ExitReason.STOP_LOSS
            
            # 2. Take Profit
            if position.side == 'BUY' and current_price >= position.take_profit:
                return ExitReason.TAKE_PROFIT
            elif position.side == 'SELL' and current_price <= position.take_profit:
                return ExitReason.TAKE_PROFIT
            
            # 3. Time-based exit
            if position.dynamic_exit.time_based_exit_hours:
                hours_held = position.holding_period.total_seconds() / 3600
                if hours_held >= position.dynamic_exit.time_based_exit_hours:
                    return ExitReason.TIME_BASED
            
            # 4. Maximum adverse excursion limit
            if position.max_adverse_excursion < -self.config['max_adverse_excursion_limit']:
                return ExitReason.RISK_MANAGEMENT
            
            # 5. Market regime change
            if self._detect_regime_change(position, current_market_regime):
                return ExitReason.REGIME_CHANGE
            
            # 6. Momentum loss
            if market_data and position.symbol in market_data:
                momentum_loss = self._detect_momentum_loss(position, market_data[position.symbol])
                if momentum_loss:
                    return ExitReason.MOMENTUM_LOSS
            
            # 7. Technical exit signals
            if market_data and position.symbol in market_data:
                technical_exit = self._check_technical_exit(position, market_data[position.symbol])
                if technical_exit:
                    return ExitReason.TECHNICAL_EXIT
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    def _detect_regime_change(self, position: EnhancedPosition, current_regime: str) -> bool:
        """Detect if market regime has changed significantly"""
        
        try:
            # Define regime compatibility
            regime_transitions = {
                'BULL_TREND': ['BEAR_TREND', 'HIGH_VOLATILITY'],
                'BEAR_TREND': ['BULL_TREND'],
                'RANGE_BOUND': ['BREAKOUT', 'HIGH_VOLATILITY'],
                'HIGH_VOLATILITY': ['LOW_VOLATILITY'],
                'LOW_VOLATILITY': ['HIGH_VOLATILITY', 'BREAKOUT'],
                'BREAKOUT': ['RANGE_BOUND'],
                'ACCUMULATION': ['DISTRIBUTION'],
                'DISTRIBUTION': ['ACCUMULATION']
            }
            
            # Check if current regime is incompatible with entry regime
            entry_regime = position.entry_market_regime.upper()
            current_regime = current_regime.upper()
            
            if entry_regime in regime_transitions:
                incompatible_regimes = regime_transitions[entry_regime]
                return current_regime in incompatible_regimes
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting regime change: {e}")
            return False
    
    def _detect_momentum_loss(self, position: EnhancedPosition, market_data: pd.DataFrame) -> bool:
        """Detect significant momentum loss"""
        
        try:
            if len(market_data) < self.config['momentum_calculation_periods']:
                return False
            
            # Calculate recent momentum
            periods = self.config['momentum_calculation_periods']
            recent_returns = market_data['close'].pct_change(periods).iloc[-1]
            
            # Compare with threshold
            threshold = position.dynamic_exit.momentum_exit_threshold
            
            if position.side == 'BUY':
                return recent_returns < threshold
            else:  # SELL
                return recent_returns > -threshold  # Inverse for short positions
            
        except Exception as e:
            logger.error(f"Error detecting momentum loss: {e}")
            return False
    
    def _check_technical_exit(self, position: EnhancedPosition, market_data: pd.DataFrame) -> bool:
        """Check for technical analysis exit signals"""
        
        try:
            if len(market_data) < 20:
                return False
            
            latest = market_data.iloc[-1]
            
            # RSI reversal
            if 'RSI_14' in market_data.columns:
                rsi = latest['RSI_14']
                if position.side == 'BUY' and rsi > 80:  # Overbought
                    return True
                elif position.side == 'SELL' and rsi < 20:  # Oversold
                    return True
            
            # MACD divergence
            if 'MACD' in market_data.columns and 'MACD_signal' in market_data.columns:
                macd_cross = (latest['MACD'] < latest['MACD_signal'] and 
                             market_data['MACD'].iloc[-2] >= market_data['MACD_signal'].iloc[-2])
                
                if position.side == 'BUY' and macd_cross:
                    return True
            
            # Moving average breakdown
            if 'EMA_20' in market_data.columns and 'EMA_50' in market_data.columns:
                if position.side == 'BUY':
                    # Exit long if short MA crosses below long MA
                    current_cross = latest['EMA_20'] < latest['EMA_50']
                    previous_cross = market_data['EMA_20'].iloc[-2] >= market_data['EMA_50'].iloc[-2]
                    return current_cross and previous_cross
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking technical exit: {e}")
            return False
    
    def _should_partial_exit(self, position: EnhancedPosition, current_price: float) -> bool:
        """Check if position should be partially closed"""
        
        try:
            if not position.dynamic_exit.partial_exit_levels:
                return False
            
            # Calculate current profit percentage
            if position.side == 'BUY':
                pnl_pct = (current_price - position.entry_price) / position.entry_price
            else:  # SELL
                pnl_pct = (position.entry_price - current_price) / position.entry_price
            
            # Check if we've hit any partial exit levels
            for profit_threshold, exit_percentage in position.dynamic_exit.partial_exit_levels:
                if pnl_pct >= profit_threshold:
                    # Check if we haven't already partially exited at this level
                    already_exited = any(
                        exit_info.get('trigger_level') == profit_threshold 
                        for exit_info in position.partial_exits
                    )
                    if not already_exited:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking partial exit: {e}")
            return False
    
    def _execute_partial_exit(self, position: EnhancedPosition, current_price: float) -> Optional[str]:
        """Execute partial position exit"""
        
        try:
            # Calculate current profit percentage
            if position.side == 'BUY':
                pnl_pct = (current_price - position.entry_price) / position.entry_price
            else:  # SELL
                pnl_pct = (position.entry_price - current_price) / position.entry_price
            
            # Find the appropriate exit level
            exit_info = None
            for profit_threshold, exit_percentage in position.dynamic_exit.partial_exit_levels:
                if pnl_pct >= profit_threshold:
                    # Check if not already executed
                    already_exited = any(
                        exit_info.get('trigger_level') == profit_threshold 
                        for exit_info in position.partial_exits
                    )
                    if not already_exited:
                        exit_info = {
                            'trigger_level': profit_threshold,
                            'exit_percentage': exit_percentage,
                            'exit_price': current_price,
                            'exit_time': datetime.now(),
                            'pnl_at_exit': pnl_pct
                        }
                        break
            
            if not exit_info:
                return None
            
            # Calculate exit quantity
            exit_quantity = position.quantity * exit_info['exit_percentage']
            exit_value = exit_quantity * current_price
            
            # Update position
            position.quantity -= exit_quantity
            position.value = position.quantity * position.entry_price
            
            # Calculate realized P&L for this partial exit
            if position.side == 'BUY':
                partial_pnl = (current_price - position.entry_price) * exit_quantity
            else:  # SELL
                partial_pnl = (position.entry_price - current_price) * exit_quantity
            
            position.realized_pnl += partial_pnl
            
            # Record the partial exit
            exit_info.update({
                'quantity_sold': exit_quantity,
                'remaining_quantity': position.quantity,
                'partial_pnl': partial_pnl,
                'exit_value': exit_value
            })
            position.partial_exits.append(exit_info)
            
            logger.info(f"Partial exit executed for {position.symbol}: "
                       f"{exit_info['exit_percentage']:.1%} at {current_price:.6f}")
            
            return f"{exit_info['exit_percentage']:.1%} at {pnl_pct:.1%} profit"
            
        except Exception as e:
            logger.error(f"Error executing partial exit: {e}")
            return None
    
    def _execute_exit(self, position: EnhancedPosition, current_price: float, 
                     exit_reason: ExitReason, paper_trade: bool = True) -> Optional[EnhancedPosition]:
        """Execute complete position exit"""
        
        try:
            # Calculate final P&L
            if position.side == 'BUY':
                gross_pnl = (current_price - position.entry_price) * position.quantity
            else:  # SELL
                gross_pnl = (position.entry_price - current_price) * position.quantity
            
            # Add any previously realized P&L from partial exits
            total_pnl = gross_pnl + position.realized_pnl
            
            # Update position for closure
            position.status = PositionStatus.CLOSED
            position.exit_price = current_price
            position.exit_time = datetime.now()
            position.exit_reason = exit_reason
            position.unrealized_pnl = 0.0
            position.realized_pnl = total_pnl
            
            # Update exit performance tracking
            self.exit_performance[exit_reason.value]['count'] += 1
            self.exit_performance[exit_reason.value]['total_pnl'] += total_pnl
            self.exit_performance[exit_reason.value]['avg_pnl'] = (
                self.exit_performance[exit_reason.value]['total_pnl'] / 
                self.exit_performance[exit_reason.value]['count']
            )
            
            logger.info(f"Position closed: {position.symbol} - {exit_reason.value} - "
                       f"P&L: {total_pnl:.2f} USDT")
            
            return position
            
        except Exception as e:
            logger.error(f"Error executing exit: {e}")
            return None
        
    def get_open_positions(self, paper_trade: bool = False) -> Dict[str, EnhancedPosition]:
        """
        Returns the dictionary of currently open positions.
        
        Args:
            paper_trade (bool): If True, returns paper trading positions. 
                                Otherwise, returns live positions.

        Returns:
            Dict[str, EnhancedPosition]: A dictionary of the open positions.
        """
        if paper_trade:
            return self.paper_positions
        else:
            return self.positions
    
    def get_position_analytics(self, paper_trade: bool = False) -> Dict:
        """Get comprehensive position analytics"""
        
        try:
            positions_to_use = self.paper_positions if paper_trade else self.positions
            closed_positions_to_use = self.paper_closed_positions if paper_trade else self.closed_positions
            
            # Current positions analysis
            current_positions = []
            total_unrealized_pnl = 0
            
            for position in positions_to_use.values():
                current_positions.append({
                    'symbol': position.symbol,
                    'side': position.side,
                    'entry_price': position.entry_price,
                    'current_value': position.value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'holding_period_hours': position.holding_period.total_seconds() / 3600,
                    'max_favorable_excursion': position.max_favorable_excursion,
                    'max_adverse_excursion': position.max_adverse_excursion,
                    'partial_exits': len(position.partial_exits)
                })
                total_unrealized_pnl += position.unrealized_pnl
            
            # Historical performance analysis
            exit_reason_stats = {}
            for reason, stats in self.exit_performance.items():
                if stats['count'] > 0:
                    exit_reason_stats[reason] = {
                        'count': stats['count'],
                        'total_pnl': stats['total_pnl'],
                        'avg_pnl': stats['avg_pnl'],
                        'success_rate': (stats['total_pnl'] > 0)
                    }
            
            # Holding period analysis
            if closed_positions_to_use:
                holding_periods = [pos.holding_period.total_seconds() / 3600 
                                 for pos in closed_positions_to_use]
                avg_holding_period = np.mean(holding_periods)
                max_holding_period = np.max(holding_periods)
                min_holding_period = np.min(holding_periods)
            else:
                avg_holding_period = max_holding_period = min_holding_period = 0
            
            # Excursion analysis
            if closed_positions_to_use:
                mfe_values = [pos.max_favorable_excursion for pos in closed_positions_to_use]
                mae_values = [abs(pos.max_adverse_excursion) for pos in closed_positions_to_use]
                
                avg_mfe = np.mean(mfe_values) if mfe_values else 0
                avg_mae = np.mean(mae_values) if mae_values else 0
            else:
                avg_mfe = avg_mae = 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'current_positions': {
                    'count': len(current_positions),
                    'total_unrealized_pnl': total_unrealized_pnl,
                    'positions': current_positions
                },
                'exit_performance': exit_reason_stats,
                'holding_period_analysis': {
                    'average_hours': avg_holding_period,
                    'max_hours': max_holding_period,
                    'min_hours': min_holding_period
                },
                'excursion_analysis': {
                    'average_max_favorable': avg_mfe,
                    'average_max_adverse': avg_mae,
                    'efficiency_ratio': avg_mfe / avg_mae if avg_mae > 0 else 0
                },
                'closed_positions_count': len(closed_positions_to_use)
            }
            
        except Exception as e:
            logger.error(f"Error getting position analytics: {e}")
            return {'error': str(e)}
    
    def get_exit_recommendations(self, paper_trade: bool = False) -> List[Dict]:
        """Get recommendations for position exits"""
        
        try:
            positions_to_use = self.paper_positions if paper_trade else self.positions
            recommendations = []
            
            for position in positions_to_use.values():
                if position.status != PositionStatus.OPEN:
                    continue
                
                position_recommendations = []
                
                # Time-based warning
                if position.dynamic_exit.time_based_exit_hours:
                    hours_held = position.holding_period.total_seconds() / 3600
                    hours_remaining = position.dynamic_exit.time_based_exit_hours - hours_held
                    
                    if hours_remaining <= self.config['time_based_warning_hours']:
                        position_recommendations.append({
                            'type': 'time_warning',
                            'message': f"Approaching time-based exit in {hours_remaining:.1f} hours",
                            'urgency': 'medium'
                        })
                
                # Adverse excursion warning
                if position.max_adverse_excursion < -0.10:  # 10% adverse excursion
                    position_recommendations.append({
                        'type': 'risk_warning',
                        'message': f"High adverse excursion: {position.max_adverse_excursion:.1%}",
                        'urgency': 'high'
                    })
                
                # Profit protection opportunity
                if position.max_favorable_excursion > 0.05:  # 5% profit achieved
                    current_pnl_pct = position.unrealized_pnl / position.value if position.value > 0 else 0
                    if current_pnl_pct < position.max_favorable_excursion * 0.7:  # Given back 30% of profits
                        position_recommendations.append({
                            'type': 'profit_protection',
                            'message': f"Consider profit protection - gave back {(position.max_favorable_excursion - current_pnl_pct):.1%}",
                            'urgency': 'medium'
                        })
                
                if position_recommendations:
                    recommendations.append({
                        'symbol': position.symbol,
                        'trade_id': position.trade_id,
                        'recommendations': position_recommendations
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting exit recommendations: {e}")
            return []
    
    def save_position_state(self, filepath: str = 'position_state.json'):
        """Save position state to file"""
        
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'positions': {
                    trade_id: {
                        'trade_id': pos.trade_id,
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'entry_time': pos.entry_time.isoformat(),
                        'value': pos.value,
                        'stop_loss': pos.stop_loss,
                        'take_profit': pos.take_profit,
                        'status': pos.status.value,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'realized_pnl': pos.realized_pnl,
                        'max_favorable_excursion': pos.max_favorable_excursion,
                        'max_adverse_excursion': pos.max_adverse_excursion,
                        'partial_exits': pos.partial_exits,
                        'entry_market_regime': pos.entry_market_regime
                    }
                    for trade_id, pos in self.positions.items()
                },
                'paper_positions': {
                    trade_id: {
                        'trade_id': pos.trade_id,
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'entry_time': pos.entry_time.isoformat(),
                        'value': pos.value,
                        'stop_loss': pos.stop_loss,
                        'take_profit': pos.take_profit,
                        'status': pos.status.value,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'realized_pnl': pos.realized_pnl,
                        'max_favorable_excursion': pos.max_favorable_excursion,
                        'max_adverse_excursion': pos.max_adverse_excursion,
                        'partial_exits': pos.partial_exits,
                        'entry_market_regime': pos.entry_market_regime
                    }
                    for trade_id, pos in self.paper_positions.items()
                },
                'exit_performance': self.exit_performance
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Position state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving position state: {e}")
            return False
    
    def load_position_state(self, filepath: str = 'position_state.json') -> bool:
        """Load position state from file"""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore exit performance
            self.exit_performance.update(state.get('exit_performance', {}))
            
            logger.info(f"Position state loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            logger.info("No position state file found")
            return False
        except Exception as e:
            logger.error(f"Error loading position state: {e}")
            return False