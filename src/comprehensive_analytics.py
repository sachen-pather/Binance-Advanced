"""
Comprehensive Performance Analytics and Monitoring System
for Advanced Cryptocurrency Trading Bot with detailed attribution,
risk metrics, and real-time monitoring capabilities.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import sqlite3
from pathlib import Path

logger = logging.getLogger("BinanceTrading.Analytics")


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time"""
    timestamp: datetime
    total_equity: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    open_positions: int
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    var_95: float


@dataclass
class TradeAnalysis:
    """Comprehensive trade analysis"""
    trade_id: str
    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: datetime
    holding_period_hours: float
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    net_pnl: float
    pnl_percentage: float
    fees: float
    max_favorable_excursion: float
    max_adverse_excursion: float
    exit_reason: str
    market_regime_entry: str
    market_regime_exit: str
    confidence_score: float
    signal_strength: float


class PerformanceDatabase:
    """SQLite database for storing performance data"""
    
    def __init__(self, db_path: str = "trading_performance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Performance snapshots table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_equity REAL,
                        unrealized_pnl REAL,
                        realized_pnl REAL,
                        daily_pnl REAL,
                        open_positions INTEGER,
                        total_trades INTEGER,
                        win_rate REAL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        volatility REAL,
                        var_95 REAL
                    )
                ''')
                
                # Trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT,
                        entry_time TEXT NOT NULL,
                        exit_time TEXT,
                        holding_period_hours REAL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        quantity REAL NOT NULL,
                        side TEXT NOT NULL,
                        net_pnl REAL,
                        pnl_percentage REAL,
                        fees REAL,
                        max_favorable_excursion REAL,
                        max_adverse_excursion REAL,
                        exit_reason TEXT,
                        market_regime_entry TEXT,
                        market_regime_exit TEXT,
                        confidence_score REAL,
                        signal_strength REAL
                    )
                ''')
                
                # Daily metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT UNIQUE NOT NULL,
                        starting_equity REAL,
                        ending_equity REAL,
                        daily_return REAL,
                        realized_pnl REAL,
                        unrealized_pnl REAL,
                        trades_opened INTEGER,
                        trades_closed INTEGER,
                        win_rate REAL,
                        largest_win REAL,
                        largest_loss REAL,
                        volatility REAL
                    )
                ''')
                
                # Strategy performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy TEXT NOT NULL,
                        date TEXT NOT NULL,
                        trades_count INTEGER,
                        wins INTEGER,
                        losses INTEGER,
                        total_pnl REAL,
                        avg_pnl REAL,
                        win_rate REAL,
                        profit_factor REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL,
                        UNIQUE(strategy, date)
                    )
                ''')
                
                conn.commit()
                logger.info("Performance database initialized")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def save_performance_snapshot(self, snapshot: PerformanceSnapshot):
        """Save performance snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_snapshots 
                    (timestamp, total_equity, unrealized_pnl, realized_pnl, daily_pnl,
                     open_positions, total_trades, win_rate, sharpe_ratio, max_drawdown,
                     volatility, var_95)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp.isoformat(),
                    snapshot.total_equity,
                    snapshot.unrealized_pnl,
                    snapshot.realized_pnl,
                    snapshot.daily_pnl,
                    snapshot.open_positions,
                    snapshot.total_trades,
                    snapshot.win_rate,
                    snapshot.sharpe_ratio,
                    snapshot.max_drawdown,
                    snapshot.volatility,
                    snapshot.var_95
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving performance snapshot: {e}")
    
    def save_trade(self, trade: TradeAnalysis):
        """Save trade analysis to database with proper enum handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert exit_reason enum to string if it's an enum
                exit_reason_str = trade.exit_reason
                if hasattr(trade.exit_reason, 'name'):
                    exit_reason_str = trade.exit_reason.name
                elif hasattr(trade.exit_reason, 'value'):
                    exit_reason_str = str(trade.exit_reason.value)
                elif not isinstance(trade.exit_reason, str):
                    exit_reason_str = str(trade.exit_reason)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO trades 
                    (trade_id, symbol, strategy, entry_time, exit_time, holding_period_hours,
                    entry_price, exit_price, quantity, side, net_pnl, pnl_percentage, fees,
                    max_favorable_excursion, max_adverse_excursion, exit_reason,
                    market_regime_entry, market_regime_exit, confidence_score, signal_strength)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.trade_id, trade.symbol, trade.strategy,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.holding_period_hours,
                    trade.entry_price, trade.exit_price, trade.quantity, trade.side,
                    trade.net_pnl, trade.pnl_percentage, trade.fees,
                    trade.max_favorable_excursion, trade.max_adverse_excursion,
                    exit_reason_str,  # Use the converted string
                    trade.market_regime_entry, trade.market_regime_exit,
                    trade.confidence_score, trade.signal_strength
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def get_performance_history(self, days: int = 30) -> pd.DataFrame:
        """Get performance history from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now() - timedelta(days=days)
                query = '''
                    SELECT * FROM performance_snapshots 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn, params=(cutoff_date.isoformat(),))
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return pd.DataFrame()
    
    def get_trades_history(self, days: int = 30) -> pd.DataFrame:
        """Get trades history from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now() - timedelta(days=days)
                query = '''
                    SELECT * FROM trades 
                    WHERE entry_time >= ? 
                    ORDER BY entry_time DESC
                '''
                df = pd.read_sql_query(query, conn, params=(cutoff_date.isoformat(),))
                if not df.empty:
                    df['entry_time'] = pd.to_datetime(df['entry_time'])
                    df['exit_time'] = pd.to_datetime(df['exit_time'])
                return df
        except Exception as e:
            logger.error(f"Error getting trades history: {e}")
            return pd.DataFrame()


class ComprehensiveAnalytics:
    """Comprehensive analytics and monitoring system"""
    
    def __init__(self, initial_equity: float = 10000):
        self.initial_equity = initial_equity
        self.db = PerformanceDatabase()
        
        # In-memory performance tracking
        self.performance_history = []
        self.trade_history = []
        self.daily_metrics = {}
        self.strategy_performance = defaultdict(dict)
        
        # Risk metrics
        self.risk_metrics = {
            'var_history': [],
            'drawdown_history': [],
            'correlation_tracking': {},
            'volatility_regime': 'normal'
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'daily_loss_pct': 0.05,      # 5% daily loss
            'drawdown_pct': 0.15,        # 15% drawdown
            'win_rate_threshold': 0.4,    # 40% win rate minimum
            'sharpe_threshold': 0.5,      # 0.5 minimum Sharpe ratio
            'var_threshold_pct': 0.1,     # 10% VaR threshold
            'correlation_threshold': 0.8   # 80% correlation threshold
        }
        
        # Performance attribution categories
        self.attribution_categories = {
            'strategy': ['trend_following', 'mean_reversion', 'breakout', 'momentum', 'accumulation'],
            'market_regime': ['BULL_TREND', 'BEAR_TREND', 'RANGE_BOUND', 'HIGH_VOLATILITY', 'LOW_VOLATILITY'],
            'timeframe': ['short', 'medium', 'long'],
            'asset_class': ['major', 'altcoin', 'defi', 'layer1']
        }
    
    def record_performance_snapshot(self, equity: float, positions: Dict, 
                                   unrealized_pnl: float, daily_pnl: float,
                                   trade_count: int, win_rate: float) -> PerformanceSnapshot:
        """Record a performance snapshot"""
        
        try:
            # Calculate metrics
            total_return = (equity - self.initial_equity) / self.initial_equity
            
            # Calculate drawdown
            if self.performance_history:
                peak_equity = max(snap.total_equity for snap in self.performance_history[-252:])  # Last year
                max_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            else:
                max_drawdown = max(0, (self.initial_equity - equity) / self.initial_equity)
            
            # Calculate volatility (annualized)
            if len(self.performance_history) > 30:
                recent_returns = [
                    (snap.total_equity - prev_snap.total_equity) / prev_snap.total_equity
                    for snap, prev_snap in zip(self.performance_history[-30:], self.performance_history[-31:-1])
                ]
                volatility = np.std(recent_returns) * np.sqrt(252) if recent_returns else 0
            else:
                volatility = 0
            
            # Calculate Sharpe ratio
            if len(self.performance_history) > 30 and volatility > 0:
                recent_returns = [
                    (snap.total_equity - prev_snap.total_equity) / prev_snap.total_equity
                    for snap, prev_snap in zip(self.performance_history[-30:], self.performance_history[-31:-1])
                ]
                avg_return = np.mean(recent_returns) * 252  # Annualized
                sharpe_ratio = avg_return / volatility
            else:
                sharpe_ratio = 0
            
            # Estimate VaR (95%)
            var_95 = equity * 0.05  # Simplified - would use proper calculation in production
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                total_equity=equity,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=equity - self.initial_equity - unrealized_pnl,
                daily_pnl=daily_pnl,
                open_positions=len(positions),
                total_trades=trade_count,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                var_95=var_95
            )
            
            # Store snapshot
            self.performance_history.append(snapshot)
            self.db.save_performance_snapshot(snapshot)
            
            # Keep only last 1000 snapshots in memory
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error recording performance snapshot: {e}")
            return None
    
    
    def _extract_position_data(self, position) -> Dict:
        """Helper method to extract data from position objects safely"""
        try:
            if isinstance(position, dict):
                return position
            
            # Extract common attributes from position objects
            data = {}
            
            # Common attribute mappings
            attr_mappings = {
                'trade_id': ['trade_id', 'id', 'position_id'],
                'symbol': ['symbol', 'trading_pair', 'asset'],
                'strategy': ['strategy', 'strategy_name', 'type'],
                'entry_time': ['entry_time', 'created_at', 'timestamp'],
                'exit_time': ['exit_time', 'closed_at', 'end_time'],
                'entry_price': ['entry_price', 'open_price', 'price'],
                'exit_price': ['exit_price', 'close_price', 'current_price'],
                'quantity': ['quantity', 'size', 'amount'],
                'side': ['side', 'direction', 'position_type'],
                'net_pnl': ['net_pnl', 'pnl', 'profit_loss'],
                'fees': ['fees', 'commission', 'cost'],
                'exit_reason': ['exit_reason', 'close_reason', 'reason']
            }
            
            for key, possible_attrs in attr_mappings.items():
                for attr in possible_attrs:
                    if hasattr(position, attr):
                        value = getattr(position, attr)
                        if value is not None:
                            data[key] = value
                            break
                
                # Set defaults if not found
                if key not in data:
                    defaults = {
                        'trade_id': f"trade_{int(datetime.now().timestamp())}",
                        'symbol': 'UNKNOWN',
                        'strategy': 'unknown',
                        'entry_time': datetime.now(),
                        'exit_time': datetime.now(),
                        'entry_price': 0,
                        'exit_price': 0,
                        'quantity': 0,
                        'side': 'BUY',
                        'net_pnl': 0,
                        'fees': 0,
                        'exit_reason': 'unknown'
                    }
                    data[key] = defaults.get(key, 0)
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting position data: {e}")
            return {}

    
    
    def analyze_trade(self, position_data: Dict, market_context: Dict = None) -> TradeAnalysis:
        """Analyze a completed trade with proper object handling"""
        
        try:
            # Handle both dictionary and object types for position_data
            if hasattr(position_data, '__dict__'):
                # Convert object to dictionary for easier handling
                pos_dict = {}
                for attr in ['trade_id', 'symbol', 'strategy', 'entry_time', 'exit_time', 
                            'entry_price', 'exit_price', 'quantity', 'side', 'net_pnl', 
                            'pnl_percentage', 'fees', 'exit_reason', 'market_regime_entry',
                            'confidence_score', 'signal_strength']:
                    if hasattr(position_data, attr):
                        pos_dict[attr] = getattr(position_data, attr)
                    else:
                        # Set default values for missing attributes
                        defaults = {
                            'trade_id': f"trade_{int(datetime.now().timestamp())}",
                            'symbol': 'UNKNOWN',
                            'strategy': 'unknown',
                            'entry_time': datetime.now(),
                            'exit_time': datetime.now(),
                            'entry_price': 0,
                            'exit_price': 0,
                            'quantity': 0,
                            'side': 'BUY',
                            'net_pnl': 0,
                            'pnl_percentage': 0,
                            'fees': 0,
                            'exit_reason': 'unknown',
                            'market_regime_entry': 'unknown',
                            'confidence_score': 0.5,
                            'signal_strength': 0.5
                        }
                        pos_dict[attr] = defaults.get(attr, 0)
            else:
                pos_dict = position_data
            
            # Extract trade information with safe defaults
            trade_id = pos_dict.get('trade_id', f"trade_{int(datetime.now().timestamp())}")
            symbol = pos_dict.get('symbol', 'UNKNOWN')
            strategy = pos_dict.get('strategy', 'unknown')
            
            # Handle datetime conversion
            entry_time = pos_dict.get('entry_time')
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time)
                except:
                    entry_time = datetime.now()
            elif entry_time is None:
                entry_time = datetime.now()
            
            exit_time = pos_dict.get('exit_time')
            if isinstance(exit_time, str):
                try:
                    exit_time = datetime.fromisoformat(exit_time)
                except:
                    exit_time = datetime.now()
            elif exit_time is None:
                exit_time = datetime.now()
            
            # Calculate holding period
            if entry_time and exit_time:
                holding_period = (exit_time - entry_time).total_seconds() / 3600
            else:
                holding_period = 0
            
            # Handle exit_reason conversion
            exit_reason = pos_dict.get('exit_reason', 'unknown')
            if hasattr(exit_reason, 'name'):
                exit_reason_str = exit_reason.name
            elif hasattr(exit_reason, 'value'):
                exit_reason_str = str(exit_reason.value)
            else:
                exit_reason_str = str(exit_reason)
            
            # Calculate additional metrics if not provided
            entry_price = pos_dict.get('entry_price', 0)
            exit_price = pos_dict.get('exit_price', 0)
            quantity = pos_dict.get('quantity', 0)
            
            # Calculate PnL if not provided
            net_pnl = pos_dict.get('net_pnl')
            if net_pnl is None and entry_price > 0 and exit_price > 0 and quantity > 0:
                if pos_dict.get('side', 'BUY').upper() == 'BUY':
                    net_pnl = (exit_price - entry_price) * quantity
                else:
                    net_pnl = (entry_price - exit_price) * quantity
            elif net_pnl is None:
                net_pnl = 0
            
            # Calculate PnL percentage
            pnl_percentage = pos_dict.get('pnl_percentage')
            if pnl_percentage is None and entry_price > 0:
                pnl_percentage = ((exit_price - entry_price) / entry_price) * 100
            elif pnl_percentage is None:
                pnl_percentage = 0
            
            # Create trade analysis
            trade_analysis = TradeAnalysis(
                trade_id=trade_id,
                symbol=symbol,
                strategy=strategy,
                entry_time=entry_time,
                exit_time=exit_time,
                holding_period_hours=holding_period,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                side=pos_dict.get('side', 'BUY'),
                net_pnl=net_pnl,
                pnl_percentage=pnl_percentage,
                fees=pos_dict.get('fees', 0),
                max_favorable_excursion=pos_dict.get('max_favorable_excursion', 0),
                max_adverse_excursion=pos_dict.get('max_adverse_excursion', 0),
                exit_reason=exit_reason_str,  # Use converted string
                market_regime_entry=pos_dict.get('market_regime_entry', 'unknown'),
                market_regime_exit=market_context.get('current_regime', 'unknown') if market_context else 'unknown',
                confidence_score=pos_dict.get('confidence_score', 0.5),
                signal_strength=pos_dict.get('signal_strength', 0.5)
            )
            
            # Store in history
            self.trade_history.append(trade_analysis)
            self.db.save_trade(trade_analysis)
            
            # Update strategy performance
            self._update_strategy_performance(trade_analysis)
            
            return trade_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trade: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
        
        
    # Fix 4: Update the main analytics call to handle objects
    def record_trade_closure(self, position, market_context: Dict = None):
        """Record a trade closure with proper object handling"""
        try:
            # Extract position data safely
            position_data = self._extract_position_data(position)
            
            # Analyze the trade
            trade_analysis = self.analyze_trade(position_data, market_context)
            
            if trade_analysis:
                logger.info(f"Trade recorded: {trade_analysis.symbol} - {trade_analysis.net_pnl:.2f} USDT")
                return trade_analysis
            else:
                logger.warning("Failed to analyze trade")
                return None
                
        except Exception as e:
            logger.error(f"Error recording trade closure: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _update_strategy_performance(self, trade: TradeAnalysis):
        """Update strategy-specific performance metrics"""
        
        try:
            strategy = trade.strategy
            today = datetime.now().date().isoformat()
            
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = {}
            
            if today not in self.strategy_performance[strategy]:
                self.strategy_performance[strategy][today] = {
                    'trades': [],
                    'total_pnl': 0,
                    'wins': 0,
                    'losses': 0
                }
            
            # Add trade to strategy performance
            day_stats = self.strategy_performance[strategy][today]
            day_stats['trades'].append(trade)
            day_stats['total_pnl'] += trade.net_pnl
            
            if trade.net_pnl > 0:
                day_stats['wins'] += 1
            else:
                day_stats['losses'] += 1
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    def generate_performance_report(self, period_days: int = 30) -> Dict:
        """Generate comprehensive performance report"""
        
        try:
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Filter performance history
            recent_snapshots = [
                snap for snap in self.performance_history
                if snap.timestamp >= start_date
            ]
            
            # Filter trade history
            recent_trades = [
                trade for trade in self.trade_history
                if trade.entry_time >= start_date
            ]
            
            if not recent_snapshots:
                return {'error': 'No performance data available for the specified period'}
            
            # Overall performance metrics
            start_equity = recent_snapshots[0].total_equity
            end_equity = recent_snapshots[-1].total_equity
            total_return = (end_equity - start_equity) / start_equity if start_equity > 0 else 0
            
            # Trade statistics
            if recent_trades:
                winning_trades = [t for t in recent_trades if t.net_pnl > 0]
                losing_trades = [t for t in recent_trades if t.net_pnl <= 0]
                
                win_rate = len(winning_trades) / len(recent_trades)
                avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0
                profit_factor = abs(sum(t.net_pnl for t in winning_trades) / sum(t.net_pnl for t in losing_trades)) if losing_trades else float('inf')
                
                # Best and worst trades
                best_trade = max(recent_trades, key=lambda t: t.net_pnl)
                worst_trade = min(recent_trades, key=lambda t: t.net_pnl)
                
                # Average holding period
                avg_holding_period = np.mean([t.holding_period_hours for t in recent_trades])
            else:
                win_rate = avg_win = avg_loss = profit_factor = avg_holding_period = 0
                best_trade = worst_trade = None
            
            # Risk metrics
            if len(recent_snapshots) > 1:
                daily_returns = [
                    (curr.total_equity - prev.total_equity) / prev.total_equity
                    for curr, prev in zip(recent_snapshots[1:], recent_snapshots[:-1])
                ]
                
                volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
                max_drawdown = max([snap.max_drawdown for snap in recent_snapshots])
                current_sharpe = recent_snapshots[-1].sharpe_ratio
                current_var = recent_snapshots[-1].var_95
            else:
                volatility = max_drawdown = current_sharpe = current_var = 0
                daily_returns = []
            
            # Strategy attribution
            strategy_breakdown = self._analyze_strategy_attribution(recent_trades)
            
            # Market regime analysis
            regime_analysis = self._analyze_market_regime_performance(recent_trades)
            
            # Risk analysis
            risk_analysis = self._analyze_risk_metrics(recent_snapshots, daily_returns)
            
            # Alerts and recommendations
            alerts = self._generate_alerts(recent_snapshots[-1] if recent_snapshots else None, recent_trades)
            
            return {
                'report_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': period_days
                },
                'performance_summary': {
                    'start_equity': start_equity,
                    'end_equity': end_equity,
                    'total_return': total_return,
                    'total_return_pct': total_return * 100,
                    'annualized_return': (total_return + 1) ** (365 / period_days) - 1 if period_days > 0 else 0,
                    'volatility': volatility,
                    'sharpe_ratio': current_sharpe,
                    'max_drawdown': max_drawdown,
                    'var_95': current_var
                },
                'trading_statistics': {
                    'total_trades': len(recent_trades),
                    'winning_trades': len(winning_trades) if recent_trades else 0,
                    'losing_trades': len(losing_trades) if recent_trades else 0,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'avg_holding_period_hours': avg_holding_period,
                    'best_trade': {
                        'symbol': best_trade.symbol,
                        'pnl': best_trade.net_pnl,
                        'pnl_pct': best_trade.pnl_percentage
                    } if best_trade else None,
                    'worst_trade': {
                        'symbol': worst_trade.symbol,
                        'pnl': worst_trade.net_pnl,
                        'pnl_pct': worst_trade.pnl_percentage
                    } if worst_trade else None
                },
                'strategy_attribution': strategy_breakdown,
                'market_regime_analysis': regime_analysis,
                'risk_analysis': risk_analysis,
                'alerts_and_recommendations': alerts,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _analyze_strategy_attribution(self, trades: List[TradeAnalysis]) -> Dict:
        """Analyze performance attribution by strategy"""
        
        try:
            if not trades:
                return {}
            
            strategy_stats = defaultdict(lambda: {
                'trades': 0, 'wins': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_pnl': 0
            })
            
            for trade in trades:
                stats = strategy_stats[trade.strategy]
                stats['trades'] += 1
                stats['total_pnl'] += trade.net_pnl
                if trade.net_pnl > 0:
                    stats['wins'] += 1
            
            # Calculate derived metrics
            for strategy, stats in strategy_stats.items():
                stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            return dict(strategy_stats)
            
        except Exception as e:
            logger.error(f"Error analyzing strategy attribution: {e}")
            return {}
    
    def _analyze_market_regime_performance(self, trades: List[TradeAnalysis]) -> Dict:
        """Analyze performance by market regime"""
        
        try:
            if not trades:
                return {}
            
            regime_stats = defaultdict(lambda: {
                'trades': 0, 'wins': 0, 'total_pnl': 0, 'win_rate': 0, 'avg_pnl': 0
            })
            
            for trade in trades:
                regime = trade.market_regime_entry
                stats = regime_stats[regime]
                stats['trades'] += 1
                stats['total_pnl'] += trade.net_pnl
                if trade.net_pnl > 0:
                    stats['wins'] += 1
            
            # Calculate derived metrics
            for regime, stats in regime_stats.items():
                stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            return dict(regime_stats)
            
        except Exception as e:
            logger.error(f"Error analyzing market regime performance: {e}")
            return {}
    
    def _analyze_risk_metrics(self, snapshots: List[PerformanceSnapshot], 
                             daily_returns: List[float]) -> Dict:
        """Analyze comprehensive risk metrics"""
        
        try:
            if not snapshots or not daily_returns:
                return {}
            
            # VaR analysis
            var_95 = np.percentile(daily_returns, 5) if daily_returns else 0
            var_99 = np.percentile(daily_returns, 1) if daily_returns else 0
            
            # Expected Shortfall (Conditional VaR)
            var_95_threshold = np.percentile(daily_returns, 5)
            expected_shortfall = np.mean([r for r in daily_returns if r <= var_95_threshold]) if daily_returns else 0
            
            # Maximum consecutive losses
            consecutive_losses = 0
            max_consecutive_losses = 0
            for ret in daily_returns:
                if ret < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            # Volatility regimes
            if len(daily_returns) >= 30:
                recent_vol = np.std(daily_returns[-30:]) * np.sqrt(252)
                historical_vol = np.std(daily_returns) * np.sqrt(252)
                vol_regime = "high" if recent_vol > historical_vol * 1.5 else "low" if recent_vol < historical_vol * 0.7 else "normal"
            else:
                recent_vol = historical_vol = 0
                vol_regime = "unknown"
            
            return {
                'var_95_daily': var_95,
                'var_99_daily': var_99,
                'expected_shortfall': expected_shortfall,
                'max_consecutive_losses': max_consecutive_losses,
                'current_volatility': recent_vol,
                'historical_volatility': historical_vol,
                'volatility_regime': vol_regime,
                'risk_adjusted_return': np.mean(daily_returns) / np.std(daily_returns) if daily_returns and np.std(daily_returns) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing risk metrics: {e}")
            return {}
    
    def _generate_alerts(self, current_snapshot: PerformanceSnapshot, 
                        recent_trades: List[TradeAnalysis]) -> List[Dict]:
        """Generate alerts based on performance thresholds"""
        
        try:
            alerts = []
            
            if not current_snapshot:
                return alerts
            
            # Daily loss alert
            daily_loss_pct = abs(current_snapshot.daily_pnl) / current_snapshot.total_equity
            if current_snapshot.daily_pnl < 0 and daily_loss_pct > self.alert_thresholds['daily_loss_pct']:
                alerts.append({
                    'type': 'risk',
                    'severity': 'high',
                    'message': f"Daily loss exceeds threshold: {daily_loss_pct:.1%}",
                    'value': daily_loss_pct,
                    'threshold': self.alert_thresholds['daily_loss_pct']
                })
            
            # Drawdown alert
            if current_snapshot.max_drawdown > self.alert_thresholds['drawdown_pct']:
                alerts.append({
                    'type': 'risk',
                    'severity': 'high',
                    'message': f"Maximum drawdown exceeds threshold: {current_snapshot.max_drawdown:.1%}",
                    'value': current_snapshot.max_drawdown,
                    'threshold': self.alert_thresholds['drawdown_pct']
                })
            
            # Win rate alert
            if current_snapshot.win_rate < self.alert_thresholds['win_rate_threshold']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"Win rate below threshold: {current_snapshot.win_rate:.1%}",
                    'value': current_snapshot.win_rate,
                    'threshold': self.alert_thresholds['win_rate_threshold']
                })
            
            # Sharpe ratio alert
            if current_snapshot.sharpe_ratio < self.alert_thresholds['sharpe_threshold']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"Sharpe ratio below threshold: {current_snapshot.sharpe_ratio:.2f}",
                    'value': current_snapshot.sharpe_ratio,
                    'threshold': self.alert_thresholds['sharpe_threshold']
                })
            
            # High volatility alert
            if current_snapshot.volatility > 0.6:  # 60% annualized volatility
                alerts.append({
                    'type': 'risk',
                    'severity': 'medium',
                    'message': f"High portfolio volatility: {current_snapshot.volatility:.1%}",
                    'value': current_snapshot.volatility,
                    'threshold': 0.6
                })
            
            # Recent performance alerts
            if recent_trades:
                recent_losses = [t for t in recent_trades[-10:] if t.net_pnl < 0]
                if len(recent_losses) >= 7:  # 7 losses in last 10 trades
                    alerts.append({
                        'type': 'performance',
                        'severity': 'high',
                        'message': f"High recent loss rate: {len(recent_losses)}/10 recent trades",
                        'value': len(recent_losses) / 10,
                        'threshold': 0.6
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return []
    
    def get_real_time_dashboard_data(self) -> Dict:
        """Get real-time data for dashboard display"""
        
        try:
            current_snapshot = self.performance_history[-1] if self.performance_history else None
            recent_trades = self.trade_history[-20:] if self.trade_history else []
            
            # Calculate today's metrics
            today = datetime.now().date()
            today_trades = [t for t in self.trade_history if t.entry_time.date() == today]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': {
                    'total_equity': current_snapshot.total_equity if current_snapshot else 0,
                    'daily_pnl': current_snapshot.daily_pnl if current_snapshot else 0,
                    'unrealized_pnl': current_snapshot.unrealized_pnl if current_snapshot else 0,
                    'open_positions': current_snapshot.open_positions if current_snapshot else 0,
                    'win_rate': current_snapshot.win_rate if current_snapshot else 0,
                    'sharpe_ratio': current_snapshot.sharpe_ratio if current_snapshot else 0,
                    'max_drawdown': current_snapshot.max_drawdown if current_snapshot else 0,
                    'volatility': current_snapshot.volatility if current_snapshot else 0
                },
                'today_activity': {
                    'trades_opened': len([t for t in today_trades if not t.exit_time]),
                    'trades_closed': len([t for t in today_trades if t.exit_time]),
                    'best_trade_pnl': max([t.net_pnl for t in today_trades], default=0),
                    'worst_trade_pnl': min([t.net_pnl for t in today_trades], default=0)
                },
                'recent_trades': [
                    {
                        'symbol': t.symbol,
                        'strategy': t.strategy,
                        'side': t.side,
                        'pnl': t.net_pnl,
                        'pnl_pct': t.pnl_percentage,
                        'exit_reason': t.exit_reason,
                        'timestamp': t.exit_time.isoformat() if t.exit_time else t.entry_time.isoformat()
                    }
                    for t in recent_trades
                ],
                'alerts': self._generate_alerts(current_snapshot, recent_trades[-50:])
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}
    
    def export_performance_data(self, format: str = 'json', days: int = 30) -> str:
        """Export performance data in various formats"""
        
        try:
            report = self.generate_performance_report(days)
            
            if format.lower() == 'json':
                return json.dumps(report, indent=2, default=str)
            elif format.lower() == 'csv':
                # Convert to CSV format (simplified)
                trades_df = pd.DataFrame([
                    {
                        'timestamp': t.exit_time or t.entry_time,
                        'symbol': t.symbol,
                        'strategy': t.strategy,
                        'pnl': t.net_pnl,
                        'pnl_pct': t.pnl_percentage,
                        'holding_hours': t.holding_period_hours,
                        'exit_reason': t.exit_reason
                    }
                    for t in self.trade_history[-100:]  # Last 100 trades
                ])
                return trades_df.to_csv(index=False)
            else:
                return json.dumps(report, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            return f"Error: {str(e)}"