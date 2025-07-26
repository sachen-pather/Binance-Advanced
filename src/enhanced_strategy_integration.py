"""
Enhanced Strategy Integration and Orchestration Module
Coordinates all components of the advanced trading system with
intelligent decision making and adaptive strategy selection.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Import all enhanced components
from enhanced_indicators import EnhancedTechnicalIndicators
from advanced_ml_engine import AdvancedMLEngine, ModelPerformanceMonitor
from advanced_market_analysis import AdvancedMarketAnalyzer, TradingOpportunity, MarketRegime
from advanced_risk_management import AdvancedRiskManager, PositionSizeRecommendation
from enhanced_position_manager import EnhancedPositionManager, ExitReason
from comprehensive_analytics import ComprehensiveAnalytics, PerformanceSnapshot

logger = logging.getLogger("BinanceTrading.EnhancedStrategy")


@dataclass
class StrategyConfiguration:
    """Configuration for strategy execution"""
    # --- Fields without default values (must be provided during creation) ---
    strategy_weights: Dict[str, float]
    regime_strategy_multipliers: Dict[str, Dict[str, float]] # MOVED TO THE TOP

    # --- Fields with default values (optional) ---
    # Execution parameters
    max_concurrent_analysis: int = 10
    analysis_timeout_seconds: int = 30
    min_opportunity_score: float = 0.6
    max_daily_trades: int = 20
    
    # Risk parameters
    portfolio_heat_limit: float = 0.8
    correlation_limit: float = 0.7
    max_drawdown_limit: float = 0.15
    
    # ML parameters
    ml_confidence_threshold: float = 0.6
    model_retrain_threshold: float = 0.55
    ensemble_min_agreement: float = 0.7
    
    # Market regime adjustments
    regime_strategy_multipliers: Dict[str, Dict[str, float]]


class EnhancedTradingStrategy:
    """Enhanced trading strategy orchestration with all advanced components"""
    
    def __init__(self, config: StrategyConfiguration, paper_trade: bool = True):
        self.config = config
        self.paper_trade = paper_trade
        
        # Initialize all components
        self.indicator_calculator = EnhancedTechnicalIndicators()
        self.ml_engine = AdvancedMLEngine()
        self.market_analyzer = AdvancedMarketAnalyzer()
        self.risk_manager = AdvancedRiskManager()
        self.position_manager = EnhancedPositionManager()
        self.analytics = ComprehensiveAnalytics()
        self.ml_monitor = ModelPerformanceMonitor()
        
        # Load saved models and state
        self.ml_engine.load_models()
        self.position_manager.load_position_state()
        
        # Strategy state
        self.current_equity = 10000.0
        self.daily_pnl = 0.0
        self.strategy_state = {
            'last_analysis_time': None,
            'last_ml_retrain': None,
            'circuit_breaker_active': False,
            'emergency_stop': False,
            'performance_degradation': False
        }
        
        # Performance tracking
        self.execution_metrics = {
            'opportunities_analyzed': 0,
            'trades_executed': 0,
            'analysis_time_avg': 0,
            'ml_predictions_made': 0,
            'risk_checks_performed': 0
        }
        
        # Thread pool for concurrent analysis
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_analysis)
        
        logger.info("Enhanced Trading Strategy initialized")
    
    async def run_comprehensive_analysis(self, data_fetcher, supported_symbols: List[str]) -> Dict:
        """Run comprehensive market analysis with all components"""
        
        try:
            analysis_start_time = datetime.now()
            
            # Step 1: Market Regime Analysis
            logger.info("Analyzing market regime...")
            regime_symbols = [s['symbol'] for s in supported_symbols[:5]]
            current_regime = self.market_analyzer.analyze_market_regime(
                data_fetcher, self.indicator_calculator, regime_symbols
            )
            
            # Step 2: Update current positions
            logger.info("Updating positions...")
            price_symbols = [s['symbol'] for s in supported_symbols[:20]]
            current_prices = await self._get_current_prices(data_fetcher, price_symbols)
            market_data_symbols = [s['symbol'] for s in supported_symbols[:10]]
            market_data = await self._get_market_data(data_fetcher, market_data_symbols)
            
            closed_positions, exit_signals = self.position_manager.update_positions(
                current_prices, market_data, current_regime.value, self.paper_trade
            )
            
            # Step 3: Process closed positions
            for closed_position in closed_positions:
            # Use the new method that handles enum conversion automatically
                trade_analysis = self.analytics.record_trade_closure(
                    closed_position, 
                    {'current_regime': current_regime.value}
                )
                
                # Update ML model performance
                if hasattr(closed_position, 'ml_prediction'):
                    actual_result = 1 if closed_position.realized_pnl > 0 else -1
                    self.ml_monitor.log_prediction(
                        closed_position.ml_prediction,
                        closed_position.ml_confidence,
                        actual_result
                    )
            
            # Step 4: Check if ML model needs retraining
            if self.ml_monitor.should_retrain(self.config.model_retrain_threshold):
                logger.info("ML model performance degraded, retraining...")
                success = self.ml_engine.retrain_if_needed(
                    data_fetcher, self.indicator_calculator, force_retrain=True
                )
                if success:
                    logger.info("ML model retrained successfully")
            
            # Step 5: Find trading opportunities
            logger.info("Analyzing trading opportunities...")
            current_positions = self.position_manager.get_open_positions(self.paper_trade)
            
            opportunities = await self._analyze_opportunities_concurrent(
                data_fetcher, supported_symbols, current_positions
            )
            
            # Step 6: Risk assessment
            logger.info("Performing risk assessment...")
            portfolio_risk = await self._assess_portfolio_risk(current_positions, market_data)
            
            # Step 7: Strategy selection and execution
            logger.info("Selecting and executing strategies...")
            executed_trades = await self._execute_selected_strategies(
                opportunities, portfolio_risk, data_fetcher
            )
            
            # Step 8: Update analytics
            self.current_equity = self._calculate_current_equity(current_positions, current_prices)
            performance_snapshot = self.analytics.record_performance_snapshot(
                self.current_equity,
                current_positions,
                sum(pos.unrealized_pnl for pos in current_positions.values()),
                self.daily_pnl,
                len(self.analytics.trade_history),
                self._calculate_current_win_rate()
            )
            
            # Step 9: Generate recommendations
            recommendations = self._generate_strategy_recommendations(
                current_regime, portfolio_risk, performance_snapshot
            )
            
            analysis_time = (datetime.now() - analysis_start_time).total_seconds()
            self.execution_metrics['analysis_time_avg'] = (
                self.execution_metrics['analysis_time_avg'] * 0.9 + analysis_time * 0.1
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'market_regime': current_regime.value,
                'opportunities_found': len(opportunities),
                'trades_executed': len(executed_trades),
                'current_equity': self.current_equity,
                'portfolio_risk': portfolio_risk,
                'performance_snapshot': performance_snapshot.__dict__ if performance_snapshot else None,
                'recommendations': recommendations,
                'execution_time_seconds': analysis_time,
                'exit_signals': exit_signals,
                'closed_positions': len(closed_positions)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _get_current_prices(self, data_fetcher, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols"""
        prices = {}
        
        for symbol_info in symbols:
            try:
                symbol = symbol_info['symbol'] if isinstance(symbol_info, dict) else symbol_info
                # Get recent data to extract current price
                df = data_fetcher.get_historical_data(symbol, '1m', '5 minutes')
                if df is not None and len(df) > 0:
                    prices[symbol] = float(df.iloc[-1]['close'])
            except Exception as e:
                logger.debug(f"Error getting price for {symbol}: {e}")
                continue
        
        return prices
    
    async def _get_market_data(self, data_fetcher, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Get market data with indicators for symbols"""
        market_data = {}
        
        for symbol_info in symbols:
            try:
                symbol = symbol_info['symbol'] if isinstance(symbol_info, dict) else symbol_info
                df = data_fetcher.get_historical_data(symbol, '1h', '10 days')
                if df is not None:
                    df_with_indicators = self.indicator_calculator.calculate_all_indicators(df)
                    if df_with_indicators is not None:
                        market_data[symbol] = df_with_indicators
            except Exception as e:
                logger.debug(f"Error getting market data for {symbol}: {e}")
                continue
        
        return market_data
    
    async def _analyze_opportunities_concurrent(self, data_fetcher, supported_symbols: List[str],
                                              current_positions: Dict) -> List[TradingOpportunity]:
        """Analyze opportunities using concurrent processing"""
        
        try:
            # Limit symbols to analyze based on volume and market cap
            symbols_to_analyze = [s['symbol'] for s in supported_symbols[:30]]  # Top 30 by volume
            
            # Create analysis tasks
            analysis_tasks = []
            for symbol in symbols_to_analyze:
                #symbol = symbol_info['symbol'] if isinstance(symbol_info, dict) else symbol_info
                
                # Skip if already have position
                if any(pos.symbol == symbol for pos in current_positions.values()):
                    continue
                
                task = self._analyze_single_symbol(symbol, data_fetcher)
                analysis_tasks.append(task)
            
            # Execute analysis tasks concurrently
            opportunities = []
            if analysis_tasks:
                results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.debug(f"Analysis task failed: {result}")
                        continue
                    elif result:
                        opportunities.extend(result)
            
            # Sort by composite score
            opportunities.sort(key=lambda x: x.confluence_score * x.confidence, reverse=True)
            
            # Filter by minimum score
            filtered_opportunities = [
                opp for opp in opportunities 
                if opp.confluence_score >= self.config.min_opportunity_score
            ]
            
            self.execution_metrics['opportunities_analyzed'] += len(opportunities)
            
            return filtered_opportunities[:10]  # Top 10 opportunities
            
        except Exception as e:
            logger.error(f"Error in concurrent opportunity analysis: {e}")
            return []
    
    async def _analyze_single_symbol(self, symbol: str, data_fetcher) -> List[TradingOpportunity]:
        """Analyze a single symbol for trading opportunities"""
        
        try:
            # Get data with indicators
            df = data_fetcher.get_historical_data(symbol, '1h', '7 days')
            if df is None or len(df) < 100:
                return []
            
            df_with_indicators = self.indicator_calculator.calculate_all_indicators(df)
            if df_with_indicators is None:
                return []
            
            # Generate buy signals
            opportunities = self.market_analyzer.generate_buy_signals(
                symbol, df_with_indicators, self.ml_engine
            )
            
            # Enhance opportunities with ML predictions
            for opportunity in opportunities:
                try:
                    # Create features for ML prediction
                    features = self.ml_engine.create_advanced_features(df_with_indicators)
                    if not features.empty:
                        ml_prediction, ml_confidence = self.ml_engine.predict_ensemble(features.iloc[[-1]])
                        
                        # Adjust opportunity based on ML prediction
                        if ml_prediction == 1:  # ML agrees with buy signal
                            opportunity.confidence *= (1 + ml_confidence * 0.3)
                            opportunity.supporting_factors.append(f"ML model confirmation (conf: {ml_confidence:.2f})")
                        elif ml_prediction == -1:  # ML disagrees
                            opportunity.confidence *= (1 - ml_confidence * 0.2)
                            opportunity.risk_factors.append(f"ML model disagreement (conf: {ml_confidence:.2f})")
                        
                        # Store ML prediction for later evaluation
                        opportunity.ml_prediction = ml_prediction
                        opportunity.ml_confidence = ml_confidence
                        
                        self.execution_metrics['ml_predictions_made'] += 1
                        
                except Exception as e:
                    logger.debug(f"Error in ML prediction for {symbol}: {e}")
            
            return opportunities
            
        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
            return []
    
    async def _assess_portfolio_risk(self, positions: Dict, market_data: Dict) -> Dict:
        """Assess comprehensive portfolio risk"""
        
        try:
            if not positions:
                return {
                    'portfolio_heat': 0,
                    'correlation_risk': 0,
                    'var_95': 0,
                    'risk_budget_used': 0,
                    'risk_warnings': []
                }
            
            # Calculate portfolio heat
            portfolio_heat = sum(pos.value for pos in positions.values()) / self.current_equity
            
            # Estimate correlation risk (simplified)
            symbols = list(positions.keys())
            correlation_risk = 0
            if len(symbols) > 1:
                # Simple correlation estimation based on asset types
                crypto_majors = sum(1 for symbol in symbols if any(major in symbol for major in ['BTC', 'ETH']))
                total_positions = len(symbols)
                correlation_risk = crypto_majors / total_positions if total_positions > 0 else 0
            
            # Calculate approximate VaR
            historical_returns = []
            for symbol, position in positions.items():
                if symbol in market_data:
                    returns = market_data[symbol]['close'].pct_change().dropna()
                    if len(returns) > 0:
                        position_weight = position.value / self.current_equity
                        weighted_returns = returns * position_weight
                        historical_returns.extend(weighted_returns.tolist())
            
            var_95 = abs(np.percentile(historical_returns, 5)) * self.current_equity if historical_returns else 0
            
            # Risk budget calculation
            max_risk_budget = self.current_equity * 0.1  # 10% of equity
            current_risk = var_95
            risk_budget_used = current_risk / max_risk_budget if max_risk_budget > 0 else 0
            
            # Generate risk warnings
            risk_warnings = []
            if portfolio_heat > self.config.portfolio_heat_limit:
                risk_warnings.append(f"Portfolio heat too high: {portfolio_heat:.1%}")
            
            if correlation_risk > self.config.correlation_limit:
                risk_warnings.append(f"High correlation risk: {correlation_risk:.1%}")
            
            if risk_budget_used > 0.8:
                risk_warnings.append(f"Risk budget nearly exhausted: {risk_budget_used:.1%}")
            
            self.execution_metrics['risk_checks_performed'] += 1
            
            return {
                'portfolio_heat': portfolio_heat,
                'correlation_risk': correlation_risk,
                'var_95': var_95,
                'risk_budget_used': risk_budget_used,
                'risk_warnings': risk_warnings
            }
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return {'error': str(e)}
    
    async def _execute_selected_strategies(self, opportunities: List[TradingOpportunity],
                                         portfolio_risk: Dict, data_fetcher) -> List[Dict]:
        """Execute selected trading strategies"""
        
        try:
            executed_trades = []
            
            # Check if we should stop trading
            if self._should_stop_trading(portfolio_risk):
                logger.warning("Trading halted due to risk limits")
                return executed_trades
            
            # Limit daily trades
            today_trades = len([t for t in self.analytics.trade_history 
                              if t.entry_time.date() == datetime.now().date()])
            
            if today_trades >= self.config.max_daily_trades:
                logger.info(f"Daily trade limit reached: {today_trades}")
                return executed_trades
            
            for opportunity in opportunities:
                try:
                    # Get current market data for position sizing
                    df = data_fetcher.get_historical_data(opportunity.symbol, '1h', '10 days')
                    if df is None:
                        continue
                    
                    df_with_indicators = self.indicator_calculator.calculate_all_indicators(df)
                    if df_with_indicators is None:
                        continue
                    
                    # Calculate optimal position size
                    position_rec = self.risk_manager.calculate_optimal_position_size(
                        opportunity.symbol,
                        opportunity.strength,
                        self.current_equity,
                        df_with_indicators,
                        getattr(opportunity, 'ml_confidence', 0.5),
                        opportunity.market_regime.value,
                        self.position_manager.get_open_positions(self.paper_trade)
                    )
                    
                    # Check if position size is acceptable
                    if position_rec.recommended_size_usdt < self.current_equity * 0.001:  # Too small
                        logger.debug(f"Position size too small for {opportunity.symbol}")
                        continue
                    
                    # Calculate stop loss and take profit
                    stop_loss, take_profit = self.risk_manager.calculate_dynamic_stop_loss(
                        opportunity.symbol,
                        opportunity.entry_price,
                        df_with_indicators,
                        position_rec.recommended_size_usdt,
                        opportunity.market_regime.value
                    )
                    
                    # Create trade information
                    trade_info = {
                        'symbol': opportunity.symbol,
                        'side': 'BUY',  # Buy-only strategy
                        'quantity': position_rec.recommended_size_usdt / opportunity.entry_price,
                        'entry_price': opportunity.entry_price,
                        'entry_time': datetime.now(),
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'value': position_rec.recommended_size_usdt,
                        'strategy': opportunity.signal_type,
                        'confidence_score': opportunity.confidence,
                        'signal_strength': opportunity.strength,
                        'market_regime': opportunity.market_regime.value,
                        'ml_prediction': getattr(opportunity, 'ml_prediction', None),
                        'ml_confidence': getattr(opportunity, 'ml_confidence', 0.5)
                    }
                    
                    # Add position to manager
                    strategy_type = opportunity.signal_type.lower().split('_')[0]
                    trade_id = self.position_manager.add_position(
                        trade_info, strategy_type, self.paper_trade
                    )
                    
                    if trade_id:
                        executed_trades.append({
                            'trade_id': trade_id,
                            'symbol': opportunity.symbol,
                            'strategy': opportunity.signal_type,
                            'position_size_usdt': position_rec.recommended_size_usdt,
                            'confidence': opportunity.confidence,
                            'expected_return': opportunity.expected_return,
                            'max_risk': opportunity.max_risk
                        })
                        
                        logger.info(f"Trade executed: {opportunity.symbol} - {opportunity.signal_type}")
                        self.execution_metrics['trades_executed'] += 1
                    
                    # Limit concurrent positions
                    current_positions = self.position_manager.get_open_positions(self.paper_trade)
                    if len(current_positions) >= 15:  # Max 15 concurrent positions
                        break
                    
                except Exception as e:
                    logger.error(f"Error executing trade for {opportunity.symbol}: {e}")
                    continue
            
            return executed_trades
            
        except Exception as e:
            logger.error(f"Error in strategy execution: {e}")
            return []
    
    def _should_stop_trading(self, portfolio_risk: Dict) -> bool:
        """Determine if trading should be stopped due to risk"""
        
        # Emergency stop
        if self.strategy_state['emergency_stop']:
            return True
        
        # Circuit breaker
        if self.strategy_state['circuit_breaker_active']:
            return True
        
        # Risk limits
        if portfolio_risk.get('portfolio_heat', 0) > 0.9:  # 90% of equity at risk
            return True
        
        if portfolio_risk.get('var_95', 0) > self.current_equity * 0.2:  # 20% VaR limit
            return True
        
        # Drawdown limit
        if self.current_equity < self.analytics.initial_equity * (1 - self.config.max_drawdown_limit):
            return True
        
        return False
    
    def _calculate_current_equity(self, positions: Dict, current_prices: Dict) -> float:
        """Calculate current total equity"""
        
        try:
            # Start with base equity (would be fetched from exchange in real implementation)
            equity = self.analytics.initial_equity
            
            # Add realized P&L
            equity += sum(trade.net_pnl for trade in self.analytics.trade_history)
            
            # Add unrealized P&L
            for position in positions.values():
                symbol = position.symbol
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    if position.side == 'BUY':
                        unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    equity += unrealized_pnl
            
            return equity
            
        except Exception as e:
            logger.error(f"Error calculating equity: {e}")
            return self.current_equity
    
    def _calculate_current_win_rate(self) -> float:
        """Calculate current win rate"""
        
        try:
            if not self.analytics.trade_history:
                return 0.5
            
            recent_trades = self.analytics.trade_history[-50:]  # Last 50 trades
            wins = sum(1 for trade in recent_trades if trade.net_pnl > 0)
            return wins / len(recent_trades)
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.5
    
    def _generate_strategy_recommendations(self, market_regime: MarketRegime,
                                         portfolio_risk: Dict, 
                                         performance_snapshot: PerformanceSnapshot) -> List[str]:
        """Generate strategic recommendations"""
        
        recommendations = []
        
        try:
            # Market regime recommendations
            if market_regime == MarketRegime.HIGH_VOLATILITY:
                recommendations.append("High volatility detected - consider reducing position sizes")
            elif market_regime == MarketRegime.LOW_VOLATILITY:
                recommendations.append("Low volatility - opportunity for larger positions")
            elif market_regime == MarketRegime.BULL_TREND:
                recommendations.append("Bull trend - favor momentum and trend-following strategies")
            elif market_regime == MarketRegime.BEAR_TREND:
                recommendations.append("Bear trend - focus on mean-reversion opportunities")
            
            # Risk recommendations
            if portfolio_risk.get('portfolio_heat', 0) > 0.7:
                recommendations.append("High portfolio heat - consider taking profits")
            
            if portfolio_risk.get('correlation_risk', 0) > 0.8:
                recommendations.append("High correlation risk - diversify into different sectors")
            
            # Performance recommendations
            if performance_snapshot and performance_snapshot.win_rate < 0.5:
                recommendations.append("Win rate below 50% - review strategy parameters")
            
            if performance_snapshot and performance_snapshot.sharpe_ratio < 0.5:
                recommendations.append("Low Sharpe ratio - focus on risk-adjusted returns")
            
            # ML model recommendations
            if self.ml_monitor.should_retrain():
                recommendations.append("ML model performance degraded - retrain recommended")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive system status"""
        
        try:
            current_positions = self.position_manager.get_open_positions(self.paper_trade)
            position_analytics = self.position_manager.get_position_analytics(self.paper_trade)
            dashboard_data = self.analytics.get_real_time_dashboard_data()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': {
                    'trading_active': not self.strategy_state['emergency_stop'],
                    'circuit_breaker': self.strategy_state['circuit_breaker_active'],
                    'paper_trade_mode': self.paper_trade,
                    'last_analysis': self.strategy_state['last_analysis_time']
                },
                'portfolio_status': {
                    'current_equity': self.current_equity,
                    'open_positions': len(current_positions),
                    'daily_pnl': self.daily_pnl,
                    'total_realized_pnl': sum(trade.net_pnl for trade in self.analytics.trade_history)
                },
                'execution_metrics': self.execution_metrics,
                'position_analytics': position_analytics,
                'dashboard_data': dashboard_data,
                'ml_model_status': {
                    'model_loaded': self.ml_engine.ensemble_model is not None,
                    'performance_monitoring': len(self.ml_monitor.prediction_history),
                    'last_retrain': self.strategy_state['last_ml_retrain']
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def save_strategy_state(self):
        """Save complete strategy state"""
        
        try:
            # Save all component states
            self.ml_engine.save_models()
            self.position_manager.save_position_state()
            
            # Save strategy state
            state = {
                'timestamp': datetime.now().isoformat(),
                'current_equity': self.current_equity,
                'daily_pnl': self.daily_pnl,
                'strategy_state': self.strategy_state,
                'execution_metrics': self.execution_metrics,
                'config': {
                    'strategy_weights': self.config.strategy_weights,
                    'min_opportunity_score': self.config.min_opportunity_score,
                    'paper_trade': self.paper_trade
                }
            }
            
            with open('enhanced_strategy_state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info("Enhanced strategy state saved")
            
        except Exception as e:
            logger.error(f"Error saving strategy state: {e}")
    
    def load_strategy_state(self):
        """Load complete strategy state"""
        
        try:
            with open('enhanced_strategy_state.json', 'r') as f:
                state = json.load(f)
            
            self.current_equity = state.get('current_equity', 10000.0)
            self.daily_pnl = state.get('daily_pnl', 0.0)
            self.strategy_state.update(state.get('strategy_state', {}))
            self.execution_metrics.update(state.get('execution_metrics', {}))
            
            logger.info("Enhanced strategy state loaded")
            return True
            
        except FileNotFoundError:
            logger.info("No strategy state file found")
            return False
        except Exception as e:
            logger.error(f"Error loading strategy state: {e}")
            return False
    
    def emergency_stop(self, reason: str = "Manual stop"):
        """Emergency stop all trading"""
        
        self.strategy_state['emergency_stop'] = True
        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
        
        # Save current state
        self.save_strategy_state()
    
    def reset_emergency_stop(self):
        """Reset emergency stop"""
        
        self.strategy_state['emergency_stop'] = False
        logger.info("Emergency stop reset - trading can resume")


# Default strategy configuration
DEFAULT_STRATEGY_CONFIG = StrategyConfiguration(
    strategy_weights={
        'trend_following': 0.3,
        'mean_reversion': 0.25,
        'breakout': 0.2,
        'momentum': 0.15,
        'accumulation': 0.1
    },
    max_concurrent_analysis=8,
    analysis_timeout_seconds=45,
    min_opportunity_score=0.65,
    max_daily_trades=15,
    portfolio_heat_limit=0.75,
    correlation_limit=0.7,
    max_drawdown_limit=0.12,
    ml_confidence_threshold=0.6,
    model_retrain_threshold=0.55,
    ensemble_min_agreement=0.7,
    regime_strategy_multipliers={
        'BULL_TREND': {
            'trend_following': 1.3,
            'momentum': 1.2,
            'breakout': 1.1,
            'mean_reversion': 0.7,
            'accumulation': 0.9
        },
        'BEAR_TREND': {
            'trend_following': 0.6,
            'momentum': 0.7,
            'breakout': 0.8,
            'mean_reversion': 1.3,
            'accumulation': 1.2
        },
        'HIGH_VOLATILITY': {
            'trend_following': 0.8,
            'momentum': 0.9,
            'breakout': 1.3,
            'mean_reversion': 1.1,
            'accumulation': 0.7
        },
        'LOW_VOLATILITY': {
            'trend_following': 1.1,
            'momentum': 1.0,
            'breakout': 0.8,
            'mean_reversion': 0.9,
            'accumulation': 1.2
        }
    }
)