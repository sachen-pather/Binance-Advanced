"""
Advanced Risk Management with Kelly Criterion, Portfolio Optimization,
and Sophisticated Position Sizing for Cryptocurrency Trading.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

from config import RISK_PARAMS

logger = logging.getLogger("BinanceTrading.AdvancedRiskManager")


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio and individual positions"""
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    expected_shortfall: float  # Conditional VaR
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    beta: float  # Beta vs market (BTC)
    correlation_with_market: float


@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation with detailed analysis"""
    symbol: str
    recommended_size_usdt: float
    recommended_size_pct: float
    kelly_fraction: float
    risk_adjusted_size: float
    confidence_level: float
    max_position_risk: float
    expected_return: float
    expected_volatility: float
    rationale: List[str]
    warnings: List[str]


class AdvancedRiskManager:
    """Advanced risk management with portfolio optimization and Kelly Criterion"""
    
    def __init__(self, risk_params=None, initial_capital=10000):
        self.risk_params = risk_params or RISK_PARAMS
        self.initial_capital = initial_capital
        self.position_history = []
        self.return_history = []
        self.correlation_matrix = pd.DataFrame()
        self.market_beta = {}
        
        # Risk model parameters
        self.risk_model_params = {
            'lookback_days': 90,
            'confidence_levels': [0.95, 0.99],
            'rebalance_threshold': 0.05,
            'max_portfolio_risk': 0.15,  # 15% portfolio VaR
            'max_single_position': 0.10,  # 10% max single position
            'max_sector_exposure': 0.30,  # 30% max sector exposure
            'correlation_threshold': 0.7,
            'kelly_multiplier': 0.25,  # Use 25% of full Kelly
            'min_sharpe_ratio': 0.5,
            'max_leverage': 1.0  # No leverage for spot trading
        }
        
        # Portfolio optimization weights
        self.optimization_weights = {
            'return': 0.4,
            'risk': 0.3,
            'diversification': 0.2,
            'momentum': 0.1
        }
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float, 
                                 confidence_adjustment: float = 1.0) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        
        try:
            if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = loss_rate
            b = avg_win / avg_loss  # Odds ratio
            p = win_rate  # Win probability
            q = 1 - p  # Loss probability
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety multiplier and confidence adjustment
            adjusted_kelly = kelly_fraction * self.risk_model_params['kelly_multiplier'] * confidence_adjustment
            
            # Ensure positive and reasonable bounds
            return max(0, min(adjusted_kelly, 0.15))  # Cap at 15%
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {e}")
            return 0.02  # Conservative fallback
    
    def calculate_portfolio_var(self, positions: Dict, returns_data: Dict, 
                           confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate portfolio Value at Risk with proper object handling"""
        
        try:
            if not positions or not returns_data:
                return 0.0, 0.0
            
            # Extract position values and symbols with object handling
            symbols = []
            position_values = []
            
            for symbol, pos in positions.items():
                # Handle different position object types
                if hasattr(pos, 'value'):
                    value = pos.value
                elif hasattr(pos, 'size') and hasattr(pos, 'entry_price'):
                    value = abs(pos.size * pos.entry_price)
                elif isinstance(pos, dict):
                    value = pos.get('value', 0)
                else:
                    value = getattr(pos, 'current_value', 0) or getattr(pos, 'market_value', 0)
                
                if value > 0:
                    symbols.append(symbol)
                    position_values.append(value)
            
            if not position_values:
                return 0.0, 0.0
            
            position_values = np.array(position_values)
            total_portfolio_value = np.sum(position_values)
            
            if total_portfolio_value == 0:
                return 0.0, 0.0
            
            # Rest of the method remains the same...
            weights = position_values / total_portfolio_value
            
            # Get returns data for symbols
            returns_matrix = []
            common_symbols = []
            
            for symbol in symbols:
                if symbol in returns_data and len(returns_data[symbol]) > 30:
                    returns_matrix.append(returns_data[symbol][-252:])
                    common_symbols.append(symbol)
            
            if len(returns_matrix) < 2:
                portfolio_returns = np.concatenate(returns_matrix) if returns_matrix else np.array([0])
                var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
                return abs(var * total_portfolio_value), total_portfolio_value * 0.02
            
            # Create returns DataFrame
            min_length = min(len(returns) for returns in returns_matrix)
            returns_df = pd.DataFrame({
                symbol: returns[-min_length:] 
                for symbol, returns in zip(common_symbols, returns_matrix)
            })
            
            # Calculate covariance matrix with shrinkage
            cov_estimator = LedoitWolf()
            cov_matrix = cov_estimator.fit(returns_df).covariance_
            
            # Monte Carlo simulation
            n_simulations = 10000
            portfolio_returns = []
            
            mean_returns = returns_df.mean().values
            
            # Adjust weights for common symbols only
            weight_mapping = {symbol: weights[symbols.index(symbol)] 
                            for symbol in common_symbols if symbol in symbols}
            adjusted_weights = np.array([weight_mapping.get(symbol, 0) for symbol in common_symbols])
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights) if np.sum(adjusted_weights) > 0 else adjusted_weights
            
            for _ in range(n_simulations):
                random_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
                portfolio_return = np.dot(adjusted_weights, random_returns)
                portfolio_returns.append(portfolio_return)
            
            portfolio_returns = np.array(portfolio_returns)
            
            # Calculate VaR
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            expected_shortfall = np.mean(portfolio_returns[portfolio_returns <= var])
            
            # Convert to dollar amounts
            var_dollar = abs(var * total_portfolio_value)
            es_dollar = abs(expected_shortfall * total_portfolio_value)
            
            return var_dollar, es_dollar
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            total_value = 0
            try:
                for pos in positions.values():
                    if hasattr(pos, 'value'):
                        total_value += pos.value
                    elif hasattr(pos, 'size') and hasattr(pos, 'entry_price'):
                        total_value += abs(pos.size * pos.entry_price)
            except:
                total_value = 10000  # Fallback
            return total_value * 0.05, total_value * 0.08
    
    def calculate_optimal_position_size(self, symbol: str, signal_strength: float, 
                                       current_equity: float, historical_data: pd.DataFrame,
                                       ml_confidence: float = 0.5, market_regime: str = 'UNKNOWN',
                                       existing_positions: Dict = None) -> PositionSizeRecommendation:
        """Calculate optimal position size using multiple risk models"""
        
        try:
            existing_positions = existing_positions or {}
            
            # Calculate basic statistics
            returns = historical_data['close'].pct_change().dropna()
            if len(returns) < 30:
                logger.warning(f"Insufficient data for {symbol} position sizing")
                return PositionSizeRecommendation(
                    symbol=symbol,
                    recommended_size_usdt=current_equity * 0.01,
                    recommended_size_pct=1.0,
                    kelly_fraction=0.01,
                    risk_adjusted_size=current_equity * 0.01,
                    confidence_level=0.3,
                    max_position_risk=0.02,
                    expected_return=0.05,
                    expected_volatility=0.3,
                    rationale=["Insufficient historical data"],
                    warnings=["Using conservative default sizing"]
                )
            
            # Historical performance metrics
            win_rate = len(returns[returns > 0]) / len(returns)
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.01
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.01
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Kelly Criterion calculation
            kelly_fraction = self.calculate_kelly_criterion(
                win_rate, avg_win, avg_loss, 
                confidence_adjustment=ml_confidence
            )
            
            # Volatility-adjusted sizing
            target_volatility = 0.15  # 15% target volatility
            vol_adjustment = min(target_volatility / volatility, 2.0) if volatility > 0 else 1.0
            
            # Signal strength adjustment
            signal_adjustment = min(signal_strength * 1.5, 2.0)
            
            # Market regime adjustment
            regime_multipliers = {
                'BULL_TREND': 1.3,
                'BEAR_TREND': 0.6,
                'RANGE_BOUND': 0.8,
                'HIGH_VOLATILITY': 0.7,
                'LOW_VOLATILITY': 1.1,
                'BREAKOUT': 1.2,
                'ACCUMULATION': 0.9,
                'UNKNOWN': 0.7
            }
            regime_adjustment = regime_multipliers.get(market_regime, 0.7)
            
            # Portfolio heat adjustment (reduce size if portfolio is already risky)
            portfolio_heat = self._calculate_portfolio_heat(existing_positions, current_equity)
            heat_adjustment = max(0.3, 1.0 - portfolio_heat)
            
            # Correlation adjustment
            correlation_adjustment = self._calculate_correlation_adjustment(
                symbol, existing_positions, historical_data
            )
            
            # Combine all adjustments
            base_size_pct = kelly_fraction
            adjusted_size_pct = (
                base_size_pct * 
                vol_adjustment * 
                signal_adjustment * 
                regime_adjustment * 
                heat_adjustment * 
                correlation_adjustment
            )
            
            # Apply limits
            max_position_pct = min(
                self.risk_model_params['max_single_position'],
                self.risk_params['max_position_size']
            )
            
            final_size_pct = min(adjusted_size_pct, max_position_pct)
            final_size_pct = max(final_size_pct, self.risk_params['min_position_size'])
            
            # Convert to dollar amount
            recommended_size_usdt = final_size_pct * current_equity
            
            # Risk metrics
            position_var = recommended_size_usdt * volatility * 2.33  # 99% VaR approximation
            expected_return = returns.mean() * 252  # Annualized
            
            # Build rationale
            rationale = [
                f"Kelly fraction: {kelly_fraction:.3f} ({win_rate:.1%} win rate)",
                f"Volatility adjustment: {vol_adjustment:.2f}x (vol: {volatility:.1%})",
                f"Signal strength: {signal_adjustment:.2f}x",
                f"Market regime ({market_regime}): {regime_adjustment:.2f}x",
                f"Portfolio heat: {heat_adjustment:.2f}x",
                f"Correlation: {correlation_adjustment:.2f}x"
            ]
            
            # Warnings
            warnings = []
            if final_size_pct >= max_position_pct:
                warnings.append(f"Position capped at maximum size ({max_position_pct:.1%})")
            if volatility > 0.5:
                warnings.append("High volatility asset - increased caution")
            if portfolio_heat > 0.3:
                warnings.append("Portfolio already has significant risk exposure")
            if correlation_adjustment < 0.8:
                warnings.append("High correlation with existing positions")
            
            return PositionSizeRecommendation(
                symbol=symbol,
                recommended_size_usdt=recommended_size_usdt,
                recommended_size_pct=final_size_pct,
                kelly_fraction=kelly_fraction,
                risk_adjusted_size=recommended_size_usdt,
                confidence_level=min(signal_strength * ml_confidence, 1.0),
                max_position_risk=position_var,
                expected_return=expected_return,
                expected_volatility=volatility,
                rationale=rationale,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            # Conservative fallback
            return PositionSizeRecommendation(
                symbol=symbol,
                recommended_size_usdt=current_equity * 0.02,
                recommended_size_pct=0.02,
                kelly_fraction=0.02,
                risk_adjusted_size=current_equity * 0.02,
                confidence_level=0.3,
                max_position_risk=current_equity * 0.01,
                expected_return=0.1,
                expected_volatility=0.3,
                rationale=["Error in calculation - using conservative default"],
                warnings=["Position sizing calculation failed"]
            )
    
    def _calculate_portfolio_heat(self, positions: Dict, current_equity: float) -> float:
        """Calculate portfolio heat (risk exposure) with proper object handling"""
        
        try:
            if not positions or current_equity == 0:
                return 0.0
            
            total_position_value = 0
            risk_weighted_exposure = 0
            
            for pos in positions.values():
                # Handle both dict and object types
                if hasattr(pos, 'value'):
                    position_value = pos.value
                elif hasattr(pos, 'size') and hasattr(pos, 'entry_price'):
                    position_value = abs(pos.size * pos.entry_price)
                elif isinstance(pos, dict):
                    position_value = pos.get('value', 0)
                else:
                    # Fallback - try to get value from common attributes
                    position_value = getattr(pos, 'current_value', 0) or getattr(pos, 'market_value', 0)
                
                total_position_value += position_value
                
                # Calculate risk multiplier
                position_weight = position_value / current_equity if current_equity > 0 else 0
                
                # Check for stop loss (handle both dict and object)
                has_stop_loss = False
                if hasattr(pos, 'stop_loss'):
                    has_stop_loss = pos.stop_loss is not None
                elif hasattr(pos, 'stop_price'):
                    has_stop_loss = pos.stop_price is not None
                elif isinstance(pos, dict):
                    has_stop_loss = 'stop_loss' in pos and pos['stop_loss'] is not None
                
                risk_multiplier = 1.0 if has_stop_loss else 1.5
                risk_weighted_exposure += position_weight * risk_multiplier
            
            portfolio_exposure = total_position_value / current_equity if current_equity > 0 else 0
            return min(risk_weighted_exposure, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio heat: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return 0.3  # Conservative assumption
    
    def _calculate_correlation_adjustment(self, symbol: str, existing_positions: Dict, 
                                        historical_data: pd.DataFrame) -> float:
        """Calculate position size adjustment based on correlation with existing positions"""
        
        try:
            if not existing_positions:
                return 1.0
            
            # This is a simplified correlation calculation
            # In practice, you'd want to calculate actual price correlations
            
            # Extract base assets
            base_asset = symbol.replace('USDT', '').replace('BTC', '').replace('ETH', '')
            
            # Check for similar assets in portfolio
            similar_assets = 0
            total_similar_exposure = 0
            
            for pos_symbol in existing_positions.keys():
                pos_base = pos_symbol.replace('USDT', '').replace('BTC', '').replace('ETH', '')
                
                # Simple correlation heuristics
                if base_asset == pos_base:
                    return 0.1  # Same asset - minimal additional exposure
                
                # Check for ecosystem correlation
                ecosystem_groups = {
                    'defi': ['UNI', 'SUSHI', 'CAKE', 'AAVE', 'COMP'],
                    'layer1': ['ETH', 'BNB', 'ADA', 'SOL', 'AVAX', 'DOT'],
                    'meme': ['DOGE', 'SHIB', 'PEPE'],
                    'exchange': ['BNB', 'FTT', 'CRO', 'LEO']
                }
                
                for ecosystem, assets in ecosystem_groups.items():
                    if base_asset in assets and pos_base in assets:
                        similar_assets += 1
                        total_similar_exposure += existing_positions[pos_symbol].get('value', 0)
            
            if similar_assets == 0:
                return 1.0
            
            # Reduce position size based on similar exposure
            correlation_adjustment = max(0.3, 1.0 - (similar_assets * 0.2))
            
            return correlation_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment: {e}")
            return 0.8  # Conservative adjustment
    
    def calculate_dynamic_stop_loss(self, symbol: str, entry_price: float, 
                                   historical_data: pd.DataFrame, position_size: float,
                                   market_regime: str = 'UNKNOWN') -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels"""
        
        try:
            # Calculate ATR for volatility-based stops
            if 'ATR_14' in historical_data.columns:
                atr = historical_data['ATR_14'].iloc[-1]
            else:
                # Fallback ATR calculation
                high_low = historical_data['high'] - historical_data['low']
                high_close = abs(historical_data['high'] - historical_data['close'].shift(1))
                low_close = abs(historical_data['low'] - historical_data['close'].shift(1))
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                atr = true_range.rolling(14).mean().iloc[-1]
            
            # Regime-based multipliers
            stop_multipliers = {
                'BULL_TREND': {'stop': 2.0, 'profit': 3.0},
                'BEAR_TREND': {'stop': 1.5, 'profit': 2.0},
                'RANGE_BOUND': {'stop': 1.2, 'profit': 1.8},
                'HIGH_VOLATILITY': {'stop': 2.5, 'profit': 3.5},
                'LOW_VOLATILITY': {'stop': 1.5, 'profit': 2.5},
                'BREAKOUT': {'stop': 1.8, 'profit': 4.0},
                'ACCUMULATION': {'stop': 2.2, 'profit': 3.2},
                'UNKNOWN': {'stop': 1.8, 'profit': 2.5}
            }
            
            multipliers = stop_multipliers.get(market_regime, stop_multipliers['UNKNOWN'])
            
            # Calculate levels
            stop_distance = atr * multipliers['stop']
            profit_distance = atr * multipliers['profit']
            
            # For buy positions
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
            
            # Risk-based adjustment
            max_risk_pct = self.risk_params.get('stop_loss_pct', 0.03)
            max_risk_amount = entry_price * max_risk_pct
            
            if stop_distance > max_risk_amount:
                stop_loss = entry_price - max_risk_amount
                # Adjust take profit proportionally
                risk_reward_ratio = profit_distance / stop_distance
                new_profit_distance = max_risk_amount * risk_reward_ratio
                take_profit = entry_price + new_profit_distance
            
            # Ensure positive values
            stop_loss = max(stop_loss, entry_price * 0.85)  # Max 15% stop loss
            take_profit = max(take_profit, entry_price * 1.02)  # Min 2% profit target
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating dynamic stops for {symbol}: {e}")
            # Fallback calculation
            stop_loss = entry_price * (1 - self.risk_params.get('stop_loss_pct', 0.03))
            take_profit = entry_price * (1 + self.risk_params.get('take_profit_pct', 0.05))
            return stop_loss, take_profit
    
    def calculate_portfolio_metrics(self, positions: Dict, historical_returns: Dict,
                                   current_equity: float) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        
        try:
            if not positions:
                return RiskMetrics(
                    var_95=0, var_99=0, expected_shortfall=0, max_drawdown=0,
                    sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                    volatility=0, beta=0, correlation_with_market=0
                )
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(positions, historical_returns)
            
            if len(portfolio_returns) < 30:
                logger.warning("Insufficient data for portfolio metrics")
                return RiskMetrics(
                    var_95=current_equity * 0.05, var_99=current_equity * 0.08,
                    expected_shortfall=current_equity * 0.1, max_drawdown=0.1,
                    sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                    volatility=0.2, beta=1.0, correlation_with_market=0.5
                )
            
            # VaR calculations
            var_95 = abs(np.percentile(portfolio_returns, 5)) * current_equity
            var_99 = abs(np.percentile(portfolio_returns, 1)) * current_equity
            
            # Expected Shortfall (Conditional VaR)
            var_95_threshold = np.percentile(portfolio_returns, 5)
            expected_shortfall = abs(np.mean(portfolio_returns[portfolio_returns <= var_95_threshold])) * current_equity
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Performance ratios
            mean_return = portfolio_returns.mean() * 252  # Annualized
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
            
            # Calmar ratio
            calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0
            
            # Beta calculation (vs BTC if available)
            beta = 1.0
            correlation_with_market = 0.5
            
            if 'BTCUSDT' in historical_returns:
                btc_returns = pd.Series(historical_returns['BTCUSDT'][-len(portfolio_returns):])
                if len(btc_returns) == len(portfolio_returns):
                    covariance = np.cov(portfolio_returns, btc_returns)[0, 1]
                    btc_variance = np.var(btc_returns)
                    beta = covariance / btc_variance if btc_variance > 0 else 1.0
                    correlation_with_market = np.corrcoef(portfolio_returns, btc_returns)[0, 1]
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                volatility=volatility,
                beta=beta,
                correlation_with_market=correlation_with_market
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return RiskMetrics(
                var_95=current_equity * 0.05, var_99=current_equity * 0.08,
                expected_shortfall=current_equity * 0.1, max_drawdown=0.1,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                volatility=0.2, beta=1.0, correlation_with_market=0.5
            )
            
    def _get_position_value(self, position) -> float:
        """Helper method to extract position value from different position object types"""
        try:
            if hasattr(position, 'value'):
                return position.value
            elif hasattr(position, 'size') and hasattr(position, 'entry_price'):
                return abs(position.size * position.entry_price)
            elif hasattr(position, 'current_value'):
                return position.current_value
            elif hasattr(position, 'market_value'):
                return position.market_value
            elif isinstance(position, dict):
                return position.get('value', 0)
            else:
                logger.warning(f"Unknown position type: {type(position)}")
                return 0.0
        except Exception as e:
            logger.error(f"Error extracting position value: {e}")
            return 0.0
    
    def _calculate_portfolio_returns(self, positions: Dict, historical_returns: Dict) -> np.array:
        """Calculate historical portfolio returns with proper object handling"""
        
        try:
            if not positions or not historical_returns:
                return np.array([])
            
            # Get position weights with object handling
            total_value = 0
            position_values = {}
            
            for symbol, pos in positions.items():
                if hasattr(pos, 'value'):
                    value = pos.value
                elif hasattr(pos, 'size') and hasattr(pos, 'entry_price'):
                    value = abs(pos.size * pos.entry_price)
                elif isinstance(pos, dict):
                    value = pos.get('value', 0)
                else:
                    value = getattr(pos, 'current_value', 0) or getattr(pos, 'market_value', 0)
                
                position_values[symbol] = value
                total_value += value
            
            if total_value == 0:
                return np.array([])
            
            weights = {symbol: value / total_value for symbol, value in position_values.items()}
            
            # Find common time period
            min_length = min(len(returns) for symbol, returns in historical_returns.items() 
                        if symbol in weights and len(returns) > 0)
            
            if min_length < 10:
                return np.array([])
            
            # Calculate weighted portfolio returns
            portfolio_returns = np.zeros(min_length)
            
            for symbol, weight in weights.items():
                if symbol in historical_returns and weight > 0:
                    symbol_returns = np.array(historical_returns[symbol][-min_length:])
                    portfolio_returns += weight * symbol_returns
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return np.array([])
    
    def check_risk_limits(self, current_equity: float, daily_pnl: float, 
                     positions: Dict, portfolio_var: float) -> Tuple[bool, List[str]]:
        """Comprehensive risk limit checking with proper object handling"""
        
        try:
            violations = []
            
            # Daily loss limit
            daily_loss_limit = self.risk_params['max_daily_loss'] * self.initial_capital
            if daily_pnl < -daily_loss_limit:
                violations.append(f"Daily loss limit exceeded: {daily_pnl:.2f} < {-daily_loss_limit:.2f}")
            
            # Portfolio VaR limit
            var_limit = self.risk_model_params['max_portfolio_risk'] * current_equity
            if portfolio_var > var_limit:
                violations.append(f"Portfolio VaR limit exceeded: {portfolio_var:.2f} > {var_limit:.2f}")
            
            # Maximum positions limit
            if len(positions) > self.risk_params['max_open_positions']:
                violations.append(f"Too many open positions: {len(positions)} > {self.risk_params['max_open_positions']}")
            
            # Single position size limits
            for symbol, position in positions.items():
                # Handle position value extraction
                if hasattr(position, 'value'):
                    position_value = position.value
                elif hasattr(position, 'size') and hasattr(position, 'entry_price'):
                    position_value = abs(position.size * position.entry_price)
                elif isinstance(position, dict):
                    position_value = position.get('value', 0)
                else:
                    position_value = getattr(position, 'current_value', 0) or getattr(position, 'market_value', 0)
                
                position_pct = position_value / current_equity if current_equity > 0 else 0
                
                if position_pct > self.risk_model_params['max_single_position']:
                    violations.append(f"{symbol} position too large: {position_pct:.1%} > {self.risk_model_params['max_single_position']:.1%}")
            
            # Drawdown limit
            if current_equity < self.initial_capital * 0.8:
                violations.append(f"Portfolio drawdown limit exceeded: {(1 - current_equity/self.initial_capital):.1%}")
            
            # Concentration risk
            if len(positions) > 0:
                largest_position = 0
                for pos in positions.values():
                    if hasattr(pos, 'value'):
                        value = pos.value
                    elif hasattr(pos, 'size') and hasattr(pos, 'entry_price'):
                        value = abs(pos.size * pos.entry_price)
                    elif isinstance(pos, dict):
                        value = pos.get('value', 0)
                    else:
                        value = getattr(pos, 'current_value', 0) or getattr(pos, 'market_value', 0)
                    
                    largest_position = max(largest_position, value)
                
                concentration_ratio = largest_position / current_equity if current_equity > 0 else 0
                
                if concentration_ratio > 0.25:
                    violations.append(f"Concentration risk: largest position is {concentration_ratio:.1%} of portfolio")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, ["Error in risk limit calculation"]
    
    def suggest_portfolio_rebalancing(self, positions: Dict, target_allocations: Dict,
                                    current_equity: float) -> List[Dict]:
        """Suggest portfolio rebalancing actions"""
        
        try:
            if not positions or current_equity == 0:
                return []
            
            rebalancing_actions = []
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            
            # Calculate current allocations
            current_allocations = {
                symbol: pos.get('value', 0) / total_value 
                for symbol, pos in positions.items()
            }
            
            # Compare with targets
            for symbol, target_pct in target_allocations.items():
                current_pct = current_allocations.get(symbol, 0)
                difference = target_pct - current_pct
                
                # Only suggest rebalancing if difference is significant
                if abs(difference) > self.risk_model_params['rebalance_threshold']:
                    action_type = 'BUY' if difference > 0 else 'REDUCE'
                    amount = abs(difference) * current_equity
                    
                    rebalancing_actions.append({
                        'symbol': symbol,
                        'action': action_type,
                        'amount_usdt': amount,
                        'current_allocation': current_pct,
                        'target_allocation': target_pct,
                        'difference': difference,
                        'priority': abs(difference)  # Higher difference = higher priority
                    })
            
            # Sort by priority
            rebalancing_actions.sort(key=lambda x: x['priority'], reverse=True)
            
            return rebalancing_actions
            
        except Exception as e:
            logger.error(f"Error suggesting rebalancing: {e}")
            return []
    
    def generate_risk_report(self, current_equity: float, positions: Dict, 
                           historical_returns: Dict, daily_pnl: float) -> Dict:
        """Generate comprehensive risk report"""
        
        try:
            # Calculate portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics(
                positions, historical_returns, current_equity
            )
            
            # Calculate portfolio VaR
            portfolio_var, expected_shortfall = self.calculate_portfolio_var(
                positions, historical_returns
            )
            
            # Check risk limits
            risk_limits_ok, violations = self.check_risk_limits(
                current_equity, daily_pnl, positions, portfolio_var
            )
            
            # Position-level analysis
            position_analysis = []
            for symbol, position in positions.items():
                position_value = position.value
                position_pct = position_value / current_equity if current_equity > 0 else 0
                
                # Estimate position VaR (simplified)
                if symbol in historical_returns and len(historical_returns[symbol]) > 30:
                    returns = np.array(historical_returns[symbol][-30:])
                    position_var = abs(np.percentile(returns, 5)) * position_value
                else:
                    position_var = position_value * 0.05  # 5% default
                
                position_analysis.append({
                    'symbol': symbol,
                    'value': position_value,
                    'allocation': position_pct,
                    'var_95': position_var,
                    'contribution_to_var': position_var / portfolio_var if portfolio_var > 0 else 0
                })
            
            # Risk concentration analysis
            risk_concentration = {
                'largest_position': max([pa['allocation'] for pa in position_analysis], default=0),
                'top_3_concentration': sum(sorted([pa['allocation'] for pa in position_analysis], reverse=True)[:3]),
                'effective_positions': len([pa for pa in position_analysis if pa['allocation'] > 0.02])  # Positions > 2%
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': current_equity,
                'daily_pnl': daily_pnl,
                'portfolio_metrics': {
                    'var_95': portfolio_metrics.var_95,
                    'var_99': portfolio_metrics.var_99,
                    'expected_shortfall': portfolio_metrics.expected_shortfall,
                    'max_drawdown': portfolio_metrics.max_drawdown,
                    'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                    'sortino_ratio': portfolio_metrics.sortino_ratio,
                    'volatility': portfolio_metrics.volatility,
                    'beta': portfolio_metrics.beta
                },
                'risk_limits': {
                    'limits_ok': risk_limits_ok,
                    'violations': violations
                },
                'position_analysis': position_analysis,
                'risk_concentration': risk_concentration,
                'recommendations': self._generate_risk_recommendations(
                    portfolio_metrics, risk_limits_ok, violations, risk_concentration
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'portfolio_value': current_equity,
                'daily_pnl': daily_pnl
            }
    
    def _generate_risk_recommendations(self, portfolio_metrics: RiskMetrics, 
                                     risk_limits_ok: bool, violations: List[str],
                                     risk_concentration: Dict) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        
        try:
            # Risk limit violations
            if not risk_limits_ok:
                recommendations.append("⚠️ Risk limit violations detected - consider reducing positions")
            
            # High concentration
            if risk_concentration['largest_position'] > 0.2:
                recommendations.append("📊 High position concentration - consider diversification")
            
            # Poor risk-adjusted returns
            if portfolio_metrics.sharpe_ratio < 0.5:
                recommendations.append("📈 Low Sharpe ratio - review strategy performance")
            
            # High volatility
            if portfolio_metrics.volatility > 0.4:
                recommendations.append("📉 High portfolio volatility - consider position size reduction")
            
            # High correlation with market
            if portfolio_metrics.correlation_with_market > 0.8:
                recommendations.append("🔗 High market correlation - diversify into uncorrelated assets")
            
            # Large drawdown
            if portfolio_metrics.max_drawdown > 0.15:
                recommendations.append("⬇️ Large historical drawdown - review risk management")
            
            # High VaR
            if portfolio_metrics.var_95 > portfolio_metrics.volatility * 0.1:  # Rough threshold
                recommendations.append("⚡ High Value at Risk - consider hedging strategies")
            
            # Insufficient diversification
            if risk_concentration['effective_positions'] < 5:
                recommendations.append("🎯 Low diversification - consider adding more positions")
            
            # Positive recommendations
            if len(recommendations) == 0:
                recommendations.append("✅ Portfolio risk profile looks healthy")
                
                if portfolio_metrics.sharpe_ratio > 1.5:
                    recommendations.append("🎉 Excellent risk-adjusted returns")
                
                if risk_concentration['effective_positions'] >= 8:
                    recommendations.append("👍 Good diversification across positions")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["⚠️ Error generating risk recommendations"]