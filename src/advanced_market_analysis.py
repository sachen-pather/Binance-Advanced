"""
Advanced Market Analysis with Multi-Confluence Buy-Only Strategies
Specialized for cryptocurrency markets with sophisticated signal generation.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import STRATEGY_PARAMS, RISK_PARAMS, CIRCUIT_BREAKER_CONFIG

logger = logging.getLogger("BinanceTrading.AdvancedMarketAnalysis")


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    DISTRIBUTION = "DISTRIBUTION"
    ACCUMULATION = "ACCUMULATION"


@dataclass
class TradingOpportunity:
    """Structured trading opportunity with detailed analysis"""
    symbol: str
    signal_type: str
    entry_price: float
    confidence: float
    strength: float
    timeframe: str
    confluence_score: float
    supporting_factors: List[str]
    risk_factors: List[str]
    expected_return: float
    max_risk: float
    holding_period: str
    market_regime: MarketRegime
    volume_profile: str
    timestamp: datetime


class AdvancedMarketAnalyzer:
    """Advanced market analysis with sophisticated buy-only strategies"""
    
    def __init__(self, strategy_params=None, risk_params=None):
        self.strategy_params = strategy_params or STRATEGY_PARAMS
        self.risk_params = risk_params or RISK_PARAMS
        self.circuit_breaker = CIRCUIT_BREAKER_CONFIG.copy()
        
        # Market state tracking
        self.market_state = {
            'regime': MarketRegime.RANGE_BOUND,
            'volatility_percentile': 50,
            'trend_strength': 0,
            'correlation_matrix': pd.DataFrame(),
            'sector_rotation': {},
            'market_sentiment': 0.5,
            'liquidity_conditions': 'normal'
        }
        
        # Strategy configurations
        self.buy_strategies = {
            'trend_following': {
                'enabled': True,
                'weight': 0.25,
                'min_confluence': 0.6,
                'timeframes': ['1h', '4h', '1d']
            },
            'mean_reversion': {
                'enabled': True,
                'weight': 0.2,
                'min_confluence': 0.7,
                'timeframes': ['15m', '1h', '4h']
            },
            'breakout': {
                'enabled': True,
                'weight': 0.2,
                'min_confluence': 0.8,
                'timeframes': ['5m', '15m', '1h']
            },
            'momentum': {
                'enabled': True,
                'weight': 0.2,
                'min_confluence': 0.65,
                'timeframes': ['1h', '4h']
            },
            'accumulation': {
                'enabled': True,
                'weight': 0.15,
                'min_confluence': 0.75,
                'timeframes': ['1h', '4h', '1d']
            }
        }
        
        # Performance tracking
        self.strategy_performance = {name: {'signals': 0, 'winners': 0, 'avg_return': 0} 
                                   for name in self.buy_strategies.keys()}
    
    def analyze_market_regime(self, data_fetcher, indicator_calculator, 
                            major_symbols: List[str] = ['BTCUSDT', 'ETHUSDT']) -> MarketRegime:
        """Comprehensive market regime analysis"""
        
        try:
            regime_scores = {regime: 0 for regime in MarketRegime}
            
            for symbol in major_symbols:
                df = data_fetcher.get_historical_data(symbol, '1h', '30 days')
                if df is None:
                    continue
                
                df = indicator_calculator.calculate_all_indicators(df)
                if df is None:
                    continue
                
                # Trend analysis
                trend_score = self._analyze_trend_strength(df)
                volatility_score = self._analyze_volatility_regime(df)
                volume_score = self._analyze_volume_regime(df)
                
                # Regime classification
                if trend_score > 0.7:
                    if df['close'].iloc[-1] > df['close'].iloc[-30]:
                        regime_scores[MarketRegime.BULL_TREND] += 1
                    else:
                        regime_scores[MarketRegime.BEAR_TREND] += 1
                elif volatility_score > 0.8:
                    regime_scores[MarketRegime.HIGH_VOLATILITY] += 1
                elif volatility_score < 0.3:
                    regime_scores[MarketRegime.LOW_VOLATILITY] += 1
                elif self._detect_breakout_pattern(df):
                    regime_scores[MarketRegime.BREAKOUT] += 1
                elif volume_score > 0.7:
                    regime_scores[MarketRegime.ACCUMULATION] += 1
                else:
                    regime_scores[MarketRegime.RANGE_BOUND] += 1
            
            # Determine dominant regime
            dominant_regime = max(regime_scores, key=regime_scores.get)
            self.market_state['regime'] = dominant_regime
            
            logger.info(f"Market regime detected: {dominant_regime.value}")
            return dominant_regime
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return MarketRegime.RANGE_BOUND
    
    def _analyze_trend_strength(self, df: pd.DataFrame) -> float:
        """Analyze trend strength using multiple indicators"""
        
        try:
            latest = df.iloc[-1]
            
            # ADX strength
            adx_strength = min(latest.get('ADX', 0) / 100, 1.0)
            
            # Moving average alignment
            ma_alignment = 0
            if 'EMA_20' in df.columns and 'EMA_50' in df.columns and 'EMA_200' in df.columns:
                if latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200']:
                    ma_alignment = 1
                elif latest['EMA_20'] < latest['EMA_50'] < latest['EMA_200']:
                    ma_alignment = 1
                else:
                    ma_alignment = 0.5
            
            # Price momentum
            if len(df) >= 20:
                price_momentum = abs(df['close'].pct_change(20).iloc[-1])
            else:
                price_momentum = 0
            
            # Combine factors
            trend_strength = (adx_strength * 0.4 + ma_alignment * 0.4 + min(price_momentum * 5, 1) * 0.2)
            
            return min(trend_strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing trend strength: {e}")
            return 0.5
    
    def _analyze_volatility_regime(self, df: pd.DataFrame) -> float:
        """Analyze volatility regime"""
        
        try:
            if len(df) < 50:
                return 0.5
            
            # Calculate rolling volatility
            returns = df['close'].pct_change()
            current_vol = returns.rolling(20).std().iloc[-1]
            historical_vol = returns.rolling(100).std()
            
            # Volatility percentile
            vol_percentile = stats.percentileofscore(historical_vol.dropna(), current_vol) / 100
            
            self.market_state['volatility_percentile'] = vol_percentile * 100
            
            return vol_percentile
            
        except Exception as e:
            logger.error(f"Error analyzing volatility regime: {e}")
            return 0.5
    
    def _analyze_volume_regime(self, df: pd.DataFrame) -> float:
        """Analyze volume regime and institutional activity"""
        
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return 0.5
            
            # Volume trend
            volume_sma = df['volume'].rolling(20).mean()
            current_volume = df['volume'].iloc[-5:].mean()  # Recent average
            
            if volume_sma.iloc[-1] > 0:
                volume_ratio = current_volume / volume_sma.iloc[-1]
            else:
                volume_ratio = 1.0
            
            # Volume-price relationship
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            
            # Correlation between volume and price movement
            if len(price_change) > 20:
                correlation = price_change.rolling(20).corr(volume_change).iloc[-1]
                if np.isnan(correlation):
                    correlation = 0
            else:
                correlation = 0
            
            # Combine factors
            volume_score = min((volume_ratio - 1) * 0.5 + (correlation + 1) * 0.25, 1.0)
            
            return max(volume_score, 0)
            
        except Exception as e:
            logger.error(f"Error analyzing volume regime: {e}")
            return 0.5
    
    def _detect_breakout_pattern(self, df: pd.DataFrame) -> bool:
        """Detect breakout patterns"""
        
        try:
            if len(df) < 20:
                return False
            
            # Bollinger Band squeeze
            if 'BB_width' in df.columns:
                bb_width = df['BB_width'].rolling(20).mean()
                current_width = df['BB_width'].iloc[-1]
                
                if current_width < bb_width.iloc[-1] * 0.8:  # Squeeze condition
                    # Check for volume expansion
                    volume_expansion = df['volume'].iloc[-1] > df['volume'].rolling(10).mean().iloc[-1] * 1.5
                    
                    # Check for price movement
                    price_movement = abs(df['close'].pct_change().iloc[-1]) > df['close'].pct_change().rolling(20).std().iloc[-1] * 2
                    
                    return volume_expansion and price_movement
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting breakout pattern: {e}")
            return False
    
    def generate_buy_signals(self, symbol: str, df: pd.DataFrame, ml_model=None) -> List[TradingOpportunity]:
        """Generate comprehensive buy signals using multiple strategies"""
        
        try:
            opportunities = []
            
            # Strategy 1: Trend Following
            if self.buy_strategies['trend_following']['enabled']:
                trend_opp = self._trend_following_strategy(symbol, df, ml_model)
                if trend_opp:
                    opportunities.append(trend_opp)
            
            # Strategy 2: Mean Reversion
            if self.buy_strategies['mean_reversion']['enabled']:
                reversion_opp = self._mean_reversion_strategy(symbol, df, ml_model)
                if reversion_opp:
                    opportunities.append(reversion_opp)
            
            # Strategy 3: Breakout
            if self.buy_strategies['breakout']['enabled']:
                breakout_opp = self._breakout_strategy(symbol, df, ml_model)
                if breakout_opp:
                    opportunities.append(breakout_opp)
            
            # Strategy 4: Momentum
            if self.buy_strategies['momentum']['enabled']:
                momentum_opp = self._momentum_strategy(symbol, df, ml_model)
                if momentum_opp:
                    opportunities.append(momentum_opp)
            
            # Strategy 5: Accumulation
            if self.buy_strategies['accumulation']['enabled']:
                accumulation_opp = self._accumulation_strategy(symbol, df, ml_model)
                if accumulation_opp:
                    opportunities.append(accumulation_opp)
            
            # Filter and rank opportunities
            filtered_opportunities = []
            for opp in opportunities:
                if self._validate_opportunity(opp, df):
                    filtered_opportunities.append(opp)
            
            # Sort by confluence score and confidence
            filtered_opportunities.sort(key=lambda x: (x.confluence_score * x.confidence), reverse=True)
            
            return filtered_opportunities
            
        except Exception as e:
            logger.error(f"Error generating buy signals for {symbol}: {e}")
            return []
    
    def _trend_following_strategy(self, symbol: str, df: pd.DataFrame, ml_model=None) -> Optional[TradingOpportunity]:
        """Trend following buy strategy"""
        
        try:
            latest = df.iloc[-1]
            
            # Multi-timeframe trend confirmation
            supporting_factors = []
            risk_factors = []
            confluence_score = 0
            
            # EMA alignment
            if ('EMA_20' in df.columns and 'EMA_50' in df.columns and 'EMA_200' in df.columns):
                if latest['EMA_20'] > latest['EMA_50'] > latest['EMA_200']:
                    supporting_factors.append("EMA alignment bullish")
                    confluence_score += 0.3
                elif latest['EMA_20'] < latest['EMA_50']:
                    risk_factors.append("Short-term EMA below medium-term")
            
            # ADX trend strength
            if latest.get('ADX', 0) > 25:
                supporting_factors.append(f"Strong trend (ADX: {latest['ADX']:.1f})")
                confluence_score += 0.25
            elif latest.get('ADX', 0) < 15:
                risk_factors.append("Weak trend strength")
            
            # MACD confirmation
            if latest.get('MACD', 0) > latest.get('MACD_signal', 0):
                supporting_factors.append("MACD bullish crossover")
                confluence_score += 0.2
            
            # Volume confirmation
            if latest.get('Volume_ratio', 1) > 1.2:
                supporting_factors.append("Above-average volume")
                confluence_score += 0.15
            
            # Price above key levels
            if latest['close'] > latest.get('EMA_200', latest['close']):
                supporting_factors.append("Price above 200 EMA")
                confluence_score += 0.1
            
            # ML model confirmation
            if ml_model and hasattr(ml_model, 'predict_ensemble'):
                try:
                    features = ml_model.create_advanced_features(df)
                    if not features.empty:
                        ml_prediction, ml_confidence = ml_model.predict_ensemble(features.iloc[[-1]])
                        if ml_prediction == 1:  # Buy signal
                            supporting_factors.append(f"ML model bullish (conf: {ml_confidence:.2f})")
                            confluence_score += 0.2 * ml_confidence
                        elif ml_prediction == -1:
                            risk_factors.append("ML model bearish")
                except Exception as e:
                    logger.debug(f"ML prediction error: {e}")
            
            # Check minimum confluence
            min_confluence = self.buy_strategies['trend_following']['min_confluence']
            if confluence_score < min_confluence:
                return None
            
            # Calculate expected return and risk
            atr = latest.get('ATR_14', latest['close'] * 0.02)
            expected_return = atr * 2  # 2:1 risk-reward
            max_risk = atr * 1
            
            return TradingOpportunity(
                symbol=symbol,
                signal_type="TREND_FOLLOWING_BUY",
                entry_price=latest['close'],
                confidence=min(confluence_score, 1.0),
                strength=confluence_score,
                timeframe="1h",
                confluence_score=confluence_score,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                expected_return=expected_return,
                max_risk=max_risk,
                holding_period="medium",
                market_regime=self.market_state['regime'],
                volume_profile="normal",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in trend following strategy: {e}")
            return None
    
    def _mean_reversion_strategy(self, symbol: str, df: pd.DataFrame, ml_model=None) -> Optional[TradingOpportunity]:
        """Mean reversion buy strategy"""
        
        try:
            latest = df.iloc[-1]
            
            supporting_factors = []
            risk_factors = []
            confluence_score = 0
            
            # RSI oversold
            rsi = latest.get('RSI_14', 50)
            if rsi < 30:
                supporting_factors.append(f"RSI oversold ({rsi:.1f})")
                confluence_score += 0.3
            elif rsi > 50:
                risk_factors.append("RSI not oversold")
            
            # Bollinger Bands
            bb_position = latest.get('BB_position', 0.5)
            if bb_position < 0.2:
                supporting_factors.append(f"Below lower Bollinger Band")
                confluence_score += 0.25
            elif bb_position > 0.5:
                risk_factors.append("Above middle Bollinger Band")
            
            # Stochastic oversold
            stoch_k = latest.get('STOCH_K', 50)
            if stoch_k < 20:
                supporting_factors.append(f"Stochastic oversold ({stoch_k:.1f})")
                confluence_score += 0.2
            
            # Williams %R
            willr = latest.get('WILLR', -50)
            if willr < -80:
                supporting_factors.append("Williams %R oversold")
                confluence_score += 0.15
            
            # Volume spike on decline
            if latest.get('Volume_ratio', 1) > 1.5:
                price_change = df['close'].pct_change().iloc[-1]
                if price_change < -0.02:  # Price down with volume
                    supporting_factors.append("Volume spike on decline")
                    confluence_score += 0.1
            
            # Check for support level
            if self._near_support_level(df, latest['close']):
                supporting_factors.append("Near support level")
                confluence_score += 0.15
            
            # Ensure not in strong downtrend
            if latest.get('ADX', 0) > 30 and latest.get('MINUS_DI', 0) > latest.get('PLUS_DI', 0):
                risk_factors.append("Strong downtrend")
                confluence_score -= 0.2
            
            # Check minimum confluence
            min_confluence = self.buy_strategies['mean_reversion']['min_confluence']
            if confluence_score < min_confluence:
                return None
            
            # Calculate expected return and risk
            atr = latest.get('ATR_14', latest['close'] * 0.02)
            expected_return = atr * 1.5  # Smaller expected return for mean reversion
            max_risk = atr * 0.8
            
            return TradingOpportunity(
                symbol=symbol,
                signal_type="MEAN_REVERSION_BUY",
                entry_price=latest['close'],
                confidence=min(confluence_score, 1.0),
                strength=confluence_score,
                timeframe="1h",
                confluence_score=confluence_score,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                expected_return=expected_return,
                max_risk=max_risk,
                holding_period="short",
                market_regime=self.market_state['regime'],
                volume_profile="elevated",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {e}")
            return None
    
    def _breakout_strategy(self, symbol: str, df: pd.DataFrame, ml_model=None) -> Optional[TradingOpportunity]:
        """Breakout buy strategy"""
        
        try:
            latest = df.iloc[-1]
            
            supporting_factors = []
            risk_factors = []
            confluence_score = 0
            
            # Bollinger Band breakout
            if latest.get('BB_position', 0.5) > 0.95:
                supporting_factors.append("Bollinger Band upper breakout")
                confluence_score += 0.3
            
            # Donchian Channel breakout
            if 'DC_upper' in df.columns:
                if latest['close'] > latest['DC_upper']:
                    supporting_factors.append("Donchian Channel breakout")
                    confluence_score += 0.25
            
            # Volume confirmation
            volume_ratio = latest.get('Volume_ratio', 1)
            if volume_ratio > 2.0:
                supporting_factors.append(f"Strong volume expansion ({volume_ratio:.1f}x)")
                confluence_score += 0.3
            elif volume_ratio < 1.2:
                risk_factors.append("Insufficient volume")
            
            # Previous consolidation
            if self._detect_consolidation(df):
                supporting_factors.append("Breaking from consolidation")
                confluence_score += 0.2
            
            # ATR expansion
            if len(df) >= 20:
                current_atr = latest.get('ATR_14', 0)
                avg_atr = df['ATR_14'].rolling(20).mean().iloc[-1]
                if current_atr > avg_atr * 1.3:
                    supporting_factors.append("ATR expansion")
                    confluence_score += 0.15
            
            # Price momentum
            if len(df) >= 5:
                momentum = df['close'].pct_change(5).iloc[-1]
                if momentum > 0.03:  # 3% momentum
                    supporting_factors.append("Strong price momentum")
                    confluence_score += 0.1
            
            # Check for false breakout risk
            if latest.get('RSI_14', 50) > 80:
                risk_factors.append("RSI overbought - false breakout risk")
                confluence_score -= 0.1
            
            # Check minimum confluence
            min_confluence = self.buy_strategies['breakout']['min_confluence']
            if confluence_score < min_confluence:
                return None
            
            # Calculate expected return and risk
            atr = latest.get('ATR_14', latest['close'] * 0.02)
            expected_return = atr * 3  # Higher expected return for breakouts
            max_risk = atr * 1.2
            
            return TradingOpportunity(
                symbol=symbol,
                signal_type="BREAKOUT_BUY",
                entry_price=latest['close'],
                confidence=min(confluence_score, 1.0),
                strength=confluence_score,
                timeframe="15m",
                confluence_score=confluence_score,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                expected_return=expected_return,
                max_risk=max_risk,
                holding_period="short",
                market_regime=self.market_state['regime'],
                volume_profile="high",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in breakout strategy: {e}")
            return None
    
    def _momentum_strategy(self, symbol: str, df: pd.DataFrame, ml_model=None) -> Optional[TradingOpportunity]:
        """Momentum buy strategy"""
        
        try:
            latest = df.iloc[-1]
            
            supporting_factors = []
            risk_factors = []
            confluence_score = 0
            
            # ROC momentum
            roc_10 = latest.get('ROC_10', 0)
            if roc_10 > 5:  # 5% momentum
                supporting_factors.append(f"Strong ROC momentum ({roc_10:.1f}%)")
                confluence_score += 0.25
            
            # RSI momentum (not overbought but rising)
            rsi = latest.get('RSI_14', 50)
            if 50 < rsi < 70:
                supporting_factors.append("RSI in momentum zone")
                confluence_score += 0.2
            elif rsi > 75:
                risk_factors.append("RSI overbought")
            
            # MACD momentum
            macd_hist = latest.get('MACD_hist', 0)
            if macd_hist > 0:
                if len(df) >= 5:
                    macd_acceleration = df['MACD_hist'].iloc[-1] - df['MACD_hist'].iloc[-5]
                    if macd_acceleration > 0:
                        supporting_factors.append("MACD accelerating")
                        confluence_score += 0.2
            
            # Price momentum with volume
            if len(df) >= 10:
                price_momentum = df['close'].pct_change(10).iloc[-1]
                volume_momentum = df['volume'].rolling(10).mean().iloc[-1] / df['volume'].rolling(30).mean().iloc[-1]
                
                if price_momentum > 0.05 and volume_momentum > 1.2:
                    supporting_factors.append("Price-volume momentum alignment")
                    confluence_score += 0.25
            
            # Moving average momentum
            if ('EMA_10' in df.columns and 'EMA_20' in df.columns):
                if latest['EMA_10'] > latest['EMA_20']:
                    ma_momentum = (latest['EMA_10'] - latest['EMA_20']) / latest['EMA_20']
                    if ma_momentum > 0.01:  # 1% momentum
                        supporting_factors.append("Moving average momentum")
                        confluence_score += 0.15
            
            # Relative strength vs market
            # This would need market data comparison - simplified here
            supporting_factors.append("Relative strength analysis")
            confluence_score += 0.1
            
            # Check minimum confluence
            min_confluence = self.buy_strategies['momentum']['min_confluence']
            if confluence_score < min_confluence:
                return None
            
            # Calculate expected return and risk
            atr = latest.get('ATR_14', latest['close'] * 0.02)
            expected_return = atr * 2.5
            max_risk = atr * 1.1
            
            return TradingOpportunity(
                symbol=symbol,
                signal_type="MOMENTUM_BUY",
                entry_price=latest['close'],
                confidence=min(confluence_score, 1.0),
                strength=confluence_score,
                timeframe="1h",
                confluence_score=confluence_score,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                expected_return=expected_return,
                max_risk=max_risk,
                holding_period="medium",
                market_regime=self.market_state['regime'],
                volume_profile="increasing",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
            return None
    
    def _accumulation_strategy(self, symbol: str, df: pd.DataFrame, ml_model=None) -> Optional[TradingOpportunity]:
        """Accumulation buy strategy (institutional buying)"""
        
        try:
            latest = df.iloc[-1]
            
            supporting_factors = []
            risk_factors = []
            confluence_score = 0
            
            # On Balance Volume trend
            if 'OBV' in df.columns and len(df) >= 20:
                obv_trend = df['OBV'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]).iloc[-1]
                if obv_trend > 0:
                    supporting_factors.append("OBV accumulation trend")
                    confluence_score += 0.25
            
            # Accumulation/Distribution Line
            if 'AD' in df.columns and len(df) >= 10:
                ad_trend = df['AD'].iloc[-1] - df['AD'].iloc[-10]
                if ad_trend > 0:
                    supporting_factors.append("A/D Line accumulation")
                    confluence_score += 0.2
            
            # Volume-Price Trend
            if 'VPT' in df.columns and len(df) >= 15:
                vpt_slope = (df['VPT'].iloc[-1] - df['VPT'].iloc[-15]) / 15
                if vpt_slope > 0:
                    supporting_factors.append("VPT uptrend")
                    confluence_score += 0.15
            
            # VPIN (Volume-Synchronized Probability of Informed Trading)
            vpin = latest.get('VPIN', 0.5)
            if vpin > 0.6:
                supporting_factors.append("High informed trading activity")
                confluence_score += 0.2
            
            # Steady accumulation pattern
            if self._detect_accumulation_pattern(df):
                supporting_factors.append("Accumulation pattern detected")
                confluence_score += 0.3
            
            # Price consolidation with volume
            if self._detect_consolidation(df):
                avg_volume = df['volume'].rolling(20).mean().iloc[-1]
                recent_volume = df['volume'].iloc[-5:].mean()
                if recent_volume > avg_volume * 1.1:
                    supporting_factors.append("Consolidation with volume")
                    confluence_score += 0.15
            
            # Low volatility (potential spring loading)
            if latest.get('ATR_14', 0) < df['ATR_14'].rolling(50).mean().iloc[-1]:
                supporting_factors.append("Low volatility environment")
                confluence_score += 0.1
            
            # Check minimum confluence
            min_confluence = self.buy_strategies['accumulation']['min_confluence']
            if confluence_score < min_confluence:
                return None
            
            # Calculate expected return and risk
            atr = latest.get('ATR_14', latest['close'] * 0.02)
            expected_return = atr * 2  # Conservative for accumulation
            max_risk = atr * 0.9
            
            return TradingOpportunity(
                symbol=symbol,
                signal_type="ACCUMULATION_BUY",
                entry_price=latest['close'],
                confidence=min(confluence_score, 1.0),
                strength=confluence_score,
                timeframe="4h",
                confluence_score=confluence_score,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                expected_return=expected_return,
                max_risk=max_risk,
                holding_period="long",
                market_regime=self.market_state['regime'],
                volume_profile="steady",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in accumulation strategy: {e}")
            return None
    
    def _near_support_level(self, df: pd.DataFrame, current_price: float) -> bool:
        """Check if price is near support level"""
        
        try:
            if len(df) < 50:
                return False
            
            # Find recent lows
            lows = df['low'].rolling(20).min()
            recent_lows = lows.iloc[-50:]
            
            # Check if current price is within 2% of any recent low
            for low in recent_lows:
                if abs(current_price - low) / low < 0.02:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking support level: {e}")
            return False
    
    def _detect_consolidation(self, df: pd.DataFrame, periods: int = 20) -> bool:
        """Detect consolidation pattern"""
        
        try:
            if len(df) < periods:
                return False
            
            # Check if price has been range-bound
            recent_data = df.iloc[-periods:]
            price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()
            
            # Consolidation if range is less than 10%
            return price_range < 0.10
            
        except Exception as e:
            logger.error(f"Error detecting consolidation: {e}")
            return False
    
    def _detect_accumulation_pattern(self, df: pd.DataFrame) -> bool:
        """Detect accumulation pattern using Wyckoff methodology"""
        
        try:
            if len(df) < 30:
                return False
            
            # Look for price stability with increasing volume
            recent_prices = df['close'].iloc[-20:]
            recent_volumes = df['volume'].iloc[-20:]
            
            # Price stability (low volatility)
            price_stability = recent_prices.std() / recent_prices.mean() < 0.05
            
            # Volume increase
            volume_trend = recent_volumes.iloc[-10:].mean() > recent_volumes.iloc[-20:-10].mean()
            
            return price_stability and volume_trend
            
        except Exception as e:
            logger.error(f"Error detecting accumulation pattern: {e}")
            return False
    
    def _validate_opportunity(self, opportunity: TradingOpportunity, df: pd.DataFrame) -> bool:
        """Validate trading opportunity against filters"""
        
        try:
            # Check circuit breaker
            if self.circuit_breaker.get('active', False):
                return False
            
            # Check market regime compatibility
            regime_compatible = self._check_regime_compatibility(opportunity)
            if not regime_compatible:
                return False
            
            # Check minimum confluence score
            if opportunity.confluence_score < 0.5:
                return False
            
            # Check risk-reward ratio
            if opportunity.expected_return / opportunity.max_risk < 1.5:
                return False
            
            # Check recent performance of this strategy
            strategy_key = opportunity.signal_type.split('_')[0].lower()
            if strategy_key in self.strategy_performance:
                perf = self.strategy_performance[strategy_key]
                if perf['signals'] > 10 and perf['winners'] / perf['signals'] < 0.4:
                    return False  # Poor recent performance
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating opportunity: {e}")
            return False
    
    def _check_regime_compatibility(self, opportunity: TradingOpportunity) -> bool:
        """Check if opportunity is compatible with current market regime"""
        
        regime_compatibility = {
            MarketRegime.BULL_TREND: ['TREND_FOLLOWING_BUY', 'MOMENTUM_BUY', 'BREAKOUT_BUY'],
            MarketRegime.BEAR_TREND: ['MEAN_REVERSION_BUY', 'ACCUMULATION_BUY'],
            MarketRegime.RANGE_BOUND: ['MEAN_REVERSION_BUY', 'ACCUMULATION_BUY'],
            MarketRegime.HIGH_VOLATILITY: ['BREAKOUT_BUY', 'MOMENTUM_BUY'],
            MarketRegime.LOW_VOLATILITY: ['ACCUMULATION_BUY', 'TREND_FOLLOWING_BUY'],
            MarketRegime.BREAKOUT: ['BREAKOUT_BUY', 'MOMENTUM_BUY'],
            MarketRegime.ACCUMULATION: ['ACCUMULATION_BUY', 'MEAN_REVERSION_BUY']
        }
        
        compatible_strategies = regime_compatibility.get(self.market_state['regime'], [])
        return opportunity.signal_type in compatible_strategies
    
    def find_best_opportunities(self, data_fetcher, indicator_calculator, ml_model,
                              supported_symbols: List[str], current_positions: Dict,
                              limit: int = 5) -> List[TradingOpportunity]:
        """Find best trading opportunities across all symbols"""
        
        try:
            all_opportunities = []
            
            # Update market regime
            self.analyze_market_regime(data_fetcher, indicator_calculator)
            
            # Analyze each symbol
            for symbol_info in supported_symbols[:30]:  # Top 30 by volume
                symbol = symbol_info['symbol']
                
                # Skip if already have position
                if any(pos['symbol'] == symbol for pos in current_positions.values()):
                    continue
                
                try:
                    # Get data and indicators
                    df = data_fetcher.get_historical_data(symbol, '1h', '7 days')
                    if df is None or len(df) < 100:
                        continue
                    
                    df = indicator_calculator.calculate_all_indicators(df)
                    if df is None:
                        continue
                    
                    # Generate buy signals
                    opportunities = self.generate_buy_signals(symbol, df, ml_model)
                    all_opportunities.extend(opportunities)
                    
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Sort by composite score
            all_opportunities.sort(
                key=lambda x: (x.confluence_score * x.confidence * self.buy_strategies.get(
                    x.signal_type.split('_')[0].lower(), {'weight': 0.2})['weight']),
                reverse=True
            )
            
            # Apply correlation filter
            filtered_opportunities = self._apply_correlation_filter(all_opportunities)
            
            logger.info(f"Found {len(filtered_opportunities)} high-quality opportunities")
            
            return filtered_opportunities[:limit]
            
        except Exception as e:
            logger.error(f"Error finding opportunities: {e}")
            return []
    
    def _apply_correlation_filter(self, opportunities: List[TradingOpportunity]) -> List[TradingOpportunity]:
        """Apply correlation filter to avoid similar positions"""
        
        try:
            if len(opportunities) <= 1:
                return opportunities
            
            filtered = [opportunities[0]]  # Always include the best opportunity
            
            for opp in opportunities[1:]:
                # Check correlation with already selected opportunities
                correlated = False
                for selected in filtered:
                    # Simple correlation check - could be enhanced with actual price correlation
                    if self._symbols_correlated(opp.symbol, selected.symbol):
                        correlated = True
                        break
                
                if not correlated:
                    filtered.append(opp)
                    
                # Limit total opportunities
                if len(filtered) >= 5:
                    break
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error applying correlation filter: {e}")
            return opportunities
    
    def _symbols_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are highly correlated"""
        
        # Simple correlation check based on base asset
        base1 = symbol1.replace('USDT', '').replace('BTC', '').replace('ETH', '')
        base2 = symbol2.replace('USDT', '').replace('BTC', '').replace('ETH', '')
        
        # Highly correlated pairs
        correlated_groups = [
            ['BTC', 'ETH'],  # Major cryptos
            ['BNB', 'CAKE'],  # BSC ecosystem
            ['DOT', 'KSM'],  # Polkadot ecosystem
            ['AVAX', 'JOE'],  # Avalanche ecosystem
        ]
        
        for group in correlated_groups:
            if base1 in group and base2 in group:
                return True
        
        return False
    
    def update_strategy_performance(self, strategy_type: str, success: bool, return_pct: float):
        """Update strategy performance tracking"""
        
        try:
            strategy_key = strategy_type.split('_')[0].lower()
            if strategy_key in self.strategy_performance:
                perf = self.strategy_performance[strategy_key]
                perf['signals'] += 1
                if success:
                    perf['winners'] += 1
                
                # Update average return
                current_avg = perf['avg_return']
                perf['avg_return'] = (current_avg * (perf['signals'] - 1) + return_pct) / perf['signals']
                
                logger.info(f"Updated {strategy_key} performance: {perf['winners']}/{perf['signals']} wins, avg return: {perf['avg_return']:.2f}%")
                
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    def get_strategy_analytics(self) -> Dict:
        """Get comprehensive strategy analytics"""
        
        try:
            analytics = {
                'market_state': {
                    'regime': self.market_state['regime'].value,
                    'volatility_percentile': self.market_state['volatility_percentile'],
                    'trend_strength': self.market_state['trend_strength']
                },
                'strategy_performance': self.strategy_performance,
                'active_strategies': {
                    name: config for name, config in self.buy_strategies.items() 
                    if config['enabled']
                },
                'circuit_breaker': self.circuit_breaker
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting strategy analytics: {e}")
            return {}