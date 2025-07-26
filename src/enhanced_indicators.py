"""
Enhanced technical indicators calculation module with 25+ indicators
across multiple timeframes for institutional-grade analysis.
Optimized for 'ta' library with manual fallbacks.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# Import ta library (confirmed working)
try:
    import ta
    HAS_TA = True
    logger_init_msg = "Using 'ta' library for technical indicators"
except ImportError:
    HAS_TA = False
    logger_init_msg = "Using manual calculations for technical indicators"

# Try other libraries as secondary options
try:
    import pandas_ta as pta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

try:
    import finta
    HAS_FINTA = True
except ImportError:
    HAS_FINTA = False

from config import STRATEGY_PARAMS

logger = logging.getLogger("BinanceTrading.EnhancedIndicators")


class EnhancedTechnicalIndicators:
    """Advanced technical indicators optimized for 'ta' library"""
    
    def __init__(self, strategy_params=None):
        self.strategy_params = strategy_params or STRATEGY_PARAMS
        self.scaler = MinMaxScaler()
        
        logger.info(logger_init_msg)
        if HAS_TA:
            logger.info("✅ Primary library 'ta' is available and working")
        
    def calculate_all_indicators(self, df: pd.DataFrame, timeframe: str = '1h') -> Optional[pd.DataFrame]:
        """Calculate comprehensive set of technical indicators"""
        if df is None or len(df) < 100:
            logger.warning(f"Insufficient data for indicator calculation: {len(df) if df is not None else 0}")
            return None
            
        try:
            # Make a copy to avoid modifying original data
            df_enhanced = df.copy()
            
            # Ensure proper column names and data types
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df_enhanced.columns:
                    logger.error(f"Missing required column: {col}")
                    return None
                # Convert to numeric
                df_enhanced[col] = pd.to_numeric(df_enhanced[col], errors='coerce')
            
            # Remove any rows with NaN values in OHLCV data
            df_enhanced = df_enhanced.dropna(subset=required_columns)
            
            if len(df_enhanced) < 50:
                logger.warning(f"Insufficient clean data after preprocessing: {len(df_enhanced)}")
                return None
            
            logger.info(f"Processing {len(df_enhanced)} data points for {timeframe}")
            
            # ========== TREND INDICATORS ==========
            df_enhanced = self._calculate_trend_indicators(df_enhanced)
            
            # ========== MOMENTUM INDICATORS ==========
            df_enhanced = self._calculate_momentum_indicators(df_enhanced)
            
            # ========== VOLATILITY INDICATORS ==========
            df_enhanced = self._calculate_volatility_indicators(df_enhanced)
            
            # ========== VOLUME INDICATORS ==========
            df_enhanced = self._calculate_volume_indicators(df_enhanced)
            
            # ========== CUSTOM COMPOSITE INDICATORS ==========
            df_enhanced = self._calculate_composite_indicators(df_enhanced)
            
            # ========== STATISTICAL INDICATORS ==========
            df_enhanced = self._calculate_statistical_indicators(df_enhanced)
            
            # ========== MARKET MICROSTRUCTURE ==========
            df_enhanced = self._calculate_microstructure_indicators(df_enhanced)
            
            # Clean data
            df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan)
            # Use forward fill then backward fill for any remaining NaN values
            df_enhanced = df_enhanced.ffill().bfill().fillna(0)
            
            logger.info(f"✅ Enhanced indicators calculated for {timeframe}: {len(df_enhanced.columns)} features")
            return df_enhanced
            
        except Exception as e:
            logger.error(f"Error calculating enhanced indicators: {e}")
            logger.error(f"Error details: {str(e)}")
            return None
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators using 'ta' library"""
        
        if HAS_TA:
            # Moving Averages using ta library
            periods = [5, 10, 20, 50, 100, 200]
            for period in periods:
                try:
                    df[f'SMA_{period}'] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator()
                    df[f'EMA_{period}'] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()
                    # WMA using manual calculation
                    weights = np.arange(1, period + 1)
                    df[f'WMA_{period}'] = df['close'].rolling(window=period).apply(
                        lambda x: np.dot(x, weights) / weights.sum() if len(x) == period else np.nan, raw=True
                    )
                except Exception as e:
                    logger.warning(f"Error calculating MA for period {period}: {e}")
            
            # MACD
            try:
                macd_indicator = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
                df['MACD'] = macd_indicator.macd()
                df['MACD_signal'] = macd_indicator.macd_signal()
                df['MACD_hist'] = macd_indicator.macd_diff()
                
                # Fast MACD
                macd_fast = ta.trend.MACD(df['close'], window_fast=5, window_slow=13, window_sign=7)
                df['MACD_fast'] = macd_fast.macd()
                df['MACD_signal_fast'] = macd_fast.macd_signal()
                df['MACD_hist_fast'] = macd_fast.macd_diff()
            except Exception as e:
                logger.warning(f"Error calculating MACD: {e}")
            
            # ADX and Directional Movement
            try:
                adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
                df['ADX'] = adx_indicator.adx()
                df['PLUS_DI'] = adx_indicator.adx_pos()
                df['MINUS_DI'] = adx_indicator.adx_neg()
            except Exception as e:
                logger.warning(f"Error calculating ADX: {e}")
            
            # Parabolic SAR
            try:
                psar_indicator = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
                df['SAR'] = psar_indicator.psar()
            except Exception as e:
                logger.warning(f"Error calculating PSAR: {e}")
            
            # Aroon
            try:
                # FIX: Explicitly set the window size, which is a required parameter.
                # The standard for Aroon is 25 periods.
                aroon_indicator = ta.trend.AroonIndicator(
                    high=df['high'],
                    low=df['low'],
                    window=25  # <-- THIS IS THE FIX
                )
                df['AROON_up'] = aroon_indicator.aroon_up()
                df['AROON_down'] = aroon_indicator.aroon_down()
                df['AROON_osc'] = aroon_indicator.aroon_up() - aroon_indicator.aroon_down()
            except Exception as e:
                logger.warning(f"Error calculating Aroon: {e}")
        
        else:
            # Fallback to manual calculations
            periods = [5, 10, 20, 50, 100, 200]
            for period in periods:
                df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()
        
        # Ichimoku Cloud (manual - consistent across all methods)
        try:
            df['TENKAN'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
            df['KIJUN'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
            df['SENKOU_A'] = ((df['TENKAN'] + df['KIJUN']) / 2).shift(26)
            df['SENKOU_B'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
            df['CHIKOU'] = df['close'].shift(-26)
        except Exception as e:
            logger.warning(f"Error calculating Ichimoku: {e}")
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum and oscillator indicators"""
        
        if HAS_TA:
            try:
                # RSI variations
                df['RSI_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
                df['RSI_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
                df['RSI_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
                
                # Stochastic Oscillator
                stoch_indicator = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
                df['STOCH_K'] = stoch_indicator.stoch()
                df['STOCH_D'] = stoch_indicator.stoch_signal()
                
                # Williams %R
                df['WILLR'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
                
                # Rate of Change
                df['ROC_10'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
                df['ROC_20'] = ta.momentum.ROCIndicator(df['close'], window=20).roc()
                
                # Momentum (manual since ta doesn't have it)
                df['MOM_10'] = df['close'] - df['close'].shift(10)
                df['MOM_20'] = df['close'] - df['close'].shift(20)
                
                # Commodity Channel Index
                df['CCI'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
                
                # Money Flow Index
                df['MFI'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
                
                # Ultimate Oscillator
                df['ULTOSC'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
                
                # TRIX
                df['TRIX'] = ta.trend.TRIXIndicator(df['close']).trix()
                
            except Exception as e:
                logger.warning(f"Error calculating momentum indicators: {e}")
        
        else:
            # Manual RSI calculation
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            df['RSI_14'] = calculate_rsi(df['close'], 14)
            df['RSI_7'] = calculate_rsi(df['close'], 7)
            df['RSI_21'] = calculate_rsi(df['close'], 21)
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility and range indicators"""
        
        if HAS_TA:
            try:
                # Bollinger Bands
                bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
                df['BB_upper'] = bb_indicator.bollinger_hband()
                df['BB_middle'] = bb_indicator.bollinger_mavg()
                df['BB_lower'] = bb_indicator.bollinger_lband()
                df['BB_width'] = bb_indicator.bollinger_wband()
                df['BB_position'] = bb_indicator.bollinger_pband()
                
                # Different BB periods
                bb_10 = ta.volatility.BollingerBands(df['close'], window=10, window_dev=2)
                df['BB_upper_10'] = bb_10.bollinger_hband()
                df['BB_middle_10'] = bb_10.bollinger_mavg()
                df['BB_lower_10'] = bb_10.bollinger_lband()
                
                # Average True Range
                df['ATR_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
                df['ATR_7'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=7).average_true_range()
                df['ATR_21'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=21).average_true_range()
                
                # True Range
                df['TRANGE'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
                
                # Keltner Channels
                kc_indicator = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
                df['KC_upper'] = kc_indicator.keltner_channel_hband()
                df['KC_lower'] = kc_indicator.keltner_channel_lband()
                df['KC_middle'] = kc_indicator.keltner_channel_mband()
                
                # Donchian Channels
                dc_indicator = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
                df['DC_upper'] = dc_indicator.donchian_channel_hband()
                df['DC_lower'] = dc_indicator.donchian_channel_lband()
                df['DC_middle'] = dc_indicator.donchian_channel_mband()
                
            except Exception as e:
                logger.warning(f"Error calculating volatility indicators: {e}")
        
        else:
            # Manual Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['BB_upper'] = sma_20 + (std_20 * 2)
            df['BB_middle'] = sma_20
            df['BB_lower'] = sma_20 - (std_20 * 2)
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        
        # Basic volume metrics
        df['Volume_SMA_20'] = df['volume'].rolling(window=20).mean()
        df['Volume_SMA_50'] = df['volume'].rolling(window=50).mean()
        df['Volume_ratio'] = df['volume'] / df['Volume_SMA_20']
        
        if HAS_TA:
            try:
                # On Balance Volume
                df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
                
                # Accumulation/Distribution Line
                df['AD'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
                
                # Chaikin Money Flow
                df['ADOSC'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
                
                # Volume Weighted Average Price
                df['VWAP'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
                
                # Force Index
                df['FI'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()
                
                # Ease of Movement
                df['EOM'] = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume']).ease_of_movement()
                
            except Exception as e:
                logger.warning(f"Error calculating volume indicators: {e}")
        
        else:
            # Manual OBV calculation
            def calculate_obv(close, volume):
                obv = pd.Series(index=close.index, dtype=float)
                obv.iloc[0] = volume.iloc[0]
                
                for i in range(1, len(close)):
                    if close.iloc[i] > close.iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                    elif close.iloc[i] < close.iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                
                return obv
            
            df['OBV'] = calculate_obv(df['close'], df['volume'])
        
        # Volume Price Trend (manual)
        df['VPT'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).fillna(0).cumsum()
        
        # Simple VWAP approximation
        if 'VWAP' not in df.columns:
            df['VWAP'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        return df
    
    def _calculate_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom composite indicators"""
        
        # Trend Strength Index (custom)
        if all(col in df.columns for col in ['ADX', 'PLUS_DI', 'MINUS_DI']):
            df['Trend_Strength'] = (df['ADX'] / 100) * (abs(df['PLUS_DI'] - df['MINUS_DI']) / 100)
        
        # Volatility Breakout Signal
        if 'ATR_14' in df.columns:
            df['VolBreakout'] = np.where(df['ATR_14'] > df['ATR_14'].rolling(window=20).mean() * 1.5, 1, 0)
        
        # Multi-timeframe RSI divergence
        if all(col in df.columns for col in ['RSI_7', 'RSI_21']):
            df['RSI_divergence'] = df['RSI_7'] - df['RSI_21']
        
        # Price-Volume Divergence
        price_momentum = df['close'].pct_change(5)
        volume_momentum = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        df['PV_divergence'] = price_momentum - (volume_momentum - 1)
        
        # Efficiency Ratio (Kaufman)
        change = abs(df['close'] - df['close'].shift(10))
        volatility = abs(df['close'] - df['close'].shift()).rolling(window=10).sum()
        df['Efficiency_Ratio'] = change / volatility
        
        # Squeeze Momentum (Bollinger Bands + Keltner Channels)
        if all(col in df.columns for col in ['BB_upper', 'BB_lower', 'KC_upper', 'KC_lower']):
            df['Squeeze'] = np.where((df['BB_upper'] < df['KC_upper']) & (df['BB_lower'] > df['KC_lower']), 1, 0)
        
        # Market Structure (Higher Highs, Lower Lows)
        df['HH'] = df['high'] > df['high'].shift(1)
        df['LL'] = df['low'] < df['low'].shift(1)
        df['Market_Structure'] = df['HH'].rolling(5).sum() - df['LL'].rolling(5).sum()
        
        return df
    
    def _calculate_statistical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical and mathematical indicators"""
        
        # Linear Regression
        def calculate_linear_regression(series, period=14):
            def linreg_stats(y):
                if len(y) < 2:
                    return np.nan, np.nan, np.nan
                x = np.arange(len(y))
                try:
                    slope, intercept = np.polyfit(x, y, 1)
                    angle = np.arctan(slope) * 180 / np.pi
                    return slope, intercept, angle
                except:
                    return np.nan, np.nan, np.nan
            
            results = series.rolling(window=period).apply(
                lambda x: linreg_stats(x)[0], raw=True  # slope
            )
            return results
        
        df['LinReg_Slope'] = calculate_linear_regression(df['close'])
        df['LinReg_Angle'] = df['LinReg_Slope'] * 180 / np.pi  # Convert to degrees
        
        # Standard Deviation and Variance
        df['StdDev'] = df['close'].rolling(window=20).std()
        df['Variance'] = df['close'].rolling(window=20).var()
        
        # Correlation with time (trend strength)
        def rolling_correlation(series, window=20):
            return series.rolling(window).apply(
                lambda x: stats.pearsonr(np.arange(len(x)), x)[0] if len(x) >= 2 else 0
            )
        
        df['Time_Correlation'] = rolling_correlation(df['close'])
        
        return df
    
    def _calculate_microstructure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure indicators"""
        
        # Basic microstructure metrics
        daily_return = abs(df['close'].pct_change())
        dollar_volume = df['close'] * df['volume']
        
        # Amihud Illiquidity Ratio
        df['Amihud_Illiq'] = daily_return / (dollar_volume + 1e-10)  # Add small value to avoid division by zero
        
        # Price Impact
        df['Price_Impact'] = daily_return / (np.sqrt(dollar_volume) + 1e-10)
        
        # VPIN approximation
        buy_volume = np.where(df['close'] > df['close'].shift(1), df['volume'], 0)
        sell_volume = np.where(df['close'] < df['close'].shift(1), df['volume'], 0)
        total_volume = buy_volume + sell_volume
        with np.errstate(divide='ignore', invalid='ignore'):
            df['VPIN'] = np.where(total_volume > 0, 
                         np.nan_to_num(abs(buy_volume - sell_volume) / total_volume), 
                         0)

        
        return df
    
    # Keep all existing analysis methods unchanged
    def get_multi_timeframe_signals(self, symbol: str, data_fetcher, timeframes: List[str] = ['1m', '5m', '15m', '1h', '4h']) -> Dict:
        """Generate signals across multiple timeframes"""
        
        multi_tf_data = {}
        signals = {}
        
        for tf in timeframes:
            try:
                # Get data for timeframe
                df = data_fetcher.get_historical_data(symbol, interval=tf, lookback='30 days')
                if df is not None:
                    # Calculate indicators
                    df_indicators = self.calculate_all_indicators(df, timeframe=tf)
                    if df_indicators is not None:
                        multi_tf_data[tf] = df_indicators
                        
                        # Generate basic signals for this timeframe
                        latest = df_indicators.iloc[-1]
                        
                        signals[tf] = {
                            'trend': self._analyze_trend(latest),
                            'momentum': self._analyze_momentum(latest),
                            'volatility': self._analyze_volatility(latest),
                            'volume': self._analyze_volume(latest),
                            'strength': self._calculate_signal_strength(latest)
                        }
                        
            except Exception as e:
                logger.error(f"Error processing {tf} timeframe for {symbol}: {e}")
                continue
        
        # Combine signals across timeframes
        combined_signal = self._combine_timeframe_signals(signals)
        
        return {
            'data': multi_tf_data,
            'signals': signals,
            'combined': combined_signal
        }
    
    def _analyze_trend(self, data: pd.Series) -> Dict:
        """Analyze trend conditions"""
        return {
            'direction': 'up' if data.get('EMA_20', 0) > data.get('EMA_50', 0) else 'down',
            'strength': data.get('ADX', 0) / 100,
            'confirmation': data.get('MACD', 0) > data.get('MACD_signal', 0)
        }
    
    def _analyze_momentum(self, data: pd.Series) -> Dict:
        """Analyze momentum conditions"""
        return {
            'rsi_signal': 'oversold' if data.get('RSI_14', 50) < 30 else 'overbought' if data.get('RSI_14', 50) > 70 else 'neutral',
            'stoch_signal': 'oversold' if data.get('STOCH_K', 50) < 20 else 'overbought' if data.get('STOCH_K', 50) > 80 else 'neutral',
            'divergence': data.get('RSI_divergence', 0)
        }
    
    def _analyze_volatility(self, data: pd.Series) -> Dict:
        """Analyze volatility conditions"""
        return {
            'bb_position': data.get('BB_position', 0.5),
            'squeeze': data.get('Squeeze', 0),
            'breakout': data.get('VolBreakout', 0),
            'expansion': data.get('BB_width', 0)
        }
    
    def _analyze_volume(self, data: pd.Series) -> Dict:
        """Analyze volume conditions"""
        return {
            'volume_ratio': data.get('Volume_ratio', 1),
            'obv_trend': 'up' if data.get('OBV', 0) > 0 else 'down',
            'vpin': data.get('VPIN', 0)
        }
    
    def _calculate_signal_strength(self, data: pd.Series) -> float:
        """Calculate overall signal strength for timeframe"""
        strength_factors = [
            data.get('ADX', 0) / 100,
            abs(data.get('RSI_14', 50) - 50) / 50,
            max(-1, min(1, data.get('Volume_ratio', 1) - 1)),  # Clamp between -1 and 1
            abs(data.get('BB_position', 0.5) - 0.5) * 2,  # Scale to 0-1
            data.get('Trend_Strength', 0)
        ]
        
        # Filter out NaN values and ensure reasonable bounds
        valid_factors = [f for f in strength_factors if not np.isnan(f) and np.isfinite(f)]
        return np.mean(valid_factors) if valid_factors else 0
    
    def _combine_timeframe_signals(self, signals: Dict) -> Dict:
        """Combine signals from multiple timeframes with weighting"""
        
        timeframe_weights = {
            '1m': 0.1,
            '5m': 0.15,
            '15m': 0.2,
            '1h': 0.25,
            '4h': 0.3
        }
        
        combined = {
            'buy_score': 0,
            'sell_score': 0,
            'confidence': 0,
            'timeframe_agreement': 0
        }
        
        total_weight = 0
        agreement_count = 0
        total_signals = 0
        
        for tf, signal_data in signals.items():
            weight = timeframe_weights.get(tf, 0.2)
            total_weight += weight
            total_signals += 1
            
            # Calculate buy/sell scores based on signal data
            trend_score = 1 if signal_data['trend']['direction'] == 'up' else -1
            momentum_score = 1 if signal_data['momentum']['rsi_signal'] == 'oversold' else -1 if signal_data['momentum']['rsi_signal'] == 'overbought' else 0
            
            signal_score = (trend_score + momentum_score) * signal_data['strength'] * weight
            
            if signal_score > 0:
                combined['buy_score'] += signal_score
                agreement_count += 1
            elif signal_score < 0:
                combined['sell_score'] += abs(signal_score)
                agreement_count += 1
        
        # Normalize scores
        if total_weight > 0:
            combined['buy_score'] /= total_weight
            combined['sell_score'] /= total_weight
            combined['timeframe_agreement'] = agreement_count / total_signals if total_signals > 0 else 0
            combined['confidence'] = max(combined['buy_score'], combined['sell_score'])
        
        return combined


if __name__ == "__main__":
    # Test the indicators
    print("🧪 Testing Enhanced Technical Indicators...")
    
    # Create sample OHLCV data
    dates = pd.date_range('2023-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    # Generate realistic price data
    price_base = 100
    returns = np.random.normal(0, 0.02, 200)
    prices = [price_base]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data with proper structure
    sample_data = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    # Ensure high >= close >= low and high >= open >= low
    for i in range(len(sample_data)):
        row = sample_data.iloc[i]
        min_price = min(row['open'], row['close'])
        max_price = max(row['open'], row['close'])
        
        sample_data.iloc[i, sample_data.columns.get_loc('low')] = min(row['low'], min_price)
        sample_data.iloc[i, sample_data.columns.get_loc('high')] = max(row['high'], max_price)
    
    print(f"📊 Created sample data with {len(sample_data)} rows")
    print(f"💰 Price range: ${sample_data['low'].min():.2f} - ${sample_data['high'].max():.2f}")
    
    # Test the indicators
    try:
        indicators = EnhancedTechnicalIndicators()
        result = indicators.calculate_all_indicators(sample_data)
        
        if result is not None:
            print(f"✅ Successfully calculated {len(result.columns)} indicators")
            print(f"📈 Available indicators include:")
            
            # Group indicators by category
            trend_indicators = [col for col in result.columns if any(x in col for x in ['SMA', 'EMA', 'MACD', 'ADX', 'AROON', 'SAR'])]
            momentum_indicators = [col for col in result.columns if any(x in col for x in ['RSI', 'STOCH', 'WILLR', 'ROC', 'MOM', 'CCI', 'MFI'])]
            volatility_indicators = [col for col in result.columns if any(x in col for x in ['BB_', 'ATR', 'KC_', 'DC_'])]
            volume_indicators = [col for col in result.columns if any(x in col for x in ['Volume', 'OBV', 'AD', 'VWAP', 'VPT'])]
            
            print(f"   📊 Trend: {len(trend_indicators)} indicators")
            print(f"   🎯 Momentum: {len(momentum_indicators)} indicators") 
            print(f"   📈 Volatility: {len(volatility_indicators)} indicators")
            print(f"   📦 Volume: {len(volume_indicators)} indicators")
            
            # Show latest values for key indicators
            latest = result.iloc[-1]
            print(f"\n🔍 Latest indicator values:")
            print(f"   RSI (14): {latest.get('RSI_14', 'N/A'):.2f}")
            print(f"   MACD: {latest.get('MACD', 'N/A'):.4f}")
            print(f"   BB Position: {latest.get('BB_position', 'N/A'):.3f}")
            print(f"   ADX: {latest.get('ADX', 'N/A'):.2f}")
            print(f"   Volume Ratio: {latest.get('Volume_ratio', 'N/A'):.2f}")
            
            print(f"\n🎉 Enhanced indicators working perfectly with 'ta' library!")
            
        else:
            print("❌ Failed to calculate indicators")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()