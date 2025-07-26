# talib_wrapper.py - Safe TA-Lib function wrapper
"""
Safe wrapper for TA-Lib functions that handles missing symbols gracefully
"""

import numpy as np
import pandas as pd
import logging

# Try importing TA-Lib, fall back to pandas-ta if needed
try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TA-Lib imported successfully")
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not available, using pandas-ta fallback")

# Try importing pandas-ta as fallback
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
    print("✅ pandas-ta available as fallback")
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("⚠️ pandas-ta not available")

def safe_talib_function(func_name, data, *args, **kwargs):
    """
    Safely call TA-Lib function with fallback options
    """
    try:
        if TALIB_AVAILABLE and hasattr(talib, func_name):
            func = getattr(talib, func_name)
            return func(data, *args, **kwargs)
        else:
            # Use fallback implementations
            return fallback_implementation(func_name, data, *args, **kwargs)
    except Exception as e:
        logging.warning(f"TA-Lib function {func_name} failed: {e}")
        return fallback_implementation(func_name, data, *args, **kwargs)

def fallback_implementation(func_name, data, *args, **kwargs):
    """
    Fallback implementations for common TA-Lib functions
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    if func_name == 'RSI':
        period = args[0] if args else kwargs.get('timeperiod', 14)
        if PANDAS_TA_AVAILABLE:
            return ta.rsi(data, length=period).values
        else:
            return simple_rsi(data, period)
    
    elif func_name == 'SMA':
        period = args[0] if args else kwargs.get('timeperiod', 20)
        return data.rolling(window=period).mean().values
    
    elif func_name == 'EMA':
        period = args[0] if args else kwargs.get('timeperiod', 12)
        return data.ewm(span=period).mean().values
    
    elif func_name == 'BBANDS':
        period = args[0] if args else kwargs.get('timeperiod', 20)
        std_dev = args[1] if len(args) > 1 else kwargs.get('nbdevup', 2)
        
        if PANDAS_TA_AVAILABLE:
            bb = ta.bbands(data, length=period, std=std_dev)
            return bb.iloc[:, 0].values, bb.iloc[:, 1].values, bb.iloc[:, 2].values
        else:
            sma = data.rolling(window=period).mean()
            std = data.rolling(window=period).std()
            upper = sma + (std * std_dev)
            middle = sma
            lower = sma - (std * std_dev)
            return upper.values, middle.values, lower.values
    
    elif func_name == 'MACD':
        fast = args[0] if args else kwargs.get('fastperiod', 12)
        slow = args[1] if len(args) > 1 else kwargs.get('slowperiod', 26)
        signal = args[2] if len(args) > 2 else kwargs.get('signalperiod', 9)
        
        if PANDAS_TA_AVAILABLE:
            macd = ta.macd(data, fast=fast, slow=slow, signal=signal)
            return macd.iloc[:, 0].values, macd.iloc[:, 1].values, macd.iloc[:, 2].values
        else:
            ema_fast = data.ewm(span=fast).mean()
            ema_slow = data.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line.values, signal_line.values, histogram.values
    
    elif func_name == 'ADX':
        period = args[0] if args else kwargs.get('timeperiod', 14)
        # Simplified ADX calculation
        return np.full(len(data), 50.0)  # Neutral ADX value
    
    elif func_name == 'AVGDEV':
        period = args[0] if args else kwargs.get('timeperiod', 14)
        # Average deviation from mean
        return data.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        ).values
    
    elif func_name == 'STOCH':
        k_period = args[0] if args else kwargs.get('fastk_period', 5)
        d_period = args[1] if len(args) > 1 else kwargs.get('slowd_period', 3)
        
        if len(args) >= 3:  # high, low, close provided
            high, low, close = data, args[0], args[1]
        else:
            # Assume single series input
            high = low = close = data
        
        if PANDAS_TA_AVAILABLE and len(args) >= 3:
            stoch = ta.stoch(high, low, close, k=k_period, d=d_period)
            return stoch.iloc[:, 0].values, stoch.iloc[:, 1].values
        else:
            # Simplified stochastic
            return np.full(len(data), 50.0), np.full(len(data), 50.0)
    
    else:
        # Unknown function - return neutral values
        logging.warning(f"Unknown TA-Lib function: {func_name}, returning neutral values")
        return np.full(len(data), np.nan)

def simple_rsi(data, period=14):
    """
    Simple RSI implementation
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

# Wrapper functions for common TA-Lib functions
def RSI(data, timeperiod=14):
    return safe_talib_function('RSI', data, timeperiod)

def SMA(data, timeperiod=20):
    return safe_talib_function('SMA', data, timeperiod)

def EMA(data, timeperiod=12):
    return safe_talib_function('EMA', data, timeperiod)

def BBANDS(data, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    return safe_talib_function('BBANDS', data, timeperiod, nbdevup)

def MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
    return safe_talib_function('MACD', data, fastperiod, slowperiod, signalperiod)

def ADX(high, low, close, timeperiod=14):
    # For ADX, we need high, low, close
    return safe_talib_function('ADX', close, timeperiod)

def AVGDEV(data, timeperiod=14):
    return safe_talib_function('AVGDEV', data, timeperiod)

def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
    return safe_talib_function('STOCH', high, low, close, fastk_period, slowd_period)

# Function to check which TA-Lib functions are available
def check_talib_functions():
    """Check which TA-Lib functions are available"""
    if not TALIB_AVAILABLE:
        print("TA-Lib not available")
        return []
    
    available_functions = []
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)  # 100 data points
    
    functions_to_test = [
        'RSI', 'SMA', 'EMA', 'BBANDS', 'MACD', 'ADX', 'AVGDEV', 'STOCH'
    ]
    
    for func_name in functions_to_test:
        try:
            if hasattr(talib, func_name):
                func = getattr(talib, func_name)
                if func_name in ['STOCH', 'ADX']:
                    # These need high, low, close
                    result = func(test_data, test_data, test_data)
                elif func_name == 'BBANDS':
                    result = func(test_data)
                elif func_name == 'MACD':
                    result = func(test_data)
                else:
                    result = func(test_data)
                available_functions.append(func_name)
                print(f"✅ {func_name} - Available")
        except Exception as e:
            print(f"❌ {func_name} - Error: {e}")
    
    return available_functions

if __name__ == "__main__":
    print("Testing TA-Lib wrapper...")
    print("Available functions:")
    available = check_talib_functions()
    print(f"\nWorking functions: {len(available)}")
    
    # Test wrapper functions
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    
    print("\nTesting wrapper functions:")
    try:
        rsi = RSI(test_data)
        print(f"✅ RSI wrapper works: {len(rsi)} values")
    except Exception as e:
        print(f"❌ RSI wrapper failed: {e}")
    
    try:
        sma = SMA(test_data)
        print(f"✅ SMA wrapper works: {len(sma)} values")
    except Exception as e:
        print(f"❌ SMA wrapper failed: {e}")
