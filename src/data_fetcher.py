"""
Data fetching module for historical data and API interactions.
--- MODIFIED to include local data fetching for backtesting ---
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os  # Import the 'os' module to handle file paths
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import ENV_CONFIG, BINANCE_CONFIG

logger = logging.getLogger("BinanceTrading.DataFetcher")


class DataFetcher:
    """Handle data fetching operations with Binance API"""
    
    def __init__(self, client=None):
        # If no client is provided, create one using the config
        if client is None:
            try:
                self.client = Client(BINANCE_CONFIG['api_key'], BINANCE_CONFIG['api_secret'])
                logger.info("DataFetcher initialized with API credentials from config")
            except Exception as e:
                logger.error(f"Failed to create Binance client: {e}")
                # Fallback to a client without credentials (for public data only)
                self.client = Client()
                logger.warning("Using public client - live trading features will not work")
        else:
            self.client = client

    # --- NEW METHOD FOR LOCAL BACKTESTING ---
    def get_historical_data_from_local(self, symbol: str, interval: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
        """
        Get historical data from a local CSV file for backtesting.
        This significantly speeds up the backtesting process.
        """
        try:
            # Construct the filename based on the convention from download_data.py
            # e.g., "data/BTCUSDT-1h.csv"
            filepath = os.path.join("data", f"{symbol}-{interval}.csv")
            
            if not os.path.exists(filepath):
                # If the local file doesn't exist, log a warning and return None.
                # The backtester should handle this gracefully.
                logger.warning(f"Local data file not found: {filepath}. Cannot load data for backtest.")
                return None

            # Read the entire CSV file into a pandas DataFrame.
            # `parse_dates=True` tells pandas to automatically convert the 'timestamp' column to datetime objects.
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            
            # Filter the DataFrame to only include the dates required for this specific backtest run.
            df = df.loc[start_date_str:end_date_str]
            
            # Make sure the data columns are in the correct numeric format.
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.info(f"Successfully loaded {len(df)} rows for {symbol} from local file: {filepath}")
            return df

        except Exception as e:
            logger.error(f"Error loading local data for {symbol}: {e}. Returning None.", exc_info=True)
            return None
    # --- END OF NEW METHOD ---


    def get_historical_data(self, symbol='BTCUSDT', interval='1h', lookback='14 days'):
        """Get historical klines/candlestick data from Binance API with error handling and retries"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Calculate start time
                lookback_time = datetime.now() - pd.Timedelta(lookback)
                
                logger.debug(f"Fetching historical data for {symbol} from {lookback_time.strftime('%Y-%m-%d %H:%M')}")
                
                klines = self.client.get_historical_klines(
                    symbol,
                    interval,
                    str(int(lookback_time.timestamp() * 1000))
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 
                    'volume', 'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                df.set_index('timestamp', inplace=True)
                
                if len(df) == 0:
                    logger.warning(f"No data returned for {symbol} {interval}")
                    return None
                    
                return df
                
            except BinanceAPIException as e:
                logger.error(f"API error getting historical data (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Error getting historical data (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
    
    def get_account_equity(self, paper_trade=False, default_equity=10000):
        """Calculate total account equity in USDT with paper trading support"""
        # If in paper trading mode, return the simulated equity
        if paper_trade:
            logger.info("Paper trading mode: Using simulated equity")
            return default_equity
            
        # Check if we have a properly authenticated client
        if not hasattr(self.client, 'API_KEY') or not self.client.API_KEY:
            logger.warning("No API key configured - using default equity")
            return default_equity
            
        # Original API-based code for live trading
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                logger.info("Attempting to get live account equity...")
                account = self.client.get_account()
                
                # Get all balances
                balances = {
                    asset['asset']: float(asset['free']) + float(asset['locked'])
                    for asset in account['balances']
                    if float(asset['free']) > 0 or float(asset['locked']) > 0
                }
                
                logger.info(f"Found balances for {len(balances)} assets")
                
                # Convert all to USDT value
                usdt_values = {}
                for asset, amount in balances.items():
                    if asset == 'USDT':
                        usdt_values[asset] = amount
                        logger.debug(f"USDT balance: {amount}")
                    else:
                        try:
                            # Try to find a direct USDT pair first
                            ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                            price = float(ticker['price'])
                            usdt_values[asset] = amount * price
                            logger.debug(f"{asset} balance: {amount} * {price} = {usdt_values[asset]} USDT")
                        except BinanceAPIException:
                            # If direct USDT pair doesn't exist, try BTC intermediate
                            try:
                                btc_ticker = self.client.get_symbol_ticker(symbol=f"{asset}BTC")
                                btc_price = float(btc_ticker['price'])
                                btc_usdt_price = float(self.client.get_symbol_ticker(symbol="BTCUSDT")['price'])
                                usdt_values[asset] = amount * btc_price * btc_usdt_price
                                logger.debug(f"{asset} via BTC: {amount} * {btc_price} * {btc_usdt_price} = {usdt_values[asset]} USDT")
                            except BinanceAPIException:
                                # If still cannot convert, skip this asset
                                logger.warning(f"Skipping asset {asset}: no direct or BTC conversion path to USDT found.")
                                usdt_values[asset] = 0
                
                total_equity = sum(usdt_values.values())
                logger.info(f"Total account equity calculated: {total_equity:.2f} USDT")
                return total_equity
                
            except Exception as e:
                logger.warning(f"Error getting account equity (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to get account equity after {max_retries} attempts. Falling back to default.")
                    return default_equity  # Default value if cannot fetch
                    
    def get_tradable_symbols(self, volume_threshold=1000000):
        """Get list of tradable symbols with sufficient volume"""
        try:
            exchange_info = self.client.get_exchange_info()
            tickers = self.client.get_ticker()
            
            # Filter symbols
            symbols = []
            for symbol_info in exchange_info['symbols']:
                # Only USDT pairs for simplicity, and ensure it's a tradable spot asset
                if (symbol_info['quoteAsset'] == 'USDT' and 
                    symbol_info['status'] == 'TRADING' and 
                    'SPOT' in symbol_info['permissions']):
                    
                    symbol = symbol_info['symbol']
                    
                    # Get 24h volume
                    ticker = next((t for t in tickers if t['symbol'] == symbol), None)
                    if ticker and 'quoteVolume' in ticker:
                        # Use quoteVolume for a more accurate representation of USDT volume
                        if float(ticker['quoteVolume']) > volume_threshold:
                            symbols.append({
                                'symbol': symbol,
                                'baseAsset': symbol_info['baseAsset'],
                                'quoteAsset': symbol_info['quoteAsset'],
                                'volume': float(ticker['quoteVolume'])
                            })
            
            # Sort by volume
            symbols.sort(key=lambda x: x['volume'], reverse=True)
            logger.info(f"Found {len(symbols)} tradable symbols above ${volume_threshold:,.0f} 24h volume.")
            return symbols[:20]  # Return the top 20 by volume
        except Exception as e:
            logger.error(f"Error getting tradable symbols: {e}")
            logger.warning("Falling back to a default list of high-volume symbols.")
            return [
                {'symbol': 'BTCUSDT'}, {'symbol': 'ETHUSDT'}, {'symbol': 'BNBUSDT'}, 
                {'symbol': 'SOLUSDT'}, {'symbol': 'XRPUSDT'}, {'symbol': 'ADAUSDT'},
                {'symbol': 'DOGEUSDT'}, {'symbol': 'AVAXUSDT'}, {'symbol': 'LINKUSDT'},
                {'symbol': 'MATICUSDT'}
            ]