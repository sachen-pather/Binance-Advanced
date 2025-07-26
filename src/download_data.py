# download_data.py

import os
import pandas as pd
from binance.client import Client
from datetime import datetime
import time

# --- Configuration ---
# This is the directory where your data will be saved.
DATA_DIR = "data" 

# Add all the symbols you want to test here. 
# It's good to have more than you need.
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT'] 

# The timeframe for your data. For HFT testing, 1m or 5m is best.
# For your current Phase 1 test, 1H is what you've been using.
INTERVAL = Client.KLINE_INTERVAL_1HOUR 

# The date range for the historical data you want to download.
# A 2-year period is excellent for robust backtesting.
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"
# ---------------------

def download_klines(symbol, interval, start_str, end_str):
    """
    Downloads historical kline data from Binance and saves it to a CSV file.
    """
    # Using a public client is fine for downloading historical data.
    client = Client() 
    
    # Define the path where the file will be saved.
    filepath = os.path.join(DATA_DIR, f"{symbol}-{interval}.csv")
    
    # Check if the file already exists to avoid re-downloading.
    if os.path.exists(filepath):
        print(f"Data for {symbol} ({interval}) already exists at {filepath}. Skipping download.")
        return

    print(f"Downloading {symbol} data for interval {interval} from {start_str} to {end_str}...")
    
    # This is the actual API call to Binance.
    klines = client.get_historical_klines(
        symbol,
        interval,
        start_str,
        end_str
    )

    # Convert the list of lists into a clean pandas DataFrame.
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # The 'timestamp' from Binance is in milliseconds; convert it to a human-readable datetime object.
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set the timestamp as the index, which is standard practice for time-series data.
    df.set_index('timestamp', inplace=True)
    
    # Save the DataFrame to a CSV file.
    df.to_csv(filepath)
    
    print(f"✅ Success! Saved {len(df)} rows of data to {filepath}")
    
    # Wait for a second before the next API call to be respectful to Binance's servers.
    time.sleep(1) 

if __name__ == "__main__":
    # Create the 'data' directory if it doesn't already exist.
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
        
    # Loop through each symbol in your list and download the data for it.
    for symbol in SYMBOLS:
        download_klines(symbol, INTERVAL, START_DATE, END_DATE)
    
    print("\n--- All data downloads complete. ---")