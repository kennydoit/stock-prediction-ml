#!/usr/bin/env python3
"""
Collect historical stock price data using Yahoo Finance
"""
import sys
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime, timedelta
import logging
import time

# Add database path
sys.path.append(str(Path(__file__).parent.parent / 'database'))
from database_manager import DatabaseManager

# Add yfinance
try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceLoader:
    """Yahoo Finance data loader"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            data['Adj Close'] = data['Close']
            data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

def collect_price_data(batch_size=10, total_symbols=None):
    """Collect historical price data for all symbols in database"""
    
    print("Collecting historical stock price data using Yahoo Finance...")
    
    # Initialize data loader and database manager
    data_loader = YahooFinanceLoader()
    db_manager = DatabaseManager()
    
    # Set date range for historical data (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    print(f"Collecting data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    with db_manager:
        # Get all symbols from database
        symbols_df = db_manager.get_symbols()
        print(f"Found {len(symbols_df)} symbols in database")
        
        if total_symbols is None:
            total_symbols = len(symbols_df)
        else:
            total_symbols = min(total_symbols, len(symbols_df))
        
        success_count = 0
        error_count = 0
        
        print(f"Processing {total_symbols} symbols...")
        
        for i in range(0, total_symbols, batch_size):
            batch = symbols_df.iloc[i:i+batch_size]
            
            print(f"\nProcessing batch {i//batch_size + 1}/{(total_symbols-1)//batch_size + 1}")
            print(f"Symbols: {', '.join(batch['symbol'].tolist())}")
            
            for _, row in batch.iterrows():
                symbol = row['symbol']
                
                try:
                    # Check if we already have recent data for this symbol
                    existing_data = db_manager.get_stock_prices(
                        symbol, 
                        start_date=(end_date - timedelta(days=7)).date()
                    )
                    
                    if not existing_data.empty:
                        print(f"  {symbol}: Already has recent data, skipping...")
                        continue
                    
                    # Fetch historical data
                    print(f"  {symbol}: Fetching data...", end="")
                    
                    stock_data = data_loader.fetch_stock_data(
                        symbol, 
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if not stock_data.empty:
                        # Insert into database
                        db_manager.insert_stock_prices(stock_data, symbol)
                        print(f" ✅ {len(stock_data)} records")
                        success_count += 1
                    else:
                        print(f" ❌ No data available")
                        error_count += 1
                        
                except Exception as e:
                    print(f"  {symbol}: ❌ Error - {str(e)}")
                    error_count += 1
            
            # Small delay between batches to be respectful
            time.sleep(1)
        
        print(f"\n{'='*60}")
        print(f"HISTORICAL DATA COLLECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Success: {success_count} symbols")
        print(f"Errors: {error_count} symbols")
        print(f"Total processed: {success_count + error_count} symbols")
        
        return success_count, error_count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Collect stock price data')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--limit', type=int, help='Limit number of symbols to process')
    
    args = parser.parse_args()
    
    collect_price_data(batch_size=args.batch_size, total_symbols=args.limit)