"""
Data loader module for fetching stock data from APIs or local CSV files.
"""
import pandas as pd
import yfinance as yf
from typing import List, Optional

class StockDataLoader:
    def __init__(self, config_path: str):
        """Initialize the data loader with configuration."""
        self.config_path = config_path
        
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance API.
        
        Args:
            symbol (str): Stock symbol (e.g., 'MSFT')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data
        """
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        return df
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str) -> None:
        """Save data to CSV in the raw data directory."""
        df.to_csv(f"data/raw/{symbol}_data.csv")
        
    def load_from_csv(self, symbol: str) -> pd.DataFrame:
        """Load data from a local CSV file."""
        return pd.read_csv(f"data/raw/{symbol}_data.csv", index_col=0, parse_dates=True)
