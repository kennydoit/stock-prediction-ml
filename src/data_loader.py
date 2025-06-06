"""
Data loader module for fetching stock data from Alpaca Markets API.
"""
import pandas as pd
import yaml
import logging
from typing import Optional
from .alpaca_data_loader import AlpacaDataLoader

logger = logging.getLogger(__name__)

class StockDataLoader:
    """
    Stock data loader using Alpaca Markets API as the primary data source.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the data loader with Alpaca configuration.
        
        Args:
            config_path (str): Path to configuration file containing Alpaca credentials
        """
        self.config_path = config_path
        self.alpaca_loader = AlpacaDataLoader(config_path)
        logger.info("StockDataLoader initialized with Alpaca data source")
        
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch stock data from Alpaca Markets API.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data with timezone-aware datetime index
        """
        return self.alpaca_loader.fetch_stock_data(symbol, start_date, end_date)
    
    def fetch_multiple_stocks(self, symbols: list, start_date: str, end_date: str) -> dict:
        """
        Fetch data for multiple stock symbols.
        
        Args:
            symbols (list): List of stock symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Dictionary with symbols as keys and DataFrames as values
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.fetch_stock_data(symbol, start_date, end_date)
                if not df.empty:
                    data[symbol] = df
                    logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                else:
                    logger.warning(f"No data retrieved for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                
        return data
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str, data_dir: str = "data/raw") -> None:
        """
        Save data to CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            symbol (str): Stock symbol for filename
            data_dir (str): Directory to save the file
        """
        import os
        os.makedirs(data_dir, exist_ok=True)
        filepath = f"{data_dir}/{symbol}_data.csv"
        df.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
        
    def load_from_csv(self, symbol: str, data_dir: str = "data/raw") -> pd.DataFrame:
        """
        Load data from a local CSV file.
        
        Args:
            symbol (str): Stock symbol
            data_dir (str): Directory containing the file
            
        Returns:
            pd.DataFrame: Loaded DataFrame with datetime index
        """
        filepath = f"{data_dir}/{symbol}_data.csv"
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Data loaded from {filepath}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
    
    def get_account_info(self) -> dict:
        """
        Get Alpaca account information.
        
        Returns:
            dict: Account information
        """
        return self.alpaca_loader.get_account_info()
    
    def get_positions(self) -> pd.DataFrame:
        """
        Get current Alpaca positions.
        
        Returns:
            pd.DataFrame: Current positions
        """
        return self.alpaca_loader.get_positions()


# For backward compatibility, create a simple factory function
def create_data_loader(config_path: str) -> StockDataLoader:
    """
    Factory function to create a data loader instance.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        StockDataLoader: Configured data loader instance
    """
    return StockDataLoader(config_path)


# Example usage
if __name__ == "__main__":
    # Test the data loader
    loader = StockDataLoader('../config.yaml')
    
    # Test single stock
    data = loader.fetch_stock_data('AAPL', '2024-01-01', '2024-01-10')
    print(f"Fetched {len(data)} records for AAPL")
    
    # Test multiple stocks
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    multi_data = loader.fetch_multiple_stocks(symbols, '2024-01-01', '2024-01-05')
    print(f"Fetched data for {len(multi_data)} symbols")
