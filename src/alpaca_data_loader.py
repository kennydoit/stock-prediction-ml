import alpaca_trade_api as tradeapi
import pandas as pd
import yaml
from datetime import datetime, timedelta
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AlpacaDataLoader:
    """
    Data loader for fetching stock data from Alpaca Markets API
    """
    
    def __init__(self, config_path: str):
        """
        Initialize Alpaca API connection
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file containing Alpaca credentials
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.alpaca_config = config.get('alpaca', {})
        
        # Initialize Alpaca API with correct base URL
        self.api = tradeapi.REST(
            key_id=self.alpaca_config.get('api_key'),
            secret_key=self.alpaca_config.get('secret_key'),
            base_url=self.alpaca_config.get('base_url', 'https://paper-api.alpaca.markets')
            # Remove api_version parameter - it's causing the duplicate /v2
        )
        
        logger.info("Alpaca API connection initialized")
    
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str, 
                        timeframe: str = '1Day') -> pd.DataFrame:
        """
        Fetch historical stock data from Alpaca
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'AAPL')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        timeframe : str
            Data timeframe ('1Min', '1Hour', '1Day', etc.)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
            
            # Get bars from Alpaca
            bars = self.api.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                adjustment='raw'
            ).df
            
            if bars.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Debug: Check what we have
            logger.info(f"Raw columns from Alpaca: {list(bars.columns)}")
            logger.info(f"DataFrame shape: {bars.shape}")
            logger.info(f"Index type: {type(bars.index)}")
            logger.info(f"Index name: {bars.index.name}")
            
            # The timestamp is already in the index, not as a column
            # Just rename the columns to our standard format
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            # Select and rename only the columns we need
            available_columns = [col for col in column_mapping.keys() if col in bars.columns]
            bars_clean = bars[available_columns].copy()
            bars_clean = bars_clean.rename(columns=column_mapping)
            
            # Ensure timezone-aware datetime index
            if bars_clean.index.tz is None:
                bars_clean.index = bars_clean.index.tz_localize('America/New_York')
            else:
                bars_clean.index = bars_clean.index.tz_convert('America/New_York')
            
            logger.info(f"Successfully fetched {len(bars_clean)} records for {symbol}")
            logger.info(f"Final columns: {list(bars_clean.columns)}")
            return bars_clean
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_account_info(self) -> dict:
        """
        Get account information
        
        Returns:
        --------
        dict
            Account information
        """
        try:
            account = self.api.get_account()
            return {
                'account_id': account.id,
                'status': account.status,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            # Check if it's an authentication issue
            if "404" in str(e) or "401" in str(e):
                logger.error("Authentication failed. Please check your API keys and base URL.")
                logger.error("Make sure you have valid Alpaca API keys in your config.yaml")
            return {}
    
    def get_positions(self) -> pd.DataFrame:
        """
        Get current positions
        
        Returns:
        --------
        pd.DataFrame
            Current positions
        """
        try:
            positions = self.api.list_positions()
            if not positions:
                return pd.DataFrame()
            
            position_data = []
            for position in positions:
                position_data.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized.plpc)
                })
            
            return pd.DataFrame(position_data)
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return pd.DataFrame()


class UnifiedDataLoader:
    """
    Unified data loader supporting both yfinance and Alpaca
    """
    def __init__(self, config_path, source='alpaca'):
        self.source = source
        self.config_path = config_path
        
        if source == 'alpaca':
            self.alpaca_loader = AlpacaDataLoader(config_path)
        elif source == 'yfinance':
            # Import only when needed to avoid dependency issues
            from .data_loader import StockDataLoader
            self.yf_loader = StockDataLoader(config_path)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def fetch_stock_data(self, symbol, start_date, end_date):
        if self.source == 'alpaca':
            return self.alpaca_loader.fetch_stock_data(symbol, start_date, end_date)
        else:
            return self.yf_loader.fetch_stock_data(symbol, start_date, end_date)


# Example usage (only when running as main module)
if __name__ == "__main__":
    # Test the Alpaca data loader
    print("Testing Alpaca data loader...")
    
    loader = AlpacaDataLoader('../config.yaml')
    data = loader.fetch_stock_data('AAPL', '2024-01-01', '2024-01-10')
    print(f"Fetched {len(data)} records")
    
    if not data.empty:
        print(data.head())