import alpaca_trade_api as tradeapi
import pandas as pd
import yaml
from datetime import datetime, timedelta
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

class AlpacaDataLoader:
    """
    Data loader for fetching stock data and news from Alpaca Markets API
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
            return bars_clean
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_news(self, symbol: str = None, start_date: str = None, 
                   end_date: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Fetch news articles from Alpaca
        
        Parameters:
        -----------
        symbol : str, optional
            Stock symbol to filter news (e.g., 'AAPL')
            If None, fetches general market news
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        limit : int
            Maximum number of articles to fetch (default: 100, max: 1000)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with news articles including timestamp, headline, summary, symbols
        """
        try:
            logger.info(f"Fetching news for symbol: {symbol}, dates: {start_date} to {end_date}")
            
            # Prepare parameters for news API
            params = {
                'limit': min(limit, 1000)  # Alpaca max limit is 1000
            }
            
            # Note: Alpaca API uses 'symbol' not 'symbols'
            if symbol:
                params['symbol'] = symbol
            
            if start_date:
                params['start'] = start_date
                
            if end_date:
                params['end'] = end_date
            
            # Debug: Print the actual API call parameters
            logger.info(f"API call parameters: {params}")
            
            # Fetch news from Alpaca
            news = self.api.get_news(**params)
            
            if not news:
                logger.warning("No news articles found")
                return pd.DataFrame()
            
            # Convert news to DataFrame
            news_data = []
            for article in news:
                try:
                    news_data.append({
                        'timestamp': pd.to_datetime(article.created_at),
                        'headline': article.headline,
                        'summary': getattr(article, 'summary', ''),
                        'content': getattr(article, 'content', ''),
                        'author': getattr(article, 'author', ''),
                        'symbols': getattr(article, 'symbols', []),
                        'url': getattr(article, 'url', ''),
                        'article_id': getattr(article, 'id', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing article: {e}")
                    continue
            
            if not news_data:
                logger.warning("No valid news articles processed")
                return pd.DataFrame()
            
            df = pd.DataFrame(news_data)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Ensure timezone-aware datetime index
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            # Sort by timestamp (newest first)
            df.sort_index(ascending=False, inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} news articles")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            # Print more detailed error information
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def fetch_news_for_symbol(self, symbol: str, days_back: int = 7, limit: int = 50) -> pd.DataFrame:
        """
        Fetch recent news for a specific symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'AAPL')
        days_back : int
            Number of days to look back for news (default: 7)
        limit : int
            Maximum number of articles (default: 50)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with news articles for the symbol
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        return self.fetch_news(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    
    def test_api_connection(self) -> dict:
        """
        Test the API connection and return status information
        
        Returns:
        --------
        dict
            Connection status and capabilities
        """
        result = {
            'connection_status': 'unknown',
            'account_access': False,
            'news_access': False,
            'data_access': False,
            'error_messages': []
        }
        
        try:
            # Test account access
            account = self.api.get_account()
            result['account_access'] = True
            result['connection_status'] = 'connected'
            logger.info("✅ Account access working")
        except Exception as e:
            result['error_messages'].append(f"Account access failed: {str(e)}")
            logger.error(f"❌ Account access failed: {str(e)}")
        
        try:
            # Test data access with a simple stock data request
            test_data = self.api.get_bars('AAPL', '1Day', limit=1)
            if test_data:
                result['data_access'] = True
                logger.info("✅ Stock data access working")
            else:
                result['error_messages'].append("Stock data access returned empty result")
        except Exception as e:
            result['error_messages'].append(f"Stock data access failed: {str(e)}")
            logger.error(f"❌ Stock data access failed: {str(e)}")
        
        try:
            # Test news access
            test_news = self.api.get_news(limit=1)
            if test_news:
                result['news_access'] = True
                logger.info("✅ News access working")
            else:
                result['error_messages'].append("News access returned empty result")
        except Exception as e:
            result['error_messages'].append(f"News access failed: {str(e)}")
            logger.error(f"❌ News access failed: {str(e)}")
        
        return result
    
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


# Example usage (only when running as main module)
if __name__ == "__main__":
    # Test the Alpaca data loader
    print("Testing Alpaca data loader...")
    
    loader = AlpacaDataLoader('../config.yaml')
    
    # Test API connection
    print("Testing API connection...")
    connection_status = loader.test_api_connection()
    print(f"Connection status: {connection_status}")
    
    # Test stock data
    print("\nTesting stock data...")
    data = loader.fetch_stock_data('AAPL', '2024-01-01', '2024-01-10')
    print(f"Fetched {len(data)} stock records")
    
    # Test news data
    print("\nTesting news data...")
    news = loader.fetch_news_for_symbol('AAPL', days_back=3, limit=5)
    print(f"Fetched {len(news)} news articles")
    
    if not news.empty:
        print("\nSample headlines:")
        for idx, row in news.head(3).iterrows():
            print(f"- {row['headline']}")