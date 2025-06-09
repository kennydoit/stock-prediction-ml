"""
Database manager for stock prediction ML project
Handles database connections, schema creation, and basic operations
"""
import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date
import yaml

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages database operations for stock prediction ML project
    """
    
    def __init__(self, db_path: str = None, config_path: str = '../config.yaml'):
        """
        Initialize database manager
        
        Parameters:
        -----------
        db_path : str
            Path to SQLite database file
        config_path : str
            Path to configuration file
        """
        if db_path is None:
            db_path = str(Path(__file__).parent / 'stock_prediction.db')
        
        self.db_path = db_path
        self.config_path = config_path
        self.connection = None
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
            self.config = {}
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def execute_script(self, script_path: str):
        """
        Execute SQL script from file
        
        Parameters:
        -----------
        script_path : str
            Path to SQL script file
        """
        if not self.connection:
            self.connect()
        
        try:
            with open(script_path, 'r') as f:
                script = f.read()
            
            self.connection.executescript(script)
            self.connection.commit()
            logger.info(f"Successfully executed script: {script_path}")
        except Exception as e:
            logger.error(f"Failed to execute script {script_path}: {e}")
            self.connection.rollback()
            raise
    
    def setup_database(self):
        """Initialize database schema"""
        schema_path = Path(__file__).parent / 'schema.sql'
        self.execute_script(str(schema_path))
        logger.info("Database schema initialized")
    
    def insert_symbol(self, symbol: str, name: str = None, sector: str = None, 
                     market_cap: str = None, exchange: str = None) -> int:
        """
        Insert or update symbol information
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        name : str
            Company name
        sector : str
            Sector classification
        market_cap : str
            Market cap category
        exchange : str
            Exchange name
            
        Returns:
        --------
        int
            Symbol ID
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            # Try to insert, update if exists
            cursor.execute("""
                INSERT OR REPLACE INTO symbols 
                (symbol, name, sector, market_cap, exchange, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (symbol, name, sector, market_cap, exchange))
            
            # Get the symbol ID
            cursor.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
            symbol_id = cursor.fetchone()[0]
            
            self.connection.commit()
            logger.debug(f"Inserted/updated symbol {symbol} with ID {symbol_id}")
            return symbol_id
            
        except Exception as e:
            logger.error(f"Failed to insert symbol {symbol}: {e}")
            self.connection.rollback()
            raise
    
    def insert_stock_prices(self, df: pd.DataFrame, symbol: str):
        """
        Insert stock price data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock price data with OHLCV columns
        symbol : str
            Stock symbol
        """
        if not self.connection:
            self.connect()
        
        # Get or create symbol ID
        symbol_id = self.insert_symbol(symbol)
        
        try:
            # Prepare data for insertion
            records = []
            for date_idx, row in df.iterrows():
                if isinstance(date_idx, str):
                    date_val = pd.to_datetime(date_idx).date()
                else:
                    date_val = date_idx.date()
                
                records.append((
                    symbol_id,
                    date_val,
                    float(row.get('Open', row.get('open', None))),
                    float(row.get('High', row.get('high', None))),
                    float(row.get('Low', row.get('low', None))),
                    float(row.get('Close', row.get('close', None))),
                    float(row.get('Adj Close', row.get('adj_close', row.get('Close', row.get('close', None))))),
                    int(row.get('Volume', row.get('volume', 0)))
                ))
            
            # Insert records
            cursor = self.connection.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO stock_prices 
                (symbol_id, date, open_price, high_price, low_price, close_price, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            self.connection.commit()
            logger.info(f"Inserted {len(records)} price records for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to insert stock prices for {symbol}: {e}")
            self.connection.rollback()
            raise
    
    def insert_news_article(self, headline: str, summary: str = None, content: str = None,
                           author: str = None, source: str = None, url: str = None,
                           published_at: datetime = None, symbols: List[str] = None) -> int:
        """
        Insert news article and link to symbols
        
        Parameters:
        -----------
        headline : str
            Article headline
        summary : str
            Article summary
        content : str
            Full article content
        author : str
            Article author
        source : str
            News source
        url : str
            Article URL
        published_at : datetime
            Publication timestamp
        symbols : List[str]
            List of related symbols
            
        Returns:
        --------
        int
            News article ID
        """
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            # Insert news article
            cursor.execute("""
                INSERT INTO news_articles 
                (headline, summary, content, author, source, url, published_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (headline, summary, content, author, source, url, published_at))
            
            news_id = cursor.lastrowid
            
            # Link to symbols if provided
            if symbols:
                for symbol in symbols:
                    symbol_id = self.insert_symbol(symbol)
                    cursor.execute("""
                        INSERT OR IGNORE INTO news_symbols (news_id, symbol_id)
                        VALUES (?, ?)
                    """, (news_id, symbol_id))
            
            self.connection.commit()
            logger.debug(f"Inserted news article {news_id}: {headline[:50]}...")
            return news_id
            
        except Exception as e:
            logger.error(f"Failed to insert news article: {e}")
            self.connection.rollback()
            raise
    
    def get_symbols(self) -> pd.DataFrame:
        """Get all symbols from database"""
        if not self.connection:
            self.connect()
        
        return pd.read_sql_query("""
            SELECT id, symbol, name, sector, market_cap, exchange, is_active
            FROM symbols 
            WHERE is_active = 1
            ORDER BY symbol
        """, self.connection)
    
    def get_stock_prices(self, symbol: str, start_date: date = None, 
                        end_date: date = None) -> pd.DataFrame:
        """
        Get stock price data for symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        start_date : date
            Start date for data
        end_date : date
            End date for data
            
        Returns:
        --------
        pd.DataFrame
            Stock price data
        """
        if not self.connection:
            self.connect()
        
        query = """
            SELECT sp.date, sp.open_price as open, sp.high_price as high,
                   sp.low_price as low, sp.close_price as close,
                   sp.adj_close, sp.volume
            FROM stock_prices sp
            JOIN symbols s ON sp.symbol_id = s.id
            WHERE s.symbol = ?
        """
        params = [symbol]
        
        if start_date:
            query += " AND sp.date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND sp.date <= ?"
            params.append(end_date)
        
        query += " ORDER BY sp.date"
        
        df = pd.read_sql_query(query, self.connection, params=params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        return df
    
    def get_news_for_symbol(self, symbol: str, start_date: date = None,
                           end_date: date = None, limit: int = None) -> pd.DataFrame:
        """
        Get news articles for symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        start_date : date
            Start date for news
        end_date : date
            End date for news
        limit : int
            Maximum number of articles
            
        Returns:
        --------
        pd.DataFrame
            News articles data
        """
        if not self.connection:
            self.connect()
        
        query = """
            SELECT na.id, na.headline, na.summary, na.author, na.source,
                   na.url, na.published_at, na.sentiment_score, na.sentiment_label
            FROM news_articles na
            JOIN news_symbols ns ON na.id = ns.news_id
            JOIN symbols s ON ns.symbol_id = s.id
            WHERE s.symbol = ?
        """
        params = [symbol]
        
        if start_date:
            query += " AND DATE(na.published_at) >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND DATE(na.published_at) <= ?"
            params.append(end_date)
        
        query += " ORDER BY na.published_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, self.connection, params=params)
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
        
        return df
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """
        Clean up old data to manage database size
        
        Parameters:
        -----------
        days_to_keep : int
            Number of days of data to keep
        """
        if not self.connection:
            self.connect()
        
        cutoff_date = datetime.now().date() - pd.Timedelta(days=days_to_keep)
        
        cursor = self.connection.cursor()
        
        try:
            # Clean old news articles
            cursor.execute("""
                DELETE FROM news_articles 
                WHERE DATE(published_at) < ?
            """, (cutoff_date,))
            
            news_deleted = cursor.rowcount
            
            # Clean old stock prices (keep more history)
            stock_cutoff = datetime.now().date() - pd.Timedelta(days=days_to_keep * 3)
            cursor.execute("""
                DELETE FROM stock_prices 
                WHERE date < ?
            """, (stock_cutoff,))
            
            prices_deleted = cursor.rowcount
            
            self.connection.commit()
            logger.info(f"Cleaned up {news_deleted} old news articles and {prices_deleted} old price records")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            self.connection.rollback()
            raise