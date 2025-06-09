# Create file: scripts/temporary_scripts/check_database_content.py
"""
Check what's actually in the database
"""

import sys
from pathlib import Path
import pandas as pd

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent / 'database'))

def check_database_content():
    """Check what's actually in the database"""
    
    print("🔍 Checking Database Content")
    print("="*30)
    
    try:
        from database_manager import DatabaseManager
        
        with DatabaseManager() as db:
            print("✅ Database connected")
            
            # Check symbols table
            try:
                symbols_df = db.get_symbols()
                print(f"\n📊 Symbols table: {len(symbols_df)} records")
                if not symbols_df.empty:
                    print(f"Sample symbols: {symbols_df['symbol'].head(10).tolist()}")
                    
                    # Check for ACN specifically
                    if 'ACN' in symbols_df['symbol'].values:
                        print("✅ ACN found in symbols table")
                    else:
                        print("❌ ACN not found in symbols table")
                else:
                    print("⚠️ Symbols table is empty")
            except Exception as e:
                print(f"❌ Error getting symbols: {e}")
            
            # Check stock_prices table with proper DataFrame handling
            try:
                test_symbols = ['ACN', 'AAPL', 'MSFT', 'SPY']
                
                for symbol in test_symbols:
                    try:
                        prices = db.get_stock_prices(symbol)
                        
                        # Handle DataFrame return properly
                        if prices is not None:
                            if isinstance(prices, pd.DataFrame):
                                if not prices.empty:  # This is the correct way for DataFrames
                                    print(f"📈 {symbol}: {len(prices)} price records (DataFrame)")
                                    
                                    # Show date range
                                    date_cols = [col for col in prices.columns if 'date' in str(col).lower()]
                                    if date_cols:
                                        date_col = date_cols[0]
                                        prices[date_col] = pd.to_datetime(prices[date_col], errors='coerce')
                                        min_date = prices[date_col].min()
                                        max_date = prices[date_col].max()
                                        print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                                        
                                        # Show latest price
                                        price_cols = [col for col in prices.columns if 'close' in str(col).lower()]
                                        if price_cols:
                                            latest_price = prices[price_cols[0]].iloc[-1]
                                            print(f"  Latest price: ${latest_price:.2f}")
                                        
                                        # Show columns
                                        print(f"  Columns: {list(prices.columns)}")
                                else:
                                    print(f"❌ {symbol}: Empty DataFrame")
                            elif isinstance(prices, list):
                                if len(prices) > 0:
                                    print(f"📈 {symbol}: {len(prices)} price records (List)")
                                else:
                                    print(f"❌ {symbol}: Empty list")
                            else:
                                print(f"❌ {symbol}: Unexpected type {type(prices)}")
                        else:
                            print(f"❌ {symbol}: None returned")
                            
                    except Exception as e:
                        print(f"❌ {symbol}: Error getting prices - {e}")
                        
            except Exception as e:
                print(f"❌ Error checking stock prices: {e}")
            
            # Direct SQL query for stock_prices table (with correct column names)
            try:
                if hasattr(db, 'connection'):
                    cursor = db.connection.cursor()
                elif hasattr(db, 'conn'):
                    cursor = db.conn.cursor()
                else:
                    print("⚠️ Cannot access database cursor")
                    return
                
                # Check table structure
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                print(f"\n📋 Available tables: {[t[0] for t in tables]}")
                
                # Check stock_prices table content
                cursor.execute("SELECT COUNT(*) FROM stock_prices;")
                count = cursor.fetchone()[0]
                print(f"📊 stock_prices table: {count:,} total records")
                
                if count > 0:
                    # Get sample data with proper column access
                    cursor.execute("""
                        SELECT sp.date, s.symbol, sp.close_price, sp.volume 
                        FROM stock_prices sp
                        JOIN symbols s ON sp.symbol_id = s.id
                        ORDER BY sp.date DESC
                        LIMIT 5;
                    """)
                    sample = cursor.fetchall()
                    
                    print(f"\n📋 Recent price records:")
                    for row in sample:
                        date, symbol, close_price, volume = row
                        print(f"  {date} | {symbol} | ${close_price:.2f} | Vol: {volume:,}")
                    
                    # Check unique symbols in stock_prices
                    cursor.execute("""
                        SELECT s.symbol, COUNT(*) as record_count
                        FROM stock_prices sp
                        JOIN symbols s ON sp.symbol_id = s.id
                        GROUP BY s.symbol
                        ORDER BY record_count DESC
                        LIMIT 10;
                    """)
                    symbols_in_prices = cursor.fetchall()
                    print(f"\n📊 Top symbols by record count:")
                    for symbol, count in symbols_in_prices:
                        print(f"  {symbol}: {count:,} records")
                    
                    # Check date ranges
                    cursor.execute("SELECT MIN(date), MAX(date) FROM stock_prices;")
                    date_range = cursor.fetchone()
                    print(f"\n📅 Date range in stock_prices: {date_range[0]} to {date_range[1]}")
                    
                    # Check recent data (last 30 days) with correct join
                    cursor.execute("""
                        SELECT s.symbol, COUNT(*) as recent_count
                        FROM stock_prices sp
                        JOIN symbols s ON sp.symbol_id = s.id
                        WHERE sp.date >= date('now', '-30 days') 
                        GROUP BY s.symbol 
                        ORDER BY recent_count DESC 
                        LIMIT 10;
                    """)
                    recent_data = cursor.fetchall()
                    print(f"\n📈 Symbols with recent data (last 30 days):")
                    for symbol, count in recent_data:
                        print(f"  {symbol}: {count} records")
                    
                    # Check ACN specifically
                    cursor.execute("""
                        SELECT sp.date, sp.close_price 
                        FROM stock_prices sp
                        JOIN symbols s ON sp.symbol_id = s.id
                        WHERE s.symbol = 'ACN'
                        ORDER BY sp.date DESC
                        LIMIT 5;
                    """)
                    acn_recent = cursor.fetchall()
                    if acn_recent:
                        print(f"\n📈 Recent ACN prices:")
                        for date, price in acn_recent:
                            print(f"  {date}: ${price:.2f}")
                    else:
                        print(f"\n❌ No ACN price data found")
                        
                else:
                    print("⚠️ stock_prices table is empty")
                    
            except Exception as e:
                print(f"❌ Error with direct SQL query: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database_content()