# Update validate_price_data.py to first inspect the DatabaseManager
"""
Simple ACN price data export using direct SQL
"""

import sys
from pathlib import Path
import pandas as pd

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

def inspect_database_manager():
    """Check what methods DatabaseManager actually has"""
    
    print("🔍 Inspecting DatabaseManager")
    print("="*30)
    
    try:
        from database.database_manager import DatabaseManager
        
        # Show all available methods
        methods = [method for method in dir(DatabaseManager) if not method.startswith('_')]
        print(f"📋 Available methods:")
        for method in methods:
            print(f"  - {method}")
        
        # Test connection
        with DatabaseManager() as db:
            print("\n✅ Database connection successful")
            
            # Check if it has cursor or connection attributes
            if hasattr(db, 'cursor'):
                print("✅ Has cursor attribute")
                cursor = db.cursor
                
                # Try to get table names
                try:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    print(f"📋 Tables: {[t[0] for t in tables]}")
                    
                    # Try to get ACN data using cursor directly
                    cursor.execute("SELECT * FROM stock_data WHERE Symbol = 'ACN' LIMIT 5;")
                    sample_data = cursor.fetchall()
                    print(f"📊 Sample ACN data: {len(sample_data)} records")
                    for row in sample_data:
                        print(f"  {row}")
                        
                except Exception as e:
                    print(f"❌ Cursor error: {e}")
            
            elif hasattr(db, 'conn'):
                print("✅ Has conn attribute")
                conn = db.conn
                
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    print(f"📋 Tables: {[t[0] for t in tables]}")
                    
                    # Try to get ACN data
                    cursor.execute("SELECT * FROM stock_data WHERE Symbol = 'ACN' LIMIT 5;")
                    sample_data = cursor.fetchall()
                    print(f"📊 Sample ACN data: {len(sample_data)} records")
                    for row in sample_data:
                        print(f"  {row}")
                        
                except Exception as e:
                    print(f"❌ Connection error: {e}")
            
            else:
                print("⚠️ No obvious cursor or connection attribute")
                
                # Try the get_stock_prices method we know works
                try:
                    acn_data = db.get_stock_prices('ACN')
                    print(f"📊 get_stock_prices works: {type(acn_data)}, length: {len(acn_data) if acn_data else 0}")
                    
                    if acn_data and len(acn_data) > 0:
                        print(f"📋 First record: {acn_data[0]}")
                        
                except Exception as e:
                    print(f"❌ get_stock_prices error: {e}")
            
    except Exception as e:
        print(f"❌ DatabaseManager inspection error: {e}")
        import traceback
        traceback.print_exc()

def export_acn_using_get_stock_prices():
    """Export ACN using the method we know works"""
    
    print("\n📊 Export using get_stock_prices")
    print("="*35)
    
    try:
        from database.database_manager import DatabaseManager
        
        with DatabaseManager() as db:
            print("✅ Database connected")
            
            # Use the method we know works
            acn_data = db.get_stock_prices('ACN')
            
            if not acn_data:
                print("❌ No ACN data found")
                return
            
            print(f"📊 Found {len(acn_data)} ACN records")
            print(f"📋 Data type: {type(acn_data)}")
            
            if isinstance(acn_data, list) and len(acn_data) > 0:
                print(f"📋 First record structure: {acn_data[0]}")
                print(f"📋 Record type: {type(acn_data[0])}")
                
                # Convert to DataFrame
                df = pd.DataFrame(acn_data)
                print(f"📋 DataFrame columns: {list(df.columns)}")
                
                # Export to CSV
                output_file = Path(__file__).parent / 'ACN_get_stock_prices_export.csv'
                df.to_csv(output_file, index=False)
                
                print(f"✅ Exported to: {output_file}")
                print(f"📊 Records: {len(df)}")
                
                # Show preview
                print(f"\n📋 First 3 rows:")
                print(df.head(3).to_string(index=False))
                
                print(f"\n📋 Last 3 rows:")
                print(df.tail(3).to_string(index=False))
                
                # Try to find date and price info
                date_cols = [col for col in df.columns if 'date' in str(col).lower()]
                price_cols = [col for col in df.columns if any(x in str(col).lower() for x in ['close', 'price', 'adj'])]
                
                if date_cols:
                    print(f"\n📅 Date column: {date_cols[0]}")
                    date_col = date_cols[0]
                    print(f"📅 Date range: {df[date_col].min()} to {df[date_col].max()}")
                
                if price_cols:
                    print(f"💰 Price column: {price_cols[0]}")
                    price_col = price_cols[0]
                    print(f"💰 Price range: ${df[price_col].min():.2f} to ${df[price_col].max():.2f}")
                    print(f"💰 Latest price: ${df[price_col].iloc[-1]:.2f}")
                
                return output_file
                
            elif isinstance(acn_data, pd.DataFrame):
                print(f"📋 Already a DataFrame with columns: {list(acn_data.columns)}")
                
                # Export directly
                output_file = Path(__file__).parent / 'ACN_dataframe_export.csv'
                acn_data.to_csv(output_file, index=False)
                
                print(f"✅ Exported to: {output_file}")
                return output_file
            
            else:
                print(f"❌ Unexpected data format: {type(acn_data)}")
                
    except Exception as e:
        print(f"❌ Export error: {e}")
        import traceback
        traceback.print_exc()

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
                    # Filter out comment lines and invalid symbols
                    valid_symbols = symbols_df[
                        ~symbols_df['symbol'].str.startswith('#') & 
                        (symbols_df['symbol'].str.len() <= 5) &  # Most stock symbols are 1-5 chars
                        symbols_df['symbol'].str.isalpha()  # Only letters
                    ]['symbol'].tolist()
                    
                    print(f"Valid symbols: {len(valid_symbols)}")
                    print(f"Sample valid symbols: {valid_symbols[:10]}")
                    
                    # Show invalid entries
                    invalid_symbols = symbols_df[
                        symbols_df['symbol'].str.startswith('#') | 
                        (symbols_df['symbol'].str.len() > 5) |
                        ~symbols_df['symbol'].str.isalpha()
                    ]['symbol'].tolist()
                    
                    if invalid_symbols:
                        print(f"\n⚠️ Invalid symbol entries found: {len(invalid_symbols)}")
                        print(f"Sample invalid: {invalid_symbols[:5]}")
                else:
                    print("⚠️ Symbols table is empty")
                    valid_symbols = []
            except Exception as e:
                print(f"❌ Error getting symbols: {e}")
                valid_symbols = ['ACN', 'AAPL', 'MSFT', 'SPY']  # Fallback
            
            # Check stock_prices table with valid symbols
            try:
                test_symbols = valid_symbols[:5] if valid_symbols else ['ACN', 'AAPL', 'MSFT', 'SPY']
                
                for symbol in test_symbols:
                    try:
                        prices = db.get_stock_prices(symbol)
                        
                        # Handle DataFrame return properly
                        if prices is not None:
                            if isinstance(prices, pd.DataFrame):
                                if not prices.empty:
                                    print(f"📈 {symbol}: {len(prices)} price records (DataFrame)")
                                    
                                    # Show date range
                                    date_cols = [col for col in prices.columns if 'date' in str(col).lower()]
                                    if date_cols:
                                        date_col = date_cols[0]
                                        prices[date_col] = pd.to_datetime(prices[date_col], errors='coerce')
                                        min_date = prices[date_col].min()
                                        max_date = prices[date_col].max()
                                        print(f"  Date range: {min_date} to {max_date}")
                                        
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
            
            # Direct SQL query for stock_prices table
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
                print(f"📊 stock_prices table: {count} total records")
                
                if count > 0:
                    # Get sample data
                    cursor.execute("SELECT * FROM stock_prices LIMIT 5;")
                    sample = cursor.fetchall()
                    
                    # Get column names
                    cursor.execute("PRAGMA table_info(stock_prices);")
                    columns = cursor.fetchall()
                    column_names = [col[1] for col in columns]
                    
                    print(f"📋 Column names: {column_names}")
                    print(f"📋 Sample records:")
                    for row in sample:
                        print(f"  {row}")
                    
                    # Check unique symbols in stock_prices
                    cursor.execute("SELECT DISTINCT Symbol FROM stock_prices;")
                    symbols_in_prices = [row[0] for row in cursor.fetchall()]
                    print(f"\n📊 Unique symbols in stock_prices: {len(symbols_in_prices)}")
                    print(f"Sample: {symbols_in_prices[:10]}")
                    
                    # Check date ranges
                    date_col = None
                    for col in column_names:
                        if 'date' in col.lower():
                            date_col = col
                            break
                    
                    if date_col:
                        cursor.execute(f"SELECT MIN({date_col}), MAX({date_col}) FROM stock_prices;")
                        date_range = cursor.fetchone()
                        print(f"📅 Date range in stock_prices: {date_range[0]} to {date_range[1]}")
                        
                        # Check recent data (last 30 days)
                        cursor.execute(f"""
                            SELECT Symbol, COUNT(*) 
                            FROM stock_prices 
                            WHERE {date_col} >= date('now', '-30 days') 
                            GROUP BY Symbol 
                            ORDER BY COUNT(*) DESC 
                            LIMIT 10;
                        """)
                        recent_data = cursor.fetchall()
                        print(f"\n📈 Symbols with recent data (last 30 days):")
                        for symbol, count in recent_data:
                            print(f"  {symbol}: {count} records")
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
    # First, inspect what methods are available
    inspect_database_manager()
    
    # Then export using the working method
    export_acn_using_get_stock_prices()

    # Check the database content
    check_database_content()