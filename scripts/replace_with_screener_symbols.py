#!/usr/bin/env python3
"""
Replace database symbols with realistic_value_screener symbols
"""
import sys
from pathlib import Path
import pandas as pd
import yaml
import logging
from datetime import datetime

# Add database path
sys.path.append(str(Path(__file__).parent.parent / 'database'))
from database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_screener_symbols():
    """Load symbols from the actual screener file"""
    
    # Direct path to your actual screener symbols
    screener_file = Path(r"C:\Users\Kenrm\repositories\stock-symbol-analyzer\data\realistic_value_screen_symbols.txt")
    
    if screener_file.exists():
        with open(screener_file, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        print(f"✅ Loaded {len(symbols)} symbols from realistic_value_screen_symbols.txt")
        print(f"File location: {screener_file}")
        return symbols, screener_file  # Return both symbols and file path
    else:
        raise FileNotFoundError(f"Screener file not found at: {screener_file}")

def replace_database_symbols():
    """Replace all database symbols with screener symbols"""
    
    print("🔄 Replacing Database Symbols with Realistic Value Screener Symbols")
    print("="*60)
    
    # Load screener symbols
    try:
        screener_symbols, screener_file = load_screener_symbols()  # Get both values
        print(f"Loaded {len(screener_symbols)} symbols from screener")
        print(f"Sample symbols: {screener_symbols[:10]}")
    except Exception as e:
        print(f"❌ Error loading screener symbols: {e}")
        return
    
    # Connect to database
    db_manager = DatabaseManager()
    
    with db_manager:
        # Clear existing data
        print("\n🗑️  Clearing existing database...")
        cursor = db_manager.connection.cursor()
        cursor.execute("DELETE FROM news_symbols")
        cursor.execute("DELETE FROM news_articles") 
        cursor.execute("DELETE FROM stock_prices")
        cursor.execute("DELETE FROM symbols")
        db_manager.connection.commit()
        print("✅ Cleared existing data")
        
        # Insert screener symbols
        print(f"\n📥 Inserting {len(screener_symbols)} screener symbols...")
        
        success_count = 0
        
        for i, symbol in enumerate(screener_symbols):
            try:
                # Determine basic sector (you can refine this later)
                if symbol in ['SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'AGG', 'TLT', 'GLD', 'SLV']:
                    sector = 'ETF'
                else:
                    sector = 'Equity'
                
                db_manager.insert_symbol(
                    symbol=symbol.upper().strip(),
                    sector=sector,
                    market_cap='Unknown'
                )
                success_count += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  Inserted {i + 1}/{len(screener_symbols)} symbols...")
                
            except Exception as e:
                print(f"  ❌ Error inserting {symbol}: {e}")
        
        print(f"\n✅ Successfully inserted {success_count}/{len(screener_symbols)} symbols")
        
        # Update universe config
        universe_config = {
            'universe_info': {
                'name': 'Realistic Value Screener Universe',
                'description': 'Validated symbols from realistic_value_screener analysis',
                'total_symbols': success_count,
                'last_updated': str(datetime.now().date()),
                'source': str(screener_file)
            },
            'symbols': screener_symbols[:success_count]
        }
        
        config_path = Path(__file__).parent.parent / 'universe_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(universe_config, f, default_flow_style=False)
        
        print(f"✅ Updated universe_config.yaml")
        
        # Show final state
        final_symbols = db_manager.get_symbols()
        print(f"\n📊 Final Database State:")
        print(f"  Total symbols: {len(final_symbols)}")
        print(f"  Ready for price data collection!")
        
        return success_count

if __name__ == "__main__":
    replace_database_symbols()