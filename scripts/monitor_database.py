#!/usr/bin/env python3
"""
Database monitoring and status script
"""
import sys
from pathlib import Path
import pandas as pd

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'database'))

from database_manager import DatabaseManager

def monitor_database():
    """Monitor database status and contents"""
    
    print("🗄️ Database Status Monitor")
    print("="*40)
    
    try:
        db_manager = DatabaseManager()
        
        with db_manager:
            # Check symbols
            symbols_df = db_manager.get_symbols()
            print(f"📊 Symbols: {len(symbols_df)} total")
            
            if not symbols_df.empty:
                sector_counts = symbols_df.groupby('sector').size()
                for sector, count in sector_counts.items():
                    print(f"  {sector}: {count}")
            
            # Check price data
            print(f"\n📈 Price Data:")
            for symbol in symbols_df['symbol'].head(5):
                prices = db_manager.get_stock_prices(symbol)
                if not prices.empty:
                    print(f"  {symbol}: {len(prices)} records ({prices.index[0]} to {prices.index[-1]})")
                else:
                    print(f"  {symbol}: No data")
        
        # Check models directory
        models_dir = Path(__file__).parent.parent / 'models'
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pkl'))
            print(f"\n🤖 Models: {len(model_files)} saved")
            
            for model_file in model_files[-3:]:  # Show last 3
                print(f"  {model_file.name}")
        else:
            print(f"\n🤖 Models: No models directory")
        
        print(f"\n✅ Database monitoring complete")
        
    except Exception as e:
        print(f"❌ Database monitoring failed: {e}")

if __name__ == "__main__":
    monitor_database()