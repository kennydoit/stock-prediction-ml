#!/usr/bin/env python3
"""
Identify which symbols failed data collection and why
"""
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add database path
sys.path.append(str(Path(__file__).parent.parent / 'database'))
from database_manager import DatabaseManager

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceLoader:
    """Yahoo Finance data loader with detailed error reporting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str):
        """Fetch stock data with detailed error reporting"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Test if symbol info is available
            try:
                info = ticker.info
                company_name = info.get('longName', 'Unknown')
            except Exception as e:
                raise Exception(f"Symbol info unavailable: {e}")
            
            # Fetch historical data
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise Exception("No historical data available")
            
            # Check data quality
            if len(data) < 10:
                raise Exception(f"Insufficient data: only {len(data)} records")
            
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            data['Adj Close'] = data['Close']
            data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            
            return data, company_name
            
        except Exception as e:
            raise Exception(str(e))

def identify_failed_symbols():
    """Identify and categorize failed symbols"""
    
    print("🔍 Identifying Failed Symbols")
    print("="*50)
    
    db_manager = DatabaseManager()
    data_loader = YahooFinanceLoader()
    
    with db_manager:
        # Get symbols without price data (likely the failed ones)
        cursor = db_manager.connection.cursor()
        cursor.execute("""
            SELECT s.symbol, s.sector, s.market_cap
            FROM symbols s
            LEFT JOIN stock_prices sp ON s.id = sp.symbol_id
            WHERE sp.symbol_id IS NULL
            ORDER BY s.sector, s.symbol
        """)
        
        failed_symbols = cursor.fetchall()
        
        if not failed_symbols:
            print("✅ No symbols without price data found!")
            return
        
        print(f"Found {len(failed_symbols)} symbols without price data")
        print("Testing each symbol to determine failure reason...\n")
        
        # Categorize failures
        failure_categories = {
            'delisted': [],
            'invalid_ticker': [],
            'insufficient_data': [],
            'api_error': [],
            'unknown': []
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        for i, (symbol, sector, market_cap) in enumerate(failed_symbols):
            print(f"[{i+1}/{len(failed_symbols)}] Testing {symbol} ({sector})...", end=" ")
            
            try:
                data, company_name = data_loader.fetch_stock_data(
                    symbol,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                print(f"✅ ACTUALLY WORKS! {company_name}")
                print(f"   {len(data)} records available - this symbol should have worked")
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"❌ {e}")
                
                # Categorize the error
                if 'delisted' in error_msg or 'not found' in error_msg:
                    failure_categories['delisted'].append((symbol, sector, str(e)))
                elif 'insufficient data' in error_msg:
                    failure_categories['insufficient_data'].append((symbol, sector, str(e)))
                elif 'symbol info unavailable' in error_msg:
                    failure_categories['invalid_ticker'].append((symbol, sector, str(e)))
                elif 'api' in error_msg or 'request' in error_msg:
                    failure_categories['api_error'].append((symbol, sector, str(e)))
                else:
                    failure_categories['unknown'].append((symbol, sector, str(e)))
        
        # Report results
        print(f"\n{'='*60}")
        print("FAILURE ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        total_failed = len(failed_symbols)
        
        for category, symbols in failure_categories.items():
            if symbols:
                print(f"\n{category.upper().replace('_', ' ')} ({len(symbols)} symbols):")
                
                if category == 'delisted':
                    print("  These symbols are no longer traded and should be removed:")
                elif category == 'invalid_ticker':
                    print("  These symbols have invalid tickers:")
                elif category == 'insufficient_data':
                    print("  These symbols don't have enough historical data:")
                elif category == 'api_error':
                    print("  These symbols had API/network issues (might work later):")
                else:
                    print("  These symbols failed for unknown reasons:")
                
                # Group by sector for better display
                by_sector = {}
                for symbol, sector, error in symbols:
                    if sector not in by_sector:
                        by_sector[sector] = []
                    by_sector[sector].append(symbol)
                
                for sector, sector_symbols in by_sector.items():
                    symbols_str = ', '.join(sector_symbols)
                    print(f"    {sector}: {symbols_str}")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        removable_count = len(failure_categories['delisted']) + len(failure_categories['invalid_ticker'])
        
        if removable_count > 0:
            print(f"1. REMOVE {removable_count} symbols that are clearly problematic")
            print("   These symbols are delisted or have invalid tickers")
        
        retry_count = len(failure_categories['api_error']) + len(failure_categories['unknown'])
        if retry_count > 0:
            print(f"2. RETRY {retry_count} symbols that might work with another attempt")
            print("   These had temporary API issues or unknown errors")
        
        if len(failure_categories['insufficient_data']) > 0:
            print(f"3. CONSIDER removing {len(failure_categories['insufficient_data'])} symbols with insufficient data")
            print("   These might be very new listings or have data quality issues")
        
        # Calculate success rate after cleanup
        potential_removals = removable_count
        remaining_symbols = len(failed_symbols) - potential_removals
        
        print(f"\nCURRENT STATUS:")
        print(f"  Total symbols: {len(failed_symbols) + 156}")  # 156 successful from your output
        print(f"  Successful: 156")
        print(f"  Failed: {len(failed_symbols)}")
        print(f"  Success rate: {156/(len(failed_symbols) + 156)*100:.1f}%")
        
        print(f"\nAFTER CLEANUP:")
        print(f"  Symbols to remove: {potential_removals}")
        print(f"  Remaining symbols: {156 + len(failed_symbols) - potential_removals}")
        print(f"  Projected success rate: {156/(156 + remaining_symbols)*100:.1f}%")

if __name__ == "__main__":
    identify_failed_symbols()