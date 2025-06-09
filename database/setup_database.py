"""
Database setup script for stock prediction ML project
"""
import logging
from pathlib import Path
import yaml
from database_manager import DatabaseManager
from symbol_universe import SymbolUniverse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_database():
    """Initialize database and populate with universe symbols"""
    
    print("Setting up stock prediction database...")
    
    # Initialize database
    db_manager = DatabaseManager()
    
    with db_manager:
        # Create schema
        print("Creating database schema...")
        db_manager.setup_database()
        
        # Load universe configuration
        universe_config_path = Path(__file__).parent.parent / 'universe_config.yaml'
        if universe_config_path.exists():
            print("Loading universe configuration...")
            with open(universe_config_path, 'r') as f:
                universe_config = yaml.safe_load(f)
            
            # Insert universe symbols
            print("Inserting universe symbols...")
            
            # Insert stocks with sector information
            for sector, symbols in universe_config.get('sector_breakdown', {}).items():
                for symbol in symbols:
                    db_manager.insert_symbol(
                        symbol=symbol,
                        sector=sector,
                        market_cap='Unknown'  # Will be updated when we fetch data
                    )
            
            # Insert ETFs
            for etf in universe_config.get('benchmark_etfs', []):
                db_manager.insert_symbol(
                    symbol=etf,
                    sector='ETF',
                    market_cap='ETF'
                )
            
            # Insert macro assets
            for macro in universe_config.get('macro_assets', []):
                db_manager.insert_symbol(
                    symbol=macro,
                    sector='Macro',
                    market_cap='Macro'
                )
            
            print(f"Inserted {universe_config['universe_info']['total_symbols']} symbols into database")
        
        else:
            print("Universe config not found. Creating sample symbols...")
            # Create some sample symbols
            sample_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
            for symbol in sample_symbols:
                db_manager.insert_symbol(symbol=symbol, sector='Technology')
    
    print("Database setup complete!")
    print(f"Database location: {db_manager.db_path}")
    
    # Display summary
    with db_manager:
        symbols_df = db_manager.get_symbols()
        print(f"\nDatabase contains {len(symbols_df)} symbols:")
        
        # Group by sector
        sector_counts = symbols_df.groupby('sector').size().sort_values(ascending=False)
        for sector, count in sector_counts.items():
            print(f"  {sector}: {count} symbols")

if __name__ == "__main__":
    setup_database()