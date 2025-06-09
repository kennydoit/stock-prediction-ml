#!/usr/bin/env python3
"""
Sync symbol universe from submodule
"""
import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime  # Add this import

# Add paths for submodule integration
submodule_path = Path(__file__).parent.parent / 'stock-prediction-ml'
if submodule_path.exists():
    sys.path.append(str(submodule_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_universe():
    """Sync universe config from submodule validated symbols"""
    
    # Import validated symbols from submodule
    try:
        # Adjust import based on your submodule structure
        from data.output import validated_symbols
        symbols_list = validated_symbols.get_symbol_list()
        
        # Update universe_config.yaml
        universe_config = {
            'universe_info': {
                'total_symbols': len(symbols_list),
                'source': 'validated_submodule',
                'last_updated': str(datetime.now().date())  # Now this will work
            },
            'stocks': symbols_list,
            # Add other categories as needed
        }
        
        config_path = Path(__file__).parent.parent / 'universe_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(universe_config, f, default_flow_style=False)
        
        print(f"✅ Synced {len(symbols_list)} validated symbols")
        
    except ImportError:
        logger.warning("Submodule not available, using existing universe_config.yaml")

if __name__ == "__main__":
    sync_universe()