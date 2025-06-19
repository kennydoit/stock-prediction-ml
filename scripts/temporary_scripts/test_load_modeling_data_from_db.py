import sys
from pathlib import Path
import yaml
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from train_enhanced_model import load_modeling_data_from_db

# Load config
yaml_path = Path(__file__).parent.parent.parent / 'config.yaml'
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

# --- Parse YAML for flags ---
periods = config['periods']
target_symbol = config['target_symbol']
peer_symbols = config.get('peer_symbols', [])

# Find which target variable is set to true
raw_targets = config['target_variable']
target_variable = None
for entry in raw_targets:
    for k, v in entry.items():
        if v:
            target_variable = k
            break
    if target_variable:
        break
if not target_variable:
    raise ValueError('No target variable set to true in config.yaml')

# Parse feature flags (expecting mapping structure)
features = config['features']
use_technical = features.get('technical', {}).get('enabled', False)
use_signals = features.get('signals', {}).get('enabled', False)
use_fred = features.get('fred', {}).get('enabled', False)

feature_flags = features.get('technical', {}).get('features', {}) if use_technical else {}
signal_flags = features.get('signals', {}).get('features', {}) if use_signals else {}
fred_flags = features.get('fred', {}).get('features', {}) if use_fred else {}

# Call loader
df = load_modeling_data_from_db(
    config,
    target_symbol,
    peer_symbols,
    periods,
    use_technical,
    use_signals,
    use_fred,
    feature_flags,
    fred_flags,
    signal_flags,
    target_variable
)

# Output to CSV
out_path = Path(__file__).parent / 'test_modeling_data_output.csv'
df.to_csv(out_path, index=False)
print(f"✅ Data written to {out_path}")
