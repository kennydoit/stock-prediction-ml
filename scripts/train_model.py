import sys
from pathlib import Path
import yaml
import pandas as pd
import joblib
from datetime import datetime
import logging
import argparse
import statsmodels.api as sm

sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'database'))
# Read in config.yaml and create masks for training, test, and strategy periods
# Add paths

def create_period_masks():
    """
    Reads in config.yaml and creates masks for training, test, and strategy periods.
    """
    import yaml
    import pandas as pd
    from datetime import datetime


    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Convert date strings to datetime objects
    train_start = config['periods']['training']['start']
    train_end = config['periods']['training']['end']
    test_start = config['periods']['test']['start']
    test_end = config['periods']['test']['end']
    strategy_start = config['periods']['strategy']['start']
    strategy_end = config['periods']['strategy']['end']

    return train_start, train_end, test_start, test_end, strategy_start, strategy_end

def load_data_from_stock_db(from_table, columns, symbols, start_date, end_date):
    import sqlite3
    import pandas as pd
    import yaml

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    project_root = str(Path(__file__).parent.parent.parent.resolve())
    print(f"🌍 Project root: {project_root}")
    def expand_path(path):
        return path.replace("$project_root", project_root)

    # --- DB paths ---
    stock_db_path = expand_path(config['stock_db_path'])

    print(f"📊 Stock DB path: {stock_db_path}")

    conn = sqlite3.connect(stock_db_path)

    # --- Connect to stock DB ---
    stock_conn = sqlite3.connect(stock_db_path)
    # --- Get symbol_id for target and peers ---
    symbol_query = f"SELECT symbol, symbol_id FROM symbols WHERE symbol IN ({', '.join(['?' for _ in symbols])})"
    symbol_df = pd.read_sql(symbol_query, stock_conn, params=symbols)

    print(f"Symbols to load: {symbols}")

    # Load data for all symbols
    stock_data_df = pd.DataFrame()

    # Assume symbol_df['symbol_id'] is a list/Series of IDs
    symbol_ids = symbol_df['symbol_id'].tolist()
    symbol_id_list = ', '.join([f"'{sid}'" for sid in symbol_ids])

    query = f"""
    SELECT date, symbol_id, {', '.join(columns)}
    FROM {from_table}
    WHERE symbol_id IN ({symbol_id_list}) AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """
    
    print(f"Executing query: {query}")

    stock_data_df = pd.read_sql(query, stock_conn)

    # Merge symbol from symbol_df onto stock_data_df
    stock_data_df = stock_data_df.merge(symbol_df[['symbol_id', 'symbol']], on='symbol_id', how='left')
    return stock_data_df

def load_data_from_fred_db(columns, start_date, end_date):
    import sqlite3
    import pandas as pd
    import yaml

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    project_root = str(Path(__file__).parent.parent.parent.resolve())
    print(f"🌍 Project root: {project_root}")
    def expand_path(path):
        return path.replace("$project_root", project_root)

    # --- DB paths ---
    fred_db_path = expand_path(config['fred_db_path'])

    print(f"📊 FRED DB path: {fred_db_path}")

    # --- DB paths ---
    conn = sqlite3.connect(fred_db_path)

    # Load data from FRED database
    query = f"""
    SELECT date, {', '.join(columns)}
    FROM fred_data_wide
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """
    
    print(f"Executing query: {query}")

    fred_data_df = pd.read_sql(query, conn)
    return fred_data_df     

def wide_table_format(df, date_col='date', symbol_col='symbol'):
    """
    Converts a DataFrame from long to wide format based on date and symbol columns.
    """
    if date_col not in df.columns or symbol_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{date_col}' and '{symbol_col}' columns")

    # Set the index to date and symbol, then unstack
    df.set_index([date_col, symbol_col], inplace=True)
    wide_df = df.unstack(level=symbol_col)

    # Flatten the MultiIndex columns
    wide_df.columns = ['__'.join(col).strip() for col in wide_df.columns.values]
    
    return wide_df.reset_index()

def main():
    # Read in config.yaml and determine which feature groups to use
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_start, train_end, test_start, test_end, strategy_start, strategy_end = create_period_masks()

    use_technical_features = config['features']['technical']['enabled']
    use_signal_features = config['features']['signals']['enabled']
    use_fred_features = config['features']['fred']['enabled']
    target_variable = config['target_variable']
    target_symbol = config['target_symbol']
    use_peers = config['peers']['use_peers']['enabled']

    print(f"Using technical features: {use_technical_features}")
    print(f"Using signal features: {use_signal_features}")
    print(f"Using FRED features: {use_fred_features}\n")
    print(f"Target variable: {target_variable}")
    print(f"Target symbol: {target_symbol}\n")

    print(f"Using peers: {use_peers}\n")

    # if using peers, get peer symbols
    if use_peers:   
        peers = config['peers']['peer_symbols']
        print(f"Peer symbols: {peers}") 

    # if using technical features, load the technical features
    if use_technical_features:
        # Get technical feature list from config
        feature_list = []
        for category, cat_data in config.get("features", {}).items():
            if cat_data.get("enabled"):
                for feat, enabled in cat_data.get("features", {}).items():
                    if enabled:
                        feature_list.append(feat)
        print(f"Using technical features: {feature_list}\n")
        
        # Extract only technical features set to true
        technical_features = []
        technical_config = config.get('features', {}).get('technical', {})
        if technical_config.get('enabled'):
            for feat, enabled in technical_config.get('features', {}).items():
                if enabled:
                    technical_features.append(feat)
        print(f"Enabled technical features: {technical_features}\n")

        # Extract only signal features set to true
        signal_features = []
        signal_config = config.get('features', {}).get('signals', {})
        if signal_config.get('enabled'):
            for feat, enabled in signal_config.get('features', {}).items():
                if enabled:
                    signal_features.append(feat)        
        print(f"Enabled signal features: {signal_features}\n")

        # Extract only FRED features set to true
        fred_features = []  
        fred_config = config.get('features', {}).get('fred', {})
        if fred_config.get('enabled'):
            for feat, enabled in fred_config.get('features', {}).items():
                if enabled:
                    fred_features.append(feat)      
        print(f"Enabled FRED features: {fred_features}\n")

        technical_data, signal_data, fred_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # Extract technical features from the database
        if use_technical_features:
            from_table = "technical_indicators"
            columns = technical_features
            technical_data = load_data_from_stock_db(from_table, columns, [target_symbol] + peers, train_start, strategy_end)
            print(f"Technical data loaded with shape: {technical_data.shape}\n")
            # if peers are used, convert technical_data to wide format 
            if use_peers:
                technical_data = wide_table_format(technical_data, date_col='date', symbol_col='symbol')
                print(f"Signal data converted to wide format with shape: {technical_data.shape}\n")

        # Extract signal features from the database
        if use_signal_features:
            from_table = "technical_trade_signals"
            columns = signal_features
            signal_data = load_data_from_stock_db(from_table, columns, [target_symbol] + peers, train_start, strategy_end)
            print(f"Signal data loaded with shape: {signal_data.shape}\n")
            # if peers are used, convert signal_data to wide format 
            if use_peers:
                signal_data = wide_table_format(signal_data, date_col='date', symbol_col='symbol')
                print(f"Signal data converted to wide format with shape: {signal_data.shape}\n")

        # Extract FRED features from the database
        if use_fred_features:
            columns = fred_features
            fred_data = load_data_from_fred_db(columns, train_start, strategy_end)
            print(f"FRED data loaded with shape: {fred_data.shape}\n")

        # Extract outcomes data from the database
        from_table = "outcomes"
        columns = [target_variable]
        outcomes_data = load_data_from_stock_db(from_table, columns, [target_symbol], train_start, strategy_end)
        print(f"Outcomes data loaded with shape: {outcomes_data.shape}\n")

  
if __name__ == "__main__":
    print("Training Model Pipeline in progress...\n")
    main()