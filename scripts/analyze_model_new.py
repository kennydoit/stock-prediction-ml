import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'database'))

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

    # Create masks
    train_mask = (lambda df: (df.index >= train_start) & (df.index <= train_end))
    test_mask = (lambda df: (df.index >= test_start) & (df.index <= test_end))
    strategy_mask = (lambda df: (df.index >= strategy_start) & (df.index <= strategy_end))

    return train_mask, test_mask, strategy_mask

def main():
    print("Starting model analysis...")
    train_mask, test_mask, strategy_mask = create_period_masks()
    latest_model_path = Path('C:/Users/Kenrm/repositories/stock-prediction-ml/data/model_outputs/enhanced_linear_regression_MSFT_with_peers_20250619_125726.pkl')

    loaded_model = pd.read_pickle(latest_model_path)
    print("Model loaded successfully:", loaded_model)

    model_coefficients_path = loaded_model.get('model_coefficients', None)

    if model_coefficients_path:
        model_coefficients = pd.read_csv(model_coefficients_path)
        print("Model coefficients loaded successfully:", model_coefficients.head())

# run analysis on the loaded model
if __name__ == "__main__":
    main()
        
