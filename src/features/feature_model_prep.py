import logging
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

def filter_technical_features(features_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter technical indicators based on configuration settings.
    """
    if features_df.empty:
        logger.warning("Empty features DataFrame provided")
        return pd.DataFrame()
        
    enabled_features = []
    tech_config = config.get('features', {}).get('technical', {})
    
    if not tech_config:
        logger.error("No technical features configuration found")
        return features_df  # Return all features if no config
    
    # Map indicator names to their column prefixes
    indicator_prefixes = {
        'rsi': 'RSI',
        'macd': 'MACD',
        'bollinger': 'BB',
        'cci': 'CCI',
        'stochastic': 'STOCH',
        'atr': 'ATR',
        'obv': 'OBV',
        'ichimoku': 'ICHIMOKU',
        'sma': 'sma'  # Add SMA mapping
    }
    
    logger.info("Available columns: %s", features_df.columns.tolist())
    
    # Build list of enabled features
    for indicator, settings in tech_config.items():
        if settings.get('enabled', False):
            prefix = indicator_prefixes.get(indicator, indicator)
            # Get all columns that contain the indicator prefix
            indicator_cols = [col for col in features_df.columns 
                            if prefix in col]  # Remove .upper() for case-sensitive matching
            if indicator_cols:
                enabled_features.extend(indicator_cols)
                logger.info(f"Found {len(indicator_cols)} columns for {indicator}")
            else:
                logger.warning(f"No columns found for enabled indicator {indicator}")
    
    # Also include lag features if they're enabled in config
    if config.get('features', {}).get('lags', {}).get('enabled', False):
        lag_cols = [col for col in features_df.columns if 'lag' in col]
        if lag_cols:
            enabled_features.extend(lag_cols)
            logger.info(f"Found {len(lag_cols)} lag columns")
        else:
            logger.warning("Lag features enabled but no lag columns found")
    
    if not enabled_features:
        logger.warning("No enabled features found, returning all features")
        return features_df
        
    logger.info(f"Total enabled features: {len(enabled_features)}")
    return features_df[enabled_features]

# Create single consistent feature preparation section
def prepare_features(all_features: pd.DataFrame, 
                    target_symbol: str, 
                    peer_symbols: list,
                    config: dict) -> tuple:
    """
    Prepare features for model training with configuration-based filtering.
    """
    if all_features.empty:
        raise ValueError("Empty features DataFrame provided")
        
    prediction_horizon = config.get('prediction_horizon', 5)  # Default to 5 days if not specified
    # Remove leakage features
    base_leakage_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close']
    leakage_features = [f'{symbol}_{feature}' 
                       for symbol in [target_symbol] + peer_symbols
                       for feature in base_leakage_features]
    
    # Get all non-leakage features
    safe_features = all_features[[col for col in all_features.columns 
                                if col not in leakage_features]]
    
    # Filter technical features
    X = filter_technical_features(safe_features, config)
    
    if X.empty:
        raise ValueError("No features remained after filtering")
        
    y = all_features[f'{target_symbol}_Close'].shift(-prediction_horizon)  # Close price 5 days ahead
    
    logger.info(f"Final feature set shape: {X.shape}")
    return X, y, list(X.columns)

def create_period_masks(data: pd.DataFrame, periods: dict) -> dict:
    """Create masks for training, test, and strategy periods"""
    return {
        period: (
            (data.index >= pd.to_datetime(config['start']).tz_localize('America/New_York')) & 
            (data.index <= pd.to_datetime(config['end']).tz_localize('America/New_York'))
        )
        for period, config in periods.items()
    }

def validate_training_period(start_date: str, end_date: str, data: pd.DataFrame) -> tuple:
    """
    Validates that requested training period exists within available data.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        data (pd.DataFrame): DataFrame with datetime index
        
    Returns:
        tuple: Validated (start_date, end_date) as timezone-aware timestamps
    """
    # Convert string dates to timezone-aware timestamps
    start = pd.to_datetime(start_date).tz_localize('America/New_York')
    end = pd.to_datetime(end_date).tz_localize('America/New_York')
    
    # Ensure data index is timezone-aware
    if data.index.tz is None:
        data.index = data.index.tz_localize('America/New_York')
    
    # Get data boundaries
    data_start = data.index.min()
    data_end = data.index.max()
    
    # Validate start date
    if start < data_start:
        logger.warning(f"Requested start date {start} before available data. Using {data_start}")
        start = data_start
    
    # Validate end date
    if end > data_end:
        logger.warning(f"Requested end date {end} after available data. Using {data_end}")
        end = data_end
    
    # Ensure start is before end
    if start >= end:
        raise ValueError(f"Invalid training period: {start} to {end}")
    
    # Check minimum data requirement
    period_length = len(data[(data.index >= start) & (data.index <= end)])
    if period_length < 252:  # One trading year
        raise ValueError(f"Insufficient training data: {period_length} days available, minimum 252 required")
    
    logger.info(f"Validated training period: {start} to {end} ({period_length} trading days)")
    return start, end