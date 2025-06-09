#!/usr/bin/env python3
"""
Linear regression model for stock prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def prepare_target_variable(features_df: pd.DataFrame, target_days: int = 1) -> pd.DataFrame:
    """Prepare target variable (future returns)"""
    
    # Assuming we have returns_1d in the features
    if 'returns_1d' not in features_df.columns:
        logger.error("returns_1d column not found in features")
        return pd.DataFrame()
    
    # Create target: future return
    target_col = f'target_return_{target_days}d'
    features_df[target_col] = features_df['returns_1d'].shift(-target_days)
    
    # Remove rows without target (last few rows)
    features_df = features_df.dropna(subset=[target_col])
    
    return features_df

def train_linear_regression_model(features_df: pd.DataFrame, config: dict):
    """Train linear regression model"""
    
    logger.info("Training linear regression model")
    
    if features_df.empty:
        logger.error("Empty features DataFrame provided")
        return None
    
    # Prepare target variable
    target_days = config.get('prediction_horizon_days', 1)
    df_with_target = prepare_target_variable(features_df, target_days)
    
    if df_with_target.empty:
        logger.error("Could not prepare target variable")
        return None
    
    target_col = f'target_return_{target_days}d'
    
    # Separate features and target
    feature_columns = [col for col in df_with_target.columns 
                      if col not in [target_col, 'returns_1d'] and not col.startswith('target_')]
    
    X = df_with_target[feature_columns]
    y = df_with_target[target_col]
    
    logger.info(f"Training with {X.shape[0]} samples and {X.shape[1]} features")
    logger.info(f"Feature columns: {feature_columns[:10]}...")  # Show first 10
    
    # Split data
    test_size = config.get('test_size', 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # Don't shuffle for time series
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_features': X_train.shape[1]
    }
    
    logger.info(f"Model trained successfully")
    logger.info(f"Train R²: {metrics['train_r2']:.4f}, Test R²: {metrics['test_r2']:.4f}")
    logger.info(f"Test MSE: {metrics['test_mse']:.6f}")
    
    return model, scaler, feature_columns, metrics

if __name__ == "__main__":
    # Test the model
    import sys
    from pathlib import Path
    import yaml
    
    # Add paths
    sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
    from features.technical_indicators import calculate_technical_indicators
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbol = config['target_symbol']
    
    # Generate features
    features = calculate_technical_indicators(symbol, config)
    
    if not features.empty:
        # Train model
        result = train_linear_regression_model(features, config)
        
        if result:
            model, scaler, feature_names, metrics = result
            print(f"✅ Model trained successfully!")
            print(f"Metrics: {metrics}")
        else:
            print("❌ Model training failed")
    else:
        print("❌ No features available")