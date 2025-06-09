#!/usr/bin/env python3
"""
Technical indicators calculation module
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add database path
sys.path.append(str(Path(__file__).parent.parent.parent / 'database'))
from database_manager import DatabaseManager

logger = logging.getLogger(__name__)

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window).mean()

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    macd_signal = calculate_ema(macd, signal)
    return macd, macd_signal

def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower, sma

def calculate_technical_indicators(symbol: str, config: dict) -> pd.DataFrame:
    """Calculate all technical indicators for a symbol"""
    
    logger.info(f"Calculating technical indicators for {symbol}")
    
    # Get price data from database
    db_manager = DatabaseManager()
    
    with db_manager:
        price_data = db_manager.get_stock_prices(symbol)
        
        if price_data.empty:
            logger.warning(f"No price data found for {symbol}")
            return pd.DataFrame()
        
        # Ensure we have the required columns
        required_cols = ['close', 'high', 'low', 'volume']
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        
        if missing_cols:
            logger.error(f"Missing columns for {symbol}: {missing_cols}")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(price_data)} price records for {symbol}")
        
        # Create features DataFrame
        features = pd.DataFrame(index=price_data.index)
        
        # Price-based features
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
        
        # Simple Moving Averages
        features['sma_5'] = calculate_sma(close, 5)
        features['sma_10'] = calculate_sma(close, 10)
        features['sma_20'] = calculate_sma(close, 20)
        features['sma_50'] = calculate_sma(close, 50)
        
        # Exponential Moving Averages
        features['ema_12'] = calculate_ema(close, 12)
        features['ema_26'] = calculate_ema(close, 26)
        
        # RSI
        features['rsi_14'] = calculate_rsi(close, 14)
        
        # MACD
        macd, macd_signal = calculate_macd(close)
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd - macd_signal
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close)
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_middle'] = bb_middle
        features['bb_width'] = bb_upper - bb_lower
        features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Volume features
        features['volume_sma'] = calculate_sma(volume, 20)
        features['volume_ratio'] = volume / features['volume_sma']
        
        # Price returns
        features['returns_1d'] = close.pct_change(1)
        features['returns_5d'] = close.pct_change(5)
        features['returns_20d'] = close.pct_change(20)
        
        # Volatility
        features['volatility_20d'] = features['returns_1d'].rolling(window=20).std()
        
        # Price position features
        features['high_low_ratio'] = high / low
        features['close_to_high'] = close / high
        features['close_to_low'] = close / low
        
        # Trend features
        features['price_above_sma20'] = (close > features['sma_20']).astype(int)
        features['price_above_sma50'] = (close > features['sma_50']).astype(int)
        features['sma_20_above_50'] = (features['sma_20'] > features['sma_50']).astype(int)
        
        # Remove rows with NaN values (from rolling calculations)
        features = features.dropna()
        
        logger.info(f"Generated {len(features)} feature records with {len(features.columns)} indicators")
        
        return features

if __name__ == "__main__":
    # Test the module
    import yaml
    
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbol = config['target_symbol']
    features = calculate_technical_indicators(symbol, config)
    
    if not features.empty:
        print(f"✅ Generated {features.shape[0]} features for {symbol}")
        print(f"Features: {list(features.columns)}")
        print(features.head())
    else:
        print(f"❌ No features generated for {symbol}")