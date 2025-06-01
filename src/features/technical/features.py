"""
Feature engineering module for technical indicators.
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import yaml

class FeatureEngineer:
    def __init__(self, config_path: str = '../config.yaml'):
        """Initialize with parameters from config file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.params = config.get('feature_parameters', {})
            
        # Set default parameters if not in config
        self.rsi_periods = self.params.get('rsi_periods', 14)
        self.macd_fast = self.params.get('macd_fast', 12)
        self.macd_slow = self.params.get('macd_slow', 26)
        self.macd_signal = self.params.get('macd_signal', 9)
        self.bb_window = self.params.get('bb_window', 20)
        self.bb_num_std = self.params.get('bb_num_std', 2)
        self.volatility_window = self.params.get('volatility_window', 20)


    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features for the dataset."""
        df = data.copy()
        
        # Technical indicators with parameters in names
        df[f'RSI_{self.rsi_periods}'] = self.calculate_rsi(df)
        
        macd_data = self.calculate_macd(df)
        macd_data.columns = [
            f'MACD_{self.macd_fast}_{self.macd_slow}',
            f'Signal_{self.macd_signal}',
            f'Histogram_{self.macd_fast}_{self.macd_slow}'
        ]
        df = pd.concat([df, macd_data], axis=1)
        
        # Basic features
        df['Returns'] = df['Close'].pct_change()
        df[f'Volatility_{self.volatility_window}'] = df['Returns'].rolling(
            window=self.volatility_window
        ).std()
        
        return df.dropna()