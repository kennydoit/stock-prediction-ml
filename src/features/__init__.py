"""Feature engineering module."""
import pandas as pd
from typing import Dict, Optional
from .technical.indicators import TechnicalIndicators

class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        self.technical = TechnicalIndicators(config)
        # Initialize sentiment as None by default
        self.sentiment = None
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer technical features for the dataset."""
        df = data.copy()
        
        # Add technical indicators only
        df = self.technical.add_indicators(df)
        
        return df

from .features import FeatureEngineer

__all__ = ['FeatureEngineer']

