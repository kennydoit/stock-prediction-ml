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
    
    def add_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optional method to add sentiment features."""
        if self.sentiment is None:
            try:
                from .sentiment.analyzer import SentimentAnalyzer
                self.sentiment = SentimentAnalyzer(self.config)
            except ImportError:
                return data
            
        return self.sentiment.get_sentiment_features(data)