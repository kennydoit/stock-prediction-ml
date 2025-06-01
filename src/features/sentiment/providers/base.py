from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class SentimentProvider(ABC):
    """Abstract base class for sentiment data providers."""
    
    @abstractmethod
    def get_sentiment(self, symbol: str, date: pd.Timestamp) -> Dict:
        """Get sentiment data for a symbol and date."""
        pass

    @abstractmethod
    def validate_response(self, response: Dict) -> bool:
        """Validate API response."""
        pass