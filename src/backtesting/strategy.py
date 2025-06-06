"""
Base trading strategy implementation and signal generation.
"""
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

@dataclass
class Position:
    symbol: str
    entry_price: float
    entry_date: pd.Timestamp
    size: int
    side: str  # 'long' or 'short'

class TradingStrategy:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: List[Position] = []
        self.trades_history: List[Dict] = []
    
    def generate_signals(self, data: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
        """Generate trading signals based on model predictions."""
        raise NotImplementedError("Subclasses must implement generate_signals()")
    
