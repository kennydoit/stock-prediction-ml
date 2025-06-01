"""Backtesting engine for evaluating trading strategies."""
from typing import Dict
import pandas as pd
import numpy as np
from .strategy import TradingStrategy

class Backtest:
    def __init__(self, strategy: TradingStrategy, data: pd.DataFrame):
        self.strategy = strategy
        self.data = data
        self.results = pd.DataFrame()
    
    def run(self, predictions: pd.Series) -> pd.DataFrame:
        """Run backtest using strategy signals."""
        signals = self.strategy.generate_signals(self.data, predictions)
        return self._calculate_returns(signals)
    
    def _calculate_returns(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy returns and metrics."""
        # Find the Close price column for target symbol
        close_col = [col for col in self.data.columns if col.endswith('_Close')][0]
        
        results = pd.DataFrame(index=self.data.index)
        results['price'] = self.data[close_col]
        results['position'] = signals['position']
        results['returns'] = self.data[close_col].pct_change()
        results['strategy_returns'] = results['position'].shift(1) * results['returns']
        results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
        results['drawdown'] = (results['cumulative_returns'] / 
                             results['cumulative_returns'].cummax() - 1)
        return results