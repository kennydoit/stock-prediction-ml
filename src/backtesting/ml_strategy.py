
"""
Machine learning based trading strategy implementation.
"""
from .strategy import TradingStrategy
import pandas as pd

class MLStrategy(TradingStrategy):
    def __init__(self, 
                 initial_capital: float = 100000,
                 entry_threshold: float = 0.01,
                 exit_threshold: float = -0.01,
                 max_trades: int = 50):  # Add this parameter
        super().__init__(initial_capital)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_trades = max_trades
        self.trade_count = 0
    
    def generate_signals(self, data: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
        """Generate trading signals based on ML model predictions."""
        signals = pd.DataFrame(index=data.index)
        signals['predicted_return'] = predictions.pct_change()
        signals['position'] = 0
        
        # Only generate new signals if under max_trades
        if self.trade_count < self.max_trades:
            # Entry signals
            signals.loc[signals['predicted_return'] > self.entry_threshold, 'position'] = 1
            signals.loc[signals['predicted_return'] < self.exit_threshold, 'position'] = -1
            
            # Count new trades (position changes)
            position_changes = signals['position'] != signals['position'].shift(1)
            new_trades = position_changes.sum() // 2  # Divide by 2 as each trade has entry + exit
            self.trade_count += new_trades
        
        return signals