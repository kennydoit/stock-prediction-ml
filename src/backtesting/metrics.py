"""Performance metrics calculation for trading strategies."""
import pandas as pd
import numpy as np
from typing import Dict

def calculate_performance_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """Calculate trading strategy performance metrics."""
    metrics = {}
    
    # Return metrics
    metrics['total_return'] = results['cumulative_returns'].iloc[-1] - 1
    metrics['annualized_return'] = ((1 + metrics['total_return']) ** 
                                  (252/len(results)) - 1)
    metrics['sharpe_ratio'] = (np.sqrt(252) * results['strategy_returns'].mean() / 
                             results['strategy_returns'].std())
    
    # Risk metrics
    metrics['max_drawdown'] = results['drawdown'].min()
    metrics['volatility'] = results['strategy_returns'].std() * np.sqrt(252)
    
    # Trading metrics
    metrics['win_rate'] = (len(results[results['strategy_returns'] > 0]) / 
                          len(results[results['strategy_returns'] != 0]))
    
    return metrics