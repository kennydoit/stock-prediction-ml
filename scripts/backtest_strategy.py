#!/usr/bin/env python3
"""
Trading strategy backtester using ML predictions
"""
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'database'))

from database_manager import DatabaseManager
from features.technical_indicators import calculate_technical_indicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8')

class MLTradingStrategy:
    """ML-based trading strategy backtester"""
    
    def __init__(self, model_package, initial_capital=100000, transaction_cost=0.001, use_signals=False, use_model=False, use_technical=False):
        self.model_package = model_package  # Store for access to features_df/target
        self.model = model_package['model']
        self.feature_names = model_package['feature_names']
        self.target_symbol = model_package['target_symbol']
        self.config = model_package['config']
        # Store new flags
        self.use_signals = use_signals
        self.use_model = use_model
        self.use_technical = use_technical
        
        # Debug model scaling information
        model_type = self.config.get('model_type', 'unknown')
        
        print(f"🤖 Model Setup:")
        print(f"  Model Type: {model_type}")
        print(f"  Use signals: {self.use_signals}")
        print(f"  Use model: {self.use_model}")
        print(f"  Use technical: {self.use_technical}")
        
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost  # 0.1% per trade
        
        self.trades = []
        self.portfolio_history = []
        
    def prepare_data(self):
        """Prepare data for backtesting using strategy period (not training period), using stored features_df and target from model_package"""
        print(f"📊 Preparing data for {self.target_symbol}...")

        # Load periods from config
        periods = self.get_periods_from_config()
        print(f"📅 Training period: {periods['training']['start']} to {periods['training']['end']}")
        print(f"📅 Strategy period: {periods['strategy']['start']} to {periods['strategy']['end']}")

        # Use features_df and target from model_package (no regeneration)
        if not hasattr(self, 'model_package'):
            raise ValueError("MLTradingStrategy must be initialized with model_package as an attribute.")
        model_package = self.model_package
        if 'features_df' not in model_package or 'target' not in model_package:
            print(f"❌ Model package missing features_df or target. Please retrain model.")
            return None
        features_df = model_package['features_df']
        target = model_package['target']

        # Get price data for actual trading
        db_manager = DatabaseManager()
        with db_manager:
            price_data = db_manager.get_stock_prices(self.target_symbol)

        # Drop 'close' from features_df if it exists to avoid join overlap
        if 'close' in features_df.columns:
            features_df = features_df.drop(columns=['close'])

        # Merge features with price data
        data = features_df.join(price_data[['close']], how='inner')
        # Add target column for backtest reference
        data['target'] = target.reindex(data.index)

        # CRITICAL: Filter to strategy period only (not training period)
        strategy_start = pd.to_datetime(periods['strategy']['start'])
        strategy_end = pd.to_datetime(periods['strategy']['end'])
        strategy_data = data[(data.index >= strategy_start) & (data.index <= strategy_end)].copy()

        if strategy_data.empty:
            print(f"❌ No data available for strategy period {strategy_start} to {strategy_end}")
            print(f"📊 Available data range: {data.index.min()} to {data.index.max()}")
            return None

        print(f"✅ Prepared {len(strategy_data)} strategy period trading days")
        print(f"📊 Features available: {len([f for f in self.feature_names if f in strategy_data.columns])}/{len(self.feature_names)}")
        print(f"🎯 Strategy period: {strategy_data.index.min()} to {strategy_data.index.max()}")

        return strategy_data

    def get_periods_from_config(self):
        """Extract training and strategy periods from config, with fallback logic"""
        
        # Try to get periods from config
        if 'periods' in self.config:
            periods = self.config['periods']
            if 'training' in periods and 'strategy' in periods:
                print("📅 Using periods from config.yaml")
                return periods
        
        print("📅 Periods not found in config - using automatic split")
        
        # Fallback: Use automatic split (90% training, 10% strategy)
        # Get all available data to determine date range
        db_manager = DatabaseManager()
        with db_manager:
            price_data = db_manager.get_stock_prices(self.target_symbol)
        
        if price_data.empty:
            raise ValueError(f"No price data available for {self.target_symbol}")
        
        # Calculate split dates
        min_date = price_data.index.min()
        max_date = price_data.index.max()
        total_days = (max_date - min_date).days
        
        # 90% for training, 10% for strategy
        training_days = int(total_days * 0.9)
        training_end = min_date + pd.Timedelta(days=training_days)
        
        periods = {
            'training': {
                'start': min_date.strftime('%Y-%m-%d'),
                'end': training_end.strftime('%Y-%m-%d')
            },
            'strategy': {
                'start': (training_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                'end': max_date.strftime('%Y-%m-%d')
            }
        }
        
        print(f"📊 Auto-calculated periods:")
        print(f"  Training: {periods['training']['start']} to {periods['training']['end']} ({training_days} days)")
        print(f"  Strategy: {periods['strategy']['start']} to {periods['strategy']['end']} ({total_days - training_days} days)")
        
        return periods
    
    def add_lag_features(self, features_df):
        """Add lag features based on model requirements"""
        
        lag_features = [f for f in self.feature_names if 'lag' in f]
        if lag_features:
            print(f"📊 Adding {len(lag_features)} lag features...")
            for feature in lag_features:
                if '_lag_' in feature:
                    base_col = feature.split('_lag_')[0]
                    lag_periods = int(feature.split('_lag_')[1])
                    
                    # Map to actual column names in the dataframe
                    col_mapping = {
                        'close': 'close',
                        'high': 'high', 
                        'low': 'low',
                        'open': 'open',
                        'volume': 'volume'
                    }
                    
                    actual_col = col_mapping.get(base_col.lower(), base_col)
                    if actual_col in features_df.columns:
                        features_df[feature] = features_df[actual_col].shift(lag_periods)
                    else:
                        features_df[feature] = 0
                        print(f"⚠️ Column {actual_col} not found for lag feature {feature}")
        
        return features_df
    
    def add_rolling_features(self, features_df):
        """Add rolling features based on model requirements"""
        
        rolling_features = [f for f in self.feature_names if 'roll' in f]
        if rolling_features:
            print(f"📊 Adding {len(rolling_features)} rolling features...")
            for feature in rolling_features:
                if '_roll_' in feature:
                    parts = feature.split('_')
                    if len(parts) >= 4:  # e.g., close_roll_5_mean
                        base_col = parts[0]
                        window = int(parts[2])
                        func = parts[3]
                        
                        # Map to actual column names
                        col_mapping = {
                            'close': 'close',
                            'high': 'high',
                            'low': 'low', 
                            'open': 'open',
                            'volume': 'volume'
                        }
                        
                        actual_col = col_mapping.get(base_col.lower(), base_col)
                        if actual_col in features_df.columns:
                            if func == 'mean':
                                features_df[feature] = features_df[actual_col].rolling(window).mean()
                            elif func == 'std':
                                features_df[feature] = features_df[actual_col].rolling(window).std()
                            elif func == 'min':
                                features_df[feature] = features_df[actual_col].rolling(window).min()
                            elif func == 'max':
                                features_df[feature] = features_df[actual_col].rolling(window).max()
                            else:
                                features_df[feature] = 0
                        else:
                            features_df[feature] = 0
                            print(f"⚠️ Column {actual_col} not found for rolling feature {feature}")
        
        return features_df
    
    def generate_signals(self, data, strategy_type='threshold', threshold=0.01, technical_signals=True):
        """Generate trading signals based on model predictions and/or technical indicators, respecting CLI flags."""
        print(f"🔮 Generating {strategy_type} signals...")

        # Debug: Check feature availability
        available_features = set(data.columns)
        required_features = set(self.feature_names)
        missing_features = required_features - available_features

        if missing_features:
            print(f"⚠️ Missing {len(missing_features)} features from model:")
            print(f"   Sample missing: {list(missing_features)[:5]}...")
            # Create missing features with appropriate default values
            for feature in missing_features:
                if 'lag' in feature:
                    base_col = feature.split('_lag_')[0]
                    lag_periods = int(feature.split('_lag_')[1])
                    if base_col in data.columns:
                        data[feature] = data[base_col].shift(lag_periods)
                    else:
                        data[feature] = 0
                elif 'roll' in feature:
                    parts = feature.split('_')
                    if len(parts) >= 4:
                        base_col = parts[0]
                        window = int(parts[2])
                        func = parts[3]
                        if base_col in data.columns:
                            if func == 'mean':
                                data[feature] = data[base_col].rolling(window).mean()
                            elif func == 'std':
                                data[feature] = data[base_col].rolling(window).std()
                            elif func == 'min':
                                data[feature] = data[base_col].rolling(window).min()
                            elif func == 'max':
                                data[feature] = data[base_col].rolling(window).max()
                            else:
                                data[feature] = 0
                        else:
                            data[feature] = 0
                else:
                    data[feature] = 0
            print(f"✅ Added {len(missing_features)} missing features")

        # Ensure all required features are present and in correct order
        feature_data = data[self.feature_names].copy()
        feature_data = feature_data.fillna(method='ffill').fillna(0)

        signals = []
        predictions = []
        confidences = []
        tech_signals = []

        # --- Technical signal logic ---
        # Use technical signals if either flag is set
        use_tech = (self.use_signals or self.use_technical) and any(col.lower() == 'close' for col in data.columns) and 'sma_10' in data.columns
        use_model = self.use_model

        # Find the actual close column name (case-insensitive)
        close_col = next((col for col in data.columns if col.lower() == 'close'), None)
        prev_price = None
        prev_sma = None
        for i, (date, row) in enumerate(feature_data.iterrows()):
            # --- Model prediction logic ---
            prediction = 0
            model_signal = 0
            if use_model:
                feature_values = row.values.reshape(1, -1)
                if np.isnan(feature_values).any():
                    feature_values = np.nan_to_num(feature_values, 0)
                try:
                    prediction = self.model.predict(feature_values)[0]
                except Exception as e:
                    print(f"⚠️ Error predicting for {date}: {e}")
                    prediction = 0
                if strategy_type == 'threshold':
                    model_signal = 1 if prediction > threshold else -1 if prediction < -threshold else 0
                elif strategy_type == 'directional':
                    model_signal = np.sign(prediction)
                else:
                    model_signal = 0
            predictions.append(prediction)

            # --- Technical signal: price crosses SMA (e.g., 10-day) ---
            tech_signal = 0
            if use_tech and close_col is not None:
                price = data.loc[date, close_col]
                sma = data.loc[date, 'sma_10']
                if prev_price is not None and prev_sma is not None:
                    if prev_price < prev_sma and price > sma:
                        tech_signal = 1  # Bullish crossover
                    elif prev_price > prev_sma and price < sma:
                        tech_signal = -1  # Bearish crossover
                prev_price = price
                prev_sma = sma
            tech_signals.append(tech_signal)

            # --- Combine model and technical signals based on flags ---
            # If both are enabled, require both to agree (strongest), or allow either (more signals)
            if use_model and use_tech:
                # OR logic: if either is nonzero, use that signal (if both nonzero and agree, use that; if both nonzero and disagree, use 0)
                if model_signal != 0 and tech_signal != 0:
                    if model_signal == tech_signal:
                        signal = model_signal
                    else:
                        signal = 0  # Disagreement: no trade (or could use model_signal or tech_signal)
                elif model_signal != 0:
                    signal = model_signal
                elif tech_signal != 0:
                    signal = tech_signal
                else:
                    signal = 0
            elif use_model:
                signal = model_signal
            elif use_tech:
                signal = tech_signal
            else:
                signal = 0  # No signals if neither flag is set

            signals.append(signal)
            # Set confidence: if using only technical, set to 1; if using model, use abs(prediction)
            if use_model:
                confidence = abs(prediction)
            elif use_tech:
                confidence = 1
            else:
                confidence = 0
            confidences.append(confidence)

        data['prediction'] = predictions
        data['signal'] = signals
        data['confidence'] = confidences
        data['tech_signal'] = tech_signals

        # Use stored target as actual returns for backtest
        if 'target' in data.columns:
            data['actual_return'] = data['target']

        # Debug: Show prediction statistics
        pred_stats = pd.Series(predictions)
        print(f"📊 Prediction Statistics:")
        print(f"  Count: {len(predictions)}")
        print(f"  Min: {pred_stats.min():.6f}")
        print(f"  Max: {pred_stats.max():.6f}")
        print(f"  Mean: {pred_stats.mean():.6f}")
        print(f"  Std: {pred_stats.std():.6f}")

        signal_counts = pd.Series(signals).value_counts()
        print(f"📈 Signals generated:")
        print(f"  Buy (1):  {signal_counts.get(1, 0)}")
        print(f"  Hold (0): {signal_counts.get(0, 0)}")
        print(f"  Sell (-1): {signal_counts.get(-1, 0)}")

        threshold_check = abs(pred_stats.max()) if abs(pred_stats.max()) > abs(pred_stats.min()) else abs(pred_stats.min())
        if threshold_check > 0:
            print(f"  Max prediction magnitude: {threshold_check:.6f}")
            print(f"  Threshold: {threshold:.6f}")
            print(f"  Ratio: {threshold_check/threshold:.2f}x threshold")

        return data
    
    def backtest_strategy(self, data, max_position_size=1.0, stop_loss=None, take_profit=None):
        """Run the backtesting simulation with optional stop-loss and take-profit"""
        print(f"🏃‍♂️ Running backtest simulation...")
        capital = self.initial_capital
        position = 0  # Current position (-1 = short, 0 = neutral, 1 = long)
        shares = 0
        entry_price = None
        self.trades = []
        self.portfolio_history = []
        
        # Ensure stop_loss and take_profit are positive (percentages)
        if stop_loss is not None and stop_loss < 0:
            print(f"⚠️ Warning: stop_loss in config is negative ({stop_loss}); should be positive. Using abs value.")
            stop_loss = abs(stop_loss)
        if take_profit is not None and take_profit < 0:
            print(f"⚠️ Warning: take_profit in config is negative ({take_profit}); should be positive. Using abs value.")
            take_profit = abs(take_profit)

        for i, (date, row) in enumerate(data.iterrows()):
            price = row['close']
            signal = row['signal']
            prediction = row['prediction']
            actual_return = row['actual_return'] if 'actual_return' in row else 0
            confidence = row['confidence'] if 'confidence' in row else 1.0
            position_size = min(max_position_size, confidence * 2)

            # --- Stop-loss and take-profit logic for long positions (PREEMPTIVE) ---
            stop_or_take_triggered = False
            if shares > 0 and entry_price is not None:
                if stop_loss is not None and price <= entry_price * (1 - stop_loss):
                    capital += shares * price * (1 - self.transaction_cost)
                    self.trades.append({
                        'date': date,
                        'action': 'stop_loss',
                        'price': price,
                        'shares': shares,
                        'value': shares * price,
                        'capital': capital,
                        'prediction': prediction
                    })
                    shares = 0
                    position = 0
                    entry_price = None
                    stop_or_take_triggered = True
                elif take_profit is not None and price >= entry_price * (1 + take_profit):
                    capital += shares * price * (1 - self.transaction_cost)
                    self.trades.append({
                        'date': date,
                        'action': 'take_profit',
                        'price': price,
                        'shares': shares,
                        'value': shares * price,
                        'capital': capital,
                        'prediction': prediction
                    })
                    shares = 0
                    position = 0
                    entry_price = None
                    stop_or_take_triggered = True
            # If stop-loss or take-profit triggered, skip rest of this bar
            if stop_or_take_triggered:
                current_portfolio_value = capital + (shares * price if shares != 0 else 0)
                self.portfolio_history.append({
                    'date': date,
                    'price': price,
                    'capital': capital,
                    'shares': shares,
                    'portfolio_value': current_portfolio_value,
                    'position': position,
                    'signal': signal,
                    'prediction': prediction,
                    'actual_return': actual_return
                })
                continue

            # --- Signal-based position closure logic (only if no stop/take triggered) ---
            if shares > 0 and signal != 0:
                capital += shares * price * (1 - self.transaction_cost)
                self.trades.append({
                    'date': date,
                    'action': 'sell',
                    'price': price,
                    'shares': shares,
                    'value': shares * price,
                    'capital': capital,
                    'prediction': prediction
                })
                shares = 0
                position = 0
                entry_price = None
            if shares < 0 and signal != 0:
                capital += -shares * price * (1 - self.transaction_cost)
                self.trades.append({
                    'date': date,
                    'action': 'cover',
                    'price': price,
                    'shares': -shares,
                    'value': -shares * price,
                    'capital': capital,
                    'prediction': prediction
                })
                shares = 0
                position = 0
                entry_price = None

            # Open new position if signal is buy or sell
            if signal == 1:
                shares_to_buy = int((capital * position_size) / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.transaction_cost)
                    if cost <= capital:
                        capital -= cost
                        shares += shares_to_buy
                        position = 1
                        entry_price = price
                        self.trades.append({
                            'date': date,
                            'action': 'buy',
                            'price': price,
                            'shares': shares_to_buy,
                            'value': cost,
                            'capital': capital,
                            'prediction': prediction,
                            'entry_price': price
                        })
            # Uncomment below to allow shorting
            # elif signal == -1:
            #     shares_to_short = int((capital * position_size) / price)
            #     if shares_to_short > 0:
            #         cost = shares_to_short * price * (1 + self.transaction_cost)
            #         if cost <= capital:
            #             capital += shares_to_short * price * (1 - self.transaction_cost)
            #             shares -= shares_to_short
            #             position = -1
            #             entry_price = price
            current_portfolio_value = capital + (shares * price if shares != 0 else 0)
            self.portfolio_history.append({
                'date': date,
                'price': price,
                'capital': capital,
                'shares': shares,
                'portfolio_value': current_portfolio_value,
                'position': position,
                'signal': signal,
                'prediction': prediction,
                'actual_return': actual_return
            })
        # Close final position
        if shares != 0:
            final_price = data['close'].iloc[-1]
            capital += shares * final_price * (1 - self.transaction_cost)
            self.trades.append({
                'date': data.index[-1],
                'action': 'close',
                'price': final_price,
                'shares': shares,
                'value': shares * final_price,
                'capital': capital,
                'prediction': 0
            })
        print(f"✅ Backtest complete!")
        print(f"  Total trades: {len(self.trades)}")
        print(f"  Final capital: ${capital:,.2f}")
        print(f"  Total return: {(capital/self.initial_capital - 1)*100:.2f}%")
        return capital
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        
        if not self.portfolio_history:
            return None
        
        df = pd.DataFrame(self.portfolio_history)
        
        # Calculate returns
        df['strategy_return'] = df['portfolio_value'].pct_change()
        df['buy_hold_value'] = self.initial_capital * (df['price'] / df['price'].iloc[0])
        df['buy_hold_return'] = df['buy_hold_value'].pct_change()
        
        # Performance metrics
        total_days = len(df)
        trading_days_per_year = 252
        
        # Returns
        total_strategy_return = (df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        total_buy_hold_return = (df['buy_hold_value'].iloc[-1] / self.initial_capital) - 1
        
        # Annualized returns
        years = total_days / trading_days_per_year
        annualized_strategy_return = (1 + total_strategy_return) ** (1/years) - 1
        annualized_buy_hold_return = (1 + total_buy_hold_return) ** (1/years) - 1
        
        # Volatility
        strategy_volatility = df['strategy_return'].std() * np.sqrt(trading_days_per_year)
        buy_hold_volatility = df['buy_hold_return'].std() * np.sqrt(trading_days_per_year)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_strategy = annualized_strategy_return / strategy_volatility if strategy_volatility > 0 else 0
        sharpe_buy_hold = annualized_buy_hold_return / buy_hold_volatility if buy_hold_volatility > 0 else 0
        
        # Maximum drawdown
        strategy_peak = df['portfolio_value'].expanding().max()
        strategy_drawdown = (df['portfolio_value'] - strategy_peak) / strategy_peak
        max_drawdown_strategy = strategy_drawdown.min()
        
        buy_hold_peak = df['buy_hold_value'].expanding().max()
        buy_hold_drawdown = (df['buy_hold_value'] - buy_hold_peak) / buy_hold_peak
        max_drawdown_buy_hold = buy_hold_drawdown.min()
        
        # Win rate
        profitable_trades = sum(1 for trade in self.trades[1:] if trade.get('action') in ['sell', 'cover'])
        if profitable_trades > 0:
            # Calculate P&L for each trade pair
            buy_trades = [t for t in self.trades if t.get('action') == 'buy']
            sell_trades = [t for t in self.trades if t.get('action') == 'sell']
            wins = sum(1 for b, s in zip(buy_trades, sell_trades) if s['price'] > b['price'])
            win_rate = wins / min(len(buy_trades), len(sell_trades)) if buy_trades and sell_trades else 0
        else:
            win_rate = 0
        
        metrics = {
            'total_strategy_return': total_strategy_return,
            'total_buy_hold_return': total_buy_hold_return,
            'annualized_strategy_return': annualized_strategy_return,
            'annualized_buy_hold_return': annualized_buy_hold_return,
            'strategy_volatility': strategy_volatility,
            'buy_hold_volatility': buy_hold_volatility,
            'sharpe_strategy': sharpe_strategy,
            'sharpe_buy_hold': sharpe_buy_hold,
            'max_drawdown_strategy': max_drawdown_strategy,
            'max_drawdown_buy_hold': max_drawdown_buy_hold,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'days_traded': total_days
        }
        
        return metrics, df
    
    def plot_results(self, df, metrics):
        """Create comprehensive performance visualization"""
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Portfolio value over time
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(df['date'], df['portfolio_value'], label='ML Strategy', linewidth=2)
        plt.plot(df['date'], df['buy_hold_value'], label='Buy & Hold', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.title(f'Portfolio Performance - {self.target_symbol}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. Cumulative returns
        ax2 = plt.subplot(2, 3, 2)
        strategy_cumret = (df['portfolio_value'] / self.initial_capital - 1) * 100
        buyhold_cumret = (df['buy_hold_value'] / self.initial_capital - 1) * 100
        
        plt.plot(df['date'], strategy_cumret, label=f'Strategy ({strategy_cumret.iloc[-1]:.1f}%)', linewidth=2)
        plt.plot(df['date'], buyhold_cumret, label=f'Buy & Hold ({buyhold_cumret.iloc[-1]:.1f}%)', linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.title('Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. Drawdown
        ax3 = plt.subplot(2, 3, 3)
        strategy_peak = df['portfolio_value'].expanding().max()
        strategy_dd = (df['portfolio_value'] - strategy_peak) / strategy_peak * 100
        
        plt.fill_between(df['date'], strategy_dd, 0, alpha=0.3, color='red')
        plt.plot(df['date'], strategy_dd, color='red', linewidth=1)
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.title(f'Strategy Drawdown (Max: {metrics["max_drawdown_strategy"]*100:.1f}%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 4. Trade signals
        ax4 = plt.subplot(2, 3, 4)
        plt.plot(df['date'], df['price'], label='Price', alpha=0.7)
        
        # Mark buy/sell signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        plt.scatter(buy_signals['date'], buy_signals['price'], 
                   color='green', marker='^', s=50, label='Buy Signal', alpha=0.8)
        plt.scatter(sell_signals['date'], sell_signals['price'], 
                   color='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
        
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title('Trading Signals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 5. Prediction vs actual returns
        ax5 = plt.subplot(2, 3, 5)
        plt.scatter(df['prediction'], df['actual_return'], alpha=0.6, s=20)
        plt.xlabel('Predicted Return')
        plt.ylabel('Actual Return')
        plt.title('Prediction Accuracy')
        
        # Add diagonal line
        min_val = min(df['prediction'].min(), df['actual_return'].min())
        max_val = max(df['prediction'].max(), df['actual_return'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.grid(True, alpha=0.3)
        
        # 6. Performance metrics table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        metrics_text = f"""
Performance Metrics

Strategy:
  Total Return: {metrics['total_strategy_return']*100:.2f}%
  Annualized: {metrics['annualized_strategy_return']*100:.2f}%
  Volatility: {metrics['strategy_volatility']*100:.2f}%
  Sharpe Ratio: {metrics['sharpe_strategy']:.2f}
  Max Drawdown: {metrics['max_drawdown_strategy']*100:.2f}%

Buy & Hold:
  Total Return: {metrics['total_buy_hold_return']*100:.2f}%
  Annualized: {metrics['annualized_buy_hold_return']*100:.2f}%
  Volatility: {metrics['buy_hold_volatility']*100:.2f}%
  Sharpe Ratio: {metrics['sharpe_buy_hold']:.2f}
  Max Drawdown: {metrics['max_drawdown_buy_hold']*100:.2f}%

Trading:
  Total Trades: {metrics['total_trades']}
  Win Rate: {metrics['win_rate']*100:.1f}%
  Days Traded: {metrics['days_traded']}
        """
        
        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot in data/model_outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_dir = Path(__file__).parent.parent / 'data' / 'model_outputs'
        plot_file = plots_dir / f'backtest_results_{self.target_symbol}_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"💾 Backtest chart saved: {plot_file}")
        
        plt.show()

def find_latest_model_for_symbol(symbol):
    """Find the latest trained model for a specific symbol"""
    
    # Fix: Look in /data/model_outputs instead of /models
    models_dir = Path(__file__).parent.parent / 'data' / 'model_outputs'
    
    # Look for models with the symbol in the filename
    symbol_models = list(models_dir.glob(f'*{symbol}*.pkl'))
    
    if not symbol_models:
        print(f"❌ No trained models found for symbol: {symbol}")
        print(f"💡 Available models in {models_dir}:")
        all_models = list(models_dir.glob('*.pkl'))
        if all_models:
            for model in all_models:
                print(f"   {model.name}")
        else:
            print(f"   No .pkl files found in {models_dir}")
            # Check if directory exists
            if not models_dir.exists():
                print(f"   Directory doesn't exist: {models_dir}")
        return None
    
    # Get the latest model for this symbol
    latest_model = max(symbol_models, key=lambda x: x.stat().st_mtime)
    print(f"📦 Found latest model: {latest_model.name}")
    return latest_model

def load_trading_config():
    """Load trading strategy parameters from config.yaml if available."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('trading', {})
    return {}

def run_backtest(target_symbol=None, initial_capital=100000, transaction_cost=0.001, max_position_size=1.0, use_signals=False, use_model=False, use_technical=False, stop_loss=None, take_profit=None):
    """Run complete backtesting workflow for specified symbol, using config values."""
    print("🏁 ML Trading Strategy Backtester")
    print("="*50)
    
    # Load model for specified symbol
    if target_symbol:
        print(f"🎯 Looking for models for symbol: {target_symbol}")
        model_file = find_latest_model_for_symbol(target_symbol)
        if model_file is None:
            print(f"\n💡 To train a model for {target_symbol}, run:")
            print(f"   python scripts/train_enhanced_model.py --symbol {target_symbol}")
            return None
    else:
        # Fallback to latest model if no symbol specified
        models_dir = Path(__file__).parent.parent / 'data' / 'model_outputs'  # Fixed path
        model_files = list(models_dir.glob('*.pkl'))
        
        if not model_files:
            print(f"❌ No trained models found in {models_dir}")
            return None
        
        model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        print("⚠️ No symbol specified, using latest model")
    
    # Load the model
    try:
        model_package = joblib.load(model_file)
        print(f"📂 Using model: {model_file.name}")
        print(f"📊 Target symbol: {model_package['target_symbol']}")
        
        # Verify symbol matches if specified
        if target_symbol and model_package['target_symbol'] != target_symbol:
            print(f"⚠️ Warning: Model is for {model_package['target_symbol']}, but {target_symbol} was requested")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Initialize strategy
    strategy = MLTradingStrategy(
        model_package=model_package,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        use_signals=use_signals,
        use_model=use_model,
        use_technical=use_technical
    )
    
    # Prepare data
    data = strategy.prepare_data()
    if data is None:
        print("❌ Failed to prepare data")
        return None
    
    # Test different strategies
    strategies = {
        'directional': {'strategy_type': 'directional'},
        'threshold_1pct': {'strategy_type': 'threshold', 'threshold': 0.01},
        'threshold_2pct': {'strategy_type': 'threshold', 'threshold': 0.02},
        'quantile': {'strategy_type': 'quantile'}
    }
    
    results = {}
    
    for strategy_name, params in strategies.items():
        print(f"\n🧪 Testing {strategy_name} strategy...")
        
        # Generate signals
        data_with_signals = strategy.generate_signals(data, **params)
        
        # Run backtest (now passes stop_loss and take_profit)
        final_capital = strategy.backtest_strategy(data_with_signals, max_position_size=max_position_size, stop_loss=stop_loss, take_profit=take_profit)
        
        # Calculate metrics
        metrics, df = strategy.calculate_performance_metrics()
        
        results[strategy_name] = {
            'final_capital': final_capital,
            'metrics': metrics,
            'data': df
        }
        
        print(f"✅ {strategy_name}: ${final_capital:,.2f} ({(final_capital/100000-1)*100:.2f}%)")
    
    # Find best strategy
    best_strategy = max(results.keys(), key=lambda x: results[x]['final_capital'])
    print(f"\n🏆 Best strategy: {best_strategy}")
    print(f"Final value: ${results[best_strategy]['final_capital']:,.2f}")
    
    # Plot results for best strategy
    best_metrics = results[best_strategy]['metrics']
    best_data = results[best_strategy]['data']
    
    # Re-run the best strategy for plotting
    strategy_params = strategies[best_strategy]
    data_with_signals = strategy.generate_signals(data, **strategy_params)
    strategy.backtest_strategy(data_with_signals, max_position_size=0.8)
    strategy.plot_results(best_data, best_metrics)
    
    # Save results - also save to /data/model_outputs for consistency
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    symbol = model_package['target_symbol']
    
    # Save in data/model_outputs directory
    results_dir = Path(__file__).parent.parent / 'data' / 'model_outputs'
    results_file = results_dir / f'backtest_summary_{symbol}_{timestamp}.csv'
    
    summary_data = []
    for name, result in results.items():
        row = {'strategy': name}
        row.update(result['metrics'])
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_file, index=False)
    print(f"💾 Results summary saved: {results_file}")
    
    return results

def main():
    """Main function with argument parsing"""
    import argparse
    parser = argparse.ArgumentParser(description='Backtest ML trading strategy for a specific symbol')
    parser.add_argument('--symbol', type=str, help='Stock symbol to backtest (e.g., ACN, AAPL)')
    parser.add_argument('--capital', type=float, help='Initial capital (overrides config)')
    parser.add_argument('--transaction-cost', type=float, help='Transaction cost (overrides config)')
    parser.add_argument('--max-position-size', type=float, help='Max position size (overrides config)')
    parser.add_argument('--use-signals', action='store_true', help='Use trading signals (rules-based, e.g. crossovers)')
    parser.add_argument('--use-model', action='store_true', help='Use predictive model for trading signals')
    parser.add_argument('--use-technical', action='store_true', help='Use technical indicators (e.g., RSI, SMA) for trading signals')
    args = parser.parse_args()

    # Load trading config from config.yaml
    trading_config = load_trading_config()

    # Use config.yaml as default, allow CLI override
    initial_capital = args.capital if args.capital is not None else trading_config.get('initial_capital', 100000)
    transaction_cost = args.transaction_cost if args.transaction_cost is not None else trading_config.get('transaction_cost', 0.001)
    max_position_size = args.max_position_size if args.max_position_size is not None else trading_config.get('max_position_size', 1.0)
    stop_loss = trading_config.get('stop_loss', None)
    take_profit = trading_config.get('take_profit', None)

    if args.symbol:
        print(f"🎯 Running backtest for: {args.symbol.upper()}")
        target_symbol = args.symbol.upper()
    else:
        print("⚠️ No symbol specified - will use latest available model")
        target_symbol = None

    # Pass new flags to run_backtest
    results = run_backtest(
        target_symbol,
        initial_capital,
        transaction_cost,
        max_position_size,
        use_signals=args.use_signals,
        use_model=args.use_model,
        use_technical=args.use_technical,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

    if results:
        print(f"\n✅ Backtest completed successfully!")
    else:
        print(f"\n❌ Backtest failed")

if __name__ == "__main__":
    main()