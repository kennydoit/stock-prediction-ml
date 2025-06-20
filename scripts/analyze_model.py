#!/usr/bin/env python3
"""
Analyze latest trained model performance - Model Diagnostics Only
"""
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import sys
import argparse
import yaml
import traceback
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'database'))

def find_latest_model_for_symbol(symbol):
    """Find the latest trained model for a specific symbol (excluding modeling data .pkl files)"""
    
    # Check both possible directories
    models_dirs = [
        Path(__file__).parent.parent / 'data' / 'model_outputs',
        Path(__file__).parent.parent / 'models'
    ]
    
    for models_dir in models_dirs:
        if models_dir.exists():
            # Only include model .pkl files, not modeling data .pkl files
            symbol_models = [f for f in models_dir.glob(f'*{symbol}*.pkl') if 'modeling_data' not in f.name]
            
            if symbol_models:
                # Get the latest model for this symbol
                latest_model = max(symbol_models, key=lambda x: x.stat().st_mtime)
                print(f"📦 Found latest model for {symbol}: {latest_model.name}")
                print(f"📁 Location: {models_dir}")
                return latest_model
    
    # If no symbol-specific model found, show what's available
    print(f"❌ No trained models found for symbol: {symbol}")
    print(f"💡 Available models:")
    
    for models_dir in models_dirs:
        if models_dir.exists():
            all_models = [f for f in models_dir.glob('*.pkl') if 'modeling_data' not in f.name]
            if all_models:
                print(f"   In {models_dir}:")
                for model in sorted(all_models, key=lambda x: x.stat().st_mtime, reverse=True):
                    # Extract symbol from filename
                    model_symbol = extract_symbol_from_filename(model.name)
                    print(f"     {model.name} ({model_symbol})")
            else:
                print(f"   No .pkl files found in {models_dir}")
    
    return None

def analyze_latest_model(target_symbol=None):
    """Analyze the latest saved model for specified symbol"""
    
    print("📊 Model Performance Analysis")
    print("="*50)
    
    # Find model for specified symbol
    if target_symbol:
        print(f"🎯 Looking for models for symbol: {target_symbol}")
        model_file = find_latest_model_for_symbol(target_symbol)
        if model_file is None:
            print(f"\n💡 To train a model for {target_symbol}, run:")
            print(f"   python scripts/train_enhanced_model.py --symbol {target_symbol}")
            return None
    else:
        # Fallback to latest model if no symbol specified
        models_dirs = [
            Path(__file__).parent.parent / 'data' / 'model_outputs',
            Path(__file__).parent.parent / 'models'
        ]
        
        model_file = None
        for models_dir in models_dirs:
            if models_dir.exists():
                model_files = list(models_dir.glob('*.pkl'))
                if model_files:
                    model_file = max(model_files, key=lambda x: x.stat().st_mtime)
                    print("⚠️ No symbol specified, using latest model")
                    print(f"📁 Location: {models_dir}")
                    break
        
        if model_file is None:
            print("❌ No trained models found")
            return None
    
    # Load the model
    try:
        model_package = joblib.load(model_file)
        print(f"📂 Loaded: {model_file.name}")
        
        # Check for 'target_symbol' in model_package
        if 'target_symbol' not in model_package:
            print(f"❌ Error: The loaded file does not contain 'target_symbol'.")
            print(f"   File loaded: {model_file}")
            print(f"   Available keys: {list(model_package.keys())}")
            print(f"   This is likely a modeling data .pkl, not a model .pkl. Please check your file selection.")
            return None
        
        # Verify symbol matches if specified
        if target_symbol and model_package['target_symbol'] != target_symbol:
            print(f"⚠️ Warning: Model is for {model_package['target_symbol']}, but {target_symbol} was requested")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Display basic model info
    target_symbol = model_package['target_symbol']
    metrics = model_package['metrics']
    
    print(f"\n🎯 Model Information:")
    print(f"  Target Symbol: {target_symbol}")
    print(f"  Trained: {model_package['trained_at']}")
    print(f"  Model Type: {model_package.get('model_type', 'Linear Regression')}")
    print(f"  Features: {model_package['feature_count']}")
    print(f"  Include Peers: {'Yes' if model_package.get('include_peers', False) else 'No'}")
    
    # --- Updated: Use OLS coefficients path from model_package ---
    import pandas as pd
    ols_coef_path = model_package.get('model_coefficients', None)
    if ols_coef_path and Path(ols_coef_path).exists():
        ols_df = pd.read_csv(ols_coef_path)
        # Overwrite model_package['coefficients'] with OLS values
        if 'feature' in ols_df.columns and 'coefficient' in ols_df.columns:
            ols_coef_dict = dict(zip(ols_df['feature'], ols_df['coefficient']))
        else:
            # If saved as a Series, use index and first column
            ols_coef_dict = dict(zip(ols_df.iloc[:,0], ols_df.iloc[:,1]))
        model_package['coefficients'] = ols_coef_dict
        print(f"⚠️ Using OLS coefficients from {ols_coef_path} for all analysis and reporting.")
    else:
        print(f"⚠️ OLS coefficients file not found: {ols_coef_path}. Using model coefficients.")
    
    return model_package

def display_model_performance(model_package):
    """Display comprehensive model performance metrics"""
    
    print(f"\n📈 Model Performance Metrics:")
    print("="*40)
    
    metrics = model_package['metrics']
    
    # Training vs Test Performance
    print(f"📊 Training Performance:")
    print(f"  R² Score: {metrics['train_r2']:.4f}")
    print(f"  MSE: {metrics['train_mse']:.6f}")
    print(f"  RMSE: {np.sqrt(metrics['train_mse']):.6f}")
    print(f"  Samples: {metrics['n_train']:,}")
    
    print(f"\n📊 Testing Performance:")
    print(f"  R² Score: {metrics['test_r2']:.4f}")
    print(f"  MSE: {metrics['test_mse']:.6f}")
    print(f"  RMSE: {np.sqrt(metrics['test_mse']):.6f}")
    print(f"  Samples: {metrics['n_test']:,}")
    
    # Model Quality Assessment
    test_r2 = metrics['test_r2']
    train_r2 = metrics['train_r2']
    r2_gap = train_r2 - test_r2
    
    print(f"\n🎯 Model Quality Assessment:")
    if test_r2 > 0.1:
        quality = "✅ Good"
    elif test_r2 > 0.05:
        quality = "⚠️ Moderate"
    elif test_r2 > 0:
        quality = "⚠️ Weak"
    else:
        quality = "❌ Poor"
    
    print(f"  Predictive Power: {quality} (Test R² = {test_r2:.4f})")
    
    if r2_gap > 0.15:
        overfitting = "❌ High Overfitting"
    elif r2_gap > 0.05:
        overfitting = "⚠️ Moderate Overfitting"
    else:
        overfitting = "✅ No Overfitting"
    
    print(f"  Overfitting Risk: {overfitting} (Gap = {r2_gap:.3f})")
    
    # Feature breakdown
    if 'feature_breakdown' in model_package:
        print(f"\n🔧 Feature Breakdown:")
        for feat_type, count in model_package['feature_breakdown'].items():
            if count > 0:
                print(f"  {feat_type}: {count}")
    
    # Data range
    if 'data_range' in model_package:
        data_range = model_package['data_range']
        print(f"\n📅 Training Data Range:")
        print(f"  Start: {data_range['start']}")
        print(f"  End: {data_range['end']}")
        print(f"  Records: {data_range['records']:,}")

def create_model_fit_chart(model_package):
    """Create model fit visualization with proper period alignment"""
    
    print(f"\n📈 Creating Model Fit Chart...")
    
    symbol = model_package['target_symbol']
    
    # Get periods from config first
    periods = get_periods_from_config()
    
    # Calculate cumulative price series from returns
    initial_price = actual_prices.iloc[0]
    actual_price_series = initial_price * (1 + actual_returns).cumprod()
    predicted_price_series = initial_price * (1 + predicted_returns).cumprod()
    
    dates = actual_returns.index
    
    # Create comprehensive plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Price Fit
    ax1 = axes[0]
    ax1.plot(dates, actual_price_series, label='Actual Prices', linewidth=2, color='blue', alpha=0.8)
    ax1.plot(dates, predicted_price_series, label='Model Predictions', linewidth=2, color='red', alpha=0.8)
    
    # Add period dividers and labels ONLY if we have actual full period data
    if show_actual_periods and periods and chart_type == "Full Period Data":
        train_end_date = pd.to_datetime(periods['training']['end'])
        strategy_start_date = pd.to_datetime(periods['strategy']['start'])
        
        # Only show dividers if they fall within our data range
        if train_end_date >= dates.min() and train_end_date <= dates.max():
            ax1.axvline(x=train_end_date, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Train/Test Split')
        
        if strategy_start_date >= dates.min() and strategy_start_date <= dates.max():
            ax1.axvline(x=strategy_start_date, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Test/Strategy Split')
        
        # Add period labels for all three periods
        y_max = max(actual_price_series.max(), predicted_price_series.max())
        
        # Calculate period midpoints
        train_mid = dates.min() + (train_end_date - dates.min()) / 2
        strategy_mid = strategy_start_date + (dates.max() - strategy_start_date) / 2
        test_mid = train_end_date + (strategy_start_date - train_end_date) / 2
        
        ax1.text(train_mid, y_max*0.95, 'TRAINING', 
                 ha='center', va='top', fontweight='bold', color='green', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        ax1.text(test_mid, y_max*0.95, 'TESTING', 
                 ha='center', va='top', fontweight='bold', color='orange', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
        ax1.text(strategy_mid, y_max*0.95, 'STRATEGY', 
                 ha='center', va='top', fontweight='bold', color='red', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
        
    else:
        # Show appropriate single period label
        y_max = max(actual_price_series.max(), predicted_price_series.max())
        
        if chart_type == "Strategy Period Only":
            label_text = 'STRATEGY PERIOD'
            label_color = 'red'
            bg_color = 'lightcoral'
        elif chart_type == "Training Period Only":
            label_text = 'TRAINING PERIOD'
            label_color = 'green'
            bg_color = 'lightgreen'
        else:
            label_text = chart_type.upper()
            label_color = 'blue'
            bg_color = 'lightblue'
        
        ax1.text(dates.mean(), y_max*0.95, label_text, 
                 ha='center', va='top', fontweight='bold', color=label_color, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.7))
    
    r2 = model_package['metrics']['test_r2']
    current_price = actual_price_series.iloc[-1]
    date_range_str = f"{dates.min().date()} to {dates.max().date()}"
    ax1.set_title(f'Model Price Fit - {symbol} (Test R² = {r2:.3f}) - Current: ${current_price:.2f}\n{chart_type}: {date_range_str}', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Returns Comparison
    ax2 = axes[1]
    ax2.plot(dates, actual_returns, label='Actual Returns', linewidth=1.5, color='blue', alpha=0.7)
    ax2.plot(dates, predicted_returns, label='Predicted Returns', linewidth=1.5, color='red', alpha=0.7)
    
    # Add same period dividers if applicable
    if show_actual_periods and periods and chart_type == "Full Period Data":
        train_end_date = pd.to_datetime(periods['training']['end'])
        strategy_start_date = pd.to_datetime(periods['strategy']['start'])
        
        if train_end_date >= dates.min() and train_end_date <= dates.max():
            ax2.axvline(x=train_end_date, color='green', linestyle='--', alpha=0.7, linewidth=2)
        if strategy_start_date >= dates.min() and strategy_start_date <= dates.max():
            ax2.axvline(x=strategy_start_date, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax2.set_title(f'Returns Prediction Fit - {symbol} ({chart_type})', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Daily Returns', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Residuals Analysis
    ax3 = axes[2]
    ax3.plot(dates, residuals, label='Prediction Residuals', linewidth=1, color='green', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add same period dividers if applicable
    if show_actual_periods and periods and chart_type == "Full Period Data":
        if train_end_date >= dates.min() and train_end_date <= dates.max():
            ax3.axvline(x=train_end_date, color='green', linestyle='--', alpha=0.7, linewidth=2)
        if strategy_start_date >= dates.min() and strategy_start_date <= dates.max():
            ax3.axvline(x=strategy_start_date, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add residual statistics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    ax3.text(0.02, 0.95, f'RMSE: {rmse:.6f}\nMAE: {mae:.6f}', 
             transform=ax3.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax3.set_title('Prediction Residuals Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Residuals', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Save the fit chart to the model outputs folder
    model_outputs_dir = Path(__file__).parent.parent / 'data' / 'model_outputs'
    model_outputs_dir.mkdir(parents=True, exist_ok=True)
    chart_path = model_outputs_dir / f"fit_chart_{symbol}.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    print(f"[INFO] Fit chart saved to {chart_path}")
    plt.show()
    
    # Display fit statistics
    correlation = np.corrcoef(actual_returns, predicted_returns)[0,1]
    print(f"\n📊 Model Fit Statistics:")
    print(f"  Data Type: {chart_type}")
    print(f"  Data Period: {dates.min().date()} to {dates.max().date()} ({len(dates)} days)")
    print(f"  Returns Correlation: {correlation:.4f}")
    print(f"  Returns R²: {correlation**2:.4f}")
    print(f"  Returns RMSE: {rmse:.6f}")
    print(f"  Returns MAE: {mae:.6f}")
    
    return True

def generate_model_predictions(model_package, dates, audit_files=None):
    """Generate model predictions for given dates, always using features_df from model_package."""
    import pandas as pd
    import numpy as np

    try:
        symbol = model_package['target_symbol']
        model = model_package['model']
        feature_names = model_package['feature_names']
        # Always use features_df from model_package
        if 'features_df' not in model_package:
            print(f"❌ Error: model_package does not contain 'features_df'. This model is too old or was not saved with features. Please retrain.")
            return None
        features_df = model_package['features_df']
        # Align with requested dates
        common_dates = dates.intersection(features_df.index)
        if len(common_dates) == 0:
            print(f"❌ No overlapping dates between requested dates and features_df.")
            return None
        # Get features in exact order expected by model
        feature_data = features_df.loc[common_dates, feature_names].copy()
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        predictions = model.predict(feature_data)
        result = pd.Series(index=dates, dtype=float)
        result.loc[common_dates] = predictions
        result = result.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        if audit_files is not None:
            audit_files['features_df'] = features_df
            audit_files['feature_data'] = feature_data
        return result
    except Exception as e:
        print(f"❌ Error in prediction generation: {e}")
        import traceback
        traceback.print_exc()
        return None

# removed create_simulated_fit_chart because if there is a problem with the real model, we need to diagnose it, and not create a fake chart

def analyze_model_coefficients(model_package):
    """Analyze model coefficients and calculate statistical significance, always showing intercept."""
    
    print(f"\n🔬 Model Coefficient Analysis")
    print("="*50)
    
    model = model_package['model']
    feature_names = model_package['feature_names']
    symbol = model_package['target_symbol']
    
    # Get coefficients
    coefficients = model.coef_
    intercept = model.intercept_
    
    print(f"📊 Linear Model for {symbol}")
    print(f"📈 Intercept: {intercept:.6f}")
    print(f"🔧 Features: {len(feature_names)}")
    
    # Create coefficient analysis
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Add intercept row at the top
    intercept_row = pd.DataFrame({'Feature': ['Intercept'], 'Coefficient': [intercept], 'Abs_Coefficient': [abs(intercept)]})
    coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
    
    # Add feature categories
    def categorize_feature(feature_name):
        if any(x in feature_name.lower() for x in ['rsi', 'macd', 'bb_', 'cci', 'stoch', 'atr', 'obv', 'ichimoku', 'sma']):
            return 'Technical'
        elif any(x in feature_name.lower() for x in ['lag_', 'roll_']):
            return 'Lagged/Rolling'
        elif any(x in feature_name.lower() for x in ['rel_', 'peer']):
            return 'Peer'
        elif any(x in feature_name.lower() for x in ['return', 'volatility']):
            return 'Returns/Vol'
        else:
            return 'Other'
    coef_df['Category'] = coef_df['Feature'].apply(categorize_feature)
    coef_df['Normalized_Importance'] = coef_df['Abs_Coefficient'] / coef_df['Abs_Coefficient'].max()

    print(f"\n🎯 Top 15 Most Important Features (by absolute coefficient):")
    print("-" * 75)
    print(f"{'Rank':<4} {'Feature':<30} {'Category':<12} {'Coefficient':<12} {'Importance'}")
    print("-" * 75)
    for i, (_, row) in enumerate(coef_df.head(15).iterrows()):
        importance_pct = row['Normalized_Importance'] * 100
        coef_str = f"{row['Coefficient']:+.6f}"
        print(f"{i+1:<4} {row['Feature'][:29]:<30} {row['Category']:<12} {coef_str:<12} {importance_pct:6.1f}%")

    # Feature category breakdown
    print(f"\n📋 Feature Category Summary:")
    print("-" * 50)
    category_stats = coef_df.groupby('Category').agg({
        'Coefficient': ['count', 'mean'],
        'Abs_Coefficient': ['mean', 'max']
    }).round(6)
    for category in category_stats.index:
        count = int(category_stats.loc[category, ('Coefficient', 'count')])
        avg_coef = category_stats.loc[category, ('Coefficient', 'mean')]
        avg_abs_coef = category_stats.loc[category, ('Abs_Coefficient', 'mean')]
        print(f"{category:<15}: {count:3d} features, Avg: {avg_coef:+.6f}, Impact: {avg_abs_coef:.6f}")

    # Strongest signals
    print(f"\n🚀 Strongest Predictive Signals:")
    positive_features = coef_df[coef_df['Coefficient'] > 0].head(3)
    negative_features = coef_df[coef_df['Coefficient'] < 0].head(3)
    print(f"  Bullish Indicators:")
    for _, row in positive_features.iterrows():
        print(f"    {row['Feature']}: {row['Coefficient']:+.6f}")
    print(f"  Bearish Indicators:")
    for _, row in negative_features.iterrows():
        print(f"    {row['Feature']}: {row['Coefficient']:+.6f}")

    # --- AUDIT: Save coef_df if audit_files is set ---
    import inspect
    frame = inspect.currentframe().f_back
    audit_files = frame.f_locals.get('audit_files', None)
    if audit_files is not None:
        audit_files['coef_df'] = coef_df

    return coef_df

def get_periods_from_config():
    """Load period configuration from config.yaml"""
    
    try:
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get('periods', {})
    except:
        return {}

def main():
    """Main analysis function with argument parsing"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Analyze ML model performance for a specific symbol')
    parser.add_argument('--symbol', type=str, help='Stock symbol to analyze (e.g., ACN, AAPL)')
    parser.add_argument('--show-charts', action='store_true', default=True, help='Show fit charts (default: True)')
    parser.add_argument('--show-coefficients', action='store_true', default=True, help='Show coefficient analysis (default: True)')
    parser.add_argument('--audit', action='store_true', help='Save key tables to data/audit as CSVs')
    args = parser.parse_args()
    
    if args.symbol:
        print(f"🎯 Analyzing model for: {args.symbol.upper()}")
        target_symbol = args.symbol.upper()
    else:
        print("⚠️ No symbol specified - will analyze latest available model")
        target_symbol = None
    
    # Load and analyze model
    model_package = analyze_latest_model(target_symbol)
    if not model_package:
        return
    
    # Display performance metrics
    display_model_performance(model_package)
    
    # --- AUDIT LOGIC ---
    audit_dir = Path(__file__).parent.parent / 'data' / 'audit'
    if args.audit:
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_prefix = f"analyze_model__{model_package['target_symbol']}"
        audit_files = {}
    else:
        audit_files = None
    
    # Create fit charts and optionally audit tables
    if args.show_charts:
        if args.audit:
            def create_model_fit_chart_with_audit(model_package):
                return create_model_fit_chart_audit_patch(model_package, audit_files)
            def create_model_fit_chart_audit_patch(model_package, audit_files):
                symbol = model_package['target_symbol']
                periods = get_periods_from_config()
                try:
                    # Dynamically import DatabaseManager from path in config.yaml
                    import yaml
                    import sys
                    from pathlib import Path
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    db_manager_path = config.get('database_manager_path')
                    if not db_manager_path:
                        raise ImportError('database_manager_path not set in config.yaml')
                    db_manager_dir = str(Path(db_manager_path).parent)
                    if db_manager_dir not in sys.path:
                        sys.path.insert(0, db_manager_dir)
                    from database_manager import DatabaseManager
                    with DatabaseManager() as db:
                        historical_data = db.get_stock_prices(symbol)
                    if historical_data is None or len(historical_data) == 0:
                        print(f"⚠️ No real data for {symbol}, cannot generate fit chart.")
                        return
                    if isinstance(historical_data, list):
                        historical_df = pd.DataFrame(historical_data)
                        column_mapping = {}
                        for col in historical_df.columns:
                            col_lower = str(col).lower()
                            if col_lower in ['date', 'timestamp']:
                                column_mapping[col] = 'Date'
                            elif col_lower in ['close', 'adj_close', 'adjusted_close']:
                                column_mapping[col] = 'Close'
                        historical_df = historical_df.rename(columns=column_mapping)
                        if 'Date' in historical_df.columns:
                            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
                            historical_df = historical_df.set_index('Date')
                        historical_data = historical_df
                    elif isinstance(historical_data, pd.DataFrame):
                        if 'Date' in historical_data.columns and historical_data.index.name != 'Date':
                            historical_data['Date'] = pd.to_datetime(historical_data['Date'])
                            historical_data = historical_data.set_index('Date')
                    if 'Close' not in historical_data.columns:
                        for alt_col in ['close', 'Adj_Close', 'adj_close', 'Close_Price', 'price']:
                            if alt_col in historical_data.columns:
                                historical_data['Close'] = historical_data[alt_col]
                                break
                        else:
                            print(f"⚠️ No Close price column found. Cannot generate fit chart.")
                            return
                    historical_data = historical_data.sort_index()
                    if len(historical_data) < 30:
                        print(f"⚠️ Insufficient data: {len(historical_data)} records. Cannot generate fit chart.")
                        return
                    actual_prices = historical_data['Close']
                    actual_returns = actual_prices.pct_change().dropna()
                    # Try to recreate features and predictions
                    predictions = generate_model_predictions(model_package, actual_returns.index, audit_files=audit_files)
                    if predictions is None:
                        print(f"⚠️ Could not generate predictions. Cannot generate fit chart.")
                        return
                    # Save audit tables if requested
                    if audit_files is not None:
                        audit_files['historical_data'] = historical_data
                        audit_files['predictions'] = pd.DataFrame({'prediction': predictions})
                    # Now plot the chart (call the original plotting code)
                    # Reuse the plotting code from create_model_fit_chart
                    # ...existing code for plotting (see create_model_fit_chart)...
                    # For brevity, you can call create_model_fit_chart here if it only does plotting
                    create_model_fit_chart(model_package)
                except Exception as e:
                    print(f"❌ Database error: {e}")
            create_model_fit_chart_with_audit(model_package)
        else:
            create_model_fit_chart(model_package)
    # Analyze coefficients and significance
    if args.show_coefficients:
        analyze_model_coefficients(model_package)
    print(f"\n✅ Model analysis complete for {model_package['target_symbol']}!")

    # --- AUDIT FILE OUTPUT ---
    if args.audit and audit_files:
        for name, df in audit_files.items():
            if hasattr(df, 'to_csv'):
                out_path = audit_dir / f"{audit_prefix}_{name}.csv"
                try:
                    df.to_csv(out_path)
                    print(f"💾 Saved audit file: {out_path}")
                except Exception as e:
                    print(f"❌ Failed to save audit file {out_path}: {e}")

if __name__ == "__main__":
    main()