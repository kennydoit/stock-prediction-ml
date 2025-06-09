#!/usr/bin/env python3
"""
Make predictions using trained models
"""
import sys
from pathlib import Path
import yaml
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import logging

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'database'))

from database_manager import DatabaseManager
from features.technical_indicators import calculate_technical_indicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_latest_model(symbol: str = None):
    """Load the latest trained model"""
    
    models_dir = Path(__file__).parent.parent / 'models'
    
    if not models_dir.exists():
        print("❌ No models directory found. Train a model first!")
        return None
    
    # Find model files
    if symbol:
        model_files = list(models_dir.glob(f'*{symbol}*.pkl'))
    else:
        model_files = list(models_dir.glob('*.pkl'))
    
    if not model_files:
        print("❌ No trained models found. Train a model first!")
        return None
    
    # Get the latest model (by timestamp in filename)
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📂 Loading model: {latest_model.name}")
    
    try:
        model_package = joblib.load(latest_model)
        
        print(f"✅ Model loaded successfully!")
        print(f"  Target symbol: {model_package['target_symbol']}")
        print(f"  Trained: {model_package['trained_at']}")
        print(f"  Features: {model_package['feature_count']}")
        print(f"  Test R²: {model_package['metrics']['test_r2']:.4f}")
        
        return model_package
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def make_single_prediction(symbol: str, model_package: dict = None):
    """Make prediction for a single symbol"""
    
    if model_package is None:
        model_package = load_latest_model(symbol)
        if model_package is None:
            return None
    
    print(f"\n🔮 Making prediction for {symbol}")
    print("="*40)
    
    # Load config
    config = model_package['config']
    
    # Generate latest features
    print("📊 Generating latest features...")
    features_df = calculate_technical_indicators(symbol, config)
    
    if features_df.empty:
        print(f"❌ No features available for {symbol}")
        return None
    
    print(f"✅ Generated features: {len(features_df)} records")
    print(f"Latest data: {features_df.index[-1]}")
    
    # Get the latest feature values
    latest_features = features_df.iloc[-1:][model_package['feature_names']]
    
    # Check for missing features
    missing_features = set(model_package['feature_names']) - set(latest_features.columns)
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
        # Fill missing with 0 or handle appropriately
        for feature in missing_features:
            latest_features[feature] = 0
    
    # Scale features
    latest_features_scaled = model_package['scaler'].transform(latest_features)
    
    # Make prediction
    prediction = model_package['model'].predict(latest_features_scaled)[0]
    
    # Get latest price for context
    db_manager = DatabaseManager()
    with db_manager:
        price_data = db_manager.get_stock_prices(symbol)
        if not price_data.empty:
            latest_price = price_data['close'].iloc[-1]
            price_date = price_data.index[-1]
        else:
            latest_price = None
            price_date = None
    
    # Create prediction result
    result = {
        'symbol': symbol,
        'prediction_date': datetime.now(),
        'latest_price': latest_price,
        'price_date': price_date,
        'predicted_return': prediction,
        'model_used': model_package['target_symbol'],
        'confidence_r2': model_package['metrics']['test_r2']
    }
    
    # Display results
    print(f"\n📈 Prediction Results:")
    print(f"  Symbol: {symbol}")
    print(f"  Latest Price: ${latest_price:.2f} ({price_date})" if latest_price else "  Latest Price: N/A")
    print(f"  Predicted Return: {prediction:.4f} ({prediction*100:.2f}%)")
    
    if latest_price:
        predicted_price = latest_price * (1 + prediction)
        print(f"  Predicted Price: ${predicted_price:.2f}")
        
        if prediction > 0:
            print(f"  📈 Bullish signal (+{prediction*100:.2f}%)")
        else:
            print(f"  📉 Bearish signal ({prediction*100:.2f}%)")
    
    print(f"  Model Confidence: {model_package['metrics']['test_r2']:.4f} R²")
    
    return result

def make_portfolio_predictions(top_n: int = 10):
    """Make predictions for multiple symbols in database"""
    
    print(f"\n📊 Portfolio Predictions (Top {top_n})")
    print("="*50)
    
    # Load model
    model_package = load_latest_model()
    if model_package is None:
        return None
    
    # Get symbols from database
    db_manager = DatabaseManager()
    with db_manager:
        symbols_df = db_manager.get_symbols()
    
    print(f"Making predictions for {len(symbols_df)} symbols...")
    
    predictions = []
    
    for i, row in symbols_df.iterrows():
        symbol = row['symbol']
        
        try:
            result = make_single_prediction(symbol, model_package)
            if result:
                predictions.append(result)
                print(f"  ✅ {symbol}: {result['predicted_return']*100:.2f}%")
            else:
                print(f"  ❌ {symbol}: Failed")
                
        except Exception as e:
            print(f"  ❌ {symbol}: Error - {e}")
    
    if not predictions:
        print("❌ No successful predictions")
        return None
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # Sort by predicted return
    predictions_df = predictions_df.sort_values('predicted_return', ascending=False)
    
    print(f"\n🏆 Top {top_n} Predictions:")
    print("="*50)
    
    for i, row in predictions_df.head(top_n).iterrows():
        symbol = row['symbol']
        pred_return = row['predicted_return']
        latest_price = row['latest_price']
        
        print(f"{i+1:2d}. {symbol:6s} | {pred_return*100:+6.2f}% | ${latest_price:7.2f}")
    
    # Save predictions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions_file = Path(__file__).parent.parent / f'predictions_{timestamp}.csv'
    predictions_df.to_csv(predictions_file, index=False)
    
    print(f"\n💾 Predictions saved to: {predictions_file}")
    
    return predictions_df

def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Make stock predictions')
    parser.add_argument('--symbol', '-s', type=str, help='Predict single symbol')
    parser.add_argument('--portfolio', '-p', action='store_true', help='Predict portfolio')
    parser.add_argument('--top', '-t', type=int, default=10, help='Top N predictions to show')
    
    args = parser.parse_args()
    
    if args.symbol:
        # Single symbol prediction
        result = make_single_prediction(args.symbol.upper())
        return result
        
    elif args.portfolio:
        # Portfolio predictions
        results = make_portfolio_predictions(args.top)
        return results
        
    else:
        # Interactive mode
        print("🔮 Stock Prediction System")
        print("="*30)
        print("1. Single symbol prediction")
        print("2. Portfolio predictions")
        
        choice = input("\nChoose option (1 or 2): ").strip()
        
        if choice == "1":
            symbol = input("Enter symbol: ").strip().upper()
            return make_single_prediction(symbol)
            
        elif choice == "2":
            top_n = input("Top N predictions (default 10): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 10
            return make_portfolio_predictions(top_n)
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()