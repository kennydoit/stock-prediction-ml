"""
Prediction script for forecasting MSFT stock based on peer analysis.
"""
from data_loader import StockDataLoader
from features import FeatureEngineer
from model import StockPredictor
import yaml
import datetime as dt

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize components
    data_loader = StockDataLoader(config['data_path'])
    feature_engineer = FeatureEngineer()
    predictor = StockPredictor.load_model('models/xgboost_msft.model')
    
    # Get current date
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=60)  # Get 60 days of historical data
    
    # Fetch latest data for MSFT and peers
    msft_data = data_loader.fetch_stock_data('MSFT', start_date.strftime('%Y-%m-%d'), 
                                           end_date.strftime('%Y-%m-%d'))
    
    # Engineer features
    features_df = feature_engineer.engineer_features(msft_data)
    
    # Make prediction
    latest_features = features_df.iloc[-1:]
    prediction = predictor.predict(latest_features)
    
    print(f"Predicted 5-day return for MSFT: {prediction[0]:.2%}")

if __name__ == "__main__":
    main()
