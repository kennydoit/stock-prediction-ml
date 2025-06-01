"""
Model training and evaluation module.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Tuple, Dict
import joblib
import os

class StockPredictor:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        self.scaler = StandardScaler()
        
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform hyperparameter tuning using RandomizedSearchCV."""
        # Define parameter space
        param_space = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 150, 200],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
        }
        
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(objective='reg:squarederror'),
            param_distributions=param_space,
            n_iter=250,  # Number of parameter settings sampled
            cv=5,       # 5-fold cross-validation
            scoring='neg_mean_squared_error',
            n_jobs=-1,  # Use all available cores
            verbose=2,
            random_state=42
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform random search
        random_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            **random_search.best_params_
        )
        
        # Return tuning results
        return {
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,  # Convert back to MSE
            'cv_results': random_search.cv_results_
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, tune: bool = False) -> Dict:
        """Train the model and return performance metrics."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Perform hyperparameter tuning if requested
        if tune:
            tuning_results = self.tune_hyperparameters(X_train, y_train)
            logger.info(f"Best parameters found: {tuning_results['best_params']}")
            logger.info(f"Best cross-validation MSE: {tuning_results['best_score']:.4f}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'train_r2': train_score,
            'test_r2': test_score
        }

    def save_model(self, path: str):
        """Save the trained model and scaler."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
    
    @classmethod
    def load_model(cls, path: str):
        """Load a trained model."""
        components = joblib.load(path)
        instance = cls()
        instance.model = components['model']
        instance.scaler = components['scaler']
        return instance
