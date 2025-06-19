#!/usr/bin/env python3
"""
Linear regression model for stock prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

def train_linear_regression_model(X, y, fit_intercept: bool = True):
    """Train linear regression model (no scaling of features or target)
    X: DataFrame of features (already selected and ordered)
    y: Series or array of target values (already aligned)
    """
    
    logger.info("Training linear regression model (no scaling)")
    
    if X.empty or y.empty:
        logger.error("Empty features or target provided")
        return None

    feature_columns = list(X.columns)
    logger.info(f"Training with {X.shape[0]} samples and {X.shape[1]} features")
    logger.info(f"Feature columns: {feature_columns[:10]}...")  # Show first 10

    # Split data
    test_size = 0.2  # Or pass as argument if needed
    n_test = int(len(X) * test_size)
    n_train = len(X) - n_test
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    # Train model (no scaling)
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_features': X_train.shape[1]
    }

    logger.info(f"Model trained successfully")
    logger.info(f"Train R²: {metrics['train_r2']:.4f}, Test R²: {metrics['test_r2']:.4f}")
    logger.info(f"Test MSE: {metrics['test_mse']:.6f}")

    return model, None, feature_columns, metrics
