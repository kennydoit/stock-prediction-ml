
import pandas as pd
import pickle
import os

def predict_linear_regression(features_df, models_dir='../models'):
    '''
    Generate predictions using the trained linear regression model.

    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature data with same structure as training data
    models_dir : str
        Directory containing saved model artifacts

    Returns:
    --------
    pd.Series
        Predictions in original scale
    '''
    # Load model artifacts
    with open(os.path.join(models_dir, 'linear_regression_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(models_dir, 'lr_feature_scaler.pkl'), 'rb') as f:
        feature_scaler = pickle.load(f)

    with open(os.path.join(models_dir, 'lr_target_scaler.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)

    with open(os.path.join(models_dir, 'lr_selected_features.pkl'), 'rb') as f:
        selected_features = pickle.load(f)

    # Clean and prepare features
    features_clean = features_df.copy()

    # Handle missing values the same way as training
    features_clean = features_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Scale features
    features_scaled = pd.DataFrame(
        feature_scaler.transform(features_clean),
        index=features_clean.index,
        columns=features_clean.columns
    )

    # Select the same features used in training
    features_selected = features_scaled[selected_features]

    # Generate predictions
    predictions_scaled = model.predict(features_selected)

    # Inverse transform to original scale
    predictions = target_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).ravel()

    return pd.Series(predictions, index=features_df.index)
