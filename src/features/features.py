import pandas as pd
import numpy as np
import logging

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._validate_config()

    def _validate_config(self):
        """Validate that required technical indicator configurations are present"""
        if 'features' not in self.config:
            raise ValueError("No 'features' section found in config")
        
        if 'technical' not in self.config['features']:
            raise ValueError("No 'technical' section found in features config")
        
        tech_config = self.config['features']['technical']
        
        # Define required indicators (only check if they exist when enabled)
        required_indicators = ['rsi', 'macd', 'bollinger', 'cci', 'stochastic', 'atr', 'obv', 'ichimoku', 'sma']
        
        for indicator in required_indicators:
            if indicator not in tech_config:
                self.logger.warning(f"No configuration found for {indicator}, skipping")
            elif tech_config[indicator].get('enabled', False):
                # Only validate parameters for enabled indicators
                self._validate_indicator_params(indicator, tech_config[indicator])

        # Validate lag configuration if enabled
        if self.config['features'].get('lags', {}).get('enabled', False):
            self._validate_lag_params()

    def _validate_indicator_params(self, indicator, params):
        """Validate parameters for a specific indicator"""
        required_params = {
            'rsi': ['period'],
            'macd': ['fast_period', 'slow_period', 'signal_period'],
            'bollinger': ['window', 'std'],
            'cci': ['period'],
            'stochastic': ['k_period', 'd_period'],
            'atr': ['period'],
            'obv': ['window'],
            'ichimoku': ['conversion_period', 'base_period'],
            'sma': []  # SMA can work with default periods or custom ones
        }
        
        if indicator in required_params:
            missing = [param for param in required_params[indicator] if param not in params]
            if missing:
                raise ValueError(f"Missing parameters for {indicator}: {missing}")

    def _validate_lag_params(self):
        """Validate lag feature parameters"""
        lag_config = self.config['features']['lags']
        required_params = ['price_columns', 'lag_periods']
        
        missing = [param for param in required_params if param not in lag_config]
        if missing:
            raise ValueError(f"Missing lag parameters: {missing}")

    def engineer_features(self, data):
        """Engineer technical features based on config settings"""
        features = data.copy()
        
        # Add technical indicators
        features = self._add_technical_indicators(features)
        
        # Add lagged features
        features = self._add_lagged_features(features)
        
        return features

    def _add_technical_indicators(self, data):
        """Add technical indicators based on config"""
        features = data.copy()
        tech_config = self.config['features']['technical']
        
        # Only calculate enabled indicators
        for indicator, settings in tech_config.items():
            if settings.get('enabled', False):
                try:
                    if indicator == 'rsi':
                        features = self._add_rsi(features, settings['period'])
                    elif indicator == 'macd':
                        features = self._add_macd(features, settings['fast_period'], 
                                                settings['slow_period'], settings['signal_period'])
                    elif indicator == 'bollinger':
                        features = self._add_bollinger_bands(features, settings['window'], settings['std'])
                    elif indicator == 'cci':
                        features = self._add_cci(features, settings['period'])
                    elif indicator == 'stochastic':
                        features = self._add_stochastic(features, settings['k_period'], settings['d_period'])
                    elif indicator == 'atr':
                        features = self._add_atr(features, settings['period'])
                    elif indicator == 'obv':
                        features = self._add_obv(features, settings['window'])
                    elif indicator == 'ichimoku':
                        features = self._add_ichimoku(features, settings['conversion_period'], 
                                                    settings['base_period'])
                    elif indicator == 'sma':
                        features = self._add_sma(features, settings.get('periods', [5, 10, 20, 50]))
                        
                    self.logger.info(f"Added {indicator} features")
                except Exception as e:
                    self.logger.error(f"Error calculating {indicator}: {e}")
        
        return features

    def _add_sma(self, data, periods=[5, 10, 20, 50]):
        """Add Simple Moving Average indicators"""
        for period in periods:
            # SMA for Close price
            data[f'Close_sma_{period}'] = data['Close'].rolling(window=period).mean()
            # SMA for Volume
            data[f'Volume_sma_{period}'] = data['Volume'].rolling(window=period).mean()
            # SMA for High
            data[f'High_sma_{period}'] = data['High'].rolling(window=period).mean()
            # SMA for Low
            data[f'Low_sma_{period}'] = data['Low'].rolling(window=period).mean()
        return data

    def _add_lagged_features(self, data):
        """Add lagged price features based on config"""
        lag_config = self.config['features'].get('lags', {})
        
        if not lag_config.get('enabled', False):
            return data
            
        features = data.copy()
        
        try:
            # Get configuration
            price_columns = lag_config['price_columns']
            lag_periods = lag_config['lag_periods']
            
            # Add basic lag features
            for column in price_columns:
                if column in features.columns:
                    for lag in lag_periods:
                        lag_col_name = f"{column}_lag_{lag}"
                        features[lag_col_name] = features[column].shift(lag)
                        
            # Add rolling window features if enabled
            rolling_config = lag_config.get('rolling_features', {})
            if rolling_config.get('enabled', False):
                features = self._add_rolling_features(features, rolling_config, price_columns)
                
            self.logger.info(f"Added lagged features for {len(price_columns)} columns with {len(lag_periods)} lag periods")
            
        except Exception as e:
            self.logger.error(f"Error adding lagged features: {e}")
            
        return features

    def _add_rolling_features(self, data, rolling_config, price_columns):
        """Add rolling window statistical features"""
        features = data.copy()
        
        windows = rolling_config.get('windows', [5, 10, 20])
        functions = rolling_config.get('functions', ['mean', 'std'])
        
        for column in price_columns:
            if column in features.columns:
                for window in windows:
                    for func in functions:
                        col_name = f"{column}_rolling_{window}_{func}"
                        
                        if func == 'mean':
                            features[col_name] = features[column].rolling(window=window).mean()
                        elif func == 'std':
                            features[col_name] = features[column].rolling(window=window).std()
                        elif func == 'min':
                            features[col_name] = features[column].rolling(window=window).min()
                        elif func == 'max':
                            features[col_name] = features[column].rolling(window=window).max()
                            
        return features

    def _add_rsi(self, data, period=14):
        """Add RSI indicator"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data

    def _add_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """Add MACD indicator"""
        exp1 = data['Close'].ewm(span=fast_period).mean()
        exp2 = data['Close'].ewm(span=slow_period).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=signal_period).mean()
        data['Histogram'] = data['MACD'] - data['Signal']
        return data

    def _add_bollinger_bands(self, data, window=20, std=2):
        """Add Bollinger Bands"""
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        data['BB_Upper'] = rolling_mean + (rolling_std * std)
        data['BB_Lower'] = rolling_mean - (rolling_std * std)
        data['BB_Middle'] = rolling_mean
        return data

    def _add_cci(self, data, period=20):
        """Add CCI indicator"""
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        data['CCI'] = (tp - sma) / (0.015 * mad)
        return data

    def _add_stochastic(self, data, k_period=14, d_period=3):
        """Add Stochastic oscillator"""
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        data['STOCH_K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        data['STOCH_D'] = data['STOCH_K'].rolling(window=d_period).mean()
        return data

    def _add_atr(self, data, period=14):
        """Add Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        data['ATR'] = true_range.rolling(window=period).mean()
        return data

    def _add_obv(self, data, window=20):
        """Add On Balance Volume"""
        obv = [0]
        for i in range(1, len(data['Close'])):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        data['OBV'] = obv
        data['OBV_SMA'] = pd.Series(obv).rolling(window=window).mean()
        return data

    def _add_ichimoku(self, data, conversion_period=9, base_period=26):
        """Add Ichimoku Cloud indicators"""
        # Conversion Line
        conv_high = data['High'].rolling(window=conversion_period).max()
        conv_low = data['Low'].rolling(window=conversion_period).min()
        data['ICHIMOKU_CONV'] = (conv_high + conv_low) / 2
        
        # Base Line
        base_high = data['High'].rolling(window=base_period).max()
        base_low = data['Low'].rolling(window=base_period).min()
        data['ICHIMOKU_BASE'] = (base_high + base_low) / 2
        
        return data
