"""Technical analysis indicators for feature engineering."""
import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self, config):
        self.params = config.get('feature_parameters', {})
        self.rsi_periods = self.params.get('rsi_periods', 14)
        self.macd_fast = self.params.get('macd_fast', 12)
        self.macd_slow = self.params.get('macd_slow', 26)
        self.macd_signal = self.params.get('macd_signal', 9)
        
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD."""
        exp1 = data['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=self.macd_signal, adjust=False).mean()
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal_line,
            'Histogram': macd - signal_line
        })
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands for price data.
        
        Args:
            data: DataFrame with OHLC data
            window: Rolling window period (default: 20)
            num_std: Number of standard deviations (default: 2)
            
        Returns:
            DataFrame with Upper, Lower, and Middle bands
        """
        # Calculate middle band (simple moving average)
        middle_band = data['Close'].rolling(window=window).mean()
        
        # Calculate standard deviation
        rolling_std = data['Close'].rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        
        return pd.DataFrame({
            'Upper': upper_band,
            'Lower': lower_band,
            'Middle': middle_band
        })
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataset."""
        df = data.copy()
        
        # Add RSI
        df['RSI'] = self.calculate_rsi(df)
        
        # Add MACD components
        macd_data = self.calculate_macd(df)
        df['MACD'] = macd_data['MACD']
        df['Signal'] = macd_data['Signal']
        df['Histogram'] = macd_data['Histogram']
        
        # Add Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df)
        df['BB_Upper'] = bb_data['Upper']
        df['BB_Lower'] = bb_data['Lower']
        df['BB_Middle'] = bb_data['Middle']
        
        return df