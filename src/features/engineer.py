def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """Add technical and sentiment features to price data."""
    df = data.copy()
    
    # Add technical features
    df['RSI'] = self.calculate_rsi(df)
    
    macd_data = self.calculate_macd(df)
    df['MACD'] = macd_data['MACD']
    df['MACD_Signal'] = macd_data['Signal']
    df['MACD_Histogram'] = macd_data['Histogram']
    
    bb_data = self.calculate_bollinger_bands(df)
    df['BB_Upper'] = bb_data['Upper']
    df['BB_Lower'] = bb_data['Lower']
    df['BB_Middle'] = bb_data['Middle']
    
    # Calculate momentum indicators
    df['ROC'] = df['Close'].pct_change(periods=10)  # 10-day Rate of Change
    df['MOM'] = df['Close'].diff(periods=10)        # 10-day Momentum
    
    # Add volatility features
    df['ATR'] = self.calculate_atr(df)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Add sentiment features
    sentiment = SentimentFeatures(self.config)
    for date in df.index:
        metrics = sentiment.get_daily_sentiment(self.symbol, date)
        for key, value in metrics.items():
            df.loc[date, key] = value
            
    return df