"""
News analysis module for sentiment analysis and news feature engineering
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
import logging
from typing import List, Dict
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """
    Analyzer for extracting features from news data
    """
    
    def __init__(self):
        """Initialize the news analyzer"""
        logger.info("NewsAnalyzer initialized")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a text using TextBlob
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with polarity and subjectivity scores
        """
        if not text or pd.isna(text):
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
                'subjectivity': blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            }
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def extract_news_features(self, news_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Extract features from news data for a specific symbol
        
        Parameters:
        -----------
        news_df : pd.DataFrame
            DataFrame with news articles
        symbol : str
            Stock symbol to analyze
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with news features aggregated by date
        """
        if news_df.empty:
            logger.warning(f"No news data provided for {symbol}")
            return pd.DataFrame()
        
        try:
            # Filter news for the specific symbol
            symbol_news = news_df[news_df['symbols'].apply(
                lambda x: symbol in x if isinstance(x, list) else symbol in str(x)
            )].copy()
            
            if symbol_news.empty:
                logger.warning(f"No news found for symbol {symbol}")
                return pd.DataFrame()
            
            # Analyze sentiment for headlines and summaries
            logger.info(f"Analyzing sentiment for {len(symbol_news)} articles for {symbol}")
            
            # Headline sentiment
            headline_sentiment = symbol_news['headline'].apply(self.analyze_sentiment)
            symbol_news['headline_polarity'] = [s['polarity'] for s in headline_sentiment]
            symbol_news['headline_subjectivity'] = [s['subjectivity'] for s in headline_sentiment]
            
            # Summary sentiment
            summary_sentiment = symbol_news['summary'].apply(self.analyze_sentiment)
            symbol_news['summary_polarity'] = [s['polarity'] for s in summary_sentiment]
            symbol_news['summary_subjectivity'] = [s['subjectivity'] for s in summary_sentiment]
            
            # Convert timestamp index to date for daily aggregation
            symbol_news['date'] = symbol_news.index.date
            
            # Aggregate news features by date
            daily_features = symbol_news.groupby('date').agg({
                'headline_polarity': ['mean', 'std', 'min', 'max'],
                'headline_subjectivity': ['mean', 'std'],
                'summary_polarity': ['mean', 'std', 'min', 'max'],
                'summary_subjectivity': ['mean', 'std'],
                'article_id': 'count'  # Number of articles per day
            }).round(4)
            
            # Flatten column names
            daily_features.columns = [f"{symbol}_news_{'_'.join(col)}" for col in daily_features.columns]
            
            # Rename article count column
            article_count_col = [col for col in daily_features.columns if 'article_id_count' in col][0]
            daily_features = daily_features.rename(columns={article_count_col: f'{symbol}_news_article_count'})
            
            # Fill NaN values (e.g., when std is 0 for single articles)
            daily_features = daily_features.fillna(0)
            
            # Convert date index to datetime
            daily_features.index = pd.to_datetime(daily_features.index)
            
            logger.info(f"Generated {len(daily_features)} days of news features for {symbol}")
            return daily_features
            
        except Exception as e:
            logger.error(f"Error extracting news features for {symbol}: {e}")
            return pd.DataFrame()
    
    def create_sentiment_signals(self, news_features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Create trading signals based on news sentiment
        
        Parameters:
        -----------
        news_features : pd.DataFrame
            DataFrame with news features
        symbol : str
            Stock symbol
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with sentiment-based signals
        """
        if news_features.empty:
            return pd.DataFrame()
        
        try:
            signals = pd.DataFrame(index=news_features.index)
            
            # Get polarity columns
            headline_polarity_col = f'{symbol}_news_headline_polarity_mean'
            summary_polarity_col = f'{symbol}_news_summary_polarity_mean'
            article_count_col = f'{symbol}_news_article_count'
            
            if headline_polarity_col in news_features.columns:
                # Overall sentiment signal (weighted by article count)
                sentiment_score = (
                    news_features[headline_polarity_col] * 0.4 +
                    news_features[summary_polarity_col] * 0.6
                ) * np.log1p(news_features[article_count_col])  # Weight by log of article count
                
                signals[f'{symbol}_sentiment_signal'] = sentiment_score
                
                # Sentiment categories
                signals[f'{symbol}_sentiment_bullish'] = (sentiment_score > 0.1).astype(int)
                signals[f'{symbol}_sentiment_bearish'] = (sentiment_score < -0.1).astype(int)
                signals[f'{symbol}_sentiment_neutral'] = (
                    (sentiment_score >= -0.1) & (sentiment_score <= 0.1)
                ).astype(int)
                
                # News momentum (change in sentiment)
                signals[f'{symbol}_sentiment_momentum'] = sentiment_score.diff()
                
            logger.info(f"Created sentiment signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error creating sentiment signals for {symbol}: {e}")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    from alpaca_data_loader import AlpacaDataLoader
    
    # Test news analysis
    print("Testing news analysis...")
    
    loader = AlpacaDataLoader('../config.yaml')
    analyzer = NewsAnalyzer()
    
    # Get news for AAPL
    news = loader.fetch_news_for_symbol('AAPL', days_back=7, limit=20)
    print(f"Fetched {len(news)} news articles")
    
    if not news.empty:
        # Extract features
        features = analyzer.extract_news_features(news, 'AAPL')
        print(f"Generated features for {len(features)} days")
        
        if not features.empty:
            print("\nNews features columns:")
            print(features.columns.tolist())
            
            # Create signals
            signals = analyzer.create_sentiment_signals(features, 'AAPL')
            print(f"Generated signals for {len(signals)} days")