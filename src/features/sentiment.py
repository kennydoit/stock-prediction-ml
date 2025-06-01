import pandas as pd
from typing import Dict, List
from newsapi import NewsApiClient

class SentimentFeatures:
    def __init__(self, config: Dict):
        self.config = config
        self.newsapi = NewsApiClient(api_key=config['api_keys']['newsapi'])
        self.lookback = config['sentiment']['lookback_days']
        
    def get_daily_sentiment(self, symbol: str, date: pd.Timestamp) -> Dict:
        """Get sentiment metrics for a single day."""
        start = date - pd.Timedelta(days=self.lookback)
        
        # Get news articles
        articles = self.newsapi.get_everything(
            q=symbol,
            from_param=start.strftime('%Y-%m-%d'),
            to=date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='relevancy'
        )
        
        return {
            'daily_sentiment_score': self._calculate_sentiment(articles),
            'news_volume': len(articles['articles'])
        }