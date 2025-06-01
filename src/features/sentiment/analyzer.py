import os
import logging
import requests
import nltk
from datetime import datetime, timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzes news sentiment for stocks."""
    
    def __init__(self, config):
        """Initialize sentiment analyzer with NLTK and News API"""
        if 'sentiment' not in config:
            raise ValueError("Missing sentiment configuration")
        
        # Load NLTK components
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.nltk_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize NLTK: {str(e)}")
            raise
        
        # Store configuration
        sentiment_config = config['sentiment']
        self.api_key = sentiment_config['api_key']
        self.endpoint = sentiment_config['api_endpoint']
        self.cache_dir = sentiment_config['cache_dir']
        self.lookback_days = int(sentiment_config.get('news_days_lookback', 7))
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Initialized SentimentAnalyzer with endpoint: {self.endpoint}")

    def get_raw_sentiment(self, symbol, start_date=None, end_date=None, page=1, page_size=100):
        """Get news articles and analyze sentiment
        
        Args:
            symbol (str): Stock symbol to search for
            start_date (datetime, optional): Start date for articles
            end_date (datetime, optional): End date for articles
            page (int, optional): Page number for pagination. Defaults to 1.
            page_size (int, optional): Number of articles per page. Defaults to 100.
        """
        try:
            # Use provided dates or default to lookback period
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=self.lookback_days)
                
            url = f"{self.endpoint}/everything"
            params = {
                'q': f'"{symbol}" AND (stock OR market OR trading)',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'page': page,
                'pageSize': page_size,
                'apiKey': self.api_key
            }
            
            logger.debug(f"Fetching news for {symbol} from {start_date} to {end_date} (page {page})")
            response = requests.get(url, params=params)
            response.raise_for_status()
            news_data = response.json()
            
            if news_data.get('status') != 'ok':
                raise ValueError(f"API Error: {news_data.get('message', 'Unknown error')}")
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            return None

    def get_sentiment_features(self, symbol, end_date, start_date=None):
        """Process news articles and return sentiment features"""
        try:
            # Get all articles using pagination
            all_articles = []
            page = 1
            while True:
                news_data = self.get_raw_sentiment(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    page=page
                )
                
                if not news_data or 'articles' not in news_data:
                    break
                    
                articles = news_data['articles']
                if not articles:
                    break
                    
                all_articles.extend(articles)
                total_results = news_data.get('totalResults', 0)
                
                # Stop if we've got all articles or hit the API limit
                if len(all_articles) >= total_results or len(articles) < 100:
                    break
                    
                page += 1
            
            if not all_articles:
                return self._get_default_features()
            
            # Analyze sentiment for each article
            sentiments = []
            for article in all_articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    sentiment = self.nltk_analyzer.polarity_scores(text)
                    sentiments.append(sentiment)
            
            if not sentiments:
                return self._get_default_features()
            
            # Calculate aggregate sentiment
            avg_sentiment = pd.DataFrame(sentiments).mean()
            return {
                'sentiment_score': float(avg_sentiment['compound']),
                'sentiment_magnitude': float(avg_sentiment['pos'] + avg_sentiment['neg']),
                'article_count': len(all_articles),
                'articles': all_articles  # Include articles for debugging
            }
            
        except Exception as e:
            logger.error(f"Error processing sentiment: {str(e)}")
            return self._get_default_features()
    
    def _get_default_features(self):
        """Return default features when no data is available"""
        return {
            'sentiment_score': 0.0,
            'sentiment_magnitude': 0.0,
            'article_count': 0
        }