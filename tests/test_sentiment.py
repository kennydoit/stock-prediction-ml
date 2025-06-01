import pytest
from src.features.sentiment.analyzer import SentimentAnalyzer
from datetime import datetime

def test_sentiment_analyzer(mock_config):
    analyzer = SentimentAnalyzer(mock_config)
    features = analyzer.get_sentiment_features('MSFT', datetime.now())
    
    assert isinstance(features, dict)
    assert all(k in features for k in ['sentiment_score', 'article_count'])