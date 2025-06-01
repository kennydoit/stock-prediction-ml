import requests
import logging
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class SentimentAPIClient:
    def __init__(self, api_key, base_url, max_retries=3, retry_delay=1.0):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = self._create_session()

    def _create_session(self):
        """Create a session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def get_sentiment(self, symbol):
        """Get sentiment data for a symbol"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            url = f"{self.base_url}/sentiment/{symbol}"
            logger.debug(f"Requesting sentiment data from: {url}")
            
            response = self.session.get(
                url,
                headers=headers,
                allow_redirects=False  # Prevent redirects
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {symbol}: {str(e)}")
            return None