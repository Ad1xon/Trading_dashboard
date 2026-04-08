"""NLP Sentiment Engine using FinBERT for financial news analysis."""

import feedparser
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SentimentEngine:
    """Singleton FinBERT sentiment analyzer with lazy loading."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.nlp_pipeline = None
        self._cache = {}
        self._initialized = True

    def _load_model(self):
        if self.nlp_pipeline is None:
            try:
                from transformers import pipeline
                self.nlp_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            except Exception as e:
                logger.error(f"Failed to load FinBERT: {e}")
                self.nlp_pipeline = False

    def fetch_rss_sentiment(self, query: str) -> float:
        """Fetch RSS news for a query and return aggregated sentiment score (-1.0 to 1.0)."""
        if query in self._cache:
            return self._cache[query]

        self._load_model()
        if not self.nlp_pipeline:
            return 0.0

        url = f"https://news.google.com/rss/search?q={query}+finance&hl=en-US&gl=US&ceid=US:en"
        try:
            feed = feedparser.parse(url)
            headlines = [entry.title for entry in feed.entries[:5]]
            if not headlines:
                return 0.0

            results = self.nlp_pipeline(headlines)
            scores = []
            for res in results:
                if res['label'] == 'positive':
                    scores.append(res['score'])
                elif res['label'] == 'negative':
                    scores.append(-res['score'])
                else:
                    scores.append(0.0)

            final_score = float(np.mean(scores))
            self._cache[query] = final_score
            return final_score
        except Exception as e:
            logger.error(f"RSS fetch failed for {query}: {e}")
            return 0.0

    def apply_sentiment_to_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Vectorized application of daily sentiment to a Range Bar DataFrame."""
        df = df.copy()
        if 'Sentiment_Score' not in df.columns:
            df['Sentiment_Score'] = 0.0

        current_sentiment = self.fetch_rss_sentiment(symbol)
        if len(df) > 0:
            df.iloc[-1, df.columns.get_loc('Sentiment_Score')] = current_sentiment
        return df