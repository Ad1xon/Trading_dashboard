"""NLP Sentiment Engine — FinBERT-based financial news analysis."""

import logging

import feedparser
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SentimentEngine:
    """Singleton FinBERT sentiment analyser with lazy model loading."""

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
        """Lazily load the FinBERT pipeline on first use."""
        if self.nlp_pipeline is None:
            try:
                from transformers import pipeline
                self.nlp_pipeline = pipeline(
                    "sentiment-analysis", model="ProsusAI/finbert",
                )
            except Exception as exc:
                logger.error("Failed to load FinBERT: %s", exc)
                self.nlp_pipeline = False

    def fetch_rss_sentiment(self, query: str) -> float:
        """Fetch Google News RSS for *query* and return aggregated sentiment."""
        if query in self._cache:
            return self._cache[query]

        self._load_model()
        if not self.nlp_pipeline:
            return 0.0

        url = (
            f"https://news.google.com/rss/search?"
            f"q={query}+finance&hl=en-US&gl=US&ceid=US:en"
        )
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
        except Exception as exc:
            logger.error("RSS fetch failed for %s: %s", query, exc)
            return 0.0

    def apply_sentiment_to_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Apply the current sentiment score to the last bar of *df*."""
        df = df.copy()
        if 'Sentiment_Score' not in df.columns:
            df['Sentiment_Score'] = 0.0

        current_sentiment = self.fetch_rss_sentiment(symbol)
        if len(df) > 0:
            df.iloc[-1, df.columns.get_loc('Sentiment_Score')] = current_sentiment
        return df
