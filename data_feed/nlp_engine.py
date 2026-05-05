"""NLP Sentiment Engine — multi-source financial sentiment aggregation with FinBERT."""

import logging

import feedparser
import pandas as pd
import numpy as np

from config import SENTIMENT_ROLLING_WINDOW

logger = logging.getLogger(__name__)

RSS_SOURCES = {
    "google_news": "https://news.google.com/rss/search?q={query}+finance&hl=en-US&gl=US&ceid=US:en",
    "yahoo_finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={query}&region=US&lang=en-US",
    "reddit_wsb": "https://www.reddit.com/r/wallstreetbets/search.rss?q={query}&sort=new&restrict_sr=on&t=week",
    "reddit_stocks": "https://www.reddit.com/r/stocks/search.rss?q={query}&sort=new&restrict_sr=on&t=week",
    "investing_com": "https://www.investing.com/rss/news_{query}.rss",
}

SOURCE_WEIGHTS = {
    "google_news": 0.35,
    "yahoo_finance": 0.25,
    "reddit_wsb": 0.15,
    "reddit_stocks": 0.15,
    "investing_com": 0.10,
}


class SentimentEngine:
    """Multi-source FinBERT sentiment analyser with lazy model loading."""

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

    def _fetch_single_source(self, source_name: str, query: str) -> list[str]:
        """Fetch headlines from a single RSS source."""
        url_template = RSS_SOURCES.get(source_name, "")
        if not url_template:
            return []
        url = url_template.format(query=query)
        try:
            feed = feedparser.parse(url)
            return [entry.title for entry in feed.entries[:5]]
        except Exception as exc:
            logger.debug("RSS fetch failed for %s/%s: %s", source_name, query, exc)
            return []

    def _score_headlines(self, headlines: list[str]) -> float:
        """Score a list of headlines using FinBERT."""
        self._load_model()
        if not self.nlp_pipeline or not headlines:
            return 0.0
        try:
            results = self.nlp_pipeline(headlines[:10])
            scores = []
            for res in results:
                if res['label'] == 'positive':
                    scores.append(res['score'])
                elif res['label'] == 'negative':
                    scores.append(-res['score'])
                else:
                    scores.append(0.0)
            return float(np.mean(scores)) if scores else 0.0
        except Exception as exc:
            logger.error("FinBERT scoring failed: %s", exc)
            return 0.0

    def fetch_rss_sentiment(self, query: str) -> float:
        """Fetch multi-source sentiment with weighted aggregation."""
        if query in self._cache:
            return self._cache[query]

        weighted_scores = []
        total_weight = 0.0

        for source_name, weight in SOURCE_WEIGHTS.items():
            headlines = self._fetch_single_source(source_name, query)
            if headlines:
                score = self._score_headlines(headlines)
                weighted_scores.append(score * weight)
                total_weight += weight

        if total_weight > 0:
            final_score = sum(weighted_scores) / total_weight
        else:
            final_score = 0.0

        self._cache[query] = final_score
        return final_score

    def fetch_source_breakdown(self, query: str) -> dict[str, float]:
        """Return per-source sentiment scores for transparency."""
        breakdown = {}
        for source_name in RSS_SOURCES:
            headlines = self._fetch_single_source(source_name, query)
            if headlines:
                breakdown[source_name] = self._score_headlines(headlines)
            else:
                breakdown[source_name] = 0.0
        return breakdown

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

    def apply_rolling_sentiment(
        self,
        df: pd.DataFrame,
        symbol: str,
        window_bars: int = SENTIMENT_ROLLING_WINDOW,
    ) -> pd.DataFrame:
        """Forward-fill sentiment across a rolling window to simulate persistent regime."""
        df = df.copy()
        if 'Sentiment_Score' not in df.columns:
            df['Sentiment_Score'] = 0.0
        current_sentiment = self.fetch_rss_sentiment(symbol)
        if len(df) > 0 and current_sentiment != 0.0:
            fill_start = max(0, len(df) - window_bars)
            df.iloc[fill_start:, df.columns.get_loc('Sentiment_Score')] = current_sentiment
        return df
