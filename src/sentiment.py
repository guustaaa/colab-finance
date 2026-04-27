"""
sentiment.py — Real-time sentiment scanner from free RSS news feeds.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) to score
financial news headlines. VADER is specifically chosen because:
- It handles financial/social media language well
- No API key required (fully local)
- Fast enough for real-time use

Keyword filtering ensures we only score news relevant to our traded pairs
and known market-moving entities (Fed officials, political figures, etc.)
"""
import feedparser
import logging
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import SENTIMENT_KEYWORDS, RSS_FEEDS, CURRENCY_NEWS_KEYWORDS

logger = logging.getLogger("sentiment")


class SentimentScanner:
    """
    Scans free financial RSS feeds for market-relevant news and
    computes sentiment scores tailored to specific currencies.
    """

    def __init__(self, feeds: list = None):
        self.analyzer = SentimentIntensityAnalyzer()
        self.feeds = feeds or RSS_FEEDS
        self.currency_keywords = CURRENCY_NEWS_KEYWORDS

    def scan_all_feeds(self) -> list:
        """
        Fetch all RSS feeds and extract relevant, scored headlines.
        """
        results = []
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:  # Cap at 20 per feed
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title} {summary}".lower()

                    # Find which currencies this text is talking about
                    mentioned_currencies = []
                    for cur, kws in self.currency_keywords.items():
                        if any(kw in text for kw in kws):
                            mentioned_currencies.append(cur)

                    if not mentioned_currencies:
                        continue

                    score = self.analyzer.polarity_scores(title)

                    results.append({
                        "title": title,
                        "source": feed_url.split("/")[2],
                        "compound": score["compound"],
                        "currencies": mentioned_currencies,
                    })
            except Exception as e:
                logger.warning(f"RSS feed error ({feed_url}): {e}")

        logger.info(f"Sentiment scan complete: {len(results)} relevant articles found")
        return results

    def get_currency_sentiment(self, currency: str, articles: list) -> float:
        """Calculate the weighted sentiment score specifically for one currency."""
        relevant = [a for a in articles if currency in a["currencies"]]
        if not relevant:
            return 0.0

        scores = [a["compound"] for a in relevant]
        weights = [abs(s) + 0.01 for s in scores]

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_pair_sentiment(self, instrument: str, articles: list = None) -> float:
        """
        Get the sentiment diff for a specific FX pair.
        E.g., EUR_USD -> Sentiment(EUR) - Sentiment(USD)
        Returns a value between -2 (bearish base, bullish quote) and +2.
        """
        if articles is None:
            articles = self.scan_all_feeds()
            
        if not articles or "_" not in instrument:
            return 0.0

        base, quote = instrument.split("_")
        base_sent = self.get_currency_sentiment(base, articles)
        quote_sent = self.get_currency_sentiment(quote, articles)

        # If base is good and quote is bad, pair goes UP
        pair_diff = base_sent - quote_sent
        logger.debug(f"[{instrument}] Sentiment -> Base({base}): {base_sent:.2f} | Quote({quote}): {quote_sent:.2f} | Diff: {pair_diff:.2f}")
        
        return pair_diff

    def get_composite_score(self) -> float:
        """Legacy global score (kept for backwards compatibility if needed)."""
        articles = self.scan_all_feeds()
        if not articles:
            return 0.0
        scores = [a["compound"] for a in articles]
        return sum(scores) / len(scores)
