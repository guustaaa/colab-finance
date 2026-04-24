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

from src.config import SENTIMENT_KEYWORDS, RSS_FEEDS

logger = logging.getLogger("sentiment")


class SentimentScanner:
    """
    Scans free financial RSS feeds for market-relevant news and
    computes a composite sentiment score.
    """

    def __init__(self, keywords: list = None, feeds: list = None):
        self.analyzer = SentimentIntensityAnalyzer()
        self.keywords = keywords or SENTIMENT_KEYWORDS
        self.feeds = feeds or RSS_FEEDS

    def scan_all_feeds(self) -> list:
        """
        Fetch all RSS feeds and extract relevant, scored headlines.

        Returns list of dicts with: title, source, compound_score, keywords_found
        """
        results = []
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:  # Cap at 20 per feed to avoid slowdown
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title} {summary}".lower()

                    # Only score if relevant keywords are present
                    matched = [kw for kw in self.keywords if kw in text]
                    if not matched:
                        continue

                    score = self.analyzer.polarity_scores(title)

                    results.append({
                        "title": title,
                        "source": feed_url.split("/")[2],  # domain name
                        "compound": score["compound"],
                        "pos": score["pos"],
                        "neg": score["neg"],
                        "keywords": matched,
                        "scanned_at": datetime.now().isoformat(),
                    })
            except Exception as e:
                logger.warning(f"RSS feed error ({feed_url}): {e}")

        logger.info(f"Sentiment scan complete: {len(results)} relevant articles found")
        return results

    def get_composite_score(self) -> float:
        """
        Get a single composite sentiment score for the market.

        Returns a value between -1 (extremely bearish) and +1 (extremely bullish).
        Returns 0 if no relevant news found (neutral assumption).

        The composite is a weighted average where more extreme scores
        carry more weight (to capture strong sentiment signals).
        """
        articles = self.scan_all_feeds()
        if not articles:
            return 0.0

        # Weighted by absolute compound score (extreme sentiment = more important)
        scores = [a["compound"] for a in articles]
        weights = [abs(s) + 0.01 for s in scores]  # +0.01 to avoid zero weights

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0

        logger.info(f"Composite sentiment: {composite:.4f} (from {len(articles)} articles)")
        return composite

    def get_detailed_report(self) -> dict:
        """Get a detailed sentiment breakdown for logging/notification."""
        articles = self.scan_all_feeds()
        if not articles:
            return {"composite": 0.0, "article_count": 0, "top_bullish": [], "top_bearish": []}

        scores = [a["compound"] for a in articles]
        sorted_articles = sorted(articles, key=lambda x: x["compound"])

        return {
            "composite": self.get_composite_score(),
            "article_count": len(articles),
            "avg_score": sum(scores) / len(scores),
            "top_bullish": [a["title"] for a in sorted_articles[-3:]],
            "top_bearish": [a["title"] for a in sorted_articles[:3]],
        }
