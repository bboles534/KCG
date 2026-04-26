from __future__ import annotations

from collections import Counter
from urllib.parse import urlparse

from app.config import Settings
from app.schemas import NewsItem


class SentimentEngine:
    POSITIVE_TERMS = {
        "beat": 1.1,
        "growth": 0.8,
        "upgrade": 1.1,
        "surge": 1.0,
        "record": 0.8,
        "profit": 0.9,
        "outperform": 1.1,
        "buyback": 0.8,
        "bullish": 1.0,
        "partnership": 0.7,
    }
    NEGATIVE_TERMS = {
        "miss": -1.1,
        "downgrade": -1.2,
        "lawsuit": -1.0,
        "fraud": -1.4,
        "plunge": -1.2,
        "warning": -0.8,
        "decline": -0.7,
        "layoff": -0.7,
        "bearish": -1.0,
        "probe": -0.8,
    }
    EVENT_TERMS = {
        "earnings",
        "guidance",
        "sec",
        "federal reserve",
        "rate",
        "acquisition",
        "merger",
        "dividend",
        "buyback",
        "cpi",
    }
    SENSATIONAL_TERMS = {
        "guaranteed",
        "shocking",
        "secret",
        "explode",
        "moon",
        "crash now",
        "insane",
        "must buy",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def enrich(self, items: list[NewsItem]) -> list[NewsItem]:
        title_counts = Counter(self._normalize_title(item.title) for item in items)
        enriched: list[NewsItem] = []
        for item in items:
            text = f"{item.title} {item.summary}".lower()
            score = 0.0
            score += sum(weight for term, weight in self.POSITIVE_TERMS.items() if term in text)
            score += sum(weight for term, weight in self.NEGATIVE_TERMS.items() if term in text)

            sentiment = "neutral"
            if score >= 0.7:
                sentiment = "positive"
            elif score <= -0.7:
                sentiment = "negative"

            credibility = self._credibility(text, item.url, title_counts[self._normalize_title(item.title)])
            relevance = min(
                1.0,
                0.45
                + 0.15 * sum(1 for term in self.EVENT_TERMS if term in text)
                + 0.1 * int(item.ticker.lower() in text),
            )
            flags: list[str] = []
            if credibility < 0.45:
                flags.append("low-credibility")
            if any(term in text for term in self.SENSATIONAL_TERMS):
                flags.append("possible-rumor")
            if title_counts[self._normalize_title(item.title)] == 1 and credibility < 0.7:
                flags.append("uncorroborated")
            if any(term in text for term in self.EVENT_TERMS):
                flags.append("market-moving")

            enriched.append(
                item.model_copy(
                    update={
                        "sentiment": sentiment,
                        "sentiment_score": score,
                        "credibility": credibility,
                        "relevance": relevance,
                        "flags": flags,
                    }
                )
            )
        return sorted(
            enriched,
            key=lambda item: (item.credibility * item.relevance, item.published_at),
            reverse=True,
        )

    def _credibility(self, text: str, url: str, corroboration_count: int) -> float:
        domain = urlparse(url).netloc.lower()
        credibility = 0.45
        if any(domain.endswith(trusted) for trusted in self.settings.trusted_news_domains):
            credibility += 0.3
        if url.startswith("https://"):
            credibility += 0.05
        if corroboration_count > 1:
            credibility += 0.1
        if any(term in text for term in self.SENSATIONAL_TERMS):
            credibility -= 0.2
        return max(0.05, min(0.99, credibility))

    @staticmethod
    def _normalize_title(title: str) -> str:
        return "".join(char for char in title.lower() if char.isalnum() or char.isspace()).strip()

