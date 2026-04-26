from __future__ import annotations

from dataclasses import dataclass
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.schemas import Candle, Prediction
from app.services.analytics import build_feature_frame


@dataclass(slots=True)
class CachedModel:
    model: Pipeline
    trained_at: float
    sample_size: int


class PredictionEngine:
    def __init__(self) -> None:
        self._cache: dict[str, CachedModel] = {}

    def predict(self, ticker: str, candles: list[Candle]) -> Prediction:
        features = build_feature_frame(candles)
        if len(features) < 50:
            return Prediction(
                label="neutral",
                probability_up=0.5,
                confidence=0.4,
                model_version="fallback",
                drivers=["not-enough-history"],
            )

        model = self._get_or_train_model(ticker, features)
        feature_columns = [column for column in features.columns if column != "future_return"]
        latest = features[feature_columns].tail(1)
        probability_up = float(model.predict_proba(latest)[0][1])
        expected_move_pct = float(features["future_return"].tail(40).std() * (probability_up - 0.5) * 12 * 100)

        label = "neutral"
        if probability_up >= 0.57:
            label = "bullish"
        elif probability_up <= 0.43:
            label = "bearish"

        confidence = min(0.95, 0.45 + abs(probability_up - 0.5) * 1.7)
        drivers = []
        feature_importance = getattr(model.named_steps["forest"], "feature_importances_", [])
        if len(feature_importance) == len(feature_columns):
            ranked = sorted(zip(feature_columns, feature_importance), key=lambda item: item[1], reverse=True)
            drivers = [name for name, _ in ranked[:3]]

        return Prediction(
            label=label,
            probability_up=probability_up,
            confidence=confidence,
            expected_move_pct=expected_move_pct,
            model_version="rf-v1-free",
            drivers=drivers,
        )

    def _get_or_train_model(self, ticker: str, features) -> Pipeline:
        cached = self._cache.get(ticker)
        if cached and time() - cached.trained_at < 1800 and cached.sample_size == len(features):
            return cached.model

        feature_columns = [column for column in features.columns if column != "future_return"]
        target = (features["future_return"] > 0).astype(int)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "forest",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_depth=6,
                        min_samples_leaf=3,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(features[feature_columns], target)
        self._cache[ticker] = CachedModel(model=model, trained_at=time(), sample_size=len(features))
        return model

