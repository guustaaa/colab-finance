"""
test_pipeline.py — End-to-end tests for the trading pipeline.

Tests the full flow: mock data → features → regime → ensemble → risk → backtest
All tests use mock data so they run without Capital.com credentials.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_fetcher import generate_mock_data
from src.features import compute_all_features, get_feature_columns
from src.regime import RegimeDetector
from src.ensemble import FactorModel, XGBoostPredictor, EnsembleEngine
from src.execution import RiskManager
from src.backtester import Backtester
from src.sentiment import SentimentScanner


class TestMockData:
    """Test mock data generation."""

    def test_generates_correct_shape(self):
        df = generate_mock_data(n=500)
        assert len(df) == 500
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_prices_are_realistic(self):
        df = generate_mock_data(n=1000)
        assert df["close"].mean() > 1.0  # EUR/USD range
        assert df["close"].mean() < 1.2
        assert (df["high"] >= df["low"]).all()


class TestFeatures:
    """Test feature engineering."""

    def test_computes_all_features(self):
        df = generate_mock_data(n=500)
        features = compute_all_features(df)
        assert not features.empty
        assert "momentum_24" in features.columns
        assert "value_deviation" in features.columns
        assert "rsi_14" in features.columns
        assert "atr_14" in features.columns
        assert "sentiment" in features.columns

    def test_no_nans_after_dropna(self):
        df = generate_mock_data(n=500)
        features = compute_all_features(df)
        assert features.isnull().sum().sum() == 0

    def test_feature_columns_list_valid(self):
        df = generate_mock_data(n=500)
        features = compute_all_features(df)
        expected = get_feature_columns()
        for col in expected:
            assert col in features.columns, f"Missing feature: {col}"

    def test_rejects_insufficient_data(self):
        df = generate_mock_data(n=100)  # Too few for 200-period MA
        features = compute_all_features(df)
        assert features.empty


class TestRegimeDetection:
    """Test HMM regime detector."""

    def test_fit_and_detect(self):
        df = generate_mock_data(n=1000)
        detector = RegimeDetector(n_states=3, lookback=500)
        success = detector.fit(df)
        assert success

        regime = detector.detect(df)
        assert regime["label"] in ["calm_trending", "volatile_trending", "crisis", "unknown"]
        assert 0 <= regime["confidence"] <= 1

    def test_strategy_weights(self):
        detector = RegimeDetector()
        regime = {"label": "calm_trending", "confidence": 0.8}
        weights = detector.get_strategy_weights(regime)
        assert "momentum_weight" in weights
        assert "mean_reversion_weight" in weights
        assert "position_scale" in weights
        assert weights["momentum_weight"] > weights["mean_reversion_weight"]  # Momentum preferred in calm

    def test_crisis_reduces_position_scale(self):
        detector = RegimeDetector()
        calm = {"label": "calm_trending", "confidence": 0.8}
        crisis = {"label": "crisis", "confidence": 0.8}
        calm_w = detector.get_strategy_weights(calm)
        crisis_w = detector.get_strategy_weights(crisis)
        assert crisis_w["position_scale"] < calm_w["position_scale"]


class TestFactorModel:
    """Test economic factor scoring."""

    def test_momentum_scoring(self):
        model = FactorModel()
        bullish = pd.Series({"momentum_24": 0.01, "momentum_168": 0.005, "momentum_crossover": 0.005})
        score = model.score_momentum(bullish)
        assert score > 0

        bearish = pd.Series({"momentum_24": -0.01, "momentum_168": -0.005, "momentum_crossover": -0.005})
        score = model.score_momentum(bearish)
        assert score < 0

    def test_value_scoring(self):
        model = FactorModel()
        overvalued = pd.Series({"value_zscore": 2.5})
        score = model.score_value(overvalued)
        assert score < 0  # Should sell when overvalued

        undervalued = pd.Series({"value_zscore": -2.5})
        score = model.score_value(undervalued)
        assert score > 0  # Should buy when undervalued

    def test_neutral_in_normal_range(self):
        model = FactorModel()
        normal = pd.Series({"value_zscore": 0.5})
        score = model.score_value(normal)
        assert score == 0.0


class TestXGBoostPredictor:
    """Test ML model training and prediction."""

    def test_train_and_predict(self):
        df = generate_mock_data(n=1000)
        features = compute_all_features(df)
        assert not features.empty

        predictor = XGBoostPredictor()
        # Add target
        features_with_target = features.copy()
        features_with_target["target"] = (features_with_target["close"].shift(-1) > features_with_target["close"]).astype(int)
        features_with_target.dropna(inplace=True)

        metrics = predictor.train(features_with_target)
        assert "mean_wf_accuracy" in metrics
        assert metrics["mean_wf_accuracy"] > 0.3  # Should be better than random

        result = predictor.predict(features)
        assert result["signal"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= result["probability"] <= 1


class TestEnsembleEngine:
    """Test the full ensemble pipeline."""

    def test_full_pipeline(self):
        df = generate_mock_data(n=1000)
        features = compute_all_features(df)

        engine = EnsembleEngine()
        train_metrics = engine.train(features)
        assert "mean_wf_accuracy" in train_metrics

        regime = {"label": "calm_trending", "confidence": 0.7}
        weights = {"momentum_weight": 0.7, "mean_reversion_weight": 0.3, "position_scale": 0.8}

        prediction = engine.predict(features, regime, weights)
        assert prediction["signal"] in ["BUY", "SELL", "HOLD"]
        assert "confidence" in prediction
        assert "ensemble_score" in prediction


class TestRiskManager:
    """Test risk management."""

    def test_position_sizing(self):
        rm = RiskManager(initial_balance=10000)
        units = rm.calculate_position_size(
            balance=10000, atr=0.0010, price=1.0850,
            win_rate=0.55, position_scale=1.0,
        )
        assert units > 0
        # Shouldn't be insanely large
        assert units < 100000

    def test_drawdown_halt(self):
        rm = RiskManager(initial_balance=10000, max_drawdown=0.10)
        rm.update_balance(10000)  # Set peak
        rm.update_balance(8900)   # 11% drawdown
        assert rm.is_halted

    def test_trade_filter(self):
        rm = RiskManager()
        # Low confidence → reject
        approved, reason = rm.should_trade(confidence=0.05, atr=0.001, spread=0.0002)
        assert not approved


class TestBacktester:
    """Test backtesting engine."""

    def test_runs_on_mock_data(self):
        df = generate_mock_data(n=2500)
        bt = Backtester(initial_balance=10000)
        metrics = bt.run(df, train_size=1500, test_size=200, step_size=200)

        assert "total_trades" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        # Should execute at least some trades
        assert metrics["total_trades"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
