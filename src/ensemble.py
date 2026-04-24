"""
ensemble.py — Production ensemble model combining factor signals + XGBoost.

Architecture:
  Economic Factors (carry, momentum, value) → raw factor scores
  Technical Features (RSI, MACD, ATR, BB)  → confirmation
  HMM Regime                               → strategy weighting
  XGBoost Meta-Learner                     → optimal combination & timing

Why this works in live trading (vs. pure deep learning):
  1. Factor signals have ECONOMIC REASONING behind them (not just patterns)
  2. XGBoost is the best model for noisy tabular data (empirically proven)
  3. Walk-forward validation prevents in-sample overfitting
  4. Regime awareness prevents applying wrong strategy to wrong market
  5. Transaction cost filtering removes low-conviction trades

Key academic sources:
  - XGBoost superiority on tabular data: Grinsztajn et al. (2022)
    "Why do tree-based models still outperform deep learning on tabular data?"
  - Walk-forward validation: White (2000) "A Reality Check for Data Snooping"
  - Factor investing in FX: Kroencke et al. (2014) "International Diversification
    Benefits with Foreign Exchange Investment Styles"
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from sklearn.model_selection import TimeSeriesSplit

from src.features import get_feature_columns
from src.config import XGB_PARAMS, WALK_FORWARD_TRAIN_SIZE, WALK_FORWARD_TEST_SIZE

logger = logging.getLogger("ensemble")


class FactorModel:
    """
    Pure factor model — generates signals from economic factors WITHOUT ML.

    This is the "floor" — the minimum intelligence level.
    It implements the three documented FX risk premia:
      1. Momentum: buy winners, sell losers
      2. Value: buy undervalued, sell overvalued
      3. Carry: buy high-yield, sell low-yield
    """

    def score_momentum(self, features: pd.Series) -> float:
        """
        Score momentum signal.

        Combines short and long-term momentum with crossover signal.
        Positive = bullish momentum, negative = bearish.
        """
        mom_24 = features.get("momentum_24", 0)
        mom_168 = features.get("momentum_168", 0)
        crossover = features.get("momentum_crossover", 0)

        # Weighted combination: short-term more weight but long-term for confirmation
        score = 0.4 * np.sign(mom_24) + 0.3 * np.sign(mom_168) + 0.3 * np.sign(crossover)
        return np.clip(score, -1, 1)

    def score_value(self, features: pd.Series) -> float:
        """
        Score value signal.

        Mean-reversion: when price deviates far from long-term average,
        bet on it returning. Stronger signal when z-score is extreme.
        """
        zscore = features.get("value_zscore", 0)

        # Contrarian: extreme positive deviation → sell, extreme negative → buy
        if abs(zscore) < 1.0:
            return 0.0  # No signal in normal range
        return np.clip(-zscore * 0.3, -1, 1)  # Inverted: high zscore = sell

    def score_carry(self, features: pd.Series) -> float:
        """
        Score carry signal.

        Positive carry proxy = currency is earning positive roll → buy.
        """
        carry = features.get("carry_proxy", 0)
        return np.clip(carry * 1000, -1, 1)  # Scale for signal magnitude

    def get_composite_signal(
        self, features: pd.Series, weights: dict
    ) -> dict:
        """
        Combine factor scores using regime-adjusted weights.

        Parameters
        ----------
        features : pd.Series
            Latest feature row.
        weights : dict
            Regime-based weights from HMM (momentum_weight, mean_reversion_weight).

        Returns
        -------
        dict with signal, strength, and breakdown.
        """
        mom_score = self.score_momentum(features)
        val_score = self.score_value(features)
        carry_score = self.score_carry(features)

        mom_w = weights.get("momentum_weight", 0.5)
        mr_w = weights.get("mean_reversion_weight", 0.5)

        # Momentum and carry are "trend" strategies, value is "mean reversion"
        composite = (
            mom_w * (0.6 * mom_score + 0.4 * carry_score)
            + mr_w * val_score
        )

        return {
            "signal": "BUY" if composite > 0.1 else ("SELL" if composite < -0.1 else "HOLD"),
            "strength": abs(composite),
            "momentum": mom_score,
            "value": val_score,
            "carry": carry_score,
            "composite": composite,
        }


class XGBoostPredictor:
    """
    XGBoost classification model that learns optimal feature combination.

    Why XGBoost and not deep learning:
      - Grinsztajn et al. (2022): "tree-based models still outperform deep learning
        on typical tabular data" — and FX features are tabular.
      - Handles missing values natively
      - Built-in regularization (L1/L2) prevents overfitting on noisy data
      - Fast to retrain → enables proper walk-forward validation
      - Feature importance is interpretable → can debug strategy
    """

    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.model = None
        self.feature_cols = get_feature_columns()
        self.lgb_model = None  # LightGBM for ensemble diversity

    def train(self, df: pd.DataFrame, target_col: str = "target") -> dict:
        """
        Train XGBoost + LightGBM ensemble using walk-forward validation.

        Walk-forward prevents look-ahead bias:
          - Train on [0, T], test on [T, T+k]
          - Retrain on [0, T+k], test on [T+k, T+2k]
          - Report average out-of-sample performance

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame with all columns from features.py
        target_col : str
            Name of the target column (1 = price went up, 0 = price went down)

        Returns
        -------
        dict with training metrics (accuracy, walk-forward scores, etc.)
        """
        if df.empty or target_col not in df.columns:
            logger.error("Empty DataFrame or missing target column")
            return {}

        # Filter to available feature columns
        available_features = [c for c in self.feature_cols if c in df.columns]
        if len(available_features) < 10:
            logger.error(f"Too few features available: {available_features}")
            return {}

        X = df[available_features]
        y = df[target_col]

        # ── Walk-Forward Cross-Validation ──
        tscv = TimeSeriesSplit(n_splits=5)
        wf_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # XGBoost
            xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            score = xgb_model.score(X_test, y_test)
            wf_scores.append(score)
            logger.info(f"Walk-forward fold {fold + 1}: accuracy = {score:.4f}")

        # ── Final Training on Full Dataset ──
        # XGBoost
        self.model = xgb.XGBClassifier(**XGB_PARAMS)
        self.model.fit(X, y, verbose=False)

        # LightGBM (for ensemble diversity)
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        self.lgb_model.fit(X, y)

        # Save models
        if self.model_path:
            xgb_path = self.model_path.replace(".joblib", "_xgb.joblib")
            lgb_path = self.model_path.replace(".joblib", "_lgb.joblib")
            joblib.dump(self.model, xgb_path)
            joblib.dump(self.lgb_model, lgb_path)
            logger.info(f"Models saved: {xgb_path}, {lgb_path}")

        # Feature importance (for debugging and transparency)
        importances = dict(zip(available_features, self.model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

        metrics = {
            "walk_forward_scores": wf_scores,
            "mean_wf_accuracy": np.mean(wf_scores),
            "std_wf_accuracy": np.std(wf_scores),
            "top_features": top_features,
            "n_train_samples": len(X),
        }

        logger.info(
            f"Training complete. Walk-forward accuracy: {metrics['mean_wf_accuracy']:.4f} "
            f"± {metrics['std_wf_accuracy']:.4f}"
        )
        logger.info(f"Top features: {[f[0] for f in top_features[:5]]}")

        return metrics

    def predict(self, features: pd.DataFrame) -> dict:
        """
        Predict direction using XGBoost + LightGBM ensemble.

        Returns probability-weighted prediction.
        Both models vote; the average probability determines the signal.
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return {"signal": "HOLD", "probability": 0.5, "confidence": 0.0}

        available = [c for c in self.feature_cols if c in features.columns]
        X = features[available]

        if X.empty:
            return {"signal": "HOLD", "probability": 0.5, "confidence": 0.0}

        # Get latest row
        X_latest = X.iloc[[-1]]

        # XGBoost probability
        xgb_prob = self.model.predict_proba(X_latest)[0]
        xgb_up_prob = xgb_prob[1] if len(xgb_prob) > 1 else xgb_prob[0]

        # LightGBM probability (if available)
        if self.lgb_model is not None:
            lgb_prob = self.lgb_model.predict_proba(X_latest)[0]
            lgb_up_prob = lgb_prob[1] if len(lgb_prob) > 1 else lgb_prob[0]
            # Average ensemble
            avg_prob = 0.5 * xgb_up_prob + 0.5 * lgb_up_prob
        else:
            avg_prob = xgb_up_prob

        confidence = abs(avg_prob - 0.5) * 2  # 0 = no confidence, 1 = max confidence

        # Only signal when confidence exceeds threshold
        if confidence < 0.15:  # ~57.5% probability threshold
            signal = "HOLD"
        elif avg_prob > 0.5:
            signal = "BUY"
        else:
            signal = "SELL"

        return {
            "signal": signal,
            "probability": float(avg_prob),
            "confidence": float(confidence),
            "xgb_prob": float(xgb_up_prob),
            "lgb_prob": float(lgb_up_prob) if self.lgb_model else None,
        }

    def load(self, model_path: str = None) -> bool:
        """Load previously trained models."""
        path = model_path or self.model_path
        try:
            xgb_path = path.replace(".joblib", "_xgb.joblib")
            lgb_path = path.replace(".joblib", "_lgb.joblib")
            self.model = joblib.load(xgb_path)
            self.lgb_model = joblib.load(lgb_path)
            logger.info(f"Models loaded from {xgb_path} and {lgb_path}")
            return True
        except Exception as e:
            logger.warning(f"Model load failed: {e}")
            return False


class EnsembleEngine:
    """
    Master ensemble that combines:
      1. Factor model signals (economic reasoning)
      2. XGBoost/LightGBM ML predictions (pattern recognition)
      3. HMM regime context (strategy selection)

    The ensemble does NOT simply average. It uses the regime to determine
    HOW MUCH to trust each sub-model:
      - In calm trending markets → ML gets more weight (patterns are stable)
      - In crisis → Factor model gets more weight (economic logic persists)
      - Low confidence → reduce position or HOLD
    """

    def __init__(self, model_path: str = ""):
        self.factor_model = FactorModel()
        self.ml_model = XGBoostPredictor(model_path=model_path)

    def train(self, df: pd.DataFrame) -> dict:
        """Train the ML component of the ensemble."""
        # Create target: did price go up in the next candle?
        df = df.copy()
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df.dropna(subset=["target"], inplace=True)
        return self.ml_model.train(df)

    def predict(
        self, features_df: pd.DataFrame, regime: dict, regime_weights: dict
    ) -> dict:
        """
        Generate the final trading signal by combining all components.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature-engineered DataFrame (output of compute_all_features)
        regime : dict
            Current regime from RegimeDetector.detect()
        regime_weights : dict
            Strategy weights from RegimeDetector.get_strategy_weights()

        Returns
        -------
        dict with: signal, confidence, position_scale, details
        """
        if features_df.empty:
            return self._default_signal()

        latest = features_df.iloc[-1]

        # 1. Factor model signal
        factor_result = self.factor_model.get_composite_signal(latest, regime_weights)

        # 2. ML model signal
        ml_result = self.ml_model.predict(features_df)

        # 3. Combine based on regime
        regime_label = regime.get("label", "unknown")

        # Regime-based trust allocation
        if regime_label == "calm_trending":
            ml_trust = 0.6
            factor_trust = 0.4
        elif regime_label == "volatile_trending":
            ml_trust = 0.4
            factor_trust = 0.6
        elif regime_label == "crisis":
            ml_trust = 0.2
            factor_trust = 0.8
        else:
            ml_trust = 0.3
            factor_trust = 0.7

        # Convert ML signal to numeric
        ml_score = (ml_result["probability"] - 0.5) * 2  # [-1, 1]
        factor_score = factor_result["composite"]

        # Weighted ensemble score
        ensemble_score = ml_trust * ml_score + factor_trust * factor_score
        ensemble_confidence = (
            ml_trust * ml_result["confidence"]
            + factor_trust * factor_result["strength"]
        )

        # Position scale from regime
        position_scale = regime_weights.get("position_scale", 0.5)

        # Final signal determination
        if ensemble_confidence < 0.10:
            signal = "HOLD"
        elif ensemble_score > 0.05:
            signal = "BUY"
        elif ensemble_score < -0.05:
            signal = "SELL"
        else:
            signal = "HOLD"

        result = {
            "signal": signal,
            "confidence": float(ensemble_confidence),
            "position_scale": float(position_scale),
            "ensemble_score": float(ensemble_score),
            "regime": regime_label,
            "details": {
                "factor": factor_result,
                "ml": ml_result,
                "ml_trust": ml_trust,
                "factor_trust": factor_trust,
            },
        }

        logger.info(
            f"Ensemble: {signal} | score={ensemble_score:.4f} | "
            f"conf={ensemble_confidence:.4f} | regime={regime_label} | "
            f"pos_scale={position_scale:.2f}"
        )
        return result

    def load_ml_model(self, path: str) -> bool:
        """Load the ML model from disk."""
        return self.ml_model.load(path)

    def _default_signal(self) -> dict:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "position_scale": 0.0,
            "ensemble_score": 0.0,
            "regime": "unknown",
            "details": {},
        }
