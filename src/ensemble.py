"""
ensemble.py — Deep ensemble: Factor model + XGBoost/LightGBM + PyTorch DNN stacker.

Architecture:
  Layer 1: Factor signals (carry, momentum, value)
  Layer 2: XGBoost + LightGBM (gradient boosted trees)
  Layer 3: Deep Neural Net stacker (learns optimal combination)
  Context: HMM regime weighting
"""
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Suppress sklearn feature name warnings (we handle this correctly)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from src.features import get_feature_columns
from src.config import XGB_PARAMS, LGB_PARAMS, WALK_FORWARD_TRAIN_SIZE, WALK_FORWARD_TEST_SIZE

logger = logging.getLogger("ensemble")

# ─────────────────────────────────────────────
# Layer 0: Factor Model
# ─────────────────────────────────────────────

class FactorModel:
    """Pure factor model — generates signals from economic factors WITHOUT ML."""

    def score_momentum(self, features: pd.Series) -> float:
        mom_12 = features.get("momentum_12", 0)
        mom_24 = features.get("momentum_24", 0)
        mom_168 = features.get("momentum_168", 0)
        crossover = features.get("momentum_crossover", 0)
        score = 0.3 * np.sign(mom_12) + 0.3 * np.sign(mom_24) + 0.2 * np.sign(mom_168) + 0.2 * np.sign(crossover)
        return np.clip(score, -1, 1)

    def score_value(self, features: pd.Series) -> float:
        zscore = features.get("value_zscore", 0)
        if abs(zscore) < 0.8:
            return 0.0
        return np.clip(-zscore * 0.35, -1, 1)

    def score_carry(self, features: pd.Series) -> float:
        carry = features.get("carry_proxy", 0)
        return np.clip(carry * 1000, -1, 1)

    def get_composite_signal(self, features: pd.Series, weights: dict) -> dict:
        mom_score = self.score_momentum(features)
        val_score = self.score_value(features)
        carry_score = self.score_carry(features)

        mom_w = weights.get("momentum_weight", 0.5)
        mr_w = weights.get("mean_reversion_weight", 0.5)

        composite = mom_w * (0.6 * mom_score + 0.4 * carry_score) + mr_w * val_score

        return {
            "signal": "BUY" if composite > 0.08 else ("SELL" if composite < -0.08 else "HOLD"),
            "strength": abs(composite),
            "momentum": mom_score, "value": val_score, "carry": carry_score,
            "composite": composite,
        }


# ─────────────────────────────────────────────
# Layer 3: Deep Neural Net Stacker
# ─────────────────────────────────────────────

class DeepStacker:
    """
    Deep neural network that stacks on top of XGB/LGB predictions.
    Uses raw features + tree model outputs as input.
    4-layer MLP with dropout and batch norm.
    """

    def __init__(self, input_dim: int = 50, hidden_dims: list = None):
        self.model = None
        self.scaler = StandardScaler()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128, 64, 32]
        self._torch_available = False
        try:
            import torch
            import torch.nn as nn
            self._torch_available = True
        except ImportError:
            logger.warning("PyTorch not available — DNN stacker disabled")

    def _build_model(self, input_dim: int):
        if not self._torch_available:
            return None
        import torch.nn as nn

        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.3),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 150, lr: float = 0.001):
        if not self._torch_available or len(X) < 50:
            return
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_scaled = self.scaler.fit_transform(X)
        self.input_dim = X_scaled.shape[1]
        self.model = self._build_model(self.input_dim)

        device = torch.device("cpu")  # XGB features are CPU; keep stacker on CPU too
        self.model = self.model.to(device)

        X_t = torch.FloatTensor(X_scaled).to(device)
        y_t = torch.FloatTensor(y.astype(np.float32)).unsqueeze(1).to(device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=min(64, len(X) // 4 + 1), shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

        logger.info(f"DNN stacker trained: {epochs} epochs, final loss={total_loss/len(loader):.4f}")

    def predict_proba(self, X: np.ndarray) -> float:
        if self.model is None or not self._torch_available:
            return 0.5
        import torch
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
            X_t = torch.FloatTensor(X_scaled)
            prob = self.model(X_t).item()
        return prob


# ─────────────────────────────────────────────
# Layer 2: XGBoost + LightGBM Predictor
# ─────────────────────────────────────────────

class XGBoostPredictor:
    """XGBoost + LightGBM + DNN stacker ensemble."""

    def __init__(self, model_path: str = "", cross_pairs: list = None):
        self.model_path = model_path
        self.model = None
        self.lgb_model = None
        self.dnn = DeepStacker()
        self.cross_pairs = cross_pairs or []
        self.feature_cols = get_feature_columns(self.cross_pairs)

    def train(self, df: pd.DataFrame, target_col: str = "target") -> dict:
        if df.empty or target_col not in df.columns:
            logger.error("Empty DataFrame or missing target column")
            return {}

        available_features = [c for c in self.feature_cols if c in df.columns]
        # Also grab any cross-pair features that exist
        for c in df.columns:
            if (c.startswith("corr_") or c.startswith("beta_")) and c not in available_features:
                available_features.append(c)

        if len(available_features) < 10:
            logger.error(f"Too few features: {len(available_features)}")
            return {}

        self.feature_cols = available_features  # update to actual available
        X = df[available_features]
        y = df[target_col]

        # ── Walk-Forward Cross-Validation ──
        n_splits = min(5, max(2, len(X) // 80))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        wf_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            score = xgb_model.score(X_test, y_test)
            wf_scores.append(score)

        # ── Final Training ──
        split = max(1, int(len(X) * 0.85))
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]

        self.model = xgb.XGBClassifier(**XGB_PARAMS)
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        self.lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        self.lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        # ── Train DNN stacker on tree outputs + raw features ──
        xgb_probs = self.model.predict_proba(X)[:, 1]
        lgb_probs = self.lgb_model.predict_proba(X)[:, 1]
        stacker_X = np.column_stack([X.values, xgb_probs, lgb_probs])
        self.dnn.train(stacker_X, y.values, epochs=200)

        # Save
        if self.model_path:
            joblib.dump(self.model, self.model_path.replace(".joblib", "_xgb.joblib"))
            joblib.dump(self.lgb_model, self.model_path.replace(".joblib", "_lgb.joblib"))
            joblib.dump(self.dnn, self.model_path.replace(".joblib", "_dnn.joblib"))

        importances = dict(zip(available_features, self.model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

        metrics = {
            "walk_forward_scores": wf_scores,
            "mean_wf_accuracy": np.mean(wf_scores),
            "std_wf_accuracy": np.std(wf_scores),
            "top_features": top_features,
            "n_train_samples": len(X),
            "n_features": len(available_features),
            "dnn_active": self.dnn.model is not None,
        }
        logger.info(f"Training done. WF acc: {metrics['mean_wf_accuracy']:.4f} ± {metrics['std_wf_accuracy']:.4f} | {len(available_features)} features | DNN={'ON' if self.dnn.model else 'OFF'}")
        return metrics

    def predict(self, features: pd.DataFrame) -> dict:
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return {"signal": "HOLD", "probability": 0.5, "confidence": 0.0}

        available = [c for c in self.feature_cols if c in features.columns]
        X = features[available]
        if X.empty:
            return {"signal": "HOLD", "probability": 0.5, "confidence": 0.0}

        X_latest = X.iloc[[-1]]

        # XGBoost probability
        xgb_prob = self.model.predict_proba(X_latest)[0]
        xgb_up = xgb_prob[1] if len(xgb_prob) > 1 else xgb_prob[0]

        # LightGBM probability
        lgb_up = xgb_up
        if self.lgb_model is not None:
            lgb_prob = self.lgb_model.predict_proba(X_latest)[0]
            lgb_up = lgb_prob[1] if len(lgb_prob) > 1 else lgb_prob[0]

        # DNN stacker probability
        dnn_up = 0.5
        if self.dnn.model is not None:
            stacker_input = np.concatenate([X_latest.values.flatten(), [xgb_up, lgb_up]])
            dnn_up = self.dnn.predict_proba(stacker_input)

        # Weighted ensemble: trees 60%, DNN 40%
        if self.dnn.model is not None:
            avg_prob = 0.3 * xgb_up + 0.3 * lgb_up + 0.4 * dnn_up
        else:
            avg_prob = 0.5 * xgb_up + 0.5 * lgb_up

        # Confidence: how far from 0.5, stretched to be more decisive
        raw_conf = abs(avg_prob - 0.5) * 2
        # Apply sigmoid stretch to amplify moderate confidence
        confidence = 1.0 / (1.0 + np.exp(-8 * (raw_conf - 0.15)))

        if confidence < 0.10:
            signal = "HOLD"
        elif avg_prob > 0.5:
            signal = "BUY"
        else:
            signal = "SELL"

        return {
            "signal": signal,
            "probability": float(avg_prob),
            "confidence": float(confidence),
            "xgb_prob": float(xgb_up),
            "lgb_prob": float(lgb_up),
            "dnn_prob": float(dnn_up),
        }

    def load(self, model_path: str = None) -> bool:
        path = model_path or self.model_path
        try:
            self.model = joblib.load(path.replace(".joblib", "_xgb.joblib"))
            self.lgb_model = joblib.load(path.replace(".joblib", "_lgb.joblib"))
            dnn_path = path.replace(".joblib", "_dnn.joblib")
            import os
            if os.path.exists(dnn_path):
                self.dnn = joblib.load(dnn_path)
            return True
        except Exception as e:
            logger.warning(f"Model load failed: {e}")
            return False


# ─────────────────────────────────────────────
# Master Ensemble
# ─────────────────────────────────────────────

class EnsembleEngine:
    """
    3-layer ensemble:
      Factor model → XGB/LGB → DNN stacker → regime-weighted output.
    """

    def __init__(self, model_path: str = "", cross_pairs: list = None):
        self.factor_model = FactorModel()
        self.ml_model = XGBoostPredictor(model_path=model_path, cross_pairs=cross_pairs)

    def train(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df.dropna(subset=["target"], inplace=True)
        return self.ml_model.train(df)

    def predict(self, features_df: pd.DataFrame, regime: dict, regime_weights: dict) -> dict:
        if features_df.empty:
            return self._default_signal()

        latest = features_df.iloc[-1]

        # Factor model
        factor_result = self.factor_model.get_composite_signal(latest, regime_weights)

        # ML model (XGB + LGB + DNN)
        ml_result = self.ml_model.predict(features_df)

        # Regime-based trust
        regime_label = regime.get("label", "unknown")
        if regime_label == "calm_trending":
            ml_trust, factor_trust = 0.7, 0.3
        elif regime_label == "volatile_trending":
            ml_trust, factor_trust = 0.5, 0.5
        elif regime_label == "crisis":
            ml_trust, factor_trust = 0.3, 0.7
        else:
            ml_trust, factor_trust = 0.5, 0.5

        ml_score = (ml_result["probability"] - 0.5) * 2
        factor_score = factor_result["composite"]

        ensemble_score = ml_trust * ml_score + factor_trust * factor_score
        ensemble_confidence = ml_trust * ml_result["confidence"] + factor_trust * factor_result["strength"]

        position_scale = regime_weights.get("position_scale", 0.5)

        if ensemble_confidence < 0.08:
            signal = "HOLD"
        elif ensemble_score > 0.03:
            signal = "BUY"
        elif ensemble_score < -0.03:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "confidence": float(ensemble_confidence),
            "position_scale": float(position_scale),
            "ensemble_score": float(ensemble_score),
            "regime": regime_label,
            "details": {
                "factor": factor_result, "ml": ml_result,
                "ml_trust": ml_trust, "factor_trust": factor_trust,
            },
        }

    def load_ml_model(self, path: str) -> bool:
        return self.ml_model.load(path)

    def _default_signal(self) -> dict:
        return {
            "signal": "HOLD", "confidence": 0.0, "position_scale": 0.0,
            "ensemble_score": 0.0, "regime": "unknown", "details": {},
        }
