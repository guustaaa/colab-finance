"""
regime.py — Hidden Markov Model regime detection.

Identifies whether the market is in one of N hidden states:
  - State 0: Low-volatility trending (momentum strategies work)
  - State 1: High-volatility trending (momentum works but needs tighter risk)
  - State 2: Choppy/Crisis (mean-reversion or stay flat)

Based on:
  - Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary
    Time Series and the Business Cycle"
  - Ang & Bekaert (2002) "Regime Switches in Interest Rates"
  - Practical FX applications: Bulla & Bulla (2006)

Key design decisions:
  - Uses log returns + realized volatility as observation features
    (NOT raw prices — stationarity is critical for HMM stability)
  - 3 states chosen as best tradeoff: enough to capture distinct regimes
    without overfitting (>4 states rarely improves FX regime detection)
  - Walk-forward retraining to handle structural breaks
"""
import warnings
import numpy as np
import pandas as pd
import joblib
import logging
from hmmlearn.hmm import GaussianHMM

# Suppress HMM convergence warnings (expected with limited data)
warnings.filterwarnings("ignore", module="hmmlearn")

from src.config import HMM_N_STATES, HMM_LOOKBACK

logger = logging.getLogger("regime")


class RegimeDetector:
    """
    Gaussian Hidden Markov Model for FX regime detection.

    Classifies market into regimes based on return/volatility dynamics.
    Used as a META-STRATEGY FILTER — does NOT generate trade signals directly.
    Instead, it modulates:
      - Which sub-strategy gets priority (momentum vs. mean-reversion)
      - Position sizing (reduced in crisis regime)
      - Whether to trade at all (flatline in uncertain regimes)
    """

    def __init__(self, n_states: int = HMM_N_STATES, lookback: int = HMM_LOOKBACK):
        self.n_states = n_states
        self.lookback = lookback
        self.model = None
        self.state_labels = {}  # maps state index → human-readable label

    def _prepare_observations(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare observation matrix for HMM fitting.

        Uses:
          - Log returns (direction + magnitude)
          - Rolling volatility (regime characteristic)
          - Return squared (captures volatility clustering / GARCH effects)
        """
        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol_24 = log_ret.rolling(24).std()
        ret_sq = log_ret ** 2

        obs = pd.DataFrame({
            "log_return": log_ret,
            "volatility": vol_24,
            "return_sq": ret_sq,
        }).dropna()

        return obs.values

    def fit(self, df: pd.DataFrame, model_path: str = None):
        """
        Fit the HMM on historical data.

        Uses the last `lookback` candles to avoid fitting on stale regime data.
        """
        # Use only the most recent data for regime fitting
        data = df.tail(self.lookback) if len(df) > self.lookback else df

        obs = self._prepare_observations(data)
        if len(obs) < 100:
            logger.warning(f"Insufficient data for HMM fitting: {len(obs)} observations")
            return False

        # Add noise regularization to prevent singular covariance
        obs = obs + np.random.normal(0, 1e-6, obs.shape)

        # Try covariance types in order of expressiveness, falling back if singular
        for cov_type in ["full", "diag", "spherical"]:
            try:
                self.model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=cov_type,
                    n_iter=200,
                    random_state=42,
                    tol=0.01,
                )
                self.model.fit(obs)
                self._label_states(obs)

                if model_path:
                    joblib.dump(self, model_path)   # save full RegimeDetector, not just self.model
                    logger.info(f"HMM model saved to {model_path}")

                logger.info(
                    f"HMM fitted successfully ({cov_type}). States: {self.state_labels}. "
                    f"Score: {self.model.score(obs):.2f}"
                )
                return True

            except Exception as e:
                logger.warning(f"HMM fit with cov_type='{cov_type}' failed: {e}")
                continue

        logger.error("HMM fitting failed with all covariance types.")
        return False

    def _label_states(self, obs: np.ndarray):
        """
        Automatically label states based on their characteristics.

        The state with the lowest volatility = "calm_trending"
        The state with medium volatility = "volatile_trending"
        The state with highest volatility = "crisis"
        """
        states = self.model.predict(obs)

        # Calculate mean volatility per state
        state_vols = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() > 0:
                # obs[:, 1] is the volatility column
                state_vols[s] = np.mean(obs[mask, 1])
            else:
                state_vols[s] = 0.0

        # Sort states by volatility
        sorted_states = sorted(state_vols.keys(), key=lambda x: state_vols[x])

        labels = ["calm_trending", "volatile_trending", "crisis"]
        self.state_labels = {}
        for i, state_idx in enumerate(sorted_states):
            if i < len(labels):
                self.state_labels[state_idx] = labels[i]

        logger.info(f"State labels assigned: {self.state_labels}")

    def detect(self, df: pd.DataFrame) -> dict:
        """
        Detect the current market regime.

        Returns a dict with:
          - state: int (state index)
          - label: str (human-readable label)
          - probabilities: dict of state probabilities
          - confidence: float (probability of most likely state)
        """
        if self.model is None:
            logger.warning("HMM not fitted. Returning default regime.")
            return {
                "state": 0,
                "label": "unknown",
                "probabilities": {},
                "confidence": 0.0,
            }

        obs = self._prepare_observations(df)
        if len(obs) < 10:
            return {
                "state": 0,
                "label": "unknown",
                "probabilities": {},
                "confidence": 0.0,
            }

        try:
            # Get state probabilities for the latest observation
            log_probs = self.model.score_samples(obs)
            posteriors = log_probs[1]  # state posteriors for each observation
            latest_probs = posteriors[-1]

            current_state = np.argmax(latest_probs)
            confidence = latest_probs[current_state]

            probs_dict = {
                self.state_labels.get(i, f"state_{i}"): float(p)
                for i, p in enumerate(latest_probs)
            }

            result = {
                "state": int(current_state),
                "label": self.state_labels.get(current_state, "unknown"),
                "probabilities": probs_dict,
                "confidence": float(confidence),
            }

            logger.info(
                f"Regime: {result['label']} (confidence: {result['confidence']:.2%})"
            )
            return result

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return {
                "state": 0,
                "label": "unknown",
                "probabilities": {},
                "confidence": 0.0,
            }

    def load(self, model_path: str) -> bool:
        """Load a previously fitted HMM model."""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"HMM model loaded from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load HMM model: {e}")
            return False

    def get_strategy_weights(self, regime: dict) -> dict:
        """
        Translate regime into strategy weights.

        This is the KEY function that connects regime detection to execution.
        Based on institutional practice:
          - Calm trending → heavy momentum weight
          - Volatile trending → balanced, tighter risk
          - Crisis → defensive, small positions, mean-reversion bias

        Returns weights for: momentum, mean_reversion, position_scale
        """
        label = regime.get("label", "unknown")
        confidence = regime.get("confidence", 0.5)

        weights = {
            "calm_trending": {
                "momentum_weight": 0.7,
                "mean_reversion_weight": 0.3,
                "position_scale": 1.0,  # Full position sizing
            },
            "volatile_trending": {
                "momentum_weight": 0.5,
                "mean_reversion_weight": 0.5,
                "position_scale": 0.6,  # Reduced sizing
            },
            "crisis": {
                "momentum_weight": 0.2,
                "mean_reversion_weight": 0.3,
                "position_scale": 0.2,  # Minimal sizing — capital preservation
            },
            "unknown": {
                "momentum_weight": 0.5,
                "mean_reversion_weight": 0.5,
                "position_scale": 0.3,  # Conservative when uncertain
            },
        }

        w = weights.get(label, weights["unknown"])

        # Scale position size by confidence (low confidence → smaller positions)
        w["position_scale"] *= max(confidence, 0.3)

        return w
