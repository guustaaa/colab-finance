"""
features.py — Feature engineering pipeline.

Implements THREE categories of features based on what works in live trading:

1. ECONOMIC FACTORS (carry, momentum, value) — the actual alpha sources
   Based on: AQR "Value & Momentum Everywhere" (2013),
             Menkhoff et al. "Currency Momentum Strategies" (2012),
             Lustig et al. "Common Risk Factors in Currency Markets" (2011)

2. TECHNICAL INDICATORS — risk management and confirmation signals
   ATR for position sizing, RSI/MACD for regime confirmation.
   NOT used as primary alpha — academically proven to be weak standalone.

3. MICROSTRUCTURE FEATURES — volume, volatility patterns
   Used to detect regime changes and filter low-quality signals.
"""
import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACD_Indicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

logger = logging.getLogger("features")


def compute_all_features(df: pd.DataFrame, sentiment: float = 0.0) -> pd.DataFrame:
    """
    Master feature engineering function.

    Takes raw OHLCV data and returns a DataFrame with all features computed.
    Drops NaN rows caused by lookback windows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data (must have open, high, low, close, volume columns).
    sentiment : float
        Current market sentiment score (-1 to +1) from news/RSS scanner.

    Returns
    -------
    pd.DataFrame with original OHLCV + all computed features.
    """
    if df.empty or len(df) < 250:
        logger.warning(f"Insufficient data for feature computation: {len(df)} rows (need >=250)")
        return pd.DataFrame()

    feat = df.copy()

    # -------- 1. RETURNS & LOG RETURNS --------
    feat["returns"] = feat["close"].pct_change()
    feat["log_returns"] = np.log(feat["close"] / feat["close"].shift(1))

    # -------- 2. ECONOMIC FACTORS --------

    # MOMENTUM - Multi-horizon (Asness, Moskowitz, Pedersen 2013)
    feat["momentum_24"] = feat["close"].pct_change(periods=24)
    feat["momentum_168"] = feat["close"].pct_change(periods=168)
    feat["momentum_crossover"] = feat["momentum_24"] - feat["momentum_168"]

    # VALUE - PPP deviation proxy
    feat["ma_200"] = feat["close"].rolling(window=200).mean()
    feat["value_deviation"] = (feat["close"] - feat["ma_200"]) / feat["ma_200"]
    feat["value_zscore"] = (
        (feat["value_deviation"] - feat["value_deviation"].rolling(100).mean())
        / feat["value_deviation"].rolling(100).std()
    )

    # CARRY - approximated via overnight return pattern
    feat["carry_proxy"] = feat["returns"].rolling(window=168).mean()

    # VOLATILITY FACTOR - realized vol for position sizing
    feat["realized_vol_24"] = feat["log_returns"].rolling(window=24).std() * np.sqrt(24)
    feat["realized_vol_168"] = feat["log_returns"].rolling(window=168).std() * np.sqrt(168)
    feat["vol_ratio"] = feat["realized_vol_24"] / feat["realized_vol_168"]

    # -------- 3. TECHNICAL INDICATORS (using ta library) --------

    # RSI - Wilder's Relative Strength Index
    rsi = RSIIndicator(close=feat["close"], window=14)
    feat["rsi_14"] = rsi.rsi()

    # MACD - Moving Average Convergence Divergence
    macd = MACD_Indicator(close=feat["close"], window_fast=12, window_slow=26, window_sign=9)
    feat["macd"] = macd.macd()
    feat["macd_signal"] = macd.macd_signal()
    feat["macd_hist"] = macd.macd_diff()

    # ATR - Average True Range (for stop loss / position sizing)
    atr = AverageTrueRange(high=feat["high"], low=feat["low"], close=feat["close"], window=14)
    feat["atr_14"] = atr.average_true_range()

    # Bollinger Bands - volatility envelope
    bb = BollingerBands(close=feat["close"], window=20, window_dev=2)
    feat["bb_upper"] = bb.bollinger_hband()
    feat["bb_mid"] = bb.bollinger_mavg()
    feat["bb_lower"] = bb.bollinger_lband()
    feat["bb_width"] = (feat["bb_upper"] - feat["bb_lower"]) / feat["bb_mid"]
    feat["bb_pct"] = (feat["close"] - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"])

    # ADX - Average Directional Index (trend strength)
    adx = ADXIndicator(high=feat["high"], low=feat["low"], close=feat["close"], window=14)
    feat["adx_14"] = adx.adx()

    # -------- 4. MICROSTRUCTURE FEATURES --------

    # Volume relative to average
    feat["vol_ma_20"] = feat["volume"].rolling(window=20).mean()
    feat["volume_ratio"] = feat["volume"] / feat["vol_ma_20"]

    # Range (high-low) as pct of close - measures intrabar volatility
    feat["range_pct"] = (feat["high"] - feat["low"]) / feat["close"]

    # Body ratio - measures candle conviction
    feat["body_ratio"] = abs(feat["close"] - feat["open"]) / (feat["high"] - feat["low"] + 1e-10)

    # -------- 5. LAG FEATURES (for XGBoost) --------
    for lag in [1, 2, 3, 5, 10]:
        feat[f"return_lag_{lag}"] = feat["returns"].shift(lag)
        feat[f"vol_lag_{lag}"] = feat["realized_vol_24"].shift(lag)

    # -------- 6. SENTIMENT (external input) --------
    feat["sentiment"] = sentiment

    # -------- DROP NaN rows from rolling windows --------
    feat.dropna(inplace=True)

    logger.info(f"Feature engineering complete: {len(feat)} rows, {len(feat.columns)} features")
    return feat


def get_feature_columns():
    """Return the list of feature column names used by the model."""
    return [
        # Economic factors
        "momentum_24", "momentum_168", "momentum_crossover",
        "value_deviation", "value_zscore",
        "carry_proxy",
        "realized_vol_24", "realized_vol_168", "vol_ratio",
        # Technical
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "atr_14", "bb_width", "bb_pct", "adx_14",
        # Microstructure
        "volume_ratio", "range_pct", "body_ratio",
        # Lags
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5", "return_lag_10",
        "vol_lag_1", "vol_lag_2", "vol_lag_3", "vol_lag_5", "vol_lag_10",
        # Sentiment
        "sentiment",
    ]
