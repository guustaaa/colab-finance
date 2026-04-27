"""
features.py — Feature engineering pipeline.

Implements FIVE categories of features based on what works in live trading:

1. ECONOMIC FACTORS (carry, momentum, value) — the actual alpha sources
2. TECHNICAL INDICATORS — risk management and confirmation signals
3. MICROSTRUCTURE FEATURES — volume, volatility patterns
4. CROSS-PAIR CORRELATION — inter-market structure
5. HIGHER-ORDER FEATURES — skew, kurtosis, interactions
"""
import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACD_Indicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

logger = logging.getLogger("features")


def compute_all_features(
    df: pd.DataFrame,
    sentiment: float = 0.0,
    cross_pair_data: dict | None = None,
) -> pd.DataFrame:
    """
    Master feature engineering function.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data (open, high, low, close, volume).
    sentiment : float
        Current market sentiment score (-1 to +1).
    cross_pair_data : dict | None
        {instrument_name: pd.DataFrame} for all other pairs.
        Used to compute cross-pair correlation features.
    """
    if df.empty or len(df) < 200:
        logger.warning(f"Insufficient data: {len(df)} rows (need >=200)")
        return pd.DataFrame()

    feat = df.copy()
    
    # Strictly deduplicate index to prevent alignment/reindex panics down the pipeline
    feat = feat[~feat.index.duplicated(keep="last")]

    # ──────── 1. RETURNS & LOG RETURNS ────────
    feat["returns"] = feat["close"].pct_change()
    feat["log_returns"] = np.log(feat["close"] / feat["close"].shift(1))

    # ──────── 2. ECONOMIC FACTORS ────────
    for p in [12, 24, 48, 168]:
        feat[f"momentum_{p}"] = feat["close"].pct_change(periods=p)
    feat["momentum_crossover"] = feat["momentum_24"] - feat["momentum_168"]

    # Value — PPP deviation proxy
    for w in [50, 100, 200]:
        feat[f"ma_{w}"] = feat["close"].rolling(window=w).mean()
    feat["value_deviation"] = (feat["close"] - feat["ma_200"]) / feat["ma_200"]
    feat["value_zscore"] = (
        (feat["value_deviation"] - feat["value_deviation"].rolling(100).mean())
        / feat["value_deviation"].rolling(100).std()
    )
    # Mean-reversion speed
    feat["mr_speed"] = feat["value_deviation"].diff()

    # Carry proxy
    feat["carry_proxy"] = feat["returns"].rolling(window=168).mean()

    # ──────── 3. VOLATILITY & HIGHER-ORDER ────────
    for w in [12, 24, 48, 168]:
        feat[f"realized_vol_{w}"] = feat["log_returns"].rolling(window=w).std() * np.sqrt(w)
    feat["vol_ratio"] = feat["realized_vol_24"] / (feat["realized_vol_168"] + 1e-10)
    feat["vol_regime_change"] = feat["realized_vol_24"] - feat["realized_vol_24"].shift(24)

    # Skewness & kurtosis (distribution shape)
    feat["skew_24"] = feat["log_returns"].rolling(24).skew()
    feat["kurt_24"] = feat["log_returns"].rolling(24).kurt()
    feat["skew_168"] = feat["log_returns"].rolling(168).skew()

    # ──────── 4. TECHNICAL INDICATORS ────────
    rsi = RSIIndicator(close=feat["close"], window=14)
    feat["rsi_14"] = rsi.rsi()

    macd = MACD_Indicator(close=feat["close"], window_fast=12, window_slow=26, window_sign=9)
    feat["macd"] = macd.macd()
    feat["macd_signal"] = macd.macd_signal()
    feat["macd_hist"] = macd.macd_diff()
    feat["macd_hist_accel"] = feat["macd_hist"].diff()  # acceleration of MACD

    atr = AverageTrueRange(high=feat["high"], low=feat["low"], close=feat["close"], window=14)
    feat["atr_14"] = atr.average_true_range()
    feat["atr_ratio"] = feat["atr_14"] / feat["close"]  # normalized ATR

    bb = BollingerBands(close=feat["close"], window=20, window_dev=2)
    feat["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    feat["bb_pct"] = (feat["close"] - bb.bollinger_lband()) / (
        bb.bollinger_hband() - bb.bollinger_lband() + 1e-10
    )

    adx = ADXIndicator(high=feat["high"], low=feat["low"], close=feat["close"], window=14)
    feat["adx_14"] = adx.adx()

    # ──────── 5. MICROSTRUCTURE ────────
    feat["vol_ma_20"] = feat["volume"].rolling(window=20).mean()
    feat["volume_ratio"] = feat["volume"] / (feat["vol_ma_20"] + 1e-10)
    feat["range_pct"] = (feat["high"] - feat["low"]) / feat["close"]
    feat["body_ratio"] = abs(feat["close"] - feat["open"]) / (feat["high"] - feat["low"] + 1e-10)
    feat["upper_shadow"] = (feat["high"] - feat[["close", "open"]].max(axis=1)) / (
        feat["high"] - feat["low"] + 1e-10
    )
    feat["lower_shadow"] = (feat[["close", "open"]].min(axis=1) - feat["low"]) / (
        feat["high"] - feat["low"] + 1e-10
    )

    # ──────── 6. INTERACTION FEATURES ────────
    feat["mom_x_vol"] = feat["momentum_24"] * feat["realized_vol_24"]
    feat["rsi_x_adx"] = (feat["rsi_14"] - 50) / 50 * feat["adx_14"] / 100
    feat["trend_strength"] = feat["adx_14"] * np.sign(feat["momentum_24"])

    # ──────── 7. LAG FEATURES ────────
    for lag in [1, 2, 3, 5, 10]:
        feat[f"return_lag_{lag}"] = feat["returns"].shift(lag)
        feat[f"vol_lag_{lag}"] = feat["realized_vol_24"].shift(lag)

    # ──────── 8. CROSS-PAIR CORRELATION ────────
    if cross_pair_data:
        feat = feat[~feat.index.duplicated(keep="last")]
        own_ret = feat["returns"]
        for pair_name, pair_df in cross_pair_data.items():
            if pair_df is None or pair_df.empty:
                continue
            # Align indices safely by ensuring unique labels
            pair_df = pair_df[~pair_df.index.duplicated(keep="last")]
            pair_ret = pair_df["close"].pct_change().reindex(feat.index)
            # Rolling correlation
            corr = own_ret.rolling(48, min_periods=24).corr(pair_ret)
            feat[f"corr_{pair_name}"] = corr
            # Rolling beta
            cov = own_ret.rolling(48, min_periods=24).cov(pair_ret)
            var = pair_ret.rolling(48, min_periods=24).var()
            feat[f"beta_{pair_name}"] = cov / (var + 1e-10)

    # ──────── 9. SENTIMENT ────────
    feat["sentiment"] = sentiment

    # ──────── DROP NaN ────────
    feat.dropna(inplace=True)

    logger.info(f"Feature engineering: {len(feat)} rows, {len(feat.columns)} features")
    return feat


def get_feature_columns(cross_pairs: list | None = None) -> list:
    """Return the list of feature column names used by the model."""
    cols = [
        # Economic factors
        "momentum_12", "momentum_24", "momentum_48", "momentum_168",
        "momentum_crossover",
        "value_deviation", "value_zscore", "mr_speed",
        "carry_proxy",
        # Volatility
        "realized_vol_12", "realized_vol_24", "realized_vol_48", "realized_vol_168",
        "vol_ratio", "vol_regime_change",
        "skew_24", "kurt_24", "skew_168",
        # Technical
        "rsi_14", "macd", "macd_signal", "macd_hist", "macd_hist_accel",
        "atr_14", "atr_ratio", "bb_width", "bb_pct", "adx_14",
        # Microstructure
        "volume_ratio", "range_pct", "body_ratio", "upper_shadow", "lower_shadow",
        # Interactions
        "mom_x_vol", "rsi_x_adx", "trend_strength",
        # Lags
        "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5", "return_lag_10",
        "vol_lag_1", "vol_lag_2", "vol_lag_3", "vol_lag_5", "vol_lag_10",
        # Sentiment
        "sentiment",
    ]
    # Cross-pair columns
    if cross_pairs:
        for p in cross_pairs:
            cols.extend([f"corr_{p}", f"beta_{p}"])
    return cols
