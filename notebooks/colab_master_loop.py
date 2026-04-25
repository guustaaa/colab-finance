# ============================================================================
# COLAB MASTER LOOP — Production Forex AI Trading System
# ============================================================================
#
# This script is designed to run entirely in Google Colab.
# It implements the full trading pipeline:
#
#   1. Mount Google Drive → load/save state
#   2. Pull latest code from GitHub
#   3. Daily retraining (HMM regime + XGBoost ensemble)
#   4. Continuous live trading loop (11 hours)
#   5. Graceful shutdown → save state → notify
#
# ARCHITECTURE RATIONALE:
# -----------------------
# Instead of using SOTA benchmark-winning models (PatchTST, N-HiTS, Chronos),
# this system uses strategies PROVEN profitable in real-world live trading:
#
# 1. ECONOMIC FACTORS (Carry, Momentum, Value)
#    - AQR "Value & Momentum Everywhere" (Asness, Moskowitz, Pedersen 2013)
#    - Proven with real institutional money over decades
#
# 2. HMM REGIME DETECTION
#    - Hamilton (1989) regime switching
#    - Used to dynamically weight strategies (not for direct signals)
#
# 3. XGBOOST + LIGHTGBM META-LEARNER
#    - "Why tree-based models still outperform deep learning on tabular data"
#      (Grinsztajn et al. 2022)
#    - Walk-forward validation prevents overfitting
#
# 4. KELLY CRITERION POSITION SIZING
#    - Mathematically optimal betting (Kelly 1956)
#    - Using fractional (1/4) Kelly for safety
#
# HOW TO RUN:
# -----------
# 1. Open this file in Google Colab
# 2. Update REPO_URL below with your GitHub repo URL
# 3. Runtime → Run all
# 4. Accept the Google Drive mount prompt
# 5. The bot will trade for 11 hours, then gracefully shutdown
# ============================================================================

# ─────────────────────────────────────────────
# CELL 1: Setup Environment
# ─────────────────────────────────────────────
import os
import sys
import time
from datetime import datetime, timedelta

# Mount Google Drive for persistent state
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted.")

# Create state directories
STATE_DIR = "/content/drive/MyDrive/ForexAI_State"
for subdir in ["models", "logs", "data"]:
    os.makedirs(os.path.join(STATE_DIR, subdir), exist_ok=True)
print(f"✅ State directory ready: {STATE_DIR}")

import subprocess, shutil

REPO_URL = "https://github.com/guustaaa/colab-finance.git"
REPO_DIR = "/content/colab-finance"

def sync_repo():
    """Self-healing repo sync: always ends with a clean, up-to-date clone."""
    # Reset CWD — a previous run may have chdir'd into the repo dir we're about to delete
    os.chdir("/content")
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        # Directory exists with a valid .git — try to pull latest
        result = subprocess.run(
            ["git", "-C", REPO_DIR, "pull", "--ff-only"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ Repository updated (git pull).")
            return
        # Pull failed (dirty state, merge conflict, etc.) — nuke and re-clone
        print(f"⚠️  Pull failed ({result.stderr.strip()}). Re-cloning fresh...")

    # Remove any leftover directory (corrupted clone, partial download, etc.)
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)

    # Fresh clone
    result = subprocess.run(
        ["git", "clone", REPO_URL, REPO_DIR],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"❌ Git clone failed:\n{result.stderr}\n"
            f"Make sure the repo is public: {REPO_URL}"
        )
    print("✅ Repository cloned.")

sync_repo()

os.chdir(REPO_DIR)
os.system("pip install -r requirements.txt -q")
sys.path.insert(0, REPO_DIR)
print("✅ Dependencies installed.")

# ─────────────────────────────────────────────
# Load credentials from Colab Secrets (🔑 icon in left sidebar)
# ─────────────────────────────────────────────
from google.colab import userdata

# Required secrets: OANDA_ACCESS_TOKEN, OANDA_ACCOUNT_ID
# Optional secret:  WEBHOOK_URL (for Discord/Slack notifications)
for secret_name in ["OANDA_ACCESS_TOKEN", "OANDA_ACCOUNT_ID", "WEBHOOK_URL"]:
    try:
        val = userdata.get(secret_name)
        if val:
            os.environ[secret_name] = val
            print(f"✅ Secret loaded: {secret_name}")
        else:
            print(f"⚠️  Secret '{secret_name}' is empty.")
    except userdata.SecretNotFoundError:
        if secret_name == "WEBHOOK_URL":
            print(f"ℹ️  Optional secret '{secret_name}' not set (notifications disabled).")
        else:
            print(f"❌ REQUIRED secret '{secret_name}' not found! Add it via 🔑 Secrets in the left sidebar.")

# ─────────────────────────────────────────────
# CELL 2: Initialize Components
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import logging

from src.config import (
    INSTRUMENTS, TRADING_GRANULARITY, TRAINING_HISTORY_COUNT,
    MAX_RUNTIME_HOURS, POLL_INTERVAL_SECONDS, WEBHOOK_URL,
    HMM_RETRAIN_INTERVAL,
)
from src.data_fetcher import OandaFetcher
from src.features import compute_all_features
from src.sentiment import SentimentScanner
from src.regime import RegimeDetector
from src.ensemble import EnsembleEngine
from src.execution import RiskManager, OandaExecutor
from src.utils import Notifier, StateManager, TradeJournal, setup_logger

# Setup logging
log_dir = os.path.join(STATE_DIR, "logs")
logger = setup_logger("colab_master", log_dir)

# Initialize all components
notifier = Notifier(WEBHOOK_URL)
state = StateManager()
fetcher = OandaFetcher()
scanner = SentimentScanner()
executor = OandaExecutor()
risk_mgr = RiskManager(initial_balance=fetcher.get_account_balance() or 10000.0)
journal = TradeJournal(state)

# Per-instrument models
regime_detectors = {}
ensembles = {}

for inst in INSTRUMENTS:
    regime_detectors[inst] = RegimeDetector()
    ensembles[inst] = EnsembleEngine(
        model_path=state.model_path(f"ensemble_{inst}.joblib")
    )

notifier.send("🚀 Forex AI System initializing...", "info")
print("✅ All components initialized.")

# ─────────────────────────────────────────────
# CELL 3: Daily Retraining
# ─────────────────────────────────────────────
def daily_retrain():
    """
    Train the HMM regime detector and XGBoost ensemble for each instrument.
    Uses the latest available historical data from OANDA.
    """
    notifier.send("📊 Starting daily retraining routine...", "info")
    results = {}

    for inst in INSTRUMENTS:
        logger.info(f"\n{'='*40}\nRetraining {inst}\n{'='*40}")
        try:
            # Fetch historical data
            df = fetcher.fetch_candles(
                inst, count=TRAINING_HISTORY_COUNT, granularity=TRADING_GRANULARITY
            )

            if df is None or df.empty or len(df) < 500:
                logger.warning(f"Insufficient data for {inst}. Skipping.")
                results[inst] = "SKIPPED: insufficient data"
                continue

            # Fit regime detector
            hmm_path = state.model_path(f"hmm_{inst}.joblib")
            regime_detectors[inst].fit(df, model_path=hmm_path)

            # Compute features and train ensemble
            sentiment = scanner.get_composite_score()
            features = compute_all_features(df, sentiment=sentiment)

            if features.empty:
                logger.warning(f"Feature engineering failed for {inst}.")
                results[inst] = "SKIPPED: feature engineering failed"
                continue

            metrics = ensembles[inst].train(features)
            results[inst] = {
                "wf_accuracy": f"{metrics.get('mean_wf_accuracy', 0):.4f}",
                "n_samples": metrics.get("n_train_samples", 0),
                "top_features": [f[0] for f in metrics.get("top_features", [])[:3]],
            }

            logger.info(f"{inst} training complete: {results[inst]}")

        except Exception as e:
            logger.error(f"Training failed for {inst}: {e}", exc_info=True)
            results[inst] = f"ERROR: {str(e)}"

    # Send training summary
    summary = "\n".join([f"  {k}: {v}" for k, v in results.items()])
    notifier.send(f"📊 Retraining complete:\n{summary}", "info")
    return results

# Run the retraining
training_results = daily_retrain()
print("✅ Daily retraining complete.")

# ─────────────────────────────────────────────
# CELL 4: Live Trading Loop
# ─────────────────────────────────────────────
def live_trading_loop():
    """
    Main continuous trading loop.

    For each instrument, every POLL_INTERVAL_SECONDS:
      1. Fetch latest candle data
      2. Compute features + sentiment
      3. Detect current regime
      4. Generate ensemble signal
      5. Check risk management
      6. Execute trade if approved
      7. Log everything
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=MAX_RUNTIME_HOURS)
    candle_count = 0
    trade_count = 0

    notifier.send(
        f"🟢 Live trading started. Will run until {end_time.strftime('%H:%M:%S')} "
        f"({MAX_RUNTIME_HOURS}h). Instruments: {', '.join(INSTRUMENTS)}",
        "info"
    )

    while datetime.now() < end_time:
        cycle_start = time.time()
        candle_count += 1

        try:
            # Update account balance
            balance = fetcher.get_account_balance()
            if balance > 0:
                risk_mgr.update_balance(balance)

            if risk_mgr.is_halted:
                notifier.send("🚨 Trading HALTED — max drawdown breached!", "error")
                break

            # Get current sentiment (shared across all instruments)
            sentiment = scanner.get_composite_score()

            for inst in INSTRUMENTS:
                try:
                    # Skip if we already have an open position for this instrument
                    if executor.has_open_position(inst):
                        logger.info(f"{inst}: Position already open. Skipping.")
                        continue

                    # Fetch recent data (enough for feature computation)
                    df = fetcher.fetch_candles(
                        inst, count=300, granularity=TRADING_GRANULARITY
                    )
                    if df is None or df.empty or len(df) < 250:
                        continue

                    # Compute features
                    features = compute_all_features(df, sentiment=sentiment)
                    if features.empty:
                        continue

                    # Detect regime
                    regime = regime_detectors[inst].detect(df)
                    weights = regime_detectors[inst].get_strategy_weights(regime)

                    # Get ensemble signal
                    prediction = ensembles[inst].predict(features, regime, weights)
                    signal = prediction["signal"]
                    confidence = prediction["confidence"]

                    if signal == "HOLD":
                        logger.info(
                            f"{inst}: HOLD | conf={confidence:.4f} | regime={regime['label']}"
                        )
                        continue

                    # Risk management pre-trade check
                    atr = features["atr_14"].iloc[-1]
                    spread = fetcher.get_spread(inst)
                    approved, reason = risk_mgr.should_trade(confidence, atr, spread)

                    if not approved:
                        logger.info(f"{inst}: Trade rejected — {reason}")
                        continue

                    # Calculate position size
                    price = features["close"].iloc[-1]
                    stats = journal.get_performance_stats()
                    win_rate = stats.get("win_rate", 0.52)
                    avg_wl = abs(stats.get("avg_win", 1.5) / stats.get("avg_loss", -1.0)) if stats.get("avg_loss", 0) != 0 else 1.5

                    units = risk_mgr.calculate_position_size(
                        balance=balance or 10000,
                        atr=atr,
                        price=price,
                        win_rate=win_rate if win_rate > 0 else 0.52,
                        avg_win_loss_ratio=avg_wl if avg_wl > 0 else 1.5,
                        position_scale=prediction["position_scale"],
                    )

                    if units <= 0:
                        continue

                    # EXECUTE THE TRADE
                    result = executor.execute_market_order(
                        instrument=inst,
                        units=units,
                        signal=signal,
                        price=price,
                        atr=atr,
                    )

                    if "error" not in result:
                        trade_count += 1
                        journal.log_trade({
                            "instrument": inst,
                            "signal": signal,
                            "price": price,
                            "units": units,
                            "atr": atr,
                            "confidence": confidence,
                            "regime": regime["label"],
                            "stop_loss": result.get("stop_loss"),
                            "take_profit": result.get("take_profit"),
                            "ensemble_score": prediction["ensemble_score"],
                        })

                        notifier.send(
                            f"💹 {signal} {units} {inst} @ {price:.5f} | "
                            f"SL={result.get('stop_loss', 'N/A')} | "
                            f"TP={result.get('take_profit', 'N/A')} | "
                            f"Conf={confidence:.2%} | Regime={regime['label']}",
                            "trade"
                        )

                except Exception as e:
                    logger.error(f"Error processing {inst}: {e}", exc_info=True)

            # Periodic HMM retraining (every N candles)
            if candle_count % HMM_RETRAIN_INTERVAL == 0:
                logger.info("Periodic HMM retraining...")
                for inst in INSTRUMENTS:
                    try:
                        df = fetcher.fetch_candles(inst, count=600, granularity=TRADING_GRANULARITY)
                        if df is not None and len(df) >= 500:
                            regime_detectors[inst].fit(df)
                    except Exception:
                        pass

            # Periodic performance report
            if candle_count % 60 == 0:  # Every ~5 hours at 5-min polling
                stats = journal.get_performance_stats()
                notifier.send(
                    f"📈 Performance update:\n"
                    f"  Trades: {stats.get('total_trades', 0)} | "
                    f"Win rate: {stats.get('win_rate', 0):.1%} | "
                    f"PnL: ${stats.get('total_pnl', 0):.2f} | "
                    f"Max DD: ${stats.get('max_drawdown', 0):.2f}",
                    "info"
                )

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            notifier.send(f"🚨 Main loop error: {str(e)}", "error")

        # Sleep until next poll
        elapsed = time.time() - cycle_start
        sleep_time = max(0, POLL_INTERVAL_SECONDS - elapsed)
        logger.info(f"Cycle complete ({elapsed:.1f}s). Sleeping {sleep_time:.0f}s...")
        time.sleep(sleep_time)

    # ── GRACEFUL SHUTDOWN ──
    logger.info("="*60)
    logger.info("GRACEFUL SHUTDOWN — Runtime limit reached")
    logger.info("="*60)

    # Final performance report
    stats = journal.get_performance_stats()
    shutdown_msg = (
        f"⏹️ Graceful shutdown after {MAX_RUNTIME_HOURS}h.\n"
        f"  Candles processed: {candle_count}\n"
        f"  Trades executed: {trade_count}\n"
        f"  Final stats: {stats}"
    )
    notifier.send(shutdown_msg, "info")
    logger.info(shutdown_msg)


# START THE LOOP
live_trading_loop()
print("\n✅ Session complete. Restart this notebook to begin a new session.")
