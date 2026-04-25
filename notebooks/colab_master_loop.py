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

# Required secrets: CAPITAL_API_KEY, CAPITAL_EMAIL, CAPITAL_PASSWORD
# Optional secret:  WEBHOOK_URL (for Discord/Slack notifications)
_missing_secrets = []
for secret_name in ["CAPITAL_API_KEY", "CAPITAL_EMAIL", "CAPITAL_PASSWORD", "WEBHOOK_URL"]:
    try:
        val = userdata.get(secret_name)
        if val:
            os.environ[secret_name] = val
            print(f"✅ Secret loaded: {secret_name}")
        else:
            print(f"⚠️  Secret '{secret_name}' is empty.")
            if secret_name != "WEBHOOK_URL":
                _missing_secrets.append(secret_name)
    except userdata.SecretNotFoundError:
        if secret_name == "WEBHOOK_URL":
            print(f"ℹ️  Optional secret '{secret_name}' not set (notifications disabled).")
        else:
            print(f"❌ REQUIRED secret '{secret_name}' not found!")
            _missing_secrets.append(secret_name)

if _missing_secrets:
    raise RuntimeError(
        f"❌ Cannot start: missing required secrets: {_missing_secrets}\n"
        f"   Add them via the 🔑 Secrets panel in the left sidebar.\n\n"
        f"   CAPITAL_API_KEY  → Go to capital.com → Settings → API integrations → Generate new key\n"
        f"   CAPITAL_EMAIL    → Your Capital.com login email\n"
        f"   CAPITAL_PASSWORD → The CUSTOM PASSWORD you set when generating the API key\n"
        f"                      (NOT your account login password!)"
    )

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
from src.data_fetcher import CapitalFetcher
from src.features import compute_all_features
from src.sentiment import SentimentScanner
from src.regime import RegimeDetector
from src.ensemble import EnsembleEngine
from src.execution import RiskManager, CapitalExecutor
from src.utils import Notifier, StateManager, TradeJournal, setup_logger

# Setup logging
log_dir = os.path.join(STATE_DIR, "logs")
logger = setup_logger("colab_master", log_dir)

# Initialize all components
notifier = Notifier(WEBHOOK_URL)
state = StateManager()
fetcher = CapitalFetcher(demo=True)
scanner = SentimentScanner()
executor = CapitalExecutor(client=fetcher.client)

# ─────────────────────────────────────────────
# Pre-flight: verify Capital.com credentials work
# ─────────────────────────────────────────────
print("\n🔍 Verifying Capital.com credentials...")
_balance = fetcher.get_account_balance()
if _balance is None or _balance <= 0:
    raise RuntimeError(
        "❌ Capital.com authentication FAILED. The bot will NOT start.\n\n"
        "   Check your Colab Secrets (🔑 left sidebar):\n"
        f"   CAPITAL_API_KEY  = '{os.environ.get('CAPITAL_API_KEY', '')[:12]}...'\n"
        f"   CAPITAL_EMAIL    = '{os.environ.get('CAPITAL_EMAIL', '')}'\n\n"
        "   Common fixes:\n"
        "   1. CAPITAL_PASSWORD must be the CUSTOM API KEY password, NOT your login password\n"
        "   2. Make sure 2FA is enabled on your Capital.com account\n"
        "   3. Check if the API key is active (not paused/expired) in Settings → API integrations"
    )
print(f"✅ Capital.com connected! Account balance: ${_balance:,.2f}")

risk_mgr = RiskManager(initial_balance=_balance)
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
# CELL 3: HEAVY TRAINING — Bulk Data Fetch + Full Model Training
# ─────────────────────────────────────────────
#
# This cell runs ONCE at startup and eats up all available Colab compute.
# Strategy:
#   1. Bulk-fetch 2 years of H1 data per pair (paginated, cached to Drive)
#   2. Train HMM + XGBoost/LightGBM ensemble on each pair in parallel
#   3. Checkpoint models to Drive every 5 minutes
#   4. When done → hand off to lean live trading loop
#
import multiprocessing
import concurrent.futures
from datetime import timezone

from src.config import (
    INSTRUMENTS, TRADING_GRANULARITY, TRAINING_HISTORY_COUNT,
    MAX_RUNTIME_HOURS, POLL_INTERVAL_SECONDS, WEBHOOK_URL,
    HMM_RETRAIN_INTERVAL, BULK_HISTORY_YEARS,
    DRIVE_DATA_DIR, DRIVE_MODELS_DIR,
)

N_CORES = multiprocessing.cpu_count()
print(f"💻 Colab has {N_CORES} CPU cores — using all of them.")

# ── STEP 1: Bulk data fetch for every pair in parallel ──
print("\n📥 Fetching 2 years of history per pair (cached to Drive)...")
_t0 = time.time()

def _fetch_pair(inst):
    cache_path = os.path.join(DRIVE_DATA_DIR, f"{inst}_H1_2y.parquet")
    df = fetcher.fetch_bulk_history(
        instrument=inst,
        years=BULK_HISTORY_YEARS,
        granularity=TRADING_GRANULARITY,
        cache_path=cache_path,
    )
    return inst, df

raw_data = {}
# Use threads (not processes) — CapitalClient session is not fork-safe
with concurrent.futures.ThreadPoolExecutor(max_workers=min(N_CORES, len(INSTRUMENTS))) as ex:
    futures = {ex.submit(_fetch_pair, inst): inst for inst in INSTRUMENTS}
    for fut in concurrent.futures.as_completed(futures):
        inst, df = fut.result()
        if df is not None and len(df) >= 2000:
            raw_data[inst] = df
            print(f"  ✅ {inst}: {len(df):,} candles ({df.index[0].date()} → {df.index[-1].date()})")
        else:
            n = len(df) if df is not None else 0
            print(f"  ⚠️  {inst}: only {n} candles — using what we have")
            if df is not None and len(df) >= 300:
                raw_data[inst] = df

print(f"\n📥 Data fetch done in {time.time() - _t0:.0f}s")
notifier.send(f"📥 Bulk data loaded: {len(raw_data)}/{len(INSTRUMENTS)} pairs ready", "info")

# ── STEP 2: Parallel training with Drive checkpointing ──
print("\n🧠 Starting parallel training (this will saturate all CPU cores)...")
print("   Models are saved to Drive every 5 minutes.\n")

_train_start = time.time()
_checkpoint_interval = 300  # 5 minutes

def _train_one_instrument(inst, df, sentiment_score):
    """
    Full training pipeline for one instrument.
    Returns (inst, metrics) or (inst, None) on failure.
    """
    from src.features import compute_all_features
    from src.regime import RegimeDetector
    from src.ensemble import EnsembleEngine

    try:
        # Feature engineering
        features = compute_all_features(df, sentiment=sentiment_score)
        if features.empty:
            return inst, None, f"feature engineering failed on {len(df)} candles"

        # Fit HMM regime detector
        hmm_path = os.path.join(DRIVE_MODELS_DIR, f"hmm_{inst}.joblib")
        rd = RegimeDetector()
        rd.fit(df, model_path=hmm_path)

        # Train ensemble (walk-forward XGB + LGB)
        ens_path = os.path.join(DRIVE_MODELS_DIR, f"ensemble_{inst}.joblib")
        ens = EnsembleEngine(model_path=ens_path)
        metrics = ens.train(features)

        # Persist to Drive immediately
        import joblib
        joblib.dump(rd,  hmm_path)
        joblib.dump(ens, ens_path)

        return inst, ens, metrics

    except Exception as e:
        import traceback
        return inst, None, traceback.format_exc()

# Get current sentiment once (shared across pairs)
_sentiment = scanner.get_composite_score()

# Train all pairs in parallel — ProcessPoolExecutor would be faster but
# scikit-learn/xgboost already use n_jobs=-1 internally, so threads suffice
_train_results = {}
_train_errors  = {}

with concurrent.futures.ThreadPoolExecutor(max_workers=len(INSTRUMENTS)) as ex:
    _futures = {
        ex.submit(_train_one_instrument, inst, raw_data[inst], _sentiment): inst
        for inst in raw_data
    }

    _completed = 0
    _next_checkpoint_log = time.time() + _checkpoint_interval

    for fut in concurrent.futures.as_completed(_futures):
        inst = _futures[fut]
        inst_out, ens_obj, metrics_or_err = fut.result()
        _completed += 1

        if ens_obj is not None:
            # Plug trained objects into the live runtime
            ensembles[inst_out]          = ens_obj
            _train_results[inst_out]     = metrics_or_err
            wf_acc = metrics_or_err.get("mean_wf_accuracy", 0)
            n_samp = metrics_or_err.get("n_train_samples", 0)
            print(f"  ✅ {inst_out}: WF accuracy={wf_acc:.4f} | n_samples={n_samp:,} "
                  f"({_completed}/{len(raw_data)} done)")
            logger.info(f"{inst_out} trained: acc={wf_acc:.4f} n={n_samp}")
        else:
            _train_errors[inst_out] = metrics_or_err
            print(f"  ❌ {inst_out} FAILED: {str(metrics_or_err)[:120]}")
            logger.error(f"{inst_out} training failed: {metrics_or_err}")

        # Periodic checkpoint log
        if time.time() > _next_checkpoint_log:
            elapsed = time.time() - _train_start
            print(f"\n  ⏱  Checkpoint: {_completed}/{len(raw_data)} pairs done | "
                  f"elapsed={elapsed:.0f}s | {N_CORES} cores running\n")
            _next_checkpoint_log = time.time() + _checkpoint_interval

_total_train_time = time.time() - _train_start
_summary_lines = []
for inst, m in _train_results.items():
    _summary_lines.append(
        f"  {inst}: acc={m.get('mean_wf_accuracy',0):.4f} | "
        f"n={m.get('n_train_samples',0):,} | "
        f"top={[f[0] for f in m.get('top_features',[])[:2]]}"
    )
for inst, err in _train_errors.items():
    _summary_lines.append(f"  {inst}: FAILED")

_summary = "\n".join(_summary_lines)
print(f"\n✅ Heavy training complete in {_total_train_time:.0f}s\n{_summary}")
notifier.send(
    f"🧠 Training done ({_total_train_time:.0f}s | {len(_train_results)}/{len(raw_data)} pairs)\n{_summary}",
    "info"
)

# Reload regime detectors from saved models (they were trained in subthreads)
import joblib as _joblib
for inst in INSTRUMENTS:
    hmm_path = os.path.join(DRIVE_MODELS_DIR, f"hmm_{inst}.joblib")
    if os.path.exists(hmm_path):
        try:
            regime_detectors[inst] = _joblib.load(hmm_path)
        except Exception:
            pass  # keep the default empty one

print("✅ All components ready. Starting live trading loop...\n")

# ─────────────────────────────────────────────
# CELL 4: Live Trading Loop (lean — models already trained)
# ─────────────────────────────────────────────
def live_trading_loop():
    """
    Lean live loop. Models are already trained; this just generates signals every 5 min.

    Every cycle:
      1. Fetch 300 recent candles per pair (fast, < 1s)
      2. Compute features + regime + ensemble signal
      3. Risk check → execute if approved
      4. Every 24 cycles: refresh HMM on recent data
      5. Every 60 cycles: performance report + save to Drive
    """
    start_time  = datetime.now()
    end_time    = start_time + timedelta(hours=MAX_RUNTIME_HOURS)
    candle_count = 0
    trade_count  = 0

    notifier.send(
        f"🟢 Live trading started. Will run until {end_time.strftime('%H:%M:%S')} "
        f"({MAX_RUNTIME_HOURS}h). Pairs: {', '.join(INSTRUMENTS)}",
        "info"
    )

    while datetime.now() < end_time:
        cycle_start = time.time()
        candle_count += 1

        try:
            # Update balance
            balance = fetcher.get_account_balance()
            if balance and balance > 0:
                risk_mgr.update_balance(balance)

            if risk_mgr.is_halted:
                notifier.send("🚨 Trading HALTED — max drawdown breached!", "error")
                break

            # Sentiment is cheap — get it once per cycle
            sentiment = scanner.get_composite_score()

            for inst in INSTRUMENTS:
                try:
                    if executor.has_open_position(inst):
                        logger.info(f"{inst}: position open — skipping")
                        continue

                    # Fetch recent 300 candles for live features
                    df = fetcher.fetch_candles(inst, count=300, granularity=TRADING_GRANULARITY)
                    if df is None or len(df) < 250:
                        continue

                    features = compute_all_features(df, sentiment=sentiment)
                    if features.empty:
                        continue

                    regime  = regime_detectors[inst].detect(df)
                    weights = regime_detectors[inst].get_strategy_weights(regime)
                    pred    = ensembles[inst].predict(features, regime, weights)

                    signal     = pred["signal"]
                    confidence = pred["confidence"]

                    if signal == "HOLD":
                        logger.info(f"{inst}: HOLD | conf={confidence:.4f} | {regime['label']}")
                        continue

                    atr    = float(features["atr_14"].iloc[-1])
                    spread = fetcher.get_spread(inst)
                    approved, reason = risk_mgr.should_trade(confidence, atr, spread)

                    if not approved:
                        logger.info(f"{inst}: rejected — {reason}")
                        continue

                    price   = float(features["close"].iloc[-1])
                    stats   = journal.get_performance_stats()
                    wr      = stats.get("win_rate", 0.52) or 0.52
                    avg_wl  = (abs(stats.get("avg_win", 1.5)) / abs(stats.get("avg_loss", 1.0))
                               if stats.get("avg_loss") else 1.5)

                    units = risk_mgr.calculate_position_size(
                        balance=balance or 1000,
                        atr=atr,
                        price=price,
                        win_rate=wr,
                        avg_win_loss_ratio=max(avg_wl, 0.5),
                        position_scale=pred["position_scale"],
                    )
                    if units <= 0:
                        continue

                    result = executor.execute_market_order(
                        instrument=inst, size=units, signal=signal,
                        price=price, atr=atr,
                    )

                    if "error" not in result:
                        trade_count += 1
                        journal.log_trade({
                            "instrument": inst, "signal": signal, "price": price,
                            "units": units, "atr": atr, "confidence": confidence,
                            "regime": regime["label"],
                            "stop_loss":  result.get("stop_loss"),
                            "take_profit": result.get("take_profit"),
                            "ensemble_score": pred["ensemble_score"],
                        })
                        notifier.send(
                            f"💹 {signal} {units:.2f}x {inst} @ {price:.5f} | "
                            f"SL={result.get('stop_loss','?')} TP={result.get('take_profit','?')} | "
                            f"conf={confidence:.2%} | {regime['label']}",
                            "trade"
                        )

                except Exception as e:
                    logger.error(f"{inst} error: {e}", exc_info=True)

            # Periodic light HMM refresh (only HMM, not XGB — fast)
            if candle_count % HMM_RETRAIN_INTERVAL == 0:
                logger.info("Light HMM refresh on recent data...")
                for inst in INSTRUMENTS:
                    try:
                        df_r = fetcher.fetch_candles(inst, count=600, granularity=TRADING_GRANULARITY)
                        if df_r is not None and len(df_r) >= 500:
                            regime_detectors[inst].fit(df_r)
                    except Exception:
                        pass

            # Performance + Drive save every ~5 hours
            if candle_count % 60 == 0:
                stats = journal.get_performance_stats()
                notifier.send(
                    f"📈 {candle_count} cycles | trades={trade_count} | "
                    f"WR={stats.get('win_rate',0):.1%} | "
                    f"PnL=${stats.get('total_pnl',0):.2f} | "
                    f"MaxDD=${stats.get('max_drawdown',0):.2f}",
                    "info"
                )
                # Persist journal to Drive
                try:
                    state.save()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            notifier.send(f"🚨 Loop error: {e}", "error")

        elapsed    = time.time() - cycle_start
        sleep_time = max(0, POLL_INTERVAL_SECONDS - elapsed)
        logger.info(f"Cycle {candle_count} done ({elapsed:.1f}s). Sleeping {sleep_time:.0f}s...")
        time.sleep(sleep_time)

    # ── Graceful shutdown ──
    logger.info("="*60 + "\nGRACEFUL SHUTDOWN\n" + "="*60)
    stats = journal.get_performance_stats()
    msg   = (
        f"⏹️ Shutdown after {MAX_RUNTIME_HOURS}h.\n"
        f"  Cycles: {candle_count} | Trades: {trade_count}\n"
        f"  Final stats: {stats}"
    )
    notifier.send(msg, "info")
    logger.info(msg)
    try:
        state.save()
    except Exception:
        pass


# START
live_trading_loop()
print("\n✅ Session complete. Restart notebook to begin a new session.")
