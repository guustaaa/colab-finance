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
    """Always leaves the repo at the exact latest commit on origin/main."""
    os.chdir("/content")

    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        # Force-sync: fetch then hard-reset (handles dirty state, conflicts, etc.)
        subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], capture_output=True)
        result = subprocess.run(
            ["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ Repository updated (git reset --hard origin/main).")
            return
        print(f"⚠️  Reset failed ({result.stderr.strip()}). Re-cloning...")

    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)

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

# Flush any stale src.* modules cached from a previous cell run
for _k in [k for k in sys.modules if k.startswith("src")]:
    del sys.modules[_k]
import importlib; importlib.invalidate_caches()

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
    HMM_RETRAIN_INTERVAL, BULK_HISTORY_YEARS,
    DRIVE_DATA_DIR, DRIVE_MODELS_DIR, COMPUTE_DEVICE,
)
from src.data_fetcher import CapitalFetcher
from src.features import compute_all_features
from src.sentiment import SentimentScanner
from src.regime import RegimeDetector
from src.ensemble import EnsembleEngine
from src.execution import RiskManager, CapitalExecutor
from src.utils import Notifier, StateManager, TradeJournal, setup_logger

# Setup logging — force-clear stale handlers from previous cell runs
log_dir = os.path.join(STATE_DIR, "logs")
for _name in ["colab_master", "data_fetcher", "regime", "ensemble", "notifier", "execution"]:
    _lg = logging.getLogger(_name)
    _lg.handlers.clear()
    _lg.propagate = False
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


N_CORES = multiprocessing.cpu_count()

# ── Device banner ──
# To force a device, add a Colab Secret or set before running:
#   COMPUTE_DEVICE = cpu | gpu | tpu
# Auto-detection: checks nvidia-smi for GPU, else CPU.
# NOTE: TPU v5e → XGBoost/LGB run on CPU (they don't support TPU).
#       Set COMPUTE_DEVICE=tpu to get 2000-tree deeper models on CPU.
print(f"💻  CPU cores : {N_CORES}")
print(f"⚙️  Device    : {COMPUTE_DEVICE.upper()}")
if COMPUTE_DEVICE == "tpu":
    print("   ℹ️  TPU detected — XGBoost/LGB use maxed CPU config (2000 trees).")
    print("      JAX-based models will use TPU in a future update.")
elif COMPUTE_DEVICE == "gpu":
    print("   🚀 GPU detected — XGBoost CUDA + LightGBM GPU enabled!")
else:
    print("   ℹ️  CPU mode — all cores active via n_jobs=-1.")

# ── STEP 1: Bulk data fetch ──
# Nuke truly poisoned caches (< 100 candles = from 429 failures)
import glob as _glob
for _pq in _glob.glob(os.path.join(DRIVE_DATA_DIR, "*_H1_2y.parquet")):
    try:
        _tmp = pd.read_parquet(_pq)
        if len(_tmp) < 100:
            os.remove(_pq)
            print(f"  🗑️  Deleted poisoned cache: {os.path.basename(_pq)} ({len(_tmp)} candles)")
    except Exception:
        os.remove(_pq)

print("\n📥 Fetching 2 years of history per pair (sequential, cached to Drive)...")
_t0 = time.time()

raw_data = {}
for _inst in INSTRUMENTS:
    _cache = os.path.join(DRIVE_DATA_DIR, f"{_inst}_H1_2y.parquet")
    _df = fetcher.fetch_bulk_history(
        instrument=_inst,
        years=BULK_HISTORY_YEARS,
        granularity=TRADING_GRANULARITY,
        cache_path=_cache,
    )
    if _df is not None and len(_df) >= 200:
        raw_data[_inst] = _df
        _tag = f"{_df.index[0].date()} → {_df.index[-1].date()}"
        _icon = "✅" if len(_df) >= 500 else "⚠️ "
        print(f"  {_icon} {_inst}: {len(_df):,} candles ({_tag})")
    else:
        print(f"  ❌ {_inst}: no usable data")
    time.sleep(2)  # 2s pause between pairs to fully reset rate limit

print(f"\n📥 Data fetch done in {time.time() - _t0:.0f}s")
notifier.send(f"📥 Bulk data loaded: {len(raw_data)}/{len(INSTRUMENTS)} pairs ready", "info")

# ── STEP 2: Parallel training with Drive checkpointing ──
print("\n🧠 Starting parallel training (this will saturate all CPU cores)...")
print("   Models are saved to Drive every 5 minutes.\n")

_train_start = time.time()
_checkpoint_interval = 300  # 5 minutes

def _train_one_instrument(inst, df, sentiment_score, all_raw_data):
    """
    Full training pipeline for one instrument.
    Returns (inst, ens, metrics) or (inst, None, error_msg) on failure.
    """
    from src.features import compute_all_features
    from src.regime import RegimeDetector
    from src.ensemble import EnsembleEngine

    try:
        # Build cross-pair data (all pairs except this one)
        cross_pair_data = {k: v for k, v in all_raw_data.items() if k != inst}
        other_pairs = [k for k in all_raw_data if k != inst]

        # Feature engineering with cross-pair correlations
        features = compute_all_features(df, sentiment=sentiment_score, cross_pair_data=cross_pair_data)
        if features.empty:
            return inst, None, f"feature engineering failed on {len(df)} candles"

        # Fit HMM regime detector
        hmm_path = os.path.join(DRIVE_MODELS_DIR, f"hmm_{inst}.joblib")
        rd = RegimeDetector()
        rd.fit(df, model_path=hmm_path)

        # Train ensemble (XGB + LGB + DNN stacker) with cross-pair awareness
        ens_path = os.path.join(DRIVE_MODELS_DIR, f"ensemble_{inst}.joblib")
        ens = EnsembleEngine(model_path=ens_path, cross_pairs=other_pairs)
        metrics = ens.train(features)

        # Persist to Drive
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
        ex.submit(_train_one_instrument, inst, raw_data[inst], _sentiment, raw_data): inst
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
            _loaded = _joblib.load(hmm_path)
            if isinstance(_loaded, RegimeDetector):
                regime_detectors[inst] = _loaded
            else:
                # Stale file (raw GaussianHMM from old code) — discard it
                logger.warning(f"{inst}: stale HMM file (type={type(_loaded).__name__}), re-creating")
                os.remove(hmm_path)
                regime_detectors[inst] = RegimeDetector()
        except Exception:
            regime_detectors[inst] = RegimeDetector()

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
    start_time   = datetime.now()
    end_time     = start_time + timedelta(hours=MAX_RUNTIME_HOURS)
    candle_count = 0
    trade_count  = 0
    _last_save          = time.time()
    _last_balance_check = time.time()


    notifier.send(
        f"🟢 Live trading started. Will run until {end_time.strftime('%H:%M:%S')} "
        f"({MAX_RUNTIME_HOURS}h). Pairs: {', '.join(INSTRUMENTS)}",
        "info"
    )

    while datetime.now() < end_time:
        cycle_start = time.time()
        candle_count += 1

        try:
            # Update balance every 60s (not every cycle — saves API quota)
            if time.time() - _last_balance_check >= 60:
                balance = fetcher.get_account_balance()
                if balance and balance > 0:
                    risk_mgr.update_balance(balance)
                _last_balance_check = time.time()

            if risk_mgr.is_halted:
                notifier.send("🚨 Trading HALTED — max drawdown breached!", "error")
                break

            # Sentiment is cheap — get it once per cycle
            sentiment = scanner.get_composite_score()

            # Fetch all pairs' candles once per cycle (for cross-pair features)
            _cycle_data = {}
            for _ci in INSTRUMENTS:
                _cdf = fetcher.fetch_candles(_ci, count=300, granularity=TRADING_GRANULARITY)
                if _cdf is not None and len(_cdf) >= 200:
                    _cycle_data[_ci] = _cdf

            for inst in INSTRUMENTS:
                try:
                    if executor.has_open_position(inst):
                        logger.info(f"{inst}: position open — skipping")
                        continue

                    df = _cycle_data.get(inst)
                    if df is None:
                        continue

                    # Cross-pair data for correlation features
                    cross_data = {k: v for k, v in _cycle_data.items() if k != inst}
                    features = compute_all_features(df, sentiment=sentiment, cross_pair_data=cross_data)
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

            # Drive save every 5 minutes wall-clock
            if time.time() - _last_save >= 300:
                stats = journal.get_performance_stats()
                notifier.send(
                    f"📈 {candle_count} cycles | trades={trade_count} | "
                    f"WR={stats.get('win_rate',0):.1%} | "
                    f"PnL=${stats.get('total_pnl',0):.2f}",
                    "info"
                )
                try:
                    state.save()
                    _last_save = time.time()
                except Exception:
                    pass


        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            notifier.send(f"🚨 Loop error: {e}", "error")

        elapsed = time.time() - cycle_start
        logger.info(f"Cycle {candle_count} done ({elapsed:.1f}s).")
        # No sleep — run continuously. Drive save happens on wall-clock timer.


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
