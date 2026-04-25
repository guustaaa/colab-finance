"""
config.py — Central configuration for the trading system.

All tunable parameters in one place. No magic numbers buried in code.
Based on institutional best practices from AQR, Menkhoff et al.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# CAPITAL.COM API
# ─────────────────────────────────────────────
CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY", "")
CAPITAL_EMAIL = os.getenv("CAPITAL_EMAIL", "")
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD", "")
CAPITAL_DEMO = os.getenv("CAPITAL_DEMO", "true").lower() == "true"

# ─────────────────────────────────────────────
# INSTRUMENTS
# ─────────────────────────────────────────────
# G10 major pairs — highest liquidity, tightest spreads,
# most studied in academic literature (Menkhoff 2012, AQR)
INSTRUMENTS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
    "USD_CAD",
    "NZD_USD",
    "USD_CHF",
]

# Default granularity for the trading loop
TRADING_GRANULARITY = "H1"  # 1-hour candles — best noise/signal tradeoff for retail
TRAINING_GRANULARITY = "H1"

# Number of candles for live-loop feature computation (recent window)
TRAINING_HISTORY_COUNT = 1000  # candles used in live loop for signal generation

# Years of history to pull for initial training (bulk fetch)
BULK_HISTORY_YEARS = 2.0  # ~17,500 H1 candles per pair

# ─────────────────────────────────────────────
# RISK MANAGEMENT
# Based on Kelly Criterion with fractional sizing
# ─────────────────────────────────────────────
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
RISK_PER_TRADE = float(os.getenv("MAX_RISK_PCT", "0.01"))  # 1% of account
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN_PCT", "0.10"))  # 10% max drawdown → halt

# ATR-based stop loss / take profit multipliers
SL_ATR_MULTIPLIER = 1.5   # Stop loss at 1.5x ATR
TP_ATR_MULTIPLIER = 2.5   # Take profit at 2.5x ATR (risk:reward = 1:1.67)

# Minimum confidence to execute a trade
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE", "0.55"))

# ─────────────────────────────────────────────
# FACTOR MODEL PARAMETERS
# Based on academic evidence from:
# - Asness, Moskowitz, Pedersen (2013) "Value & Momentum Everywhere"
# - Menkhoff et al. (2012) "Currency Momentum Strategies"
# ─────────────────────────────────────────────

# Momentum lookback periods (in candles at TRADING_GRANULARITY)
MOMENTUM_FAST_PERIOD = 24    # 1 day of H1
MOMENTUM_SLOW_PERIOD = 168   # 1 week of H1
MOMENTUM_SIGNAL_PERIOD = 720 # 1 month of H1

# Carry proxy — interest rate differential approximation
# In practice, we use the swap rate from Capital.com (overnight financing charges)
# For now, approximated from forward-spot differential

# Value — PPP deviation
# We use a long-term moving average as a PPP proxy (200-period ~ 8 days on H1)
VALUE_MA_PERIOD = 200

# Volatility targeting period
VOLATILITY_LOOKBACK = 24  # 1 day rolling vol

# ─────────────────────────────────────────────
# HMM REGIME DETECTION
# Based on Hamilton (1989), refined for FX in:
# Ang & Bekaert (2002) "Regime Switches in Interest Rates"
# ─────────────────────────────────────────────
HMM_N_STATES = 3          # low-vol trending, high-vol trending, crisis/choppy
HMM_LOOKBACK = 500        # candles of data to fit HMM on
HMM_RETRAIN_INTERVAL = 24 # retrain every 24 candles (1 day on H1)

# ─────────────────────────────────────────────
# XGBOOST META-LEARNER
# Tuned for max Colab CPU utilisation (all cores, large ensembles)
# ─────────────────────────────────────────────
import os as _os
_N_JOBS = int(_os.getenv("N_JOBS", "-1"))  # -1 = use all CPU cores

XGB_PARAMS = {
    "n_estimators": 1000,       # 5x more trees — better averaging of weak learners
    "learning_rate": 0.01,      # lower LR → needs more trees but generalises better
    "max_depth": 6,             # deeper trees for complex FX patterns
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 3,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "random_state": 42,
    "eval_metric": "logloss",
    "n_jobs": _N_JOBS,          # saturate all CPU cores
    "tree_method": "hist",      # fastest CPU method
    "early_stopping_rounds": 50,  # stop when val loss stops improving
}

LGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_samples": 20,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "random_state": 42,
    "n_jobs": _N_JOBS,
    "verbose": -1,
}

# Walk-forward validation — much larger windows now we have 2y of data
WALK_FORWARD_TRAIN_SIZE = 10000  # ~14 months of H1
WALK_FORWARD_TEST_SIZE  = 500    # ~3 weeks
WALK_FORWARD_STEP       = 500    # retrain every 3 weeks of data

# ─────────────────────────────────────────────
# SENTIMENT
# ─────────────────────────────────────────────
SENTIMENT_KEYWORDS = [
    "trump", "musk", "fed", "fomc", "powell", "lagarde", "boj",
    "inflation", "cpi", "ppi", "nfp", "gdp", "unemployment",
    "rate hike", "rate cut", "hawkish", "dovish",
    "war", "sanctions", "tariff", "trade war",
    "usd", "eur", "gbp", "jpy", "aud", "cad", "nzd", "chf",
    "recession", "default", "banking crisis",
]

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD=X",
    "https://www.forexlive.com/feed/news",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "https://feeds.reuters.com/reuters/businessNews",
]

# ─────────────────────────────────────────────
# NOTIFICATIONS
# ─────────────────────────────────────────────
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# ─────────────────────────────────────────────
# COLAB RUNTIME
# ─────────────────────────────────────────────
MAX_RUNTIME_HOURS = 11       # Graceful shutdown before Colab kills us
POLL_INTERVAL_SECONDS = 300  # 5 minutes between live checks (match H1 candle buildup)

# ─────────────────────────────────────────────
# GOOGLE DRIVE STATE PATHS (when running in Colab)
# ─────────────────────────────────────────────
DRIVE_STATE_DIR = "/content/drive/MyDrive/ForexAI_State"
DRIVE_MODELS_DIR = f"{DRIVE_STATE_DIR}/models"
DRIVE_LOGS_DIR = f"{DRIVE_STATE_DIR}/logs"
DRIVE_DATA_DIR = f"{DRIVE_STATE_DIR}/data"
