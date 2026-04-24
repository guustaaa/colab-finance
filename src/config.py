"""
config.py — Central configuration for the trading system.

All tunable parameters in one place. No magic numbers buried in code.
Based on institutional best practices from AQR, Menkhoff et al.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# OANDA API
# ─────────────────────────────────────────────
OANDA_TOKEN = os.getenv("OANDA_ACCESS_TOKEN", "")
OANDA_ACCOUNT = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENV = os.getenv("OANDA_ENVIRONMENT", "practice")

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

# Number of candles for training history
TRAINING_HISTORY_COUNT = 4000  # ~6 months of H1 data

# ─────────────────────────────────────────────
# RISK MANAGEMENT
# Based on Kelly Criterion with fractional sizing
# ─────────────────────────────────────────────
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PCT", "0.01"))  # 1% of account
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN_PCT", "0.10"))     # 10% max drawdown → halt

# ATR-based stop loss / take profit multipliers
ATR_SL_MULTIPLIER = 1.5   # Stop loss at 1.5x ATR
ATR_TP_MULTIPLIER = 2.5   # Take profit at 2.5x ATR (risk:reward = 1:1.67)

# Minimum expected profit (in pips) after spread to take a trade
MIN_EDGE_AFTER_COSTS = 3.0  # pips — filters out low-conviction signals

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
# In practice, we use the swap rate from OANDA (overnight financing charges)
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
# ─────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,     # L1 regularization — prevents overfitting on noise
    "reg_lambda": 1.0,    # L2 regularization
    "random_state": 42,
    "eval_metric": "logloss",
}

# Walk-forward validation
WALK_FORWARD_TRAIN_SIZE = 2000  # candles
WALK_FORWARD_TEST_SIZE = 200    # candles (~8 days of H1)
WALK_FORWARD_STEP = 200         # retrain every 200 candles

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
