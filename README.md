# 🧠 Forex AI — Research-Driven Automated Trading System

> **Philosophy:** We don't chase SOTA benchmarks. We use strategies proven profitable with real money.

## What This Is NOT
- ❌ Another "85% accurate" model that dies in live trading
- ❌ Deep learning for the sake of deep learning
- ❌ Overfit-to-backtest benchmark hunting

## What This IS
- ✅ Economic factor model (carry, momentum, value) — proven with institutional money
- ✅ HMM regime detection — adapts to market conditions dynamically
- ✅ XGBoost + LightGBM ensemble — best model for noisy tabular data (Grinsztajn et al. 2022)
- ✅ Walk-forward validation — no look-ahead bias
- ✅ Kelly Criterion position sizing — mathematically optimal betting
- ✅ ATR-based risk management — dynamic stop loss and take profit
- ✅ Transaction cost filtering — don't trade if edge < cost

## Architecture

```
OANDA Data → Feature Engineering → [Economic Factors + Technical + Sentiment]
                                              ↓
                                    HMM Regime Detector
                                              ↓
                              Regime-Weighted Ensemble Engine
                              (Factor Model + XGBoost + LightGBM)
                                              ↓
                              Risk Manager (Kelly + ATR + Drawdown)
                                              ↓
                              OANDA Execution (or HOLD)
                                              ↓
                              Trade Journal → Walk-Forward Retrainer
```

## Academic Foundation

| Component | Based On | Citation |
|-----------|----------|----------|
| Momentum Factor | Cross-sectional FX momentum | Asness, Moskowitz, Pedersen (2013) "Value & Momentum Everywhere" |
| Value Factor | PPP mean-reversion | Taylor (1995), Menkhoff et al. (2017) |
| Carry Factor | Interest rate differential | Lustig, Roussanov, Verdelhan (2011) |
| Regime Detection | Hidden Markov Models | Hamilton (1989), Ang & Bekaert (2002) |
| XGBoost superiority | Tree-based > DL on tabular | Grinsztajn et al. (2022) |
| Position Sizing | Kelly Criterion | Kelly (1956), Thorp (2006) |
| Walk-Forward | Realistic validation | White (2000) "Reality Check for Data Snooping" |

## Quick Start

### 1. Setup OANDA Demo Account
1. Go to [OANDA Demo](https://www.oanda.com/demo-account/)
2. Create a free practice account
3. Go to **Manage API Access** → generate a personal access token
4. Copy `.env.example` to `.env` and fill in your credentials

### 2. Run in Google Colab (Zero Local Hardware)
1. Open `notebooks/colab_master_loop.py` in Google Colab
2. Update `REPO_URL` with your GitHub repo URL
3. `Runtime → Run all`
4. Accept the Google Drive mount prompt
5. The bot trades for 11 hours, then gracefully shuts down

### 3. Run Tests Locally
```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

### 4. Run Backtest Locally
```python
from src.data_fetcher import generate_mock_data
from src.backtester import Backtester

df = generate_mock_data(n=3000)
bt = Backtester(initial_balance=10000)
metrics = bt.run(df)
print(metrics)
```

## Project Structure
```
├── src/
│   ├── config.py          # All tunable parameters (documented with citations)
│   ├── data_fetcher.py    # OANDA V20 API interface
│   ├── features.py        # Feature engineering (economic factors + technical)
│   ├── sentiment.py       # RSS news sentiment scanner (VADER)
│   ├── regime.py          # HMM regime detector (3-state)
│   ├── ensemble.py        # Factor Model + XGBoost + LightGBM ensemble
│   ├── execution.py       # OANDA trade execution + risk management
│   ├── backtester.py      # Walk-forward backtesting engine
│   └── utils.py           # Notifications, state management, trade journal
├── notebooks/
│   └── colab_master_loop.py  # Production Colab orchestrator
├── tests/
│   └── test_pipeline.py   # End-to-end test suite
├── models/                # Saved model weights (on Google Drive in production)
├── data/                  # Historical data cache
├── requirements.txt
├── .env.example
└── README.md
```

## Risk Disclaimer
This is experimental software for educational purposes. Forex trading involves substantial risk of loss. Never trade with money you can't afford to lose. Past performance does not guarantee future results. Always start with a demo account.
