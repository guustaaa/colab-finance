"""
data_fetcher.py — OANDA V20 API interface for historical and live price data.

Uses the oandapyV20 library for clean REST API communication.
Handles pagination, error recovery, and data normalization.
"""
import os
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts

load_dotenv()
logger = logging.getLogger("data_fetcher")


class OandaFetcher:
    """
    Fetches price data from OANDA V20 API.

    Handles both historical bulk fetches (for training) and
    live single-candle fetches (for the trading loop).
    """

    def __init__(self, token: str = "", account_id: str = "", environment: str = "practice"):
        self.token = token or os.getenv("OANDA_ACCESS_TOKEN", "")
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID", "")
        self.environment = environment or os.getenv("OANDA_ENVIRONMENT", "practice")
        self.client = None

        if self.token:
            self.client = oandapyV20.API(
                access_token=self.token, environment=self.environment
            )
        else:
            logger.warning(
                "No OANDA_ACCESS_TOKEN found. Data fetching will fail unless mocked."
            )

    def fetch_candles(
        self,
        instrument: str,
        count: int = 500,
        granularity: str = "H1",
        include_incomplete: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch historical candlestick data from OANDA.

        For large counts (>5000), automatically paginates.

        Returns DataFrame with columns: open, high, low, close, volume
        indexed by datetime.
        """
        if self.client is None:
            logger.error("OANDA client not initialized.")
            return pd.DataFrame()

        all_candles = []
        remaining = count
        params_base = {"granularity": granularity, "price": "M"}

        # OANDA limits to 5000 candles per request
        while remaining > 0:
            batch_size = min(remaining, 5000)
            params = {**params_base, "count": batch_size}

            # If we already have data, paginate from the earliest candle
            if all_candles:
                params["to"] = all_candles[0]["time"]

            try:
                req = instruments.InstrumentsCandles(
                    instrument=instrument, params=params
                )
                self.client.request(req)
                candles = req.response.get("candles", [])

                if not candles:
                    break

                all_candles = candles + all_candles
                remaining -= len(candles)

                # If we got fewer candles than requested, we've hit the end
                if len(candles) < batch_size:
                    break

            except Exception as e:
                logger.error(f"OANDA fetch error for {instrument}: {e}")
                break

        return self._candles_to_dataframe(all_candles, include_incomplete)

    def fetch_latest(
        self, instrument: str, granularity: str = "H1", n: int = 1
    ) -> pd.DataFrame:
        """Fetch the N most recent candles (including the incomplete current one)."""
        return self.fetch_candles(
            instrument, count=n, granularity=granularity, include_incomplete=True
        )

    def get_account_balance(self) -> float:
        """Get the current account balance for position sizing."""
        if self.client is None:
            return 0.0
        try:
            req = accounts.AccountDetails(accountID=self.account_id)
            self.client.request(req)
            return float(req.response["account"]["balance"])
        except Exception as e:
            logger.error(f"Failed to fetch account balance: {e}")
            return 0.0

    def get_spread(self, instrument: str) -> float:
        """
        Get the current spread for an instrument (in price units).
        Used for transaction cost filtering.
        """
        if self.client is None:
            return 0.0002  # Default ~2 pip spread estimate
        try:
            params = {"count": 1, "granularity": "S5", "price": "BA"}
            req = instruments.InstrumentsCandles(
                instrument=instrument, params=params
            )
            self.client.request(req)
            candles = req.response.get("candles", [])
            if candles:
                ask = float(candles[-1]["ask"]["c"])
                bid = float(candles[-1]["bid"]["c"])
                return ask - bid
        except Exception as e:
            logger.warning(f"Spread fetch failed for {instrument}: {e}")
        return 0.0002

    def _candles_to_dataframe(
        self, candles: list, include_incomplete: bool
    ) -> pd.DataFrame:
        """Convert OANDA candle response to a clean DataFrame."""
        rows = []
        for c in candles:
            if not include_incomplete and not c.get("complete", True):
                continue
            rows.append(
                {
                    "time": c["time"],
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "volume": int(c["volume"]),
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df


def generate_mock_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic mock forex data for testing without API credentials.

    Uses geometric Brownian motion with mean-reversion (Ornstein-Uhlenbeck)
    to simulate realistic FX price dynamics.
    """
    np.random.seed(seed)

    # OU process parameters (calibrated to EUR/USD H1 dynamics)
    dt = 1 / 24  # 1 hour in days
    mu = 1.0850   # long-term mean
    theta = 0.01  # mean-reversion speed
    sigma = 0.0008  # volatility per step

    prices = [mu]
    for _ in range(n - 1):
        dp = theta * (mu - prices[-1]) * dt + sigma * np.random.randn()
        prices.append(prices[-1] + dp)

    prices = np.array(prices)

    # Generate OHLCV from the close prices
    dates = pd.date_range(start="2024-01-01", periods=n, freq="h")
    noise = np.random.uniform(0.0001, 0.0005, n)

    df = pd.DataFrame(
        {
            "open": prices + np.random.randn(n) * 0.0002,
            "high": prices + noise,
            "low": prices - noise,
            "close": prices,
            "volume": np.random.randint(100, 5000, n),
        },
        index=dates,
    )
    df.index.name = "time"
    return df
