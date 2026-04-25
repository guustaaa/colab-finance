"""
data_fetcher.py — Capital.com REST API data fetcher + mock data generator.

Capital.com API Reference: https://open-api.capital.com/
- Base Demo URL: https://demo-api-capital.backend-capital.com
- Base Live URL: https://api-capital.backend-capital.com
- Session expires after 10 minutes of inactivity
- Rate limit: 10 requests/second, 1 req/0.1s for orders
"""
import os
import time
import logging
import requests
import numpy as np
import pandas as pd

logger = logging.getLogger("data_fetcher")

# ──────────────────────────────────────────────────────────
# Capital.com REST API Client
# ──────────────────────────────────────────────────────────

class CapitalClient:
    """
    Thin wrapper around the Capital.com REST API.
    Handles session management (auto-refresh) and authenticated requests.
    """

    DEMO_URL = "https://demo-api-capital.backend-capital.com"
    LIVE_URL = "https://api-capital.backend-capital.com"

    # Capital.com uses "epics" for instrument names
    EPIC_MAP = {
        "EUR_USD": "EURUSD",
        "GBP_USD": "GBPUSD",
        "USD_JPY": "USDJPY",
        "AUD_USD": "AUDUSD",
        "USD_CAD": "USDCAD",
        "NZD_USD": "NZDUSD",
        "USD_CHF": "USDCHF",
    }

    # Reverse map for converting epic back to our internal names
    REVERSE_EPIC_MAP = {v: k for k, v in EPIC_MAP.items()}

    def __init__(self, demo: bool = True):
        self.base_url = self.DEMO_URL if demo else self.LIVE_URL
        self.api_key = os.environ.get("CAPITAL_API_KEY", "")
        self.email = os.environ.get("CAPITAL_EMAIL", "")
        self.password = os.environ.get("CAPITAL_PASSWORD", "")
        self.cst = None
        self.security_token = None
        self._session_time = 0

        if not self.api_key or not self.email or not self.password:
            logger.warning(
                "Capital.com credentials not set. "
                "Set CAPITAL_API_KEY, CAPITAL_EMAIL, CAPITAL_PASSWORD."
            )

    def _ensure_session(self):
        """Create or refresh session if expired (10 min timeout)."""
        if self.cst and (time.time() - self._session_time < 540):  # 9 min safety
            return True
        return self._create_session()

    def _create_session(self) -> bool:
        """POST /api/v1/session — create a new trading session."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/session",
                headers={
                    "X-CAP-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "identifier": self.email,
                    "password": self.password,
                    "encryptedPassword": False,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                self.cst = resp.headers.get("CST")
                self.security_token = resp.headers.get("X-SECURITY-TOKEN")
                self._session_time = time.time()
                logger.info("Capital.com session created successfully.")
                return True
            else:
                logger.error(f"Session creation failed [{resp.status_code}]: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            return False

    def _auth_headers(self) -> dict:
        """Headers required for all authenticated requests."""
        return {
            "X-SECURITY-TOKEN": self.security_token or "",
            "CST": self.cst or "",
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict = None) -> dict | None:
        """Authenticated GET request with auto-session refresh."""
        if not self._ensure_session():
            return None
        try:
            resp = requests.get(
                f"{self.base_url}{path}",
                headers=self._auth_headers(),
                params=params,
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.error(f"GET {path} failed [{resp.status_code}]: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"GET {path} error: {e}")
            return None

    def _post(self, path: str, data: dict) -> dict | None:
        """Authenticated POST request."""
        if not self._ensure_session():
            return None
        try:
            resp = requests.post(
                f"{self.base_url}{path}",
                headers=self._auth_headers(),
                json=data,
                timeout=15,
            )
            if resp.status_code in (200, 201):
                result = resp.json() if resp.text else {}
                # Also capture dealReference from response
                result["_status_code"] = resp.status_code
                result["_headers"] = dict(resp.headers)
                return result
            else:
                logger.error(f"POST {path} failed [{resp.status_code}]: {resp.text}")
                return {"error": resp.text, "_status_code": resp.status_code}
        except Exception as e:
            logger.error(f"POST {path} error: {e}")
            return {"error": str(e)}

    def _delete(self, path: str) -> dict | None:
        """Authenticated DELETE request."""
        if not self._ensure_session():
            return None
        try:
            resp = requests.delete(
                f"{self.base_url}{path}",
                headers=self._auth_headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json() if resp.text else {"status": "OK"}
            else:
                logger.error(f"DELETE {path} failed [{resp.status_code}]: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"DELETE {path} error: {e}")
            return None

    def to_epic(self, instrument: str) -> str:
        """Convert our instrument name (EUR_USD) to Capital.com epic (EURUSD)."""
        return self.EPIC_MAP.get(instrument, instrument)


# ──────────────────────────────────────────────────────────
# Data Fetcher (replaces OandaFetcher)
# ──────────────────────────────────────────────────────────

class CapitalFetcher:
    """
    Fetches OHLCV candle data and account info from Capital.com.
    Drop-in replacement for the old OandaFetcher.
    """

    # Capital.com resolution strings
    RESOLUTION_MAP = {
        "M5": "MINUTE_5",
        "M15": "MINUTE_15",
        "M30": "MINUTE_30",
        "H1": "HOUR",
        "H4": "HOUR_4",
        "D": "DAY",
        "W": "WEEK",
    }

    def __init__(self, demo: bool = True):
        self.client = CapitalClient(demo=demo)

    def get_account_balance(self) -> float | None:
        """Fetch the current account balance."""
        data = self.client._get("/api/v1/accounts")
        if data and "accounts" in data:
            for acc in data["accounts"]:
                if acc.get("preferred", False):
                    return acc.get("balance", {}).get("balance", 0.0)
            # Fallback to first account
            if data["accounts"]:
                return data["accounts"][0].get("balance", {}).get("balance", 0.0)
        return None

    def fetch_candles(
        self,
        instrument: str,
        count: int = 500,
        granularity: str = "H1",
    ) -> pd.DataFrame | None:
        """
        Fetch historical OHLCV candles from Capital.com.

        GET /api/v1/prices/{epic}?resolution=HOUR&max={count}

        Returns DataFrame with columns: open, high, low, close, volume
        """
        epic = self.client.to_epic(instrument)
        resolution = self.RESOLUTION_MAP.get(granularity, "HOUR")

        data = self.client._get(
            f"/api/v1/prices/{epic}",
            params={"resolution": resolution, "max": min(count, 1000)},
        )
        if data is None or "prices" not in data:
            logger.error(f"Failed to fetch candles for {instrument} ({epic})")
            return None

        prices = data["prices"]
        if not prices:
            logger.warning(f"No price data returned for {instrument}")
            return None

        # Capital.com returns bid/ask OHLC — use mid-price
        rows = []
        for candle in prices:
            snap = candle.get("snapshotTimeUTC", candle.get("snapshotTime", ""))
            bid = candle.get("closePrice", {})  # bid close
            ask = candle.get("highPrice", {})    # will get from structure

            # Each candle has: openPrice, closePrice, highPrice, lowPrice
            # Each sub-dict has: bid, ask
            o_bid = candle.get("openPrice", {}).get("bid", 0)
            o_ask = candle.get("openPrice", {}).get("ask", 0)
            h_bid = candle.get("highPrice", {}).get("bid", 0)
            h_ask = candle.get("highPrice", {}).get("ask", 0)
            l_bid = candle.get("lowPrice", {}).get("bid", 0)
            l_ask = candle.get("lowPrice", {}).get("ask", 0)
            c_bid = candle.get("closePrice", {}).get("bid", 0)
            c_ask = candle.get("closePrice", {}).get("ask", 0)

            rows.append({
                "time": snap,
                "open": (o_bid + o_ask) / 2,
                "high": (h_bid + h_ask) / 2,
                "low": (l_bid + l_ask) / 2,
                "close": (c_bid + c_ask) / 2,
                "volume": candle.get("lastTradedVolume", 0),
            })

        df = pd.DataFrame(rows)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)

        logger.info(f"Fetched {len(df)} candles for {instrument}")
        return df

    def get_spread(self, instrument: str) -> float:
        """Get the current bid-ask spread for an instrument."""
        epic = self.client.to_epic(instrument)
        data = self.client._get(f"/api/v1/markets/{epic}")
        if data and "snapshot" in data:
            bid = data["snapshot"].get("bid", 0)
            offer = data["snapshot"].get("offer", 0)
            if bid > 0 and offer > 0:
                return offer - bid
        return 0.0002  # Default spread fallback

    def get_market_details(self, instrument: str) -> dict | None:
        """Get full market details including min size, lot size, etc."""
        epic = self.client.to_epic(instrument)
        return self.client._get(f"/api/v1/markets/{epic}")

    def get_client_sentiment(self, instrument: str) -> dict | None:
        """Get client sentiment (% long vs short) for an instrument."""
        epic = self.client.to_epic(instrument)
        data = self.client._get(f"/api/v1/clientsentiment/{epic}")
        return data


# ──────────────────────────────────────────────────────────
# Mock Data Generator (unchanged — used for backtesting)
# ──────────────────────────────────────────────────────────

def generate_mock_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic EUR/USD-like OHLCV data using Geometric Brownian Motion
    with mean reversion (Ornstein-Uhlenbeck process).

    Used for backtesting and testing the full pipeline without live API access.
    """
    np.random.seed(seed)

    # OU process parameters (calibrated to EUR/USD daily)
    mu = 1.08       # Long-run mean
    theta = 0.01    # Mean reversion speed
    sigma = 0.005   # Volatility

    prices = np.zeros(n)
    prices[0] = mu + np.random.normal(0, 0.01)

    for i in range(1, n):
        dW = np.random.normal(0, 1)
        prices[i] = prices[i-1] + theta * (mu - prices[i-1]) + sigma * dW

    # Generate OHLCV from close prices — columns built in standard order
    opens  = np.roll(prices, 1); opens[0] = prices[0]
    noise  = np.random.uniform(0.0001, 0.003, n)
    highs  = np.maximum(opens, prices) + noise
    lows   = np.minimum(opens, prices) - noise
    volume = np.random.randint(100, 10000, n)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volume},
        index=pd.date_range("2023-01-01", periods=n, freq="h"),
    )
    return df
