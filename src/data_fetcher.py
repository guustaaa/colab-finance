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

    def _get(self, path: str, params: dict = None, _retries: int = 5) -> dict | None:
        """Authenticated GET with auto-session refresh and 429 backoff."""
        if not self._ensure_session():
            return None
        for attempt in range(_retries):
            try:
                resp = requests.get(
                    f"{self.base_url}{path}",
                    headers=self._auth_headers(),
                    params=params,
                    timeout=15,
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    wait = min(2 ** attempt, 30)  # 1, 2, 4, 8, 16, cap 30
                    logger.warning(f"429 rate-limited on {path} — backoff {wait}s (attempt {attempt+1}/{_retries})")
                    time.sleep(wait)
                    continue
                logger.error(f"GET {path} failed [{resp.status_code}]: {resp.text}")
                return None
            except Exception as e:
                logger.error(f"GET {path} error: {e}")
                return None
        logger.error(f"GET {path} exhausted {_retries} retries (429)")
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

    @staticmethod
    def _parse_candles(prices: list) -> pd.DataFrame:
        """Parse Capital.com candle list → mid-price OHLCV DataFrame."""
        rows = []
        for c in prices:
            snap = c.get("snapshotTimeUTC", c.get("snapshotTime", ""))
            def mid(key):
                d = c.get(key, {})
                b, a = d.get("bid", 0), d.get("ask", 0)
                return (b + a) / 2 if b and a else (b or a)
            rows.append({
                "time":   snap,
                "open":   mid("openPrice"),
                "high":   mid("highPrice"),
                "low":    mid("lowPrice"),
                "close":  mid("closePrice"),
                "volume": c.get("lastTradedVolume", 0),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"])
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    def fetch_candles(
        self,
        instrument: str,
        count: int = 500,
        granularity: str = "H1",
    ) -> pd.DataFrame | None:
        """
        Fetch the most recent `count` OHLCV candles (up to 1000 per call).
        Use fetch_bulk_history for training data > 1000 candles.
        """
        epic = self.client.to_epic(instrument)
        resolution = self.RESOLUTION_MAP.get(granularity, "HOUR")
        data = self.client._get(
            f"/api/v1/prices/{epic}",
            params={"resolution": resolution, "max": min(count, 1000)},
        )
        if data is None or "prices" not in data or not data["prices"]:
            logger.error(f"Failed to fetch candles for {instrument} ({epic})")
            return None
        df = self._parse_candles(data["prices"])
        logger.info(f"Fetched {len(df)} candles for {instrument}")
        return df if not df.empty else None

    def fetch_bulk_history(self, epic: str) -> pd.DataFrame:
        import os
        import pandas as pd
        
        cache_dir = "/kaggle/working/ForexAI_State/data_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{epic}_history.csv")
        
        cached_df = pd.DataFrame()
        last_cached_date = None
        
        # 1. Load Local Cache
        if os.path.exists(cache_path):
            cached_df = pd.read_csv(cache_path)
            cached_df['timestamp'] = pd.to_datetime(cached_df['timestamp'])
            cached_df = cached_df.sort_values('timestamp').reset_index(drop=True)
            
            if not cached_df.empty:
                last_cached_date = cached_df['timestamp'].iloc[-1]
                logger.info(f"[{epic}] 📦 Cache found! Contains {len(cached_df)} rows. Fetching only NEW data since {last_cached_date.strftime('%Y-%m-%d')}")

        new_data_frames = []
        
        # -------------------------------------------------------------
        # 2. YOUR EXISTING FETCH LOOP (Add the cache breaker)
        # Note: Keep your existing Capital.com session logic here.
        # This is pseudo-code for where to put the cache breaker:
        # -------------------------------------------------------------
        
        # for batch in range(total_batches):
        #     batch_df = ... # (Your existing fetch logic)
        #     
        #     if batch_df is not None and not batch_df.empty:
        #         oldest_batch_date = batch_df['timestamp'].min()
        #         new_data_frames.append(batch_df)
        #
        #         # 🛑 SMART CACHE BREAKER: If we paginated back into history we already have, stop fetching!
        #         if last_cached_date and oldest_batch_date <= last_cached_date:
        #             logger.info(f"[{epic}] 🛑 Reached cached historical data. Stopping fetch early.")
        #             break 
        
        # 3. Combine, Clean, and Save
        newly_fetched_df = pd.concat(new_data_frames, ignore_index=True) if new_data_frames else pd.DataFrame()
        final_df = pd.concat([cached_df, newly_fetched_df], ignore_index=True)
        
        # Drop duplicates in case the API overlap fetched a candle twice
        if not final_df.empty:
            final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
            final_df = final_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            final_df.to_csv(cache_path, index=False)
            
        return final_df

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
