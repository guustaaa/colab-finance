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

    def fetch_bulk_history(
        self,
        instrument: str,
        years: float = 2.0,
        granularity: str = "H1",
        cache_path: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Paginate backwards to collect up to `years` of H1 history.

        Capital.com allows GET /api/v1/prices/{epic}?from=...&to=...&max=1000
        We walk backwards in 1000-candle windows until we have enough data.

        Pages are cached to Google Drive so re-runs don't re-fetch everything.

        Parameters
        ----------
        instrument : str
            Internal instrument name, e.g. "EUR_USD"
        years : float
            Approximate years of history to fetch (default 2 = ~17k H1 candles)
        granularity : str
            Candle size (H1 default)
        cache_path : str | None
            If set, save/load parquet cache here (e.g. Drive path)
        """
        import time as _time
        from datetime import timezone

        # ── Try loading from cache first ──
        if cache_path and os.path.exists(cache_path):
            try:
                cached = pd.read_parquet(cache_path)
                if not cached.empty and len(cached) >= 300:
                    age_hours = (pd.Timestamp.now(tz="UTC") - cached.index[-1]).total_seconds() / 3600
                    if age_hours < 24:
                        logger.info(f"[{instrument}] Loaded {len(cached)} candles from cache (age {age_hours:.1f}h)")
                        return cached
                    logger.info(f"[{instrument}] Cache stale ({age_hours:.0f}h old). Refreshing...")
                elif not cached.empty:
                    logger.warning(f"[{instrument}] Cache too small ({len(cached)} candles). Re-fetching.")
            except Exception as e:
                logger.warning(f"[{instrument}] Cache read failed: {e}. Re-fetching.")

        epic = self.client.to_epic(instrument)
        resolution = self.RESOLUTION_MAP.get(granularity, "HOUR")

        # Candles per hour based on granularity
        candles_per_day = {"MINUTE": 1440, "MINUTE_5": 288, "MINUTE_15": 96,
                           "MINUTE_30": 48, "HOUR": 24, "HOUR_4": 6, "DAY": 1}.get(resolution, 24)
        target_candles = int(years * 365 * candles_per_day)

        all_frames = []
        # Start from now and walk backwards
        to_dt = pd.Timestamp.now(tz="UTC")
        batch_size = 1000
        max_batches = (target_candles // batch_size) + 2  # safety ceiling
        consecutive_empty = 0

        logger.info(f"[{instrument}] Fetching {target_candles} candles ({years:.1f}y) in {max_batches} batches...")

        for batch_num in range(max_batches):
            # Walk backwards by batch_size candles worth of time
            hours_per_batch = batch_size / candles_per_day
            from_dt = to_dt - pd.Timedelta(hours=hours_per_batch * 1.05)  # slight overlap

            data = self.client._get(
                f"/api/v1/prices/{epic}",
                params={
                    "resolution": resolution,
                    "max": batch_size,
                    "from": from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "to":   to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )

            if data is None or not data.get("prices"):
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    logger.warning(f"[{instrument}] 3 empty batches in a row — stopping early at batch {batch_num}")
                    break
                to_dt = from_dt  # move window back anyway
                continue

            consecutive_empty = 0
            batch_df = self._parse_candles(data["prices"])
            if batch_df.empty:
                to_dt = from_dt
                continue

            all_frames.append(batch_df)
            total_so_far = sum(len(f) for f in all_frames)

            # Move window back to just before the oldest candle in this batch
            to_dt = batch_df.index[0] - pd.Timedelta(minutes=1)

            logger.info(
                f"[{instrument}] Batch {batch_num+1}/{max_batches}: "
                f"+{len(batch_df)} candles | total={total_so_far} | oldest={batch_df.index[0].date()}"
            )

            if total_so_far >= target_candles:
                logger.info(f"[{instrument}] Target reached ({total_so_far} >= {target_candles}). Done.")
                break

            _time.sleep(0.3)  # respect 10 req/s rate limit with margin

        if not all_frames:
            logger.error(f"[{instrument}] No data fetched at all!")
            return None

        df = pd.concat(all_frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]  # remove overlap duplicates
        logger.info(f"[{instrument}] Total: {len(df)} candles | {df.index[0].date()} → {df.index[-1].date()}")

        # ── Save to cache ──
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                df.to_parquet(cache_path)
                logger.info(f"[{instrument}] Cached to {cache_path}")
            except Exception as e:
                logger.warning(f"[{instrument}] Cache write failed: {e}")

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
