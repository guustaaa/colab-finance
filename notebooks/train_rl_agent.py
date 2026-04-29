import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import ta  

warnings.filterwarnings("ignore", category=FutureWarning)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from src.data_fetcher import CapitalFetcher
from src.rl_agent import RLAgent

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("rl_trainer")

def build_timeframe_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resamples base M5 data into higher timeframes and extracts features."""
    if timeframe != "5min":
        df = df.resample(timeframe).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
    df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    
    # Sentiment Proxy
    rsi_norm = (df['momentum_rsi'] - 50.0) / 50.0
    macd_norm = (df['trend_macd_diff'] / df['close']) * 1000.0
    df['sentiment'] = (rsi_norm + macd_norm).clip(lower=-1.0, upper=1.0)
    
    # Prefix columns to avoid collisions when merging
    df.columns = [f"{timeframe}_{c}" for c in df.columns]
    return df

def generate_multi_timeframe_matrix(df_m5: pd.DataFrame) -> pd.DataFrame:
    logger.info("Synthesizing M15 and H1 macro horizons from base M5 data...")
    
    # Generate timeframe features
    df_5m = build_timeframe_features(df_m5.copy(), "5min")
    df_15m = build_timeframe_features(df_m5.copy(), "15min")
    df_1h = build_timeframe_features(df_m5.copy(), "1H")
    
    # Stitch timeframes together using forward-fill (so M5 always has the latest H1 context)
    df_merged = df_5m.copy()
    df_merged = df_merged.join(df_15m, how='outer').ffill()
    df_merged = df_merged.join(df_1h, how='outer').ffill()
    
    # Drop rows until all timeframes have warmed up
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    
    # Lock execution columns to exact JAX positions (Base M5 close is always index 3, sentiment index 4)
    core_cols = ['5min_open', '5min_high', '5min_low', '5min_close', '5min_sentiment']
    other_cols = [c for c in df_merged.columns if c not in core_cols]
    df_merged = df_merged[core_cols + other_cols]
    
    return df_merged

def main():
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        os.environ["CAPITAL_API_KEY"] = user_secrets.get_secret("CAPITAL_API_KEY")
        os.environ["CAPITAL_EMAIL"] = user_secrets.get_secret("CAPITAL_EMAIL")
        os.environ["CAPITAL_PASSWORD"] = user_secrets.get_secret("CAPITAL_PASSWORD")
        logger.info("✅ Kaggle Secrets successfully loaded")
    except Exception as e:
        logger.error(f"🚨 Failed to load Kaggle secrets: {e}")
        return

    fetcher = CapitalFetcher()
    target_pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF"]
    data_dict = {}
    
    for pair in target_pairs:
        logger.info(f"Fetching max M5 history for {pair}...")
        # Fetching M5 takes more API calls but gives execution precision
        df = fetcher.fetch_bulk_history(pair, granularity="M5", years=1.0) 
        
        if df is not None and not df.empty:
            df = generate_multi_timeframe_matrix(df)
            data_dict[pair] = df
            logger.info(f"✅ {pair} Multi-Timeframe Matrix Ready: {len(df)} rows x {len(df.columns)} dimensions.")
        else:
            logger.warning(f"⚠️ Failed to fetch data for {pair}, skipping...")

    if not data_dict:
        logger.error("🚨 No data was fetched. Exiting.")
        return

    logger.info("🧠 Spawning CUSTOM SAC Multi-Timeframe Environments...")
    agent = RLAgent(model_path="/kaggle/working/ForexAI_State/models/rl_custom_sac_v1.pkl")
    agent.train(data_dict=data_dict, total_timesteps=500_000_000)

if __name__ == "__main__":
    main()