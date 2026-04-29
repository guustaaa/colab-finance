import os
import sys
import logging
import warnings
import pandas as pd
import ta  

# Silence the Pandas/TA FutureWarnings
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

def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing Technical Indicators & Sentiment Math...")
    df = ta.add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )
    df = df.dropna().reset_index(drop=True)
    return df

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
        logger.info(f"Fetching max history for {pair}...")
        df = fetcher.fetch_bulk_history(pair) 
        
        if df is not None and not df.empty:
            df = add_market_context(df)
            data_dict[pair] = df
            logger.info(f"✅ {pair} Features Ready: {len(df)} rows x {len(df.columns)} dimensions.")
        else:
            logger.warning(f"⚠️ Failed to fetch data for {pair}, skipping...")

    if not data_dict:
        logger.error("🚨 No data was fetched. Exiting.")
        return

    logger.info("🧠 Spawning Hybrid PST-Trader Matrix Environments...")
    agent = RLAgent(model_path="/kaggle/working/ForexAI_State/models/rl_pst_trader_v9.pkl")
    agent.train(data_dict=data_dict, total_timesteps=500_000_000)

if __name__ == "__main__":
    main()