import os
import sys
import time
import logging

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import INSTRUMENTS, DRIVE_MODELS_DIR, TRADING_GRANULARITY, BULK_HISTORY_YEARS
from src.data_fetcher import CapitalFetcher
from src.features import compute_all_features
from src.sentiment import SentimentScanner
from src.rl_agent import RLAgent
from src.utils import setup_logger

setup_logger("rl_trainer", "./logs")
logger = logging.getLogger("rl_trainer")
# notebooks/train_rl_agent.py
# ... (Keep your imports and secret loading) ...

def main():
    # 1. Fetch data for MULTIPLE pairs to force the network to learn generalized sentiment
    target_pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF"]
    data_dict = {}
    
    # Assuming your fetcher is initialized here
    for pair in target_pairs:
        logger.info(f"Fetching max history for {pair}...")
        # Replace this with your exact CapitalFetcher call
        df = fetcher.fetch_bulk_history(pair) 
        data_dict[pair] = df
        logger.info(f"✅ {pair} Features Ready: {len(df)} rows.")

    logger.info("🧠 Spawning Matrix Environments and Initializing PPO...")
    agent = RLAgent(model_path="/kaggle/working/ForexAI_State/models/rl_ppo_agent.pkl")
    
    # 2. PUSH THE T4 GPUs TO MAXIMUM CAPACITY
    agent.train(
        data_dict=data_dict,
        total_timesteps=50_000_000,  # 50 Million micro-decisions
        n_envs=4096,                 # 4,096 Parallel universes simulated at once
        batch_size=8192              # Massive batches to saturate the 2x T4s
    )

if __name__ == "__main__":
    main()
