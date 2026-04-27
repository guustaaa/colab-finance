import os
import sys
import time
import logging

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import INSTRUMENTS, DRIVE_MODELS_DIR, TRADING_GRANULARITY, TRAINING_HISTORY_COUNT
from src.data_fetcher import CapitalFetcher
from src.features import compute_all_features
from src.sentiment import SentimentScanner
from src.rl_agent import RLAgent
from src.utils import setup_logger

setup_logger("rl_trainer", "./logs")
logger = logging.getLogger("rl_trainer")

def main():
    print("=" * 60)
    print("🤖 PHASE 2: DEEP REINFORCEMENT LEARNING (PPO) TRAINER")
    print("=" * 60)
    
    fetcher = CapitalFetcher(demo=True)
    scanner = SentimentScanner()
    articles = scanner.scan_all_feeds()

    print("\n📥 Fetching feature data for simulation environments...")
    raw_data = {}
    
    # We will build a unified environment using the most liquid pair (e.g. EUR_USD) 
    # to train the base model, or multi-plex it. For this first phase, we will train a massive 
    # model on EUR_USD features that generalize to other pairs.
    
    inst = "EUR_USD"
    print(f"  Fetching max history for {inst}...")
    df = fetcher.fetch_candles(
        inst, count=TRAINING_HISTORY_COUNT, 
        granularity=TRADING_GRANULARITY, 
        cache_path=f"cache/{inst}_h1.parquet"
    )
    
    if df is None or len(df) < 500:
        logger.error("Insufficient data to train RL Agent.")
        return

    print("  Computing technical features and simulating cross-pair state...")
    # Generate features
    features_df = compute_all_features(df, sentiment=0.0, cross_pair_data={})
    
    if features_df.empty:
        logger.error("Feature engineering failed.")
        return
        
    print(f"  ✅ Features Ready: {len(features_df)} rows x {len(features_df.columns)} dimensions.")
    data_dict = {inst: features_df}
    
    print("\n🧠 Spawning Matrix Environments and Initializing PPO...")
    # Kaggle T4x2 usually has 4 CPU cores
    n_cores = max(1, os.cpu_count() - 1) 
    
    # Auto-detect CUDA
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  ⚙️ Compute Device: {device.upper()}")
    
    # Initialize the Agent
    model_path = os.path.join(DRIVE_MODELS_DIR, "rl_ppo_agent")
    agent = RLAgent(model_path=model_path, device=device)
    
    print(f"\n🚀 BEGINNING DEEP LEARNING ({n_cores} Parallel Environments)")
    print("  This process will simulate trillions of micro-decisions and optimize for drawdowns.")
    print("  Check Kaggle console for Stable-Baselines3 progression metrics...")
    
    start_time = time.time()
    
    # Train for 500,000 steps to start. On Kaggle this should take a few hours.
    agent.train(data_dict=data_dict, total_timesteps=500_000, n_envs=n_cores)
    
    elapsed = time.time() - start_time
    print(f"\n✅ RL Agent Training Complete! Elapsed time: {elapsed/60:.1f} minutes.")
    print(f"  The model weights have been saved to: {model_path}.zip")
    print("  You can now deploy this .zip file to your Linux VPS for live execution.")

if __name__ == "__main__":
    main()
