import os
import logging
from src.data_fetcher import CapitalFetcher
from src.rl_agent import RLAgent

# Suppress TF/JAX CUDA factory warnings before any heavy JAX imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("rl_trainer")

def main():
    # Initialize the fetcher to use your loaded Capital.com API secrets
    fetcher = CapitalFetcher()
    
    # 1. Fetch data for MULTIPLE pairs to force the network to learn generalized sentiment
    target_pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF"]
    data_dict = {}
    
    for pair in target_pairs:
        logger.info(f"Fetching max history for {pair}...")
        
        # Fetch the bulk historical data
        df = fetcher.fetch_bulk_history(pair) 
        
        # Ensure it actually returned data before adding to our matrix
        if df is not None and not df.empty:
            data_dict[pair] = df
            logger.info(f"✅ {pair} Features Ready: {len(df)} rows.")
        else:
            logger.warning(f"⚠️ Failed to fetch data for {pair}, skipping...")

    if not data_dict:
        logger.error("🚨 No data was fetched across any currency pairs. Exiting.")
        return

    logger.info("🧠 Spawning Matrix Environments and Initializing PPO...")
    
    # Point the agent to where you want the weights saved
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