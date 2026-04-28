import os
import sys
import logging

# Dynamically add the repository root to the Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Now Python can find 'src'
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
    # 0. Inject Kaggle Secrets into the environment variables
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        os.environ["CAPITAL_API_KEY"] = user_secrets.get_secret("CAPITAL_API_KEY")
        os.environ["CAPITAL_EMAIL"] = user_secrets.get_secret("CAPITAL_EMAIL")
        os.environ["CAPITAL_PASSWORD"] = user_secrets.get_secret("CAPITAL_PASSWORD")
        logger.info("✅ Kaggle Secrets successfully loaded into os.environ")
    except Exception as e:
        logger.error(f"🚨 Failed to load Kaggle secrets: {e}. Are they attached to this notebook?")
        return

    # Initialize the fetcher
    fetcher = CapitalFetcher()
    
    # 1. Fetch data for MULTIPLE pairs
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

    logger.info("🧠 Spawning Hybrid PST-Trader Matrix Environments...")
    
    # Ensure it's looking for the new hybrid model file
    agent = RLAgent(model_path="/kaggle/working/ForexAI_State/models/rl_hybrid_agent.pkl")
    
    # 2. PUSH THE T4 GPUs TO MAXIMUM CAPACITY
    # Note: n_envs and batch_size are now hardcoded in the agent for max VRAM usage
    agent.train(
        data_dict=data_dict,
        total_timesteps=500_000_000
    )

if __name__ == "__main__":
    main()