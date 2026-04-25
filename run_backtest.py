import sys
import os

# Add the current directory to the path so src can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_fetcher import generate_mock_data
from src.backtester import Backtester

def main():
    print("Generating 3000 candles of realistic mock data (incorporating mean reversion and random walk)...")
    df = generate_mock_data(n=3000)
    
    print("\nRunning Walk-Forward Backtester with transaction costs and slippage...")
    bt = Backtester(initial_balance=10000.0)
    # Using 1000 candles for training, 250 for testing, rolling forward by 250 each time
    metrics = bt.run(df, train_size=1000, test_size=250, step_size=250)
    
    print("\n" + "="*50)
    print("FINAL BACKTEST METRICS (Walk-Forward Out-Of-Sample)")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:25s}: {value}")
    print("="*50)

if __name__ == "__main__":
    main()
