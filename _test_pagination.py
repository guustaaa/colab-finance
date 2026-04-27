import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from src.data_fetcher import CapitalFetcher
import pandas as pd

fetcher = CapitalFetcher(demo=True)
epic = "EURUSD"
to_dt = pd.Timestamp.now(tz="UTC")

print("Testing with 'from' and 'to'")
from_dt = to_dt - pd.Timedelta(hours=1000)
data1 = fetcher.client._get(
    f"/api/v1/prices/{epic}",
    params={
        "resolution": "HOUR",
        "max": 1000,
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "to":   to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
    }
)
if data1 and "prices" in data1:
    print("With from/to:", len(data1["prices"]), "candles")
else:
    print("With from/to failed or empty:", data1)

print("\nTesting with only 'to'")
data2 = fetcher.client._get(
    f"/api/v1/prices/{epic}",
    params={
        "resolution": "HOUR",
        "max": 1000,
        "to":   to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
    }
)
if data2 and "prices" in data2:
    print("Only to:", len(data2["prices"]), "candles")
    
    # Check if we can paginate further back
    oldest_dt = pd.to_datetime(data2["prices"][0]["snapshotTimeUTC"]).tz_localize("UTC")
    print("Oldest date in batch 1:", oldest_dt)
    
    data3 = fetcher.client._get(
        f"/api/v1/prices/{epic}",
        params={
            "resolution": "HOUR",
            "max": 1000,
            "to":   (oldest_dt - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    if data3 and "prices" in data3:
        print("Only to (batch 2):", len(data3["prices"]), "candles")
        print("Oldest date in batch 2:", pd.to_datetime(data3["prices"][0]["snapshotTimeUTC"]))
    else:
        print("Batch 2 failed or empty:", data3)
else:
    print("Only to failed or empty:", data2)
