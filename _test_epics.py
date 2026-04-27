import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from src.data_fetcher import CapitalFetcher

fetcher = CapitalFetcher(demo=True)
epics_to_test = ["US500", "US100", "GOLD", "OIL_CRUDE", "BRENT", "DXY", "VIX"]

print("Testing Epics...")
for e in epics_to_test:
    data = fetcher.client._get(f"/api/v1/markets/{e}")
    if data and "instrument" in data:
        print(f"SUCCESS: {e} -> {data['instrument'].get('name')}")
    else:
        print(f"FAILED: {e}")
