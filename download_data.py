import argparse
import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timezone
from tqdm import tqdm

# ۲۰ نماد پرحجم و اصلی
SUBSET_SYMBOLS = [
    "BTC-SWAP-USDT", "ETH-SWAP-USDT", "SOL-SWAP-USDT", "XRP-SWAP-USDT",
    "BNB-SWAP-USDT", "DOGE-SWAP-USDT", "ADA-SWAP-USDT", "AVAX-SWAP-USDT",
    "LINK-SWAP-USDT", "DOT-SWAP-USDT", "LTC-SWAP-USDT", "BCH-SWAP-USDT",
    "NEAR-SWAP-USDT", "APT-SWAP-USDT", "ARB-SWAP-USDT", "OP-SWAP-USDT",
    "ATOM-SWAP-USDT", "UNI-SWAP-USDT", "FIL-SWAP-USDT", "INJ-SWAP-USDT"
]

# ... (همون لیست ۸۸ تایی قبلی برای ALL_SYMBOLS)

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
OUTPUT_DIR = "data"

def download_all(symbols, days_back):
    exchange = ccxt.toobit({"enableRateLimit": True})
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - (days_back * 24 * 60 * 60 * 1000)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for symbol in tqdm(symbols, desc="Downloading"):
        for tf in TIMEFRAMES:
            file_path = os.path.join(OUTPUT_DIR, f"{symbol}_{tf}.parquet")
            if os.path.exists(file_path):
                continue
                
            all_candles = []
            current_since = since_ms
            
            while True:
                try:
                    candles = exchange.fetch_ohlcv(symbol, timeframe=tf, since=current_since, limit=1000)
                    if not candles:
                        break
                    
                    all_candles.extend(candles)
                    last_ts = candles[-1][0]
                    
                    if last_ts >= now_ms:
                        break
                        
                    current_since = last_ts + 1
                    time.sleep(exchange.rateLimit / 1000)
                    
                except Exception as e:
                    time.sleep(5)
                    continue
            
            if all_candles:
                df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df.to_parquet(file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", action="store_true", help="Only download 20 high-volume symbols")
    parser.add_argument("--days", type=int, default=365, help="Days back")
    args = parser.parse_args()
    
    symbols = SUBSET_SYMBOLS if args.subset else ALL_SYMBOLS
    print(f"🚀 Downloading {args.days} days for {len(symbols)} symbols")
    download_all(symbols, args.days)
    print("🎉 Done!")