import os
import shutil
import time
from datetime import datetime, timezone, timedelta
import ccxt
import pandas as pd
from tqdm import tqdm

# ------------------- تنظیمات -------------------
SYMBOLS = [
    "BTC-SWAP-USDT", "ETH-SWAP-USDT", "SOL-SWAP-USDT", "XRP-SWAP-USDT",
    "BNB-SWAP-USDT", "DOGE-SWAP-USDT", "ADA-SWAP-USDT", "AVAX-SWAP-USDT",
    "LINK-SWAP-USDT", "DOT-SWAP-USDT", "LTC-SWAP-USDT", "BCH-SWAP-USDT",
    "NEAR-SWAP-USDT", "APT-SWAP-USDT", "ARB-SWAP-USDT", "OP-SWAP-USDT",
    "ATOM-SWAP-USDT", "UNI-SWAP-USDT", "FIL-SWAP-USDT", "INJ-SWAP-USDT"
]  # ۲۰ نماد پرحجم

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
DAYS_BACK = 365
OUTPUT_DIR = "data"

# پاک کردن داده‌های قدیمی برای شروع تمیز
if os.path.exists(OUTPUT_DIR):
    print("🧹 پاک کردن داده‌های قدیمی...")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_all():
    exchange = ccxt.toobit({"enableRateLimit": True, "timeout": 30000})
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    for symbol in tqdm(SYMBOLS, desc="Symbols"):
        for tf in TIMEFRAMES:
            file_path = os.path.join(OUTPUT_DIR, f"{symbol}_{tf}.parquet")
            
            # محاسبه زمان شروع
            if tf == "1d":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            elif tf == "4h":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            elif tf == "1h":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            elif tf == "15m":
                since = now_ms - (90 * 24 * 60 * 60 * 1000) # 90 روز برای 15m
            elif tf == "5m":
                since = now_ms - (30 * 24 * 60 * 60 * 1000) # 30 روز برای 5m
            
            all_candles = []
            current_since = since
            
            # تلاش برای دریافت داده‌ها
            while True:
                try:
                    candles = exchange.fetch_ohlcv(symbol, timeframe=tf, since=current_since, limit=1000)
                    if not candles:
                        break
                    
                    all_candles.extend(candles)
                    last_ts = candles[-1][0]
                    
                    if last_ts >= now_ms - (60 * 60 * 1000): # اگر به ۱ ساعت اخیر رسیدیم
                        break
                        
                    current_since = last_ts + 1
                    time.sleep(exchange.rateLimit / 1000)
                    
                except Exception as e:
                    print(f"⚠️ خطا در دریافت {symbol} {tf}: {e}")
                    time.sleep(5)
                    continue
            
            if all_candles:
                df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df.to_parquet(file_path, index=False)
                print(f"  ✅ {symbol} {tf}: {len(df)} کندل ذخیره شد.")

if __name__ == "__main__":
    print(f"🚀 شروع دانلود {DAYS_BACK} روز داده برای {len(SYMBOLS)} نماد...")
    download_all()
    print("🎉 دانلود کامل شد!")