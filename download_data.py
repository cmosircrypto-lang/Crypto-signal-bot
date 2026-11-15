import os
import shutil
import time
from datetime import datetime, timezone
import ccxt
import pandas as pd
from tqdm import tqdm

# ------------------- تنظیمات -------------------
# در OKX نمادها به صورت BTC/USDT:USDT هستند
SYMBOLS_OKX = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "BNB/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
    "NEAR/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT",
    "ATOM/USDT:USDT", "UNI/USDT:USDT", "FIL/USDT:USDT", "INJ/USDT:USDT"
]

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
DAYS_BACK = 30
OUTPUT_DIR = "data"

# پاک کردن کامل پوشه data برای شروع تمیز
if os.path.exists(OUTPUT_DIR):
    print("🧹 پاک کردن داده‌های قدیمی...")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_all():
    # اتصال به OKX
    exchange = ccxt.okx({"enableRateLimit": True})
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    for symbol in tqdm(SYMBOLS_OKX, desc="Symbols"):
        # تبدیل نام OKX به نام ربات (مثلاً BTC/USDT:USDT -> BTC-SWAP-USDT)
        # تا بک‌تست بتواند فایل‌ها را با همان نام‌های اصلی پیدا کند
        base_name = symbol.split("/")[0]
        if base_name.startswith("1000"):
            # برای نمادهای خاص مثل 1000PEPE
            bot_name = f"{base_name}-SWAP-USDT"
        else:
            bot_name = f"{base_name}-SWAP-USDT"
            
        for tf in TIMEFRAMES:
            file_path = os.path.join(OUTPUT_DIR, f"{bot_name}_{tf}.parquet")
            
            # تعیین زمان شروع بر اساس تایم‌فریم
            if tf == "1d":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            elif tf == "4h":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            elif tf == "1h":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            elif tf == "15m":
                since = now_ms - (120 * 24 * 60 * 60 * 1000) # ۱۲۰ روز برای ۱۵ دقیقه
            elif tf == "5m":
                since = now_ms - (30 * 24 * 60 * 60 * 1000) # ۳۰ روز برای ۵ دقیقه
            
            all_candles = []
            current_since = since
            
            while True:
                try:
                    candles = exchange.fetch_ohlcv(symbol, timeframe=tf, since=current_since, limit=100)
                    if not candles:
                        break
                    
                    all_candles.extend(candles)
                    last_ts = candles[-1][0]
                    
                    # اگر به زمان حال رسیدیم
                    if last_ts >= now_ms - (60 * 60 * 1000):
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
                print(f"  ✅ {bot_name} {tf}: {len(df)} کندل ذخیره شد.")

if __name__ == "__main__":
    print(f"🚀 شروع دانلود {DAYS_BACK} روز داده از OKX برای {len(SYMBOLS_OKX)} نماد...")
    download_all()
    print("🎉 دانلود کامل شد!")