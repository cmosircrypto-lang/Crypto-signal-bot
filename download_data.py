import os
import subprocess
import time
from datetime import datetime, timezone
import ccxt
import pandas as pd
from tqdm import tqdm

SYMBOLS_OKX = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "BNB/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT",
    "AVAX/USDT:USDT", "LINK/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT",
    "BCH/USDT:USDT", "TON/USDT:USDT", "SUI/USDT:USDT", "NEAR/USDT:USDT",
    "APT/USDT:USDT", "ARB/USDT:USDT", "OP/USDT:USDT", "ATOM/USDT:USDT",
    "UNI/USDT:USDT", "AAVE/USDT:USDT", "FIL/USDT:USDT", "INJ/USDT:USDT",
    "TIA/USDT:USDT", "JUP/USDT:USDT", "ENA/USDT:USDT", "WIF/USDT:USDT",
    "1000PEPE/USDT:USDT", "1000SHIB/USDT:USDT", "WLD/USDT:USDT", "SEI/USDT:USDT",
    "RUNE/USDT:USDT", "LDO/USDT:USDT", "STX/USDT:USDT", "IMX/USDT:USDT",
    "HBAR/USDT:USDT", "ICP/USDT:USDT", "ETC/USDT:USDT", "XLM/USDT:USDT",
    "CRV/USDT:USDT", "SNX/USDT:USDT", "COMP/USDT:USDT", "SUSHI/USDT:USDT",
    "1INCH/USDT:USDT", "GRT/USDT:USDT", "DYDX/USDT:USDT", "GMX/USDT:USDT",
    "PENDLE/USDT:USDT", "ONDO/USDT:USDT", "JTO/USDT:USDT", "PYTH/USDT:USDT",
    "STRK/USDT:USDT", "MANTA/USDT:USDT", "1000BONK/USDT:USDT", "1000FLOKI/USDT:USDT",
    "MEME/USDT:USDT", "ORDI/USDT:USDT", "1000SATS/USDT:USDT", "1000RATS/USDT:USDT",
    "POPCAT/USDT:USDT", "BRETT/USDT:USDT", "TURBO/USDT:USDT", "DOGS/USDT:USDT",
    "FET/USDT:USDT", "TAO/USDT:USDT", "AKT/USDT:USDT", "ILV/USDT:USDT",
    "AXS/USDT:USDT", "SAND/USDT:USDT", "MANA/USDT:USDT", "GALA/USDT:USDT",
    "ALGO/USDT:USDT", "VET/USDT:USDT", "EGLD/USDT:USDT", "FLOW/USDT:USDT",
    "KAVA/USDT:USDT", "ROSE/USDT:USDT", "KAS/USDT:USDT", "THETA/USDT:USDT",
    "CHZ/USDT:USDT", "ENJ/USDT:USDT", "ZIL/USDT:USDT", "ONE/USDT:USDT",
    "QTUM/USDT:USDT", "IOTA/USDT:USDT", "NEO/USDT:USDT"
]

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
DAYS_BACK = 120  # ۱۲۰ روز
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_lfs():
    print("🔧 راه‌اندازی Git LFS...")
    subprocess.run(["git", "lfs", "install"], check=True)
    subprocess.run(["git", "lfs", "track", "*.parquet"], check=True)
    print("✅ Git LFS آماده است")

def download_all():
    exchange = ccxt.okx({"enableRateLimit": True})
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    for symbol in tqdm(SYMBOLS_OKX, desc="Symbols"):
        base_name = symbol.split("/")[0]
        bot_name = f"{base_name}-SWAP-USDT"
        
        for tf in TIMEFRAMES:
            file_path = os.path.join(OUTPUT_DIR, f"{bot_name}_{tf}.parquet")
            
            if os.path.exists(file_path):
                print(f"⏩ {bot_name} {tf} already exists, skipping...")
                continue
            
            if tf == "1d":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            elif tf == "4h":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            elif tf == "1h":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            elif tf == "15m":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            elif tf == "5m":
                since = now_ms - (DAYS_BACK * 24 * 60 * 60 * 1000)
            
            all_candles = []
            current_since = since
            
            while True:
                try:
                    candles = exchange.fetch_ohlcv(symbol, timeframe=tf, since=current_since, limit=100)
                    if not candles:
                        break
                    
                    all_candles.extend(candles)
                    last_ts = candles[-1][0]
                    
                    if last_ts >= now_ms - (60 * 60 * 1000):
                        break
                        
                    current_since = last_ts + 1
                    time.sleep(exchange.rateLimit / 1000)
                    
                except Exception as e:
                    print(f"⚠️ خطا در {symbol} {tf}: {e}")
                    time.sleep(5)
                    continue
            
            if all_candles:
                df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df.to_parquet(file_path, index=False)
                print(f"  ✅ {bot_name} {tf}: {len(df)} کندل ذخیره شد.")

if __name__ == "__main__":
    setup_lfs()
    print(f"🚀 شروع دانلود {DAYS_BACK} روز داده از OKX برای {len(SYMBOLS_OKX)} نماد...")
    download_all()
    print("🎉 دانلود کامل شد!")