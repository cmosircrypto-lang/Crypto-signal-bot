import json
import os
import time
import pandas as pd
import ccxt
from datetime import datetime, timezone
import excel_log

# --- تنظیمات ---
BOTS_FILES = [
    "last_signals.json",
    "last_signals_v10.json",
    "last_signals_v20.json",
    "last_signals_v30.json",
    "last_signals_v40.json",
    "last_signals_v50.json",
    "last_signals_v60.json",
    "last_signals_v70.json",
]
BOT_NAMES = {
    "last_signals.json": "crypto-signal-bot",
    "last_signals_v10.json": "crypto-signal-bot-v10",
    "last_signals_v20.json": "crypto-signal-bot-v20",
    "last_signals_v30.json": "crypto-signal-bot-v30",
    "last_signals_v40.json": "crypto-signal-bot-v40",
    "last_signals_v50.json": "crypto-signal-bot-v50",
    "last_signals_v60.json": "crypto-signal-bot-v60",
    "last_signals_v70.json": "crypto-signal-bot-v70",
}

def now_ms():
    return int(time.time() * 1000)

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def check_trade_result(exchange, symbol, data):
    direction = data["direction"]
    tp = data["tp"]
    sl = data["sl"]
    opened_ms = data.get("opened_ms")

    if not opened_ms:
        return None, None

    try:
        # دریافت دیتای ۵ دقیقه‌ای از زمان باز شدن معامله
        candles = exchange.fetch_ohlcv(symbol, timeframe="5m", since=int(opened_ms), limit=1000)
    except Exception as e:
        print(f"⚠️ خطا در دریافت دیتای {symbol}: {e}")
        return None, None

    for cd in candles:
        if cd[0] < opened_ms:
            continue
        high, low = cd[2], cd[3]
        
        if direction == "LONG":
            if low <= sl: return "SL", sl
            if high >= tp: return "TP", tp
        else: # SHORT
            if high >= sl: return "SL", sl
            if low <= tp: return "TP", tp
            
    return None, None

def main():
    print("🚀 شروع فرایند بازیابی معاملات گم‌شده...")
    exchange = ccxt.toobit({'enableRateLimit': True})
    
    total_recovered = 0
    
    for filename in BOTS_FILES:
        if not os.path.exists(filename):
            continue
            
        bot_name = BOT_NAMES.get(filename, "unknown-bot")
        signals = load_json(filename)
        updated = False
        
        print(f"\n🔍 بررسی فایل {filename} برای ربات {bot_name}...")
        
        for symbol, data in list(signals.items()):
            if data.get("status") == "OPEN":
                print(f"  ⏳ بررسی معامله باز: {symbol} {data['direction']} | Entry: {data['entry']}")
                
                result, close_price = check_trade_result(exchange, symbol, data)
                
                if result:
                    data["status"] = "CLOSED"
                    data["result"] = result
                    data["closed_ms"] = now_ms()
                    data["close_price"] = float(close_price)
                    data["close_telegram_sent"] = True # جلوگیری از ارسال پیام تکراری به تلگرام
                    data["symbol"] = symbol
                    updated = True
                    total_recovered += 1
                    
                    # ثبت در اکسل
                    excel_log.log_result(bot_name, symbol, result, data["closed_ms"])
                    
                    print(f"  ✅ معامله {symbol} در گذشته بسته شده بود! نتیجه: {result} در قیمت {close_price}")
                else:
                    print(f"  ⏸️ معامله {symbol} هنوز باز است.")
        
        if updated:
            save_json(filename, signals)
            print(f"💾 فایل {filename} به‌روزرسانی شد.")
            
    print(f"\n🎉 پایان! تعداد {total_recovered} معامله گم‌شده بازیابی و در اکسل ثبت شد.")

if __name__ == "__main__":
    main()