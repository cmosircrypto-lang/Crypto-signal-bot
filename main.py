import os
import requests
import pandas as pd
import ccxt

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
TOOBIT_API_KEY = os.environ.get("TOOBIT_API_KEY")
TOOBIT_SECRET = os.environ.get("TOOBIT_SECRET")

SYMBOLS = [
    "SOL-SWAP-USDT",
    "NEAR-SWAP-USDT",
    "ADA-SWAP-USDT",
    "XRP-SWAP-USDT",
    "SUI-SWAP-USDT",
]

def fetch_data(symbol):
    """دریافت کندل فیوچرز از Toobit"""
    exchange = ccxt.toobit({
        'apiKey': TOOBIT_API_KEY,
        'secret': TOOBIT_SECRET,
    })
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def analyze(df):
    """تحلیل با RSI (موقت)"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    last_rsi = df['rsi'].iloc[-1]
    last_price = df['close'].iloc[-1]

    if last_rsi < 30:
        return "BUY", last_rsi, last_price
    elif last_rsi > 70:
        return "SELL", last_rsi, last_price
    return None, last_rsi, last_price

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": message})

if __name__ == "__main__":
    print("Starting multi-symbol analysis...")
    for symbol in SYMBOLS:
        try:
            print(f"--- Checking {symbol} ---")
            df = fetch_data(symbol)
            signal, rsi, price = analyze(df)
            print(f"{symbol} | Price: {price} | RSI: {rsi:.2f} | Signal: {signal}")

            if signal:
                msg = f"🚨 سیگنال {signal}\nنماد: {symbol}\nقیمت: {price}\nRSI: {rsi:.2f}"
                send_telegram(msg)
                print(f"Signal for {symbol} sent to Telegram!")
        except Exception as e:
            print(f"⚠️ خطا در {symbol}: {e}")
            continue

    print("Analysis complete.")