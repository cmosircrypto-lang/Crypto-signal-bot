import os
import json
import requests
import pandas as pd
import ccxt
from datetime import datetime
from zoneinfo import ZoneInfo
import jdatetime

# --- تنظیمات ---
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

HEARTBEAT_FILE = "server_messages.json"
MAX_HEARTBEATS = 4

# --- ابزارهای عمومی ---
def get_iran_time():
    """تاریخ شمسی و میلادی + ساعت به وقت ایران"""
    now = datetime.now(ZoneInfo("Asia/Tehran"))
    date_gregorian = now.strftime("%Y-%m-%d")
    now_jalali = jdatetime.datetime.fromgregorian(datetime=now)
    date_jalali = now_jalali.strftime("%Y/%m/%d")
    time_str = now.strftime("%H:%M:%S")
    return date_gregorian, date_jalali, time_str

def send_telegram(message):
    """ارسال پیام به تلگرام و برگرداندن message_id"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": message}, timeout=10)
        if r.status_code == 200:
            return r.json().get("result", {}).get("message_id")
    except Exception as e:
        print(f"⚠️ خطا در ارسال تلگرام: {e}")
    return None

def delete_telegram(message_id):
    """حذف پیام تلگرام"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "message_id": message_id}, timeout=10)
    except Exception as e:
        print(f"⚠️ خطا در حذف پیام: {e}")

# --- Heartbeat ---
def load_heartbeat_ids():
    try:
        with open(HEARTBEAT_FILE, "r") as f:
            return json.load(f).get("message_ids", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_heartbeat_ids(ids):
    with open(HEARTBEAT_FILE, "w") as f:
        json.dump({"message_ids": ids}, f)

def send_heartbeat():
    """ارسال پیام Heartbeat و مدیریت حذف پیام‌های قدیمی"""
    date_gregorian, date_jalali, time_str = get_iran_time()
    msg = f"""🖥️ سرور آنلاین
━━━━━━━━━━━━━━━━━
📅 تاریخ شمسی: {date_jalali}
📅 تاریخ میلادی: {date_gregorian}
⏰ ساعت: {time_str}
━━━━━━━━━━━━━━━━━"""

    ids = load_heartbeat_ids()
    new_id = send_telegram(msg)

    if new_id:
        ids.append(new_id)
        while len(ids) > MAX_HEARTBEATS:
            old_id = ids.pop(0)
            delete_telegram(old_id)
        save_heartbeat_ids(ids)

# --- داده و تحلیل ---
def create_exchange():
    return ccxt.toobit({
        'apiKey': TOOBIT_API_KEY,
        'secret': TOOBIT_SECRET,
        'enableRateLimit': True,
    })

def fetch_ohlcv(exchange, symbol, timeframe, limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def calculate_indicators(df):
    """محاسبه EMA، RSI، MACD، ATR"""
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal_line

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    return df

def analyze(symbol, exchange):
    """تحلیل چند تایم‌فریمی و تولید سیگنال"""
    # تایم ۴ ساعته برای تشخیص روند
    df_4h = fetch_ohlcv(exchange, symbol, '4h', 100)
    df_4h = calculate_indicators(df_4h)
    last_4h = df_4h.iloc[-1]

    trend_long = last_4h['ema20'] > last_4h['ema50']
    trend_short = last_4h['ema20'] < last_4h['ema50']

    # تایم ۱ ساعته برای ورود
    df_1h = fetch_ohlcv(exchange, symbol, '1h', 100)
    df_1h = calculate_indicators(df_1h)
    last_1h = df_1h.iloc[-1]
    prev_1h = df_1h.iloc[-2]

    entry_price = last_1h['close']
    atr = last_1h['atr']

    # --- دیباگ: چاپ وضعیت شرط‌ها ---
    print(f"  [DEBUG 4H] EMA20: {last_4h['ema20']:.4f} | EMA50: {last_4h['ema50']:.4f} | Trend: {'LONG' if trend_long else 'SHORT' if trend_short else 'NONE'}")
    print(f"  [DEBUG 1H] RSI prev: {prev_1h['rsi']:.2f} | RSI now: {last_1h['rsi']:.2f} | MACD hist: {last_1h['macd_hist']:.6f}")
    print(f"  [DEBUG CHK] trend_long={trend_long} | rsi_cond={prev_1h['rsi'] < 35 and last_1h['rsi'] > 35} | macd_cond={last_1h['macd_hist'] > 0}")

    signal = None
    sl = tp = None

    # شرط LONG
    if trend_long:
        if prev_1h['rsi'] < 35 and last_1h['rsi'] > 35 and last_1h['macd_hist'] > 0:
            signal = "LONG"
            sl = entry_price - (atr * 1.5)
            tp = entry_price + (atr * 3.0)

    # شرط SHORT
    elif trend_short:
        if prev_1h['rsi'] > 65 and last_1h['rsi'] < 65 and last_1h['macd_hist'] < 0:
            signal = "SHORT"
            sl = entry_price + (atr * 1.5)
            tp = entry_price - (atr * 3.0)

    return signal, entry_price, sl, tp, last_1h['rsi'], atr

def format_number(value):
    """فرمت عدد بر اساس بزرگی قیمت"""
    if value >= 1000:
        return f"{value:.2f}"
    elif value >= 1:
        return f"{value:.4f}"
    else:
        return f"{value:.6f}"

# --- اجرای اصلی ---
if __name__ == "__main__":
    print("=" * 50)
    print("Starting bot cycle...")

    send_heartbeat()
    print("✅ Heartbeat sent.")

    exchange = create_exchange()
    date_gregorian, date_jalali, time_str = get_iran_time()

    for symbol in SYMBOLS:
        try:
            print(f"--- Checking {symbol} ---")
            signal, entry, sl, tp, rsi, atr = analyze(symbol, exchange)

            if signal is None:
                print(f"{symbol} | Price: {entry} | RSI: {rsi:.2f} | No signal")
                continue

            if signal == "LONG":
                rr = (tp - entry) / (entry - sl)
                emoji = "🟢"
            else:
                rr = (entry - tp) / (sl - entry)
                emoji = "🔴"

            msg = f"""{emoji} سیگنال {signal}
━━━━━━━━━━━━━━━━━
📌 نماد: {symbol}
💰 قیمت ورود: {format_number(entry)}
🎯 حد سود (TP): {format_number(tp)}
🛑 حد ضرر (SL): {format_number(sl)}
📊 نسبت R:R: 1:{rr:.1f}
📈 RSI: {rsi:.2f}
📉 ATR: {format_number(atr)}
━━━━━━━━━━━━━━━━━
📅 تاریخ شمسی: {date_jalali}
📅 تاریخ میلادی: {date_gregorian}
⏰ ساعت: {time_str}
━━━━━━━━━━━━━━━━━"""

            send_telegram(msg)
            print(f"✅ {signal} signal for {symbol} sent!")

        except Exception as e:
            print(f"⚠️ خطا در {symbol}: {e}")
            continue

    print("Analysis complete.")
    print("=" * 50)