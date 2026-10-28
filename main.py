import os
import json
import time
import requests
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import jdatetime

# =====================================================================
# تنظیمات
# =====================================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
TOOBIT_API_KEY = os.environ.get("TOOBIT_API_KEY")   # اختیاری (برای داده‌ی عمومی لازم نیست)
TOOBIT_SECRET = os.environ.get("TOOBIT_SECRET")     # اختیاری

SYMBOLS = [
    # --- Core Majors (Top Liquidity) ---
    "BTC-SWAP-USDT", "ETH-SWAP-USDT", "SOL-SWAP-USDT", "XRP-SWAP-USDT",
    "BNB-SWAP-USDT", "DOGE-SWAP-USDT", "ADA-SWAP-USDT", "TRX-SWAP-USDT",
    "AVAX-SWAP-USDT", "LINK-SWAP-USDT", "DOT-SWAP-USDT", "LTC-SWAP-USDT",
    "BCH-SWAP-USDT", "TON-SWAP-USDT", "SUI-SWAP-USDT", "NEAR-SWAP-USDT",
    "APT-SWAP-USDT", "ARB-SWAP-USDT", "OP-SWAP-USDT", "ATOM-SWAP-USDT",

    # --- High Volume / Popular Altcoins ---
    "UNI-SWAP-USDT", "AAVE-SWAP-USDT", "FIL-SWAP-USDT", "INJ-SWAP-USDT",
    "TIA-SWAP-USDT", "JUP-SWAP-USDT", "ENA-SWAP-USDT", "WIF-SWAP-USDT",
    "1000PEPE-SWAP-USDT", "1000SHIB-SWAP-USDT", "WLD-SWAP-USDT", "SEI-SWAP-USDT",
    "RUNE-SWAP-USDT", "LDO-SWAP-USDT", "STX-SWAP-USDT", "IMX-SWAP-USDT",
    "HBAR-SWAP-USDT", "ICP-SWAP-USDT", "ETC-SWAP-USDT", "XLM-SWAP-USDT",

    # --- DeFi & Layer 2 ---
    "CRV-SWAP-USDT", "SNX-SWAP-USDT",
    "COMP-SWAP-USDT", "SUSHI-SWAP-USDT", "1INCH-SWAP-USDT", "GRT-SWAP-USDT",
    "DYDX-SWAP-USDT", "GMX-SWAP-USDT", "PENDLE-SWAP-USDT", "ONDO-SWAP-USDT",
    "JTO-SWAP-USDT", "PYTH-SWAP-USDT", "STRK-SWAP-USDT", "MANTA-SWAP-USDT",

    # --- Meme & Community Coins ---
    "1000BONK-SWAP-USDT", "1000FLOKI-SWAP-USDT", "MEME-SWAP-USDT", "ORDI-SWAP-USDT",
    "1000SATS-SWAP-USDT", "1000RATS-SWAP-USDT", "POPCAT-SWAP-USDT",
    "BRETT-SWAP-USDT", "TURBO-SWAP-USDT", "1000000MOG-SWAP-USDT", "DOGS-SWAP-USDT",

    # --- AI, Gaming & Metaverse ---
    "FET-SWAP-USDT", "TAO-SWAP-USDT",
    "AKT-SWAP-USDT", "ILV-SWAP-USDT",
    "AXS-SWAP-USDT", "SAND-SWAP-USDT", "MANA-SWAP-USDT", "GALA-SWAP-USDT",

    # --- Infrastructure & Others ---
    "ALGO-SWAP-USDT", "VET-SWAP-USDT", "EGLD-SWAP-USDT", "FLOW-SWAP-USDT",
    "KAVA-SWAP-USDT", "ROSE-SWAP-USDT", "KAS-SWAP-USDT",
    "THETA-SWAP-USDT", "CHZ-SWAP-USDT", "ENJ-SWAP-USDT", "ZIL-SWAP-USDT",
    "ONE-SWAP-USDT", "QTUM-SWAP-USDT", "IOTA-SWAP-USDT", "NEO-SWAP-USDT"
]

HEARTBEAT_FILE = "server_messages.json"
SIGNALS_FILE = "last_signals.json"
MAX_HEARTBEATS = 4
MAX_SIGNALS_PER_CYCLE = 3

# --- پارامترهای استراتژی ---
CROSS_LOOKBACK = 3          # کراس EMA در ۴ساعته تا چند کندل بسته‌ی اخیر معتبر است
RSI_LONG_RANGE = (50, 70)   # RSI 15m برای لانگ (بالای ۷۰ = تعقیب قیمت)
RSI_SHORT_RANGE = (30, 50)  # RSI 15m برای شورت
VOLUME_MULT = 1.0           # حجم کندل بسته باید از میانگین ۲۰ کندل قبل بیشتر باشد
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
MIN_SL_PCT = 0.003          # حداقل فاصله‌ی SL (۰.۳٪) تا کارمزد تمام سود را نخورد
MAX_ENTRY_DRIFT_ATR = 0.5   # اگر قیمت لحظه‌ای بیش از 0.5 ATR از close کندل دور شده، سیگنال کهنه است
BTC_FILTER_4H = True        # سیگنال خلاف روند ۴ساعته‌ی BTC رد شود
BTC_FILTER_1H = False       # فیلتر ۱ساعته‌ی BTC (سخت‌گیرانه؛ پیش‌فرض خاموش)

TF_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000}


# =====================================================================
# ابزارهای عمومی
# =====================================================================
def get_iran_time():
    now = datetime.now(ZoneInfo("Asia/Tehran"))
    date_gregorian = now.strftime("%Y-%m-%d")
    now_jalali = jdatetime.datetime.fromgregorian(datetime=now)
    date_jalali = now_jalali.strftime("%Y/%m/%d")
    time_str = now.strftime("%H:%M:%S")
    return date_gregorian, date_jalali, time_str


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": message}, timeout=10)
        if r.status_code == 200:
            return r.json().get("result", {}).get("message_id")
    except Exception as e:
        print(f"⚠️ خطا در ارسال تلگرام: {e}")
    return None


def delete_telegram(message_id):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "message_id": message_id}, timeout=10)
    except Exception as e:
        print(f"⚠️ خطا در حذف پیام: {e}")


def format_number(value):
    if value >= 1000:
        return f"{value:.2f}"
    elif value >= 1:
        return f"{value:.4f}"
    else:
        return f"{value:.6f}"


# =====================================================================
# Heartbeat
# =====================================================================
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


# =====================================================================
# مدیریت معاملات
# =====================================================================
def load_signals():
    try:
        with open(SIGNALS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_signals(data):
    with open(SIGNALS_FILE, "w") as f:
        json.dump(data, f)


def check_trade_result(exchange, symbol, data):
    """
    نتیجه‌ی معامله را با high/low کندل‌های ۵ دقیقه‌ای از زمان ورود بررسی می‌کند.
    اگر در یک کندل هر دو (TP و SL) لمس شده باشند، محافظه‌کارانه SL حساب می‌شود.
    خروجی: (result, price) یا (None, None)
    """
    direction = data["direction"]
    tp = data["tp"]
    sl = data["sl"]
    opened_ms = data.get("opened_ms")

    if opened_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe="5m", since=int(opened_ms), limit=1000)
        except Exception as e:
            print(f"⚠️ خطا در دریافت کندل‌های {symbol}: {e}")
            candles = []
        for c in candles:
            if c[0] < opened_ms:
                continue
            high, low = c[2], c[3]
            if direction == "LONG":
                hit_sl, hit_tp = low <= sl, high >= tp
            else:
                hit_sl, hit_tp = high >= sl, low <= tp
            if hit_sl:
                return "SL", sl
            if hit_tp:
                return "TP", tp

    # پوشش فاصله‌ی کوتاه بین ورود و اولین کندل / معاملات قدیمی بدون opened_ms
    price = exchange.fetch_ticker(symbol)["last"]
    if direction == "LONG":
        if price <= sl:
            return "SL", price
        if price >= tp:
            return "TP", price
    else:
        if price >= sl:
            return "SL", price
        if price <= tp:
            return "TP", price
    return None, None


def update_open_trades(exchange):
    signals = load_signals()
    updated = False
    date_gregorian, date_jalali, time_str = get_iran_time()

    for symbol, data in list(signals.items()):
        if data.get("status") != "OPEN":
            continue
        try:
            result, close_price = check_trade_result(exchange, symbol, data)
            if not result:
                continue

            entry = data["entry"]
            direction = data["direction"]
            signals[symbol]["status"] = "CLOSED"
            signals[symbol]["result"] = result
            updated = True
            emoji = "🎯" if result == "TP" else "🛑"
            title = "حد سود (TP) لمس شد" if result == "TP" else "حد ضرر (SL) لمس شد"
            print(f"{emoji} {symbol} {direction} hit {result} at {close_price}")

            close_msg = f"""{emoji} بسته شدن معامله
━━━━━━━━━━━━━━━━━
📌 نماد: {symbol}
📊 جهت: {direction}
🏁 نتیجه: {title}
💰 قیمت بسته شدن: {format_number(close_price)}
🚪 قیمت ورود: {format_number(entry)}
━━━━━━━━━━━━━━━━━
📅 تاریخ شمسی: {date_jalali}
📅 تاریخ میلادی: {date_gregorian}
⏰ ساعت: {time_str}
━━━━━━━━━━━━━━━━━"""
            send_telegram(close_msg)
        except Exception as e:
            print(f"⚠️ خطا در بررسی {symbol}: {e}")
            continue
    if updated:
        save_signals(signals)


# =====================================================================
# داده و اندیکاتورها
# =====================================================================
def create_exchange():
    # برای داده‌ی عمومی نیازی به کلید نیست؛ اگر کلید تنظیم نشده باشد، بدون آن ساخته می‌شود
    params = {"enableRateLimit": True}
    if TOOBIT_API_KEY and TOOBIT_SECRET:
        params.update({"apiKey": TOOBIT_API_KEY, "secret": TOOBIT_SECRET})
    return ccxt.toobit(params)


def fetch_ohlcv(exchange, symbol, timeframe, limit=150):
    """کندل‌ها را می‌گیرد و کندل در حال تشکیل (ناقص) را حذف می‌کند؛ پس iloc[-1] همیشه آخرین کندل بسته است."""
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    now_ms = int(time.time() * 1000)
    if len(df) and df["timestamp"].iloc[-1] + TF_MS[timeframe] > now_ms:
        df = df.iloc[:-1]
    return df.reset_index(drop=True)


def calculate_indicators(df):
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    # RSI به روش Wilder (مطابق TradingView)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    df["rsi"] = rsi

    # ATR به روش Wilder
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return df


def get_data(exchange, symbol, timeframe, cache):
    """دریافت داده + اندیکاتور با کش (هر نماد/تایم‌فریم فقط یک بار در هر چرخه)."""
    key = (symbol, timeframe)
    if key not in cache:
        df = fetch_ohlcv(exchange, symbol, timeframe, 150)
        if len(df) < 40:
            raise ValueError(f"داده‌ی کافی برای {symbol} {timeframe} نیست ({len(df)} کندل)")
        cache[key] = calculate_indicators(df)
    return cache[key]


def trend_of(df):
    last = df.iloc[-1]
    if last["ema9"] > last["ema21"]:
        return "LONG"
    if last["ema9"] < last["ema21"]:
        return "SHORT"
    return "NONE"


def get_btc_trend(exchange, cache):
    """روند BTC را (با کندل بسته) در ۴ساعته و ۱ساعته برمی‌گرداند."""
    try:
        t4 = trend_of(get_data(exchange, "BTC-SWAP-USDT", "4h", cache))
        t1 = trend_of(get_data(exchange, "BTC-SWAP-USDT", "1h", cache))
        return t4, t1
    except Exception as e:
        print(f"⚠️ خطا در دریافت روند بیت‌کوین: {e}")
        return "UNKNOWN", "UNKNOWN"


def find_recent_cross(df, direction, lookback=CROSS_LOOKBACK):
    """
    آخرین کراس EMA9/EMA21 را در `lookback` کندل بسته‌ی اخیر پیدا می‌کند.
    خروجی: (age, cross_timestamp) که age=0 یعنی آخرین کندل بسته؛ یا (None, None).
    کراس فقط وقتی معتبر است که وضعیت فعلی هنوز هم‌جهت باشد.
    """
    now = trend_of(df)
    if now != direction:
        return None, None
    n = len(df)
    for age in range(lookback):
        i = n - 1 - age
        cur, prev = df.iloc[i], df.iloc[i - 1]
        if direction == "LONG" and prev["ema9"] <= prev["ema21"] and cur["ema9"] > cur["ema21"]:
            return age, int(df["timestamp"].iloc[i])
        if direction == "SHORT" and prev["ema9"] >= prev["ema21"] and cur["ema9"] < cur["ema21"]:
            return age, int(df["timestamp"].iloc[i])
    return None, None


# =====================================================================
# تحلیل (بدون ارسال؛ فقط کاندیدا تولید می‌کند)
# =====================================================================
def analyze(symbol, exchange, cache):
    """
    اگر نماد شرایط ورود داشته باشد یک dict کاندیدا (با امتیاز) برمی‌گرداند، وگرنه None.
    مراحل به ترتیب ارزان‌ترین فیلتر است تا درخواست‌های اضافه نشود.
    """
    # ۱) ۴ساعته: کراس تازه (حداکثر CROSS_LOOKBACK کندل قبل)
    df_4h = get_data(exchange, symbol, "4h", cache)
    direction = None
    cross_age = cross_ts = None
    for d in ("LONG", "SHORT"):
        age, ts = find_recent_cross(df_4h, d)
        if age is not None:
            direction, cross_age, cross_ts = d, age, ts
            break
    if direction is None:
        print(f"  {symbol}: no 4H cross")
        return None

    # ۲) ۱ساعته: تایید روند
    df_1h = get_data(exchange, symbol, "1h", cache)
    last_1h = df_1h.iloc[-1]
    if trend_of(df_1h) != direction:
        print(f"  {symbol}: {direction} 4H cross but 1H trend disagrees")
        return None

    # ۳) ۱۵ دقیقه: مومنتوم و حجم (همه روی کندل بسته)
    df_15m = get_data(exchange, symbol, "15m", cache)
    last_15m = df_15m.iloc[-1]
    rsi = float(last_15m["rsi"])
    atr = float(last_15m["atr"])
    close_15m = float(last_15m["close"])

    if np.isnan(rsi) or np.isnan(atr) or atr <= 0:
        print(f"  {symbol}: indicators not ready")
        return None

    rsi_lo, rsi_hi = RSI_LONG_RANGE if direction == "LONG" else RSI_SHORT_RANGE
    rsi_ok = rsi_lo <= rsi <= rsi_hi

    current_volume = float(last_15m["volume"])
    avg_volume = float(df_15m["volume"].iloc[-21:-1].mean())
    vol_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0
    vol_ok = vol_ratio >= VOLUME_MULT

    print(f"  {symbol}: {direction} 4H cross(age={cross_age}) | 1H ok | "
          f"RSI {rsi:.1f} ok={rsi_ok} | Vol x{vol_ratio:.2f} ok={vol_ok}")

    if not (rsi_ok and vol_ok):
        return None

    # حداقل فاصله‌ی SL نسبت به قیمت
    if (atr * SL_ATR_MULT) / close_15m < MIN_SL_PCT:
        print(f"  {symbol}: SL too tight ({(atr * SL_ATR_MULT) / close_15m:.2%}) -> skipped")
        return None

    # ۴) امتیاز برای رتبه‌بندی (۰ تا ۱)
    vol_score = min(vol_ratio, 3.0) / 3.0
    fresh_score = (CROSS_LOOKBACK - cross_age) / CROSS_LOOKBACK
    atr_1h = float(last_1h["atr"]) if not np.isnan(last_1h["atr"]) else 0.0
    gap_1h = abs(float(last_1h["ema9"]) - float(last_1h["ema21"]))
    trend_score = min(gap_1h / atr_1h, 1.0) if atr_1h > 0 else 0.0
    rsi_score = min(abs(rsi - 50) / 20.0, 1.0)
    score = 0.35 * vol_score + 0.25 * fresh_score + 0.20 * trend_score + 0.20 * rsi_score

    return {
        "symbol": symbol,
        "direction": direction,
        "close": close_15m,
        "atr": atr,
        "rsi": rsi,
        "volume": current_volume,
        "avg_volume": avg_volume,
        "cross_ts": cross_ts,
        "score": score,
    }


# =====================================================================
# اجرای اصلی
# =====================================================================
def build_signal_message(c, entry, sl, tp, rr):
    date_gregorian, date_jalali, time_str = get_iran_time()
    emoji = "🟢" if c["direction"] == "LONG" else "🔴"
    return f"""{emoji} سیگنال {c['direction']}
━━━━━━━━━━━━━━━━━
📌 نماد: {c['symbol']}
💰 قیمت ورود: {format_number(entry)}
🎯 حد سود (TP): {format_number(tp)}
🛑 حد ضرر (SL): {format_number(sl)}
📊 نسبت R:R: 1:{rr:.1f}
📈 RSI: {c['rsi']:.2f}
📉 ATR: {format_number(c['atr'])}
📊 Volume: {c['volume']:.2f} (Avg: {c['avg_volume']:.2f})
━━━━━━━━━━━━━━━━━
📅 تاریخ شمسی: {date_jalali}
📅 تاریخ میلادی: {date_gregorian}
⏰ ساعت: {time_str}
━━━━━━━━━━━━━━━━━"""


if __name__ == "__main__":
    print("=" * 50)
    print("Starting bot cycle (closed candles + 4H cross window + BTC filter + volume + ranking)...")

    send_heartbeat()
    print("✅ Heartbeat sent.")

    exchange = create_exchange()

    update_open_trades(exchange)
    signals_data = load_signals()

    cache = {}
    btc_trend_4h, btc_trend_1h = get_btc_trend(exchange, cache)
    print(f"🌍 BTC 4H Trend: {btc_trend_4h} | BTC 1H Trend: {btc_trend_1h}")

    # ---------- مرحله ۱: تحلیل همه‌ی نمادها و جمع‌آوری کاندیداها ----------
    candidates = []
    for symbol in SYMBOLS:
        try:
            print(f"--- Checking {symbol} ---")

            prev = signals_data.get(symbol)
            if prev and prev.get("status") == "OPEN":
                print(f"⏳ {symbol} has an OPEN trade. Skipping.")
                continue

            c = analyze(symbol, exchange, cache)
            if c is None:
                continue

            # جلوگیری از سیگنال تکراری روی همان کراس (بعد از بسته شدن معامله)
            if prev and prev.get("cross_ts") == c["cross_ts"] and prev.get("direction") == c["direction"]:
                print(f"♻️ {symbol} {c['direction']}: this 4H cross was already signaled. Skipping.")
                continue

            # فیلتر BTC (برای خود BTC اعمال نمی‌شود)
            if symbol != "BTC-SWAP-USDT":
                if BTC_FILTER_4H and btc_trend_4h in ("LONG", "SHORT") and c["direction"] != btc_trend_4h:
                    print(f"🚫 {symbol} {c['direction']} rejected. BTC 4H trend is {btc_trend_4h}.")
                    continue
                if BTC_FILTER_1H and btc_trend_1h in ("LONG", "SHORT") and c["direction"] != btc_trend_1h:
                    print(f"🚫 {symbol} {c['direction']} rejected. BTC 1H trend is {btc_trend_1h}.")
                    continue

            candidates.append(c)
            print(f"✅ Candidate: {symbol} {c['direction']} score={c['score']:.3f}")
        except Exception as e:
            print(f"⚠️ خطا در {symbol}: {e}")
            continue

    # ---------- مرحله ۲: رتبه‌بندی و ارسال بهترین‌ها ----------
    candidates.sort(key=lambda x: x["score"], reverse=True)
    print(f"📋 {len(candidates)} candidate(s): " +
          ", ".join(f"{c['symbol']}({c['score']:.2f})" for c in candidates))

    signals_sent_this_cycle = 0
    for c in candidates:
        if signals_sent_this_cycle >= MAX_SIGNALS_PER_CYCLE:
            break
        symbol = c["symbol"]
        try:
            # قیمت لحظه‌ای؛ اگر خیلی از close کندل دور شده، سیگنال کهنه است
            entry = float(exchange.fetch_ticker(symbol)["last"])
            if abs(entry - c["close"]) > c["atr"] * MAX_ENTRY_DRIFT_ATR:
                print(f"⌛ {symbol}: price drifted too far from signal candle. Skipping.")
                continue

            atr = c["atr"]
            if c["direction"] == "LONG":
                sl = entry - atr * SL_ATR_MULT
                tp = entry + atr * TP_ATR_MULT
                rr = (tp - entry) / (entry - sl)
            else:
                sl = entry + atr * SL_ATR_MULT
                tp = entry - atr * TP_ATR_MULT
                rr = (entry - tp) / (sl - entry)

            if sl <= 0 or tp <= 0:
                print(f"⚠️ {symbol}: invalid SL/TP. Skipping.")
                continue

            send_telegram(build_signal_message(c, entry, sl, tp, rr))

            signals_data[symbol] = {
                "direction": c["direction"],
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "status": "OPEN",
                "cross_ts": c["cross_ts"],
                "score": round(c["score"], 4),
                "opened_ms": int(time.time() * 1000),
                "time": datetime.now(timezone.utc).isoformat(),
            }
            save_signals(signals_data)
            signals_sent_this_cycle += 1
            print(f"✅ {c['direction']} signal for {symbol} sent! "
                  f"({signals_sent_this_cycle}/{MAX_SIGNALS_PER_CYCLE})")
        except Exception as e:
            print(f"⚠️ خطا در ارسال {symbol}: {e}")
            continue

    print("Analysis complete.")
    print("=" * 50)
