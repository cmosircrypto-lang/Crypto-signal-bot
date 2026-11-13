"""
ربات‌های سیگنال با تمرکز روی کیفیت بالا (Win Rate)
  1) crypto-signal-bot      : کراس EMA 9/21 در ۴ساعته + تایید قوی ۱ساعته
  2) crypto-signal-bot-v10  : شکست رنج تمیز در ۱ساعته با حجم قوی
  3) crypto-signal-bot-v20  : MTF Pullback (Daily EMA50/200 + 1H Swing/Donchian)
  4) crypto-signal-bot-v30  : AMD + نقدینگی سشن (رنج آسیا → سوییپ → FVG)
  5) crypto-signal-bot-v40  : رژیم بازار (ADX) — نسخه سخت‌تر کیفیت
  6) crypto-signal-bot-v50  : SMC/ICT ترکیبی
  7) crypto-signal-bot-v60  : فیلتر BTC (1D+4H) + پولبک هم‌جهت روی نمادها
  8) crypto-signal-bot-v70  : SMC Winner (نسخه بهینه‌شده با بک‌تست ۱ ساله)
"""
import os
import json
import time
import requests
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import jdatetime
import excel_log

# =====================================================================
# تنظیمات عمومی
# =====================================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
TOOBIT_API_KEY = os.environ.get("TOOBIT_API_KEY")
TOOBIT_SECRET = os.environ.get("TOOBIT_SECRET")

SYMBOLS = [
    # --- Core Majors ---
    "BTC-SWAP-USDT", "ETH-SWAP-USDT", "SOL-SWAP-USDT", "XRP-SWAP-USDT",
    "BNB-SWAP-USDT", "DOGE-SWAP-USDT", "ADA-SWAP-USDT", "TRX-SWAP-USDT",
    "AVAX-SWAP-USDT", "LINK-SWAP-USDT", "DOT-SWAP-USDT", "LTC-SWAP-USDT",
    "BCH-SWAP-USDT", "TON-SWAP-USDT", "SUI-SWAP-USDT", "NEAR-SWAP-USDT",
    "APT-SWAP-USDT", "ARB-SWAP-USDT", "OP-SWAP-USDT", "ATOM-SWAP-USDT",

    # --- High Volume / Popular ---
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

    # --- Meme & Community ---
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

BTC_SYMBOL = "BTC-SWAP-USDT"
MAX_HEARTBEATS = 2
MAX_SIGNALS_PER_RUN = 5
SUMMARY_FILE = "daily_summary.json"
SUMMARY_HOUR = 8

MIN_SL_PCT = 0.0035
MAX_ENTRY_DRIFT_ATR = 0.45

TF_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

# --- ربات ۱: EMA Cross (بهبود یافته) ---
B1_CROSS_LOOKBACK = 3
B1_RSI_LONG = (45, 68)
B1_RSI_SHORT = (32, 55)
B1_VOLUME_MULT = 1.5
B1_SL_ATR = 1.6
B1_TP_ATR = 3.4
B1_MIN_SCORE = 0.70
B1_MIN_ADX_4H = 18
B1_MIN_ADX_1H = 16
B1_MIN_SLOPE_ATR = 0.08

# --- ربات ۲: Breakout (کیفیت بالاتر) ---
B2_LOOKBACK = 18
B2_VOLUME_MULT = 1.8
B2_CLOSE_POS = 0.72
B2_MAX_EXTENSION_ATR = 0.85
B2_RSI_LONG = (52, 72)
B2_RSI_SHORT = (28, 48)
B2_SL_ATR, B2_TP_ATR = 1.4, 3.0
B2_MAX_RANGE_ATR = 6.5
B2_MIN_BODY_ATR = 0.45
B2_MIN_SCORE = 0.62
B2_MIN_ADX = 18

# --- ربات ۳: MTF Pullback ---
B3_RSI_LONG = (31, 45)
B3_RSI_SHORT = (55, 69)
B3_NEAR_PCT = 0.015
B3_BODY_MIN = 0.42
B3_ATR_MULT = 0.85
B3_RR = 2.1
B3_MIN_SEP = 0.005
B3_MIN_PULLBACK = 1.005
B3_COOLDOWN_HOURS = 3
B3_MIN_SCORE = 0.55

# --- ربات v30: AMD (سخت‌گیرانه‌تر) ---
B30_ASIA_START_H = 0
B30_ASIA_END_H = 8
B30_ACTIVE_START_H = 7
B30_ACTIVE_END_H = 20
B30_SWEEP_MAX_AGE_1H = 4
B30_FVG_RECENT = 8
B30_SL_BUFFER_ATR = 0.15
B30_MIN_RR = 2.1
B30_MAX_RISK_ATR = 2.0
B30_MIN_RANGE_ATR = 1.0
B30_MAX_RANGE_ATR = 5.0
B30_MIN_SCORE = 0.60
B30_COOLDOWN_HOURS = 4
B30_TREND_GAP_4H_ATR = 0.30

# --- ربات ۴: Regime ADX ---
B4_ADX_TREND = 24
B4_ADX_RANGE = 18
B4_TREND_GAP_4H_ATR = 0.30
B4_TREND_TOUCH_ATR = 0.18
B4_TREND_MAX_EXT_ATR = 0.70
B4_TREND_RSI_LONG = (38, 55)
B4_TREND_RSI_SHORT = (45, 62)
B4_TREND_SL_ATR, B4_TREND_TP_ATR = 1.3, 2.8
B4_TREND_MIN_VOL = 0.90
B4_RANGE_RSI_LONG_MAX = 28
B4_RANGE_RSI_SHORT_MIN = 72
B4_RANGE_SL_BUFFER_ATR = 0.45
B4_RANGE_MIN_RR = 2.1
B4_MIN_SCORE = 0.64
B4_COOLDOWN_HOURS = 3

# --- ربات ۵: SMC ---
B5_LOOKBACK_1H = 24
B5_SWEEP_RECENT = 2
B5_RANGE_BARS = 40
B5_FVG_RECENT = 8
B5_TREND_GAP_4H_ATR = 0.25
B5_SL_BUFFER_ATR = 0.12
B5_MIN_RR = 2.1
B5_MAX_RISK_ATR = 2.2
B5_MIN_SCORE = 0.60
B5_COOLDOWN_HOURS = 4
B5_MIN_FVG_ATR = 0.25
B5_MIN_VOLUME_RATIO = 0.90

# --- ربات ۶: BTC Filter ---
B6_MIN_GAP_ATR = 0.28
B6_MIN_ADX = 20
B6_MAX_EXT_ATR = 1.3
B6_1H_ALIGN_BONUS = 0.12
B6_ALT_GAP_ATR = 0.22
B6_TOUCH_ATR = 0.22
B6_MAX_EXT_ALT_ATR = 1.0
B6_RSI_LONG = (35, 55)
B6_RSI_SHORT = (45, 65)
B6_VOLUME_MULT = 1.25
B6_SL_ATR, B6_TP_ATR = 1.4, 2.95
B6_MIN_SCORE = 0.60
B6_COOLDOWN_HOURS = 3

# --- ربات ۸: SMC Winner (نسخه برنده بک‌تست) ---
B8_SWEEP_LOOKBACK = 10
B8_TP_ATR_MULT = 2.5
B8_COOLDOWN_HOURS = 4


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


def send_telegram(message, max_retries=3, timeout=20):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                url,
                json={"chat_id": CHAT_ID, "text": message},
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json().get("result", {}).get("message_id")
            print(f"⚠️ تلگرام status={r.status_code} (تلاش {attempt}/{max_retries})")
        except Exception as e:
            print(f"⚠️ خطا در ارسال تلگرام (تلاش {attempt}/{max_retries}): {e}")
        if attempt < max_retries:
            time.sleep(2 * attempt)
    return None


def delete_telegram(message_id, max_retries=2, timeout=15):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteMessage"
    for attempt in range(1, max_retries + 1):
        try:
            requests.post(
                url,
                json={"chat_id": CHAT_ID, "message_id": message_id},
                timeout=timeout,
            )
            return
        except Exception as e:
            print(f"⚠️ خطا در حذف پیام (تلاش {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(1)


def format_number(value):
    if value >= 1000:
        return f"{value:.2f}"
    elif value >= 1:
        return f"{value:.4f}"
    else:
        return f"{value:.6f}"


def load_json(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def append_trade_csv(bot_name, record):
    import csv
    fname = f"trades_{bot_name.replace('crypto-signal-bot', 'bot')}.csv"
    file_exists = os.path.exists(fname)
    fields = ["bot", "symbol", "direction", "entry", "sl", "tp", "exit",
              "result", "pnl_r", "setup", "score", "opened_ms", "closed_ms"]
    risk = abs(float(record.get("entry", 0) or 0) - float(record.get("sl", 0) or 0))
    pnl_r = ""
    if risk > 0 and record.get("result") in ("TP", "SL"):
        if record["result"] == "TP":
            pnl_r = round(abs(float(record["tp"]) - float(record["entry"])) / risk, 3)
        else:
            pnl_r = -1.0
    row = {
        "bot": bot_name,
        "symbol": record.get("symbol", ""),
        "direction": record.get("direction"),
        "entry": record.get("entry"),
        "sl": record.get("sl"),
        "tp": record.get("tp"),
        "exit": record.get("close_price"),
        "result": record.get("result"),
        "pnl_r": pnl_r,
        "setup": record.get("setup", ""),
        "score": record.get("score", ""),
        "opened_ms": record.get("opened_ms", ""),
        "closed_ms": record.get("closed_ms", ""),
    }
    try:
        with open(fname, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"⚠️ CSV log error: {e}")


def now_ms():
    return int(time.time() * 1000)


# =====================================================================
# داده و اندیکاتورها
# =====================================================================
class Ctx:
    def __init__(self, exchange):
        self.exchange = exchange
        self.cache = {}
        self.tickers = {}
        self.btc_filter = None


def create_exchange():
    params = {"enableRateLimit": True}
    if TOOBIT_API_KEY and TOOBIT_SECRET:
        params.update({"apiKey": TOOBIT_API_KEY, "secret": TOOBIT_SECRET})
    return ccxt.toobit(params)


def fetch_ohlcv(exchange, symbol, timeframe, limit=150):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if len(df) and df["timestamp"].iloc[-1] + TF_MS[timeframe] > now_ms():
        df = df.iloc[:-1]
    return df.reset_index(drop=True)


def calculate_indicators(df):
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

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

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_safe = df["atr"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_safe
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    bb_mid = df["close"].rolling(20).mean()
    bb_sd = df["close"].rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_up"] = bb_mid + 2 * bb_sd
    df["bb_lo"] = bb_mid - 2 * bb_sd
    return df


def get_data(ctx, symbol, timeframe):
    key = (symbol, timeframe)
    if key not in ctx.cache:
        limit = 300 if timeframe == "1d" else 150
        df = fetch_ohlcv(ctx.exchange, symbol, timeframe, limit)

        if len(df) < 40:
            raise ValueError(f"داده‌ی کافی برای {symbol} {timeframe} نیست ({len(df)} کندل)")
        ctx.cache[key] = calculate_indicators(df)
    return ctx.cache[key]


def get_price(ctx, symbol):
    if symbol not in ctx.tickers:
        ctx.tickers[symbol] = float(ctx.exchange.fetch_ticker(symbol)["last"])
    return ctx.tickers[symbol]


def trend_of(df):
    last = df.iloc[-1]
    if last["ema9"] > last["ema21"]:
        return "LONG"
    if last["ema9"] < last["ema21"]:
        return "SHORT"
    return "NONE"


def get_btc_trend(ctx):
    try:
        t4 = trend_of(get_data(ctx, BTC_SYMBOL, "4h"))
        t1 = trend_of(get_data(ctx, BTC_SYMBOL, "1h"))
        return t4, t1
    except Exception as e:
        print(f"⚠️ خطا در دریافت روند بیت‌کوین: {e}")
        return "UNKNOWN", "UNKNOWN"


def atr_levels(direction, atr, sl_mult, tp_mult):
    def levels(entry):
        if direction == "LONG":
            return entry - atr * sl_mult, entry + atr * tp_mult
        return entry + atr * sl_mult, entry - atr * tp_mult
    return levels


def ratio(cur, avg):
    return cur / avg if avg and avg > 0 else 0.0


# =====================================================================
# استراتژی ۱: کراس EMA
# =====================================================================
def find_recent_cross(df, direction, lookback=B1_CROSS_LOOKBACK):
    if trend_of(df) != direction:
        return None, None
    n = len(df)
    for age in range(lookback):
        i = n - 1 - age
        if i < 1:
            break
        cur, prev = df.iloc[i], df.iloc[i - 1]
        if direction == "LONG" and prev["ema9"] <= prev["ema21"] and cur["ema9"] > cur["ema21"]:
            return age, int(df["timestamp"].iloc[i])
        if direction == "SHORT" and prev["ema9"] >= prev["ema21"] and cur["ema9"] < cur["ema21"]:
            return age, int(df["timestamp"].iloc[i])
    return None, None


def analyze_cross(symbol, ctx):
    df_4h = get_data(ctx, symbol, "4h")
    direction = None
    cross_age = None
    cross_ts = None

    for d in ("LONG", "SHORT"):
        age, ts = find_recent_cross(df_4h, d, lookback=B1_CROSS_LOOKBACK)
        if age is not None:
            direction = d
            cross_age = age
            cross_ts = ts
            break

    if direction is None:
        return None

    last4 = df_4h.iloc[-1]
    atr_4h = float(last4["atr"])
    if np.isnan(atr_4h) or atr_4h <= 0:
        return None

    adx_4h = float(last4["adx"])
    if np.isnan(adx_4h):
        return None

    if adx_4h < B1_MIN_ADX_4H:
        print(f"  {symbol}: weak 4H ADX ({adx_4h:.1f})")
        return None

    ema9_4h = float(last4["ema9"])
    ema21_4h = float(last4["ema21"])
    ema50_4h = float(last4["ema50"])

    gap_4h = abs(ema9_4h - ema21_4h)
    if gap_4h < 0.35 * atr_4h:
        print(f"  {symbol}: weak 4H trend")
        return None

    ema9_slope = (ema9_4h - float(df_4h["ema9"].iloc[-4]))
    slope_atr = (abs(ema9_slope) / atr_4h)

    if slope_atr < B1_MIN_SLOPE_ATR:
        print(f"  {symbol}: EMA slope too weak")
        return None

    if direction == "LONG" and ema9_slope <= 0:
        return None
    if direction == "SHORT" and ema9_slope >= 0:
        return None

    if direction == "LONG":
        if ema21_4h <= ema50_4h:
            print(f"  {symbol}: LONG cross but EMA21 below EMA50")
            return None
    else:
        if ema21_4h >= ema50_4h:
            print(f"  {symbol}: SHORT cross but EMA21 above EMA50")
            return None

    df_1h = get_data(ctx, symbol, "1h")
    if trend_of(df_1h) != direction:
        return None

    last_1h = df_1h.iloc[-1]
    atr_1h = float(last_1h["atr"])
    if np.isnan(atr_1h) or atr_1h <= 0:
        return None

    adx_1h = float(last_1h["adx"])
    if np.isnan(adx_1h):
        return None

    if adx_1h < B1_MIN_ADX_1H:
        print(f"  {symbol}: weak 1H ADX ({adx_1h:.1f})")
        return None

    gap_1h = abs(float(last_1h["ema9"]) - float(last_1h["ema21"]))
    if gap_1h < 0.40 * atr_1h:
        print(f"  {symbol}: weak 1H trend")
        return None

    vol_1h = float(last_1h["volume"])
    avg_vol_1h = float(df_1h["volume"].iloc[-21:-1].mean())
    vol_ratio_1h = ratio(vol_1h, avg_vol_1h)

    if vol_ratio_1h < B1_VOLUME_MULT:
        print(f"  {symbol}: weak volume (x{vol_ratio_1h:.2f})")
        return None

    df_15m = get_data(ctx, symbol, "15m")
    last = df_15m.iloc[-1]
    rsi = float(last["rsi"])
    if np.isnan(rsi):
        return None

    lo, hi = (B1_RSI_LONG if direction == "LONG" else B1_RSI_SHORT)
    if not (lo <= rsi <= hi):
        print(f"  {symbol}: RSI {rsi:.1f} not in range")
        return None

    close = float(last["close"])
    if (atr_1h * B1_SL_ATR) / close < MIN_SL_PCT:
        return None

    trend1_score = (min(gap_1h / atr_1h, 1.5) / 1.5)
    trend4_score = (min(gap_4h / atr_4h, 1.5) / 1.5)
    vol_score = (min(vol_ratio_1h, 3.0) / 3.0)
    cross_score = (B1_CROSS_LOOKBACK - cross_age) / B1_CROSS_LOOKBACK
    rsi_score = min(abs(rsi - 50) / 20.0, 1.0)
    adx_score = min(adx_4h / 35.0, 1.0)

    score = (0.25 * trend1_score + 0.25 * trend4_score + 0.20 * adx_score + 
             0.15 * vol_score + 0.10 * cross_score + 0.05 * rsi_score)

    print(f"  {symbol}: {direction} | score={score:.3f} | ADX4={adx_4h:.1f} | ADX1={adx_1h:.1f} | vol={vol_ratio_1h:.2f}")

    if score < B1_MIN_SCORE:
        print(f"  {symbol}: score too low ({score:.3f})")
        return None

    return {
        "symbol": symbol, "direction": direction, "close": close,
        "atr": atr_1h, "rsi": rsi, "volume": vol_1h, "avg_volume": avg_vol_1h,
        "signal_key": cross_ts, "score": score,
        "setup": f"EMA Cross PRO (4H age {cross_age})",
        "levels": atr_levels(direction, atr_1h, B1_SL_ATR, B1_TP_ATR),
    }


# =====================================================================
# استراتژی ۲: Breakout
# =====================================================================
def analyze_breakout(symbol, ctx):
    df = get_data(ctx, symbol, "1h")
    lb = B2_LOOKBACK
    if len(df) < lb + 5:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    range_high = float(df["high"].iloc[-(lb + 1):-1].max())
    range_low = float(df["low"].iloc[-(lb + 1):-1].min())
    range_size = range_high - range_low

    close = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    open_ = float(last["open"])
    atr = float(last["atr"])
    rsi = float(last["rsi"])
    rng = high - low
    body = abs(close - open_)

    if np.isnan(atr) or atr <= 0 or rng <= 0 or range_size <= 0:
        return None

    if range_size > B2_MAX_RANGE_ATR * atr:
        print(f"  {symbol}: range too wide ({range_size / atr:.1f} ATR)")
        return None

    direction = level = None

    if (close > range_high and float(prev["close"]) <= range_high and
            (close - low) / rng >= B2_CLOSE_POS and (close - range_high) <= B2_MAX_EXTENSION_ATR * atr):
        direction, level = "LONG", range_high

    elif (close < range_low and float(prev["close"]) >= range_low and
          (high - close) / rng >= B2_CLOSE_POS and (range_low - close) <= B2_MAX_EXTENSION_ATR * atr):
        direction, level = "SHORT", range_low

    if direction is None:
        return None

    if body < B2_MIN_BODY_ATR * atr:
        print(f"  {symbol}: {direction} breakout but body too weak ({body / atr:.2f} ATR)")
        return None

    adx = float(last["adx"]) if not np.isnan(last["adx"]) else 0.0
    if adx < B2_MIN_ADX:
        print(f"  {symbol}: breakout ADX too weak ({adx:.1f})")
        return None

    volume = float(last["volume"])
    avg_volume = float(df["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(volume, avg_volume)

    lo, hi = B2_RSI_LONG if direction == "LONG" else B2_RSI_SHORT
    rsi_ok = lo <= rsi <= hi
    vol_ok = vol_ratio >= B2_VOLUME_MULT

    df_4h = get_data(ctx, symbol, "4h")
    l4 = df_4h.iloc[-1]
    slow_trend = "LONG" if l4["ema21"] > l4["ema50"] else "SHORT" if l4["ema21"] < l4["ema50"] else "NONE"
    trend_ok = slow_trend == direction

    print(f"  {symbol}: {direction} breakout 1H | RSI {rsi:.1f} ok={rsi_ok} | Vol x{vol_ratio:.2f} ok={vol_ok} | 4H ok={trend_ok}")

    if not (rsi_ok and vol_ok and trend_ok):
        reasons = []
        if not rsi_ok: reasons.append(f"RSI {rsi:.1f} خارج {lo}-{hi}")
        if not vol_ok: reasons.append(f"حجم {vol_ratio:.2f} < {B2_VOLUME_MULT}")
        if not trend_ok: reasons.append(f"روند ۴ساعته {slow_trend} مخالف {direction}")
        print(f"    -> rejected: " + " | ".join(reasons))
        return None

    if (atr * B2_SL_ATR) / close < MIN_SL_PCT:
        return None

    strength = min(abs(close - level) / atr, 1.0)
    body_score = min(body / atr, 1.5) / 1.5
    atr_4h = float(l4["atr"]) if not np.isnan(l4["atr"]) else 0.0
    gap_4h = abs(float(l4["ema21"]) - float(l4["ema50"]))
    trend_score = min(gap_4h / atr_4h, 1.0) if atr_4h > 0 else 0.0

    score = (0.35 * min(vol_ratio, 4.0) / 4.0 + 0.25 * strength +
             0.20 * body_score + 0.20 * trend_score)

    if score < B2_MIN_SCORE:
        print(f"    -> rejected (score {score:.3f} < {B2_MIN_SCORE})")
        return None

    return {
        "symbol": symbol, "direction": direction, "close": close,
        "atr": atr, "rsi": rsi, "volume": volume, "avg_volume": avg_volume,
        "signal_key": int(last["timestamp"]), "score": score,
        "setup": f"Breakout {lb}x1H ({'high' if direction == 'LONG' else 'low'} {format_number(level)})",
        "levels": atr_levels(direction, atr, B2_SL_ATR, B2_TP_ATR),
    }


# =====================================================================
# استراتژی ۳: MTF Pullback (Daily Bias + 1H Entry)
# =====================================================================
def analyze_mtf_pullback(symbol, ctx):
    try:
        df_1d = get_data(ctx, symbol, "1d")
    except Exception:
        return None
    if len(df_1d) < 200:
        return None

    ema50_d = df_1d["close"].ewm(span=50, adjust=False).mean()
    ema200_d = df_1d["close"].ewm(span=200, adjust=False).mean()
    e50 = float(ema50_d.iloc[-1])
    e200 = float(ema200_d.iloc[-1])
    close_d = float(df_1d["close"].iloc[-1])
    if any(np.isnan(x) for x in (e50, e200)) or e200 <= 0:
        return None

    sep = abs(e50 - e200) / e200
    if close_d > e200 and e50 > e200 and sep >= B3_MIN_SEP:
        direction = "LONG"
    elif close_d < e200 and e50 < e200 and sep >= B3_MIN_SEP:
        direction = "SHORT"
    else:
        return None

    try:
        df = get_data(ctx, symbol, "1h")
    except Exception:
        return None
    if len(df) < 50:
        return None

    last = df.iloc[-1]
    atr = float(last["atr"])
    rsi = float(last["rsi"])
    ema20 = float(last["ema21"])
    if np.isnan(atr) or np.isnan(rsi) or atr <= 0:
        return None

    c = float(last["close"]); o = float(last["open"])
    h = float(last["high"]); l = float(last["low"])
    rng = h - l
    if rng <= 0:
        return None
    body_ratio = abs(c - o) / rng
    volume = float(last["volume"])
    avg_volume = float(df["volume"].iloc[-21:-1].mean()) if len(df) >= 21 else volume

    donch_high = float(df["high"].iloc[-20:].max())
    donch_low = float(df["low"].iloc[-20:].min())

    look = df.iloc[-18:-1]
    highs = look["high"].values
    lows = look["low"].values
    swing_lows, swing_highs = [], []
    for i in range(2, len(look) - 2):
        if lows[i] <= min(lows[i-2:i+3]): swing_lows.append(float(lows[i]))
        if highs[i] >= max(highs[i-2:i+3]): swing_highs.append(float(highs[i]))

    if direction == "LONG":
        levels_list = ([swing_lows[-1]] if swing_lows else []) + [donch_low]
        level = min(levels_list)
        if float(look["high"].max()) < level * B3_MIN_PULLBACK:
            return None
        dist = abs(c - level) / level
        if dist > B3_NEAR_PCT:
            return None
        if not (B3_RSI_LONG[0] <= rsi <= B3_RSI_LONG[1]):
            return None
        if body_ratio < B3_BODY_MIN or c <= o:
            return None
        if c < ema20:
            return None
        risk = max(c - level + atr * 0.1, atr * B3_ATR_MULT)
        if risk / c < MIN_SL_PCT:
            risk = c * MIN_SL_PCT * 1.15
        sl_level = c - risk

        def levels(entry):
            r = entry - sl_level
            if r <= 0: return None
            return sl_level, entry + B3_RR * r

        score = (0.30 * min(sep / 0.02, 1.0) + 0.25 * (1.0 - dist / B3_NEAR_PCT) +
                 0.25 * min(body_ratio / 0.7, 1.0) + 0.20 * min(abs(rsi - 38) / 15.0, 1.0))
    else:
        levels_list = ([swing_highs[-1]] if swing_highs else []) + [donch_high]
        level = max(levels_list)
        if float(look["low"].min()) > level * (2 - B3_MIN_PULLBACK):
            return None
        dist = abs(c - level) / level
        if dist > B3_NEAR_PCT:
            return None
        if not (B3_RSI_SHORT[0] <= rsi <= B3_RSI_SHORT[1]):
            return None
        if body_ratio < B3_BODY_MIN or c >= o:
            return None
        if c > ema20:
            return None
        risk = max(level - c + atr * 0.1, atr * B3_ATR_MULT)
        if risk / c < MIN_SL_PCT:
            risk = c * MIN_SL_PCT * 1.15
        sl_level = c + risk

        def levels(entry):
            r = sl_level - entry
            if r <= 0: return None
            return sl_level, entry - B3_RR * r

        score = (0.30 * min(sep / 0.02, 1.0) + 0.25 * (1.0 - dist / B3_NEAR_PCT) +
                 0.25 * min(body_ratio / 0.7, 1.0) + 0.20 * min(abs(rsi - 62) / 15.0, 1.0))

    if score < B3_MIN_SCORE:
        print(f"  {symbol}: MTF score too low ({score:.3f})")
        return None

    print(f"  {symbol}: MTF {direction} | RSI {rsi:.1f} | body {body_ratio:.2f} | dist {dist*100:.2f}% | score {score:.3f}")

    return {
        "symbol": symbol, "direction": direction, "close": c, "atr": atr, "rsi": rsi,
        "volume": volume, "avg_volume": avg_volume, "signal_key": int(last["timestamp"]),
        "score": score, "setup": f"MTF Pullback 1H (Daily {direction})", "levels": levels,
    }


# =====================================================================
# استراتژی v30: AMD + Session Liquidity
# =====================================================================
def _utc_hour(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour


def _in_active_session():
    h = datetime.now(timezone.utc).hour
    return B30_ACTIVE_START_H <= h < B30_ACTIVE_END_H


def _asia_range_1h(df1):
    highs, lows, ts = [], [], []
    for i in range(len(df1)):
        h = _utc_hour(int(df1["timestamp"].iloc[i]))
        if B30_ASIA_START_H <= h < B30_ASIA_END_H:
            highs.append(float(df1["high"].iloc[i]))
            lows.append(float(df1["low"].iloc[i]))
            ts.append(int(df1["timestamp"].iloc[i]))

    if len(highs) < 3:
        return None

    now_h = datetime.now(timezone.utc).hour
    if B30_ASIA_START_H <= now_h < B30_ASIA_END_H:
        return None

    asia_hi = max(highs[-8:]) if len(highs) >= 8 else max(highs)
    asia_lo = min(lows[-8:]) if len(lows) >= 8 else min(lows)
    asia_end_ts = ts[-1]
    return {"high": asia_hi, "low": asia_lo, "end_ts": asia_end_ts}


def analyze_amd_session(symbol, ctx):
    if not _in_active_session():
        return None

    df4 = get_data(ctx, symbol, "4h")
    l4 = df4.iloc[-1]
    bias_4h = trend_of(df4)
    atr4 = float(l4["atr"])
    if bias_4h not in ("LONG", "SHORT") or np.isnan(atr4) or atr4 <= 0:
        return None
    gap4 = abs(float(l4["ema9"]) - float(l4["ema21"]))
    if gap4 < B30_TREND_GAP_4H_ATR * atr4:
        return None

    df1 = get_data(ctx, symbol, "1h")
    if len(df1) < 30:
        return None
    atr1 = float(df1["atr"].iloc[-1])
    if np.isnan(atr1) or atr1 <= 0:
        return None

    asia = _asia_range_1h(df1)
    if asia is None:
        return None

    rng_size = asia["high"] - asia["low"]
    if rng_size < B30_MIN_RANGE_ATR * atr1 or rng_size > B30_MAX_RANGE_ATR * atr1:
        print(f"  {symbol}: Asia range invalid ({rng_size / atr1:.1f} ATR)")
        return None

    hi1 = df1["high"].values
    lo1 = df1["low"].values
    c1 = df1["close"].values
    ts1 = df1["timestamp"].values
    n1 = len(df1)

    sweep = None
    for i in range(n1 - 1, max(0, n1 - 1 - B30_SWEEP_MAX_AGE_1H) - 1, -1):
        if int(ts1[i]) <= asia["end_ts"]:
            break
        if lo1[i] < asia["low"] and c1[i] > asia["low"]:
            sweep = {"direction": "LONG", "extreme": float(lo1[i]), "level": asia["low"], "ts": int(ts1[i]), "target": asia["high"]}
            break
        if hi1[i] > asia["high"] and c1[i] < asia["high"]:
            sweep = {"direction": "SHORT", "extreme": float(hi1[i]), "level": asia["high"], "ts": int(ts1[i]), "target": asia["low"]}
            break

    if sweep is None:
        return None

    if sweep["direction"] != bias_4h:
        print(f"  {symbol}: AMD sweep {sweep['direction']} مخالف 4H={bias_4h}")
        return None

    direction = sweep["direction"]

    df15 = get_data(ctx, symbol, "15m")
    h, l, o, c, v, ts15 = (df15[k].values for k in ("high", "low", "open", "close", "volume", "timestamp"))
    last = len(df15) - 1
    price = float(c[last])

    fvg = None
    for i in range(last - 1, max(2, last - B30_FVG_RECENT) - 1, -1):
        if int(ts15[i]) < sweep["ts"]:
            break
        if direction == "LONG" and h[i - 2] < l[i]:
            bottom, top = float(h[i - 2]), float(l[i])
            if (c[i + 1:last] < bottom).any():
                continue
            mid = (bottom + top) / 2
            if l[last] <= top and c[last] >= mid * 0.98:
                fvg = {"bottom": bottom, "top": top, "ts": int(ts15[i])}
                break
        if direction == "SHORT" and l[i - 2] > h[i]:
            bottom, top = float(h[i]), float(l[i - 2])
            if (c[i + 1:last] > top).any():
                continue
            mid = (bottom + top) / 2
            if h[last] >= bottom and c[last] <= mid * 1.02:
                fvg = {"bottom": bottom, "top": top, "ts": int(ts15[i])}
                break

    if fvg is None:
        print(f"  {symbol}: AMD {direction} | sweep ok | FVG missing")
        return None

    if direction == "LONG":
        sl = sweep["extreme"] - B30_SL_BUFFER_ATR * atr1
        tp = sweep["target"]
    else:
        sl = sweep["extreme"] + B30_SL_BUFFER_ATR * atr1
        tp = sweep["target"]

    def levels(entry):
        if direction == "LONG":
            if not (sl < entry < tp): return None
            risk, reward = entry - sl, tp - entry
        else:
            if not (tp < entry < sl): return None
            risk, reward = sl - entry, entry - tp
        if risk <= 0 or risk > B30_MAX_RISK_ATR * atr1 or reward / risk < B30_MIN_RR:
            return None
        return sl, tp

    if levels(price) is None:
        print(f"  {symbol}: AMD {direction} rejected (RR/levels)")
        return None

    risk = abs(price - sl)
    if risk / price < MIN_SL_PCT:
        return None
    rr = abs(tp - price) / risk

    depth = abs(sweep["level"] - sweep["extreme"]) / atr1
    vol = float(v[last])
    avg_vol = float(df15["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(vol, avg_vol)
    rsi = float(df15["rsi"].iloc[-1]) if not np.isnan(df15["rsi"].iloc[-1]) else 50.0

    score = (0.35 * min(depth / 0.6, 1.0) + 0.30 * min(rr / 3.0, 1.0) +
             0.20 * min(vol_ratio / 2.0, 1.0) + 0.15 * min(gap4 / atr4, 1.0))

    if score < B30_MIN_SCORE:
        print(f"  {symbol}: AMD score too low ({score:.3f})")
        return None

    print(f"  {symbol}: AMD {direction} | Asia sweep ok | FVG ok | RR {rr:.2f} | score {score:.3f}")
    return {
        "symbol": symbol, "direction": direction, "close": price, "atr": atr1, "rsi": rsi,
        "volume": vol, "avg_volume": avg_vol, "signal_key": fvg["ts"], "score": score,
        "setup": f"AMD: Asia sweep + FVG ({direction})", "levels": levels,
    }


# =====================================================================
# استراتژی ۴: رژیم ADX
# =====================================================================
def analyze_regime(symbol, ctx):
    df = get_data(ctx, symbol, "1h")
    last = df.iloc[-1]

    close, open_, high, low = (float(last[k]) for k in ("close", "open", "high", "low"))
    atr = float(last["atr"])
    rsi = float(last["rsi"])
    adx = float(last["adx"])
    if any(np.isnan(v) for v in (atr, rsi, adx)) or atr <= 0:
        return None

    volume = float(last["volume"])
    avg_volume = float(df["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(volume, avg_volume)

    if adx >= B4_ADX_TREND: mode = "trend"
    elif adx <= B4_ADX_RANGE: mode = "range"
    else:
        print(f"  {symbol}: ADX {adx:.1f} بین دو رژیم، بدون معامله")
        return None

    if mode == "trend":
        df_4h = get_data(ctx, symbol, "4h")
        l4 = df_4h.iloc[-1]
        direction = trend_of(df_4h)
        atr_4h = float(l4["atr"])
        if direction not in ("LONG", "SHORT") or np.isnan(atr_4h) or atr_4h <= 0:
            return None
        gap_4h = abs(float(l4["ema9"]) - float(l4["ema21"]))
        if gap_4h < B4_TREND_GAP_4H_ATR * atr_4h:
            return None

        try:
            btc_4h = trend_of(get_data(ctx, BTC_SYMBOL, "4h"))
            if btc_4h in ("LONG", "SHORT") and direction != btc_4h:
                print(f"  {symbol}: TREND {direction} rejected (BTC 4H={btc_4h})")
                return None
        except Exception: pass

        ema21 = float(last["ema21"])
        if direction == "LONG":
            touched = (low <= ema21 + B4_TREND_TOUCH_ATR * atr and close > ema21 and close > open_ and close - ema21 <= B4_TREND_MAX_EXT_ATR * atr)
            rsi_ok = B4_TREND_RSI_LONG[0] <= rsi <= B4_TREND_RSI_LONG[1]
        else:
            touched = (high >= ema21 - B4_TREND_TOUCH_ATR * atr and close < ema21 and close < open_ and ema21 - close <= B4_TREND_MAX_EXT_ATR * atr)
            rsi_ok = B4_TREND_RSI_SHORT[0] <= rsi <= B4_TREND_RSI_SHORT[1]

        vol_ok = vol_ratio >= B4_TREND_MIN_VOL

        print(f"  {symbol}: ADX {adx:.1f} TREND {direction} | پولبک ok={touched} | RSI {rsi:.1f} ok={rsi_ok} | Vol x{vol_ratio:.2f} ok={vol_ok}")
        if not (touched and rsi_ok and vol_ok):
            return None
        if (atr * B4_TREND_SL_ATR) / close < MIN_SL_PCT:
            return None

        score = (0.45 * min(adx / 40.0, 1.0) + 0.30 * min(gap_4h / atr_4h, 1.0) + 0.25 * min(vol_ratio / 2.0, 1.0))
        if score < B4_MIN_SCORE:
            print(f"  {symbol}: score too low ({score:.3f})")
            return None

        return {
            "symbol": symbol, "direction": direction, "close": close, "atr": atr, "rsi": rsi,
            "volume": volume, "avg_volume": avg_volume, "signal_key": int(last["timestamp"]), "score": score,
            "setup": "رژیم روند: پولبک به میانگین ۱H", "levels": atr_levels(direction, atr, B4_TREND_SL_ATR, B4_TREND_TP_ATR),
        }

    bb_lo, bb_mid, bb_up = (float(last[k]) for k in ("bb_lo", "bb_mid", "bb_up"))
    if any(np.isnan(v) for v in (bb_lo, bb_mid, bb_up)):
        return None

    if low <= bb_lo and close > bb_lo and rsi <= B4_RANGE_RSI_LONG_MAX:
        direction = "LONG"
        sl = low - B4_RANGE_SL_BUFFER_ATR * atr
        tp = bb_mid
    elif high >= bb_up and close < bb_up and rsi >= B4_RANGE_RSI_SHORT_MIN:
        direction = "SHORT"
        sl = high + B4_RANGE_SL_BUFFER_ATR * atr
        tp = bb_mid
    else:
        print(f"  {symbol}: ADX {adx:.1f} RANGE | بدون بازگشت از باند (RSI {rsi:.1f})")
        return None

    if (direction == "LONG" and not (sl < close < tp)) or (direction == "SHORT" and not (tp < close < sl)):
        return None
    risk = abs(close - sl)
    rr = abs(tp - close) / risk
    if rr < B4_RANGE_MIN_RR or risk / close < MIN_SL_PCT:
        print(f"  {symbol}: RANGE {direction} rejected (RR {rr:.2f})")
        return None

    def levels(entry):
        if direction == "LONG": return (sl, tp) if sl < entry < tp else None
        return (sl, tp) if tp < entry < sl else None

    score = (0.50 * min(rr / 3.0, 1.0) + 0.30 * min(abs(rsi - 50) / 25.0, 1.0) + 0.20 * min(vol_ratio / 2.0, 1.0))
    if score < B4_MIN_SCORE:
        print(f"  {symbol}: score too low ({score:.3f})")
        return None

    print(f"  {symbol}: ADX {adx:.1f} RANGE {direction} | RSI {rsi:.1f} | RR {rr:.2f}")
    return {
        "symbol": symbol, "direction": direction, "close": close, "atr": atr, "rsi": rsi,
        "volume": volume, "avg_volume": avg_volume, "signal_key": int(last["timestamp"]), "score": score,
        "setup": "رژیم رنج: بازگشت از باند به میانگین ۱H", "levels": levels,
    }


# =====================================================================
# استراتژی ۵: SMC/ICT
# =====================================================================
# =====================================================================
# استراتژی ۵: SMC/ICT
# =====================================================================
def analyze_smc(symbol, ctx):
    df4 = get_data(ctx, symbol, "4h")
    l4 = df4.iloc[-1]
    direction = trend_of(df4)
    atr4 = float(l4["atr"])
    
    if direction not in ("LONG", "SHORT") or np.isnan(atr4) or atr4 <= 0:
        return None
        
    trend_gap = abs(float(l4["ema9"]) - float(l4["ema21"]))
    if trend_gap < B5_TREND_GAP_4H_ATR * atr4:
        return None

    df1 = get_data(ctx, symbol, "1h")
    n1 = len(df1)
    
    if n1 < B5_RANGE_BARS + B5_LOOKBACK_1H + 5:
        return None
        
    atr1 = float(df1["atr"].iloc[-1])
    if np.isnan(atr1) or atr1 <= 0:
        return None

    hi1 = df1["high"].values
    lo1 = df1["low"].values
    cl1 = df1["close"].values
    ts1 = df1["timestamp"].values

    sweep = None
    for m in range(n1 - 1, n1 - 1 - B5_SWEEP_RECENT, -1):
        win = slice(m - B5_LOOKBACK_1H, m)
        
        if direction == "LONG":
            level = float(lo1[win].min())
            if lo1[m] < level < cl1[m]:  # <--- اینجا اصلاح شد
                sweep = {
                    "ts": int(ts1[m]),
                    "extreme": float(lo1[m]),
                    "level": level
                }
                break
        else:
            level = float(hi1[win].max())
            if cl1[m] < level < hi1[m]:  # <--- اینجا اصلاح شد
                sweep = {
                    "ts": int(ts1[m]),
                    "extreme": float(hi1[m]),
                    "level": level
                }
                break

    if sweep is None:
        return None

    # BOS FILTER
    bos_ok = False
    if direction == "LONG":
        recent_high = float(df1["high"].iloc[-10:-1].max())
        if float(df1["close"].iloc[-1]) > recent_high:
            bos_ok = True
    else:
        recent_low = float(df1["low"].iloc[-10:-1].min())
        if float(df1["close"].iloc[-1]) < recent_low:
            bos_ok = True

    if not bos_ok:
        print(f"  {symbol}: BOS missing")
        return None

    rng = slice(n1 - B5_RANGE_BARS, n1)
    rng_hi = float(hi1[rng].max())
    rng_lo = float(lo1[rng].min())
    eq = (rng_hi + rng_lo) / 2

    df15 = get_data(ctx, symbol, "15m")
    h = df15["high"].values
    l = df15["low"].values
    o = df15["open"].values
    c = df15["close"].values
    v = df15["volume"].values
    ts15 = df15["timestamp"].values
    
    last = len(df15) - 1
    price = float(c[last])
    
    if (direction == "LONG" and price > eq):
        return None
    if (direction == "SHORT" and price < eq):
        return None

    fvg = None
    fvg_size = 0
    
    for i in range(last - 1, max(2, last - B5_FVG_RECENT) - 1, -1):
        if ts15[i - 1] < sweep["ts"]:
            break
            
        if direction == "LONG" and h[i - 2] < l[i]:
            bottom = float(h[i - 2])
            top = float(l[i])
            if (c[i + 1:last] < bottom).any():
                continue
            if (l[last] <= top and c[last] > o[last]):
                fvg_size = abs(top - bottom)
                if fvg_size < (B5_MIN_FVG_ATR * atr1):
                    continue
                fvg = {"ts": int(ts15[i])}
                break
                
        if direction == "SHORT" and l[i - 2] > h[i]:
            bottom = float(h[i])
            top = float(l[i - 2])
            if (c[i + 1:last] > top).any():
                continue
            if (h[last] >= bottom and c[last] < o[last]):
                fvg_size = abs(top - bottom)
                if fvg_size < (B5_MIN_FVG_ATR * atr1):
                    continue
                fvg = {"ts": int(ts15[i])}
                break

    if fvg is None:
        print(f"  {symbol}: no quality FVG")
        return None

    vol = float(v[last])
    avg_vol = float(df15["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(vol, avg_vol)
    
    if vol_ratio < B5_MIN_VOLUME_RATIO:
        print(f"  {symbol}: weak volume")
        return None

    if direction == "LONG":
        sl = sweep["extreme"] - B5_SL_BUFFER_ATR * atr1
        tp = rng_hi
    else:
        sl = sweep["extreme"] + B5_SL_BUFFER_ATR * atr1
        tp = rng_lo

    def levels(entry):
        if direction == "LONG":
            if not (sl < entry < tp): return None
            risk = entry - sl
            reward = tp - entry
        else:
            if not (tp < entry < sl): return None
            risk = sl - entry
            reward = entry - tp
            
        if risk <= 0: return None
        if risk > (B5_MAX_RISK_ATR * atr1): return None
        if reward / risk < B5_MIN_RR: return None
        return sl, tp

    if levels(price) is None:
        return None
        
    risk = abs(price - sl)
    if risk / price < MIN_SL_PCT:
        return None
        
    rr = abs(tp - price) / risk
    depth = abs(sweep["level"] - sweep["extreme"]) / atr1
    
    score = (
        0.30 * min(depth / 0.60, 1.0) +
        0.25 * min(vol_ratio / 2.0, 1.0) +
        0.25 * min(rr / 3.0, 1.0) +
        0.20 * min(fvg_size / atr1, 1.0)
    )
    
    if score < B5_MIN_SCORE:
        print(f"  {symbol}: score too low ({score:.3f})")
        return None
        
    rsi = float(df15["rsi"].iloc[-1])
    if np.isnan(rsi):
        rsi = 50.0
        
    print(f"  {symbol}: SMC {direction} | BOS | FVG | RR={rr:.2f} | score={score:.3f}")
    
    return {
        "symbol": symbol,
        "direction": direction,
        "close": price,
        "atr": atr1,
        "rsi": rsi,
        "volume": vol,
        "avg_volume": avg_vol,
        "signal_key": fvg["ts"],
        "score": score,
        "setup": "SMC PRO (Sweep+BOS+FVG)",
        "levels": levels,
    }
    
    
        
        
        
            
        
  


# =====================================================================
# استراتژی ۶: فیلتر BTC + پولبک هم‌جهت
# =====================================================================
def get_btc_filter(ctx):
    try:
        df_1d = get_data(ctx, BTC_SYMBOL, "1d")
        df_4h = get_data(ctx, BTC_SYMBOL, "4h")
        df_1h = get_data(ctx, BTC_SYMBOL, "1h")
    except Exception as e:
        print(f"⚠️ BTC filter data error: {e}")
        return None

    t_1d, t_4h, t_1h = trend_of(df_1d), trend_of(df_4h), trend_of(df_1h)

    if t_1d not in ("LONG", "SHORT") or t_4h not in ("LONG", "SHORT"):
        print(f"🌍 BTC filter: no clear bias (1D={t_1d}, 4H={t_4h})")
        return None
    if t_1d != t_4h:
        print(f"🌍 BTC filter: 1D/4H conflict (1D={t_1d}, 4H={t_4h})")
        return None

    bias = t_1d
    last = df_4h.iloc[-1]
    atr, close = float(last["atr"]), float(last["close"])
    ema9, ema21 = float(last["ema9"]), float(last["ema21"])
    adx = float(last["adx"]) if not np.isnan(last["adx"]) else 0.0

    if np.isnan(atr) or atr <= 0: return None

    gap = abs(ema9 - ema21)
    gap_atr = gap / atr
    strength_ok = (gap_atr >= B6_MIN_GAP_ATR) or (adx >= B6_MIN_ADX)
    if not strength_ok:
        print(f"🌍 BTC filter: weak trend (gap={gap_atr:.2f} ATR, ADX={adx:.1f})")
        return None

    ext_atr = abs(close - ema21) / atr
    if ext_atr > B6_MAX_EXT_ATR:
        print(f"🌍 BTC filter: overextended ({ext_atr:.2f} ATR from EMA21)")
        return None

    align_1h = t_1h == bias
    print(f"🌍 BTC filter OK | bias={bias} | 1D={t_1d} 4H={t_4h} 1H={t_1h} | gap={gap_atr:.2f} ADX={adx:.1f} ext={ext_atr:.2f} | 1H_align={align_1h}")

    return {"bias": bias, "t_1d": t_1d, "t_4h": t_4h, "t_1h": t_1h, "align_1h": align_1h,
            "gap_atr": gap_atr, "adx": adx, "ext_atr": ext_atr}


def analyze_btc_aligned(symbol, ctx):
    btc = getattr(ctx, "btc_filter", None)
    if not btc: return None
    bias = btc["bias"]

    df_4h = get_data(ctx, symbol, "4h")
    if trend_of(df_4h) != bias: return None

    l4 = df_4h.iloc[-1]
    atr_4h = float(l4["atr"])
    if np.isnan(atr_4h) or atr_4h <= 0: return None
    gap_4h = abs(float(l4["ema9"]) - float(l4["ema21"]))
    if gap_4h < B6_ALT_GAP_ATR * atr_4h: return None

    df_1h = get_data(ctx, symbol, "1h")
    last = df_1h.iloc[-1]
    close, open_, high, low = (float(last[k]) for k in ("close", "open", "high", "low"))
    atr, rsi, ema21 = float(last["atr"]), float(last["rsi"]), float(last["ema21"])

    if np.isnan(atr) or np.isnan(rsi) or atr <= 0: return None

    if bias == "LONG":
        touched = (low <= ema21 + B6_TOUCH_ATR * atr and close > ema21 and close > open_ and close - ema21 <= B6_MAX_EXT_ALT_ATR * atr)
        rsi_ok = B6_RSI_LONG[0] <= rsi <= B6_RSI_LONG[1]
    else:
        touched = (high >= ema21 - B6_TOUCH_ATR * atr and close < ema21 and close < open_ and ema21 - close <= B6_MAX_EXT_ALT_ATR * atr)
        rsi_ok = B6_RSI_SHORT[0] <= rsi <= B6_RSI_SHORT[1]

    volume = float(last["volume"])
    avg_volume = float(df_1h["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(volume, avg_volume)
    vol_ok = vol_ratio >= B6_VOLUME_MULT

    print(f"  {symbol}: BTC={bias} | pullback ok={touched} | RSI {rsi:.1f} ok={rsi_ok} | Vol x{vol_ratio:.2f} ok={vol_ok}")

    if not (touched and rsi_ok and vol_ok): return None
    if (atr * B6_SL_ATR) / close < MIN_SL_PCT: return None

    trend_score = min(gap_4h / atr_4h, 1.2) / 1.2
    vol_score = min(vol_ratio / 2.5, 1.0)
    btc_score = 0.55 + (B6_1H_ALIGN_BONUS if btc["align_1h"] else 0.0)
    btc_score += 0.15 * min(btc["gap_atr"] / 0.6, 1.0)
    score = (0.35 * btc_score + 0.30 * trend_score + 0.20 * vol_score + 0.15 * min(abs(rsi - 50) / 20.0, 1.0))

    if score < B6_MIN_SCORE:
        print(f"  {symbol}: score too low ({score:.3f})")
        return None

    return {
        "symbol": symbol, "direction": bias, "close": close, "atr": atr, "rsi": rsi,
        "volume": volume, "avg_volume": avg_volume, "signal_key": int(last["timestamp"]), "score": score,
        "setup": f"BTC-aligned pullback 1H (BTC {bias})", "levels": atr_levels(bias, atr, B6_SL_ATR, B6_TP_ATR),
    }


# =====================================================================
# استراتژی ۸: SMC Winner (نسخه بک‌تست شده با سود ۱۴.۷۷ دلار)
# =====================================================================
def analyze_smc_winner(symbol, ctx):
    try:
        df4 = get_data(ctx, symbol, "4h")
    except Exception:
        return None
    if len(df4) < 20:
        return None
    
    curr = df4.iloc[-1]
    atr = float(curr["atr"])
    if np.isnan(atr) or atr <= 0:
        return None

    # پیدا کردن سطوح نقدینگی (سوییپ) در ۱۰ کندل اخیر
    prev_low = float(df4["low"].iloc[-B8_SWEEP_LOOKBACK-1:-1].min())
    prev_high = float(df4["high"].iloc[-B8_SWEEP_LOOKBACK-1:-1].max())
    
    close = float(curr["close"])
    low = float(curr["low"])
    high = float(curr["high"])
    rsi = float(curr["rsi"]) if not np.isnan(curr["rsi"]) else 50.0
    
    direction = None
    sl_dist = 0
    tp_dist = 0

    # سوییپ کف و بازگشت -> خرید
    if low < prev_low and close > prev_low:
        direction = "LONG"
        sl_dist = close - (low - atr * 0.2)
        tp_dist = atr * B8_TP_ATR_MULT
        
    # سوییپ سقف و بازگشت -> فروش
    elif high > prev_high and close < prev_high:
        direction = "SHORT"
        sl_dist = (high + atr * 0.2) - close
        tp_dist = atr * B8_TP_ATR_MULT
    else:
        return None

    if sl_dist <= 0 or close <= 0:
        return None
        
    if sl_dist / close < MIN_SL_PCT:
        return None

    # محاسبه امتیاز ساده برای این ربات
    score = 0.8 if (direction == "LONG" and rsi < 50) or (direction == "SHORT" and rsi > 50) else 0.6

    def levels(entry):
        if direction == "LONG":
            return entry - sl_dist, entry + tp_dist
        else:
            return entry + sl_dist, entry - tp_dist

    return {
        "symbol": symbol, "direction": direction, "close": close,
        "atr": atr, "rsi": rsi,
        "volume": float(curr["volume"]), "avg_volume": float(df4["volume"].iloc[-21:-1].mean()),
        "signal_key": int(curr["timestamp"]), "score": score,
        "setup": f"SMC Winner ({direction} Sweep)",
        "levels": levels,
    }


# =====================================================================
# تعریف ربات‌ها
# =====================================================================
class Bot:
    def __init__(self, name, label, analyze, heartbeat_file, signals_file,
                 cooldown_hours=0, max_signals=MAX_SIGNALS_PER_RUN,
                 btc_filter_4h=True, btc_filter_1h=False, live=True):
        self.name = name
        self.label = label
        self.analyze = analyze
        self.heartbeat_file = heartbeat_file
        self.signals_file = signals_file
        self.cooldown_ms = int(cooldown_hours * 3600 * 1000)
        self.max_signals = max_signals
        self.btc_filter_4h = btc_filter_4h
        self.btc_filter_1h = btc_filter_1h
        self.live = live

    @property
    def title_tag(self):
        return f"[{self.name}]" + (f" ({self.label})" if self.label else "")


BOTS = [
    Bot("crypto-signal-bot", "", analyze_cross,
        "server_messages.json", "last_signals.json"),
    Bot("crypto-signal-bot-v10", "Breakout", analyze_breakout,
        "server_messages_v10.json", "last_signals_v10.json"),
    Bot("crypto-signal-bot-v20", "MTF Pullback", analyze_mtf_pullback,
        "server_messages_v20.json", "last_signals_v20.json",
        cooldown_hours=B3_COOLDOWN_HOURS),
    Bot("crypto-signal-bot-v30", "AMD Session", analyze_amd_session,
        "server_messages_v30.json", "last_signals_v30.json",
        cooldown_hours=B30_COOLDOWN_HOURS, btc_filter_4h=True),
    Bot("crypto-signal-bot-v40", "Regime", analyze_regime,
        "server_messages_v40.json", "last_signals_v40.json",
        cooldown_hours=B4_COOLDOWN_HOURS, btc_filter_4h=False),
    Bot("crypto-signal-bot-v50", "SMC", analyze_smc,
        "server_messages_v50.json", "last_signals_v50.json",
        cooldown_hours=B5_COOLDOWN_HOURS, live=True),
    Bot("crypto-signal-bot-v60", "BTC Filter", analyze_btc_aligned,
        "server_messages_v60.json", "last_signals_v60.json",
        cooldown_hours=B6_COOLDOWN_HOURS, btc_filter_4h=False),
    
    # --- ربات جدید ۸ (SMC Winner) ---
    Bot("crypto-signal-bot-v70", "SMC Winner", analyze_smc_winner,
        "server_messages_v70.json", "last_signals_v70.json",
        cooldown_hours=B8_COOLDOWN_HOURS, btc_filter_4h=False),
]
LIVE_BOTS = [b for b in BOTS if b.live]


# =====================================================================
# Heartbeat
# =====================================================================
HEARTBEAT_FILE = "server_messages.json"

def send_heartbeat_all(results, btc_4h, btc_1h, warning=None):
    date_gregorian, date_jalali, time_str = get_iran_time()

    for bot in LIVE_BOTS[1:]:
        old = load_json(bot.heartbeat_file, {}).get("message_ids", [])
        if old:
            for mid in old: delete_telegram(mid)
            save_json(bot.heartbeat_file, {"message_ids": []})

    lines, ok = [], 0
    for bot in LIVE_BOTS:
        r = results.get(bot.name, "ERROR")
        if r == "ERROR": lines.append(f"❌ {bot.name}")
        else:
            ok += 1
            lines.append(f"✅ {bot.name}" + (f"  ({r} سیگنال جدید)" if r else ""))

    extra = f"\n⚠️ {warning}" if warning else ""
    msg = f"""🖥️ سیستم آنلاین | {ok}/{len(LIVE_BOTS)} ربات سالم{extra}
━━━━━━━━━━━━━━━━━
{chr(10).join(lines)}
━━━━━━━━━━━━━━━━━
🌍 BTC: 4H={btc_4h} | 1H={btc_1h}
📅 تاریخ شمسی: {date_jalali}
📅 تاریخ میلادی: {date_gregorian}
⏰ ساعت: {time_str}
━━━━━━━━━━━━━━━━━"""

    ids = load_json(HEARTBEAT_FILE, {}).get("message_ids", [])
    new_id = send_telegram(msg)
    if new_id:
        ids.append(new_id)
        while len(ids) > MAX_HEARTBEATS: delete_telegram(ids.pop(0))
        save_json(HEARTBEAT_FILE, {"message_ids": ids})


# =====================================================================
# مدیریت معاملات
# =====================================================================
def check_trade_result(exchange, symbol, data):
    direction, tp, sl = data["direction"], data["tp"], data["sl"]
    opened_ms = data.get("opened_ms")

    if opened_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe="5m", since=int(opened_ms), limit=1000)
        except Exception as e:
            print(f"⚠️ خطا در دریافت کندل‌های {symbol}: {e}")
            candles = []
        for cd in candles:
            if cd[0] < opened_ms: continue
            high, low = cd[2], cd[3]
            if direction == "LONG": hit_sl, hit_tp = low <= sl, high >= tp
            else: hit_sl, hit_tp = high >= sl, low <= tp
            if hit_sl: return "SL", sl
            if hit_tp: return "TP", tp

    price = float(exchange.fetch_ticker(symbol)["last"])
    if direction == "LONG":
        if price <= sl: return "SL", price
        if price >= tp: return "TP", price
    else:
        if price >= sl: return "SL", price
        if price <= tp: return "TP", price
    return None, None


def trade_r(rec):
    if rec.get("status") != "CLOSED": return None
    risk = abs(rec["entry"] - rec["sl"])
    if risk <= 0: return None
    return abs(rec["tp"] - rec["entry"]) / risk if rec.get("result") == "TP" else -1.0


def build_close_message(bot, symbol, direction, result, close_price, entry):
    date_gregorian, date_jalali, time_str = get_iran_time()
    emoji = "🎯" if result == "TP" else "🛑"
    title = "حد سود (TP) لمس شد" if result == "TP" else "حد ضرر (SL) لمس شد"
    return f"""{emoji} {bot.title_tag} بسته شدن معامله
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


CLOSE_MSG_MAX_ATTEMPTS = 5

def update_open_trades(bot, exchange):
    signals = load_json(bot.signals_file, {})
    updated = False

    for symbol, data in list(signals.items()):
        if data.get("status") == "OPEN":
            try:
                result, close_price = check_trade_result(exchange, symbol, data)
            except Exception as e:
                print(f"⚠️ خطا در بررسی {symbol}: {e}")
                continue
            if result:
                data["status"] = "CLOSED"
                data["result"] = result
                data["closed_ms"] = now_ms()
                data["close_price"] = float(close_price)
                data["close_telegram_sent"] = False
                data["close_msg_attempts"] = 0
                data["symbol"] = symbol
                updated = True
                append_trade_csv(bot.name, data)
                excel_log.log_result(bot.name, symbol, result, data["closed_ms"])
                print(f"{'🎯' if result == 'TP' else '🛑'} {symbol} {data['direction']} hit {result} at {close_price}")

        if (data.get("status") == "CLOSED" and data.get("close_telegram_sent") is False
                and data.get("close_msg_attempts", 0) < CLOSE_MSG_MAX_ATTEMPTS):
            msg_id = send_telegram(build_close_message(
                bot, symbol, data["direction"], data["result"], data["close_price"], data["entry"]))
            data["close_msg_attempts"] = data.get("close_msg_attempts", 0) + 1
            data["close_telegram_sent"] = bool(msg_id)
            updated = True
            if not msg_id:
                print(f"⚠️ {symbol}: پیام بسته شدن ارسال نشد (تلاش {data['close_msg_attempts']})")

    if updated:
        save_json(bot.signals_file, signals)


def build_signal_message(bot, c, entry, sl, tp, rr):
    date_gregorian, date_jalali, time_str = get_iran_time()
    emoji = "🟢" if c["direction"] == "LONG" else "🔴"
    return f"""{emoji} {bot.title_tag} سیگنال {c['direction']}
━━━━━━━━━━━━━━━━━
📌 نماد: {c['symbol']}
🔎 ستاپ: {c['setup']}
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


# =====================================================================
# چرخه‌ی یک ربات
# =====================================================================
def run_bot(bot, ctx, btc_4h, btc_1h):
    print("=" * 50)
    print(f"▶ {bot.name} {('(' + bot.label + ')') if bot.label else ''}")

    update_open_trades(bot, ctx.exchange)
    signals = load_json(bot.signals_file, {})

    candidates = []
    t_now = now_ms()
    for symbol in SYMBOLS:
        try:
            prev = signals.get(symbol)
            if prev and prev.get("status") == "OPEN":
                print(f"⏳ {symbol} has an OPEN trade. Skipping.")
                continue
            if (prev and bot.cooldown_ms and prev.get("closed_ms")
                    and t_now - prev["closed_ms"] < bot.cooldown_ms):
                print(f"❄️ {symbol} in cooldown. Skipping.")
                continue

            c = bot.analyze(symbol, ctx)
            if c is None:
                continue

            prev_key = (prev.get("signal_key", prev.get("cross_ts")) if prev else None)
            if prev and prev_key == c["signal_key"] and prev.get("direction") == c["direction"]:
                print(f"♻️ {symbol} {c['direction']}: this setup was already signaled. Skipping.")
                continue

            if symbol != BTC_SYMBOL:
                if bot.btc_filter_4h and btc_4h in ("LONG", "SHORT") and c["direction"] != btc_4h:
                    print(f"🚫 {symbol} {c['direction']} rejected. BTC 4H trend is {btc_4h}.")
                    continue
                if bot.btc_filter_1h and btc_1h in ("LONG", "SHORT") and c["direction"] != btc_1h:
                    print(f"🚫 {symbol} {c['direction']} rejected. BTC 1H trend is {btc_1h}.")
                    continue

            candidates.append(c)
            print(f"✅ Candidate: {symbol} {c['direction']} score={c['score']:.3f}")
        except Exception as e:
            print(f"⚠️ خطا در {symbol}: {e}")
            continue

    candidates.sort(key=lambda x: x["score"], reverse=True)
    print(f"📋 {len(candidates)} candidate(s): " + ", ".join(f"{c['symbol']}({c['score']:.2f})" for c in candidates))

    sent = 0
    for c in candidates:
        if sent >= bot.max_signals: break
        symbol = c["symbol"]
        try:
            entry = get_price(ctx, symbol)
            if abs(entry - c["close"]) > c["atr"] * MAX_ENTRY_DRIFT_ATR:
                print(f"⌛ {symbol}: price drifted too far from signal candle. Skipping.")
                continue

            lv = c["levels"](entry)
            if lv is None:
                print(f"⚠️ {symbol}: price already beyond stop level. Skipping.")
                continue
            sl, tp = lv
            if sl <= 0 or tp <= 0:
                print(f"⚠️ {symbol}: invalid SL/TP. Skipping.")
                continue
            rr = abs(tp - entry) / abs(entry - sl)

            msg_id = send_telegram(build_signal_message(bot, c, entry, sl, tp, rr))

            record = {
                "direction": c["direction"], "entry": entry, "tp": tp, "sl": sl,
                "signal_key": c["signal_key"], "setup": c["setup"],
                "score": round(c["score"], 4),
                "time": datetime.now(timezone.utc).isoformat(),
                "telegram_sent": bool(msg_id), "telegram_message_id": msg_id,
            }
            if msg_id:
                record["status"] = "OPEN"
                record["opened_ms"] = now_ms()
                sent += 1
                excel_log.log_signal(bot.name, symbol, c["direction"], entry, tp, sl)
                print(f"✅ {c['direction']} signal for {symbol} sent! ({sent}/{bot.max_signals})")
            else:
                record["status"] = "NOT_SENT"
                print(f"⚠️ {symbol}: پیام تلگرام ارسال نشد؛ معامله ثبت نشد (NOT_SENT).")
            signals[symbol] = record
            save_json(bot.signals_file, signals)
        except Exception as e:
            print(f"⚠️ خطا در ارسال {symbol}: {e}")
            continue

    print(f"{bot.name}: analysis complete ({sent} signal(s) sent).")
    return sent


# =====================================================================
# خلاصه‌ی روزانه
# =====================================================================
def maybe_send_daily_summary():
    now = datetime.now(ZoneInfo("Asia/Tehran"))
    today = now.strftime("%Y-%m-%d")
    state = load_json(SUMMARY_FILE, {})
    if now.hour < SUMMARY_HOUR or state.get("date") == today:
        return

    date_gregorian, date_jalali, time_str = get_iran_time()
    lines = []
    for bot in LIVE_BOTS:
        recs = list(load_json(bot.signals_file, {}).values())
        closed = [r for r in recs if r.get("status") == "CLOSED"]
        opened = [r for r in recs if r.get("status") == "OPEN"]
        rs = [x for x in (trade_r(r) for r in closed) if x is not None]
        if rs:
            wins = sum(1 for r in closed if r.get("result") == "TP")
            lines.append(f"🤖 {bot.name}\n"
                         f"   بسته‌شده: {len(rs)} | برد: {wins} ({wins / len(rs):.0%}) | "
                         f"R کل: {sum(rs):+.1f} | میانگین R: {sum(rs) / len(rs):+.2f}\n"
                         f"   باز: {len(opened)}")
        else:
            lines.append(f"🤖 {bot.name}\n   هنوز معامله‌ی بسته‌شده‌ای نیست | باز: {len(opened)}")

    msg = ("📊 خلاصه‌ی روزانه‌ی ربات‌ها\n━━━━━━━━━━━━━━━━━\n" + "\n".join(lines) +
           f"\n━━━━━━━━━━━━━━━━━\n📅 تاریخ شمسی: {date_jalali}\n📅 تاریخ میلادی: {date_gregorian}\n"
           f"⏰ ساعت: {time_str}\n━━━━━━━━━━━━━━━━━")
    if send_telegram(msg):
        save_json(SUMMARY_FILE, {"date": today})


# =====================================================================
# اجرای اصلی
# =====================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Starting multi-bot cycle (Quality Mode): " + ", ".join(b.name for b in LIVE_BOTS))

    exchange = create_exchange()
    ctx = Ctx(exchange)

    btc_4h, btc_1h = get_btc_trend(ctx)
    print(f"🌍 BTC 4H Trend: {btc_4h} | BTC 1H Trend: {btc_1h}")
    warning = "دریافت داده‌ی BTC ناموفق بود" if btc_4h == "UNKNOWN" else None

    ctx.btc_filter = get_btc_filter(ctx)
    if not ctx.btc_filter:
        print("🌍 BTC filter blocked — v60 will skip all symbols this cycle.")

    results = {}
    for bot in LIVE_BOTS:
        try:
            results[bot.name] = run_bot(bot, ctx, btc_4h, btc_1h)
        except Exception as e:
            results[bot.name] = "ERROR"
            print(f"⚠️ خطای کلی در {bot.name}: {e}")

    try:
        send_heartbeat_all(results, btc_4h, btc_1h, warning)
        print("✅ Heartbeat sent.")
    except Exception as e:
        print(f"⚠️ خطا در heartbeat: {e}")

    try:
        maybe_send_daily_summary()
    except Exception as e:
        print(f"⚠️ خطا در خلاصه‌ی روزانه: {e}")

    print("All bots complete.")
    print("=" * 50)