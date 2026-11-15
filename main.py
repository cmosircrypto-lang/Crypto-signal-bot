"""
ربات‌های سیگنال با تمرکز روی کیفیت بالا (Win Rate) — نسخه اصلاح‌شده v2
  اصلاحات این نسخه:
    1) B30_ACTIVE_END_H = 22 (پنجره فعال v30 بازتر)
    2) get_btc_filter با لاگ کامل و دلیل دقیق block
    3) __main__ با لاگ BTC 1D
  اصلاحات نسخه قبل (حفظ شده):
    - check_trade_result درست در لمس همزمان SL/TP
    - B30_ACTIVE_START_H = 8 (رفع تناقض)
    - _asia_range_1h بدون وابستگی به now()
    - Ctx.ref_time_ms برای backtest
    - get_price در backtest از close کندل
    - analyze_amd_session از ctx.now_utc_hour()
    - پارامترهای v70 شل‌شده + meta
    - run_bot: opened_ms در backtest از timestamp کندل
"""
import os
import json
import time
import requests
import numpy as np
import pandas as pd
import ccxt
from collections import Counter, defaultdict
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
    "BTC-SWAP-USDT", "ETH-SWAP-USDT", "SOL-SWAP-USDT", "XRP-SWAP-USDT",
    "BNB-SWAP-USDT", "DOGE-SWAP-USDT", "ADA-SWAP-USDT", "TRX-SWAP-USDT",
    "AVAX-SWAP-USDT", "LINK-SWAP-USDT", "DOT-SWAP-USDT", "LTC-SWAP-USDT",
    "BCH-SWAP-USDT", "TON-SWAP-USDT", "SUI-SWAP-USDT", "NEAR-SWAP-USDT",
    "APT-SWAP-USDT", "ARB-SWAP-USDT", "OP-SWAP-USDT", "ATOM-SWAP-USDT",
    "UNI-SWAP-USDT", "AAVE-SWAP-USDT", "FIL-SWAP-USDT", "INJ-SWAP-USDT",
    "TIA-SWAP-USDT", "JUP-SWAP-USDT", "ENA-SWAP-USDT", "WIF-SWAP-USDT",
    "1000PEPE-SWAP-USDT", "1000SHIB-SWAP-USDT", "WLD-SWAP-USDT", "SEI-SWAP-USDT",
    "RUNE-SWAP-USDT", "LDO-SWAP-USDT", "STX-SWAP-USDT", "IMX-SWAP-USDT",
    "HBAR-SWAP-USDT", "ICP-SWAP-USDT", "ETC-SWAP-USDT", "XLM-SWAP-USDT",
    "CRV-SWAP-USDT", "SNX-SWAP-USDT",
    "COMP-SWAP-USDT", "SUSHI-SWAP-USDT", "1INCH-SWAP-USDT", "GRT-SWAP-USDT",
    "DYDX-SWAP-USDT", "GMX-SWAP-USDT", "PENDLE-SWAP-USDT", "ONDO-SWAP-USDT",
    "JTO-SWAP-USDT", "PYTH-SWAP-USDT", "STRK-SWAP-USDT", "MANTA-SWAP-USDT",
    "1000BONK-SWAP-USDT", "1000FLOKI-SWAP-USDT", "MEME-SWAP-USDT", "ORDI-SWAP-USDT",
    "1000SATS-SWAP-USDT", "1000RATS-SWAP-USDT", "POPCAT-SWAP-USDT",
    "BRETT-SWAP-USDT", "TURBO-SWAP-USDT", "1000000MOG-SWAP-USDT", "DOGS-SWAP-USDT",
    "FET-SWAP-USDT", "TAO-SWAP-USDT",
    "AKT-SWAP-USDT", "ILV-SWAP-USDT",
    "AXS-SWAP-USDT", "SAND-SWAP-USDT", "MANA-SWAP-USDT", "GALA-SWAP-USDT",
    "ALGO-SWAP-USDT", "VET-SWAP-USDT", "EGLD-SWAP-USDT", "FLOW-SWAP-USDT",
    "KAVA-SWAP-USDT", "ROSE-SWAP-USDT", "KAS-SWAP-USDT",
    "THETA-SWAP-USDT", "CHZ-SWAP-USDT", "ENJ-SWAP-USDT", "ZIL-SWAP-USDT",
    "ONE-SWAP-USDT", "QTUM-SWAP-USDT", "IOTA-SWAP-USDT", "NEO-SWAP-USDT"
]

BTC_SYMBOL = "BTC-SWAP-USDT"
MAX_HEARTBEATS = 2
MAX_SIGNALS_PER_RUN = 5
MAX_OPEN_SAME_DIR = 3
EXTRA_LOG_FIELDS = ["fvg_size_atr", "fvg_age_15m", "gap_pos", "last_candle_with_trade",
                    "sweep_depth_atr", "sweep_age_h", "asia_range_atr", "gap4_atr", "risk_atr"]
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

# --- ربات ۱: EMA Cross (تغییر) ---
B1_CROSS_LOOKBACK = 6
B1_RSI_LONG = (45, 68)
B1_RSI_SHORT = (32, 55)
B1_VOLUME_MULT = 1.5
B1_SL_ATR = 1.6
B1_TP_ATR = 3.4
B1_MIN_SCORE = 0.70
B1_MIN_ADX_4H = 18
B1_MIN_ADX_1H = 16
B1_MIN_SLOPE_ATR = 0.08

# --- ربات ۲: Breakout (بدون تغییر در پارامترها، فقط غیرفعال) ---
# (در بخش Bot ها، live=False می‌شود)
B2_LOOKBACK = 18
B2_VOLUME_MULT = 1.8
B2_CLOSE_POS = 0.65
B2_MAX_EXTENSION_ATR = 1.2
B2_RSI_LONG = (52, 72)
B2_RSI_SHORT = (28, 48)
B2_SL_ATR, B2_TP_ATR = 1.4, 3.0
B2_MAX_RANGE_ATR = 6.5
B2_MIN_BODY_ATR = 0.45
B2_MIN_SCORE = 0.62
B2_MIN_ADX = 18

# --- ربات ۳: MTF Pullback (بدون تغییر) ---
B3_RSI_LONG = (31, 45)
B3_RSI_SHORT = (55, 69)
B3_NEAR_PCT = 0.03
B3_BODY_MIN = 0.42
B3_ATR_MULT = 0.85
B3_RR = 2.1
B3_MIN_SEP = 0.005
B3_MIN_PULLBACK = 1.005
B3_COOLDOWN_HOURS = 3
B3_MIN_SCORE = 0.55

# --- ربات v30: AMD (تغییر) ---
B30_ASIA_START_H = 0
B30_ASIA_END_H = 8
B30_ACTIVE_START_H = 12
B30_ACTIVE_END_H = 21
B30_SWEEP_MAX_AGE_1H = 4
B30_FVG_RECENT = 8
B30_SL_BUFFER_ATR = 0.15
B30_MIN_RR = 2.1
B30_MAX_RISK_ATR = 2.0
B30_MIN_RANGE_ATR = 1.0
B30_MAX_RANGE_ATR = 5.0
B30_MIN_SCORE = 0.70       # ← افزایش از 0.60
B30_COOLDOWN_HOURS = 4
B30_TREND_GAP_4H_ATR = 0.30

# --- ربات ۴: Regime (تغییر) ---
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
B4_MIN_SCORE = 0.65        # ← افزایش از 0.55
B4_COOLDOWN_HOURS = 3

# --- ربات ۵: SMC (تغییر) ---
B5_LOOKBACK_1H = 24
B5_SWEEP_RECENT = 24        # ← افزایش از 4
B5_RANGE_BARS = 40
B5_FVG_RECENT = 8
B5_TREND_GAP_4H_ATR = 0.15
B5_SL_BUFFER_ATR = 0.12
B5_MIN_RR = 2.1
B5_MAX_RISK_ATR = 2.2
B5_MIN_SCORE = 0.60
B5_COOLDOWN_HOURS = 4
B5_MIN_FVG_ATR = 0.25
B5_MIN_VOLUME_RATIO = 0.90

# --- ربات ۶: BTC Filter (تغییر) ---
B6_MIN_GAP_ATR = 0.28
B6_MIN_ADX = 20
B6_MAX_EXT_ATR = 6.0       # ← افزایش از 4.5
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

# --- ربات ۸: SMC Winner (بدون تغییر) ---
B8_SWEEP_LOOKBACK = 10
B8_TP_ATR_MULT = 2.0
B8_COOLDOWN_HOURS = 4
B8_SL_BUFFER_ATR = 0.2
B8_MIN_SL_ATR = 0.6
B8_MAX_SL_ATR = 3.0
B8_MAX_SL_PCT = 0.07
B8_MIN_RR = 1.2
B8_MIN_VOL_RATIO = 0.9
B8_RSI_MAX_LONG = 72
B8_RSI_MIN_SHORT = 28
B8_MIN_SCORE = 0.35
B8_MAX_SIGNALS_PER_RUN = 2


# =====================================================================
# Funnel
# =====================================================================
FUNNEL = defaultdict(Counter)
_CUR = {"bot": None}


def rej(label="unlabeled"):
    FUNNEL[_CUR["bot"]][label] += 1
    return None


def funnel_count(label):
    FUNNEL[_CUR["bot"]][label] += 1


def print_funnel(bot_name, top=12):
    c = FUNNEL.pop(bot_name, None)
    if not c:
        return
    print(f"📉 {bot_name} funnel:")
    for k, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
        print(f"   {k}: {n}")


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
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return None
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
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
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
              "result", "pnl_r", "setup", "score",
              "rsi", "vol_ratio", "atr_pct", "risk_pct", "rr", "drift_atr",
              "btc_4h", "btc_1h", "open_same_dir"] + EXTRA_LOG_FIELDS + [
              "hold_hours", "opened_ms", "closed_ms"]
    if file_exists:
        try:
            with open(fname, "r", encoding="utf-8") as f:
                old_header = f.readline().strip().split(",")
            if old_header != fields:
                os.replace(fname, fname + ".old")
                file_exists = False
        except Exception as e:
            print(f"⚠️ CSV header check error: {e}")
    hold_hours = ""
    try:
        if record.get("opened_ms") and record.get("closed_ms"):
            hold_hours = round((record["closed_ms"] - record["opened_ms"]) / 3600000, 2)
    except Exception:
        pass
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
        "rsi": record.get("rsi", ""),
        "vol_ratio": record.get("vol_ratio", ""),
        "atr_pct": record.get("atr_pct", ""),
        "risk_pct": record.get("risk_pct", ""),
        "rr": record.get("rr", ""),
        "drift_atr": record.get("drift_atr", ""),
        "btc_4h": record.get("btc_4h", ""),
        "btc_1h": record.get("btc_1h", ""),
        "open_same_dir": record.get("open_same_dir", ""),
        "hold_hours": hold_hours,
        "opened_ms": record.get("opened_ms", ""),
        "closed_ms": record.get("closed_ms", ""),
    }
    for k in EXTRA_LOG_FIELDS:
        row[k] = record.get(k, "")
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
# Ctx (backtest-friendly)
# =====================================================================
class Ctx:
    def __init__(self, exchange, ref_time_ms=None):
        self.exchange = exchange
        self.cache = {}
        self.tickers = {}
        self.btc_filter = None
        self.ref_time_ms = ref_time_ms  # None → live

    def now_utc_hour(self):
        if self.ref_time_ms is None:
            return datetime.now(timezone.utc).hour
        return datetime.fromtimestamp(self.ref_time_ms / 1000, tz=timezone.utc).hour


# =====================================================================
# صرافی و داده
# =====================================================================
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
    if ctx.ref_time_ms is not None:
        df = get_data(ctx, symbol, "1h")
        return float(df["close"].iloc[-1])
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
        return rej("01_no_recent_4h_cross")

    last4 = df_4h.iloc[-1]
    atr_4h = float(last4["atr"])
    if np.isnan(atr_4h) or atr_4h <= 0:
        return rej("02_bad_atr_4h")

    adx_4h = float(last4["adx"])
    if np.isnan(adx_4h):
        return rej("03_bad_adx_4h")

    if adx_4h < B1_MIN_ADX_4H:
        return rej("04_adx_4h")

    ema9_4h = float(last4["ema9"])
    ema21_4h = float(last4["ema21"])
    ema50_4h = float(last4["ema50"])

    gap_4h = abs(ema9_4h - ema21_4h)
    if gap_4h < 0.35 * atr_4h:
        return rej("05_gap_4h")

    ema9_slope = (ema9_4h - float(df_4h["ema9"].iloc[-4]))
    slope_atr = (abs(ema9_slope) / atr_4h)

    if slope_atr < B1_MIN_SLOPE_ATR:
        return rej("06_slope_weak")

    if direction == "LONG" and ema9_slope <= 0:
        return rej("07_slope_direction")
    if direction == "SHORT" and ema9_slope >= 0:
        return rej("07_slope_direction")

    if direction == "LONG":
        if ema21_4h <= ema50_4h:
            return rej("08_ema21_vs_ema50")
    else:
        if ema21_4h >= ema50_4h:
            return rej("08_ema21_vs_ema50")

    df_1h = get_data(ctx, symbol, "1h")
    if trend_of(df_1h) != direction:
        return rej("09_trend_1h_mismatch")

    last_1h = df_1h.iloc[-1]
    atr_1h = float(last_1h["atr"])
    if np.isnan(atr_1h) or atr_1h <= 0:
        return rej("10_bad_atr_1h")

    adx_1h = float(last_1h["adx"])
    if np.isnan(adx_1h):
        return rej("11_bad_adx_1h")

    if adx_1h < B1_MIN_ADX_1H:
        return rej("12_adx_1h")

    gap_1h = abs(float(last_1h["ema9"]) - float(last_1h["ema21"]))
    if gap_1h < 0.40 * atr_1h:
        return rej("13_gap_1h")

    vol_1h = float(last_1h["volume"])
    avg_vol_1h = float(df_1h["volume"].iloc[-21:-1].mean())
    vol_ratio_1h = ratio(vol_1h, avg_vol_1h)

    if vol_ratio_1h < B1_VOLUME_MULT:
        return rej("14_volume_1h")

    df_15m = get_data(ctx, symbol, "15m")
    last = df_15m.iloc[-1]
    rsi = float(last["rsi"])
    if np.isnan(rsi):
        return rej("15_bad_rsi")

    lo, hi = (B1_RSI_LONG if direction == "LONG" else B1_RSI_SHORT)
    if not (lo <= rsi <= hi):
        return rej("16_rsi_15m")

    close = float(last["close"])
    if (atr_1h * B1_SL_ATR) / close < MIN_SL_PCT:
        return rej("17_min_sl_pct")

    trend1_score = (min(gap_1h / atr_1h, 1.5) / 1.5)
    trend4_score = (min(gap_4h / atr_4h, 1.5) / 1.5)
    vol_score = (min(vol_ratio_1h, 3.0) / 3.0)
    cross_score = (B1_CROSS_LOOKBACK - cross_age) / B1_CROSS_LOOKBACK
    rsi_score = min(abs(rsi - 50) / 20.0, 1.0)
    adx_score = min(adx_4h / 35.0, 1.0)

    score = (0.25 * trend1_score + 0.25 * trend4_score + 0.20 * adx_score +
             0.15 * vol_score + 0.10 * cross_score + 0.05 * rsi_score)

    if score < B1_MIN_SCORE:
        return rej("18_score")

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
        return rej("01_short_data")

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
        return rej("02_bad_values")

    if range_size > B2_MAX_RANGE_ATR * atr:
        return rej("03_range_too_wide")

    direction = level = None

    if (close > range_high and float(prev["close"]) <= range_high and
            (close - low) / rng >= B2_CLOSE_POS and (close - range_high) <= B2_MAX_EXTENSION_ATR * atr):
        direction, level = "LONG", range_high

    elif (close < range_low and float(prev["close"]) >= range_low and
          (high - close) / rng >= B2_CLOSE_POS and (range_low - close) <= B2_MAX_EXTENSION_ATR * atr):
        direction, level = "SHORT", range_low

    if direction is None:
        return rej("04_no_breakout")

    if body < B2_MIN_BODY_ATR * atr:
        return rej("05_body_weak")

    adx = float(last["adx"]) if not np.isnan(last["adx"]) else 0.0
    if adx < B2_MIN_ADX:
        return rej("06_adx_weak")

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

    if not (rsi_ok and vol_ok and trend_ok):
        tags = []
        if not rsi_ok: tags.append("rsi")
        if not vol_ok: tags.append("vol")
        if not trend_ok: tags.append("trend4h")
        return rej("07_filters_" + "+".join(tags))

    if (atr * B2_SL_ATR) / close < MIN_SL_PCT:
        return rej("08_min_sl_pct")

    strength = min(abs(close - level) / atr, 1.0)
    body_score = min(body / atr, 1.5) / 1.5
    atr_4h = float(l4["atr"]) if not np.isnan(l4["atr"]) else 0.0
    gap_4h = abs(float(l4["ema21"]) - float(l4["ema50"]))
    trend_score = min(gap_4h / atr_4h, 1.0) if atr_4h > 0 else 0.0

    score = (0.35 * min(vol_ratio, 4.0) / 4.0 + 0.25 * strength +
             0.20 * body_score + 0.20 * trend_score)

    if score < B2_MIN_SCORE:
        return rej("09_score")

    return {
        "symbol": symbol, "direction": direction, "close": close,
        "atr": atr, "rsi": rsi, "volume": volume, "avg_volume": avg_volume,
        "signal_key": int(last["timestamp"]), "score": score,
        "setup": f"Breakout {lb}x1H ({'high' if direction == 'LONG' else 'low'} {format_number(level)})",
        "levels": atr_levels(direction, atr, B2_SL_ATR, B2_TP_ATR),
    }


# =====================================================================
# استراتژی ۳: MTF Pullback
# =====================================================================
def analyze_mtf_pullback(symbol, ctx):
    try:
        df_1d = get_data(ctx, symbol, "1d")
    except Exception:
        return rej("01_no_1d_data")
    if len(df_1d) < 200:
        return rej("02_1d_lt_200_candles")

    ema50_d = df_1d["close"].ewm(span=50, adjust=False).mean()
    ema200_d = df_1d["close"].ewm(span=200, adjust=False).mean()
    e50 = float(ema50_d.iloc[-1])
    e200 = float(ema200_d.iloc[-1])
    close_d = float(df_1d["close"].iloc[-1])
    if any(np.isnan(x) for x in (e50, e200)) or e200 <= 0:
        return rej("03_bad_daily_ema")

    sep = abs(e50 - e200) / e200
    if close_d > e200 and e50 > e200 and sep >= B3_MIN_SEP:
        direction = "LONG"
    elif close_d < e200 and e50 < e200 and sep >= B3_MIN_SEP:
        direction = "SHORT"
    else:
        return rej("04_no_daily_bias")

    try:
        df = get_data(ctx, symbol, "1h")
    except Exception:
        return rej("05_no_1h_data")
    if len(df) < 50:
        return rej("06_1h_short_data")

    last = df.iloc[-1]
    atr = float(last["atr"])
    rsi = float(last["rsi"])
    ema20 = float(last["ema21"])
    if np.isnan(atr) or np.isnan(rsi) or atr <= 0:
        return rej("07_bad_atr_rsi")

    c = float(last["close"]); o = float(last["open"])
    h = float(last["high"]); l = float(last["low"])
    rng = h - l
    if rng <= 0:
        return rej("08_zero_range")
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
            return rej("09_no_pullback_move")
        dist = abs(c - level) / level
        if dist > B3_NEAR_PCT:
            return rej("10_not_near_level")
        if not (B3_RSI_LONG[0] <= rsi <= B3_RSI_LONG[1]):
            return rej("11_rsi")
        if body_ratio < B3_BODY_MIN or c <= o:
            return rej("12_candle_body_dir")
        if c < ema20:
            return rej("13_vs_ema21")
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
            return rej("09_no_pullback_move")
        dist = abs(c - level) / level
        if dist > B3_NEAR_PCT:
            return rej("10_not_near_level")
        if not (B3_RSI_SHORT[0] <= rsi <= B3_RSI_SHORT[1]):
            return rej("11_rsi")
        if body_ratio < B3_BODY_MIN or c >= o:
            return rej("12_candle_body_dir")
        if c > ema20:
            return rej("13_vs_ema21")
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
        return rej("14_score")

    return {
        "symbol": symbol, "direction": direction, "close": c, "atr": atr, "rsi": rsi,
        "volume": volume, "avg_volume": avg_volume, "signal_key": int(last["timestamp"]),
        "score": score, "setup": f"MTF Pullback 1H (Daily {direction})", "levels": levels,
    }


# =====================================================================
# استراتژی v30: AMD + Session Liquidity (اصلاح‌شده)
# =====================================================================
def _utc_hour(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour


def _in_active_session(ctx):
    h = ctx.now_utc_hour()
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

    asia_hi = max(highs[-8:])
    asia_lo = min(lows[-8:])
    asia_end_ts = ts[-1]
    return {"high": asia_hi, "low": asia_lo, "end_ts": asia_end_ts}


def analyze_amd_session(symbol, ctx):
    if not _in_active_session(ctx):
        return rej("01_outside_active_session")

    df4 = get_data(ctx, symbol, "4h")
    l4 = df4.iloc[-1]
    bias_4h = trend_of(df4)
    atr4 = float(l4["atr"])
    if bias_4h not in ("LONG", "SHORT") or np.isnan(atr4) or atr4 <= 0:
        return rej("02_bad_4h_bias_atr")
    gap4 = abs(float(l4["ema9"]) - float(l4["ema21"]))
    if gap4 < B30_TREND_GAP_4H_ATR * atr4:
        return rej("03_gap_4h")

    df1 = get_data(ctx, symbol, "1h")
    if len(df1) < 30:
        return rej("04_1h_short_data")
    atr1 = float(df1["atr"].iloc[-1])
    if np.isnan(atr1) or atr1 <= 0:
        return rej("05_bad_atr_1h")

    asia = _asia_range_1h(df1)
    if asia is None:
        return rej("06_no_asia_range")

    rng_size = asia["high"] - asia["low"]
    if rng_size < B30_MIN_RANGE_ATR * atr1 or rng_size > B30_MAX_RANGE_ATR * atr1:
        return rej("07_asia_range_invalid")

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
        return rej("08_no_sweep")

    if sweep["direction"] != bias_4h:
        return rej("09_sweep_vs_4h_bias")

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
                fvg = {"bottom": bottom, "top": top, "ts": int(ts15[i]), "i": i}
                break
        if direction == "SHORT" and l[i - 2] > h[i]:
            bottom, top = float(h[i]), float(l[i - 2])
            if (c[i + 1:last] > top).any():
                continue
            mid = (bottom + top) / 2
            if h[last] >= bottom and c[last] <= mid * 1.02:
                fvg = {"bottom": bottom, "top": top, "ts": int(ts15[i]), "i": i}
                break

    if fvg is None:
        return rej("10_no_fvg")

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
        return rej("11_levels_rr")

    risk = abs(price - sl)
    if risk / price < MIN_SL_PCT:
        return rej("12_min_sl_pct")
    rr = abs(tp - price) / risk

    depth = abs(sweep["level"] - sweep["extreme"]) / atr1
    vol = float(v[last])
    avg_vol = float(df15["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(vol, avg_vol)
    rsi = float(df15["rsi"].iloc[-1]) if not np.isnan(df15["rsi"].iloc[-1]) else 50.0

    score = (0.35 * min(depth / 0.6, 1.0) + 0.30 * min(rr / 3.0, 1.0) +
             0.20 * min(vol_ratio / 2.0, 1.0) + 0.15 * min(gap4 / atr4, 1.0))

    if score < B30_MIN_SCORE:
        return rej("13_score")

    fvg_size = fvg["top"] - fvg["bottom"]
    if direction == "LONG":
        gap_pos = (float(c[last]) - fvg["bottom"]) / fvg_size if fvg_size > 0 else ""
        with_trade = 1 if c[last] > o[last] else 0
    else:
        gap_pos = (fvg["top"] - float(c[last])) / fvg_size if fvg_size > 0 else ""
        with_trade = 1 if c[last] < o[last] else 0

    meta = {
        "fvg_size_atr": round(fvg_size / atr1, 3),
        "fvg_age_15m": int(last - fvg["i"]),
        "gap_pos": round(gap_pos, 3) if gap_pos != "" else "",
        "last_candle_with_trade": with_trade,
        "sweep_depth_atr": round(depth, 3),
        "sweep_age_h": round((int(ts1[-1]) - sweep["ts"]) / 3600000, 1),
        "asia_range_atr": round(rng_size / atr1, 3),
        "gap4_atr": round(gap4 / atr4, 3),
        "risk_atr": round(risk / atr1, 3),
    }

    return {
        "symbol": symbol, "direction": direction, "close": price, "atr": atr1, "rsi": rsi,
        "volume": vol, "avg_volume": avg_vol, "signal_key": fvg["ts"], "score": score,
        "setup": f"AMD: Asia sweep + FVG ({direction})", "levels": levels,
        "meta": meta,
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
        return rej("01_bad_atr_rsi_adx")

    volume = float(last["volume"])
    avg_volume = float(df["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(volume, avg_volume)

    if adx >= B4_ADX_TREND: mode = "trend"
    elif adx <= B4_ADX_RANGE: mode = "range"
    else:
        return rej("02_adx_between_regimes")

    if mode == "trend":
        df_4h = get_data(ctx, symbol, "4h")
        l4 = df_4h.iloc[-1]
        direction = trend_of(df_4h)
        atr_4h = float(l4["atr"])
        if direction not in ("LONG", "SHORT") or np.isnan(atr_4h) or atr_4h <= 0:
            return rej("T1_bad_4h_dir_atr")
        gap_4h = abs(float(l4["ema9"]) - float(l4["ema21"]))
        if gap_4h < B4_TREND_GAP_4H_ATR * atr_4h:
            return rej("T2_gap_4h")

        try:
            btc_4h = trend_of(get_data(ctx, BTC_SYMBOL, "4h"))
            if btc_4h in ("LONG", "SHORT") and direction != btc_4h:
                return rej("T3_btc_4h_mismatch")
        except Exception:
            pass

        ema21 = float(last["ema21"])
        if direction == "LONG":
            touched = (low <= ema21 + B4_TREND_TOUCH_ATR * atr and close > ema21 and close > open_ and close - ema21 <= B4_TREND_MAX_EXT_ATR * atr)
            rsi_ok = B4_TREND_RSI_LONG[0] <= rsi <= B4_TREND_RSI_LONG[1]
        else:
            touched = (high >= ema21 - B4_TREND_TOUCH_ATR * atr and close < ema21 and close < open_ and ema21 - close <= B4_TREND_MAX_EXT_ATR * atr)
            rsi_ok = B4_TREND_RSI_SHORT[0] <= rsi <= B4_TREND_RSI_SHORT[1]

        vol_ok = vol_ratio >= B4_TREND_MIN_VOL

        if not (touched and rsi_ok and vol_ok):
            tags = []
            if not touched: tags.append("pullback")
            if not rsi_ok: tags.append("rsi")
            if not vol_ok: tags.append("vol")
            return rej("T4_filters_" + "+".join(tags))
        if (atr * B4_TREND_SL_ATR) / close < MIN_SL_PCT:
            return rej("T5_min_sl_pct")

        score = (0.45 * min(adx / 40.0, 1.0) + 0.30 * min(gap_4h / atr_4h, 1.0) + 0.25 * min(vol_ratio / 2.0, 1.0))
        if score < B4_MIN_SCORE:
            return rej("T6_score")

        return {
            "symbol": symbol, "direction": direction, "close": close, "atr": atr, "rsi": rsi,
            "volume": volume, "avg_volume": avg_volume, "signal_key": int(last["timestamp"]), "score": score,
            "setup": "رژیم روند: پولبک به میانگین ۱H", "levels": atr_levels(direction, atr, B4_TREND_SL_ATR, B4_TREND_TP_ATR),
        }

    bb_lo, bb_mid, bb_up = (float(last[k]) for k in ("bb_lo", "bb_mid", "bb_up"))
    if any(np.isnan(v) for v in (bb_lo, bb_mid, bb_up)):
        return rej("R1_bad_bollinger")

    if low <= bb_lo and close > bb_lo and rsi <= B4_RANGE_RSI_LONG_MAX:
        direction = "LONG"
        sl = low - B4_RANGE_SL_BUFFER_ATR * atr
        tp = bb_mid
    elif high >= bb_up and close < bb_up and rsi >= B4_RANGE_RSI_SHORT_MIN:
        direction = "SHORT"
        sl = high + B4_RANGE_SL_BUFFER_ATR * atr
        tp = bb_mid
    else:
        return rej("R2_no_band_reversal")

    if (direction == "LONG" and not (sl < close < tp)) or (direction == "SHORT" and not (tp < close < sl)):
        return rej("R3_bad_levels_order")
    risk = abs(close - sl)
    rr = abs(tp - close) / risk
    if rr < B4_RANGE_MIN_RR or risk / close < MIN_SL_PCT:
        return rej("R4_rr_or_min_sl")

    def levels(entry):
        if direction == "LONG": return (sl, tp) if sl < entry < tp else None
        return (sl, tp) if tp < entry < sl else None

    score = (0.50 * min(rr / 3.0, 1.0) + 0.30 * min(abs(rsi - 50) / 25.0, 1.0) + 0.20 * min(vol_ratio / 2.0, 1.0))
    if score < B4_MIN_SCORE:
        return rej("R5_score")

    return {
        "symbol": symbol, "direction": direction, "close": close, "atr": atr, "rsi": rsi,
        "volume": volume, "avg_volume": avg_volume, "signal_key": int(last["timestamp"]), "score": score,
        "setup": "رژیم رنج: بازگشت از باند به میانگین ۱H", "levels": levels,
    }


# =====================================================================
# استراتژی ۵: SMC/ICT
# =====================================================================
def analyze_smc(symbol, ctx):
    df4 = get_data(ctx, symbol, "4h")
    l4 = df4.iloc[-1]
    direction = trend_of(df4)
    atr4 = float(l4["atr"])

    if direction not in ("LONG", "SHORT") or np.isnan(atr4) or atr4 <= 0:
        return rej("01_bad_4h_dir_atr")

    trend_gap = abs(float(l4["ema9"]) - float(l4["ema21"]))
    if trend_gap < B5_TREND_GAP_4H_ATR * atr4:
        return rej("02_gap_4h")

    df1 = get_data(ctx, symbol, "1h")
    n1 = len(df1)

    if n1 < B5_RANGE_BARS + B5_LOOKBACK_1H + 5:
        return rej("03_1h_short_data")

    atr1 = float(df1["atr"].iloc[-1])
    if np.isnan(atr1) or atr1 <= 0:
        return rej("04_bad_atr_1h")

    hi1 = df1["high"].values
    lo1 = df1["low"].values
    cl1 = df1["close"].values
    ts1 = df1["timestamp"].values

    sweep = None
    for m in range(n1 - 1, n1 - 1 - B5_SWEEP_RECENT, -1):
        win = slice(m - B5_LOOKBACK_1H, m)
        if direction == "LONG":
            level = float(lo1[win].min())
            if lo1[m] < level < cl1[m]:
                sweep = {"ts": int(ts1[m]), "extreme": float(lo1[m]), "level": level}
                break
        else:
            level = float(hi1[win].max())
            if cl1[m] < level < hi1[m]:
                sweep = {"ts": int(ts1[m]), "extreme": float(hi1[m]), "level": level}
                break

    if sweep is None:
        return rej("05_no_sweep")

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
        return rej("06_no_bos")

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
        return rej("07_price_vs_equilibrium")
    if (direction == "SHORT" and price < eq):
        return rej("07_price_vs_equilibrium")

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
        return rej("08_no_quality_fvg")

    vol = float(v[last])
    avg_vol = float(df15["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(vol, avg_vol)

    if vol_ratio < B5_MIN_VOLUME_RATIO:
        return rej("09_weak_volume")

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
        return rej("10_levels_rr")

    risk = abs(price - sl)
    if risk / price < MIN_SL_PCT:
        return rej("11_min_sl_pct")

    rr = abs(tp - price) / risk
    depth = abs(sweep["level"] - sweep["extreme"]) / atr1

    score = (
        0.30 * min(depth / 0.60, 1.0) +
        0.25 * min(vol_ratio / 2.0, 1.0) +
        0.25 * min(rr / 3.0, 1.0) +
        0.20 * min(fvg_size / atr1, 1.0)
    )

    if score < B5_MIN_SCORE:
        return rej("12_score")

    rsi = float(df15["rsi"].iloc[-1])
    if np.isnan(rsi):
        rsi = 50.0

    return {
        "symbol": symbol, "direction": direction, "close": price, "atr": atr1, "rsi": rsi,
        "volume": vol, "avg_volume": avg_vol, "signal_key": fvg["ts"], "score": score,
        "setup": "SMC PRO (Sweep+BOS+FVG)", "levels": levels,
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
    print(f"🌍 BTC filter check: 1D={t_1d} | 4H={t_4h} | 1H={t_1h}")

    if t_1d not in ("LONG", "SHORT") or t_4h not in ("LONG", "SHORT"):
        print(f"🌍 BTC filter BLOCKED: trend unclear (1D={t_1d}, 4H={t_4h})")
        return None
    if t_1d != t_4h:
        print(f"🌍 BTC filter BLOCKED: 1D/4H conflict (1D={t_1d}, 4H={t_4h})")
        return None

    bias = t_1d
    last = df_4h.iloc[-1]
    atr, close = float(last["atr"]), float(last["close"])
    ema9, ema21 = float(last["ema9"]), float(last["ema21"])
    adx = float(last["adx"]) if not np.isnan(last["adx"]) else 0.0

    if np.isnan(atr) or atr <= 0:
        print(f"🌍 BTC filter BLOCKED: bad ATR ({atr})")
        return None

    gap = abs(ema9 - ema21)
    gap_atr = gap / atr
    strength_ok = (gap_atr >= B6_MIN_GAP_ATR) or (adx >= B6_MIN_ADX)
    if not strength_ok:
        print(f"🌍 BTC filter BLOCKED: weak trend (gap={gap_atr:.2f} ATR, ADX={adx:.1f})")
        return None

    ext_atr = abs(close - ema21) / atr
    if ext_atr > B6_MAX_EXT_ATR:
        print(f"🌍 BTC filter BLOCKED: overextended ({ext_atr:.2f} ATR from EMA21)")
        return None

    align_1h = t_1h == bias
    print(f"🌍 BTC filter PASSED | bias={bias} | gap={gap_atr:.2f} ADX={adx:.1f} ext={ext_atr:.2f} | 1H_align={align_1h}")

    return {"bias": bias, "t_1d": t_1d, "t_4h": t_4h, "t_1h": t_1h, "align_1h": align_1h,
            "gap_atr": gap_atr, "adx": adx, "ext_atr": ext_atr}


def analyze_btc_aligned(symbol, ctx):
    btc = getattr(ctx, "btc_filter", None)
    if not btc:
        # اگر فیلتر BTC موجود نبود، از روند ۴ ساعته خود نماد استفاده کن
        try:
            df_4h = get_data(ctx, symbol, "4h")
            bias = trend_of(df_4h)
            if bias not in ("LONG", "SHORT"):
                return rej("01_no_bias")
        except Exception:
            return rej("01_btc_filter_blocked")
    else:
        bias = btc["bias"]

    df_4h = get_data(ctx, symbol, "4h")
    if trend_of(df_4h) != bias: 
        return rej("02_symbol_4h_vs_btc_bias")

    l4 = df_4h.iloc[-1]
    atr_4h = float(l4["atr"])
    if np.isnan(atr_4h) or atr_4h <= 0: 
        return rej("03_bad_atr_4h")
    gap_4h = abs(float(l4["ema9"]) - float(l4["ema21"]))
    if gap_4h < B6_ALT_GAP_ATR * atr_4h: 
        return rej("04_gap_4h")

    df_1h = get_data(ctx, symbol, "1h")
    last = df_1h.iloc[-1]
    close, open_, high, low = (float(last[k]) for k in ("close", "open", "high", "low"))
    atr, rsi, ema21 = float(last["atr"]), float(last["rsi"]), float(last["ema21"])

    if np.isnan(atr) or np.isnan(rsi) or atr <= 0: 
        return rej("05_bad_atr_rsi_1h")

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

    if not (touched and rsi_ok and vol_ok):
        tags = []
        if not touched: tags.append("pullback")
        if not rsi_ok: tags.append("rsi")
        if not vol_ok: tags.append("vol")
        return rej("06_filters_" + "+".join(tags))
    if (atr * B6_SL_ATR) / close < MIN_SL_PCT: 
        return rej("07_min_sl_pct")

    trend_score = min(gap_4h / atr_4h, 1.2) / 1.2
    vol_score = min(vol_ratio / 2.5, 1.0)
    btc_score = 0.55 + (B6_1H_ALIGN_BONUS if btc and btc.get("align_1h") else 0.0)
    if btc:
        btc_score += 0.15 * min(btc["gap_atr"] / 0.6, 1.0)
    score = (0.35 * btc_score + 0.30 * trend_score + 0.20 * vol_score + 0.15 * min(abs(rsi - 50) / 20.0, 1.0))

    if score < B6_MIN_SCORE:
        return rej("08_score")

    return {
        "symbol": symbol, "direction": bias, "close": close, "atr": atr, "rsi": rsi,
        "volume": volume, "avg_volume": avg_volume, "signal_key": int(last["timestamp"]), "score": score,
        "setup": f"BTC-aligned pullback 1H (BTC {bias})", "levels": atr_levels(bias, atr, B6_SL_ATR, B6_TP_ATR),
    }


# =====================================================================
# استراتژی ۸: SMC Winner (v70) - نسخه اصلاح‌شده
# =====================================================================
def analyze_smc_winner(symbol, ctx):
    try:
        df4 = get_data(ctx, symbol, "4h")
    except Exception:
        return rej("01_no_4h_data")
    if len(df4) < 60:
        return rej("02_4h_short_data")

    curr = df4.iloc[-1]
    atr = float(curr["atr"])
    if np.isnan(atr) or atr <= 0:
        return rej("03_bad_atr")

    close = float(curr["close"])
    open_ = float(curr["open"])
    low = float(curr["low"])
    high = float(curr["high"])
    if close <= 0:
        return rej("04_bad_price")
    rsi = float(curr["rsi"]) if not np.isnan(curr["rsi"]) else 50.0
    ema21 = float(curr["ema21"])
    ema50 = float(curr["ema50"])

    prev_low = float(df4["low"].iloc[-B8_SWEEP_LOOKBACK - 1:-1].min())
    prev_high = float(df4["high"].iloc[-B8_SWEEP_LOOKBACK - 1:-1].max())

    if low < prev_low and close > prev_low and close > open_:
        direction = "LONG"
        if not (ema21 > ema50 and close > ema50):
            return rej("06_trend_filter")
        wick = min(open_, close) - low
        sl_dist = close - (low - atr * B8_SL_BUFFER_ATR)
    elif high > prev_high and close < prev_high and close < open_:
        direction = "SHORT"
        if not (ema21 < ema50 and close < ema50):
            return rej("06_trend_filter")
        wick = high - max(open_, close)
        sl_dist = (high + atr * B8_SL_BUFFER_ATR) - close
    else:
        return rej("05_no_sweep")

    if sl_dist <= 0:
        return rej("07_sl_dist_invalid")

    sl_dist = max(sl_dist, atr * B8_MIN_SL_ATR)
    if sl_dist > atr * B8_MAX_SL_ATR or sl_dist / close > B8_MAX_SL_PCT:
        return rej("08_sl_too_far")
    if sl_dist / close < MIN_SL_PCT:
        return rej("09_min_sl_pct")

    tp_dist = atr * B8_TP_ATR_MULT
    rr = tp_dist / sl_dist
    if rr < B8_MIN_RR:
        return rej("10_rr_low")

    volume = float(curr["volume"])
    avg_volume = float(df4["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(volume, avg_volume)
    if vol_ratio < B8_MIN_VOL_RATIO:
        return rej("11_volume")

    if direction == "LONG" and rsi > B8_RSI_MAX_LONG:
        return rej("12_rsi_extreme")
    if direction == "SHORT" and rsi < B8_RSI_MIN_SHORT:
        return rej("12_rsi_extreme")

    trend_score = min(abs(ema21 - ema50) / atr / 1.0, 1.0)
    vol_score = min(vol_ratio / 2.0, 1.0)
    wick_score = min((wick / atr) / 1.0, 1.0)
    rr_score = min(rr / 2.5, 1.0)
    score = 0.30 * trend_score + 0.25 * vol_score + 0.25 * wick_score + 0.20 * rr_score
    if score < B8_MIN_SCORE:
        return rej("13_score")

    def levels(entry):
        if direction == "LONG":
            return entry - sl_dist, entry + tp_dist
        return entry + sl_dist, entry - tp_dist

    sweep_depth = abs(prev_low - low) if direction == "LONG" else abs(high - prev_high)
    meta = {
        "fvg_size_atr": "",
        "fvg_age_15m": "",
        "gap_pos": "",
        "last_candle_with_trade": 1 if ((direction == "LONG" and close > open_) or (direction == "SHORT" and close < open_)) else 0,
        "sweep_depth_atr": round(sweep_depth / atr, 3),
        "sweep_age_h": 0,
        "asia_range_atr": "",
        "gap4_atr": "",
        "risk_atr": round(sl_dist / atr, 3),
    }

    return {
        "symbol": symbol, "direction": direction, "close": close,
        "atr": atr, "rsi": rsi, "volume": volume, "avg_volume": avg_volume,
        "signal_key": int(curr["timestamp"]), "score": score,
        "setup": f"SMC Winner ({direction} Sweep, RR {rr:.1f})",
        "levels": levels, "meta": meta,
    }


# =====================================================================
# Bot
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
        "server_messages.json", "last_signals.json",
        live=True),
    Bot("crypto-signal-bot-v10", "Breakout", analyze_breakout,
        "server_messages_v10.json", "last_signals_v10.json",
        live=False),
    Bot("crypto-signal-bot-v20", "MTF Pullback", analyze_mtf_pullback,
        "server_messages_v20.json", "last_signals_v20.json",
        cooldown_hours=B3_COOLDOWN_HOURS, live=True),  # ← فعال شد
    Bot("crypto-signal-bot-v30", "AMD Session", analyze_amd_session,
        "server_messages_v30.json", "last_signals_v30.json",
        cooldown_hours=B30_COOLDOWN_HOURS, btc_filter_4h=True, live=False),
    Bot("crypto-signal-bot-v40", "Regime", analyze_regime,
        "server_messages_v40.json", "last_signals_v40.json",
        cooldown_hours=B4_COOLDOWN_HOURS, btc_filter_4h=False, live=False),
    Bot("crypto-signal-bot-v50", "SMC", analyze_smc,
        "server_messages_v50.json", "last_signals_v50.json",
        cooldown_hours=B5_COOLDOWN_HOURS, live=True),
    Bot("crypto-signal-bot-v60", "BTC Filter", analyze_btc_aligned,
        "server_messages_v60.json", "last_signals_v60.json",
        cooldown_hours=B6_COOLDOWN_HOURS, btc_filter_4h=False, live=True),  # ← فعال شد
    Bot("crypto-signal-bot-v70", "SMC Winner", analyze_smc_winner,
        "server_messages_v70.json", "last_signals_v70.json",
        cooldown_hours=B8_COOLDOWN_HOURS, max_signals=B8_MAX_SIGNALS_PER_RUN,
        btc_filter_4h=True, live=True),
]
LIVE_BOTS = [b for b in BOTS if b.live]


# =====================================================================
# Heartbeat
# =====================================================================
HEARTBEAT_FILE = "server_messages.json"


def send_heartbeat_all(results, btc_4h, btc_1h, warning=None):
    date_gregorian, date_jalali, time_str = get_iran_time()

    # ۱. پاک کردن پیام‌های قبلی برای همه ربات‌ها
    for bot in LIVE_BOTS[1:]:
        old_ids = load_json(bot.heartbeat_file, {}).get("message_ids", [])
        for mid in old_ids: 
            delete_telegram(mid)
        save_json(bot.heartbeat_file, {"message_ids": []})

    # ۲. پاک کردن پیام‌های قبلی برای ربات اول
    old_main_ids = load_json(HEARTBEAT_FILE, {}).get("message_ids", [])
    for mid in old_main_ids:
        delete_telegram(mid)
    
    # ۳. ساخت پیام جدید
    lines, ok = [], 0
    for bot in LIVE_BOTS:
        r = results.get(bot.name, "ERROR")
        if r == "ERROR": 
            lines.append(f"❌ {bot.name}")
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

    # ۴. ارسال پیام جدید و ذخیره ID آن
    new_id = send_telegram(msg)
    if new_id:
        # ذخیره فقط ID جدید (چون قبلی‌ها پاک شدند)
        save_json(HEARTBEAT_FILE, {"message_ids": [new_id]})


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
            if cd[0] < opened_ms:
                continue
            high, low, close = cd[2], cd[3], cd[4]

            if direction == "LONG":
                hit_sl, hit_tp = low <= sl, high >= tp
            else:
                hit_sl, hit_tp = high >= sl, low <= tp

            if hit_sl and hit_tp:
                dist_tp = abs(close - tp)
                dist_sl = abs(close - sl)
                if dist_tp < dist_sl:
                    return "TP", tp
                return "SL", sl

            if hit_sl:
                return "SL", sl
            if hit_tp:
                return "TP", tp

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
# چرخه ربات
# =====================================================================
def run_bot(bot, ctx, btc_4h, btc_1h):
    _CUR["bot"] = bot.name
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
                funnel_count("~skip_open_trade")
                continue
            if (prev and bot.cooldown_ms and prev.get("closed_ms")
                    and t_now - prev["closed_ms"] < bot.cooldown_ms):
                funnel_count("~skip_cooldown")
                continue

            funnel_count("0_analyzed")
            c = bot.analyze(symbol, ctx)
            if c is None:
                continue
            funnel_count("PASSED_analyze")

            prev_key = (prev.get("signal_key", prev.get("cross_ts")) if prev else None)
            if prev and prev_key == c["signal_key"] and prev.get("direction") == c["direction"]:
                funnel_count("~already_signaled")
                continue

            if symbol != BTC_SYMBOL:
                if bot.btc_filter_4h and btc_4h in ("LONG", "SHORT") and c["direction"] != btc_4h:
                    funnel_count("~btc_4h_filter")
                    continue
                if bot.btc_filter_1h and btc_1h in ("LONG", "SHORT") and c["direction"] != btc_1h:
                    funnel_count("~btc_1h_filter")
                    continue

            candidates.append(c)
        except Exception as e:
            funnel_count("EXCEPTION")
            print(f"⚠️ خطا در {symbol}: {e}")
            continue

    candidates.sort(key=lambda x: x["score"], reverse=True)

    sent = 0
    open_dir = Counter(r.get("direction") for r in signals.values() if r.get("status") == "OPEN")
    for c in candidates:
        if sent >= bot.max_signals: break
        symbol = c["symbol"]
        try:
            if open_dir[c["direction"]] >= MAX_OPEN_SAME_DIR:
                funnel_count("~cap_same_direction")
                continue
            entry = get_price(ctx, symbol)
            if abs(entry - c["close"]) > c["atr"] * MAX_ENTRY_DRIFT_ATR:
                funnel_count("~send_price_drift")
                continue

            lv = c["levels"](entry)
            if lv is None:
                funnel_count("~send_levels_none")
                continue
            sl, tp = lv
            if sl <= 0 or tp <= 0:
                funnel_count("~send_invalid_sl_tp")
                continue
            rr = abs(tp - entry) / abs(entry - sl)

            msg_id = send_telegram(build_signal_message(bot, c, entry, sl, tp, rr))

            opened_ts = ctx.ref_time_ms if ctx.ref_time_ms is not None else now_ms()

            record = {
                "direction": c["direction"], "entry": entry, "tp": tp, "sl": sl,
                "signal_key": c["signal_key"], "setup": c["setup"],
                "score": round(c["score"], 4),
                "rsi": round(c["rsi"], 2),
                "vol_ratio": round(ratio(c["volume"], c["avg_volume"]), 2),
                "atr_pct": round(c["atr"] / entry * 100, 3),
                "risk_pct": round(abs(entry - sl) / entry * 100, 3),
                "rr": round(rr, 2),
                "drift_atr": round(abs(entry - c["close"]) / c["atr"], 3),
                "btc_4h": btc_4h, "btc_1h": btc_1h,
                "open_same_dir": open_dir[c["direction"]],
                **c.get("meta", {}),
                "time": datetime.now(timezone.utc).isoformat(),
                "telegram_sent": bool(msg_id), "telegram_message_id": msg_id,
            }
            if msg_id:
                record["status"] = "OPEN"
                record["opened_ms"] = opened_ts
                sent += 1
                open_dir[c["direction"]] += 1
                funnel_count("SENT_signal")
                excel_log.log_signal(bot.name, symbol, c["direction"], entry, tp, sl)
            else:
                record["status"] = "NOT_SENT"
                funnel_count("~telegram_not_sent")
            signals[symbol] = record
            save_json(bot.signals_file, signals)
        except Exception as e:
            funnel_count("EXCEPTION_send")
            print(f"⚠️ خطا در ارسال {symbol}: {e}")
            continue

    print_funnel(bot.name)
    return sent


# =====================================================================
# خلاصه روزانه
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
    # ← اصلاح: چاپ 1D هم برای دیباگ
    try:
        btc_1d_df = get_data(ctx, BTC_SYMBOL, "1d")
        btc_1d = trend_of(btc_1d_df)
    except Exception:
        btc_1d = "UNKNOWN"
    print(f"🌍 BTC Trend: 1D={btc_1d} | 4H={btc_4h} | 1H={btc_1h}")
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
    except Exception as e:
        print(f"⚠️ خطا در heartbeat: {e}")

    try:
        maybe_send_daily_summary()
    except Exception as e:
        print(f"⚠️ خطا در خلاصه‌ی روزانه: {e}")

    print("All bots complete.")
    print("=" * 50)