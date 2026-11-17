"""
ربات‌های سیگنال با تمرکز روی کیفیت بالا (Win Rate) — نسخه سبک v3
فقط ۶ ربات فعال نگه داشته شده‌اند:
- crypto-signal-bot (EMA Cross)، v60 (BTC Filter)، v70 (SMC Winner)
- v0a, v60a, v70a با مدیریت سرمایه و چرخه جبران
- ربات‌های غیرفعال (v10, v20, v30, v40, v50) به همراه استراتژی و منطقشان حذف شده‌اند
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
    "CRV-SWAP-USDT", "SNX-SWAP-USDT", "COMP-SWAP-USDT", "SUSHI-SWAP-USDT",
    "1INCH-SWAP-USDT", "GRT-SWAP-USDT", "DYDX-SWAP-USDT", "GMX-SWAP-USDT",
    "PENDLE-SWAP-USDT", "ONDO-SWAP-USDT", "JTO-SWAP-USDT", "PYTH-SWAP-USDT",
    "STRK-SWAP-USDT", "MANTA-SWAP-USDT", "1000BONK-SWAP-USDT", "1000FLOKI-SWAP-USDT",
    "MEME-SWAP-USDT", "ORDI-SWAP-USDT", "1000SATS-SWAP-USDT", "1000RATS-SWAP-USDT",
    "POPCAT-SWAP-USDT", "BRETT-SWAP-USDT", "TURBO-SWAP-USDT", "1000000MOG-SWAP-USDT",
    "DOGS-SWAP-USDT", "FET-SWAP-USDT", "TAO-SWAP-USDT", "AKT-SWAP-USDT",
    "ILV-SWAP-USDT", "AXS-SWAP-USDT", "SAND-SWAP-USDT", "MANA-SWAP-USDT",
    "GALA-SWAP-USDT", "ALGO-SWAP-USDT", "VET-SWAP-USDT", "EGLD-SWAP-USDT",
    "FLOW-SWAP-USDT", "KAVA-SWAP-USDT", "ROSE-SWAP-USDT", "KAS-SWAP-USDT",
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

# =====================================================================
# تنظیمات مدیریت سرمایه (برای ربات‌های v0a, v60a, v70a)
# =====================================================================
INITIAL_BALANCE = 100.0   # بالانس اولیه هر ربات (دلار)
RISK_PERCENT = 3.0        # ریسک هر معامله (۳٪)
SL_PERCENT = 2.0          # حد ضرر (۲٪ از قیمت ورود)

TF_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

# --- تنظیمات استراتژی‌ها (فقط استراتژی‌های فعال) ---
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

B6_MIN_GAP_ATR = 0.28
B6_MIN_ADX = 20
B6_MAX_EXT_ATR = 6.0
B6_1H_ALIGN_BONUS = 0.12
B6_ALT_GAP_ATR = 0.22
B6_TOUCH_ATR = 0.22
B6_MAX_EXT_ALT_ATR = 1.0
B6_RSI_LONG = (40, 60)
B6_RSI_SHORT = (40, 60)
B6_VOLUME_MULT = 1.4
B6_SL_ATR, B6_TP_ATR = 1.4, 2.95
B6_MIN_SCORE = 0.65
B6_COOLDOWN_HOURS = 3

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
B8_MAX_SIGNALS_PER_RUN = 5

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
            r = requests.post(url, json={"chat_id": CHAT_ID, "text": message}, timeout=timeout)
            if r.status_code == 200:
                return r.json().get("result", {}).get("message_id")
            print(f"⚠️ تلگرام status={r.status_code} (تلاش {attempt}/{max_retries})")
        except Exception as e:
            print(f"⚠️ خطا در ارسال تلگرام: {e}")
        if attempt < max_retries:
            time.sleep(2 * attempt)
    return None

def delete_telegram(message_id, max_retries=2, timeout=15):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteMessage"
    for attempt in range(1, max_retries + 1):
        try:
            requests.post(url, json={"chat_id": CHAT_ID, "message_id": message_id}, timeout=timeout)
            return
        except Exception:
            if attempt < max_retries:
                time.sleep(1)

def format_number(value):
    if value >= 1000: return f"{value:.2f}"
    elif value >= 1: return f"{value:.4f}"
    else: return f"{value:.6f}"

def load_json(path, default):
    try:
        with open(path, "r") as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f)

def append_trade_csv(bot_name, record):
    import csv
    fname = f"trades_{bot_name.replace('crypto-signal-bot', 'bot')}.csv"
    file_exists = os.path.exists(fname)
    fields = ["bot", "symbol", "direction", "entry", "sl", "tp", "exit",
              "result", "pnl_r", "setup", "score", "rsi", "vol_ratio", "atr_pct",
              "risk_pct", "rr", "drift_atr", "btc_4h", "btc_1h", "open_same_dir",
              "position_size", "margin", "leverage", "profit_usd", "balance_after",
              "cycle_status"] + EXTRA_LOG_FIELDS + ["hold_hours", "opened_ms", "closed_ms"]
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
    except Exception: pass
    risk = abs(float(record.get("entry", 0) or 0) - float(record.get("sl", 0) or 0))
    pnl_r = ""
    if risk > 0 and record.get("result") in ("TP", "SL"):
        pnl_r = round(abs(float(record["tp"]) - float(record["entry"])) / risk, 3) if record["result"] == "TP" else -1.0
    row = {
        "bot": bot_name, "symbol": record.get("symbol", ""), "direction": record.get("direction"),
        "entry": record.get("entry"), "sl": record.get("sl"), "tp": record.get("tp"),
        "exit": record.get("close_price"), "result": record.get("result"), "pnl_r": pnl_r,
        "setup": record.get("setup", ""), "score": record.get("score", ""),
        "rsi": record.get("rsi", ""), "vol_ratio": record.get("vol_ratio", ""),
        "atr_pct": record.get("atr_pct", ""), "risk_pct": record.get("risk_pct", ""),
        "rr": record.get("rr", ""), "drift_atr": record.get("drift_atr", ""),
        "btc_4h": record.get("btc_4h", ""), "btc_1h": record.get("btc_1h", ""),
        "open_same_dir": record.get("open_same_dir", ""), "position_size": record.get("position_size", ""),
        "margin": record.get("margin", ""), "leverage": record.get("leverage", ""),
        "profit_usd": record.get("profit_usd", ""), "balance_after": record.get("balance_after", ""),
        "cycle_status": record.get("cycle_status", ""),
        "hold_hours": hold_hours, "opened_ms": record.get("opened_ms", ""), "closed_ms": record.get("closed_ms", "")
    }
    for k in EXTRA_LOG_FIELDS: row[k] = record.get(k, "")
    try:
        with open(fname, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not file_exists: w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"⚠️ CSV log error: {e}")

def now_ms(): return int(time.time() * 1000)

# =====================================================================
# Ctx و صرافی
# =====================================================================
class Ctx:
    def __init__(self, exchange, ref_time_ms=None):
        self.exchange = exchange
        self.cache = {}
        self.tickers = {}
        self.btc_filter = None
        self.ref_time_ms = ref_time_ms
    def now_utc_hour(self):
        if self.ref_time_ms is None: return datetime.now(timezone.utc).hour
        return datetime.fromtimestamp(self.ref_time_ms / 1000, tz=timezone.utc).hour

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
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    df["rsi"] = rsi
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_safe = df["atr"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_safe
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["adx"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    bb_mid = df["close"].rolling(20).mean()
    bb_sd = df["close"].rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_up"] = bb_mid + 2*bb_sd
    df["bb_lo"] = bb_mid - 2*bb_sd
    return df

def get_data(ctx, symbol, timeframe):
    key = (symbol, timeframe)
    if key not in ctx.cache:
        limit = 300 if timeframe == "1d" else 150
        df = fetch_ohlcv(ctx.exchange, symbol, timeframe, limit)
        if len(df) < 40:
            raise ValueError(f"داده کافی نیست: {symbol} {timeframe}")
        ctx.cache[key] = calculate_indicators(df)
    return ctx.cache[key]

def get_price(ctx, symbol):
    if ctx.ref_time_ms is not None:
        return float(get_data(ctx, symbol, "1h")["close"].iloc[-1])
    if symbol not in ctx.tickers:
        ctx.tickers[symbol] = float(ctx.exchange.fetch_ticker(symbol)["last"])
    return ctx.tickers[symbol]

def trend_of(df):
    last = df.iloc[-1]
    if last["ema9"] > last["ema21"]: return "LONG"
    if last["ema9"] < last["ema21"]: return "SHORT"
    return "NONE"

def get_btc_trend(ctx):
    try:
        return trend_of(get_data(ctx, BTC_SYMBOL, "4h")), trend_of(get_data(ctx, BTC_SYMBOL, "1h"))
    except Exception as e:
        print(f"⚠️ خطا در BTC: {e}")
        return "UNKNOWN", "UNKNOWN"

def atr_levels(direction, atr, sl_mult, tp_mult):
    def levels(entry):
        if direction == "LONG": return entry - atr*sl_mult, entry + atr*tp_mult
        return entry + atr*sl_mult, entry - atr*tp_mult
    return levels

def ratio(cur, avg): return cur / avg if avg and avg > 0 else 0.0

# =====================================================================
# استراتژی‌ها (فقط استراتژی‌های فعال: EMA Cross، BTC-aligned، SMC Winner)
# =====================================================================
def find_recent_cross(df, direction, lookback=B1_CROSS_LOOKBACK):
    if trend_of(df) != direction: return None, None
    n = len(df)
    for age in range(lookback):
        i = n - 1 - age
        if i < 1: break
        cur, prev = df.iloc[i], df.iloc[i-1]
        if direction == "LONG" and prev["ema9"] <= prev["ema21"] and cur["ema9"] > cur["ema21"]:
            return age, int(df["timestamp"].iloc[i])
        if direction == "SHORT" and prev["ema9"] >= prev["ema21"] and cur["ema9"] < cur["ema21"]:
            return age, int(df["timestamp"].iloc[i])
    return None, None

def analyze_cross(symbol, ctx):
    df_4h = get_data(ctx, symbol, "4h")
    direction, cross_age, cross_ts = None, None, None
    for d in ("LONG", "SHORT"):
        age, ts = find_recent_cross(df_4h, d, lookback=B1_CROSS_LOOKBACK)
        if age is not None: direction, cross_age, cross_ts = d, age, ts; break
    if direction is None: return rej("01_no_recent_4h_cross")
    last4 = df_4h.iloc[-1]
    atr_4h = float(last4["atr"])
    if np.isnan(atr_4h) or atr_4h <= 0: return rej("02_bad_atr_4h")
    adx_4h = float(last4["adx"])
    if np.isnan(adx_4h): return rej("03_bad_adx_4h")
    if adx_4h < B1_MIN_ADX_4H: return rej("04_adx_4h")
    ema9_4h, ema21_4h, ema50_4h = float(last4["ema9"]), float(last4["ema21"]), float(last4["ema50"])
    gap_4h = abs(ema9_4h - ema21_4h)
    if gap_4h < 0.35 * atr_4h: return rej("05_gap_4h")
    ema9_slope = ema9_4h - float(df_4h["ema9"].iloc[-4])
    slope_atr = abs(ema9_slope) / atr_4h
    if slope_atr < B1_MIN_SLOPE_ATR: return rej("06_slope_weak")
    if direction == "LONG" and ema9_slope <= 0: return rej("07_slope_direction")
    if direction == "SHORT" and ema9_slope >= 0: return rej("07_slope_direction")
    if direction == "LONG" and ema21_4h <= ema50_4h: return rej("08_ema21_vs_ema50")
    if direction == "SHORT" and ema21_4h >= ema50_4h: return rej("08_ema21_vs_ema50")
    df_1h = get_data(ctx, symbol, "1h")
    if trend_of(df_1h) != direction: return rej("09_trend_1h_mismatch")
    last_1h = df_1h.iloc[-1]
    atr_1h = float(last_1h["atr"])
    if np.isnan(atr_1h) or atr_1h <= 0: return rej("10_bad_atr_1h")
    adx_1h = float(last_1h["adx"])
    if np.isnan(adx_1h): return rej("11_bad_adx_1h")
    if adx_1h < B1_MIN_ADX_1H: return rej("12_adx_1h")
    gap_1h = abs(float(last_1h["ema9"]) - float(last_1h["ema21"]))
    if gap_1h < 0.40 * atr_1h: return rej("13_gap_1h")
    vol_1h = float(last_1h["volume"])
    avg_vol_1h = float(df_1h["volume"].iloc[-21:-1].mean())
    vol_ratio_1h = ratio(vol_1h, avg_vol_1h)
    if vol_ratio_1h < B1_VOLUME_MULT: return rej("14_volume_1h")
    df_15m = get_data(ctx, symbol, "15m")
    last = df_15m.iloc[-1]
    rsi = float(last["rsi"])
    if np.isnan(rsi): return rej("15_bad_rsi")
    lo, hi = (B1_RSI_LONG if direction == "LONG" else B1_RSI_SHORT)
    if not (lo <= rsi <= hi): return rej("16_rsi_15m")
    close = float(last["close"])
    if (atr_1h * B1_SL_ATR) / close < MIN_SL_PCT: return rej("17_min_sl_pct")
    trend1_score = min(gap_1h / atr_1h, 1.5) / 1.5
    trend4_score = min(gap_4h / atr_4h, 1.5) / 1.5
    vol_score = min(vol_ratio_1h, 3.0) / 3.0
    cross_score = (B1_CROSS_LOOKBACK - cross_age) / B1_CROSS_LOOKBACK
    rsi_score = min(abs(rsi - 50) / 20.0, 1.0)
    adx_score = min(adx_4h / 35.0, 1.0)
    score = (0.25*trend1_score + 0.25*trend4_score + 0.20*adx_score + 0.15*vol_score + 0.10*cross_score + 0.05*rsi_score)
    if score < B1_MIN_SCORE: return rej("18_score")
    return {"symbol": symbol, "direction": direction, "close": close, "atr": atr_1h,
            "rsi": rsi, "volume": vol_1h, "avg_volume": avg_vol_1h,
            "signal_key": cross_ts, "score": score,
            "setup": f"EMA Cross PRO (4H age {cross_age})",
            "levels": atr_levels(direction, atr_1h, B1_SL_ATR, B1_TP_ATR)}

def get_btc_filter(ctx):
    try:
        df_1d, df_4h, df_1h = get_data(ctx, BTC_SYMBOL, "1d"), get_data(ctx, BTC_SYMBOL, "4h"), get_data(ctx, BTC_SYMBOL, "1h")
    except Exception as e:
        print(f"⚠️ BTC filter data error: {e}")
        return None
    t_1d, t_4h, t_1h = trend_of(df_1d), trend_of(df_4h), trend_of(df_1h)
    print(f"🌍 BTC filter check: 1D={t_1d} | 4H={t_4h} | 1H={t_1h}")
    if t_1d not in ("LONG","SHORT") or t_4h not in ("LONG","SHORT"):
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
    gap_atr = abs(ema9 - ema21) / atr
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
    return {"bias":bias,"t_1d":t_1d,"t_4h":t_4h,"t_1h":t_1h,"align_1h":align_1h,"gap_atr":gap_atr,"adx":adx,"ext_atr":ext_atr}

def analyze_btc_aligned(symbol, ctx):
    btc = getattr(ctx, "btc_filter", None)
    if not btc:
        try:
            df_4h = get_data(ctx, symbol, "4h")
            bias = trend_of(df_4h)
            if bias not in ("LONG","SHORT"): return rej("01_no_bias")
        except Exception: return rej("01_btc_filter_blocked")
    else:
        bias = btc["bias"]
    df_4h = get_data(ctx, symbol, "4h")
    if trend_of(df_4h) != bias: return rej("02_symbol_4h_vs_btc_bias")
    l4 = df_4h.iloc[-1]
    atr_4h = float(l4["atr"])
    if np.isnan(atr_4h) or atr_4h <= 0: return rej("03_bad_atr_4h")
    gap_4h = abs(float(l4["ema9"]) - float(l4["ema21"]))
    if gap_4h < B6_ALT_GAP_ATR * atr_4h: return rej("04_gap_4h")
    df_1h = get_data(ctx, symbol, "1h")
    last = df_1h.iloc[-1]
    close, open_, high, low = (float(last[k]) for k in ("close","open","high","low"))
    atr, rsi, ema21 = float(last["atr"]), float(last["rsi"]), float(last["ema21"])
    if np.isnan(atr) or np.isnan(rsi) or atr <= 0: return rej("05_bad_atr_rsi_1h")
    if bias == "LONG":
        touched = (low <= ema21 + B6_TOUCH_ATR*atr and close > ema21 and close > open_ and close - ema21 <= B6_MAX_EXT_ALT_ATR*atr)
        rsi_ok = B6_RSI_LONG[0] <= rsi <= B6_RSI_LONG[1]
    else:
        touched = (high >= ema21 - B6_TOUCH_ATR*atr and close < ema21 and close < open_ and ema21 - close <= B6_MAX_EXT_ALT_ATR*atr)
        rsi_ok = B6_RSI_SHORT[0] <= rsi <= B6_RSI_SHORT[1]
    volume, avg_volume = float(last["volume"]), float(df_1h["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(volume, avg_volume)
    vol_ok = vol_ratio >= B6_VOLUME_MULT
    if not (touched and rsi_ok and vol_ok):
        tags = []
        if not touched: tags.append("pullback")
        if not rsi_ok: tags.append("rsi")
        if not vol_ok: tags.append("vol")
        return rej("06_filters_" + "+".join(tags))
    if (atr * B6_SL_ATR) / close < MIN_SL_PCT: return rej("07_min_sl_pct")
    trend_score = min(gap_4h/atr_4h, 1.2)/1.2
    vol_score = min(vol_ratio/2.5, 1.0)
    btc_score = 0.55 + (B6_1H_ALIGN_BONUS if btc and btc.get("align_1h") else 0.0)
    if btc: btc_score += 0.15 * min(btc["gap_atr"]/0.6, 1.0)
    score = 0.35*btc_score + 0.30*trend_score + 0.20*vol_score + 0.15*min(abs(rsi-50)/20.0,1.0)
    if score < B6_MIN_SCORE: return rej("08_score")
    return {"symbol": symbol, "direction": bias, "close": close, "atr": atr, "rsi": rsi,
            "volume": volume, "avg_volume": avg_volume, "signal_key": int(last["timestamp"]),
            "score": score, "setup": f"BTC-aligned pullback 1H (BTC {bias})",
            "levels": atr_levels(bias, atr, B6_SL_ATR, B6_TP_ATR)}

def analyze_smc_winner(symbol, ctx):
    try: df4 = get_data(ctx, symbol, "4h")
    except Exception: return rej("01_no_4h_data")
    if len(df4) < 60: return rej("02_4h_short_data")
    curr = df4.iloc[-1]
    atr = float(curr["atr"])
    if np.isnan(atr) or atr <= 0: return rej("03_bad_atr")
    close, open_, low, high = float(curr["close"]), float(curr["open"]), float(curr["low"]), float(curr["high"])
    if close <= 0: return rej("04_bad_price")
    rsi = float(curr["rsi"]) if not np.isnan(curr["rsi"]) else 50.0
    ema21, ema50 = float(curr["ema21"]), float(curr["ema50"])
    prev_low = float(df4["low"].iloc[-B8_SWEEP_LOOKBACK-1:-1].min())
    prev_high = float(df4["high"].iloc[-B8_SWEEP_LOOKBACK-1:-1].max())
    if low < prev_low and close > prev_low and close > open_:
        direction = "LONG"
        if not (ema21 > ema50 and close > ema50): return rej("06_trend_filter")
        wick = min(open_, close) - low
        sl_dist = close - (low - atr * B8_SL_BUFFER_ATR)
    elif high > prev_high and close < prev_high and close < open_:
        direction = "SHORT"
        if not (ema21 < ema50 and close < ema50): return rej("06_trend_filter")
        wick = high - max(open_, close)
        sl_dist = (high + atr * B8_SL_BUFFER_ATR) - close
    else: return rej("05_no_sweep")
    if sl_dist <= 0: return rej("07_sl_dist_invalid")
    sl_dist = max(sl_dist, atr * B8_MIN_SL_ATR)
    if sl_dist > atr * B8_MAX_SL_ATR or sl_dist / close > B8_MAX_SL_PCT: return rej("08_sl_too_far")
    if sl_dist / close < MIN_SL_PCT: return rej("09_min_sl_pct")
    tp_dist = atr * B8_TP_ATR_MULT
    rr = tp_dist / sl_dist
    if rr < B8_MIN_RR: return rej("10_rr_low")
    volume, avg_volume = float(curr["volume"]), float(df4["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(volume, avg_volume)
    if vol_ratio < B8_MIN_VOL_RATIO: return rej("11_volume")
    if direction == "LONG" and rsi > B8_RSI_MAX_LONG: return rej("12_rsi_extreme")
    if direction == "SHORT" and rsi < B8_RSI_MIN_SHORT: return rej("12_rsi_extreme")
    trend_score = min(abs(ema21 - ema50)/atr, 1.0)
    vol_score = min(vol_ratio/2.0, 1.0)
    wick_score = min(wick/atr, 1.0)
    rr_score = min(rr/2.5, 1.0)
    score = 0.30*trend_score + 0.25*vol_score + 0.25*wick_score + 0.20*rr_score
    if score < B8_MIN_SCORE: return rej("13_score")
    def levels(entry):
        if direction == "LONG": return entry - sl_dist, entry + tp_dist
        return entry + sl_dist, entry - tp_dist
    sweep_depth = abs(prev_low - low) if direction == "LONG" else abs(high - prev_high)
    meta = {"fvg_size_atr":"","fvg_age_15m":"","gap_pos":"",
            "last_candle_with_trade":1 if ((direction=="LONG" and close>open_) or (direction=="SHORT" and close<open_)) else 0,
            "sweep_depth_atr":round(sweep_depth/atr,3),"sweep_age_h":0,
            "asia_range_atr":"","gap4_atr":"","risk_atr":round(sl_dist/atr,3)}
    return {"symbol": symbol, "direction": direction, "close": close, "atr": atr, "rsi": rsi,
            "volume": volume, "avg_volume": avg_volume, "signal_key": int(curr["timestamp"]),
            "score": score, "setup": f"SMC Winner ({direction} Sweep, RR {rr:.1f})",
            "levels": levels, "meta": meta}

# =====================================================================
# مدیریت سرمایه و چرخه جبران
# =====================================================================
def calculate_position_sizing(bot, entry, sl, tp, balance, in_recovery, recovery_losses):
    sl_dist_pct = abs(entry - sl) / entry if entry > 0 else 0
    rr_ratio = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 2.0
    if not in_recovery:
        risk_amount = balance * (bot.risk_percent / 100.0)
    else:
        total_loss = sum(recovery_losses)
        risk_amount = total_loss / rr_ratio if rr_ratio > 0 else 0
    position_value = risk_amount / sl_dist_pct if sl_dist_pct > 0 else 0
    leverage_needed = position_value / balance if balance > 0 else 1.0
    position_size = position_value / entry if entry > 0 else 0
    return {"position_size": round(position_size, 6), "position_value": round(position_value, 2),
            "leverage": round(max(1.0, leverage_needed), 2), "risk_amount": round(risk_amount, 2)}

def update_open_trades_with_sizing(bot, exchange, signals):
    updated = False
    for symbol, data in list(signals.items()):
        if symbol.startswith("__"): continue
        if data.get("status") == "OPEN":
            try: result, close_price = check_trade_result(exchange, symbol, data)
            except Exception as e:
                print(f"⚠️ خطا در بررسی {symbol}: {e}"); continue
            if result:
                data["status"] = "CLOSED"
                data["result"] = result
                data["closed_ms"] = now_ms()
                data["close_price"] = float(close_price)
                data["close_telegram_sent"] = False
                data["close_msg_attempts"] = 0
                data["symbol"] = symbol
                entry_price = data["entry"]
                pos_size = data.get("position_size", 0)
                if result == "TP":
                    profit_usd = abs(data["tp"] - entry_price) * pos_size
                    new_balance = signals.get("__balance__", bot.initial_balance) + profit_usd
                    signals["__balance__"] = new_balance
                    signals["__in_recovery__"] = False
                    signals["__recovery_losses__"] = []
                    data["cycle_status"] = "ENDED (Win)"
                else:
                    loss_usd = abs(entry_price - data["sl"]) * pos_size
                    new_balance = signals.get("__balance__", bot.initial_balance) - loss_usd
                    signals["__balance__"] = new_balance
                    losses = signals.get("__recovery_losses__", [])
                    losses.append(loss_usd)
                    signals["__recovery_losses__"] = losses
                    signals["__in_recovery__"] = True
                    data["cycle_status"] = f"IN RECOVERY (Loss #{len(losses)})"
                data["profit_usd"] = round(profit_usd if result == "TP" else -loss_usd, 2)
                data["balance_after"] = round(new_balance, 2)
                updated = True
                append_trade_csv(bot.name, data)
                excel_log.log_result(bot.name, symbol, result, data["closed_ms"])
                print(f"{'🎯' if result == 'TP' else '🛑'} {symbol} {data['direction']} hit {result} at {close_price}")
        if (data.get("status") == "CLOSED" and data.get("close_telegram_sent") is False
                and data.get("close_msg_attempts", 0) < CLOSE_MSG_MAX_ATTEMPTS):
            msg_id = send_telegram(build_close_message(
                bot, symbol, data["direction"], data["result"], data["close_price"], data["entry"],
                profit_usd=data.get("profit_usd"), balance_after=data.get("balance_after"),
                cycle_status=data.get("cycle_status")))
            data["close_msg_attempts"] = data.get("close_msg_attempts", 0) + 1
            data["close_telegram_sent"] = bool(msg_id)
            updated = True
    return updated

# =====================================================================
# توابع مدیریت معاملات
# =====================================================================
CLOSE_MSG_MAX_ATTEMPTS = 5

def check_trade_result(exchange, symbol, data):
    direction, tp, sl = data["direction"], data["tp"], data["sl"]
    opened_ms = data.get("opened_ms")
    if opened_ms:
        try: candles = exchange.fetch_ohlcv(symbol, timeframe="5m", since=int(opened_ms), limit=1000)
        except Exception: candles = []
        for cd in candles:
            if cd[0] < opened_ms: continue
            high, low, close = cd[2], cd[3], cd[4]
            if direction == "LONG": hit_sl, hit_tp = low <= sl, high >= tp
            else: hit_sl, hit_tp = high >= sl, low <= tp
            if hit_sl and hit_tp:
                dist_tp = abs(close - tp); dist_sl = abs(close - sl)
                return ("TP", tp) if dist_tp < dist_sl else ("SL", sl)
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

def build_close_message(bot, symbol, direction, result, close_price, entry, profit_usd=None, balance_after=None, cycle_status=None):
    date_gregorian, date_jalali, time_str = get_iran_time()
    emoji = "🎯" if result == "TP" else "🛑"
    title = "حد سود (TP) لمس شد" if result == "TP" else "حد ضرر (SL) لمس شد"
    fin_lines = ""
    if profit_usd is not None:
        fin_lines = f"""
💰 سود/ضرر: ${profit_usd:+.2f}
💵 موجودی جدید: ${balance_after:.2f}
🔄 وضعیت چرخه: {cycle_status}
"""
    return f"""{emoji} {bot.title_tag} بسته شدن معامله
━━━━━━━━━━━━━━━━━
📌 نماد: {symbol}
📊 جهت: {direction}
🏁 نتیجه: {title}
💰 قیمت بسته شدن: {format_number(close_price)}
🚪 قیمت ورود: {format_number(entry)}
{fin_lines}
━━━━━━━━━━━━━━━━━
📅 تاریخ شمسی: {date_jalali}
📅 تاریخ میلادی: {date_gregorian}
⏰ ساعت: {time_str}
━━━━━━━━━━━━━━━━━"""

def build_signal_message(bot, c, entry, sl, tp, rr, sizing=None):
    date_gregorian, date_jalali, time_str = get_iran_time()
    emoji = "🟢" if c["direction"] == "LONG" else "🔴"
    fin_lines = ""
    if sizing:
        fin_lines = f"""
📦 حجم پوزیشن: {sizing['position_size']}
💵 ارزش دلاری: ${sizing['position_value']}
⚙️ اهرم: {sizing['leverage']}x
"""
    return f"""{emoji} {bot.title_tag} سیگنال {c['direction']}
━━━━━━━━━━━━━━━━━
📌 نماد: {c['symbol']}
🔎 ستاپ: {c['setup']}
💰 قیمت ورود: {format_number(entry)}
🎯 حد سود (TP): {format_number(tp)}
🛑 حد ضرر (SL): {format_number(sl)}
📊 نسبت R:R: 1:{rr:.1f}
{fin_lines}
━━━━━━━━━━━━━━━━━
📅 تاریخ شمسی: {date_jalali}
📅 تاریخ میلادی: {date_gregorian}
⏰ ساعت: {time_str}
━━━━━━━━━━━━━━━━━"""

# =====================================================================
# update_open_trades (برای ربات‌های بدون مدیریت سرمایه)
# =====================================================================
def update_open_trades(bot, exchange):
    signals = load_json(bot.signals_file, {})
    updated = False
    for symbol, data in list(signals.items()):
        if symbol.startswith("__"): continue
        if data.get("status") == "OPEN":
            try: result, close_price = check_trade_result(exchange, symbol, data)
            except Exception as e:
                print(f"⚠️ خطا در بررسی {symbol}: {e}"); continue
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
    if updated: save_json(bot.signals_file, signals)

# =====================================================================
# Bot
# =====================================================================
class Bot:
    def __init__(self, name, label, analyze, heartbeat_file, signals_file,
                 cooldown_hours=0, max_signals=MAX_SIGNALS_PER_RUN,
                 btc_filter_4h=True, btc_filter_1h=False, live=True,
                 single_trade=False, has_sizing=False,
                 initial_balance=INITIAL_BALANCE, risk_percent=RISK_PERCENT, sl_percent=SL_PERCENT):
        self.name = name
        self.label = label
        self.analyze = analyze
        self.heartbeat_file = heartbeat_file
        self.signals_file = signals_file
        self.cooldown_ms = int(cooldown_hours * 3600 * 1000)
        # ربات‌های تک‌معامله هیچ‌وقت نباید در یک اجرا بیش از یک سیگنال بدهند
        self.max_signals = 1 if single_trade else max_signals
        self.btc_filter_4h = btc_filter_4h
        self.btc_filter_1h = btc_filter_1h
        self.live = live
        self.single_trade = single_trade
        self.has_sizing = has_sizing
        self.initial_balance = initial_balance
        self.risk_percent = risk_percent
        self.sl_percent = sl_percent
    @property
    def title_tag(self):
        return f"[{self.name}]" + (f" ({self.label})" if self.label else "")

# =====================================================================
# BOTS نهایی (۶ ربات فعال)
# =====================================================================
BOTS = [
    # --- ۳ ربات اصلی ---
    Bot("crypto-signal-bot", "EMA Cross", analyze_cross,
        "server_messages.json", "last_signals.json", live=True),
    Bot("crypto-signal-bot-v60", "BTC Filter", analyze_btc_aligned,
        "server_messages_v60.json", "last_signals_v60.json",
        cooldown_hours=B6_COOLDOWN_HOURS, btc_filter_4h=False, live=True),
    Bot("crypto-signal-bot-v70", "SMC Winner", analyze_smc_winner,
        "server_messages_v70.json", "last_signals_v70.json",
        cooldown_hours=B8_COOLDOWN_HOURS, max_signals=B8_MAX_SIGNALS_PER_RUN,
        btc_filter_4h=True, live=True),
    # --- ۳ ربات با مدیریت سرمایه (تک‌معامله) ---
    Bot("crypto-signal-bot-v0a", "EMA Cross (Managed)", analyze_cross,
        "server_messages_v0a.json", "last_signals_v0a.json",
        live=True, single_trade=True, has_sizing=True,
        initial_balance=100.0, risk_percent=3.0, sl_percent=2.0),
    Bot("crypto-signal-bot-v60a", "BTC Filter (Managed)", analyze_btc_aligned,
        "server_messages_v60a.json", "last_signals_v60a.json",
        cooldown_hours=B6_COOLDOWN_HOURS, btc_filter_4h=False,
        live=True, single_trade=True, has_sizing=True,
        initial_balance=100.0, risk_percent=3.0, sl_percent=2.0),
    Bot("crypto-signal-bot-v70a", "SMC Winner (Managed)", analyze_smc_winner,
        "server_messages_v70a.json", "last_signals_v70a.json",
        cooldown_hours=B8_COOLDOWN_HOURS, max_signals=1,
        btc_filter_4h=True, live=True, single_trade=True, has_sizing=True,
        initial_balance=100.0, risk_percent=3.0, sl_percent=2.0),
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
        for mid in old: delete_telegram(mid)
        save_json(bot.heartbeat_file, {"message_ids": []})
    old_main_ids = load_json(HEARTBEAT_FILE, {}).get("message_ids", [])
    for mid in old_main_ids: delete_telegram(mid)
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
    new_id = send_telegram(msg)
    if new_id: save_json(HEARTBEAT_FILE, {"message_ids": [new_id]})

# =====================================================================
# خلاصه روزانه
# =====================================================================
def trade_r(rec):
    if rec.get("status") != "CLOSED": return None
    risk = abs(rec["entry"] - rec["sl"])
    if risk <= 0: return None
    return abs(rec["tp"] - rec["entry"]) / risk if rec.get("result") == "TP" else -1.0

def maybe_send_daily_summary():
    now = datetime.now(ZoneInfo("Asia/Tehran"))
    today = now.strftime("%Y-%m-%d")
    state = load_json(SUMMARY_FILE, {})
    if now.hour < SUMMARY_HOUR or state.get("date") == today: return
    date_gregorian, date_jalali, time_str = get_iran_time()
    lines = []
    for bot in LIVE_BOTS:
        recs = [r for k, r in load_json(bot.signals_file, {}).items() if not k.startswith("__")]
        closed = [r for r in recs if r.get("status") == "CLOSED"]
        opened = [r for r in recs if r.get("status") == "OPEN"]
        rs = [x for x in (trade_r(r) for r in closed) if x is not None]
        if rs:
            wins = sum(1 for r in closed if r.get("result") == "TP")
            lines.append(f"🤖 {bot.name}\n   بسته‌شده: {len(rs)} | برد: {wins} ({wins/len(rs):.0%}) | R کل: {sum(rs):+.1f} | باز: {len(opened)}")
        else:
            lines.append(f"🤖 {bot.name}\n   هنوز معامله‌ی بسته‌شده‌ای نیست | باز: {len(opened)}")
    msg = ("📊 خلاصه‌ی روزانه‌ی ربات‌ها\n━━━━━━━━━━━━━━━━━\n" + "\n".join(lines) +
           f"\n━━━━━━━━━━━━━━━━━\n📅 تاریخ شمسی: {date_jalali}\n📅 تاریخ میلادی: {date_gregorian}\n⏰ ساعت: {time_str}\n━━━━━━━━━━━━━━━━━")
    if send_telegram(msg): save_json(SUMMARY_FILE, {"date": today})

# =====================================================================
# چرخه ربات (با مدیریت سرمایه)
# =====================================================================
def run_bot(bot, ctx, btc_4h, btc_1h):
    _CUR["bot"] = bot.name
    print("=" * 50)
    print(f"▶ {bot.name} {('(' + bot.label + ')') if bot.label else ''}")
    signals = load_json(bot.signals_file, {})
    # ابتدا وضعیت معاملات باز بررسی شود (قبل از چک تک‌معامله)
    if bot.has_sizing:
        if update_open_trades_with_sizing(bot, ctx.exchange, signals):
            save_json(bot.signals_file, signals)
    else:
        update_open_trades(bot, ctx.exchange)
        signals = load_json(bot.signals_file, {})
    if bot.single_trade:
        open_trades = [s for k, s in signals.items() if not k.startswith("__") and s.get("status") == "OPEN"]
        if open_trades:
            print(f"⏳ {bot.name} has an OPEN trade. Skipping this cycle.")
            return 0
    candidates = []
    t_now = now_ms()
    for symbol in SYMBOLS:
        try:
            prev = signals.get(symbol)
            if prev and prev.get("status") == "OPEN": funnel_count("~skip_open_trade"); continue
            if (prev and bot.cooldown_ms and prev.get("closed_ms") and t_now - prev["closed_ms"] < bot.cooldown_ms):
                funnel_count("~skip_cooldown"); continue
            funnel_count("0_analyzed")
            c = bot.analyze(symbol, ctx)
            if c is None: continue
            funnel_count("PASSED_analyze")
            prev_key = (prev.get("signal_key", prev.get("cross_ts")) if prev else None)
            if prev and prev_key == c["signal_key"] and prev.get("direction") == c["direction"]:
                funnel_count("~already_signaled"); continue
            if symbol != BTC_SYMBOL:
                if bot.btc_filter_4h and btc_4h in ("LONG","SHORT") and c["direction"] != btc_4h:
                    funnel_count("~btc_4h_filter"); continue
                if bot.btc_filter_1h and btc_1h in ("LONG","SHORT") and c["direction"] != btc_1h:
                    funnel_count("~btc_1h_filter"); continue
            candidates.append(c)
        except Exception as e:
            funnel_count("EXCEPTION")
            print(f"⚠️ خطا در {symbol}: {e}")
            continue
    candidates.sort(key=lambda x: x["score"], reverse=True)
    sent = 0
    open_dir = Counter(s.get("direction") for k, s in signals.items() if not k.startswith("__") and s.get("status") == "OPEN")
    for c in candidates:
        if sent >= bot.max_signals: break
        symbol = c["symbol"]
        try:
            if open_dir[c["direction"]] >= MAX_OPEN_SAME_DIR: funnel_count("~cap_same_direction"); continue
            entry = get_price(ctx, symbol)
            if abs(entry - c["close"]) > c["atr"] * MAX_ENTRY_DRIFT_ATR:
                funnel_count("~send_price_drift"); continue
            lv = c["levels"](entry)
            if lv is None: funnel_count("~send_levels_none"); continue
            sl, tp = lv
            if sl <= 0 or tp <= 0: funnel_count("~send_invalid_sl_tp"); continue
            rr = abs(tp - entry) / abs(entry - sl)
            sizing = None
            if bot.has_sizing:
                balance = signals.get("__balance__", bot.initial_balance)
                in_recovery = signals.get("__in_recovery__", False)
                recovery_losses = signals.get("__recovery_losses__", [])
                sizing = calculate_position_sizing(bot, entry, sl, tp, balance, in_recovery, recovery_losses)
            msg_id = send_telegram(build_signal_message(bot, c, entry, sl, tp, rr, sizing))
            opened_ts = ctx.ref_time_ms if ctx.ref_time_ms is not None else now_ms()
            record = {
                "direction": c["direction"], "entry": entry, "tp": tp, "sl": sl,
                "signal_key": c["signal_key"], "setup": c["setup"],
                "score": round(c["score"], 4), "rsi": round(c["rsi"], 2),
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
            if sizing:
                record.update({"position_size": sizing["position_size"], "position_value": sizing["position_value"],
                               "leverage": sizing["leverage"], "risk_amount": sizing["risk_amount"]})
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
# اجرای اصلی
# =====================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Starting multi-bot cycle (Quality Mode): " + ", ".join(b.name for b in LIVE_BOTS))
    exchange = create_exchange()
    ctx = Ctx(exchange)
    btc_4h, btc_1h = get_btc_trend(ctx)
    try: btc_1d = trend_of(get_data(ctx, BTC_SYMBOL, "1d"))
    except Exception: btc_1d = "UNKNOWN"
    print(f"🌍 BTC Trend: 1D={btc_1d} | 4H={btc_4h} | 1H={btc_1h}")
    warning = "دریافت داده‌ی BTC ناموفق بود" if btc_4h == "UNKNOWN" else None
    ctx.btc_filter = get_btc_filter(ctx)
    if not ctx.btc_filter: print("🌍 BTC filter blocked — v60 will skip all symbols this cycle.")
    results = {}
    for bot in LIVE_BOTS:
        try: results[bot.name] = run_bot(bot, ctx, btc_4h, btc_1h)
        except Exception as e:
            results[bot.name] = "ERROR"
            print(f"⚠️ خطای کلی در {bot.name}: {e}")
    try: send_heartbeat_all(results, btc_4h, btc_1h, warning)
    except Exception as e: print(f"⚠️ خطا در heartbeat: {e}")
    try: maybe_send_daily_summary()
    except Exception as e: print(f"⚠️ خطا در خلاصه‌ی روزانه: {e}")
    print("All bots complete.")
    print("=" * 50)
