"""
نسخه تست — فقط ربات‌های سری a (v0a, v60a, v70a)
- هر ربات فقط یک معامله باز (single_trade)
- مدیریت سرمایه با ریسک ۱٪ از بالانس
- بدون تلگرام، بدون CSV، فقط print در کنسول
- آماده برای بک‌تست روی داده OKX یا اجرای زنده
"""
import os
import json
import time
import numpy as np
import pandas as pd
import ccxt
from collections import Counter, defaultdict
from datetime import datetime, timezone

# =====================================================================
# تنظیمات
# =====================================================================
BACKTEST_DIR = "Backtest"
OKX_DATA_DIR = os.path.join(BACKTEST_DIR, "okx_data")  # فرمت رو با فایل‌هات هماهنگ کن
TRADES_CSV = os.path.join(BACKTEST_DIR, "trades.csv")

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
MAX_OPEN_SAME_DIR = 3
MIN_SL_PCT = 0.0035
MAX_ENTRY_DRIFT_ATR = 0.45

# مدیریت سرمایه — تست اول: ریسک ۱٪
INITIAL_BALANCE = 100.0
RISK_PERCENT = 1.0
MAX_LEVERAGE = 10.0
MAX_RISK_PER_TRADE_PCT = 3.0
RECOVERY_ENABLED = True
MAX_RECOVERY_LOSSES = 4

TF_MS = {
    "5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

# پارامترهای استراتژی‌ها — دقیقاً مثل فایل تستی که فرستادی
B1_CROSS_LOOKBACK = 6
B1_RSI_LONG = (45, 68); B1_RSI_SHORT = (32, 55)
B1_VOLUME_MULT = 1.5
B1_SL_ATR = 1.6; B1_TP_ATR = 3.4
B1_MIN_SCORE = 0.70
B1_MIN_ADX_4H = 18; B1_MIN_ADX_1H = 16
B1_MIN_SLOPE_ATR = 0.08

B6_MIN_GAP_ATR = 0.28; B6_MIN_ADX = 20; B6_MAX_EXT_ATR = 6.0
B6_1H_ALIGN_BONUS = 0.12
B6_ALT_GAP_ATR = 0.22; B6_TOUCH_ATR = 0.22; B6_MAX_EXT_ALT_ATR = 1.0
B6_RSI_LONG = (40, 60); B6_RSI_SHORT = (40, 60)
B6_VOLUME_MULT = 1.4
B6_SL_ATR, B6_TP_ATR = 1.4, 2.95
B6_MIN_SCORE = 0.65
B6_COOLDOWN_HOURS = 3

B8_SWEEP_LOOKBACK = 10
B8_TP_ATR_MULT = 2.0
B8_COOLDOWN_HOURS = 4
B8_SL_BUFFER_ATR = 0.2
B8_MIN_SL_ATR = 0.6; B8_MAX_SL_ATR = 3.0
B8_MAX_SL_PCT = 0.07
B8_MIN_RR = 1.2
B8_MIN_VOL_RATIO = 0.9
B8_RSI_MAX_LONG = 72; B8_RSI_MIN_SHORT = 28
B8_MIN_SCORE = 0.35

# =====================================================================
# Funnel (فقط print)
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
    if not c: return
    print(f"📉 {bot_name} funnel:")
    for k, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
        print(f"   {k}: {n}")

# =====================================================================
# ابزارها
# =====================================================================
def load_json(path, default):
    try:
        with open(path, "r") as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f: json.dump(data, f)

def now_ms(): return int(time.time() * 1000)

# =====================================================================
# Ctx — پشتیبانی از backtest (ref_time_ms) و live
# =====================================================================
class Ctx:
    def __init__(self, exchange=None, ref_time_ms=None, data_loader=None):
        self.exchange = exchange
        self.cache = {}
        self.tickers = {}
        self.btc_filter = None
        self.ref_time_ms = ref_time_ms
        self.data_loader = data_loader  # برای بک‌تست: تابع (symbol, tf) -> DataFrame

def create_exchange():
    return ccxt.okx({"enableRateLimit": True})

def fetch_ohlcv_live(exchange, symbol, timeframe, limit=150):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    if len(df) and df["timestamp"].iloc[-1] + TF_MS[timeframe] > now_ms():
        df = df.iloc[:-1]
    return df.reset_index(drop=True)

def calculate_indicators(df):
    df = df.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss==0)&(avg_gain>0)), 100.0)
    rsi = rsi.where(~((avg_loss==0)&(avg_gain==0)), 50.0)
    df["rsi"] = rsi
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    up = df["high"].diff(); down = -df["low"].diff()
    plus_dm = up.where((up>down)&(up>0), 0.0)
    minus_dm = down.where((down>up)&(down>0), 0.0)
    atr_safe = df["atr"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_safe
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["adx"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    bb_mid = df["close"].rolling(20).mean()
    bb_sd = df["close"].rolling(20).std()
    df["bb_mid"] = bb_mid; df["bb_up"] = bb_mid+2*bb_sd; df["bb_lo"] = bb_mid-2*bb_sd
    return df

def get_data(ctx, symbol, timeframe):
    key = (symbol, timeframe)
    if key in ctx.cache: return ctx.cache[key]
    if ctx.data_loader is not None:
        df = ctx.data_loader(symbol, timeframe)
        if df is None or len(df) < 40:
            raise ValueError(f"داده کافی نیست: {symbol} {timeframe}")
        if ctx.ref_time_ms is not None:
            df = df[df["timestamp"] <= ctx.ref_time_ms].reset_index(drop=True)
    else:
        limit = 300 if timeframe == "1d" else 150
        df = fetch_ohlcv_live(ctx.exchange, symbol, timeframe, limit)
        if len(df) < 40:
            raise ValueError(f"داده کافی نیست: {symbol} {timeframe}")
    ctx.cache[key] = calculate_indicators(df)
    return ctx.cache[key]

def get_price(ctx, symbol):
    if ctx.data_loader is not None:
        df = get_data(ctx, symbol, "1h")
        return float(df["close"].iloc[-1])
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
    except Exception:
        return "UNKNOWN", "UNKNOWN"

def atr_levels(direction, atr, sl_mult, tp_mult):
    def levels(entry):
        if direction == "LONG": return entry - atr*sl_mult, entry + atr*tp_mult
        return entry + atr*sl_mult, entry - atr*tp_mult
    return levels

def ratio(cur, avg): return cur/avg if avg and avg > 0 else 0.0

# =====================================================================
# استراتژی‌ها (همان کد تستی — بدون تغییر)
# =====================================================================
def find_recent_cross(df, direction, lookback=B1_CROSS_LOOKBACK):
    if trend_of(df) != direction: return None, None
    n = len(df)
    for age in range(lookback):
        i = n - 1 - age
        if i < 1: break
        cur, prev = df.iloc[i], df.iloc[i-1]
        if direction=="LONG" and prev["ema9"]<=prev["ema21"] and cur["ema9"]>cur["ema21"]:
            return age, int(df["timestamp"].iloc[i])
        if direction=="SHORT" and prev["ema9"]>=prev["ema21"] and cur["ema9"]<cur["ema21"]:
            return age, int(df["timestamp"].iloc[i])
    return None, None

def analyze_cross(symbol, ctx):
    df_4h = get_data(ctx, symbol, "4h")
    direction = cross_age = cross_ts = None
    for d in ("LONG","SHORT"):
        age, ts = find_recent_cross(df_4h, d)
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
    if gap_4h < 0.35*atr_4h: return rej("05_gap_4h")
    ema9_slope = ema9_4h - float(df_4h["ema9"].iloc[-4])
    if abs(ema9_slope)/atr_4h < B1_MIN_SLOPE_ATR: return rej("06_slope_weak")
    if direction=="LONG" and ema9_slope<=0: return rej("07_slope_direction")
    if direction=="SHORT" and ema9_slope>=0: return rej("07_slope_direction")
    if direction=="LONG" and ema21_4h<=ema50_4h: return rej("08_ema21_vs_ema50")
    if direction=="SHORT" and ema21_4h>=ema50_4h: return rej("08_ema21_vs_ema50")
    df_1h = get_data(ctx, symbol, "1h")
    if trend_of(df_1h) != direction: return rej("09_trend_1h_mismatch")
    last_1h = df_1h.iloc[-1]
    atr_1h = float(last_1h["atr"])
    if np.isnan(atr_1h) or atr_1h <= 0: return rej("10_bad_atr_1h")
    adx_1h = float(last_1h["adx"])
    if np.isnan(adx_1h): return rej("11_bad_adx_1h")
    if adx_1h < B1_MIN_ADX_1H: return rej("12_adx_1h")
    gap_1h = abs(float(last_1h["ema9"]) - float(last_1h["ema21"]))
    if gap_1h < 0.40*atr_1h: return rej("13_gap_1h")
    vol_1h = float(last_1h["volume"])
    avg_vol_1h = float(df_1h["volume"].iloc[-21:-1].mean())
    vol_ratio_1h = ratio(vol_1h, avg_vol_1h)
    if vol_ratio_1h < B1_VOLUME_MULT: return rej("14_volume_1h")
    df_15m = get_data(ctx, symbol, "15m")
    last = df_15m.iloc[-1]
    rsi = float(last["rsi"])
    if np.isnan(rsi): return rej("15_bad_rsi")
    lo, hi = (B1_RSI_LONG if direction=="LONG" else B1_RSI_SHORT)
    if not (lo <= rsi <= hi): return rej("16_rsi_15m")
    close = float(last["close"])
    if (atr_1h*B1_SL_ATR)/close < MIN_SL_PCT: return rej("17_min_sl_pct")
    trend1_score = min(gap_1h/atr_1h, 1.5)/1.5
    trend4_score = min(gap_4h/atr_4h, 1.5)/1.5
    vol_score = min(vol_ratio_1h, 3.0)/3.0
    cross_score = (B1_CROSS_LOOKBACK-cross_age)/B1_CROSS_LOOKBACK
    rsi_score = min(abs(rsi-50)/20.0, 1.0)
    adx_score = min(adx_4h/35.0, 1.0)
    score = 0.25*trend1_score + 0.25*trend4_score + 0.20*adx_score + 0.15*vol_score + 0.10*cross_score + 0.05*rsi_score
    if score < B1_MIN_SCORE: return rej("18_score")
    return {"symbol": symbol, "direction": direction, "close": close, "atr": atr_1h,
            "rsi": rsi, "volume": vol_1h, "avg_volume": avg_vol_1h,
            "signal_key": cross_ts, "score": score,
            "setup": f"EMA Cross PRO (4H age {cross_age})",
            "levels": atr_levels(direction, atr_1h, B1_SL_ATR, B1_TP_ATR)}

def get_btc_filter(ctx):
    try:
        df_1d = get_data(ctx, BTC_SYMBOL, "1d")
        df_4h = get_data(ctx, BTC_SYMBOL, "4h")
        df_1h = get_data(ctx, BTC_SYMBOL, "1h")
    except Exception:
        return None
    t_1d, t_4h, t_1h = trend_of(df_1d), trend_of(df_4h), trend_of(df_1h)
    if t_1d not in ("LONG","SHORT") or t_4h not in ("LONG","SHORT"): return None
    if t_1d != t_4h: return None
    bias = t_1d
    last = df_4h.iloc[-1]
    atr, close = float(last["atr"]), float(last["close"])
    ema9, ema21 = float(last["ema9"]), float(last["ema21"])
    adx = float(last["adx"]) if not np.isnan(last["adx"]) else 0.0
    if np.isnan(atr) or atr <= 0: return None
    gap_atr = abs(ema9-ema21)/atr
    if not ((gap_atr >= B6_MIN_GAP_ATR) or (adx >= B6_MIN_ADX)): return None
    ext_atr = abs(close-ema21)/atr
    if ext_atr > B6_MAX_EXT_ATR: return None
    return {"bias":bias,"t_1d":t_1d,"t_4h":t_4h,"t_1h":t_1h,
            "align_1h": t_1h==bias,"gap_atr":gap_atr,"adx":adx,"ext_atr":ext_atr}

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
    gap_4h = abs(float(l4["ema9"])-float(l4["ema21"]))
    if gap_4h < B6_ALT_GAP_ATR*atr_4h: return rej("04_gap_4h")
    df_1h = get_data(ctx, symbol, "1h")
    last = df_1h.iloc[-1]
    close, open_, high, low = (float(last[k]) for k in ("close","open","high","low"))
    atr, rsi, ema21 = float(last["atr"]), float(last["rsi"]), float(last["ema21"])
    if np.isnan(atr) or np.isnan(rsi) or atr <= 0: return rej("05_bad_atr_rsi_1h")
    if bias=="LONG":
        touched = (low<=ema21+B6_TOUCH_ATR*atr and close>ema21 and close>open_ and close-ema21<=B6_MAX_EXT_ALT_ATR*atr)
        rsi_ok = B6_RSI_LONG[0]<=rsi<=B6_RSI_LONG[1]
    else:
        touched = (high>=ema21-B6_TOUCH_ATR*atr and close<ema21 and close<open_ and ema21-close<=B6_MAX_EXT_ALT_ATR*atr)
        rsi_ok = B6_RSI_SHORT[0]<=rsi<=B6_RSI_SHORT[1]
    volume, avg_volume = float(last["volume"]), float(df_1h["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(volume, avg_volume)
    vol_ok = vol_ratio >= B6_VOLUME_MULT
    if not (touched and rsi_ok and vol_ok):
        tags = []
        if not touched: tags.append("pullback")
        if not rsi_ok: tags.append("rsi")
        if not vol_ok: tags.append("vol")
        return rej("06_filters_"+ "+".join(tags))
    if (atr*B6_SL_ATR)/close < MIN_SL_PCT: return rej("07_min_sl_pct")
    trend_score = min(gap_4h/atr_4h, 1.2)/1.2
    vol_score = min(vol_ratio/2.5, 1.0)
    btc_score = 0.55 + (B6_1H_ALIGN_BONUS if btc and btc.get("align_1h") else 0.0)
    if btc: btc_score += 0.15*min(btc["gap_atr"]/0.6, 1.0)
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
    if low<prev_low and close>prev_low and close>open_:
        direction = "LONG"
        if not (ema21>ema50 and close>ema50): return rej("06_trend_filter")
        wick = min(open_, close)-low
        sl_dist = close - (low - atr*B8_SL_BUFFER_ATR)
    elif high>prev_high and close<prev_high and close<open_:
        direction = "SHORT"
        if not (ema21<ema50 and close<ema50): return rej("06_trend_filter")
        wick = high-max(open_, close)
        sl_dist = (high+atr*B8_SL_BUFFER_ATR)-close
    else: return rej("05_no_sweep")
    if sl_dist <= 0: return rej("07_sl_dist_invalid")
    sl_dist = max(sl_dist, atr*B8_MIN_SL_ATR)
    if sl_dist > atr*B8_MAX_SL_ATR or sl_dist/close > B8_MAX_SL_PCT: return rej("08_sl_too_far")
    if sl_dist/close < MIN_SL_PCT: return rej("09_min_sl_pct")
    tp_dist = atr*B8_TP_ATR_MULT
    rr = tp_dist/sl_dist
    if rr < B8_MIN_RR: return rej("10_rr_low")
    volume, avg_volume = float(curr["volume"]), float(df4["volume"].iloc[-21:-1].mean())
    vol_ratio = ratio(volume, avg_volume)
    if vol_ratio < B8_MIN_VOL_RATIO: return rej("11_volume")
    if direction=="LONG" and rsi>B8_RSI_MAX_LONG: return rej("12_rsi_extreme")
    if direction=="SHORT" and rsi<B8_RSI_MIN_SHORT: return rej("12_rsi_extreme")
    trend_score = min(abs(ema21-ema50)/atr, 1.0)
    vol_score = min(vol_ratio/2.0, 1.0)
    wick_score = min(wick/atr, 1.0)
    rr_score = min(rr/2.5, 1.0)
    score = 0.30*trend_score + 0.25*vol_score + 0.25*wick_score + 0.20*rr_score
    if score < B8_MIN_SCORE: return rej("13_score")
    def levels(entry):
        if direction=="LONG": return entry-sl_dist, entry+tp_dist
        return entry+sl_dist, entry-tp_dist
    return {"symbol": symbol, "direction": direction, "close": close, "atr": atr, "rsi": rsi,
            "volume": volume, "avg_volume": avg_volume, "signal_key": int(curr["timestamp"]),
            "score": score, "setup": f"SMC Winner ({direction} Sweep, RR {rr:.1f})",
            "levels": levels, "meta": {"risk_atr": round(sl_dist/atr, 3)}}

# =====================================================================
# مدیریت سرمایه
# =====================================================================
def calculate_position_sizing(bot, entry, sl, tp, balance, in_recovery, recovery_losses):
    sl_dist_pct = abs(entry-sl)/entry if entry > 0 else 0
    rr_ratio = abs(tp-entry)/abs(entry-sl) if abs(entry-sl) > 0 else 2.0
    if not in_recovery or not RECOVERY_ENABLED:
        risk_amount = balance * (bot.risk_percent/100.0)
    else:
        total_loss = sum(recovery_losses)
        risk_amount = total_loss/rr_ratio if rr_ratio > 0 else 0
    risk_amount = min(risk_amount, balance*(MAX_RISK_PER_TRADE_PCT/100.0))
    position_value = risk_amount/sl_dist_pct if sl_dist_pct > 0 else 0
    max_position_value = balance*MAX_LEVERAGE
    if position_value > max_position_value:
        position_value = max_position_value
        risk_amount = position_value*sl_dist_pct
    leverage_needed = position_value/balance if balance > 0 else 1.0
    position_size = position_value/entry if entry > 0 else 0
    return {"position_size": round(position_size, 6), "position_value": round(position_value, 2),
            "leverage": round(max(1.0, leverage_needed), 2), "risk_amount": round(risk_amount, 2)}

def check_trade_result_bt(ctx, symbol, data):
    """بررسی TP/SL روی کندل‌های 5m بعد از opened_ms، محدود به بازه بک‌تست."""
    direction, tp, sl = data["direction"], data["tp"], data["sl"]
    opened_ms = data["opened_ms"]
    try:
        df = ctx.data_loader(symbol, "5m")
        if df is None: return None, None
        df = df[df["timestamp"] >= opened_ms]
        if ctx.ref_time_ms is not None:
            df = df[df["timestamp"] <= ctx.ref_time_ms]
        for _, cd in df.iterrows():
            high, low, close = cd["high"], cd["low"], cd["close"]
            if direction=="LONG": hit_sl, hit_tp = low<=sl, high>=tp
            else: hit_sl, hit_tp = high>=sl, low<=tp
            if hit_sl and hit_tp:
                return ("TP", tp) if abs(close-tp) < abs(close-sl) else ("SL", sl)
            if hit_sl: return "SL", sl
            if hit_tp: return "TP", tp
    except Exception:
        pass
    return None, None

def check_trade_result_live(exchange, symbol, data):
    direction, tp, sl = data["direction"], data["tp"], data["sl"]
    opened_ms = data.get("opened_ms")
    if opened_ms:
        try: candles = exchange.fetch_ohlcv(symbol, timeframe="5m", since=int(opened_ms), limit=1000)
        except Exception: candles = []
        for cd in candles:
            if cd[0] < opened_ms: continue
            high, low, close = cd[2], cd[3], cd[4]
            if direction=="LONG": hit_sl, hit_tp = low<=sl, high>=tp
            else: hit_sl, hit_tp = high>=sl, low<=tp
            if hit_sl and hit_tp:
                return ("TP", tp) if abs(close-tp)<abs(close-sl) else ("SL", sl)
            if hit_sl: return "SL", sl
            if hit_tp: return "TP", tp
    price = float(exchange.fetch_ticker(symbol)["last"])
    if direction=="LONG":
        if price<=sl: return "SL", price
        if price>=tp: return "TP", price
    else:
        if price>=sl: return "SL", price
        if price<=tp: return "TP", price
    return None, None

# =====================================================================
# Bot
# =====================================================================
class Bot:
    def __init__(self, name, label, analyze, state_file, cooldown_hours=0,
                 btc_filter_4h=True, btc_filter_1h=False,
                 single_trade=True, has_sizing=True, risk_percent=RISK_PERCENT):
        self.name = name; self.label = label; self.analyze = analyze
        self.state_file = state_file
        self.cooldown_ms = int(cooldown_hours*3600*1000)
        self.btc_filter_4h = btc_filter_4h
        self.btc_filter_1h = btc_filter_1h
        self.single_trade = single_trade
        self.has_sizing = has_sizing
        self.initial_balance = INITIAL_BALANCE
        self.risk_percent = risk_percent
    @property
    def title_tag(self): return f"[{self.name}]" + (f" ({self.label})" if self.label else "")

BOTS = [
    Bot("crypto-signal-bot-v0a", "EMA Cross (Managed)", analyze_cross,
        os.path.join(BACKTEST_DIR, "state_v0a.json"), risk_percent=RISK_PERCENT),
    Bot("crypto-signal-bot-v60a", "BTC Filter (Managed)", analyze_btc_aligned,
        os.path.join(BACKTEST_DIR, "state_v60a.json"),
        cooldown_hours=B6_COOLDOWN_HOURS, btc_filter_4h=False, risk_percent=RISK_PERCENT),
    Bot("crypto-signal-bot-v70a", "SMC Winner (Managed)", analyze_smc_winner,
        os.path.join(BACKTEST_DIR, "state_v70a.json"),
        cooldown_hours=B8_COOLDOWN_HOURS, btc_filter_4h=True, risk_percent=RISK_PERCENT),
]

# =====================================================================
# به‌روزرسانی معاملات باز + مدیریت سرمایه
# =====================================================================
def update_open_trades(bot, signals, ctx, is_backtest):
    updated = False
    for symbol, data in list(signals.items()):
        if symbol.startswith("__"): continue
        if data.get("status") != "OPEN": continue
        try:
            if is_backtest:
                result, close_price = check_trade_result_bt(ctx, symbol, data)
            else:
                result, close_price = check_trade_result_live(ctx.exchange, symbol, data)
        except Exception as e:
            print(f"⚠️ خطا در بررسی {symbol}: {e}"); continue
        if not result: continue
        data["status"] = "CLOSED"
        data["result"] = result
        data["closed_ms"] = ctx.ref_time_ms if is_backtest else now_ms()
        data["close_price"] = float(close_price)
        entry_price = data["entry"]; pos_size = data.get("position_size", 0)
        if result == "TP":
            profit_usd = abs(data["tp"]-entry_price)*pos_size
            new_balance = signals.get("__balance__", bot.initial_balance) + profit_usd
            signals["__balance__"] = new_balance
            signals["__in_recovery__"] = False
            signals["__recovery_losses__"] = []
            data["cycle_status"] = "ENDED (Win)"
        else:
            loss_usd = abs(entry_price-data["sl"])*pos_size
            new_balance = signals.get("__balance__", bot.initial_balance) - loss_usd
            signals["__balance__"] = new_balance
            losses = signals.get("__recovery_losses__", [])
            losses.append(loss_usd)
            if not RECOVERY_ENABLED:
                signals["__recovery_losses__"] = []
                signals["__in_recovery__"] = False
                data["cycle_status"] = "LOSS (no recovery)"
            elif len(losses) >= MAX_RECOVERY_LOSSES:
                signals["__recovery_losses__"] = []
                signals["__in_recovery__"] = False
                data["cycle_status"] = f"ENDED (Recovery aborted after {len(losses)})"
            else:
                signals["__recovery_losses__"] = losses
                signals["__in_recovery__"] = True
                data["cycle_status"] = f"IN RECOVERY (Loss #{len(losses)})"
        data["profit_usd"] = round(profit_usd if result=="TP" else -loss_usd, 2)
        data["balance_after"] = round(new_balance, 2)
        updated = True
        print(f"{'🎯' if result=='TP' else '🛑'} {bot.name} | {symbol} {data['direction']} {result} | "
              f"balance={new_balance:.2f} | cycle={data['cycle_status']}")
        # ثبت در trades.csv
        append_trade_row(bot, symbol, data)
    return updated

def append_trade_row(bot, symbol, data):
    import csv
    fields = ["bot","symbol","direction","entry","sl","tp","exit","result",
              "risk_pct","rr","position_size","position_value","leverage",
              "profit_usd","balance_after","cycle_status","opened_ms","closed_ms","hold_hours"]
    file_exists = os.path.exists(TRADES_CSV)
    hold_hours = ""
    try:
        if data.get("opened_ms") and data.get("closed_ms"):
            hold_hours = round((data["closed_ms"]-data["opened_ms"])/3600000, 2)
    except Exception: pass
    row = {
        "bot": bot.name, "symbol": symbol, "direction": data.get("direction"),
        "entry": data.get("entry"), "sl": data.get("sl"), "tp": data.get("tp"),
        "exit": data.get("close_price"), "result": data.get("result"),
        "risk_pct": data.get("risk_pct"), "rr": data.get("rr"),
        "position_size": data.get("position_size"), "position_value": data.get("position_value"),
        "leverage": data.get("leverage"), "profit_usd": data.get("profit_usd"),
        "balance_after": data.get("balance_after"), "cycle_status": data.get("cycle_status"),
        "opened_ms": data.get("opened_ms"), "closed_ms": data.get("closed_ms"),
        "hold_hours": hold_hours,
    }
    os.makedirs(os.path.dirname(TRADES_CSV) or ".", exist_ok=True)
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists: w.writeheader()
        w.writerow(row)

# =====================================================================
# چرخه ربات
# =====================================================================
def run_bot(bot, ctx, btc_4h, btc_1h, is_backtest):
    _CUR["bot"] = bot.name
    signals = load_json(bot.state_file, {})
    if update_open_trades(bot, signals, ctx, is_backtest):
        save_json(bot.state_file, signals)
    # چک تک‌معامله
    open_trades = [s for k,s in signals.items() if not k.startswith("__") and s.get("status")=="OPEN"]
    if bot.single_trade and open_trades:
        return 0
    if signals.get("__balance__", bot.initial_balance) <= 0:
        return 0
    candidates = []
    t_now = ctx.ref_time_ms if is_backtest else now_ms()
    for symbol in SYMBOLS:
        try:
            prev = signals.get(symbol)
            if prev and prev.get("status") == "OPEN": funnel_count("~skip_open_trade"); continue
            if prev and bot.cooldown_ms and prev.get("closed_ms") and t_now - prev["closed_ms"] < bot.cooldown_ms:
                funnel_count("~skip_cooldown"); continue
            funnel_count("0_analyzed")
            c = bot.analyze(symbol, ctx)
            if c is None: continue
            funnel_count("PASSED_analyze")
            prev_key = prev.get("signal_key") if prev else None
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
            continue
    candidates.sort(key=lambda x: x["score"], reverse=True)
    sent = 0
    for c in candidates:
        if sent >= 1: break  # single_trade: حداکثر ۱ سیگنال در هر چرخه
        symbol = c["symbol"]
        try:
            entry = get_price(ctx, symbol)
            if abs(entry - c["close"]) > c["atr"]*MAX_ENTRY_DRIFT_ATR:
                funnel_count("~send_price_drift"); continue
            sl, tp = c["levels"](entry)
            if sl <= 0 or tp <= 0: funnel_count("~send_invalid_sl_tp"); continue
            rr = abs(tp-entry)/abs(entry-sl)
            balance = signals.get("__balance__", bot.initial_balance)
            in_recovery = signals.get("__in_recovery__", False)
            recovery_losses = signals.get("__recovery_losses__", [])
            sizing = calculate_position_sizing(bot, entry, sl, tp, balance, in_recovery, recovery_losses)
            record = {
                "direction": c["direction"], "entry": entry, "tp": tp, "sl": sl,
                "signal_key": c["signal_key"], "setup": c["setup"],
                "score": round(c["score"],4), "rsi": round(c["rsi"],2),
                "vol_ratio": round(ratio(c["volume"], c["avg_volume"]),2),
                "atr_pct": round(c["atr"]/entry*100,3),
                "risk_pct": round(abs(entry-sl)/entry*100,3),
                "rr": round(rr,2),
                "status": "OPEN",
                "opened_ms": ctx.ref_time_ms if is_backtest else now_ms(),
                **c.get("meta", {}),
                **sizing,
            }
            signals[symbol] = record
            save_json(bot.state_file, signals)
            sent += 1
            funnel_count("SENT_signal")
            print(f"📤 {bot.name} | {symbol} {c['direction']} @ {entry} | SL={sl} TP={tp} | "
                  f"size={sizing['position_size']} lev={sizing['leverage']}x risk=${sizing['risk_amount']}")
        except Exception as e:
            funnel_count("EXCEPTION_send")
            continue
    print_funnel(bot.name)
    return sent

# =====================================================================
# اجرای زنده
# =====================================================================
def run_live_cycle():
    exchange = create_exchange()
    ctx = Ctx(exchange=exchange)
    btc_4h, btc_1h = get_btc_trend(ctx)
    ctx.btc_filter = get_btc_filter(ctx)
    for bot in BOTS:
        try: run_bot(bot, ctx, btc_4h, btc_1h, is_backtest=False)
        except Exception as e: print(f"⚠️ خطای کلی {bot.name}: {e}")

# =====================================================================
# بک‌تست — قدم به قدم روی داده OKX
# =====================================================================
def load_okx_csv(symbol, timeframe):
    """فرمت: Backtest/okx_data/{symbol}_{timeframe}.csv با ستون‌های timestamp,open,high,low,close,volume"""
    path = os.path.join(OKX_DATA_DIR, f"{symbol}_{timeframe}.csv")
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    for col in ("timestamp","open","high","low","close","volume"):
        if col not in df.columns: return None
    df["timestamp"] = df["timestamp"].astype("int64")
    return df.sort_values("timestamp").reset_index(drop=True)

def run_backtest(start_ms, end_ms, step_ms=4*3600*1000):
    """
    روی محور زمان از start_ms تا end_ms با گام step_ms جلو می‌ره.
    در هر گام، ctx.ref_time_ms ست می‌شه و همه ربات‌ها یک چرخه اجرا می‌شن.
    """
    # پاک‌سازی state و trades قبلی
    for bot in BOTS:
        if os.path.exists(bot.state_file): os.remove(bot.state_file)
    if os.path.exists(TRADES_CSV): os.remove(TRADES_CSV)

    # پیش‌بارگذاری داده برای همه نمادها و تایم‌فریم‌های لازم
    tfs = ["5m","15m","1h","4h","1d"]
    data_cache = {}
    for sym in SYMBOLS + [BTC_SYMBOL]:
        for tf in tfs:
            df = load_okx_csv(sym, tf)
            if df is not None: data_cache[(sym,tf)] = df
    print(f"📦 داده بارگذاری شد: {len(data_cache)} (symbol,tf)")

    def loader(symbol, timeframe):
        return data_cache.get((symbol, timeframe))

    t = start_ms
    cycle = 0
    while t <= end_ms:
        cycle += 1
        ctx = Ctx(exchange=None, ref_time_ms=t, data_loader=loader)
        btc_4h, btc_1h = get_btc_trend(ctx)
        ctx.btc_filter = get_btc_filter(ctx)
        for bot in BOTS:
            try: run_bot(bot, ctx, btc_4h, btc_1h, is_backtest=True)
            except Exception as e: print(f"⚠️ {bot.name} @ {t}: {e}")
        if cycle % 10 == 0:
            print(f"⏱ cycle={cycle} t={datetime.fromtimestamp(t/1000, tz=timezone.utc)}")
        t += step_ms
    print(f"✅ بک‌تست تمام شد. trades در {TRADES_CSV}")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"
    if mode == "backtest":
        # بازه‌ی نمونه — بعداً با تاریخ داده‌ی خودت هماهنگ کن
        start = int(datetime(2024,1,1,tzinfo=timezone.utc).timestamp()*1000)
        end   = int(datetime(2025,1,1,tzinfo=timezone.utc).timestamp()*1000)
        run_backtest(start, end, step_ms=4*3600*1000)
    else:
        run_live_cycle()