#!/usr/bin/env python3
"""
بک‌تست ربات‌های سری a (v0a, v60a, v70a) روی داده‌ی OKX.

- فقط از داده‌ی 15m کش‌شده استفاده می‌کند و 1h/4h/1d را با resample دقیق می‌سازد.
- بدون ccxt، بدون تلگرام، بدون وابستگی به فایل ربات‌های زنده.
- ورودی‌ها: --days --end --step-minutes --set KEY=VAL --bots ...
- خروجی: trades.csv (همه‌ی معاملات) + report.txt (خلاصه با سرتیتر «نتایج بک‌تست»)
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# =====================================================================
# نمادها (فرمت Toobit — نگاشت به OKX با OKX_BASE_ALIASES)
# =====================================================================
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
    "ONE-SWAP-USDT", "QTUM-SWAP-USDT", "IOTA-SWAP-USDT", "NEO-SWAP-USDT",
]
BTC_SYMBOL = "BTC-SWAP-USDT"

OKX_BASE_ALIASES = {
    "1000PEPE": "PEPE", "1000SHIB": "SHIB", "1000BONK": "BONK",
    "1000FLOKI": "FLOKI", "1000SATS": "SATS", "1000RATS": "RATS",
    "1000000MOG": "MOG",
}

def to_okx_inst(symbol):
    base = symbol.split("-")[0]
    base = OKX_BASE_ALIASES.get(base, base)
    return f"{base}-USDT-SWAP"

# =====================================================================
# تنظیمات قابل override با --set
# =====================================================================
CONFIG = {
    # مدیریت سرمایه
    "INITIAL_BALANCE": 100.0,
    "RISK_PERCENT": 1.0,
    "MAX_LEVERAGE": 10.0,
    "MAX_RISK_PER_TRADE_PCT": 3.0,
    "RECOVERY_ENABLED": True,
    "MAX_RECOVERY_LOSSES": 4,
    # عمومی
    "MIN_SL_PCT": 0.0035,
    "MAX_ENTRY_DRIFT_ATR": 0.45,
    "MAX_OPEN_SAME_DIR": 3,
    # B1
    "B1_CROSS_LOOKBACK": 6, "B1_RSI_LONG": (45,68), "B1_RSI_SHORT": (32,55),
    "B1_VOLUME_MULT": 1.5, "B1_SL_ATR": 1.6, "B1_TP_ATR": 3.4,
    "B1_MIN_SCORE": 0.70, "B1_MIN_ADX_4H": 18, "B1_MIN_ADX_1H": 16,
    "B1_MIN_SLOPE_ATR": 0.08,
    # B6
    "B6_MIN_GAP_ATR": 0.28, "B6_MIN_ADX": 20, "B6_MAX_EXT_ATR": 6.0,
    "B6_1H_ALIGN_BONUS": 0.12, "B6_ALT_GAP_ATR": 0.22, "B6_TOUCH_ATR": 0.22,
    "B6_MAX_EXT_ALT_ATR": 1.0, "B6_RSI_LONG": (40,60), "B6_RSI_SHORT": (40,60),
    "B6_VOLUME_MULT": 1.4, "B6_SL_ATR": 1.4, "B6_TP_ATR": 2.95,
    "B6_MIN_SCORE": 0.65, "B6_COOLDOWN_HOURS": 3,
    # B8
    "B8_SWEEP_LOOKBACK": 10, "B8_TP_ATR_MULT": 2.0, "B8_COOLDOWN_HOURS": 4,
    "B8_SL_BUFFER_ATR": 0.2, "B8_MIN_SL_ATR": 0.6, "B8_MAX_SL_ATR": 3.0,
    "B8_MAX_SL_PCT": 0.07, "B8_MIN_RR": 1.2, "B8_MIN_VOL_RATIO": 0.9,
    "B8_RSI_MAX_LONG": 72, "B8_RSI_MIN_SHORT": 28, "B8_MIN_SCORE": 0.35,
}
def C(key): return CONFIG[key]

def parse_overrides(pairs):
    for p in pairs:
        if "=" not in p:
            print(f"⚠️ override نامعتبر: {p}"); continue
        k, v = p.split("=", 1)
        k = k.strip(); v = v.strip()
        if k not in CONFIG:
            print(f"⚠️ پارامتر ناشناخته: {k}"); continue
        old = CONFIG[k]
        try:
            if isinstance(old, bool): new = v.lower() in ("1","true","yes")
            elif isinstance(old, int): new = int(v)
            elif isinstance(old, float): new = float(v)
            elif isinstance(old, tuple): new = tuple(int(x) for x in v.strip("()").split(","))
            else: new = v
            CONFIG[k] = new
            print(f"🔧 override: {k} = {new} (قبلی: {old})")
        except Exception as e:
            print(f"⚠️ خطا در override {k}={v}: {e}")

# =====================================================================
# Funnel
# =====================================================================
FUNNEL = defaultdict(Counter)
_CUR = {"bot": None}
def rej(label="unlabeled"): FUNNEL[_CUR["bot"]][label] += 1; return None
def funnel_count(label): FUNNEL[_CUR["bot"]][label] += 1
def print_funnel(bot_name, top=10):
    c = FUNNEL.pop(bot_name, None)
    if not c: return
    print(f"📉 {bot_name} funnel:")
    for k, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
        print(f"   {k}: {n}")

# =====================================================================
# بارگذاری داده + resample
# =====================================================================
TF_MS = {"15m": 15*60*1000, "1h": 60*60*1000, "4h": 4*60*60*1000, "1d": 24*60*60*1000}
_TF_RULE = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}

_DATA_CACHE = {}   # (okx_inst, tf) -> df

def load_base_csv(data_dir, okx_inst):
    path = os.path.join(data_dir, "15m", f"{okx_inst}.csv.gz")
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            with gzip.open(path, "rt") as f: df = pd.read_csv(f)
        except Exception: return None
    for c in ("ts","open","high","low","close","volume"):
        if c not in df.columns: return None
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def resample_df(df15, rule):
    if df15 is None or len(df15) == 0: return None
    df = df15.copy()
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt")
    o = df["open"].resample(rule, label="left", closed="left").first()
    h = df["high"].resample(rule, label="left", closed="left").max()
    l = df["low"].resample(rule, label="left", closed="left").min()
    c = df["close"].resample(rule, label="left", closed="left").last()
    v = df["volume"].resample(rule, label="left", closed="left").sum()
    out = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna()
    out = out.reset_index()
    # تبدیل ۱۰۰٪ امن به epoch میلی‌ثانیه (مستقل از ns/us/ms بودن dtype)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    out["timestamp"] = ((out["dt"] - epoch) // pd.Timedelta(milliseconds=1)).astype("int64")
    return out[["timestamp","open","high","low","close","volume"]]

def get_df(data_dir, symbol, tf):
    key = (symbol, tf)
    if key in _DATA_CACHE: return _DATA_CACHE[key]
    okx = to_okx_inst(symbol)
    base = load_base_csv(data_dir, okx)
    if base is None:
        _DATA_CACHE[key] = None; return None
    if tf == "15m":
        df = base.rename(columns={"ts":"timestamp"})[["timestamp","open","high","low","close","volume"]]
    else:
        df = resample_df(base, _TF_RULE[tf])
    _DATA_CACHE[key] = df
    return df

# =====================================================================
# اندیکاتورها
# =====================================================================
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
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-df["close"].shift()).abs(),
                    (df["low"]-df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    up = df["high"].diff(); down = -df["low"].diff()
    plus_dm = up.where((up>down)&(up>0), 0.0)
    minus_dm = down.where((down>up)&(down>0), 0.0)
    atr_safe = df["atr"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_safe
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return df

def get_data(ctx, symbol, tf):
    key = (symbol, tf)
    if key in ctx.cache: return ctx.cache[key]
    df = ctx.data_loader(symbol, tf)
    if df is None or len(df) < 60:
        raise ValueError(f"داده کافی نیست: {symbol} {tf}")
    # فقط کندل‌های قبل یا مساوی ref_time
    df = df[df["timestamp"] <= ctx.ref_time_ms].reset_index(drop=True)
    if len(df) < 60: raise ValueError(f"داده کافی نیست بعد از فیلتر: {symbol} {tf}")
    ctx.cache[key] = calculate_indicators(df)
    return ctx.cache[key]

def trend_of(df):
    last = df.iloc[-1]
    if last["ema9"] > last["ema21"]: return "LONG"
    if last["ema9"] < last["ema21"]: return "SHORT"
    return "NONE"

def ratio(cur, avg): return cur/avg if avg and avg>0 else 0.0

def atr_levels(direction, atr, sl_mult, tp_mult):
    def levels(entry):
        if direction == "LONG": return entry - atr*sl_mult, entry + atr*tp_mult
        return entry + atr*sl_mult, entry - atr*tp_mult
    return levels

# =====================================================================
# استراتژی‌ها (همان منطق نسخه زنده، فقط IND از CONFIG)
# =====================================================================
def find_recent_cross(df, direction, lookback):
    if trend_of(df) != direction: return None, None
    n = len(df)
    for age in range(lookback):
        i = n-1-age
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
        age, ts = find_recent_cross(df_4h, d, C("B1_CROSS_LOOKBACK"))
        if age is not None: direction, cross_age, cross_ts = d, age, ts; break
    if direction is None: return rej("01_no_cross")
    last4 = df_4h.iloc[-1]
    atr_4h = float(last4["atr"])
    if np.isnan(atr_4h) or atr_4h<=0: return rej("02_atr4h")
    adx_4h = float(last4["adx"])
    if np.isnan(adx_4h) or adx_4h < C("B1_MIN_ADX_4H"): return rej("03_adx4h")
    ema9, ema21, ema50 = float(last4["ema9"]), float(last4["ema21"]), float(last4["ema50"])
    gap_4h = abs(ema9-ema21)
    if gap_4h < 0.35*atr_4h: return rej("04_gap4h")
    ema9_slope = ema9 - float(df_4h["ema9"].iloc[-4])
    if abs(ema9_slope)/atr_4h < C("B1_MIN_SLOPE_ATR"): return rej("05_slope")
    if (direction=="LONG" and ema9_slope<=0) or (direction=="SHORT" and ema9_slope>=0): return rej("06_slope_dir")
    if (direction=="LONG" and ema21<=ema50) or (direction=="SHORT" and ema21>=ema50): return rej("07_ema21_50")
    df_1h = get_data(ctx, symbol, "1h")
    if trend_of(df_1h) != direction: return rej("08_trend1h")
    last1 = df_1h.iloc[-1]
    atr_1h = float(last1["atr"])
    if np.isnan(atr_1h) or atr_1h<=0: return rej("09_atr1h")
    adx_1h = float(last1["adx"])
    if np.isnan(adx_1h) or adx_1h < C("B1_MIN_ADX_1H"): return rej("10_adx1h")
    gap_1h = abs(float(last1["ema9"])-float(last1["ema21"]))
    if gap_1h < 0.40*atr_1h: return rej("11_gap1h")
    vol = float(last1["volume"]); avg_vol = float(df_1h["volume"].iloc[-21:-1].mean())
    vr = ratio(vol, avg_vol)
    if vr < C("B1_VOLUME_MULT"): return rej("12_vol")
    df_15 = get_data(ctx, symbol, "15m")
    last15 = df_15.iloc[-1]
    rsi = float(last15["rsi"])
    if np.isnan(rsi): return rej("13_rsi")
    lo, hi = C("B1_RSI_LONG") if direction=="LONG" else C("B1_RSI_SHORT")
    if not (lo<=rsi<=hi): return rej("14_rsi_range")
    close = float(last15["close"])
    if (atr_1h*C("B1_SL_ATR"))/close < C("MIN_SL_PCT"): return rej("15_min_sl")
    t1 = min(gap_1h/atr_1h,1.5)/1.5; t4 = min(gap_4h/atr_4h,1.5)/1.5
    vs = min(vr,3.0)/3.0; cs = (C("B1_CROSS_LOOKBACK")-cross_age)/C("B1_CROSS_LOOKBACK")
    rs = min(abs(rsi-50)/20.0,1.0); ads = min(adx_4h/35.0,1.0)
    score = 0.25*t1+0.25*t4+0.20*ads+0.15*vs+0.10*cs+0.05*rs
    if score < C("B1_MIN_SCORE"): return rej("16_score")
    return {"symbol":symbol,"direction":direction,"close":close,"atr":atr_1h,"rsi":rsi,
            "volume":vol,"avg_volume":avg_vol,"signal_key":cross_ts,"score":score,
            "setup":f"EMA Cross PRO (4H age {cross_age})",
            "levels":atr_levels(direction, atr_1h, C("B1_SL_ATR"), C("B1_TP_ATR"))}

def get_btc_filter(ctx):
    try:
        d1, d4, h1 = get_data(ctx,BTC_SYMBOL,"1d"), get_data(ctx,BTC_SYMBOL,"4h"), get_data(ctx,BTC_SYMBOL,"1h")
    except Exception: return None
    t1d, t4h, t1h = trend_of(d1), trend_of(d4), trend_of(h1)
    if t1d not in ("LONG","SHORT") or t4h not in ("LONG","SHORT") or t1d!=t4h: return None
    bias = t1d; last = d4.iloc[-1]
    atr, close = float(last["atr"]), float(last["close"])
    ema9, ema21 = float(last["ema9"]), float(last["ema21"])
    adx = float(last["adx"]) if not np.isnan(last["adx"]) else 0.0
    if np.isnan(atr) or atr<=0: return None
    gap_atr = abs(ema9-ema21)/atr
    if not (gap_atr>=C("B6_MIN_GAP_ATR") or adx>=C("B6_MIN_ADX")): return None
    ext = abs(close-ema21)/atr
    if ext > C("B6_MAX_EXT_ATR"): return None
    return {"bias":bias,"t_1d":t1d,"t_4h":t4h,"t_1h":t1h,
            "align_1h":t1h==bias,"gap_atr":gap_atr,"adx":adx,"ext_atr":ext}

def analyze_btc_aligned(symbol, ctx):
    btc = getattr(ctx, "btc_filter", None)
    if not btc:
        try:
            df4 = get_data(ctx, symbol, "4h"); bias = trend_of(df4)
            if bias not in ("LONG","SHORT"): return rej("01_no_bias")
        except Exception: return rej("01_btc_blocked")
    else: bias = btc["bias"]
    df4 = get_data(ctx, symbol, "4h")
    if trend_of(df4) != bias: return rej("02_dir4h")
    l4 = df4.iloc[-1]; atr4 = float(l4["atr"])
    if np.isnan(atr4) or atr4<=0: return rej("03_atr4h")
    gap4 = abs(float(l4["ema9"])-float(l4["ema21"]))
    if gap4 < C("B6_ALT_GAP_ATR")*atr4: return rej("04_gap4h")
    df1 = get_data(ctx, symbol, "1h"); l1 = df1.iloc[-1]
    close, open_, high, low = (float(l1[k]) for k in ("close","open","high","low"))
    atr, rsi, ema21 = float(l1["atr"]), float(l1["rsi"]), float(l1["ema21"])
    if np.isnan(atr) or np.isnan(rsi) or atr<=0: return rej("05_atr_rsi")
    if bias=="LONG":
        touched = (low<=ema21+C("B6_TOUCH_ATR")*atr and close>ema21 and close>open_ and close-ema21<=C("B6_MAX_EXT_ALT_ATR")*atr)
        rsi_ok = C("B6_RSI_LONG")[0]<=rsi<=C("B6_RSI_LONG")[1]
    else:
        touched = (high>=ema21-C("B6_TOUCH_ATR")*atr and close<ema21 and close<open_ and ema21-close<=C("B6_MAX_EXT_ALT_ATR")*atr)
        rsi_ok = C("B6_RSI_SHORT")[0]<=rsi<=C("B6_RSI_SHORT")[1]
    vol, avg_vol = float(l1["volume"]), float(df1["volume"].iloc[-21:-1].mean())
    vr = ratio(vol, avg_vol)
    if not (touched and rsi_ok and vr>=C("B6_VOLUME_MULT")): return rej("06_filters")
    if (atr*C("B6_SL_ATR"))/close < C("MIN_SL_PCT"): return rej("07_min_sl")
    ts = min(gap4/atr4,1.2)/1.2; vs = min(vr/2.5,1.0)
    btc_s = 0.55 + (C("B6_1H_ALIGN_BONUS") if btc and btc.get("align_1h") else 0)
    if btc: btc_s += 0.15*min(btc["gap_atr"]/0.6,1.0)
    score = 0.35*btc_s+0.30*ts+0.20*vs+0.15*min(abs(rsi-50)/20.0,1.0)
    if score < C("B6_MIN_SCORE"): return rej("08_score")
    return {"symbol":symbol,"direction":bias,"close":close,"atr":atr,"rsi":rsi,
            "volume":vol,"avg_volume":avg_vol,"signal_key":int(l1["timestamp"]),"score":score,
            "setup":f"BTC-aligned pullback 1H (BTC {bias})",
            "levels":atr_levels(bias, atr, C("B6_SL_ATR"), C("B6_TP_ATR"))}

def analyze_smc_winner(symbol, ctx):
    try: df4 = get_data(ctx, symbol, "4h")
    except Exception: return rej("01_no4h")
    if len(df4) < 60: return rej("02_short4h")
    cur = df4.iloc[-1]; atr = float(cur["atr"])
    if np.isnan(atr) or atr<=0: return rej("03_atr")
    close, open_, low, high = float(cur["close"]), float(cur["open"]), float(cur["low"]), float(cur["high"])
    if close<=0: return rej("04_price")
    rsi = float(cur["rsi"]) if not np.isnan(cur["rsi"]) else 50.0
    ema21, ema50 = float(cur["ema21"]), float(cur["ema50"])
    prev_low = float(df4["low"].iloc[-C("B8_SWEEP_LOOKBACK")-1:-1].min())
    prev_high = float(df4["high"].iloc[-C("B8_SWEEP_LOOKBACK")-1:-1].max())
    if low<prev_low and close>prev_low and close>open_:
        direction="LONG"
        if not (ema21>ema50 and close>ema50): return rej("06_trend")
        wick = min(open_,close)-low; sl_dist = close-(low-atr*C("B8_SL_BUFFER_ATR"))
    elif high>prev_high and close<prev_high and close<open_:
        direction="SHORT"
        if not (ema21<ema50 and close<ema50): return rej("06_trend")
        wick = high-max(open_,close); sl_dist = (high+atr*C("B8_SL_BUFFER_ATR"))-close
    else: return rej("05_no_sweep")
    if sl_dist<=0: return rej("07_sl")
    sl_dist = max(sl_dist, atr*C("B8_MIN_SL_ATR"))
    if sl_dist > atr*C("B8_MAX_SL_ATR") or sl_dist/close > C("B8_MAX_SL_PCT"): return rej("08_sl_far")
    if sl_dist/close < C("MIN_SL_PCT"): return rej("09_min_sl")
    tp_dist = atr*C("B8_TP_ATR_MULT")
    rr = tp_dist/sl_dist
    if rr < C("B8_MIN_RR"): return rej("10_rr")
    vol, avg_vol = float(cur["volume"]), float(df4["volume"].iloc[-21:-1].mean())
    vr = ratio(vol, avg_vol)
    if vr < C("B8_MIN_VOL_RATIO"): return rej("11_vol")
    if direction=="LONG" and rsi>C("B8_RSI_MAX_LONG"): return rej("12_rsi")
    if direction=="SHORT" and rsi<C("B8_RSI_MIN_SHORT"): return rej("12_rsi")
    ts = min(abs(ema21-ema50)/atr,1.0); vs = min(vr/2.0,1.0)
    ws = min(wick/atr,1.0); rs = min(rr/2.5,1.0)
    score = 0.30*ts+0.25*vs+0.25*ws+0.20*rs
    if score < C("B8_MIN_SCORE"): return rej("13_score")
    def levels(entry):
        if direction=="LONG": return entry-sl_dist, entry+tp_dist
        return entry+sl_dist, entry-tp_dist
    return {"symbol":symbol,"direction":direction,"close":close,"atr":atr,"rsi":rsi,
            "volume":vol,"avg_volume":avg_vol,"signal_key":int(cur["timestamp"]),"score":score,
            "setup":f"SMC Winner ({direction} Sweep, RR {rr:.1f})","levels":levels}

# =====================================================================
# مدیریت سرمایه
# =====================================================================
def calc_sizing(risk_percent, entry, sl, tp, balance, in_recovery, losses):
    sl_pct = abs(entry-sl)/entry if entry>0 else 0
    rr = abs(tp-entry)/abs(entry-sl) if abs(entry-sl)>0 else 2.0
    if not in_recovery or not C("RECOVERY_ENABLED"):
        risk_amt = balance*(risk_percent/100.0)
    else:
        risk_amt = sum(losses)/rr if rr>0 else 0
    risk_amt = min(risk_amt, balance*(C("MAX_RISK_PER_TRADE_PCT")/100.0))
    pos_val = risk_amt/sl_pct if sl_pct>0 else 0
    max_val = balance*C("MAX_LEVERAGE")
    if pos_val > max_val:
        pos_val = max_val; risk_amt = pos_val*sl_pct
    lev = pos_val/balance if balance>0 else 1.0
    size = pos_val/entry if entry>0 else 0
    return {"position_size":round(size,6),"position_value":round(pos_val,2),
            "leverage":round(max(1.0,lev),2),"risk_amount":round(risk_amt,2)}

# =====================================================================
# بررسی TP/SL روی کندل‌های 15m
# =====================================================================
def check_trade_result(ctx, symbol, data):
    direction, tp, sl = data["direction"], data["tp"], data["sl"]
    opened_ms = data["opened_ms"]
    df = ctx.data_loader(symbol, "15m")
    if df is None: return None, None
    df = df[(df["timestamp"] >= opened_ms) & (df["timestamp"] <= ctx.ref_time_ms)]
    for _, cd in df.iterrows():
        high, low, close = cd["high"], cd["low"], cd["close"]
        if direction=="LONG": hit_sl, hit_tp = low<=sl, high>=tp
        else: hit_sl, hit_tp = high>=sl, low<=tp
        if hit_sl and hit_tp:
            return ("TP", tp) if abs(close-tp)<abs(close-sl) else ("SL", sl)
        if hit_sl: return "SL", sl
        if hit_tp: return "TP", tp
    return None, None

# =====================================================================
# Bot + state (در حافظه — بک‌تست)
# =====================================================================
class Bot:
    def __init__(self, name, label, analyze, cooldown_hours=0,
                 btc_filter_4h=True, btc_filter_1h=False, risk_percent=None):
        self.name = name; self.label = label; self.analyze = analyze
        self.cooldown_ms = int(cooldown_hours*3600*1000)
        self.btc_filter_4h = btc_filter_4h
        self.btc_filter_1h = btc_filter_1h
        self.risk_percent = risk_percent if risk_percent is not None else C("RISK_PERCENT")
        self.signals = {}   # state در حافظه

ALL_BOTS = [
    Bot("crypto-signal-bot-v0a", "EMA Cross (Managed)", analyze_cross),
    Bot("crypto-signal-bot-v60a", "BTC Filter (Managed)", analyze_btc_aligned,
        cooldown_hours=C("B6_COOLDOWN_HOURS"), btc_filter_4h=False),
    Bot("crypto-signal-bot-v70a", "SMC Winner (Managed)", analyze_smc_winner,
        cooldown_hours=C("B8_COOLDOWN_HOURS"), btc_filter_4h=True),
]

# =====================================================================
# به‌روزرسانی معاملات باز
# =====================================================================
def update_open(bot, ctx, trades):
    for symbol, data in list(bot.signals.items()):
        if symbol.startswith("__") or data.get("status")!="OPEN": continue
        try: result, close_price = check_trade_result(ctx, symbol, data)
        except Exception: continue
        if not result: continue
        data["status"] = "CLOSED"
        data["result"] = result
        data["closed_ms"] = ctx.ref_time_ms
        data["close_price"] = float(close_price)
        entry = data["entry"]; pos = data.get("position_size", 0)
        if result == "TP":
            profit = abs(data["tp"]-entry)*pos
            new_bal = bot.signals.get("__balance__", C("INITIAL_BALANCE")) + profit
            bot.signals["__balance__"] = new_bal
            bot.signals["__in_recovery__"] = False
            bot.signals["__recovery_losses__"] = []
            data["cycle_status"] = "ENDED (Win)"
        else:
            loss = abs(entry-data["sl"])*pos
            new_bal = bot.signals.get("__balance__", C("INITIAL_BALANCE")) - loss
            bot.signals["__balance__"] = new_bal
            losses = bot.signals.get("__recovery_losses__", []); losses.append(loss)
            if not C("RECOVERY_ENABLED"):
                bot.signals["__recovery_losses__"] = []; bot.signals["__in_recovery__"] = False
                data["cycle_status"] = "LOSS (no recovery)"
            elif len(losses) >= C("MAX_RECOVERY_LOSSES"):
                bot.signals["__recovery_losses__"] = []; bot.signals["__in_recovery__"] = False
                data["cycle_status"] = f"ENDED (Recovery aborted after {len(losses)})"
            else:
                bot.signals["__recovery_losses__"] = losses
                bot.signals["__in_recovery__"] = True
                data["cycle_status"] = f"IN RECOVERY (Loss #{len(losses)})"
        data["profit_usd"] = round(profit if result=="TP" else -loss, 2)
        data["balance_after"] = round(new_bal, 2)
        trades.append({
            "bot": bot.name, "symbol": symbol, "direction": data["direction"],
            "entry": entry, "sl": data["sl"], "tp": data["tp"], "exit": float(close_price),
            "result": result, "risk_pct": data.get("risk_pct"), "rr": data.get("rr"),
            "position_size": data.get("position_size"), "leverage": data.get("leverage"),
            "profit_usd": data["profit_usd"], "balance_after": data["balance_after"],
            "cycle_status": data["cycle_status"],
            "opened_ms": data["opened_ms"], "closed_ms": data["closed_ms"],
            "hold_hours": round((data["closed_ms"]-data["opened_ms"])/3600000,2),
        })

# =====================================================================
# چرخه ربات
# =====================================================================
def run_bot(bot, ctx, btc_4h, btc_1h, trades):
    _CUR["bot"] = bot.name
    update_open(bot, ctx, trades)
    open_trades = [s for k,s in bot.signals.items() if not k.startswith("__") and s.get("status")=="OPEN"]
    if open_trades: return 0   # single_trade: فقط یک معامله باز
    if bot.signals.get("__balance__", C("INITIAL_BALANCE")) <= 0: return 0
    candidates = []
    t = ctx.ref_time_ms
    for symbol in SYMBOLS:
        try:
            prev = bot.signals.get(symbol)
            if prev and prev.get("status")=="OPEN": continue
            if prev and bot.cooldown_ms and prev.get("closed_ms") and t-prev["closed_ms"]<bot.cooldown_ms: continue
            c = bot.analyze(symbol, ctx)
            if c is None: continue
            prev_key = prev.get("signal_key") if prev else None
            if prev and prev_key==c["signal_key"] and prev.get("direction")==c["direction"]: continue
            if symbol != BTC_SYMBOL:
                if bot.btc_filter_4h and btc_4h in ("LONG","SHORT") and c["direction"]!=btc_4h: continue
                if bot.btc_filter_1h and btc_1h in ("LONG","SHORT") and c["direction"]!=btc_1h: continue
            candidates.append(c)
        except Exception: continue
    if not candidates: return 0
    candidates.sort(key=lambda x: x["score"], reverse=True)
    c = candidates[0]  # single_trade
    try:
        entry = float(get_data(ctx, c["symbol"], "15m")["close"].iloc[-1])
        if abs(entry-c["close"]) > c["atr"]*C("MAX_ENTRY_DRIFT_ATR"): return 0
        sl, tp = c["levels"](entry)
        if sl<=0 or tp<=0: return 0
        rr = abs(tp-entry)/abs(entry-sl)
        bal = bot.signals.get("__balance__", C("INITIAL_BALANCE"))
        sizing = calc_sizing(bot.risk_percent, entry, sl, tp, bal,
                             bot.signals.get("__in_recovery__", False),
                             bot.signals.get("__recovery_losses__", []))
        bot.signals[c["symbol"]] = {
            "direction": c["direction"], "entry": entry, "tp": tp, "sl": sl,
            "signal_key": c["signal_key"], "setup": c["setup"], "score": round(c["score"],4),
            "risk_pct": round(abs(entry-sl)/entry*100,3), "rr": round(rr,2),
            "status": "OPEN", "opened_ms": ctx.ref_time_ms,
            **sizing,
        }
        return 1
    except Exception:
        return 0

# =====================================================================
# بک‌تست
# =====================================================================
def run_backtest(data_dir, start_ms, end_ms, step_ms, bots, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    trades = []
    def loader(symbol, tf): return get_df(data_dir, symbol, tf)
    t = start_ms; cycle = 0; t0 = datetime.now(timezone.utc)
    total_cycles = (end_ms-start_ms)//step_ms + 1
    # --- DEBUG: check data coverage once ---
    print("DATA CHECK:", flush=True)
    print(f"  backtest window: {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc):%Y-%m-%d} → "
          f"{datetime.fromtimestamp(end_ms/1000, tz=timezone.utc):%Y-%m-%d}", flush=True)
    print(f"  start_ms={start_ms}  end_ms={end_ms}", flush=True)
    for sym in [BTC_SYMBOL] + SYMBOLS[:5]:
        df = get_df(data_dir, sym, "15m")
        if df is None or len(df) == 0:
            print(f"  {sym} 15m: NO DATA", flush=True)
            continue
        lo = int(df["timestamp"].iloc[0]); hi = int(df["timestamp"].iloc[-1])
        d0 = datetime.fromtimestamp(lo/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        d1 = datetime.fromtimestamp(hi/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"  {sym} 15m: n={len(df)} | {d0} → {d1} | ts=[{lo} .. {hi}]", flush=True)
        # also try 4h resample once
        df4 = get_df(data_dir, sym, "4h")
        if df4 is None or len(df4) == 0:
            print(f"    -> 4h resample: EMPTY/FAIL", flush=True)
        else:
            lo4 = int(df4["timestamp"].iloc[0]); hi4 = int(df4["timestamp"].iloc[-1])
            def _safe_dt(ms):
                try:
                    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                except (ValueError, OSError, OverflowError):
                    return f"BAD_TS({ms})"
            print(f"    -> 4h resample: n={len(df4)} | {_safe_dt(lo4)} → {_safe_dt(hi4)} | ts=[{lo4} .. {hi4}]", flush=True)
    print("-" * 40, flush=True)
    while t <= end_ms:
        cycle += 1
        ctx = type("C", (), {})()
        ctx.ref_time_ms = t; ctx.cache = {}; ctx.data_loader = loader
        try:
            btc_4h, btc_1h = trend_of(get_data(ctx,BTC_SYMBOL,"4h")), trend_of(get_data(ctx,BTC_SYMBOL,"1h"))
        except Exception:
            btc_4h, btc_1h = "UNKNOWN","UNKNOWN"
        try: ctx.btc_filter = get_btc_filter(ctx)
        except Exception: ctx.btc_filter = None
        for bot in bots:
            try:
                run_bot(bot, ctx, btc_4h, btc_1h, trades)
            except Exception as e:
                if cycle == 1 or cycle % 500 == 0:
                    print(f"⚠️ {bot.name} error @cycle {cycle}: {type(e).__name__}: {e}", flush=True)
        # --- DEBUG: data health check ---
        if cycle == 1 or cycle % 1000 == 0:
            try:
                # raw (unfiltered) frames from loader
                raw15 = ctx.data_loader(BTC_SYMBOL, "15m")
                raw4  = ctx.data_loader(BTC_SYMBOL, "4h")
                def _rng(df, name):
                    if df is None or len(df) == 0:
                        return f"{name}: EMPTY"
                    lo = int(df["timestamp"].iloc[0]); hi = int(df["timestamp"].iloc[-1])
                    d0 = datetime.fromtimestamp(lo/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                    d1 = datetime.fromtimestamp(hi/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                    return f"{name}: n={len(df)} | {d0} → {d1} | ts=[{lo} .. {hi}]"
                ref_d = datetime.fromtimestamp(ctx.ref_time_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                print(f"DEBUG c={cycle} ref={ctx.ref_time_ms} ({ref_d})", flush=True)
                print(f"  {_rng(raw15, 'raw15')}", flush=True)
                print(f"  {_rng(raw4,  'raw4 ')}", flush=True)
                if raw4 is not None and len(raw4):
                    n_le = int((raw4["timestamp"] <= ctx.ref_time_ms).sum())
                    print(f"  4h rows with ts<=ref: {n_le}", flush=True)
                # now try filtered get_data
                df4 = get_data(ctx, BTC_SYMBOL, "4h")
                print(f"  get_data 4h OK, rows={len(df4)}", flush=True)
            except Exception as e:
                print(f"DEBUG c={cycle} BTC FAILED: {type(e).__name__}: {e}", flush=True)
        if cycle % 200 == 0:
            el = (datetime.now(timezone.utc)-t0).total_seconds()
            eta = el/cycle*(total_cycles-cycle) if cycle < total_cycles else 0
            print(f"⏱ cycle {cycle}/{total_cycles} | trades={len(trades)} | ETA {eta/60:.1f}m", flush=True)
    # --- DEBUG: print funnels ---
    print("=" * 50, flush=True)
    print("FUNNEL SUMMARY (why signals were rejected):", flush=True)
    for bot in bots:
        print_funnel(bot.name, top=15)
    print("=" * 50, flush=True)
    # ذخیره
    if trades:
        pd.DataFrame(trades).to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    else:
        pd.DataFrame(columns=["bot","symbol","direction","entry","sl","tp","exit","result",
                              "profit_usd","balance_after"]).to_csv(os.path.join(out_dir,"trades.csv"), index=False)
    return trades

def build_report(bots, trades, start_ms, end_ms, step_min):
    lines = []
    lines.append("نتایج بک‌تست")
    lines.append("━"*40)
    d0 = datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    d1 = datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).strftime("%Y-%m-%d")
    lines.append(f"بازه: {d0} → {d1} | گام: {step_min} دقیقه")
    lines.append(f"ریسک پایه: {C('RISK_PERCENT')}% | اهرم سقف: {C('MAX_LEVERAGE')}x | Recovery: {C('RECOVERY_ENABLED')}")
    lines.append("")
    df = pd.DataFrame(trades) if trades else pd.DataFrame()
    total_win = total_loss = 0
    all_r = []
    for bot in bots:
        lines.append(f"🤖 {bot.name} ({bot.label})")
        if df.empty or bot.name not in df["bot"].values:
            lines.append("   بدون معامله")
            lines.append("")
            continue
        b = df[df["bot"]==bot.name]
        n = len(b); wins = int((b["result"]=="TP").sum()); losses = n-wins
        total_win += wins; total_loss += losses
        wr = 100*wins/n if n else 0
        pnl = b["profit_usd"].sum()
        rs = []
        for _, r in b.iterrows():
            risk = abs(r["entry"]-r["sl"]); rr_val = abs(r["tp"]-r["entry"])/risk if risk>0 else 0
            rs.append(rr_val if r["result"]=="TP" else -1.0)
        all_r += rs
        final_bal = b["balance_after"].iloc[-1]
        peak = b["balance_after"].cummax(); dd = (peak-b["balance_after"])/peak
        max_dd = 100*dd.max() if len(dd) else 0
        avg_hold = b["hold_hours"].mean()
        lines.append(f"   معاملات: {n} | برد: {wins} ({wr:.0f}%) | مجموع R: {sum(rs):+.1f}")
        lines.append(f"   PnL: ${pnl:+.2f} | بالانس نهایی: ${final_bal:.2f} | حداکثر افت: -{max_dd:.1f}%")
        lines.append(f"   میانگین نگهداری: {avg_hold:.1f}h | میانگین R: {np.mean(rs):+.2f}")
        lines.append("")
    lines.append("━"*40)
    n_all = total_win+total_loss
    wr_all = 100*total_win/n_all if n_all else 0
    lines.append(f"📊 جمع کل: {n_all} معامله | برد {total_win} ({wr_all:.0f}%) | R کل {sum(all_r):+.1f}")
    lines.append("━"*40)
    return "\n".join(lines)

# =====================================================================
# main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--end", default=None)
    ap.add_argument("--step-minutes", type=int, default=15)
    ap.add_argument("--data-dir", default="data/okx")
    ap.add_argument("--out-dir", default="backtest_results")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--bots", nargs="*", default=None)
    args = ap.parse_args()

    if args.set: parse_overrides(args.set)

    if args.end: end_dt = datetime.strptime(args.end,"%Y-%m-%d").replace(tzinfo=timezone.utc)
    else: end_dt = datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
    end_ms = int(end_dt.timestamp()*1000)
    start_ms = end_ms - args.days*24*3600*1000
    step_ms = args.step_minutes*60*1000

    bots = ALL_BOTS
    if args.bots:
        names = set(args.bots)
        bots = [b for b in ALL_BOTS if b.name in names]
        if not bots:
            print(f"⚠️ هیچ رباتی با این اسم‌ها پیدا نشد: {names}")
            sys.exit(1)
        print(f"🎯 ربات‌های انتخابی: {[b.name for b in bots]}")

    print(f"بک‌تست: {datetime.fromtimestamp(start_ms/1000,tz=timezone.utc):%Y-%m-%d} → "
          f"{end_dt:%Y-%m-%d} | گام {args.step_minutes}m | {len(bots)} ربات")
    trades = run_backtest(args.data_dir, start_ms, end_ms, step_ms, bots, args.out_dir)
    report = build_report(bots, trades, start_ms, end_ms, args.step_minutes)
    with open("report.txt","w",encoding="utf-8") as f: f.write(report)
    print(report)

if __name__ == "__main__":
    main()