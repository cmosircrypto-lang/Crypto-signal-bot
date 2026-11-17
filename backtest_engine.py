import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import importlib.util
import sys
import warnings
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

spec = importlib.util.spec_from_file_location("main", "main.py")
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)

DATA_DIR = "data"
RESULTS_FILE = "backtest_results.csv"
IND_CACHE = {}
CURRENT_TS = 0
FUNNEL = defaultdict(Counter)

def load_df_with_indicators(symbol, tf):
    key = (symbol, tf)
    if key in IND_CACHE:
        return IND_CACHE[key]
    path = os.path.join(DATA_DIR, f"{symbol}_{tf}.parquet")
    if not os.path.exists(path):
        raise ValueError(f"Missing data: {path}")
    df = pd.read_parquet(path)
    df = main.calculate_indicators(df)
    IND_CACHE[key] = df
    return df

def mock_get_data(ctx, symbol, timeframe):
    df_full = load_df_with_indicators(symbol, timeframe)
    if timeframe == "4h":
        current_ts_rounded = (CURRENT_TS // (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000)
    elif timeframe == "1d":
        current_ts_rounded = (CURRENT_TS // (24 * 60 * 60 * 1000)) * (24 * 60 * 60 * 1000)
    elif timeframe == "15m":
        current_ts_rounded = (CURRENT_TS // (15 * 60 * 1000)) * (15 * 60 * 1000)
    elif timeframe == "5m":
        current_ts_rounded = (CURRENT_TS // (5 * 60 * 1000)) * (5 * 60 * 1000)
    else:
        current_ts_rounded = CURRENT_TS
    df_slice = df_full[df_full["timestamp"] <= current_ts_rounded].copy()
    if len(df_slice) < 40:
        raise ValueError(f"Not enough data at {CURRENT_TS}")
    return df_slice

main.get_data = mock_get_data

class MockExchange:
    def fetch_ticker(self, symbol): 
        return {"last": 0.0}
    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        df = load_df_with_indicators(symbol, timeframe)
        if since is not None:
            df = df[df["timestamp"] >= since]
        if timeframe == "5m":
            current_ts_rounded = (CURRENT_TS // (5 * 60 * 1000)) * (5 * 60 * 1000)
        else:
            current_ts_rounded = CURRENT_TS
        df = df[df["timestamp"] <= current_ts_rounded]
        return df.head(limit)[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()

def rej_collector(bot_name):
    def _rej(label="unlabeled"):
        FUNNEL[bot_name][label] += 1
        return None
    return _rej

# =====================================================================
# بک‌تست با مدیریت سرمایه
# =====================================================================
def run_backtest_with_sizing(bot):
    global CURRENT_TS
    print(f"\n▶ Backtesting {bot.name} ({bot.label}) with SIZING...")
    
    main.rej = rej_collector(bot.name)
    
    btc_df = load_df_with_indicators(main.BTC_SYMBOL, "1h")
    start_idx = 300
    DAYS = 120
    end_idx = min(len(btc_df) - 100, start_idx + 24 * DAYS)
    
    trades = []
    open_trade = None
    cooldown_until = 0
    analyzed = 0
    signal_attempts = 0
    
    # وضعیت مدیریت سرمایه
    balance = bot.initial_balance
    in_recovery = False
    recovery_losses = []
    
    for i in range(start_idx, end_idx):
        CURRENT_TS = int(btc_df["timestamp"].iloc[i])
        
        if (i - start_idx) % 100 == 0 and (i - start_idx) > 0:
            print(f"  📊 Progress: {((i - start_idx) / (end_idx - start_idx) * 100):.0f}%")
        
        ctx = main.Ctx(MockExchange(), ref_time_ms=CURRENT_TS)
        
        # ۱. بررسی بسته شدن معامله باز
        if open_trade:
            try:
                df_5m = load_df_with_indicators(open_trade["symbol"], "5m")
                candles_5m = df_5m[(df_5m["timestamp"] >= open_trade["opened_ms"]) & (df_5m["timestamp"] <= CURRENT_TS)]
            except:
                candles_5m = []
                
            closed = False
            for _, cd in candles_5m.iterrows():
                high, low, close = cd["high"], cd["low"], cd["close"]
                direction, tp, sl = open_trade["direction"], open_trade["tp"], open_trade["sl"]
                
                if direction == "LONG": 
                    hit_sl, hit_tp = low <= sl, high >= tp
                else: 
                    hit_sl, hit_tp = high >= sl, low <= tp
                    
                if hit_sl and hit_tp:
                    res = "TP" if abs(close - tp) < abs(close - sl) else "SL"
                    closed = True
                elif hit_sl: 
                    res, closed = "SL", True
                elif hit_tp: 
                    res, closed = "TP", True
                    
                if closed:
                    open_trade["result"] = res
                    open_trade["closed_ms"] = int(cd["timestamp"])
                    trades.append(open_trade)
                    
                    # محاسبه سود/ضرر دلاری
                    entry_price = open_trade["entry"]
                    pos_size = open_trade.get("position_size", 0)
                    
                    if res == "TP":
                        profit_usd = abs(open_trade["tp"] - entry_price) * pos_size
                        balance += profit_usd
                        in_recovery = False
                        recovery_losses = []
                        cycle_status = "ENDED (Win)"
                    else:
                        loss_usd = abs(entry_price - open_trade["sl"]) * pos_size
                        balance -= loss_usd
                        recovery_losses.append(loss_usd)
                        in_recovery = True
                        cycle_status = f"IN RECOVERY (Loss #{len(recovery_losses)})"
                    
                    open_trade["profit_usd"] = round(profit_usd if res == "TP" else -loss_usd, 2)
                    open_trade["balance_after"] = round(balance, 2)
                    open_trade["cycle_status"] = cycle_status
                    
                    open_trade = None
                    cooldown_until = CURRENT_TS + (bot.cooldown_ms or 0)
                    break
        
        if open_trade or CURRENT_TS < cooldown_until: 
            continue
            
        # ۲. جستجوی سیگنال جدید
        best_signal = None
        for symbol in main.SYMBOLS:
            if not os.path.exists(os.path.join(DATA_DIR, f"{symbol}_1h.parquet")): 
                continue
            
            try:
                analyzed += 1
                c = bot.analyze(symbol, ctx)
                if c is None: 
                    continue
                
                if symbol != main.BTC_SYMBOL and bot.btc_filter_4h:
                    btc_4h_df = load_df_with_indicators(main.BTC_SYMBOL, "4h")
                    btc_4h_ts = (CURRENT_TS // (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000)
                    btc_4h_df = btc_4h_df[btc_4h_df["timestamp"] <= btc_4h_ts]
                    if len(btc_4h_df) > 20:
                        btc_trend = main.trend_of(btc_4h_df)
                        if btc_trend != c["direction"]:
                            FUNNEL[bot.name]["~btc_filter"] += 1
                            continue
                            
                if best_signal is None or c["score"] > best_signal["score"]:
                    best_signal = c
                    signal_attempts += 1
            except Exception as e:
                FUNNEL[bot.name]["EXCEPTION"] += 1
                continue
                
        # ۳. باز کردن معامله با محاسبه حجم
        if best_signal:
            c = best_signal
            entry = c["close"]
            lv = c["levels"](entry)
            if lv:
                sl, tp = lv
                
                # محاسبه مدیریت سرمایه
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
                
                open_trade = {
                    "symbol": c["symbol"], "direction": c["direction"],
                    "entry": entry, "tp": tp, "sl": sl, "score": c["score"],
                    "setup": c["setup"], "opened_ms": CURRENT_TS, "status": "OPEN",
                    "position_size": round(position_size, 6),
                    "position_value": round(position_value, 2),
                    "leverage": round(max(1.0, leverage_needed), 2),
                    "risk_amount": round(risk_amount, 2),
                    "balance_before": round(balance, 2),
                }
    
    print(f"  ✅ {len(trades)} trades | Final Balance: ${balance:.2f}")
    print(f"  📊 Full funnel for {bot.name}:")
    for k in sorted(FUNNEL[bot.name].keys()):
        print(f"     {k}: {FUNNEL[bot.name][k]}")
    
    return trades, balance

# =====================================================================
# بک‌تست بدون مدیریت سرمایه (ربات‌های اصلی)
# =====================================================================
def run_backtest_simple(bot):
    global CURRENT_TS
    print(f"\n▶ Backtesting {bot.name} ({bot.label})...")
    
    main.rej = rej_collector(bot.name)
    
    btc_df = load_df_with_indicators(main.BTC_SYMBOL, "1h")
    start_idx = 300
    DAYS = 120
    end_idx = min(len(btc_df) - 100, start_idx + 24 * DAYS)
    
    trades = []
    open_trade = None
    cooldown_until = 0
    analyzed = 0
    signal_attempts = 0
    
    for i in range(start_idx, end_idx):
        CURRENT_TS = int(btc_df["timestamp"].iloc[i])
        
        if (i - start_idx) % 100 == 0 and (i - start_idx) > 0:
            print(f"  📊 Progress: {((i - start_idx) / (end_idx - start_idx) * 100):.0f}%")
        
        ctx = main.Ctx(MockExchange(), ref_time_ms=CURRENT_TS)
        
        if open_trade:
            try:
                df_5m = load_df_with_indicators(open_trade["symbol"], "5m")
                candles_5m = df_5m[(df_5m["timestamp"] >= open_trade["opened_ms"]) & (df_5m["timestamp"] <= CURRENT_TS)]
            except:
                candles_5m = []
                
            closed = False
            for _, cd in candles_5m.iterrows():
                high, low, close = cd["high"], cd["low"], cd["close"]
                direction, tp, sl = open_trade["direction"], open_trade["tp"], open_trade["sl"]
                
                if direction == "LONG": 
                    hit_sl, hit_tp = low <= sl, high >= tp
                else: 
                    hit_sl, hit_tp = high >= sl, low <= tp
                    
                if hit_sl and hit_tp:
                    res = "TP" if abs(close - tp) < abs(close - sl) else "SL"
                    closed = True
                elif hit_sl: 
                    res, closed = "SL", True
                elif hit_tp: 
                    res, closed = "TP", True
                    
                if closed:
                    open_trade["result"] = res
                    open_trade["closed_ms"] = int(cd["timestamp"])
                    trades.append(open_trade)
                    open_trade = None
                    cooldown_until = CURRENT_TS + (bot.cooldown_ms or 0)
                    break
        
        if open_trade or CURRENT_TS < cooldown_until: 
            continue
            
        best_signal = None
        for symbol in main.SYMBOLS:
            if not os.path.exists(os.path.join(DATA_DIR, f"{symbol}_1h.parquet")): 
                continue
            
            try:
                analyzed += 1
                c = bot.analyze(symbol, ctx)
                if c is None: 
                    continue
                
                if symbol != main.BTC_SYMBOL and bot.btc_filter_4h:
                    btc_4h_df = load_df_with_indicators(main.BTC_SYMBOL, "4h")
                    btc_4h_ts = (CURRENT_TS // (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000)
                    btc_4h_df = btc_4h_df[btc_4h_df["timestamp"] <= btc_4h_ts]
                    if len(btc_4h_df) > 20:
                        btc_trend = main.trend_of(btc_4h_df)
                        if btc_trend != c["direction"]:
                            FUNNEL[bot.name]["~btc_filter"] += 1
                            continue
                            
                if best_signal is None or c["score"] > best_signal["score"]:
                    best_signal = c
                    signal_attempts += 1
            except Exception as e:
                FUNNEL[bot.name]["EXCEPTION"] += 1
                continue
                
        if best_signal:
            c = best_signal
            entry = c["close"]
            lv = c["levels"](entry)
            if lv:
                sl, tp = lv
                open_trade = {
                    "symbol": c["symbol"], "direction": c["direction"],
                    "entry": entry, "tp": tp, "sl": sl, "score": c["score"],
                    "setup": c["setup"], "opened_ms": CURRENT_TS, "status": "OPEN",
                }
    
    print(f"  ✅ {len(trades)} trades | analyzed={analyzed} | signal_attempts={signal_attempts}")
    print(f"  📊 Full funnel for {bot.name}:")
    for k in sorted(FUNNEL[bot.name].keys()):
        print(f"     {k}: {FUNNEL[bot.name][k]}")
    
    return trades, None

# =====================================================================
# اجرای اصلی
# =====================================================================
def main_run():
    all_results = []
    
    for bot in main.LIVE_BOTS:
        global IND_CACHE
        IND_CACHE = {}
        
        if bot.has_sizing:
            trades, final_balance = run_backtest_with_sizing(bot)
            for t in trades: 
                t["bot"] = bot.name
            all_results.extend(trades)
            print(f"\n💰 Final Balance for {bot.name}: ${final_balance:.2f}")
        else:
            trades, _ = run_backtest_simple(bot)
            for t in trades: 
                t["bot"] = bot.name
            all_results.extend(trades)
        
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_FILE, index=False)
        print(f"\n🎉 Saved {len(all_results)} trades to {RESULTS_FILE}")
        
        summary = df.groupby("bot").agg(
            trades=("result", "count"), 
            wins=("result", lambda x: (x == "TP").sum()), 
            total_r=("pnl_r", "sum") if "pnl_r" in df.columns else ("result", "count"),
            avg_r=("pnl_r", "mean") if "pnl_r" in df.columns else ("result", "count"),
        ).reset_index()
        
        if "pnl_r" in df.columns:
            summary["win_rate"] = (summary["wins"] / summary["trades"] * 100).round(1)
            summary["total_r"] = summary["total_r"].round(2)
            print("\n" + "=" * 60 + "\n📊 Backtest Summary\n" + "=" * 60)
            print(summary.to_string(index=False))
    else:
        print("❌ No trades generated.")

if __name__ == "__main__":
    main_run()