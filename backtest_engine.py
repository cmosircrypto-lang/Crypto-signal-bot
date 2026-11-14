import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import importlib.util
import sys
import warnings
warnings.filterwarnings("ignore")

# ایمپورت توابع از main.py (بدون اجرای خودکار main)
spec = importlib.util.spec_from_file_location("main", "main.py")
main = importlib.util.module_from_spec(spec)
sys.modules["main"] = main
spec.loader.exec_module(main)

DATA_DIR = "data"
RESULTS_FILE = "backtest_results.csv"
DATA_CACHE = {}

def load_df(symbol, tf):
    key = (symbol, tf)
    if key in DATA_CACHE:
        return DATA_CACHE[key]
    path = os.path.join(DATA_DIR, f"{symbol}_{tf}.parquet")
    if not os.path.exists(path):
        raise ValueError(f"Missing data: {path}")
    df = pd.read_parquet(path)
    df = main.calculate_indicators(df)
    DATA_CACHE[key] = df
    return df

class MockExchange:
    def fetch_ticker(self, symbol): return {"last": 0.0}
    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        df = load_df(symbol, timeframe)
        if since is not None:
            df = df[df["timestamp"] >= since]
        return df.head(limit)[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()

def mock_get_data(ctx, symbol, timeframe):
    df = load_df(symbol, timeframe)
    if len(df) < 40:
        raise ValueError(f"Not enough data")
    return df

main.get_data = mock_get_data

def run_backtest(bot):
    print(f"\n▶ Backtesting {bot.name} ({bot.label})...")
    btc_df = load_df(main.BTC_SYMBOL, "1h")
    
    start_idx = 300
    end_idx = len(btc_df) - 100
    
    trades = []
    open_trade = None
    cooldown_until = 0
    
    for i in range(start_idx, end_idx):
        current_ts = int(btc_df["timestamp"].iloc[i])
        ctx = main.Ctx(MockExchange(), ref_time_ms=current_ts)
        
        if open_trade:
            try:
                df_5m = load_df(open_trade["symbol"], "5m")
                candles_5m = df_5m[(df_5m["timestamp"] >= open_trade["opened_ms"]) & (df_5m["timestamp"] <= current_ts)]
            except:
                candles_5m = []
                
            closed = False
            for _, cd in candles_5m.iterrows():
                high, low, close = cd["high"], cd["low"], cd["close"]
                direction, tp, sl = open_trade["direction"], open_trade["tp"], open_trade["sl"]
                
                if direction == "LONG": hit_sl, hit_tp = low <= sl, high >= tp
                else: hit_sl, hit_tp = high >= sl, low <= tp
                    
                if hit_sl and hit_tp:
                    res = "TP" if abs(close - tp) < abs(close - sl) else "SL"
                    closed = True
                elif hit_sl: res, closed = "SL", True
                elif hit_tp: res, closed = "TP", True
                    
                if closed:
                    open_trade["result"] = res
                    open_trade["closed_ms"] = int(cd["timestamp"])
                    trades.append(open_trade)
                    risk = abs(open_trade["entry"] - open_trade["sl"])
                    open_trade["pnl_r"] = round(abs(open_trade["tp"] - open_trade["entry"]) / risk if res == "TP" else -1.0, 3)
                    open_trade = None
                    cooldown_until = current_ts + (bot.cooldown_ms or 0)
                    break
        
        if open_trade or current_ts < cooldown_until: continue
            
        best_signal = None
        for symbol in main.SYMBOLS:
            try:
                if not os.path.exists(os.path.join(DATA_DIR, f"{symbol}_1h.parquet")): continue
                ctx.ref_time_ms = current_ts
                c = bot.analyze(symbol, ctx)
                if c is None: continue
                    
                if symbol != main.BTC_SYMBOL and bot.btc_filter_4h:
                    btc_4h_df = load_df(main.BTC_SYMBOL, "4h")
                    btc_4h_df = btc_4h_df[btc_4h_df["timestamp"] <= current_ts]
                    if len(btc_4h_df) > 20:
                        btc_trend = main.trend_of(btc_4h_df)
                        if btc_trend != c["direction"]: continue
                            
                if best_signal is None or c["score"] > best_signal["score"]: best_signal = c
            except: continue
                
        if best_signal:
            c = best_signal
            entry = c["close"]
            lv = c["levels"](entry)
            if lv:
                sl, tp = lv
                open_trade = {"symbol": c["symbol"], "direction": c["direction"], "entry": entry, "tp": tp, "sl": sl, "score": c["score"], "setup": c["setup"], "opened_ms": current_ts, "status": "OPEN"}
    
    print(f"  ✅ {len(trades)} trades completed")
    return trades

def main_run():
    all_results = []
    for bot in main.LIVE_BOTS:
        global DATA_CACHE
        DATA_CACHE = {}
        trades = run_backtest(bot)
        for t in trades: t["bot"] = bot.name
        all_results.extend(trades)
        
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_FILE, index=False)
        print(f"\n🎉 Saved {len(all_results)} trades to {RESULTS_FILE}")
        summary = df.groupby("bot").agg(trades=("result", "count"), wins=("result", lambda x: (x == "TP").sum()), total_r=("pnl_r", "sum"), avg_r=("pnl_r", "mean")).reset_index()
        summary["win_rate"] = (summary["wins"] / summary["trades"] * 100).round(1)
        summary["total_r"] = summary["total_r"].round(2)
        print("\n" + "=" * 60 + "\n📊 Backtest Summary\n" + "=" * 60)
        print(summary.to_string(index=False))
    else:
        print("❌ No trades generated.")

if __name__ == "__main__":
    main_run()