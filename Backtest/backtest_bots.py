#!/usr/bin/env python3
"""
بک‌تست ۶ ربات روی داده‌ی تاریخی OKX (کش‌شده توسط download_okx_data.py)

روش کار:
  * فایل ربات‌ها (crypto_signal_bots_lite.py) مستقیماً import می‌شود؛ یعنی استراتژی‌ها،
    فیلتر BTC، کولدان، سقف جهت‌ها، مدیریت سرمایه و چرخه‌ی جبران عیناً همان کد زنده‌اند و
    حتی خود تابع run_bot اجرا می‌شود. فقط این‌ها جایگزین شده‌اند:
        دریافت داده / قیمت لحظه‌ای / بررسی TP-SL  ← داده‌ی تاریخی
        تلگرام / اکسل / CSV / فایل JSON            ← حافظه (هیچ چیزی ارسال یا ذخیره نمی‌شود)
        ساعت سیستم                                 ← ساعت شبیه‌سازی
  * تایم‌فریم‌های 1h / 4h / 1d از روی کندل 15m ساخته می‌شوند (UTC).
  * در هر لحظه فقط کندل‌های بسته‌شده دیده می‌شوند (بدون look-ahead).
  * هر ربات هر --step-minutes دقیقه یک‌بار اجرا می‌شود (مثل کرون زنده). قیمت ورود = close آخرین کندل 15m.
  * TP/SL از روی کندل‌های 15m بعد از ورود بررسی می‌شود. اگر در یک کندل هر دو لمس شوند
    (--ambiguous live)، مثل کد زنده نزدیک‌تر بودن close به TP یا SL تعیین می‌کند؛
    با --ambiguous sl حالت محافظه‌کارانه (SL) استفاده می‌شود.

اجرا:
    python backtest_bots.py
    python backtest_bots.py --step-minutes 15 --fee-pct 0.05
    python backtest_bots.py --bots crypto-signal-bot-v70 crypto-signal-bot-v70a
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import time
import types
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import download_okx_data as dl            # فقط برای to_okx_inst و ثابت‌ها

BAR = {"5m": 300_000, "15m": 900_000}
TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

STATE = {"t": 0}
STORE = {}            # جایگزین فایل‌های JSON ربات‌ها
TRADES = []           # معاملات بسته‌شده
FUN = defaultdict(Counter)
DATA = {}             # symbol -> tf -> {"df","end_ts"}
RAW = {}              # symbol -> آرایه‌های numpy کندل پایه
STEP_SLICE = {}
STEP_ANALYZE = {}
CFG = {"ambiguous": "live"}
_msg = [0]


# ----------------------------------------------------------------------------
# بارگذاری ربات‌ها با جایگزینی وابستگی‌های بیرونی
# ----------------------------------------------------------------------------
def load_bots_module(path):
    for name in ("ccxt", "jdatetime", "excel_log"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["excel_log"].log_signal = lambda *a, **k: None
    sys.modules["excel_log"].log_result = lambda *a, **k: None
    spec = importlib.util.spec_from_file_location("bots_lite", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# داده
# ----------------------------------------------------------------------------
def resample(df, tf_ms, bar_ms):
    b = (df["ts"] // tf_ms) * tf_ms
    g = df.groupby(b)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"), n=("ts", "count"))
    out = out[out["n"] == tf_ms // bar_ms].drop(columns="n")
    out.index.name = "timestamp"
    return out.reset_index()


def load_data(B, symbols, data_dir, bar):
    bar_ms = BAR[bar]
    ok, missing = [], []
    for k, sym in enumerate(symbols, 1):
        inst = dl.to_okx_inst(sym)
        path = os.path.join(data_dir, bar, f"{inst}.csv.gz")
        if not os.path.exists(path):
            missing.append(sym)
            continue
        raw = pd.read_csv(path).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        if len(raw) < 100:
            missing.append(sym)
            continue
        RAW[sym] = {"ts": raw["ts"].to_numpy(np.int64), "end": raw["ts"].to_numpy(np.int64) + bar_ms,
                    "high": raw["high"].to_numpy(float), "low": raw["low"].to_numpy(float),
                    "close": raw["close"].to_numpy(float)}
        per_tf = {}
        for tf, ms in TF_MS.items():
            if ms == bar_ms:
                d = raw.rename(columns={"ts": "timestamp"}).copy()
            else:
                d = resample(raw, ms, bar_ms)
            d = B.calculate_indicators(d)
            per_tf[tf] = {"df": d, "end_ts": d["timestamp"].to_numpy(np.int64) + ms}
        DATA[sym] = per_tf
        ok.append(sym)
        if k % 10 == 0:
            print(f"  بارگذاری داده: {k}/{len(symbols)}", flush=True)
    return ok, missing


# ----------------------------------------------------------------------------
# جایگزین‌های توابع ربات
# ----------------------------------------------------------------------------
def p_get_data(ctx, symbol, timeframe):
    key = (symbol, timeframe)
    hit = STEP_SLICE.get(key)
    if hit is not None:
        return hit
    d = DATA[symbol][timeframe]
    n = int(np.searchsorted(d["end_ts"], ctx.ref_time_ms, side="right"))
    limit = 300 if timeframe == "1d" else 150
    if n < 40:
        raise ValueError(f"داده کافی نیست: {symbol} {timeframe}")
    df = d["df"].iloc[max(0, n - limit):n]
    STEP_SLICE[key] = df
    return df


def p_get_price(ctx, symbol):
    return float(p_get_data(ctx, symbol, "15m")["close"].iloc[-1])


def p_check_trade_result(exchange, symbol, data):
    R = RAW[symbol]
    t = STATE["t"]
    start = max(int(data["opened_ms"]), int(data.get("_scan_ms", 0)))
    i0 = int(np.searchsorted(R["ts"], start, side="left"))
    i1 = int(np.searchsorted(R["end"], t, side="right"))
    if i1 <= i0:
        return None, None
    h, l, c = R["high"][i0:i1], R["low"][i0:i1], R["close"][i0:i1]
    tp, sl = data["tp"], data["sl"]
    if data["direction"] == "LONG":
        hs, ht = l <= sl, h >= tp
    else:
        hs, ht = h >= sl, l <= tp
    anyhit = hs | ht
    if not anyhit.any():
        data["_scan_ms"] = int(R["end"][i1 - 1])
        return None, None
    j = int(np.argmax(anyhit))
    data["_hit_ms"] = int(R["ts"][i0 + j])
    if hs[j] and ht[j]:
        data["_ambig"] = True
        if CFG["ambiguous"] == "sl":
            return "SL", sl
        return ("TP", tp) if abs(c[j] - tp) < abs(c[j] - sl) else ("SL", sl)
    return ("TP", tp) if ht[j] else ("SL", sl)


def patch_bots(B, symbols):
    B.SYMBOLS = list(symbols)
    B.get_iran_time = lambda: ("", "", "")
    B.send_telegram = lambda *a, **k: (_msg.__setitem__(0, _msg[0] + 1) or _msg[0])
    B.delete_telegram = lambda *a, **k: None
    B.now_ms = lambda: STATE["t"]

    def load_json(path, default):
        if path not in STORE:
            STORE[path] = default
        return STORE[path]

    def save_json(path, data):
        STORE[path] = data

    B.load_json, B.save_json = load_json, save_json
    B.append_trade_csv = lambda bot_name, record: TRADES.append({**record, "bot": bot_name})
    B.get_data, B.get_price, B.check_trade_result = p_get_data, p_get_price, p_check_trade_result

    def collect(bot_name, top=12):
        c = B.FUNNEL.pop(bot_name, None)
        if c:
            FUN[bot_name].update(c)
    B.print_funnel = collect


def wrap_cached(bot):
    """v0a/v60a/v70a همان تحلیل ربات‌های اصلی را دارند؛ نتیجه در هر گام یک‌بار محاسبه می‌شود."""
    orig = bot.analyze

    def cached(symbol, ctx):
        key = (orig, symbol)
        if key not in STEP_ANALYZE:
            STEP_ANALYZE[key] = orig(symbol, ctx)
        return STEP_ANALYZE[key]
    bot.analyze = cached


# ----------------------------------------------------------------------------
# آمار
# ----------------------------------------------------------------------------
def trade_pnl_r(t):
    risk = abs(t["entry"] - t["sl"])
    if risk <= 0:
        return 0.0
    return abs(t["tp"] - t["entry"]) / risk if t["result"] == "TP" else -1.0


def max_drawdown(series, start=0.0):
    peak, dd, cur = start, 0.0, start
    for x in series:
        cur += x
        peak = max(peak, cur)
        dd = max(dd, peak - cur)
    return dd


def ms_to_date(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def bot_stats(bot, trades, fee_pct, blown_at):
    tr = sorted([t for t in trades if t["bot"] == bot.name], key=lambda t: t.get("_hit_ms", t["closed_ms"]))
    st = {"bot": bot.name, "label": bot.label, "trades": len(tr)}
    open_n = sum(1 for k, v in STORE.get(bot.signals_file, {}).items()
                 if not k.startswith("__") and v.get("status") == "OPEN")
    st["open_at_end"] = open_n
    if not tr:
        return st, tr
    r = [trade_pnl_r(t) for t in tr]
    for t, x in zip(tr, r):
        t["pnl_r"] = round(x, 3)
        risk_pct = abs(t["entry"] - t["sl"]) / t["entry"] * 100
        t["pnl_r_net"] = round(x - (2 * fee_pct / risk_pct if risk_pct > 0 else 0), 3)
    rn = [t["pnl_r_net"] for t in tr]
    wins = [x for x in r if x > 0]
    loss = [x for x in r if x < 0]
    hold = [(t.get("_hit_ms", t["closed_ms"]) - t["opened_ms"]) / 3.6e6 for t in tr]
    st.update({
        "wins": len(wins), "losses": len(loss),
        "win_rate_pct": round(100 * len(wins) / len(tr), 1),
        "total_R": round(sum(r), 2), "avg_R": round(sum(r) / len(r), 3),
        "profit_factor": round(sum(wins) / abs(sum(loss)), 2) if loss else None,
        "max_dd_R": round(max_drawdown(r), 2),
        "total_R_after_fees": round(sum(rn), 2), "max_dd_R_after_fees": round(max_drawdown(rn), 2),
        "avg_rr": round(float(np.mean([abs(t["tp"] - t["entry"]) / abs(t["entry"] - t["sl"]) for t in tr])), 2),
        "avg_hold_h": round(float(np.mean(hold)), 1),
        "longs": sum(1 for t in tr if t["direction"] == "LONG"),
        "shorts": sum(1 for t in tr if t["direction"] == "SHORT"),
        "ambiguous_candles": sum(1 for t in tr if t.get("_ambig")),
        "first_trade": ms_to_date(tr[0]["opened_ms"]), "last_trade": ms_to_date(tr[-1]["opened_ms"]),
    })
    # بدترین رشته‌ی ضرر پشت‌سرهم
    streak = best = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    st["max_loss_streak"] = best
    if bot.has_sizing:
        bal = [bot.initial_balance] + [t["balance_after"] for t in tr if "balance_after" in t]
        peak, dd = bal[0], 0.0
        for b in bal:
            peak = max(peak, b)
            dd = max(dd, (peak - b) / peak * 100 if peak > 0 else 100.0)
        st.update({
            "initial_balance": bot.initial_balance, "final_balance": round(bal[-1], 2),
            "return_pct": round((bal[-1] / bot.initial_balance - 1) * 100, 1),
            "min_balance": round(min(bal), 2), "max_dd_pct": round(dd, 1),
            "max_leverage": max([float(t.get("leverage", 1) or 1) for t in tr]),
            "blown": blown_at is not None, "blown_date": ms_to_date(blown_at) if blown_at else None,
        })
    # ماهانه
    monthly = defaultdict(float)
    for t, x in zip(tr, r):
        monthly[ms_to_date(t.get("_hit_ms", t["closed_ms"]))[:7]] += x
    st["monthly_R"] = {k: round(v, 2) for k, v in sorted(monthly.items())}
    return st, tr


def print_report(stats, funnels):
    print("\n" + "=" * 100)
    print("نتایج بک‌تست")
    print("=" * 100)
    hdr = f"{'bot':<26}{'trades':>7}{'win%':>7}{'totR':>8}{'avgR':>7}{'PF':>6}{'maxDD_R':>9}{'R(fee)':>8}{'hold_h':>8}{'L/S':>9}"
    print(hdr)
    print("-" * len(hdr))
    for s in stats:
        if not s["trades"]:
            print(f"{s['bot']:<26}{0:>7}  (بدون معامله)")
            continue
        print(f"{s['bot']:<26}{s['trades']:>7}{s['win_rate_pct']:>7}{s['total_R']:>8}{s['avg_R']:>7}"
              f"{str(s['profit_factor']):>6}{s['max_dd_R']:>9}{s['total_R_after_fees']:>8}{s['avg_hold_h']:>8}"
              f"{str(s['longs']) + '/' + str(s['shorts']):>9}")
    man = [s for s in stats if "final_balance" in s]
    if man:
        print("\nربات‌های مدیریت سرمایه (بالانس اولیه و ریسک ۳٪ + چرخه‌ی جبران، بدون کارمزد):")
        h2 = f"{'bot':<26}{'start$':>8}{'final$':>10}{'ret%':>9}{'min$':>9}{'maxDD%':>8}{'maxLev':>8}{'lossStreak':>11}  blown"
        print(h2)
        print("-" * len(h2))
        for s in man:
            print(f"{s['bot']:<26}{s['initial_balance']:>8}{s['final_balance']:>10}{s['return_pct']:>9}"
                  f"{s['min_balance']:>9}{s['max_dd_pct']:>8}{s['max_leverage']:>8}{s['max_loss_streak']:>11}  "
                  f"{'YES ' + s['blown_date'] if s['blown'] else 'no'}")
    print("\nR ماهانه:")
    months = sorted({m for s in stats for m in s.get("monthly_R", {})})
    if months:
        print(f"{'month':<9}" + "".join(f"{s['bot'].replace('crypto-signal-bot', 'bot'):>12}" for s in stats))
        for m in months:
            print(f"{m:<9}" + "".join(f"{s.get('monthly_R', {}).get(m, 0):>12}" for s in stats))
    print("\nپرتکرارترین دلایل رد سیگنال (۵ مورد اول، بدون ~skip ها):")
    for name, c in funnels.items():
        top = [(k, v) for k, v in c.most_common() if not k.startswith(("0_", "~skip"))][:5]
        print(f"  {name}: " + " | ".join(f"{k}={v}" for k, v in top))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="بک‌تست ربات‌ها روی داده‌ی OKX")
    ap.add_argument("--bots-file", default=os.path.join(HERE, "crypto_signal_bots_lite.py"))
    ap.add_argument("--data", default=os.path.join(HERE, "data", "okx"))
    ap.add_argument("--bar", default="15m", choices=list(BAR))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--end", default=None, help="پایان بک‌تست (YYYY-MM-DD, UTC). پیش‌فرض: انتهای داده‌ی BTC")
    ap.add_argument("--step-minutes", type=int, default=60, help="فاصله‌ی اجرای ربات‌ها (مضرب ۱۵)")
    ap.add_argument("--bots", nargs="*", default=None, help="فقط این ربات‌ها (نام کامل)")
    ap.add_argument("--fee-pct", type=float, default=0.05, help="کارمزد هر طرف به درصد (فقط برای ستون R(fee))")
    ap.add_argument("--ambiguous", default="live", choices=["live", "sl"])
    ap.add_argument("--out", default=os.path.join(HERE, "backtest_results"))
    args = ap.parse_args()
    CFG["ambiguous"] = args.ambiguous
    bar_ms = BAR[args.bar]
    if args.step_minutes % 15:
        sys.exit("--step-minutes باید مضرب ۱۵ باشد")

    B = load_bots_module(args.bots_file)
    symbols = list(B.SYMBOLS)
    print(f"بارگذاری داده از {args.data} ...")
    t0 = time.time()
    ok, missing = load_data(B, symbols, args.data, args.bar)
    if B.BTC_SYMBOL not in ok:
        sys.exit("داده‌ی BTC پیدا نشد؛ اول download_okx_data.py را اجرا کن.")
    if missing:
        print(f"⚠️ {len(missing)} نماد بدون داده (کنار گذاشته شد): {', '.join(missing)}")
    print(f"{len(ok)} نماد آماده ({time.time() - t0:.0f}s)")

    btc_last = int(RAW[B.BTC_SYMBOL]["end"][-1])
    end_ms = int(datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) \
        if args.end else (btc_last // TF_MS["1d"]) * TF_MS["1d"]
    start_ms = end_ms - args.days * TF_MS["1d"]
    btc_first = int(RAW[B.BTC_SYMBOL]["ts"][0])
    if btc_first > start_ms - 30 * TF_MS["1d"]:
        print(f"⚠️ داده‌ی قبل از شروع کمتر از ۳۰ روز است ({ms_to_date(btc_first)}) — اندیکاتورهای ابتدای بازه کم‌دقت‌اند.")
    print(f"بازه‌ی بک‌تست: {ms_to_date(start_ms)} → {ms_to_date(end_ms)} | گام: {args.step_minutes} دقیقه")

    patch_bots(B, ok)
    bots = [b for b in B.LIVE_BOTS if not args.bots or b.name in args.bots]
    if not bots:
        sys.exit("ربات مطابق --bots پیدا نشد")
    for b in bots:
        wrap_cached(b)
    active = list(bots)
    blown = {}

    step_ms = args.step_minutes * 60_000
    steps = list(range(start_ms, end_ms, step_ms))
    print(f"{len(steps):,} گام × {len(bots)} ربات × {len(ok)} نماد ...\n")
    t0 = time.time()
    devnull = io.StringIO()
    for k, t in enumerate(steps, 1):
        STATE["t"] = t
        STEP_SLICE.clear()
        STEP_ANALYZE.clear()
        ctx = B.Ctx(None, ref_time_ms=t)
        with contextlib.redirect_stdout(devnull):
            btc_4h, btc_1h = B.get_btc_trend(ctx)
            ctx.btc_filter = B.get_btc_filter(ctx)
            for bot in list(active):
                try:
                    B.run_bot(bot, ctx, btc_4h, btc_1h)
                except Exception as e:
                    print(f"خطا {bot.name}: {e}", file=sys.stderr)
                if bot.has_sizing and STORE.get(bot.signals_file, {}).get("__balance__", 1) <= 0:
                    blown[bot.name] = t
                    active.remove(bot)
            devnull.seek(0)
            devnull.truncate(0)
        if k % 200 == 0 or k == len(steps):
            el = time.time() - t0
            print(f"  {k:,}/{len(steps):,} | {ms_to_date(t)} | معاملات بسته: {len(TRADES)} | "
                  f"گذشته {el / 60:.1f}m | باقی‌مانده ~{el / k * (len(steps) - k) / 60:.1f}m", flush=True)

    # خروجی
    os.makedirs(args.out, exist_ok=True)
    stats, all_tr = [], []
    for b in bots:
        s, tr = bot_stats(b, TRADES, args.fee_pct, blown.get(b.name))
        stats.append(s)
        all_tr += tr
    print_report(stats, FUN)

    skip = {"telegram_sent", "telegram_message_id", "close_telegram_sent", "close_msg_attempts", "time"}
    keys = ["bot", "symbol", "direction", "setup", "entry", "sl", "tp", "close_price", "result", "pnl_r", "pnl_r_net",
            "opened_ms", "_hit_ms", "closed_ms", "score", "rsi", "vol_ratio", "atr_pct", "risk_pct", "rr",
            "position_size", "position_value", "leverage", "profit_usd", "balance_after", "cycle_status"]
    keys += sorted({k for t in all_tr for k in t} - set(keys) - skip)
    all_tr.sort(key=lambda t: t["opened_ms"])
    with open(os.path.join(args.out, "trades.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_tr)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"period": [ms_to_date(start_ms), ms_to_date(end_ms)], "step_minutes": args.step_minutes,
                   "symbols_used": len(ok), "symbols_missing": missing, "ambiguous_mode": args.ambiguous,
                   "fee_pct_per_side": args.fee_pct, "bots": stats,
                   "funnel": {k: dict(v) for k, v in FUN.items()}}, f, ensure_ascii=False, indent=2)
    print(f"\nذخیره شد: {os.path.join(args.out, 'trades.csv')} و summary.json")


if __name__ == "__main__":
    main()
