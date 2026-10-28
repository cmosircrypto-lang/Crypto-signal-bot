"""
بک‌تست هر سه ربات روی داده‌ی تاریخی توبیت، با همان توابع تحلیل main.py.

قواعد شبیه‌سازی:
- تصمیم روی کندل ۱۵دقیقه‌ی بسته گرفته می‌شود؛ ورود روی open کندل بعدی (با لغزش).
- اندیکاتورها فقط با کندل‌های بسته‌ی همان لحظه محاسبه می‌شوند (بدون نگاه به آینده).
- خروج با high/low کندل‌های ۱۵دقیقه؛ اگر در یک کندل هر دو لمس شوند، SL حساب می‌شود.
- کارمزد رفت‌وبرگشت و لغزش در R خالص لحاظ می‌شود.
- سقف سیگنال در هر اجرا و قفل هر نماد تا بسته شدن معامله رعایت می‌شود.
- معاملات باز در پایان بازه، جدا گزارش می‌شوند (با قیمت آخر، نه در آمار اصلی).
"""
import os
import io
import json
import time
import contextlib
import numpy as np
import pandas as pd
import main as M

BT_DAYS = int(os.environ.get("BT_DAYS", "90"))
BT_SYMBOLS = int(os.environ.get("BT_SYMBOLS", "0"))      # 0 = همه‌ی نمادها
FEE_PER_SIDE = float(os.environ.get("BT_FEE", "0.0006"))  # ۰.۰۶٪ به ازای هر طرف
SLIP = float(os.environ.get("BT_SLIP", "0.0005"))         # ۰.۰۵٪ لغزش به ازای هر طرف
WARMUP_MS = 30 * 24 * 3600 * 1000
MIN_CANDLES = 120
PAGE = 1000
M15 = M.TF_MS["15m"]
COLS = ["timestamp", "open", "high", "low", "close", "volume"]
RESULTS_FILE = "backtest_results.json"
BTC = M.BTC_SYMBOL
_DEVNULL = open(os.devnull, "w")


class BTCtx:
    """زمینه‌ی سازگار با توابع تحلیل: فقط cache از پیش‌پرشده دارد و هیچ درخواستی نمی‌زند."""
    def __init__(self, cache):
        self.exchange = None
        self.cache = cache
        self.tickers = {}


# =====================================================================
# دریافت داده
# =====================================================================
def fetch_history(ex, symbol, tf, since_ms, until_ms):
    step = M.TF_MS[tf]
    rows, since = [], since_ms
    while since < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=PAGE)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + step
        if nxt <= since:
            break
        since = nxt
    df = pd.DataFrame(rows, columns=COLS).drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"] + step <= until_ms]          # فقط کندل‌های بسته
    return df.reset_index(drop=True)


def load_data(ex, symbols, since, until):
    data = {}
    for sym in symbols:
        try:
            f = {}
            for tf in ("15m", "1h", "4h"):
                df = fetch_history(ex, sym, tf, since, until)
                if len(df) < MIN_CANDLES:
                    raise ValueError(f"فقط {len(df)} کندل {tf}")
                df = M.calculate_indicators(df)   # علّی است؛ مقدار هر ردیف فقط به گذشته وابسته است
                f[tf] = {"df": df, "ts": df["timestamp"].values, "o": df["open"].values,
                         "h": df["high"].values, "l": df["low"].values, "c": df["close"].values}
            data[sym] = f
        except Exception as e:
            print(f"skip {sym}: {e}")
    return data


# =====================================================================
# هم‌ترازی زمانی (تصمیم در پایان هر کندل ۱۵دقیقه‌ای)
# =====================================================================
def build_align(f, grid):
    T = grid + M15
    al = {}
    for tf in ("1h", "4h"):
        al[tf] = np.searchsorted(f[tf]["ts"], T - M.TF_MS[tf], side="right") - 1
    al["15m"] = np.searchsorted(f["15m"]["ts"], grid, side="right") - 1
    return al


def cache_at(sym, f, al, j):
    cache = {}
    for tf in ("15m", "1h", "4h"):
        k = al[tf][j]
        if k + 1 < 60:
            return None
        cache[(sym, tf)] = f[tf]["df"].iloc[: k + 1]
    # بررسی بدون نگاه به آینده: آخرین کندل هر تایم‌فریم باید تا لحظه‌ی تصمیم بسته شده باشد
    return cache


def trend_arr(f, al):
    df = f["4h"]["df"]
    e9, e21 = df["ema9"].values, df["ema21"].values
    k = al
    valid = k >= 0
    kk = np.where(valid, k, 0)
    out = np.where(e9[kk] > e21[kk], "LONG", np.where(e9[kk] < e21[kk], "SHORT", "NONE")).astype(object)
    out[~valid] = "UNKNOWN"
    return out


def trend_arr_tf(f, tf, k):
    df = f[tf]["df"]
    e9, e21 = df["ema9"].values, df["ema21"].values
    valid = k >= 0
    kk = np.where(valid, k, 0)
    out = np.where(e9[kk] > e21[kk], "LONG", np.where(e9[kk] < e21[kk], "SHORT", "NONE")).astype(object)
    out[~valid] = "UNKNOWN"
    return out


# =====================================================================
# شبیه‌سازی یک ربات
# =====================================================================
def trade_R(d, exit_px):
    sign = 1 if d["dir"] == "LONG" else -1
    gross = sign * (exit_px - d["entry"]) / d["entry"]
    risk = abs(d["entry"] - d["sl"]) / d["entry"]
    return float((gross - 2 * FEE_PER_SIDE) / risk)


def simulate(bot, data, grid, aligns, btc4, btc1, start_j):
    syms = list(data.keys())
    open_tr, closed, last_key, cool = {}, [], {}, {}
    taken = 0

    for j in range(start_j, len(grid) - 1):
        T = int(grid[j]) + M15

        # ---------- ۱) خروج: کندل ۱۵دقیقه‌ی j ----------
        for sym in list(open_tr):
            d = open_tr[sym]
            f = data[sym]["15m"]
            k = aligns[sym]["15m"][j]
            if k < 0 or f["ts"][k] != grid[j]:
                continue
            hi, lo = f["h"][k], f["l"][k]
            if d["dir"] == "LONG":
                sl_hit, tp_hit = lo <= d["sl"], hi >= d["tp"]
            else:
                sl_hit, tp_hit = hi >= d["sl"], lo <= d["tp"]
            if sl_hit or tp_hit:
                if sl_hit:
                    result = "SL"
                    px = d["sl"] * (1 - SLIP) if d["dir"] == "LONG" else d["sl"] * (1 + SLIP)
                else:
                    result, px = "TP", d["tp"]
                closed.append({"symbol": sym, "dir": d["dir"], "result": result,
                               "R": trade_R(d, px), "exit_ms": T, "setup": d["setup"]})
                del open_tr[sym]
                cool[sym] = T + bot.cooldown_ms

        # ---------- ۲) کاندیداها ----------
        btc_t4, btc_t1 = btc4[j], btc1[j]
        cands = []
        for sym in syms:
            if sym in open_tr:
                continue
            if cool.get(sym, 0) > grid[j]:
                continue
            cache = cache_at(sym, data[sym], aligns[sym], j)
            if cache is None:
                continue
            try:
                with contextlib.redirect_stdout(_DEVNULL):
                    c = bot.analyze(sym, BTCtx(cache))
            except Exception:
                c = None
            if c is None:
                continue
            if sym != BTC:
                if bot.btc_filter_4h and btc_t4 in ("LONG", "SHORT") and c["direction"] != btc_t4:
                    continue
                if bot.btc_filter_1h and btc_t1 in ("LONG", "SHORT") and c["direction"] != btc_t1:
                    continue
            key = (c["signal_key"], c["direction"])
            if last_key.get(sym) == key:
                continue
            c["sym"] = sym
            cands.append(c)

        # ---------- ۳) انتخاب بهترین‌ها و ورود روی open کندل بعدی ----------
        cands.sort(key=lambda x: x["score"], reverse=True)
        sent = 0
        for c in cands:
            if sent >= bot.max_signals:
                break
            sym = c["sym"]
            f = data[sym]["15m"]
            kn = aligns[sym]["15m"][j + 1]
            if kn < 0 or f["ts"][kn] != grid[j + 1]:
                continue
            nxt_open = f["o"][kn]
            if abs(nxt_open - c["close"]) > c["atr"] * M.MAX_ENTRY_DRIFT_ATR:
                continue
            entry = nxt_open * (1 + SLIP) if c["direction"] == "LONG" else nxt_open * (1 - SLIP)
            lv = c["levels"](entry)
            if lv is None:
                continue
            sl, tp = lv
            if sl <= 0 or tp <= 0:
                continue
            open_tr[sym] = {"dir": c["direction"], "entry": entry, "sl": sl, "tp": tp,
                            "setup": c["setup"]}
            last_key[sym] = (c["signal_key"], c["direction"])
            sent += 1
            taken += 1

    # معاملات باز در پایان: با قیمت آخر (mark-to-market) جدا گزارش می‌شوند
    last_j = len(grid) - 2
    open_end, mtm = [], 0.0
    for sym, d in open_tr.items():
        f = data[sym]["15m"]
        k = aligns[sym]["15m"][last_j]
        if k < 0:
            continue
        r = trade_R(d, f["c"][k])
        mtm += r
        open_end.append({"symbol": sym, "dir": d["dir"], "mtm_R": round(r, 3)})

    months = (grid[-2] - grid[start_j]) / (30.44 * 24 * 3600 * 1000)
    return {"closed": closed, "taken": taken, "open_end": open_end, "mtm_R": mtm, "months": months}


# =====================================================================
# خلاصه‌ی آماری
# =====================================================================
def summarize(bot, res):
    closed = sorted(res["closed"], key=lambda x: x["exit_ms"])
    Rs = [c["R"] for c in closed]
    n = len(Rs)
    wins = sum(1 for c in closed if c["result"] == "TP")
    total = float(sum(Rs))
    gw = float(sum(r for r in Rs if r > 0))
    gl = float(-sum(r for r in Rs if r < 0))
    cum = peak = mdd = 0.0
    for r in Rs:
        cum += r
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    per_sym = {}
    for c in closed:
        per_sym[c["symbol"]] = per_sym.get(c["symbol"], 0.0) + c["R"]
    ranked = sorted(per_sym.items(), key=lambda x: x[1], reverse=True)
    months = max(res["months"], 1e-9)
    return {
        "bot": bot.name,
        "months": round(months, 2),
        "signals_taken": res["taken"],
        "signals_per_month": round(res["taken"] / months, 1),
        "closed": n,
        "still_open_end": len(res["open_end"]),
        "win_rate": round(wins / n, 3) if n else None,
        "avg_R_net": round(total / n, 3) if n else None,
        "total_R_net": round(total, 2),
        "profit_factor": round(gw / gl, 2) if gl > 0 else None,
        "max_drawdown_R": round(mdd, 2),
        "open_mtm_R": round(res["mtm_R"], 2),
        "best_symbols": [(s, round(v, 2)) for s, v in ranked[:3]],
        "worst_symbols": [(s, round(v, 2)) for s, v in ranked[-3:]],
        "trades": closed,
        "open_end_trades": res["open_end"],
    }


def format_report(stats):
    lines = ["📊 نتیجه‌ی بک‌تست", f"بازه: {stats[0]['months']} ماه | کارمزد {FEE_PER_SIDE:.2%} و لغزش {SLIP:.2%} به ازای هر طرف",
             "━━━━━━━━━━━━━━━━━"]
    for s in stats:
        if s["closed"] == 0:
            lines.append(f"🤖 {s['bot']}\n   هیچ معامله‌ی بسته‌ای نبود | سیگنال: {s['signals_taken']}")
            continue
        pf = s["profit_factor"] if s["profit_factor"] is not None else "∞"
        lines.append(
            f"🤖 {s['bot']}\n"
            f"   سیگنال: {s['signals_taken']} ({s['signals_per_month']} در ماه) | بسته: {s['closed']} | باز در پایان: {s['still_open_end']}\n"
            f"   برد: {s['win_rate']:.0%} | میانگین R: {s['avg_R_net']:+.2f} | R کل: {s['total_R_net']:+.1f}\n"
            f"   Profit Factor: {pf} | افت سرمایه‌ی ماکسیمم: {s['max_drawdown_R']:.1f}R"
        )
    lines.append("━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


# =====================================================================
# اجرا
# =====================================================================
def run(ex, send=True, days=BT_DAYS, symbols=None):
    until = (int(time.time() * 1000) // M15) * M15
    since = until - (days * 24 * 3600 * 1000 + WARMUP_MS)
    syms = list(symbols) if symbols else list(M.SYMBOLS)
    if BT_SYMBOLS and not symbols:
        syms = syms[:BT_SYMBOLS]
    if BTC not in syms:
        syms = [BTC] + syms

    print(f"Loading data for {len(syms)} symbols...")
    data = load_data(ex, syms, since, until)
    if BTC not in data:
        raise RuntimeError("BTC data unavailable")

    grid = data[BTC]["15m"]["ts"]
    start_j = int(np.searchsorted(grid, grid[0] + WARMUP_MS))
    if start_j >= len(grid) - 2:
        raise RuntimeError(f"داده‌ی BTC کافی نیست: {len(grid)} کندل ۱۵دقیقه‌ای برای warmup و بازه")
    aligns = {s: build_align(data[s], grid) for s in data}
    btc4 = trend_arr_tf(data[BTC], "4h", aligns[BTC]["4h"])
    btc1 = trend_arr_tf(data[BTC], "1h", aligns[BTC]["1h"])

    stats = []
    for bot in M.BOTS:
        print(f"▶ backtest {bot.name} ...")
        res = simulate(bot, data, grid, aligns, btc4, btc1, start_j)
        st = summarize(bot, res)
        stats.append(st)
        print(f"  {bot.name}: taken={st['signals_taken']} closed={st['closed']} "
              f"win={st['win_rate']} avgR={st['avg_R_net']} totalR={st['total_R_net']}")

    report = format_report(stats)
    print(report)
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"days": days, "fee": FEE_PER_SIDE, "slip": SLIP, "stats": stats}, fh, ensure_ascii=False)
    if send:
        M.send_telegram(report)
    return stats


if __name__ == "__main__":
    run(M.create_exchange(), send=True)
