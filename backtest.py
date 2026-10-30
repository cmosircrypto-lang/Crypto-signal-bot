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
BT_WARMUP_DAYS = float(os.environ.get("BT_WARMUP_DAYS", "10"))  # ۱۵دقیقه‌ای حدود ۳۶ روز تاریخچه دارد
WARMUP_MS = int(BT_WARMUP_DAYS * 24 * 3600 * 1000)
MIN_CANDLES = 120
MIN_CANDLES_1D = 40                                     # فیلتر BTC روزانه (ربات ۶) به حدود ۴۰ کندل نیاز دارد
D1_EXTRA_MS = 200 * 24 * 3600 * 1000                    # تاریخچه‌ی اضافه‌ی روزانه برای گرم شدن EMA
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
        self.btc_filter = None   # برای ربات ۶: نتیجه‌ی get_btc_filter در همان لحظه‌ی تصمیم


# =====================================================================
# دریافت داده
# =====================================================================
def fetch_history(ex, symbol, tf, since_ms, until_ms):
    """
    تاریخچه‌ی کندل‌ها را صفحه‌صفحه می‌گیرد.
    ابتدا با since به جلو می‌رود. اگر صرافی since را نادیده بگیرد (فقط آخرین کندل‌ها را بدهد)،
    با endTime به عقب صفحه‌بندی می‌کند.
    """
    step = M.TF_MS[tf]
    rows = []

    # --- روش ۱: جلو با since
    since = since_ms
    while since < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=PAGE)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + step
        if nxt <= since:
            break
        since = nxt
    df = pd.DataFrame(rows, columns=COLS).drop_duplicates("timestamp")
    mode = "since"

    # --- روش ۲ (اگر روش ۱ به ابتدای بازه نرسید): عقب با endTime
    if len(df) and df["timestamp"].min() > since_ms + 2 * step:
        rows, end = [], until_ms
        for _ in range(10000):
            batch = ex.fetch_ohlcv(symbol, timeframe=tf, limit=PAGE, params={"endTime": int(end)})
            if not batch:
                break
            rows.extend(batch)
            earliest = min(b[0] for b in batch)
            if earliest <= since_ms or earliest - 1 >= end:
                break
            end = earliest - 1
        df = pd.DataFrame(rows, columns=COLS).drop_duplicates("timestamp")
        mode = "endTime"

    df = df.sort_values("timestamp")
    df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] + step <= until_ms)]   # فقط کندل‌های بسته
    print(f"  fetch {symbol} {tf}: {len(df)} کندل (روش: {mode})")
    return df.reset_index(drop=True)


def load_data(ex, symbols, since, until):
    data = {}
    for sym in symbols:
        try:
            f = {}
            tfs = ("15m", "1h", "4h", "1d") if sym == BTC else ("15m", "1h", "4h")
            for tf in tfs:
                if tf == "1d":
                    # فقط برای فیلتر BTC؛ تاریخچه‌ی بیشتری می‌گیریم تا EMA روزانه گرم شود
                    df = fetch_history(ex, sym, tf, since - D1_EXTRA_MS, until)
                    min_c = MIN_CANDLES_1D
                else:
                    df = fetch_history(ex, sym, tf, since, until)
                    min_c = MIN_CANDLES
                if len(df) < min_c:
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


def btc_filters_series(data, aligns, grid):
    """
    برای هر کندل ۱۵دقیقه‌ای، نتیجه‌ی get_btc_filter همان لحظه را محاسبه می‌کند (همان منطق main.py).
    فیلتر روزانه/۴ساعته‌ی BTC فقط با کندل‌های بسته‌شده تا لحظه‌ی تصمیم ساخته می‌شود.
    """
    out = [None] * len(grid)
    if BTC not in data or "1d" not in data[BTC]:
        return out
    f, al = data[BTC], aligns[BTC]
    for j in range(len(grid)):
        cache = cache_at(BTC, f, al, j)
        if cache is None:
            continue
        k1 = int(al["1d"][j])
        if k1 + 1 < MIN_CANDLES_1D:
            continue
        cache[(BTC, "1d")] = f["1d"]["df"].iloc[: k1 + 1]
        try:
            with contextlib.redirect_stdout(_DEVNULL):
                out[j] = M.get_btc_filter(BTCtx(cache))
        except Exception:
            out[j] = None
    return out


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


def simulate(bot, data, grid, aligns, btc4, btc1, start_j, btc_filters=None):
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
                               "R": trade_R(d, px), "exit_ms": T, "entry_ms": d["entry_ms"],
                               "setup": d["setup"],
                               "risk_pct": abs(d["entry"] - d["sl"]) / d["entry"],
                               "rr": abs(d["tp"] - d["entry"]) / abs(d["entry"] - d["sl"])})
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
            bctx = BTCtx(cache)
            bctx.btc_filter = btc_filters[j] if btc_filters is not None else None
            try:
                with contextlib.redirect_stdout(_DEVNULL):
                    c = bot.analyze(sym, bctx)
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
                            "setup": c["setup"], "entry_ms": int(grid[j + 1])}
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
    return {"closed": closed, "taken": taken, "open_end": open_end, "mtm_R": mtm, "months": months,
            "t0": int(grid[start_j]), "t1": int(grid[-2])}


# =====================================================================
# خلاصه‌ی آماری
# =====================================================================
def setup_kind(s):
    """نام کلی الگو را بدون اعداد و جزئیات (سن کراس، سطح قیمت) برمی‌گرداند."""
    words = [w for w in str(s).split("(")[0].split() if not any(ch.isdigit() for ch in w)]
    return " ".join(words) or "-"


def random_baseline(res, data, grid, aligns, start_j, reps=None, seed=42):
    """
    پایه‌ی تصادفی: برای هر معامله‌ی واقعی، جهت و هندسه‌ی ریسک/ریوارد (درصد SL و نسبت R:R) حفظ می‌شود،
    اما نماد و زمان ورود تصادفی انتخاب می‌شود. خروج با همان قواعد (SL-first، کارمزد و لغزش) اعمال می‌شود.
    اگر ربات واقعاً ورودها را خوب انتخاب کند، میانگین R آن باید از توزیع این پایه بالاتر باشد.
    """
    reps = reps or int(os.environ.get("BT_BASE_REPS", "200"))
    tmpl = [t for t in res["closed"] if "risk_pct" in t]
    if not tmpl:
        return None
    rng = np.random.default_rng(seed)
    syms = list(data.keys())
    lo_j, hi_j = start_j, len(grid) - 3
    rep_means, rep_wins = [], []
    for _ in range(reps):
        Rs, wins = [], 0
        for t in tmpl:
            pick = None
            for _try in range(50):
                sym = syms[int(rng.integers(len(syms)))]
                j = int(rng.integers(lo_j, hi_j + 1))
                f = data[sym]["15m"]
                k = int(aligns[sym]["15m"][j + 1])
                if k >= 0 and f["ts"][k] == grid[j + 1]:
                    pick = (sym, k)
                    break
            if pick is None:
                continue
            sym, k = pick
            f = data[sym]["15m"]
            long_ = t["dir"] == "LONG"
            sign = 1 if long_ else -1
            entry = f["o"][k] * (1 + SLIP) if long_ else f["o"][k] * (1 - SLIP)
            rp, rr = t["risk_pct"], t["rr"]
            if long_:
                sl, tp = entry * (1 - rp), entry * (1 + rp * rr)
                sl_h = f["l"][k:] <= sl
                tp_h = f["h"][k:] >= tp
            else:
                sl, tp = entry * (1 + rp), entry * (1 - rp * rr)
                sl_h = f["h"][k:] >= sl
                tp_h = f["l"][k:] <= tp
            hit = sl_h | tp_h
            if hit.any():
                i = int(np.argmax(hit))
                if sl_h[i]:
                    px = sl * (1 - SLIP) if long_ else sl * (1 + SLIP)
                else:
                    px = tp
                    wins += 1
            else:
                px = f["c"][-1]            # mark-to-market در پایان داده
            gross = sign * (px - entry) / entry
            Rs.append((gross - 2 * FEE_PER_SIDE) / rp)
        if Rs:
            rep_means.append(float(np.mean(Rs)))
            rep_wins.append(wins / len(Rs))
    actual = float(np.mean([t["R"] for t in tmpl]))
    arr = np.array(rep_means)
    return {
        "reps": len(rep_means),
        "mean": round(float(arr.mean()), 3),
        "p5": round(float(np.percentile(arr, 5)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "win_mean": round(float(np.mean(rep_wins)), 3),
        "actual": round(actual, 3),
        "percentile": round(float((arr < actual).mean() * 100), 1),
    }


def breakdown(trades, field, norm=lambda x: str(x)):
    """آمار به تفکیک یک فیلد برای تشخیص این‌که زیان از کدام بخش می‌آید."""
    groups = {}
    for t in trades:
        groups.setdefault(norm(t.get(field, "-")), []).append(t)
    out = {}
    for k, items in groups.items():
        Rs = [x["R"] for x in items]
        n = len(Rs)
        out[k] = {
            "n": n,
            "win_rate": round(sum(1 for x in items if x["result"] == "TP") / n, 3),
            "avg_R": round(sum(Rs) / n, 3),
            "total_R": round(sum(Rs), 2),
        }
    return out


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

    # تقسیم بازه به دو نیمه‌ی زمانی بر اساس زمان ورود: برای بررسی پایداری نتیجه
    mid = (res["t0"] + res["t1"]) / 2
    def half_stats(items):
        Rs = [x["R"] for x in items]
        n = len(Rs)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "win_rate": round(sum(1 for x in items if x["result"] == "TP") / n, 3),
            "avg_R": round(sum(Rs) / n, 3),
            "total_R": round(sum(Rs), 2),
            "by_direction": breakdown(items, "dir"),
        }
    halves = {
        "نیمه‌ی اول": half_stats([c for c in closed if c["entry_ms"] < mid]),
        "نیمه‌ی دوم": half_stats([c for c in closed if c["entry_ms"] >= mid]),
    }

    return {
        "halves": halves,
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
        "exits": {"TP": sum(1 for c in closed if c["result"] == "TP"),
                  "SL": sum(1 for c in closed if c["result"] == "SL")},
        "by_direction": breakdown(closed, "dir"),
        "by_setup": breakdown(closed, "setup", setup_kind),
        "best_symbols": [(s, round(v, 2)) for s, v in ranked[:3]],
        "worst_symbols": [(s, round(v, 2)) for s, v in ranked[-3:]],
        "trades": closed,
        "open_end_trades": res["open_end"],
    }


def _half_line(name, h):
    if h["n"] == 0:
        return f"\n   {name}: معامله‌ای نبود"
    dirs = " | ".join(f"{d} {v['n']}: برد {v['win_rate']:.0%} R {v['avg_R']:+.2f}"
                      for d, v in sorted(h["by_direction"].items()))
    return (f"\n   {name}: {h['n']} معامله | برد {h['win_rate']:.0%} | میانگین R {h['avg_R']:+.2f} | R کل {h['total_R']:+.1f}"
            f"\n      ({dirs})")


def _base_line(s):
    b = s.get("baseline")
    if not b or s["closed"] == 0:
        return ""
    return (f"\n   پایه‌ی تصادفی (همان جهت و ریسک/ریوارد، ورود تصادفی): میانگین R {b['mean']:+.3f} "
            f"[{b['p5']:+.3f} تا {b['p95']:+.3f}] | برد تصادفی {b['win_mean']:.0%}"
            f"\n   ربات {s['avg_R_net']:+.3f} در صدک {b['percentile']:.0f}ام پایه")


def format_report(stats, market=None):
    lines = ["📊 نتیجه‌ی بک‌تست", f"بازه: {stats[0]['months']} ماه | کارمزد {FEE_PER_SIDE:.2%} و لغزش {SLIP:.2%} به ازای هر طرف"]
    if market:
        lines.append(f"🌍 بازار در بازه: BTC {market['btc_pct']:+.1f}% | میانه‌ی {market['symbols']} نماد {market['median_pct']:+.1f}%")
    lines.append("━━━━━━━━━━━━━━━━━")
    for s in stats:
        if s["closed"] == 0:
            lines.append(f"🤖 {s['bot']}\n   هیچ معامله‌ی بسته‌ای نبود | سیگنال: {s['signals_taken']}")
            continue
        pf = s["profit_factor"] if s["profit_factor"] is not None else "∞"
        lines.append(
            f"🤖 {s['bot']}\n"
            f"   سیگنال: {s['signals_taken']} ({s['signals_per_month']} در ماه) | بسته: {s['closed']} | باز در پایان: {s['still_open_end']}\n"
            f"   برد: {s['win_rate']:.0%} | میانگین R: {s['avg_R_net']:+.2f} | R کل: {s['total_R_net']:+.1f}\n"
            f"   Profit Factor: {pf} | افت سرمایه‌ی ماکسیمم: {s['max_drawdown_R']:.1f}R\n"
            f"   خروج: TP {s['exits']['TP']} / SL {s['exits']['SL']}"
            + "".join(
                f"\n   {d}: {v['n']} معامله | برد {v['win_rate']:.0%} | میانگین R {v['avg_R']:+.2f}"
                for d, v in sorted(s["by_direction"].items())
            )
            + "".join(
                f"\n   setup «{k}»: {v['n']} | برد {v['win_rate']:.0%} | R کل {v['total_R']:+.1f}"
                for k, v in sorted(s["by_setup"].items(), key=lambda x: -x[1]["n"])[:4]
            )
            + "".join(
                _half_line(name, h) for name, h in s["halves"].items()
            )
            + _base_line(s)
        )
    lines.append("━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


# =====================================================================
# اجرا
# =====================================================================
def market_move(data, grid, start_j):
    """تغییر قیمت BTC و میانه‌ی تغییر همه‌ی نمادها در بازه‌ی ارزیابی؛ برای فهم جهت کلی بازار."""
    rets = {}
    t_start = grid[start_j]
    for sym, f in data.items():
        ts, c = f["15m"]["ts"], f["15m"]["c"]
        i0 = int(np.searchsorted(ts, t_start))
        if i0 < len(ts) and ts[i0] == t_start and c[i0] > 0:
            rets[sym] = float(c[-1] / c[i0] - 1)
    med = float(np.median(list(rets.values()))) if rets else 0.0
    return {"btc_pct": round(rets.get(BTC, 0.0) * 100, 2),
            "median_pct": round(med * 100, 2), "symbols": len(rets)}


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
    if "1d" in data[BTC]:
        # کندل روزانه‌ی بسته‌شده تا لحظه‌ی تصمیم (همان قاعده‌ی هم‌ترازی ۴ساعته و ۱ساعته)
        aligns[BTC]["1d"] = np.searchsorted(data[BTC]["1d"]["ts"], grid + M15 - M.TF_MS["1d"],
                                            side="right") - 1
    print("  محاسبه‌ی فیلتر BTC برای هر کندل ۱۵دقیقه‌ای (ربات ۶)...")
    btc_filters = btc_filters_series(data, aligns, grid)
    n_ok = sum(1 for x in btc_filters if x)
    print(f"  فیلتر BTC فعال در {n_ok} از {len(grid)} کندل")

    market = market_move(data, grid, start_j)
    print(f"  بازار در بازه: BTC {market['btc_pct']:+.2f}% | میانه‌ی نمادها {market['median_pct']:+.2f}%")

    stats = []
    for bot in M.BOTS:
        print(f"▶ backtest {bot.name} ...")
        res = simulate(bot, data, grid, aligns, btc4, btc1, start_j, btc_filters)
        st = summarize(bot, res)
        st["baseline"] = random_baseline(res, data, grid, aligns, start_j)
        stats.append(st)
        print(f"  {bot.name}: taken={st['signals_taken']} closed={st['closed']} "
              f"win={st['win_rate']} avgR={st['avg_R_net']} totalR={st['total_R_net']}")

    report = format_report(stats, market)
    print(report)
    with open(RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump({"days": days, "fee": FEE_PER_SIDE, "slip": SLIP, "market": market, "stats": stats},
                  fh, ensure_ascii=False)
    if send:
        M.send_telegram(report)
    return stats


if __name__ == "__main__":
    run(M.create_exchange(), send=True)
