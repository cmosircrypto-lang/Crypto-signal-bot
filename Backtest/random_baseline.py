#!/usr/bin/env python3
"""
آزمون مبنا با ورود تصادفی (Monte Carlo)

برای هر ربات، همان معاملات واقعیِ trades.csv را می‌گیرد و با «همان فاصله‌ی SL و TP و همان نماد»
ولی ورود تصادفی شبیه‌سازی می‌کند. سپس میانگین R واقعی را با توزیع تصادفی مقایسه می‌کند.

  A) زمان تصادفی + جهت تصادفی   → آیا منطق ورود (زمان و جهت) چیزی بیشتر از شانس دارد؟
  B) همان زمان واقعی + جهت تصادفی → آیا فقط «جهت» اطلاعات دارد؟

p-value = نسبت شبیه‌سازی‌هایی که میانگین R آن‌ها >= میانگین R واقعی است (هرچه کوچک‌تر، لبه‌ی واقعی‌تر).
"""
import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import backtest_bots as B

HORIZON = 14 * 96            # حداکثر ۱۴ روز نگهداری در شبیه‌سازی (کندل 15m)
BAR = 15 * 60 * 1000


class SymData:
    def __init__(self, df):
        self.ts = df["timestamp"].to_numpy(dtype="int64")
        self.h = df["high"].to_numpy(dtype=float)
        self.l = df["low"].to_numpy(dtype=float)
        self.c = df["close"].to_numpy(dtype=float)


def simulate(sd, t_ms, direction, sl_pct, tp_pct):
    """ورود با close آخرین کندل بسته‌شده تا t_ms. خروجی: R یا None (اگر در افق تعیین‌تکلیف نشد)."""
    i0 = int(np.searchsorted(sd.ts, t_ms, side="left"))     # اولین کندلی که ts >= t
    if i0 < 1 or i0 >= len(sd.ts) - 1:
        return None
    entry = sd.c[i0 - 1]
    hi, lo, cl = sd.h[i0:i0 + HORIZON], sd.l[i0:i0 + HORIZON], sd.c[i0:i0 + HORIZON]
    if direction == "LONG":
        sl, tp = entry * (1 - sl_pct), entry * (1 + tp_pct)
        hit_sl, hit_tp = lo <= sl, hi >= tp
    else:
        sl, tp = entry * (1 + sl_pct), entry * (1 - tp_pct)
        hit_sl, hit_tp = hi >= sl, lo <= tp
    any_hit = hit_sl | hit_tp
    if not any_hit.any():
        return None
    k = int(np.argmax(any_hit))
    if hit_sl[k] and hit_tp[k]:
        win = abs(cl[k] - tp) < abs(cl[k] - sl)
    else:
        win = bool(hit_tp[k])
    return tp_pct / sl_pct if win else -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="backtest_results/trades.csv")
    ap.add_argument("--exchange", default="okx", choices=["okx", "toobit"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--end", required=True)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="report_random.txt")
    args = ap.parse_args()

    B.EXCHANGE["name"] = args.exchange
    data_dir = args.data_dir or f"data/{args.exchange}"
    end_ms = int(datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    start_ms = end_ms - args.days * 86400_000
    t_hi = end_ms - 2 * 86400_000          # آخرین زمان مجاز ورود تصادفی

    tr = pd.read_csv(args.trades)
    if tr.empty:
        print("trades.csv خالی است"); sys.exit(0)
    tr["sl_pct"] = (tr["entry"] - tr["sl"]).abs() / tr["entry"]
    tr["tp_pct"] = (tr["tp"] - tr["entry"]).abs() / tr["entry"]
    tr["R"] = np.where(tr["result"] == "TP", tr["tp_pct"] / tr["sl_pct"], -1.0)

    sdata = {}
    for sym in tr["symbol"].unique():
        df = B.get_df(data_dir, sym, "15m")
        if df is not None:
            sdata[sym] = SymData(df)
    tr = tr[tr["symbol"].isin(sdata)].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    lines = ["", "آزمون ورود تصادفی (Monte Carlo)", "━" * 40,
             f"شبیه‌سازی: {args.sims} بار | همان SL%/TP% و نماد هر معامله‌ی واقعی",
             "A = زمان+جهت تصادفی | B = همان زمان واقعی، جهت تصادفی",
             "p = احتمال اینکه ورود تصادفی حداقل به اندازه‌ی ربات خوب باشد", ""]

    groups = [(b.replace("crypto-signal-bot-", ""), g) for b, g in tr.groupby("bot")] + [("ALL", tr)]
    for name, g in groups:
        rows = list(g.itertuples(index=False))
        real = float(g["R"].mean())
        resA, resB = [], []
        for _ in range(args.sims):
            ra, rb = [], []
            for r in rows:
                sd = sdata[r.symbol]
                lo_t = max(start_ms, int(sd.ts[0]) + 30 * 86400_000)
                t = lo_t + int(rng.integers(0, max(1, (t_hi - lo_t) // 3600_000))) * 3600_000
                x = simulate(sd, t, "LONG" if rng.random() < .5 else "SHORT", r.sl_pct, r.tp_pct)
                if x is not None: ra.append(x)
                y = simulate(sd, int(r.opened_ms), "LONG" if rng.random() < .5 else "SHORT", r.sl_pct, r.tp_pct)
                if y is not None: rb.append(y)
            if ra: resA.append(np.mean(ra))
            if rb: resB.append(np.mean(rb))
        resA, resB = np.array(resA), np.array(resB)
        pA = float((resA >= real).mean()) if len(resA) else float("nan")
        pB = float((resB >= real).mean()) if len(resB) else float("nan")
        lines.append(f"🤖 {name} | معاملات: {len(g)} | میانگین R واقعی: {real:+.3f}")
        lines.append(f"   A: تصادفی {resA.mean():+.3f} ± {resA.std():.3f} | p = {pA:.3f}")
        lines.append(f"   B: تصادفی {resB.mean():+.3f} ± {resB.std():.3f} | p = {pB:.3f}")
        lines.append("")
    lines.append("━" * 40)
    text = "\n".join(lines)
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
