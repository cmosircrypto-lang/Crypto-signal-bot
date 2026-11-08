#!/usr/bin/env python3
"""
مقایسه خودکار عملکرد ۷ ربات بر اساس فایل‌های CSV معاملات بسته‌شده

فایل‌های مورد انتظار (در همان پوشه اجرا):
  trades_bot.csv
  trades_bot-v10.csv
  trades_bot-v20.csv
  trades_bot-v30.csv
  trades_bot-v40.csv
  trades_bot-v50.csv
  trades_bot-v60.csv

استفاده:
  python compare_bots.py
  python compare_bots.py --min-trades 50
  python compare_bots.py --export ranking.csv
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import pandas as pd
import numpy as np


# نام‌های پیش‌فرض مطابق append_trade_csv در main.py
BOT_FILES = [
    ("crypto-signal-bot", "trades_bot.csv", "EMA Cross"),
    ("crypto-signal-bot-v10", "trades_bot-v10.csv", "Breakout"),
    ("crypto-signal-bot-v20", "trades_bot-v20.csv", "MTF Pullback"),
    ("crypto-signal-bot-v30", "trades_bot-v30.csv", "AMD Session"),
    ("crypto-signal-bot-v40", "trades_bot-v40.csv", "Regime ADX"),
    ("crypto-signal-bot-v50", "trades_bot-v50.csv", "SMC"),
    ("crypto-signal-bot-v60", "trades_bot-v60.csv", "BTC Filter"),
]


def load_trades(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ خطا در خواندن {path}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    # نرمال‌سازی
    if "pnl_r" in df.columns:
        df["pnl_r"] = pd.to_numeric(df["pnl_r"], errors="coerce")
    else:
        df["pnl_r"] = np.nan

    # اگر pnl_r خالی بود از result بساز
    if df["pnl_r"].isna().any() and "result" in df.columns:
        risk_ok = (
            pd.to_numeric(df.get("entry"), errors="coerce")
            - pd.to_numeric(df.get("sl"), errors="coerce")
        ).abs()
        tp_r = (
            pd.to_numeric(df.get("tp"), errors="coerce")
            - pd.to_numeric(df.get("entry"), errors="coerce")
        ).abs() / risk_ok.replace(0, np.nan)

        mask = df["pnl_r"].isna()
        df.loc[mask & (df["result"] == "TP"), "pnl_r"] = tp_r[mask & (df["result"] == "TP")]
        df.loc[mask & (df["result"] == "SL"), "pnl_r"] = -1.0

    df = df.dropna(subset=["pnl_r"])
    return df


def metrics(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_win_streak": 0,
            "max_lose_streak": 0,
            "max_drawdown_r": 0.0,
        }

    pnl = df["pnl_r"].astype(float).values
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    total_r = float(np.sum(pnl))
    avg_r = float(np.mean(pnl))
    wr = float(len(wins) / len(pnl) * 100)

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(np.abs(losses.sum())) if len(losses) else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    # streaks
    max_w = max_l = cur_w = cur_l = 0
    for x in pnl:
        if x > 0:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)

    # equity curve drawdown in R
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    return {
        "trades": len(pnl),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": round(wr, 1),
        "total_r": round(total_r, 2),
        "avg_r": round(avg_r, 3),
        "profit_factor": round(pf, 2) if pf != float("inf") else 999.0,
        "expectancy": round(avg_r, 3),
        "max_win_streak": max_w,
        "max_lose_streak": max_l,
        "max_drawdown_r": round(max_dd, 2),
    }


def score_bot(m: dict, min_trades: int) -> float:
    """
    امتیاز ترکیبی برای رتبه‌بندی.
    تا قبل از min_trades امتیاز کاهش می‌یابد (نمونه آماری کم).
    """
    if m["trades"] == 0:
        return -999.0

    # نمونه آماری
    sample_factor = min(1.0, m["trades"] / max(min_trades, 1))

    # مؤلفه‌ها
    # Total R مهم‌ترین است، بعد expectancy، بعد PF، بعد win rate
    s = (
        0.40 * m["total_r"]
        + 0.25 * (m["avg_r"] * 10)          # مقیاس
        + 0.20 * min(m["profit_factor"], 5)  # سقف برای PF خیلی بالا
        + 0.10 * (m["win_rate"] / 20.0)
        + 0.05 * max(m["max_drawdown_r"], -20) / 5.0  # جریمه دراودان
    )
    return round(s * sample_factor, 3)


def main():
    parser = argparse.ArgumentParser(description="مقایسه عملکرد ربات‌ها از روی CSV")
    parser.add_argument("--dir", default=".", help="پوشه فایل‌های CSV")
    parser.add_argument("--min-trades", type=int, default=100,
                        help="حداقل معامله برای اعتبار کامل رتبه (پیش‌فرض ۱۰۰)")
    parser.add_argument("--export", default="", help="خروجی ranking به CSV")
    parser.add_argument("--show-all", action="store_true", help="نمایش ربات‌های بدون معامله هم")
    args = parser.parse_args()

    base = Path(args.dir)
    rows = []

    print("=" * 72)
    print("📊 مقایسه خودکار ربات‌ها")
    print(f"حداقل معامله برای اعتبار کامل: {args.min_trades}")
    print("=" * 72)

    for bot_id, fname, label in BOT_FILES:
        fpath = base / fname
        df = load_trades(str(fpath))
        m = metrics(df)
        m["bot"] = bot_id
        m["label"] = label
        m["file"] = fname
        m["exists"] = fpath.exists()
        m["score"] = score_bot(m, args.min_trades)
        m["ready"] = m["trades"] >= args.min_trades
        rows.append(m)

        status = "✅ READY" if m["ready"] else (f"⏳ {m['trades']}/{args.min_trades}" if m["trades"] else "❌ no data")
        if m["trades"] == 0 and not args.show_all:
            print(f"\n{label:16} | {status}")
            continue

        print(f"\n🤖 {label} ({bot_id})")
        print(f"   فایل          : {fname} {'(موجود)' if m['exists'] else '(نیست)'}")
        print(f"   معاملات       : {m['trades']}  |  برد: {m['wins']}  |  باخت: {m['losses']}")
        print(f"   Win Rate      : {m['win_rate']}%")
        print(f"   Total R       : {m['total_r']:+.2f}")
        print(f"   Avg R         : {m['avg_r']:+.3f}")
        print(f"   Profit Factor : {m['profit_factor']}")
        print(f"   Max DD (R)    : {m['max_drawdown_r']}")
        print(f"   Streak W/L    : {m['max_win_streak']} / {m['max_lose_streak']}")
        print(f"   Score         : {m['score']}  |  {status}")

    # Ranking
    ranked = sorted(rows, key=lambda x: x["score"], reverse=True)
    print("\n" + "=" * 72)
    print("🏆 رتبه‌بندی (بالاتر = بهتر)")
    print("=" * 72)
    print(f"{'#':<3} {'ربات':<16} {'n':>4} {'WR%':>6} {'TotalR':>8} {'AvgR':>7} {'PF':>6} {'Score':>8} {'وضعیت'}")
    print("-" * 72)

    for i, m in enumerate(ranked, 1):
        if m["trades"] == 0 and not args.show_all:
            continue
        st = "READY" if m["ready"] else f"{m['trades']}/{args.min_trades}"
        print(
            f"{i:<3} {m['label']:<16} {m['trades']:>4} {m['win_rate']:>6.1f} "
            f"{m['total_r']:>+8.2f} {m['avg_r']:>+7.3f} {m['profit_factor']:>6.2f} "
            f"{m['score']:>8.3f} {st}"
        )

    ready = [m for m in ranked if m["ready"]]
    if ready:
        best = ready[0]
        print("\n" + "=" * 72)
        print(f"⭐ پیشنهاد ربات اصلی: {best['label']} ({best['bot']})")
        print(f"   Total R={best['total_r']:+.2f} | WR={best['win_rate']}% | "
              f"PF={best['profit_factor']} | n={best['trades']}")
        print("=" * 72)
    else:
        print(f"\n⏳ هنوز هیچ رباتی به {args.min_trades} معامله نرسیده.")
        if any(m["trades"] > 0 for m in ranked):
            interim = max((m for m in ranked if m["trades"] > 0), key=lambda x: x["score"])
            print(f"   پیشرو فعلی (نمونه ناکافی): {interim['label']} "
                  f"(n={interim['trades']}, score={interim['score']})")

    if args.export:
        out = pd.DataFrame(ranked)
        cols = ["label", "bot", "trades", "wins", "losses", "win_rate",
                "total_r", "avg_r", "profit_factor", "max_drawdown_r",
                "max_win_streak", "max_lose_streak", "score", "ready", "file"]
        out = out[[c for c in cols if c in out.columns]]
        out.to_csv(args.export, index=False)
        print(f"\n💾 خروجی ذخیره شد: {args.export}")


if __name__ == "__main__":
    main()
