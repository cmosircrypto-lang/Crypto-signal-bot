#!/usr/bin/env python3
"""
دانلود داده‌ی کندل از OKX (فیوچرز دائمی USDT) و ذخیره‌ی دائمی روی دیسک.

- فقط یک تایم‌فریم پایه (پیش‌فرض 15m) دانلود می‌شود؛ تایم‌فریم‌های 1h/4h/1d
  در بک‌تست از روی همین داده ساخته می‌شوند (resample دقیق).
- لیست نمادها مستقیماً از فایل ربات‌ها (SYMBOLS) خوانده می‌شود.
- هر نماد بعد از دانلود در فایل جدا ذخیره می‌شود: data/okx/15m/BTC-USDT-SWAP.csv.gz
- اجرای دوباره، فقط بخش‌های ناقص (قدیمی‌تر یا جدیدتر از فایل موجود) را دانلود می‌کند؛
  اگر همه‌چیز کامل باشد هیچ درخواستی به OKX زده نمی‌شود.
- اگر وسط کار قطع شد، دوباره اجرا کن؛ از همان‌جا ادامه می‌دهد.

مثال:
    python download_okx_data.py                      # ۳۶۵ روز + ۶۰ روز warmup تا امروز (UTC)
    python download_okx_data.py --end 2026-11-17     # پایان ثابت (برای نتایج تکرارپذیر)
    python download_okx_data.py --symbols BTC-SWAP-USDT ETH-SWAP-USDT   # فقط چند نماد
"""
import argparse
import ast
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

BASE = "https://www.okx.com"
BAR_MS = {
    "5m": 300_000,     # دقیق‌تر برای تشخیص TP/SL ولی حدود ۳ برابر زمان دانلود
    "15m": 900_000,    # پیش‌فرض
}
COLS = ["ts", "open", "high", "low", "close", "volume"]
MIN_INTERVAL = 0.12          # حدود ۸ درخواست در ثانیه (سقف OKX برای این endpoint: 20 درخواست / 2 ثانیه)

# نمادهایی که در Toobit با پیشوند 1000 هستند ولی در OKX بدون آن معامله می‌شوند.
# (قیمت‌ها فقط یک ضریب ثابت دارند و استراتژی‌ها نسبت به مقیاس بی‌تفاوت‌اند)
OKX_BASE_ALIASES = {
    "1000PEPE": "PEPE", "1000SHIB": "SHIB", "1000BONK": "BONK",
    "1000FLOKI": "FLOKI", "1000SATS": "SATS", "1000RATS": "RATS",
    "1000000MOG": "MOG",
}


def to_okx_inst(symbol):
    """BTC-SWAP-USDT  ->  BTC-USDT-SWAP"""
    base = symbol.split("-")[0]
    base = OKX_BASE_ALIASES.get(base, base)
    return f"{base}-USDT-SWAP"


def load_symbols_from_bots_file(path):
    """SYMBOLS را بدون import کردن فایل ربات‌ها از روی کد می‌خواند."""
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SYMBOLS" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f"SYMBOLS در {path} پیدا نشد")


_last_call = [0.0]


def okx_get(session, path, params, retries=8):
    for attempt in range(retries):
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = session.get(BASE + path, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            j = r.json()
            code = j.get("code")
            if code == "0":
                return j.get("data", [])
            if code == "50011":            # Too many requests
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"OKX error {code}: {j.get('msg')}")
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("OKX: تعداد تلاش‌ها تمام شد")


def fetch_range(session, inst, bar, start_ms, end_ms):
    """کندل‌های بسته‌شده با ts در بازه‌ی [start_ms, end_ms) را از جدید به قدیم می‌گیرد."""
    rows = []
    after = end_ms
    while True:
        data = okx_get(session, "/api/v5/market/history-candles",
                       {"instId": inst, "bar": bar, "limit": "100", "after": str(after)})
        if not data:
            break
        for r in data:
            ts = int(r[0])
            if start_ms <= ts < end_ms and r[8] == "1":
                vol = r[6] if r[6] not in ("", None) else r[5]     # volCcy (به واحد کوین)
                rows.append((ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(vol)))
        oldest = int(data[-1][0])
        if oldest <= start_ms or oldest >= after:
            break
        after = oldest
    return rows


def read_cached(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    try:
        df = pd.read_csv(path)
        return df[COLS]
    except Exception as e:
        print(f"   ⚠️ فایل کش خراب است ({e}) — از نو دانلود می‌شود")
        return pd.DataFrame(columns=COLS)


def save_cached(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def update_symbol(session, inst, bar, start_ms, end_ms, out_dir):
    bar_ms = BAR_MS[bar]
    path = os.path.join(out_dir, bar, f"{inst}.csv.gz")
    old = read_cached(path)
    new_rows = []
    if old.empty:
        new_rows += fetch_range(session, inst, bar, start_ms, end_ms)
    else:
        lo, hi = int(old["ts"].min()), int(old["ts"].max())
        if start_ms < lo:                                   # بخش قدیمی‌تر
            new_rows += fetch_range(session, inst, bar, start_ms, lo)
        if end_ms > hi + bar_ms:                            # بخش جدیدتر
            new_rows += fetch_range(session, inst, bar, hi + bar_ms, end_ms)
    added = len(new_rows)
    if added:
        df = pd.concat([old, pd.DataFrame(new_rows, columns=COLS)], ignore_index=True)
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        save_cached(path, df)
    else:
        df = old
    return df, added


def describe(df, start_ms, end_ms, bar_ms):
    if df.empty:
        return "بدون داده"
    lo, hi = int(df["ts"].min()), int(df["ts"].max())
    first = max(lo, start_ms)
    expected = (min(hi + bar_ms, end_ms) - first) // bar_ms
    have = int(((df["ts"] >= first) & (df["ts"] < end_ms)).sum())
    cov = 100.0 * have / expected if expected > 0 else 0.0
    d0 = datetime.fromtimestamp(lo / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    d1 = datetime.fromtimestamp(hi / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return f"{d0} → {d1} | {have:,} کندل | پوشش {cov:.2f}%"


def main():
    ap = argparse.ArgumentParser(description="دانلود و کش داده‌ی OKX")
    ap.add_argument("--bots-file", default="crypto_signal_bots_lite.py")
    ap.add_argument("--symbols", nargs="*", default=None, help="لیست نماد (فرمت ربات‌ها). پیش‌فرض: همه")
    ap.add_argument("--bar", default="15m", choices=list(BAR_MS))
    ap.add_argument("--days", type=int, default=365, help="طول بازه‌ی بک‌تست")
    ap.add_argument("--warmup-days", type=int, default=60, help="داده‌ی اضافه قبل از شروع برای محاسبه اندیکاتورها")
    ap.add_argument("--end", default=None, help="تاریخ پایان UTC (YYYY-MM-DD). پیش‌فرض: امروز")
    ap.add_argument("--out", default="data/okx")
    args = ap.parse_args()

    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=args.days + args.warmup_days)
    end_ms, start_ms = int(end_dt.timestamp() * 1000), int(start_dt.timestamp() * 1000)
    bar_ms = BAR_MS[args.bar]

    symbols = args.symbols or load_symbols_from_bots_file(args.bots_file)
    print(f"بازه: {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d} (UTC) | تایم‌فریم: {args.bar} | {len(symbols)} نماد")
    print(f"پوشه‌ی ذخیره: {os.path.abspath(args.out)}")

    session = requests.Session()
    session.headers["User-Agent"] = "okx-backtest-downloader/1.0"

    listed = None
    try:
        data = okx_get(session, "/api/v5/public/instruments", {"instType": "SWAP"})
        listed = {d["instId"] for d in data if d.get("state") == "live"}
        print(f"OKX: {len(listed)} قرارداد SWAP فعال")
    except Exception as e:
        print(f"⚠️ دریافت لیست قراردادها ناموفق بود ({e}) — همه‌ی نمادها امتحان می‌شوند")

    missing, failed, t0 = [], [], time.time()
    for i, sym in enumerate(symbols, 1):
        inst = to_okx_inst(sym)
        if listed is not None and inst not in listed:
            print(f"[{i}/{len(symbols)}] {inst}: در OKX وجود ندارد — رد شد")
            missing.append(sym)
            continue
        try:
            df, added = update_symbol(session, inst, args.bar, start_ms, end_ms, args.out)
        except KeyboardInterrupt:
            print("\n⏹ متوقف شد. دوباره اجرا کن تا ادامه بدهد (نمادهای کامل‌شده دوباره دانلود نمی‌شوند).")
            sys.exit(1)
        except Exception as e:
            print(f"[{i}/{len(symbols)}] {inst}: ❌ خطا: {e}")
            failed.append(sym)
            continue
        elapsed = time.time() - t0
        eta = elapsed / i * (len(symbols) - i)
        tag = f"+{added:,} جدید" if added else "از کش (بدون دانلود)"
        print(f"[{i}/{len(symbols)}] {inst}: {tag} | {describe(df, start_ms, end_ms, bar_ms)} | ETA {eta/60:.0f}m")

    print("=" * 60)
    if missing:
        print(f"نمادهای موجود نبودن در OKX ({len(missing)}): {', '.join(missing)}")
    if failed:
        print(f"نمادهای ناموفق ({len(failed)}) — دوباره اجرا کن: {', '.join(failed)}")
    print("تمام شد.")


if __name__ == "__main__":
    main()
