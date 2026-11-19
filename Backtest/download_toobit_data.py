#!/usr/bin/env python3
"""
دانلود داده‌ی کندل 15m از Toobit (فیوچرز دائمی USDT-M) و ذخیره‌ی دائمی روی دیسک.

- Endpoint: GET https://api.toobit.com/quote/v1/klines  (حداکثر ۱۰۰۰ کندل در هر درخواست)
- نمادها همان فرمت ربات‌ها هستند (BTC-SWAP-USDT)؛ نیازی به نگاشت نیست.
- هر نماد در یک فایل جدا ذخیره می‌شود: data/toobit/15m/BTC-SWAP-USDT.csv.gz
- اجرای دوباره فقط بخش‌های ناقص (قدیمی‌تر/جدیدتر) را می‌گیرد.
- در پایان برای هر نماد تاریخ اولین کندل چاپ می‌شود تا عمق تاریخچه‌ی Toobit مشخص شود.

مثال:
    python download_toobit_data.py --end 2026-11-17
    python download_toobit_data.py --end 2026-11-17 --symbols BTC-SWAP-USDT ETH-SWAP-USDT
"""
import argparse
import ast
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

BASE = os.environ.get("TOOBIT_BASE", "https://api.toobit.com")
BAR_MS = {"5m": 300_000, "15m": 900_000}
COLS = ["ts", "open", "high", "low", "close", "volume"]
MIN_INTERVAL = 0.15          # فاصله‌ی حداقل بین درخواست‌ها (ثانیه)
PAGE = 1000


def load_symbols_from_bots_file(path):
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SYMBOLS" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f"SYMBOLS در {path} پیدا نشد")


_last_call = [0.0]


class NoData(Exception):
    pass


def tb_get(session, path, params, retries=8):
    for attempt in range(retries):
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = session.get(BASE + path, params=params, timeout=25)
            if r.status_code in (418, 429):
                time.sleep(2 * (attempt + 1))
                continue
            try:
                j = r.json()
            except ValueError:
                r.raise_for_status()
                raise
            if isinstance(j, dict) and "code" in j and j.get("code") not in (0, 200, "0", "200"):
                if int(j["code"]) == -1128:          # No data
                    raise NoData()
                raise RuntimeError(f"Toobit error {j.get('code')}: {j.get('msg')}")
            r.raise_for_status()
            return j
        except NoData:
            raise
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Toobit: تعداد تلاش‌ها تمام شد")


def fetch_range(session, sym, bar, start_ms, end_ms):
    """
    کندل‌های کاملاً بسته‌شده با ts در [start_ms, end_ms) را می‌گیرد.
    Toobit وقتی endTime داده شود، آخرین ۱۰۰۰ کندل قبل از آن را برمی‌گرداند
    (نه ۱۰۰۰ کندل بعد از startTime)؛ پس از جدید به قدیم صفحه‌بندی می‌کنیم.
    """
    bar_ms = BAR_MS[bar]
    rows, e = [], end_ms
    now_ms = int(time.time() * 1000)
    while e > start_ms:
        try:
            data = tb_get(session, "/quote/v1/klines",
                          {"symbol": sym, "interval": bar, "startTime": start_ms,
                           "endTime": e - 1, "limit": PAGE})
        except NoData:
            data = []
        if not isinstance(data, list):
            raise RuntimeError(f"پاسخ نامعتبر: {str(data)[:120]}")
        if not data:
            break                           # قدیمی‌تر از این داده‌ای وجود ندارد
        data.sort(key=lambda r: int(r[0]))
        for r in data:
            ts = int(r[0])
            if start_ms <= ts < end_ms and ts + bar_ms <= now_ms:
                rows.append((ts, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        oldest = int(data[0][0])
        if oldest >= e:
            break                           # پیشرفتی نشد
        e = oldest                          # صفحه‌ی بعد: قبل از قدیمی‌ترین کندل این صفحه
        if len(data) < PAGE and oldest <= start_ms + bar_ms:
            break
    return rows


def read_cached(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    try:
        return pd.read_csv(path)[COLS]
    except Exception as e:
        print(f"   ⚠️ فایل کش خراب است ({e}) — از نو دانلود می‌شود")
        return pd.DataFrame(columns=COLS)


def save_cached(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def update_symbol(session, sym, bar, start_ms, end_ms, out_dir):
    bar_ms = BAR_MS[bar]
    path = os.path.join(out_dir, bar, f"{sym}.csv.gz")
    old = read_cached(path)
    new_rows = []
    if old.empty:
        new_rows += fetch_range(session, sym, bar, start_ms, end_ms)
    else:
        lo, hi = int(old["ts"].min()), int(old["ts"].max())
        if start_ms < lo:
            new_rows += fetch_range(session, sym, bar, start_ms, lo)
        if end_ms > hi + bar_ms:
            new_rows += fetch_range(session, sym, bar, hi + bar_ms, end_ms)
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
    ap = argparse.ArgumentParser(description="دانلود و کش داده‌ی Toobit")
    ap.add_argument("--bots-file", default="backtest_bots.py")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--bar", default="15m", choices=list(BAR_MS))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--warmup-days", type=int, default=60)
    ap.add_argument("--end", default=None, help="تاریخ پایان UTC (YYYY-MM-DD)")
    ap.add_argument("--out", default="data/toobit")
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
    session.headers["User-Agent"] = "toobit-backtest-downloader/1.0"

    listed = None
    try:
        info = tb_get(session, "/api/v1/exchangeInfo", {})
        contracts = info.get("contracts", []) if isinstance(info, dict) else []
        listed = {c["symbol"] for c in contracts if c.get("symbol") and not c.get("inverse")}
        if listed:
            print(f"Toobit: {len(listed)} قرارداد USDT-M")
        else:
            listed = None
    except Exception as e:
        print(f"⚠️ دریافت لیست قراردادها ناموفق بود ({e}) — همه‌ی نمادها امتحان می‌شوند")

    missing, failed, short, t0 = [], [], [], time.time()
    for i, sym in enumerate(symbols, 1):
        if listed is not None and sym not in listed:
            print(f"[{i}/{len(symbols)}] {sym}: در Toobit وجود ندارد — رد شد")
            missing.append(sym)
            continue
        try:
            df, added = update_symbol(session, sym, args.bar, start_ms, end_ms, args.out)
        except KeyboardInterrupt:
            print("\n⏹ متوقف شد. دوباره اجرا کن تا ادامه بدهد.")
            sys.exit(1)
        except Exception as e:
            print(f"[{i}/{len(symbols)}] {sym}: ❌ خطا: {e}")
            failed.append(sym)
            continue
        if df.empty:
            print(f"[{i}/{len(symbols)}] {sym}: بدون داده")
            short.append(sym)
            continue
        if int(df["ts"].min()) > start_ms + 7 * 86400_000:
            short.append(sym)
        elapsed = time.time() - t0
        eta = elapsed / i * (len(symbols) - i)
        tag = f"+{added:,} جدید" if added else "از کش (بدون دانلود)"
        print(f"[{i}/{len(symbols)}] {sym}: {tag} | {describe(df, start_ms, end_ms, bar_ms)} | ETA {eta/60:.0f}m")

    print("=" * 60)
    if missing:
        print(f"نمادهای موجود نبودن در Toobit ({len(missing)}): {', '.join(missing)}")
    if short:
        print(f"نمادهایی که تاریخچه‌شان از ابتدای بازه شروع نمی‌شود ({len(short)}): {', '.join(short)}")
    if failed:
        print(f"نمادهای ناموفق ({len(failed)}) — دوباره اجرا کن: {', '.join(failed)}")
    print("تمام شد.")


if __name__ == "__main__":
    main()
