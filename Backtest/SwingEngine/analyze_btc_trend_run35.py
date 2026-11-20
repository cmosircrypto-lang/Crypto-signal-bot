#!/usr/bin/env python3
"""Compare Run #35 executed entries with BTC 1h/4h trend without lookahead."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RUN35_ARTIFACT = Path("run35-results.zip")
DATA_ROOT = Path("Backtest/data/okx")
TRADE_SUFFIX = "trades_risk_0.05_rr_1.5.csv"
TIMEFRAME_MS = {"1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000}


def load_run35_trades(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        matches = [n for n in archive.namelist() if n.endswith(TRADE_SUFFIX)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one baseline trade ledger ending {TRADE_SUFFIX}; found {matches}")
        with archive.open(matches[0]) as stream:
            trades = pd.read_csv(stream)
    required = {"symbol", "direction", "opened_at"}
    missing = required - set(trades.columns)
    if missing:
        raise RuntimeError(f"Run #35 trade ledger missing columns: {sorted(missing)}")
    trades["opened_at"] = pd.to_numeric(trades["opened_at"], errors="raise").astype("int64")
    trades["direction"] = trades["direction"].astype(str).str.upper()
    trades = trades[trades["direction"].isin(["BUY", "SELL"])].copy()
    if trades.empty:
        raise RuntimeError("No BUY/SELL trades found in Run #35 baseline ledger")
    return trades


def btc_trend_at_entries(timeframe: str, entry_times: np.ndarray) -> np.ndarray:
    path = DATA_ROOT / timeframe / "BTC-USDT-SWAP.csv.gz"
    if not path.is_file():
        candidates = sorted((DATA_ROOT / timeframe).glob("*BTC*USDT*SWAP*.csv.gz"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Cannot uniquely locate BTC-USDT-SWAP {timeframe} data; candidates={candidates}")
        path = candidates[0]

    candles = pd.read_csv(path, compression="gzip")
    required = {"ts", "close"}
    if not required.issubset(candles.columns):
        raise RuntimeError(f"{path} must contain ts and close")
    candles["ts"] = pd.to_numeric(candles["ts"], errors="raise").astype("int64")
    candles["close"] = pd.to_numeric(candles["close"], errors="raise")
    candles = candles.sort_values("ts").drop_duplicates("ts")
    ema20 = candles["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = candles["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    close_times = candles["ts"].to_numpy() + TIMEFRAME_MS[timeframe]
    close_values = candles["close"].to_numpy()
    e20 = ema20.to_numpy()
    e50 = ema50.to_numpy()

    indices = np.searchsorted(close_times, entry_times, side="right") - 1
    states = np.full(len(entry_times), "UNAVAILABLE", dtype=object)
    valid = indices >= 0
    idx = indices[valid]
    enough = np.isfinite(e20[idx]) & np.isfinite(e50[idx])
    valid_positions = np.flatnonzero(valid)
    states[valid_positions[~enough]] = "NEUTRAL"
    usable_positions = valid_positions[enough]
    usable_idx = idx[enough]
    bullish = (close_values[usable_idx] > e20[usable_idx]) & (e20[usable_idx] > e50[usable_idx])
    bearish = (close_values[usable_idx] < e20[usable_idx]) & (e20[usable_idx] < e50[usable_idx])
    states[usable_positions[bullish]] = "BUY"
    states[usable_positions[bearish]] = "SELL"
    states[usable_positions[~(bullish | bearish)]] = "NEUTRAL"
    return states


def summarize(trades: pd.DataFrame, timeframe: str, states: np.ndarray) -> dict:
    directions = trades["direction"].to_numpy()
    aligned = states == directions
    opposite = ((states == "BUY") & (directions == "SELL")) | ((states == "SELL") & (directions == "BUY"))
    neutral = states == "NEUTRAL"
    unavailable = states == "UNAVAILABLE"
    total = len(trades)
    return {
        "timeframe": timeframe,
        "trades": total,
        "aligned": int(aligned.sum()),
        "aligned_pct": round(float(aligned.mean() * 100), 2),
        "opposite": int(opposite.sum()),
        "opposite_pct": round(float(opposite.mean() * 100), 2),
        "neutral": int(neutral.sum()),
        "neutral_pct": round(float(neutral.mean() * 100), 2),
        "unavailable": int(unavailable.sum()),
        "unavailable_pct": round(float(unavailable.mean() * 100), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=RUN35_ARTIFACT)
    parser.add_argument("--output", type=Path, default=Path("btc_trend_alignment_run35"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    trades = load_run35_trades(args.artifact)
    entry_times = trades["opened_at"].to_numpy(dtype=np.int64)
    summaries = []
    details = trades.copy()
    for timeframe in ("4h", "1h"):
        states = btc_trend_at_entries(timeframe, entry_times)
        details[f"btc_{timeframe}_trend"] = states
        summaries.append(summarize(trades, timeframe, states))

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "btc_trend_alignment_summary.csv", index=False)
    details.to_csv(args.output / "btc_trend_alignment_trade_details.csv", index=False)

    lines = [
        "# Bitcoin trend alignment — Run #35",
        "",
        "Scope: executed and closed trades from the baseline scenario (risk 0.05%, RR 1.5), not every generated signal and not all nine scenario ledgers.",
        "",
        "Trend rule at entry time (using only fully closed BTC candles): BUY when close > EMA20 > EMA50; SELL when close < EMA20 < EMA50; otherwise NEUTRAL. EMA periods are 20 and 50.",
        "",
        "| Timeframe | Trades | Aligned | Aligned % | Opposite | Opposite % | Neutral | Neutral % | Unavailable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['timeframe']} | {row['trades']} | {row['aligned']} | {row['aligned_pct']}% | "
            f"{row['opposite']} | {row['opposite_pct']}% | {row['neutral']} | {row['neutral_pct']}% | {row['unavailable']} |"
        )
    lines += [
        "",
        "The detailed CSV contains each trade and the BTC trend classification on both timeframes.",
        "The Run #35 artifact stores opened_at rather than the original signal_ts; the runner fills at the signal candle close, so opened_at is used as the signal/entry decision timestamp.",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
