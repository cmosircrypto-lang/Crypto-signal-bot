#!/usr/bin/env python3
"""Compare Run #35 trades with BTC trend and evaluate post-entry trend filters."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RUN35_ARTIFACT = Path("run35-results.zip")
DATA_ROOT = Path("Backtest/data/okx")
TRADE_SUFFIX = "trades_risk_0.05_rr_1.5.csv"
TIMEFRAME_MS = {"1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}
INITIAL_CAPITAL = 100.0


def load_run35_trades(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        matches = [n for n in archive.namelist() if n.endswith(TRADE_SUFFIX)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one baseline trade ledger ending {TRADE_SUFFIX}; found {matches}")
        with archive.open(matches[0]) as stream:
            trades = pd.read_csv(stream)
    required = {"symbol", "direction", "opened_at", "closed_at", "pnl"}
    missing = required - set(trades.columns)
    if missing:
        raise RuntimeError(f"Run #35 trade ledger missing columns: {sorted(missing)}")
    trades["opened_at"] = pd.to_numeric(trades["opened_at"], errors="raise").astype("int64")
    trades["closed_at"] = pd.to_numeric(trades["closed_at"], errors="raise").astype("int64")
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="raise")
    trades["direction"] = trades["direction"].astype(str).str.upper()
    trades = trades[trades["direction"].isin(["BUY", "SELL"])].copy()
    if trades.empty:
        raise RuntimeError("No BUY/SELL trades found in Run #35 baseline ledger")
    return trades


def btc_trend_at_entries(timeframe: str, entry_times: np.ndarray) -> np.ndarray:
    if timeframe == "1d":
        hourly_path = DATA_ROOT / "1h" / "BTC-USDT-SWAP.csv.gz"
        if not hourly_path.is_file():
            candidates = sorted((DATA_ROOT / "1h").glob("*BTC*USDT*SWAP*.csv.gz"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"Cannot uniquely locate BTC 1h data for daily aggregation; candidates={candidates}")
            hourly_path = candidates[0]
        hourly = pd.read_csv(hourly_path, compression="gzip")
        required = {"ts", "close"}
        if not required.issubset(hourly.columns):
            raise RuntimeError(f"{hourly_path} must contain ts and close")
        hourly["ts"] = pd.to_numeric(hourly["ts"], errors="raise").astype("int64")
        hourly["close"] = pd.to_numeric(hourly["close"], errors="raise")
        hourly = hourly.sort_values("ts").drop_duplicates("ts")
        hourly["utc_day"] = (hourly["ts"] // TIMEFRAME_MS["1d"]) * TIMEFRAME_MS["1d"]
        rows = []
        for day, group in hourly.groupby("utc_day", sort=True):
            expected = day + np.arange(24, dtype=np.int64) * TIMEFRAME_MS["1h"]
            actual = group["ts"].to_numpy(dtype=np.int64)
            if len(actual) == 24 and np.array_equal(actual, expected):
                rows.append({"ts": int(day), "close": float(group.iloc[-1]["close"])})
        candles = pd.DataFrame(rows)
        if candles.empty:
            raise RuntimeError("No complete UTC daily BTC candles could be derived from 1h data")
        return _trend_states_from_candles(candles, "1d", entry_times)

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
    return _trend_states_from_candles(candles, timeframe, entry_times)


def _trend_states_from_candles(candles: pd.DataFrame, timeframe: str, entry_times: np.ndarray) -> np.ndarray:
    ema20 = candles["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = candles["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    close_times = candles["ts"].to_numpy(dtype=np.int64) + TIMEFRAME_MS[timeframe]
    close_values = candles["close"].to_numpy(dtype=float)
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


def alignment_counts(trades: pd.DataFrame, timeframe: str, states: np.ndarray) -> dict:
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


def performance_metrics(name: str, trades: pd.DataFrame) -> dict:
    ordered = trades.sort_values(["closed_at", "opened_at"]).copy()
    pnl = ordered["pnl"].astype(float)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    breakeven = int((pnl == 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(pnl[pnl < 0].sum())
    net_pnl = float(pnl.sum())
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Realized-equity curve: aggregate trades closed at the same timestamp.
    by_close = ordered.groupby("closed_at", sort=True)["pnl"].sum()
    equity = INITIAL_CAPITAL + by_close.cumsum()
    equity_with_start = np.concatenate(([INITIAL_CAPITAL], equity.to_numpy(dtype=float)))
    peaks = np.maximum.accumulate(equity_with_start)
    drawdowns = peaks - equity_with_start
    max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0
    drawdown_pct = np.divide(
        drawdowns, peaks, out=np.zeros_like(drawdowns), where=peaks != 0
    ) * 100.0

    directions = ordered["direction"]
    return {
        "filter": name,
        "trades": int(len(ordered)),
        "buy_trades": int((directions == "BUY").sum()),
        "sell_trades": int((directions == "SELL").sum()),
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate_pct": round(wins / len(ordered) * 100, 2) if len(ordered) else 0.0,
        "gross_profit": round(gross_profit, 8),
        "gross_loss": round(gross_loss, 8),
        "net_pnl": round(net_pnl, 8),
        "ending_equity": round(INITIAL_CAPITAL + net_pnl, 8),
        "profit_factor": round(profit_factor, 6) if np.isfinite(profit_factor) else "INF",
        "max_drawdown": round(max_drawdown, 8),
        "max_drawdown_pct": round(float(drawdown_pct.max()) if len(drawdown_pct) else 0.0, 4),
        "avg_pnl_per_trade": round(net_pnl / len(ordered), 8) if len(ordered) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=RUN35_ARTIFACT)
    parser.add_argument("--output", type=Path, default=Path("btc_trend_alignment_run35"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    trades = load_run35_trades(args.artifact)
    entry_times = trades["opened_at"].to_numpy(dtype=np.int64)
    details = trades.copy()
    alignment_summaries = []
    for timeframe in ("1d", "4h", "1h"):
        states = btc_trend_at_entries(timeframe, entry_times)
        details[f"btc_{timeframe}_trend"] = states
        alignment_summaries.append(alignment_counts(trades, timeframe, states))

    details["pass_1d_filter"] = details["btc_1d_trend"].to_numpy() == details["direction"].to_numpy()
    details["pass_4h_filter"] = details["btc_4h_trend"].to_numpy() == details["direction"].to_numpy()
    details["pass_1h_filter"] = details["btc_1h_trend"].to_numpy() == details["direction"].to_numpy()
    details["pass_1d_4h_filter"] = details["pass_1d_filter"] & details["pass_4h_filter"]
    details["pass_1d_1h_filter"] = details["pass_1d_filter"] & details["pass_1h_filter"]
    details["pass_4h_1h_filter"] = details["pass_4h_filter"] & details["pass_1h_filter"]
    details["pass_all_three_filters"] = details["pass_1d_filter"] & details["pass_4h_filter"] & details["pass_1h_filter"]

    filters = {
        "no_filter_baseline": np.ones(len(details), dtype=bool),
        "1d_aligned_only": details["pass_1d_filter"].to_numpy(dtype=bool),
        "4h_aligned_only": details["pass_4h_filter"].to_numpy(dtype=bool),
        "1h_aligned_only": details["pass_1h_filter"].to_numpy(dtype=bool),
        "1d_and_4h_aligned": details["pass_1d_4h_filter"].to_numpy(dtype=bool),
        "1d_and_1h_aligned": details["pass_1d_1h_filter"].to_numpy(dtype=bool),
        "4h_and_1h_aligned": details["pass_4h_1h_filter"].to_numpy(dtype=bool),
        "all_1d_4h_1h_aligned": details["pass_all_three_filters"].to_numpy(dtype=bool),
    }
    performance = []
    for name, mask in filters.items():
        selected = details.loc[mask].copy()
        performance.append(performance_metrics(name, selected))
        if name != "no_filter_baseline":
            selected.to_csv(args.output / f"filtered_trades_{name}.csv", index=False)

    alignment = pd.DataFrame(alignment_summaries)
    performance_df = pd.DataFrame(performance)
    alignment.to_csv(args.output / "btc_trend_alignment_summary.csv", index=False)
    performance_df.to_csv(args.output / "btc_filter_performance_summary.csv", index=False)
    details.to_csv(args.output / "btc_trend_alignment_trade_details.csv", index=False)

    lines = [
        "# Bitcoin trend alignment and filter test — Run #35",
        "",
        "Scope: executed and closed trades from the baseline scenario (risk 0.05%, RR 1.5). This is a historical post-filter of those executed trades, not a full portfolio re-simulation; removed trades are not replaced by other signals.",
        "",
        "Trend rule at entry time, using only fully closed BTC candles: BUY when close > EMA20 > EMA50; SELL when close < EMA20 < EMA50; otherwise NEUTRAL. EMA periods are 20 and 50. Daily candles are built from exactly 24 complete UTC 1h candles; incomplete days are excluded.",
        "",
        "## Alignment at entry",
        "",
        "| Timeframe | Trades | Aligned | Aligned % | Opposite | Opposite % | Neutral | Neutral % | Unavailable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in alignment_summaries:
        lines.append(
            f"| {row['timeframe']} | {row['trades']} | {row['aligned']} | {row['aligned_pct']}% | "
            f"{row['opposite']} | {row['opposite_pct']}% | {row['neutral']} | {row['neutral_pct']}% | {row['unavailable']} |"
        )
    lines += [
        "",
        "## Performance after applying filters",
        "",
        "| Filter | Trades | BUY | SELL | Wins | Losses | Win rate | Net PnL | Ending equity | Profit Factor | Max DD | Max DD % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in performance:
        lines.append(
            f"| {row['filter']} | {row['trades']} | {row['buy_trades']} | {row['sell_trades']} | "
            f"{row['wins']} | {row['losses']} | {row['win_rate_pct']}% | {row['net_pnl']} | "
            f"{row['ending_equity']} | {row['profit_factor']} | {row['max_drawdown']} | {row['max_drawdown_pct']}% |"
        )
    lines += [
        "",
        "Interpretation caution: PnL and equity are calculated by summing the original realized trade PnL values in close-time order, starting from $100. The original trade sizes are retained. This isolates the effect of excluding trades but does not recalculate position sizes from a filtered equity curve, free-slot replacement entries, or overlapping-position portfolio constraints.",
        "",
        "The requested comparisons are 1d alone, 1d+4h, 1d+1h, and all three timeframes. 4h alone, 1h alone, and 4h+1h are also retained as comparison controls. The detailed CSV contains every original trade, BTC trend states, and pass/fail flags for each filter. Separate CSV ledgers contain trades retained by each filter.",
        "Run #35 stores opened_at rather than original signal_ts; opened_at is used as the entry decision timestamp because the runner enters at the signal candle close.",
    ]
    report = "\n".join(lines) + "\n"
    (args.output / "README.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
