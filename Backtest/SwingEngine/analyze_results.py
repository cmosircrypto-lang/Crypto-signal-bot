"""Generate attribution reports from Swing Engine trade and skip ledgers.

This module is analysis-only: it reads completed backtest CSVs and never
changes signals, fills, exits, sizing, or portfolio state.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _scenario_from_path(path: Path, prefix: str) -> str:
    name = path.stem
    return name[len(prefix):] if name.startswith(prefix) else name


def _aggregate_trades(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=keys + [
            "closed_trades", "wins", "losses", "win_rate_pct",
            "net_pnl", "gross_profit", "gross_loss", "profit_factor",
            "average_pnl", "average_r",
        ])
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        risk = pd.to_numeric(group["risk_amount"], errors="coerce")
        valid = risk > 0
        avg_r = float((pnl[valid] / risk[valid]).mean()) if valid.any() else 0.0
        rows.append({
            **dict(zip(keys, values)),
            "closed_trades": int(len(group)),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": wins / len(group) * 100 if len(group) else 0.0,
            "net_pnl": float(pnl.sum()),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": gross_profit / gross_loss if gross_loss else float("inf") if gross_profit else 0.0,
            "average_pnl": float(pnl.mean()) if len(pnl) else 0.0,
            "average_r": avg_r,
        })
    return pd.DataFrame(rows)


def generate_diagnostics(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    trade_files = sorted(root.glob("trades_risk_*_rr_*.csv"))
    if not trade_files:
        raise FileNotFoundError(f"No trades_risk_*_rr_*.csv files found in {root}")

    trade_parts = []
    for path in trade_files:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["scenario"] = _scenario_from_path(path, "trades_risk_")
        frame["entry_month"] = pd.to_datetime(
            pd.to_numeric(frame["opened_at"], errors="coerce"),
            unit="ms", utc=True,
        ).dt.strftime("%Y-%m")
        frame["exit_month"] = pd.to_datetime(
            pd.to_numeric(frame["closed_at"], errors="coerce"),
            unit="ms", utc=True,
        ).dt.strftime("%Y-%m")
        trade_parts.append(frame)

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    if trades.empty:
        raise ValueError("Trade ledgers exist, but contain no closed trades")

    outputs = []
    groupings = {
        "by_method": ["scenario", "method"],
        "by_symbol": ["scenario", "symbol"],
        "by_entry_month": ["scenario", "entry_month"],
        "by_exit_month": ["scenario", "exit_month"],
        "by_direction": ["scenario", "direction"],
        "by_touch_number": ["scenario", "touch_number"],
        "by_exit_reason": ["scenario", "exit_reason"],
        "by_symbol_method": ["scenario", "symbol", "method"],
    }
    for name, keys in groupings.items():
        path = root / f"diagnostics_{name}.csv"
        _aggregate_trades(trades, keys).to_csv(path, index=False)
        outputs.append(path)

    skipped_files = sorted(root.glob("skipped_*.csv"))
    skip_parts = []
    for path in skipped_files:
        frame = pd.read_csv(path)
        if not frame.empty:
            frame["scenario"] = _scenario_from_path(path, "skipped_")
            skip_parts.append(frame)
    if skip_parts:
        skipped = pd.concat(skip_parts, ignore_index=True)
        for name, keys in {
            "skipped_by_reason": ["scenario", "skip_reason"],
            "skipped_by_method": ["scenario", "method", "skip_reason"],
            "skipped_by_symbol": ["scenario", "symbol", "skip_reason"],
        }.items():
            result = (
                skipped.groupby(keys, dropna=False)
                .size().reset_index(name="skipped_signals")
                .sort_values("skipped_signals", ascending=False)
            )
            path = root / f"diagnostics_{name}.csv"
            result.to_csv(path, index=False)
            outputs.append(path)

    # Compact cross-scenario ranking for quick inspection.
    overall = _aggregate_trades(trades, ["scenario"])
    overall = overall.sort_values(["profit_factor", "net_pnl"], ascending=False)
    path = root / "diagnostics_scenario_ranking.csv"
    overall.to_csv(path, index=False)
    outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="Backtest/results/swing_engine")
    args = parser.parse_args()
    paths = generate_diagnostics(args.input_dir)
    for path in paths:
        print(f"[DIAGNOSTIC] saved {path}", flush=True)


if __name__ == "__main__":
    main()
