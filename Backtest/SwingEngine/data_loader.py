"""Validated loader for the OKX 15m/1h/4h research datasets.

CSV files are expected at Backtest/data/okx/{15m,1h,4h}/*.csv.gz.
The timestamp column is the candle OPEN time in Unix milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Callable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["ts", "open", "high", "low", "close", "volume"]
TIMEFRAME_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}
EXPECTED_SYMBOLS = 76


class DataValidationError(ValueError):
    """Raised when a dataset file violates the agreed data contract."""


@dataclass
class SymbolData:
    symbol: str
    frames: Dict[str, pd.DataFrame]


def _validate_frame(path: Path, timeframe: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, compression="gzip")
    except Exception as exc:
        raise DataValidationError(f"Cannot read {path}: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"{path.name}: missing columns {missing}")

    frame = frame[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    values = frame[REQUIRED_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataValidationError(f"{path.name}: null, non-numeric or infinite values")

    ts = frame["ts"].to_numpy(dtype=float)
    if not np.equal(ts, np.floor(ts)).all():
        raise DataValidationError(f"{path.name}: timestamps are not integer milliseconds")
    frame["ts"] = frame["ts"].astype("int64")

    if frame.empty:
        raise DataValidationError(f"{path.name}: empty dataset")
    if frame["ts"].duplicated().any():
        raise DataValidationError(f"{path.name}: duplicate timestamps")
    if not frame["ts"].is_monotonic_increasing:
        raise DataValidationError(f"{path.name}: timestamps are not strictly ordered")

    interval = TIMEFRAME_MS[timeframe]
    if (frame["ts"] % interval != 0).any():
        raise DataValidationError(
            f"{path.name}: {timeframe} timestamps are not UTC-aligned"
        )

    if (frame["volume"] < 0).any():
        raise DataValidationError(f"{path.name}: negative volume")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise DataValidationError(f"{path.name}: invalid high")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise DataValidationError(f"{path.name}: invalid low")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError(f"{path.name}: non-positive OHLC price")

    return frame.reset_index(drop=True)


def load_okx_data(
    data_root: str | Path = "Backtest/data/okx",
    expected_symbols: int | None = EXPECTED_SYMBOLS,
    progress_callback: Callable[[str, str, int, int], None] | None = None,
) -> Dict[str, SymbolData]:
    """Load and validate all symbols across the three available timeframes.

    Different symbols may have different history lengths. We deliberately do
    not force their first/last timestamps to match.
    """
    root = Path(data_root)
    paths: dict[str, dict[str, Path]] = {}

    for timeframe in TIMEFRAME_MS:
        folder = root / timeframe
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        files = sorted(folder.glob("*.csv.gz"))
        if not files:
            raise FileNotFoundError(f"No *.csv.gz files found in {folder}")

        paths[timeframe] = {
            file.name.removesuffix(".csv.gz"): file for file in files
        }

    symbol_sets = {tf: set(items) for tf, items in paths.items()}
    base_symbols = symbol_sets["15m"]

    for timeframe, symbols in symbol_sets.items():
        if symbols != base_symbols:
            missing = sorted(base_symbols - symbols)
            extra = sorted(symbols - base_symbols)
            raise DataValidationError(
                f"{timeframe} symbol file set differs from 15m; "
                f"missing={missing}, extra={extra}"
            )

    if expected_symbols is not None and len(base_symbols) != expected_symbols:
        raise DataValidationError(
            f"Expected {expected_symbols} symbols, found {len(base_symbols)}"
        )

    result: Dict[str, SymbolData] = {}
    symbols = sorted(base_symbols)
    total = len(symbols)
    for index, symbol in enumerate(symbols, start=1):
        frames = {}
        for timeframe in TIMEFRAME_MS:
            frames[timeframe] = _validate_frame(paths[timeframe][symbol], timeframe)
            if progress_callback is not None:
                progress_callback(timeframe, symbol, index, total)
        result[symbol] = SymbolData(symbol=symbol, frames=frames)

    return result


def dataset_report(data: Dict[str, SymbolData]) -> pd.DataFrame:
    """Return one coverage/row-count record per symbol and timeframe."""
    rows = []
    for symbol, item in sorted(data.items()):
        for timeframe, frame in item.frames.items():
            interval = TIMEFRAME_MS[timeframe]
            rows.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(frame),
                "first_open_utc": pd.to_datetime(
                    int(frame["ts"].iloc[0]), unit="ms", utc=True
                ).isoformat(),
                "last_open_utc": pd.to_datetime(
                    int(frame["ts"].iloc[-1]), unit="ms", utc=True
                ).isoformat(),
                "last_close_utc": pd.to_datetime(
                    int(frame["ts"].iloc[-1]) + interval, unit="ms", utc=True
                ).isoformat(),
            })
    return pd.DataFrame(rows)


def latest_closed_index(
    frame: pd.DataFrame,
    timeframe: str,
    decision_time_ms: int,
) -> int | None:
    """Index of the latest fully closed candle at decision_time_ms.

    Since ts is candle OPEN time, a bar is usable only when
    ts + timeframe duration <= decision_time_ms. This prevents HTF lookahead.
    """
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    close_times = frame["ts"].to_numpy(dtype=np.int64) + TIMEFRAME_MS[timeframe]
    eligible = np.flatnonzero(close_times <= int(decision_time_ms))
    return int(eligible[-1]) if len(eligible) else None
