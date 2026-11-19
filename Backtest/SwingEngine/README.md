# Swing Engine — initial isolated scaffold

This directory is intentionally separate from the legacy backtest code.

## Dataset layout

The loader expects the already validated OKX archives to be extracted so the
repository contains:

- `Backtest/data/okx/15m/*.csv.gz`
- `Backtest/data/okx/1h/*.csv.gz`
- `Backtest/data/okx/4h/*.csv.gz`

Each file must contain `ts,open,high,low,close,volume`. `ts` is candle OPEN
time in Unix milliseconds.

## Run

From the repository root:

```bash
pip install pandas numpy
python Backtest/SwingEngine/smoke_test.py
```

## Portfolio rules implemented

- Maximum 6 simultaneous trades.
- Maximum 3 BUY and maximum 3 SELL.
- One open position per symbol.
- A symbol with an open position is skipped entirely by the signal scanner.
- When its trade closes, the symbol becomes eligible for scanning again.

This scaffold does not yet generate signals or execute backtests. It validates
the dataset and the agreed portfolio constraints first.

## No-lookahead rule

A candle is available to a decision only when its close time has passed:
`candle_open_timestamp + timeframe_duration <= decision_timestamp`.
