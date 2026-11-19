# Swing Engine — isolated OKX perpetual backtest

This engine is simulation-only. It does not submit live orders and does not
modify the legacy scanner or live trading code.

## Dataset layout

The loader expects validated OKX archives at:

- `Backtest/data/okx/15m/*.csv.gz`
- `Backtest/data/okx/1h/*.csv.gz`
- `Backtest/data/okx/4h/*.csv.gz`

Each file must contain `ts,open,high,low,close,volume`. The `ts` value is
the candle OPEN time in Unix milliseconds. The default run validates all 76
symbols across all three timeframes.

## Run the full nine-combination backtest

From the repository root:

```bash
pip install pandas numpy
python Backtest/SwingEngine/backtest_runner.py
```

By default, the runner downloads public OKX instrument specifications and
applies linear USDT SWAP contract lot-size, minimum-size, and tick-size rules.
No API key is required. If a calculated order is below the exchange minimum,
the trade is skipped and counted; the runner never increases risk to force
an order.

For theoretical base-quantity sizing without exchange lot/minimum constraints:

```bash
python Backtest/SwingEngine/backtest_runner.py --legacy-sizing
```

The optional `--require-4h-alignment` flag enables the 4h trend filter. It is
off by default.

## Test matrix

Initial capital is $100. Every run uses the same signals, entry, and stop
placement, with zero fees, zero slippage, and zero funding:

| Risk per trade | RR values |
|---|---|
| 0.05% | 1.5, 2.0, 2.5 |
| 1.0% | 1.5, 2.0, 2.5 |
| 1.5% | 1.5, 2.0, 2.5 |

The stop is placed at the far wick edge of the source 1h Order Block:
below the lower wick for BUY and above the upper wick for SELL. The target is
calculated from the entry-to-stop distance multiplied by the selected RR.

## Execution and portfolio rules

- Maximum 6 simultaneous positions.
- Maximum 3 BUY and maximum 3 SELL.
- One open position per symbol; symbols with an open position are not rescanned.
- Entries occur at the close of the signal 15m candle.
- Exit checks begin on the next 15m candle, never on the entry candle.
- If SL and TP are both touched in one 15m candle, SL is assumed first.
- Risk is calculated from current equity and is not multiplied by leverage.
- No fees, slippage, or funding are applied at this stage.
- Positions still open at the dataset end are reported separately and are not
  force-closed at an invented price.

## Outputs

Files are written to `Backtest/results/swing_engine/`:

- `summary_9_combinations.csv` and `.json`
- One `trades_*.csv` ledger for each of the nine scenarios

The summary includes trade count, win rate, profit factor, average R,
maximum realized-equity drawdown, ending equity, open positions, and skipped
trade counts.

## No-lookahead rule

A candle is available to a decision only when its close time has passed:
`candle_open_timestamp + timeframe_duration <= decision_timestamp`.


## Live progress and stall diagnosis

The runner reports progress from the beginning of the job, not only during
the nine portfolio simulations:

- Dataset validation/loading: completed symbols out of 76, with a 30-second
  heartbeat while a compressed file is being read.
- Order Block detection: current symbol and scanned 1h candle count.
- Signal preparation: current symbol, Order Block scan count, and 15m candle
  scan count. Long inner scans emit progress callbacks every 50,000 candles.
- Timeline construction: symbols indexed for the global 15m close timeline.
- Each of the nine scenarios: progress approximately every 5%, active UTC
  month, elapsed time, and estimated remaining time.
- Saving: completed trade ledgers out of nine.

A `[HEARTBEAT]` line is emitted every 30 seconds and identifies the exact
stage and current symbol/operation. If the same operation remains active for
a long time, the heartbeat continues to show that it is alive but not
advancing its sub-step counter; this helps distinguish a slow nested scan
from a silent or stalled process. If no sub-step callback arrives for two
minutes, the runner also prints `[STALL WARNING]` with the current stage and
operation. This warning does not kill or alter the run. The heartbeat is
diagnostic and does not change signal rules, event ordering, position sizing,
or calculated results.
